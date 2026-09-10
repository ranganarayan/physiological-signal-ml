# WESAD Pipeline: Code Walkthrough
*Signal → Filter → Feature → Model*

---

## Overview: What the pipeline does

We take raw wrist sensor data from 15 subjects doing a stress protocol, process it into 60-second windows, and train classifiers to distinguish **stress** (TSST protocol) from **non-stress** (baseline, amusement, meditation).

The pipeline has four stages:

```
Raw .pkl file
     ↓
[1] Load & calibrate signals       wesad_loader.py
     ↓
[2] Filter, window, artifact-gate  signal_processing.py
     ↓
[3a] Extract HRV features          hrv_features.py  →  RandomForest
[3b] Stack raw channels            cnn_classifier.py →  1D CNN
     ↓
[4] Leave-One-Subject-Out CV       loso_cv.py
```

---

## Stage 1 — Loading signals (`wesad_loader.py`)

### What's in the raw file

Each subject is a `.pkl` file. Inside:

```python
raw["signal"]["wrist"]   # dict of sensor arrays
raw["label"]             # 1D array at 700 Hz
```

The four wrist sensors and their sampling rates:

| Sensor | Rate | Shape | What it measures |
|--------|------|-------|-----------------|
| BVP    | 64 Hz | (N,) | Blood volume pulse (PPG) — light absorption changes with each heartbeat |
| ACC    | 32 Hz | (N, 3) | 3-axis accelerometer — x/y/z motion |
| TEMP   | 4 Hz  | (N,) | Skin temperature |
| EDA    | 4 Hz  | (N,) | Electrodermal activity (skin conductance — sweat) |

### The critical unit fix (line 72)

```python
# BEFORE (wrong — native E4 units, NOT g):
self.acc = wrist["ACC"].astype(np.float32)

# AFTER (correct):
self.acc = wrist["ACC"].astype(np.float32) / 64.0   # (N, 3) in g
```

The Empatica E4 encodes accelerometer values as integers where **1g = 64 counts**. Raw values range from -128 to 127 (mean magnitude ≈ 63.4, which is ~1g). Without the `/64`, every threshold comparison downstream operates in the wrong units — our 0.15g threshold was effectively 0.0023g, which flagged nearly every window as artifact.

### Label alignment

Labels are recorded at 700 Hz, but signals run at 4–64 Hz. We downsample by **majority vote** in each block:

```python
ratio = 700 // target_fs        # e.g. 700 // 64 = 10 samples per block
trimmed.reshape(n_blocks, ratio)
# → take argmax of bincount per row
```

This means each sample inherits the most common label in its corresponding 700 Hz block. Label meanings:

| Code | Meaning | Binary mapping |
|------|---------|----------------|
| 0 | Undefined / transition | discarded |
| 1 | Baseline | 0 (non-stress) |
| 2 | Stress (TSST) | 1 (stress) |
| 3 | Amusement | 0 (non-stress) |
| 4 | Meditation | 0 (non-stress) |

---

## Stage 2 — Signal processing (`signal_processing.py`)

### Filters applied per signal

**BVP / PPG — bandpass 0.5–4.0 Hz**

```python
bvp_filt = bandpass_filter(subject.bvp, fs=64, low_hz=0.5, high_hz=4.0)
```

Why these frequencies? A resting heart rate of 30 bpm = 0.5 Hz; 240 bpm = 4.0 Hz. The bandpass:
- Removes **baseline wander** (< 0.5 Hz) — slow drift from breathing or movement
- Removes **high-frequency noise** (> 4 Hz) — electronics noise, motion at high frequencies

The filter is a 4th-order Butterworth, run with `filtfilt` (zero-phase). Zero-phase means no time shift in the output — critical because HRV features depend on the exact timing between heartbeat peaks. A causal filter (applied in one direction only) would shift peaks by half the filter delay.

**TEMP and EDA — lowpass at 1.0 Hz**

```python
temp_filt = lowpass_filter(subject.temp, fs=4, cutoff_hz=1.0)
eda_filt  = lowpass_filter(subject.eda,  fs=4, cutoff_hz=1.0)
```

Temperature and skin conductance change very slowly (seconds to minutes). A 1 Hz lowpass removes any fast noise while preserving the physiologically meaningful trend.

**ACC — no frequency filter before windowing**

ACC is not filtered before windowing. Instead, a separate high-pass at 0.5 Hz is applied *inside* the motion artifact detector to remove gravity from the magnitude calculation.

### Normalization

After filtering, every signal is z-scored **per subject**:

```python
bvp_norm = zscore_normalize(bvp_filt)   # (signal - mean) / std
```

This is per-subject, not global. Different people have different baseline EDA levels, skin temperatures, PPG amplitudes. Z-scoring within a subject lets the model focus on relative changes (stress causes EDA to rise *from your baseline*) rather than absolute values that vary between people.

### Windowing — 60-second windows, 30-second step

```python
cfg = WindowConfig(window_s=60, step_s=30)
```

A 60-second window at BVP rate (64 Hz) = **3,840 samples**. Step of 30 seconds = **50% overlap**. Overlap means more training examples and smoother transitions.

Each sensor gets its own window size:
- BVP:     60s × 64 Hz = 3,840 samples
- ACC:     60s × 32 Hz = 1,920 samples
- TEMP/EDA: 60s × 4 Hz =   240 samples

**Label assignment per window** uses majority vote. A window is discarded if fewer than 80% of its samples have a defined label — this removes transition windows between protocol phases.

### Motion artifact detection

```python
def motion_artifact_mask(acc, fs_acc=32, window_s=60, step_s=30, threshold=0.15):
```

Steps:
1. **High-pass ACC at 0.5 Hz** — removes the constant ~1g gravity vector
2. **Compute vector magnitude** (RMS across x/y/z) per sample
3. **Flag a window** if more than 20% of its samples exceed 0.15g

```python
artifact[i] = (chunk > threshold).mean() > 0.20
```

With the unit fix in place, this flags about **8.7% of windows** — a realistic rate for subjects seated in a lab. Before the fix, raw counts (~63) were being compared to 0.15, flagging nearly everything.

---

## Stage 3a — Feature extraction (`hrv_features.py`)

The RandomForest branch uses **29 handcrafted features** per window. No raw signal — just numbers.

### Why HRV?

The autonomic nervous system (ANS) controls heart rate. Stress activates the **sympathetic branch** (fight-or-flight), which speeds the heart and reduces beat-to-beat variability. Relaxation activates the **parasympathetic branch**, which increases variability. HRV is the variance in time between consecutive heartbeats — it is a direct readout of ANS balance.

### Step 1: Detect PPG peaks

```python
peaks = detect_ppg_peaks(bvp, fs=64)
```

Tries `neurokit2` first (more robust), falls back to `scipy.signal.find_peaks` with a minimum distance of 0.33s (capping at ~180 bpm).

### Step 2: RR intervals

```python
rr_ms = np.diff(peaks) / fs * 1000.0   # milliseconds between consecutive peaks
```

Normal range: 500–1200 ms (50–120 bpm).

### Time-domain HRV features

| Feature | What it captures |
|---------|-----------------|
| `hr_mean` | Average heart rate in bpm |
| `hr_std` | HR variability |
| `sdnn` | std(RR) in ms — overall HRV, decreases under stress |
| `rmssd` | Short-term HRV — parasympathetic tone; drops sharply under stress |
| `pnn50` | Fraction of consecutive RR differences > 50ms |

### Poincaré features

The Poincaré plot graphs RR_n vs RR_{n+1}. The cloud forms an ellipse.

```python
sd1 = std((RR_{n+1} - RR_n) / sqrt(2))   # ellipse width — short-term variability
sd2 = std((RR_{n+1} + RR_n) / sqrt(2))   # ellipse length — long-term variability
```

`SD1` is dominated by parasympathetic activity. Under stress, SD1 shrinks. `SD1/SD2` is a useful summary ratio.

### Frequency-domain HRV

```
LF band: 0.04–0.15 Hz
HF band: 0.15–0.40 Hz
```

RR intervals are not evenly spaced in time, so we use **Lomb-Scargle** periodogram (designed for unevenly-sampled data):

- **HF power** (0.15–0.40 Hz) reflects **respiratory sinus arrhythmia** — heart speeds slightly on inhale, slows on exhale. Breathing at 12–25 breaths/min falls here. Purely parasympathetic. Drops under stress.
- **LF power** (0.04–0.15 Hz) reflects mixed sympathetic + parasympathetic. Rises under stress.
- **LF/HF ratio** increases under stress — used as a sympathovagal balance index.

### PPG morphology (3 features)

Shape features per beat: amplitude (systolic peak minus diastolic trough), rise time (trough to peak in ms), and pulse width at 50% amplitude (FWHM). These relate to vascular tone and cardiac output.

### Accelerometer features (9 features)

```python
mag = sqrt(x^2 + y^2 + z^2)
```

Mean magnitude, std magnitude, signal magnitude area (SMA), plus mean and std per axis. Give the RF context about motion level.

### EDA features (5 features)

Mean, std, min, max, **slope**. Under stress: EDA mean rises, slope is positive (conductance increasing as sweat glands activate — sympathetic arousal).

### TEMP features (5 features)

Mean, std, min, max, slope. Under stress: peripheral skin temperature drops as blood is redirected to muscles.

### Final feature vector: 29 features total

```
5 time-domain HRV  +  3 Poincaré  +  4 frequency-domain  +  3 PPG morphology
9 ACC  +  5 TEMP  +  5 EDA
= 29 features
```

---

## Stage 3b — 1D CNN (`cnn_classifier.py`)

The CNN takes **raw preprocessed signal windows** and learns its own representations — no handcrafted features.

### Input preparation

ACC is at 32 Hz; BVP at 64 Hz. ACC is linearly interpolated to 64 Hz, then all four channels are stacked:

```
Input shape: (batch, 4 channels, 3840 time steps)
  Channel 0: BVP
  Channels 1–3: ACC x, y, z (upsampled to 64 Hz)
```

### Architecture

```
ConvBlock 1: Conv1d(4→32, kernel=7) → BatchNorm → ReLU → MaxPool(2)
  → (batch, 32, 1920)

ConvBlock 2: Conv1d(32→64, kernel=5) → BatchNorm → ReLU → MaxPool(2)
  → (batch, 64, 960)

ConvBlock 3: Conv1d(64→128, kernel=3) → BatchNorm → ReLU → MaxPool(2)
  → (batch, 128, 480)

AdaptiveAvgPool → squeeze
  → (batch, 128)

Dropout(0.4) → Linear(128→256) → ReLU
Dropout(0.4) → Linear(256→64)  → ReLU
              → Linear(64→2)
  → (batch, 2)  logits for [non-stress, stress]
```

**Why 1D convolution?** A kernel of size 7 slides across 7 consecutive time samples, learning local temporal patterns — the shape of a heartbeat, a motion burst. Stacking three conv layers builds hierarchical representations (beats → rhythms → patterns).

**Why BatchNorm?** With ~80 training windows per subject per LOSO fold, training is unstable. BatchNorm normalizes activations in each layer, smoothing the loss landscape.

**Why AdaptiveAvgPool?** Instead of a fixed flatten, this averages across all remaining time steps. The architecture works with any input length — changing `window_s` doesn't require changing the model.

**Why Dropout 0.4?** 14 subjects, ~100K parameters — the network will memorize training subjects without regularization. Dropout zeros 40% of activations randomly during training.

### Class weighting

```python
weights = 1.0 / class_counts     # ~24% stress, ~76% non-stress
criterion = CrossEntropyLoss(weight=weights)
```

Without this, a model can reach 76% accuracy by always predicting non-stress. Weighting makes a misclassified stress window ~3× more costly.

### Optimizer

- **AdamW** with weight decay 1e-4
- **Cosine annealing** LR schedule: starts at 1e-3, decays to ~0 over 30 epochs

---

## Stage 4 — Leave-One-Subject-Out CV

**Why LOSO?** With 14 subjects, standard k-fold would put windows from the same person in both train and test sets. The model could memorize subject-specific signatures (this person's baseline HR, their EDA level) and report inflated accuracy. LOSO trains on 13 subjects, tests on the 14th, repeated 14 times. This tests generalization to **unseen people**.

High standard deviation in results (RF F1 = 0.833 ± 0.293) is expected — some subjects show much stronger or more consistent stress responses than others. This inter-subject variability is the core challenge in wearable stress detection.

---

## Key numbers at a glance

| | Before fix | After fix |
|--|--|--|
| ACC units | raw E4 counts | g (÷ 64) |
| Artifact threshold comparison | counts vs 0.15 | g vs 0.15 |
| Windows flagged | ~99% | ~8.7% (122/1396) |

| Metric | RF (no artifact filter) | CNN (no artifact filter) |
|--------|------------------------|--------------------------|
| Accuracy | 0.928 ± 0.115 | 0.647 ± 0.119 |
| F1 | 0.833 ± 0.293 | 0.367 ± 0.168 |
| AUC | 0.949 ± 0.126 | 0.553 ± 0.192 |

*New results with `remove_artifacts=True` pending current pipeline run.*
