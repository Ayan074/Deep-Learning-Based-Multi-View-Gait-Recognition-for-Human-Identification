<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Dataset-CASIA--B-00D4AA?style=for-the-badge&logo=databricks&logoColor=white" alt="CASIA-B"/>
  <img src="https://img.shields.io/badge/Task-Biometric%20ID-FF6B6B?style=for-the-badge&logo=fingerprint&logoColor=white" alt="Biometric"/>
  <img src="https://img.shields.io/badge/License-Academic-8B5CF6?style=for-the-badge&logo=creativecommons&logoColor=white" alt="License"/>
</p>

<h1 align="center">🚶‍♂️ Gait-Based Human Identification<br/>Using Deep Learning</h1>

<p align="center">
  <strong>Identify individuals by the way they walk — no face, no fingerprint, just their gait.</strong>
</p>

<p align="center">
  <em>A CNN-based deep learning system that extracts 256-dimensional gait embeddings from silhouette sequences<br/>
  and performs person re-identification across multiple viewing angles and walking conditions.</em>
</p>

---

<br/>

## 📌 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [📂 Project Structure](#-project-structure)
- [📊 Dataset — CASIA-B](#-dataset--casia-b)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🚀 How to Run](#-how-to-run)
- [📈 Model Architecture](#-model-architecture)
- [🧪 Evaluation Protocol](#-evaluation-protocol)
- [🖼️ Demo & Visualization](#️-demo--visualization)
- [🛠️ Technical Details](#️-technical-details)
- [📜 Citation](#-citation)
- [📝 License](#-license)

---

<br/>

## 🎯 Project Overview

**Gait recognition** is a biometric identification technique that recognizes people based on their **unique walking patterns**. Unlike other biometrics (face, fingerprint, iris), gait can be captured **at a distance** and **without the subject's cooperation**, making it invaluable for:

- 🎥 **Surveillance & Security** — identify individuals in CCTV footage
- 🏥 **Healthcare** — detect gait abnormalities and monitor rehabilitation
- 🔐 **Access Control** — non-intrusive identity verification
- 🕵️ **Forensics** — identify suspects from walking footage

This project implements an **end-to-end deep learning pipeline** that:

1. **Preprocesses** raw silhouette images with aspect-ratio-preserving normalization
2. **Trains** a CNN feature extractor to learn discriminative 256-D gait embeddings
3. **Evaluates** identification accuracy across 11 viewing angles using cosine distance matching
4. **Demonstrates** real-time identification with visual comparison output

---

<br/>

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🧠 Deep Learning Pipeline
- Custom **3-layer CNN** with BatchNorm & Dropout
- **Temporal max-pooling** across 30 frames
- **256-dimensional** embedding space
- Label smoothing & learning rate scheduling

</td>
<td width="50%">

### 📐 Multi-View Evaluation
- **11 viewing angles** (0° → 180°)
- **Rank-1** and **Rank-5** accuracy metrics
- Gallery-probe matching with **cosine distance**
- Cross-condition testing (Normal / Bag / Coat)

</td>
</tr>
<tr>
<td>

### 🖼️ Smart Preprocessing
- Automatic **silhouette detection** & cropping
- **Aspect-ratio preserving** resize to 64×64
- Zero-padded centering on canvas
- Batch serialization to `.pkl` for fast I/O

</td>
<td>

### 🎬 Interactive Demo
- Visual **probe vs. gallery** comparison
- Confidence scoring via distance metrics
- Support for all walking conditions
- Matplotlib-based frame visualization

</td>
</tr>
</table>

---

<br/>

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GAIT IDENTIFICATION PIPELINE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐ │
│  │  Raw      │    │ Preprocessing │    │  CNN      │    │ Embedding│ │
│  │  CASIA-B  │───▶│ • Detect     │───▶│ Feature   │───▶│ Space    │ │
│  │  Images   │    │ • Crop       │    │ Extractor │    │ (256-D)  │ │
│  │  320×240  │    │ • Resize     │    │ Conv1→3   │    │          │ │
│  │           │    │ • Pad to     │    │ MaxPool   │    │          │ │
│  │           │    │   64×64      │    │ FC Layers │    │          │ │
│  └──────────┘    └──────────────┘    └───────────┘    └─────┬────┘ │
│                                                              │      │
│                                                              ▼      │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────────┐ │
│  │  Result   │    │  Cosine      │    │  Gallery Database         │ │
│  │  Subject  │◀──│  Distance    │◀──│  NM-01 @ 90° per subject  │ │
│  │  ID Match │    │  Matching    │    │  (Reference Embeddings)   │ │
│  └──────────┘    └──────────────┘    └───────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

<br/>

## 📂 Project Structure

```
Gait-Based_Human_Identification_Using_Deep_Learning/
│
├── 📄 README.md                    ← You are here
├── 📄 requirements.txt             ← Python dependencies
│
├── 📁 src/                         ← Source code
│   ├── model.py                    ← GaitModel CNN architecture
│   ├── dataset.py                  ← PyTorch Dataset & DataLoader
│   ├── preprocess.py               ← Raw image → 64×64 .pkl pipeline
│   ├── train.py                    ← Training loop with validation
│   ├── evaluate.py                 ← Multi-view Rank-1/5 evaluation
│   ├── demo.py                     ← Visual identification demo
│   ├── view_data.py                ← Data visualization utility
│   └── Gait_project.ipynb          ← Jupyter notebook (all-in-one)
│
├── 📁 data/                        ← ⚠️ NOT included (see below)
│   ├── raw/                        ← CASIA-B silhouette images
│   └── processed/                  ← Preprocessed .pkl sequences
│
├── 📁 weights/                     ← ⚠️ NOT included (too large)
│   ├── gait_model.pth              ← Latest trained weights (~65 MB)
│   └── gait_model_best.pth         ← Best validation checkpoint (~65 MB)
│
└── 📁 data_information/            ← Dataset documentation
    └── DATASET_INFO.md             ← Full dataset guide & download links
```

---

<br/>

## 📊 Dataset — CASIA-B

> [!IMPORTANT]
> **The dataset and preprocessed data are NOT included in this repository** due to their massive size (~12+ GB combined). You must download the dataset separately.

### 🔗 Download Link

<table>
<tr>
<td>

| | |
|:---|:---|
| **Dataset** | **CASIA Gait Database B (CASIA-B)** |
| **Source** | Institute of Automation, Chinese Academy of Sciences |
| **🌐 Official Page** | **[http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)** |
| **📄 Paper** | [Yu et al., ICPR 2006](https://ieeexplore.ieee.org/document/1699873) |
| **Usage** | Academic / Non-commercial research only |

</td>
</tr>
</table>

> 📖 **For detailed download instructions, directory structure, and dataset statistics, see:**
> **[`data_information/DATASET_INFO.md`](data_information/DATASET_INFO.md)**

### Dataset at a Glance

```
124 Subjects  ×  10 Walking Conditions  ×  11 View Angles  =  13,640 Sequences
```

| Property | Value |
|:---|:---|
| Subjects | **124** individuals |
| Normal walking (nm) | 6 sequences per subject |
| Carrying bag (bg) | 2 sequences per subject |
| Wearing coat (cl) | 2 sequences per subject |
| Viewing angles | 0°, 18°, 36°, 54°, 72°, **90°**, 108°, 126°, 144°, 162°, 180° |
| Raw resolution | 320 × 240 px |
| Processed resolution | 64 × 64 px |

### Data Split

| Split | Subjects | Purpose |
|:---|:---|:---|
| 🟢 **Training** | 001 → 074 (74 subjects) | Learning gait embeddings |
| 🔵 **Validation** | 20% of training samples | Hyperparameter tuning |
| 🔴 **Testing** | 075 → 124 (50 subjects) | Unseen identity evaluation |

> [!WARNING]
> **The trained model weights (`weights/`) are also too large for GitHub (~130 MB).**
> You must train the model yourself after setting up the dataset, or contact the author for pre-trained weights.

---

<br/>

## ⚙️ Installation & Setup

### Prerequisites

- **Python** 3.8 or higher
- **CUDA** (optional, for GPU acceleration)
- **pip** package manager

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/Gait-Based_Human_Identification_Using_Deep_Learning.git
cd Gait-Based_Human_Identification_Using_Deep_Learning
```

### Step 2 — Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 <strong>Dependencies List</strong></summary>

| Package | Purpose |
|:---|:---|
| `torch` | Deep learning framework |
| `torchvision` | Image transforms & utilities |
| `opencv-python` | Image reading & preprocessing |
| `numpy` | Numerical computations |
| `tqdm` | Progress bars |
| `scikit-learn` | Metrics (optional) |
| `matplotlib` | Visualization & plotting |

</details>

### Step 4 — Download & Place Dataset

1. Download **CASIA-B silhouette dataset** from the [official source](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)
2. Extract into `data/raw/` following the structure in [`data_information/DATASET_INFO.md`](data_information/DATASET_INFO.md)
3. Verify: `data/raw/001/nm-01/090/` should contain `.png` silhouette images

---

<br/>

## 🚀 How to Run

> [!NOTE]
> All scripts must be run from the `src/` directory. Make sure the dataset is in place before running.

### 📋 Complete Pipeline (Step-by-Step)

---

### Step 1 — Preprocess Raw Images

Converts raw silhouette images into normalized 64×64 `.pkl` sequences.

```bash
cd src
python preprocess.py
```

> ⏱️ **Estimated Time:** 15–30 minutes depending on hardware
>
> **What it does:**
> - Reads each frame from `data/raw/`
> - Detects silhouette bounding box
> - Crops and resizes with aspect ratio preservation
> - Pads to 64×64 canvas
> - Saves as `.pkl` per angle in `data/processed/`

---

### Step 2 — Train the Model

Trains the CNN feature extractor with cross-entropy loss and label smoothing.

```bash
python train.py
```

> ⏱️ **Estimated Time:** 2–4 hours on GPU, 8–12 hours on CPU

| Hyperparameter | Value |
|:---|:---|
| Batch Size | 16 |
| Learning Rate | 0.001 (with ReduceLROnPlateau) |
| Epochs | 50 |
| Optimizer | Adam (weight_decay=1e-4) |
| Loss | CrossEntropy (label_smoothing=0.1) |
| Frame Count | 30 per sequence |

> **Output:** Best model saved to `weights/gait_model_best.pth`

---

### Step 3 — Evaluate Multi-View Accuracy

Computes Rank-1 and Rank-5 identification accuracy across all 11 viewing angles.

```bash
python evaluate.py
```

> **Protocol:**
> - **Gallery:** NM-01 @ 90° (one embedding per test subject)
> - **Probe:** NM-02 @ all angles
> - **Matching:** Cosine distance, nearest neighbor

**Expected Output:**
```
======================================
FINAL MULTI-VIEW RESULTS SUMMARY
======================================

000° : Rank-1 = XX.XX% | Rank-5 = XX.XX%
018° : Rank-1 = XX.XX% | Rank-5 = XX.XX%
...
180° : Rank-1 = XX.XX% | Rank-5 = XX.XX%

--------------------------------------
Average Rank-1 Accuracy : XX.XX%
Average Rank-5 Accuracy : XX.XX%
--------------------------------------
```

---

### Step 4 — Run Interactive Demo

Visual demo that identifies a probe subject against a gallery of known identities.

```bash
python demo.py
```

> Shows a side-by-side comparison of the probe sequence (e.g., wearing a coat) and the matched gallery sequence (normal walk), along with the identification result.

---

### 🔍 View Processed Data

Quickly visualize preprocessed silhouette frames.

```bash
python view_data.py
```

---

### 📓 Jupyter Notebook

For an all-in-one interactive experience:

```bash
jupyter notebook src/Gait_project.ipynb
```

---

<br/>

## 📈 Model Architecture

<table>
<tr><th>Component</th><th>Architecture</th><th>Output Shape</th></tr>
<tr>
<td><strong>Input</strong></td>
<td>30 grayscale silhouette frames</td>
<td><code>[B, 30, 1, 64, 64]</code></td>
</tr>
<tr>
<td><strong>Conv Block 1</strong></td>
<td>Conv2d(1→32) → BatchNorm → ReLU → MaxPool</td>
<td><code>[B×30, 32, 32, 32]</code></td>
</tr>
<tr>
<td><strong>Conv Block 2</strong></td>
<td>Conv2d(32→64) → BatchNorm → ReLU → MaxPool</td>
<td><code>[B×30, 64, 16, 16]</code></td>
</tr>
<tr>
<td><strong>Conv Block 3</strong></td>
<td>Conv2d(64→128) → BatchNorm → ReLU</td>
<td><code>[B×30, 128, 16, 16]</code></td>
</tr>
<tr>
<td><strong>Temporal Pooling</strong></td>
<td>Max-pool across 30 frames</td>
<td><code>[B, 128×16×16]</code></td>
</tr>
<tr>
<td><strong>FC Layer 1</strong></td>
<td>Linear(32768→512) → BatchNorm → ReLU → Dropout(0.5)</td>
<td><code>[B, 512]</code></td>
</tr>
<tr>
<td><strong>FC Layer 2 (Embedding)</strong></td>
<td>Linear(512→256)</td>
<td><code>[B, 256]</code></td>
</tr>
</table>

> **Key Design Choice:** Temporal max-pooling captures the most discriminative frame-level features across the entire gait cycle, creating a single compact representation per sequence.

---

<br/>

## 🧪 Evaluation Protocol

```
┌─────────────────────────────────────────────────┐
│             EVALUATION PROTOCOL                  │
├─────────────────────────────────────────────────┤
│                                                  │
│  GALLERY (Reference Database)                    │
│  ├── Condition: NM-01 (Normal Walk)              │
│  ├── Angle: 90° (Side View)                      │
│  └── Subjects: 075 → 124 (50 people)             │
│                                                  │
│  PROBE (Query / Test)                            │
│  ├── Condition: NM-02 (Different Normal Walk)    │
│  ├── Angles: ALL (0° → 180°, 11 angles)          │
│  └── Subjects: 075 → 124 (same 50 people)        │
│                                                  │
│  MATCHING                                        │
│  ├── Distance: Cosine Similarity                 │
│  ├── Rank-1: Top prediction == True ID?          │
│  └── Rank-5: True ID in top 5 predictions?       │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

<br/>

## 🖼️ Demo & Visualization

The demo script (`demo.py`) provides a visual proof of the system's capabilities:

```
┌─────────────────────────────────────────────────────────┐
│              Gait Identification Demo                    │
│     True ID: 077 | Predicted ID: 077 (SUCCESS!)          │
├──────────┬──────────┬──────────┬──────────┬──────────────┤
│ PROBE    │ Frame 1  │ Frame 2  │ Frame 3  │ Frame 4  ... │
│ (cl-01)  │ 🟨🟨🟨  │ 🟨🟨🟨  │ 🟨🟨🟨  │ 🟨🟨🟨      │
├──────────┼──────────┼──────────┼──────────┼──────────────┤
│ MATCHED  │ Frame 1  │ Frame 2  │ Frame 3  │ Frame 4  ... │
│ (nm-01)  │ 🟩🟩🟩  │ 🟩🟩🟩  │ 🟩🟩🟩  │ 🟩🟩🟩      │
└──────────┴──────────┴──────────┴──────────┴──────────────┘

Top Row: Probe subject walking in a COAT (challenging condition)
Bot Row: Matched gallery subject walking NORMALLY
→ Model correctly identifies the person despite clothing change!
```

---

<br/>

## 🛠️ Technical Details

### Preprocessing Pipeline

| Step | Operation | Details |
|:---|:---|:---|
| 1 | **Read** | Load grayscale silhouette (320×240) |
| 2 | **Detect** | Find non-zero pixels (white silhouette on black) |
| 3 | **Crop** | Tight bounding box around person |
| 4 | **Resize** | Scale to fit 64×64 while preserving aspect ratio |
| 5 | **Pad** | Center on black 64×64 canvas with zero-padding |
| 6 | **Serialize** | Save as `.pkl` NumPy array per angle |

### Training Details

| Component | Configuration |
|:---|:---|
| **Framework** | PyTorch |
| **Device** | Auto-select (CUDA if available) |
| **Data Split** | 74 train subjects / 50 test subjects |
| **Validation** | 20% of training samples (random, seed=42) |
| **Regularization** | Dropout(0.5) + Weight Decay(1e-4) + Label Smoothing(0.1) |
| **LR Schedule** | ReduceLROnPlateau (factor=0.5, patience=5) |
| **Checkpointing** | Best model saved by validation accuracy |

---

<br/>

## 📜 Citation

If you use this project or the CASIA-B dataset, please cite:

```bibtex
@article{yu2006framework,
  title     = {A framework for evaluating the effect of view angle, clothing 
               and carrying condition on gait recognition},
  author    = {Yu, Shiqi and Tan, Daoliang and Tan, Tieniu},
  journal   = {18th International Conference on Pattern Recognition (ICPR'06)},
  volume    = {4},
  pages     = {441--444},
  year      = {2006},
  publisher = {IEEE}
}
```

---

<br/>

## 📝 License

This project is intended for **academic and educational purposes only**.

- The **CASIA-B dataset** is restricted to non-commercial research use.
- The **source code** in this repository is available for learning and research.
- See the official CASIA page for dataset-specific licensing terms.

---

<br/>

<p align="center">
  <strong>🚶‍♂️ Walk. Capture. Identify.</strong>
  <br/><br/>
  <em>Built with ❤️ using PyTorch</em>
  <br/><br/>
  <img src="https://img.shields.io/badge/Made%20with-PyTorch-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Powered%20by-CASIA--B-00D4AA?style=flat-square" alt="CASIA-B"/>
  <img src="https://img.shields.io/badge/Status-Research%20Project-8B5CF6?style=flat-square" alt="Status"/>
</p>
