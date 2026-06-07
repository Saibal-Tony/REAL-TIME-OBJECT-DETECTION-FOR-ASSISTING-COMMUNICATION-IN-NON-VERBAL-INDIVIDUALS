<p align="center">
  <h1 align="center">🤟 Real-Time Sign Language Detection</h1>
  <h3 align="center">Assistive Communication System for Non-Verbal Individuals</h3>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9.13-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.10.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.8.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/CUDA-13.2-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

---

## 📑 Table of Contents

- [About the Project](#-about-the-project)
- [Real World Impact](#-real-world-impact)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [System Requirements](#-system-requirements)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the Project](#-running-the-project)
- [Training Your Own Model](#-training-your-own-model)
- [Active Learning Pipeline](#-active-learning-pipeline)
- [Model Performance](#-model-performance)
- [Common Issues & Fixes](#-common-issues--fixes)
- [Roadmap](#-roadmap)
- [Dependencies](#-dependencies)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 About the Project

A **real-time sign language recognition system** built to assist communication for non-verbal individuals using computer vision and deep learning.

The system detects **American Sign Language (ASL) hand gestures** from a live webcam feed using a custom-trained **SSD MobileNet V2** object detection model and displays the detected letter on screen in real time.

### The Problem

Millions of people worldwide are non-verbal due to:
- Autism Spectrum Disorder
- Cerebral Palsy
- ALS / Motor Neuron Disease
- Post-surgery recovery (tracheotomy, etc.)
- Stroke
- Deaf/Hard of hearing

These individuals often rely on sign language to communicate, but most people around them don't understand it. This creates a **massive communication barrier** in daily life, healthcare, and education.

### The Solution

This system acts as a **real-time sign language interpreter** — detecting hand gestures and converting them to text (and soon speech), allowing non-verbal individuals to communicate with anyone.

---

## 🌍 Real World Impact

| Use Case | How This Helps |
|----------|---------------|
| 🏥 Healthcare | Patients who cannot speak after surgery can communicate with doctors/nurses |
| 🏫 Education | Deaf students can interact with hearing teachers without an interpreter |
| 👨‍👩‍👧 Daily Life | Non-verbal individuals can communicate with family and strangers |
| 📞 Video Calls | Real-time sign language interpretation during calls |
| 🏛️ Public Services | Kiosks and information terminals accessible to deaf community |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   WEBCAM INPUT                      │
│              (Live video stream)                    │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│           SSD MobileNet V2 FPN (640x640)            │
│         TensorFlow Object Detection API             │
│                                                     │
│  • Detects hand region in frame                     │
│  • Classifies gesture as A-Z letter                 │
│  • Returns bounding box + confidence score          │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│              POST-PROCESSING                        │
│  • Filter detections by confidence (>70%)           │
│  • Draw bounding box on frame                       │
│  • Display label + confidence score                 │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                 OUTPUT                              │
│  • Real-time annotated video display                │
│  • Detected letter shown on screen                  │
│  • (Coming soon) Text-to-speech output              │
└─────────────────────────────────────────────────────┘
```

### Training Pipeline Architecture

```
Raw Images (Webcam)
       ↓
LabelImg (Manual Annotation → .xml files)
       ↓
generate_tfrecord.py (Convert to .record files)
       ↓
SSD MobileNet V2 Fine-tuning
       ↓
Trained Model (saved_model/)
       ↓
Auto-Label remaining images (Active Learning)
       ↓
Final Training with all labeled data
       ↓
Export → TFLite (Mobile) / TFJS (Web) / SavedModel (Desktop)
```

---

## 🛠️ Tech Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Deep Learning | TensorFlow | 2.10.0 | Model training & inference |
| Object Detection | TF Object Detection API | 2.10 | SSD MobileNet pipeline |
| Computer Vision | OpenCV | 4.8.0 | Webcam capture & display |
| Image Processing | Pillow | 9.5.0 | Image manipulation |
| Data Processing | NumPy | 1.24.3 | Array operations |
| Visualization | Matplotlib | 3.7.2 | Training visualization |
| Annotation Tool | LabelImg | latest | Manual image labeling |
| GPU Acceleration | CUDA + cuDNN | 13.2 + 8.9 | GPU training |
| Language | Python | 3.9.13 | Core language |
| Notebook | Jupyter | latest | Development environment |
| IDE | VS Code | latest | Code editor |

---

## 💻 System Requirements

### Minimum Requirements

| Component | Minimum |
|-----------|---------|
| OS | Windows 10 (64-bit) |
| CPU | Intel Core i5 / AMD Ryzen 5 |
| RAM | 8 GB |
| GPU | NVIDIA GTX 1050 (4GB VRAM) |
| Storage | 10 GB free space |
| Python | 3.9.x |

### Recommended Requirements (This Project)

| Component | Used In This Project |
|-----------|---------------------|
| OS | Windows 11 |
| CPU | AMD Ryzen 5 4600H |
| RAM | 16 GB DDR4 |
| GPU | NVIDIA GeForce GTX 1650 (4GB VRAM) |
| Storage | 20 GB free space |
| Python | 3.9.13 |
| CUDA | 13.2 |
| Driver | 596.49 |

> ⚠️ **CPU-only training is possible** but very slow. A NVIDIA GPU with CUDA support is strongly recommended.

> ⚠️ **TensorFlow 2.10.0** is the last version with **native Windows GPU support**. TF 2.11+ on Windows runs on CPU only.

---

## 📁 Project Structure

```
REAL-TIME-OBJECT-DETECTION/
│
├── 📓 1_Complete_Pipeline_FINAL.ipynb   ← Main pipeline (all steps)
├── 🐍 detect_webcam.py                  ← Real-time detection script
├── 🐍 auto_label.py                     ← Auto-labeling script
├── 📄 requirements.txt                  ← All dependencies with versions
├── 📄 README.md                         ← This file
├── 📄 .gitignore                        ← Git ignore rules
│
├── Tensorflow/
│   ├── models/                          ← TF Object Detection API (cloned)
│   │   └── research/
│   │       └── object_detection/        ← Core TFOD API
│   │
│   ├── protoc/                          ← Protobuf compiler
│   │   └── bin/
│   │       └── protoc.exe
│   │
│   ├── scripts/
│   │   └── generate_tfrecord.py         ← Converts XML → TFRecord
│   │
│   ├── labelimg/                        ← LabelImg annotation tool
│   │
│   └── workspace/
│       ├── annotations/
│       │   ├── label_map.pbtxt          ← Class labels (A-Z)
│       │   ├── train.record             ← Training data (TF format)
│       │   └── test.record              ← Testing data (TF format)
│       │
│       ├── images/
│       │   ├── collectedimages/         ← Raw collected images
│       │   │   ├── A/                   ← Images + .xml labels for A
│       │   │   ├── B/
│       │   │   └── ... (A-Z folders)
│       │   ├── train_label/             ← Initial training split (80%)
│       │   ├── test_label/              ← Initial testing split (20%)
│       │   ├── train/                   ← Final training split (80%)
│       │   └── test/                    ← Final testing split (20%)
│       │
│       ├── models/
│       │   ├── my_ssd_mobnet_initial/   ← Initial trained model
│       │   │   ├── pipeline.config
│       │   │   ├── ckpt-0.*             ← Checkpoints
│       │   │   ├── ckpt-1.*
│       │   │   ├── ckpt-2.*             ← Step 2000 checkpoint
│       │   │   └── export/
│       │   │       └── saved_model/     ← Exported inference model
│       │   │
│       │   └── my_ssd_mobnet_final/     ← Final trained model
│       │       └── export/
│       │           └── saved_model/     ← Final inference model
│       │
│       └── pre-trained-models/
│           └── ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/
│               ├── checkpoint/          ← Pretrained weights
│               └── pipeline.config      ← Base config
│
└── ObjEnv/                              ← Python virtual environment (not in git)
```

---

## ⚙️ Installation & Setup

### Step 1 — Install Python 3.9.13

Download the Windows 64-bit installer:

👉 **Direct Download:** https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe

During installation:
- ✅ Check **"Add Python 3.9 to PATH"**
- ✅ Click **"Customize installation"**
- ✅ Ensure **pip** and **py launcher** are checked
- ✅ Choose **"Install for all users"** (optional but recommended)

Verify:
```bash
python --version
# Output: Python 3.9.13
```

> ⚠️ If you have multiple Python versions, use the full path:
> `C:\Users\YourName\AppData\Local\Programs\Python\Python39\python.exe --version`

---

### Step 2 — Clone This Repository

```bash
git clone https://github.com/yourusername/sign-language-detection.git
cd sign-language-detection
```

---

### Step 3 — Create Virtual Environment

```bash
# Using Python 3.9 explicitly (recommended if multiple versions installed)
C:\Users\YourName\AppData\Local\Programs\Python\Python39\python.exe -m venv ObjEnv

# Activate (Windows PowerShell)
ObjEnv\Scripts\activate

# Activate (Windows CMD)
ObjEnv\Scripts\activate.bat

# You should now see (ObjEnv) at the start of your terminal prompt
```

---

### Step 4 — Upgrade pip

```bash
python -m pip install --upgrade pip
# Should upgrade to pip 26.x
```

---

### Step 5 — Install All Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ This installs ~2GB of packages including TensorFlow. May take 10-15 minutes depending on your internet speed.

> ⚠️ Do NOT upgrade packages if pip suggests it — versions are pinned for compatibility.

---

### Step 6 — Fix cuDNN DLLs (Required for GPU)

TensorFlow 2.10 expects cuDNN 8.x DLL names, but the pip package ships cuDNN 9.x. Run these copies in PowerShell:

```powershell
# Navigate to your project folder first
cd "path\to\your\project"

$cudnn = "ObjEnv\lib\site-packages\nvidia\cudnn\bin"

# Copy with TF 2.10 compatible names
copy "$cudnn\cudnn64_9.dll"             "$cudnn\cudnn64_8.dll"
copy "$cudnn\cudnn_adv_infer64_8.dll"   "$cudnn\cudnn_adv_infer64_8.dll"
copy "$cudnn\cudnn_ops_infer64_8.dll"   "$cudnn\cudnn_ops_infer64_8.dll"
copy "$cudnn\cudnn_cnn_infer64_8.dll"   "$cudnn\cudnn_cnn_infer64_8.dll"
```

---

### Step 7 — Add CUDA Libraries to System PATH (Permanent)

Run this **once** in PowerShell (restart terminal after):

```powershell
$base = "$PWD\ObjEnv\lib\site-packages\nvidia"
$cudaPaths = @(
    "$base\cublas\bin",
    "$base\cuda_runtime\bin",
    "$base\cudnn\bin",
    "$base\cufft\bin",
    "$base\cusolver\bin",
    "$base\cusparse\bin"
)
$currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
$newPath = $currentPath + ";" + ($cudaPaths -join ";")
[System.Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
Write-Host "✅ Done! Please restart your terminal."
```

---

### Step 8 — Verify GPU Setup

```bash
python -c "import tensorflow as tf; print('TF Version:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

Expected output:
```
TF Version: 2.10.0
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If GPU shows `[]` — check that cuDNN PATH was added and restart terminal.

---

### Step 9 — Install TF Object Detection API

Open `1_Complete_Pipeline_FINAL.ipynb` in VS Code:

1. Select **ObjEnv (Python 3.9.13)** as the kernel
2. Run **Cell 1** (GPU Setup)
3. Run **Cell 2** (Paths Setup)
4. Run **Cell 3** (wget)
5. Run **Cell 4** (Clone TF Models — runs once)
6. Run **Cell 5** (Install TFOD API — runs once, takes ~5 minutes)
7. Run **Cell 6** (Download Pretrained Model — runs once, ~30MB)

> ✅ Cells 4, 5, 6 are guarded with flag files — they auto-skip on subsequent runs.

---

### Step 10 — Patch model_lib_v2.py (Required Fix)

This patches a known JIT compilation bug with TF 2.10 + CUDA 13.x:

```python
# Run this cell in the notebook once
model_lib_path = r'path\to\ObjEnv\lib\site-packages\object_detection\model_lib_v2.py'

with open(model_lib_path, 'r') as f:
    content = f.read()

content = content.replace(
    'if record_summaries:',
    'if False:  # disabled to fix JIT compilation error'
)
content = content.replace(
    'lambda: global_step % num_steps_per_iteration == 0)',
    'True)'
)

with open(model_lib_path, 'w') as f:
    f.write(content)

print('✅ Patched!')
```

---

## 🚀 Running the Project

### Real-Time Detection (after training)

```bash
# Activate environment
ObjEnv\Scripts\activate

# Run webcam detection
python detect_webcam.py
```

- A **camera window** will open showing live detection
- Show your hand making ASL letter gestures
- Detected letter appears with bounding box and confidence score
- Press **Q** to quit

> ⚠️ Use `opencv-contrib-python` (not `opencv-python`) for the window to appear:
> ```bash
> pip uninstall opencv-python opencv-python-headless -y
> pip install opencv-contrib-python==4.8.0.76
> ```

---

## 📊 Training Your Own Model

### Notebook Cell Reference

| Cell | What it does | When to run |
|------|-------------|-------------|
| **Cell 1** | GPU + cuDNN setup | ▶️ Every time |
| **Cell 2** | Define all paths | ▶️ Every time |
| **Cell 3** | Install wget | ▶️ Every time |
| **Cell 4** | Clone TF Models repo | 🔁 Once (auto) |
| **Cell 5** | Install TFOD API + protoc | 🔁 Once (auto) |
| **Cell 6** | Download SSD MobileNet pretrained | 🔁 Once (auto) |
| **Cell 7** | Create label_map.pbtxt | 📝 Before each training |
| **Cell 8** | Split images → train_label / test_label | 📝 After labeling |
| **Cell 9** | Generate train.record + test.record | 📝 Before training |
| **Cell 10** | Update pipeline.config | 📝 Before training |
| **Cell 11** | Train initial model (2000 steps) | 🚂 Train |
| **Cell 12** | Export trained model | 📦 After training |
| **Cell 13** | Test detection per class | 🔍 After export |
| **Cell 14** | Auto-label remaining images | 🤖 After initial training |
| **Cell 15** | Final split (all labeled images) | 📝 After auto-labeling |
| **Cell 16** | Final training (10000 steps) | 🚂 Final train |
| **Cell 17** | Export final model | 📦 After final training |
| **Cell 18** | Real-time webcam detection | 🎥 Demo |

---

### Collecting Images

In Cell 3 the webcam captures images automatically:

```python
labels      = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
number_imgs = 40  # images per class
```

- **40 images per class** is recommended minimum
- **100 images per class** for production quality
- Tips for better data:
  - 💡 Vary lighting conditions (bright, dim, natural)
  - 📐 Vary angles (straight, slight tilt left/right)
  - 👕 Vary backgrounds (plain, busy, dark, light)
  - ↔️ Vary distances (close, medium, far from camera)
  - 🖐️ Try both hands if applicable

---

### Labeling Images with LabelImg

LabelImg opens automatically from Cell 5 (Labeling section):

```bash
# Opens automatically or run manually:
cd Tensorflow/labelimg
python labelImg.py
```

**Setup in LabelImg:**
1. Click **Open Dir** → select `Tensorflow/workspace/images/collectedimages/A`
2. Click **Change Save Dir** → select the same folder (save XML next to images)
3. Make sure format shows **PascalVOC** (bottom left) — not YOLO
4. Label at least **20 images per class** before training

**Keyboard Shortcuts:**

| Key | Action |
|-----|--------|
| `W` | Draw bounding box |
| `D` | Next image |
| `A` | Previous image |
| `Ctrl + S` | Save label |
| `Ctrl + D` | Duplicate last label |
| `Del` | Delete selected box |
| `Space` | Flag image |

**Tips:**
- Draw box **tightly** around the hand gesture
- Include the whole hand but exclude arm as much as possible
- For letters like C, O — include the open space inside the hand

---

### Training Configuration

Key settings in `pipeline.config` (updated automatically by Cell 10):

```
num_classes:          24 (A-Y excluding J,Z) or 26 (all)
batch_size:           2  (for 4GB VRAM GPU)
num_train_steps:      2000 (initial) / 10000 (final)
fine_tune_checkpoint: path to pretrained or previous model
image_resizer:        640 x 640
learning_rate_base:   0.08
warmup_steps:         100
```

**What to watch during training:**
```
Step 100  → total_loss should be < 2.0
Step 500  → total_loss should be < 1.0
Step 1000 → total_loss should be < 0.5
Step 2000 → total_loss should be < 0.3 (good initial model)
```

If loss is not dropping, check:
- Images are labeled correctly
- Label map matches class names exactly
- TF Records were generated after last label update

---

### Fine-Tuning from Previous Model

When adding new classes, fine-tune from your last checkpoint instead of scratch:

```python
# In Cell 10 — change fine_tune_checkpoint to your trained model:
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(
    'Tensorflow', 'workspace', 'models',
    'my_ssd_mobnet_initial', 'ckpt-2'  # ckpt-2 = step 2000
)
pipeline_config.train_config.fine_tune_checkpoint_type = 'detection'
```

This makes training **3-4x faster** as the model already knows basic features.

---

## 🔄 Active Learning Pipeline

This project uses **active learning** to minimize manual labeling effort:

```
Phase 1: Manual Labeling
─────────────────────────
Label 20 images per class manually in LabelImg
                ↓
Phase 2: Initial Training
─────────────────────────
Train model for 2000 steps on manually labeled images
Loss target: < 0.3
                ↓
Phase 3: Auto-Labeling
─────────────────────────
Run trained model on remaining unlabeled images
Confidence threshold: 0.5
Auto-generates .xml files for high-confidence detections
Low-confidence images → manual_review.json
                ↓
Phase 4: Review
─────────────────────────
Check manual_review.json
Manually label flagged images in LabelImg
                ↓
Phase 5: Final Training
─────────────────────────
Train on ALL labeled images (manual + auto) for 10000 steps
Loss target: < 0.2
```

**Result:** Reduces manual labeling by **60-70%** while maintaining model quality.

### Auto-Label Output

```
✅ Auto-labeled  : 31 images  ← automatically labeled
⏭️  Skipped       : 20 images  ← already had manual labels (safe)
⚠️  Manual review : 0 images   ← needs human check
```

---

## 📈 Model Performance

### Class A Results (Initial Training)

| Metric | Value |
|--------|-------|
| Training Steps | 2000 |
| Final Loss | 0.2196 |
| Training Images | 16 |
| Test Images | 4 |
| Detection Confidence | 93–99% |
| Auto-Label Success Rate | 100% (0 manual reviews) |
| Inference Speed | ~10-15 FPS (GTX 1650) |

### Loss Progression

```
Step  100 → Loss: 0.70  ████████████████████░░░░░░░░░░
Step  300 → Loss: 0.34  ██████████░░░░░░░░░░░░░░░░░░░░
Step  500 → Loss: 0.46  █████████████░░░░░░░░░░░░░░░░░
Step 1000 → Loss: 0.23  ███████░░░░░░░░░░░░░░░░░░░░░░░
Step 1500 → Loss: 0.26  ████████░░░░░░░░░░░░░░░░░░░░░░
Step 2000 → Loss: 0.21  ██████░░░░░░░░░░░░░░░░░░░░░░░░  ✅
```

### Model Architecture Details

```
Model:       SSD MobileNet V2 FPN Lite
Input:       640 × 640 × 3 (RGB)
Backbone:    MobileNet V2 (pretrained on ImageNet)
Neck:        Feature Pyramid Network (FPN)
Head:        Weight-shared convolutional predictor
Anchors:     Multiscale (levels 3-7)
Loss:        Focal loss (classification) + Smooth L1 (localization)
Optimizer:   Momentum (cosine decay learning rate)
```

---

## 🐛 Common Issues & Fixes

### ❌ `cudnn64_8.dll not found`
**Cause:** TF 2.10 expects cuDNN 8.x naming but pip ships 9.x
```powershell
copy "ObjEnv\lib\site-packages\nvidia\cudnn\bin\cudnn64_9.dll" `
     "ObjEnv\lib\site-packages\nvidia\cudnn\bin\cudnn64_8.dll"
```

---

### ❌ `No module named 'scipy'`
**Cause:** scipy not installed or wrong env active
```bash
pip install scipy==1.11.4
```

---

### ❌ `tf-models-official reinstalls to 2.20.0`
**Cause:** setup.py install pulls latest version automatically
```bash
pip install tf-models-official==2.10.1 --force-reinstall
```

---

### ❌ `JIT compilation failed` / `Can't find libdevice`
**Cause:** XLA tries to JIT compile on GPU but can't find CUDA nvvm
```python
# Add to train_wrapper.py:
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['XLA_FLAGS']    = '--xla_gpu_cuda_data_dir=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.2'
```

---

### ❌ `The paging file is too small`
**Cause:** Not enough virtual memory for TF to load
- `Windows + R` → `sysdm.cpl` → Advanced → Performance Settings
- Advanced → Virtual Memory → Change
- Initial size: `8192 MB` | Maximum: `16384 MB`
- Restart PC

---

### ❌ `cv2.imshow` window doesn't appear
**Cause:** `opencv-python` pip wheel has no GUI support
```bash
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-contrib-python==4.8.0.76
```

---

### ❌ `ResourceExhaustedError` when loading model
**Cause:** Previous model still in GPU memory
```python
import gc, tensorflow as tf
try: del detect_fn
except: pass
gc.collect()
tf.keras.backend.clear_session()
```

---

### ❌ `DLL load failed — paging file too small`
**Cause:** System RAM is too low when starting TF
- Close Chrome, Ollama, and other heavy apps
- Increase virtual memory (see above)
- Restart VS Code

---

### ❌ `Protobuf version conflict`
**Cause:** TF 2.10 requires protobuf < 3.20
```bash
pip install protobuf==3.19.6
```

---

### ❌ Detection shows `?` instead of class name
**Cause:** Class index offset issue in visualization
```python
# Change:
detections['detection_classes'] + 1
# To:
detections['detection_classes']
```

---

## 🗺️ Roadmap

### Phase 1 — Static Letter Detection (Current)
- [x] Project setup and environment configuration
- [x] Image collection pipeline (webcam)
- [x] Manual labeling workflow (LabelImg)
- [x] Initial model training (Class A — 99% confidence)
- [x] Active learning / auto-labeling pipeline
- [x] Real-time webcam detection
- [ ] Complete all static letters (A–I, K–Y) — *in progress*
- [ ] Full A–Z detection (excluding J, Z)

### Phase 2 — Motion Gesture Detection
- [ ] J detection (MediaPipe + LSTM)
- [ ] Z detection (MediaPipe + LSTM)
- [ ] Integrate static + motion detection into single pipeline

### Phase 3 — Common Phrases
- [ ] "Hello" (wave gesture)
- [ ] "Thank you"
- [ ] "Please"
- [ ] "Yes" / "No"
- [ ] "Help"
- [ ] "Me / I"
- [ ] "See you later"

### Phase 4 — Output & Communication
- [ ] Text-to-speech (pyttsx3)
- [ ] Letter → word builder
- [ ] Auto-complete suggestions
- [ ] On-screen text display with history

### Phase 5 — Deployment
- [ ] TFLite export → Android app
- [ ] TensorFlow.js export → Web app
- [ ] FastAPI backend for REST API
- [ ] Desktop app (PyQt5/Tkinter GUI)

---

## 📦 Dependencies

Full list with versions and purpose:

```
tensorflow==2.10.0              # Core ML framework (last with Windows GPU support)
protobuf==3.19.6                # TF model config parsing
numpy==1.24.3                   # Numerical computing
opencv-contrib-python==4.8.0.76 # Computer vision + GUI support
matplotlib==3.7.2               # Plotting and visualization
pillow==9.5.0                   # Image processing
scipy==1.11.4                   # Scientific computing (required by TFOD)
wget==3.2                       # File downloading
ipykernel==6.23.3               # Jupyter kernel support
PyQt5==5.15.7                   # GUI framework (LabelImg)
PyQt5-sip==12.11.0              # PyQt5 bindings
lxml==4.9.3                     # XML parsing (LabelImg)
tf-models-official==2.10.1      # TF Model Garden
nvidia-cuda-runtime-cu11        # CUDA runtime DLLs
nvidia-cublas-cu11              # GPU BLAS operations
nvidia-cudnn-cu11==8.9.4.25     # Deep learning GPU acceleration
nvidia-cufft-cu11               # GPU FFT operations
nvidia-cusolver-cu11            # GPU linear algebra
nvidia-cusparse-cu11            # GPU sparse operations
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** your feature branch
   ```bash
   git checkout -b feature/AddPhraseDetection
   ```
3. **Commit** your changes
   ```bash
   git commit -m "Add: phrase detection for Hello gesture"
   ```
4. **Push** to the branch
   ```bash
   git push origin feature/AddPhraseDetection
   ```
5. **Open** a Pull Request

### Contribution Ideas
- Add more gesture classes (phrases, numbers)
- Improve model accuracy with data augmentation
- Build mobile app (Android/iOS)
- Add text-to-speech integration
- Create web demo with TensorFlow.js
- Add support for other sign language systems (BSL, ISL, etc.)

---

## 📄 License

Distributed under the **MIT License**.

```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgements

- [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) — Core detection framework
- [SSD MobileNet V2 Paper](https://arxiv.org/abs/1801.04381) — Model architecture
- [LabelImg](https://github.com/tzutalin/labelImg) — Image annotation tool
- [Nicholas Renotte](https://github.com/nicknochnack) — Tutorial inspiration
- [NVIDIA](https://developer.nvidia.com/cuda-toolkit) — CUDA toolkit
- [American Sign Language University](https://www.lifeprint.com/) — ASL reference

---

## 📬 Contact

**Your Name**
- 📧 Email: your.email@example.com
- 💼 LinkedIn: linkedin.com/in/yourprofile
- 🐙 GitHub: github.com/yourusername

**Project Link:** https://github.com/yourusername/sign-language-detection

---

<p align="center">
  <b>Built with ❤️ for accessibility and inclusion</b><br/>
  <i>Making communication barrier-free, one gesture at a time</i>
</p>
