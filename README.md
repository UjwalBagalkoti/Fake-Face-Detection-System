# ForensicFusion — Deepfake Detection System

ForensicFusion is an end-to-end deepfake detection system for manipulated faces in images and videos. It combines spatial visual evidence, frequency-domain artifacts, and temporal consistency to produce a calibrated real/fake prediction.

## Architecture

```text
Image / Video
      │
      ▼
Face detection + preprocessing
      │
      ├── Spatial stream ─── EfficientNet CNN
      │
      ├── Frequency stream ─ FFT log-magnitude + CNN
      │
      └── Temporal stream ── Transformer over video frames
                            │
                            ▼
                       Fusion layer
                            │
                            ▼
                 Calibrated prediction
```

### Components

- Face detection and normalization with OpenCV
- EfficientNet-B0/B4 spatial feature extraction
- FFT-based frequency representation
- Feature fusion for image classification
- Transformer encoder for temporal video reasoning
- Validation-based threshold calibration
- Accuracy, precision, recall, F1 and ROC-AUC evaluation
- Gradient-based visual explanation
- FastAPI inference service
- Browser dashboard for image and video uploads
- Test-time augmentation for image inference
- Calibrated uncertain-decision band
- Cross-dataset evaluation utilities
- Compression and blur robustness evaluation
- GitHub Actions CI

## Dataset layout

Use a source-aware split whenever the dataset provides source video or identity information. For image training:

```text
data/images/
├── train/
│   ├── real/*.jpg
│   └── fake/*.jpg
├── val/
│   ├── real/*.jpg
│   └── fake/*.jpg
└── test/
    ├── real/*.jpg
    └── fake/*.jpg
```

For video experiments:

```text
data/video_frames/
├── train/
│   ├── real/<video_id>/*.jpg
│   └── fake/<video_id>/*.jpg
├── val/
│   ├── real/<video_id>/*.jpg
│   └── fake/<video_id>/*.jpg
└── test/
    ├── real/<video_id>/*.jpg
    └── fake/<video_id>/*.jpg
```

Keep all frames from the same source video in the same split. Do not randomly distribute frames from one video across train and test.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the image detector

EfficientNet-B4:

```bash
python -m training.train_image --data data/images --backbone efficientnet_b4 --epochs 15 --batch-size 24 --pretrained
```

EfficientNet-B0 for lighter hardware:

```bash
python -m training.train_image --data data/images --backbone efficientnet_b0 --epochs 10 --batch-size 32 --pretrained
```

The best checkpoint is written to `models/best_image_model.pth`.

## Evaluate

```bash
python -m training.evaluate --data data/images --model models/best_image_model.pth
```

The evaluation flow selects the operating threshold from validation data and reports the final metrics on the untouched test split.

## Train the video detector

```bash
python -m training.train_video --data data/video_frames --image-model models/best_image_model.pth --epochs 8 --batch-size 2
```

## Run the application

Set the model checkpoint path:

```bash
# Windows PowerShell
$env:FORENSIC_MODEL="models/best_image_model.pth"
$env:FORENSIC_VIDEO_MODEL="models/best_video_model.pth"
python run.py
```

```bash
# Linux/macOS
export FORENSIC_MODEL=models/best_image_model.pth
export FORENSIC_VIDEO_MODEL=models/best_video_model.pth
python run.py
```

Open `http://127.0.0.1:8000`.

## API

### Health

`GET /health`

### Image inference

`POST /api/predict/image` with a multipart image file.

### Video inference

`POST /api/predict/video` with an MP4, MOV, WebM or AVI file.

## Explainability

`detection/explain.py` provides a gradient-based heatmap over the CNN feature map. This is intended to show which facial regions contributed to a prediction; it is not independent proof of manipulation.

## Robustness evaluation

Run the stress suite against the untouched test split:

```bash
python -m scripts.robustness_eval --data data/images --model models/best_image_model.pth
```

This evaluates native images plus JPEG compression and blur variants without changing the trained threshold.

## Evaluation protocol

For a credible reported result, record:

- Dataset and exact version
- Source/identity-aware split protocol
- Training configuration and random seed
- Checkpoint identifier
- Accuracy
- ROC-AUC
- Precision, recall and F1
- Threshold selected on validation data
- Per-manipulation metrics when available
- Cross-dataset generalization results

Avoid using the same source identities or source videos in both training and test sets.

## Project structure

```text
ForensicFusion/
├── app/                    FastAPI service
├── configs/                configuration files
├── detection/              models, preprocessing and inference
├── docs/                   model documentation and project notes
├── models/                 trained checkpoints
├── scripts/                dataset and evaluation utilities
├── static/                 web interface
├── training/               datasets, augmentation, training and metrics
├── tests/                  automated tests
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Limitations

Detector performance can change under unseen generators, strong compression, resizing, adversarial edits, poor face crops, unusual lighting, and distribution shift. Predictions should be treated as model evidence rather than definitive proof of authenticity.
