# Model Card — Fake Face Detection System

## Intended use
Research, development, evaluation and forensic-method experimentation for manipulated-face detection.

## Architecture
The image path combines an EfficientNet spatial encoder with an FFT-based frequency encoder and a fusion head. The video path adds a Transformer encoder over frame-level fused embeddings.

## Training
Train with licensed benchmark datasets using source-aware train/validation/test splits. Threshold calibration is performed on validation data; the test split is reserved for final reporting.

## Evaluation
Report accuracy, ROC-AUC, precision, recall and F1. For video, also report temporal/frame-level agreement and, where possible, per-manipulation and cross-dataset results.

## Limitations
Distribution shift, unseen generators, compression, resizing, adversarial edits, face-detection failures and dataset bias can affect predictions.

## Responsible use
The detector should not be used as the sole basis for identity verification, legal decisions, hiring decisions, moderation, or claims about a person's intent or honesty.
