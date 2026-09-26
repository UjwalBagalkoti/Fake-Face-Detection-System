# Technical decisions

## Why CNN?
EfficientNet is a strong image feature extractor and a clean baseline for facial manipulation artifacts.

## Why frequency features?
Some synthesis, resampling and compression artifacts are easier to express in frequency space. A dedicated FFT branch provides complementary evidence.

## Why a Transformer?
Video forgery can produce frame-to-frame inconsistencies. Temporal attention lets the model aggregate evidence across sampled frames.

## Why threshold calibration?
A 0.5 sigmoid cutoff is arbitrary. The project chooses an operating threshold on validation data and carries that fixed threshold to the test set.
