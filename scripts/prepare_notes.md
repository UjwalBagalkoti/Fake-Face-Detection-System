# Dataset preparation notes

Keep source identities and videos isolated between train/validation/test. For video datasets, split by video (and preferably identity), then extract/crop frames after the split is created.

Recommended benchmark families:
- FaceForensics++
- Celeb-DF / Celeb-DF++
- DF40

Store processed faces in the directory structure described in README.md.
