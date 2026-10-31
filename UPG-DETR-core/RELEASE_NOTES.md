# Release preparation

The original project was read only. This directory is an independent release copy.

Python comments and docstrings, YAML comments, and commented-out experiments were removed. Core computation in the encoder, decoder, prior predictor, attention modules, matcher, and losses was preserved. Files unrelated to the detection training dependency chain were omitted.

Release-specific adjustments:

- Registration modules expose detection, the HGNetV2 backbone, and COCO-format datasets only.
- `train.py` respects the YAML mixed-precision setting, defaults to training, has no checkpoint selected by default, and selects categories from the annotation JSON. The unused hardcoded dataset-name dictionary was removed; mappings are initialized in place before model and dataset construction.
- The optional per-class AP export helper and its validation-only call were removed. Standard aggregate evaluation and recall remain.
- Dataset paths are repository-relative. The active four-level M configuration is shared by three dataset templates.
- Total training batch size is set to 8 for the release configuration. The source value was 2. Other active model, loss, optimizer, and augmentation computations are retained.
- Dependency declarations include modules imported by the retained implementation.

No datasets, checkpoints, pretrained parameters, visualizations, or experimental results are distributed in this directory. Validation and source-integrity reports are kept outside the upload directory.
