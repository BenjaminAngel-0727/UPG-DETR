# UPG-DETR

Core model and training implementation of UPG-DETR: Learning Uncertainty-Guided Spatial Priors for Detection Transformers.

This release contains the four-level HGNetV2-B2 model, LUPE, UGCA, PMRA, decoder, matching and losses, COCO-format data loading, optimizer, EMA, distributed training, and standard validation. Additional paper experiments, visualization scripts, weights, datasets, and historical logs are not included.

## Environment

Use Python 3.10 and install `requirements.txt`. The supplied environment follows the original PyTorch 2.1.0 / torchvision 0.16.0 requirements. Choose a PyTorch build compatible with your CUDA installation.

```bash
pip install -r requirements.txt
```

## Configuration

Run commands from the repository root. The default configuration is `configs/upg_detr_m_visdrone.yml`.

| Configuration | Classes | Dataset configuration |
| --- | ---: | --- |
| `configs/upg_detr_m_visdrone.yml` | 10 | `configs/dataset/visdrone.yml` |
| `configs/upg_detr_m_coco.yml` | 80 | `configs/dataset/coco.yml` |
| `configs/upg_detr_m_custom.yml` | 17 | `configs/dataset/custom.yml` |

Set `img_folder`, `ann_file`, and `num_classes` in the selected dataset configuration. Category names and ID mappings are read from the annotation JSON rather than a hardcoded dataset list. Use the same category definitions in training and validation annotations. `remap_mscoco_category: true` maps annotation IDs to contiguous training labels and restores annotation IDs for evaluation.

The custom template uses the 17-entry label space present in the source implementation. If your annotations contain 13 categories, set `num_classes: 13`; use the actual annotation category count, including categories without instances. This template does not define a new data split.

The main model settings are in `configs/include/upg_detr_m.yml`: four feature levels with strides 4/8/16/32, four decoder layers, 256 hidden channels, and 132 epochs. `configs/include/dataloader.yml` uses total batch size 8 and input size 640. The optimizer uses AdamW with learning rate 0.0002. Data augmentation and the model/loss settings follow the supplied source files; these configurations are not a claim that every manuscript setting has been independently reconstructed.

Backbone pretraining remains enabled as in the source configuration. The backbone can download its upstream pretrained parameters when training is launched. No parameters are bundled here. To disable this download and use random initialization, pass `-u HGNetv2.pretrained=False`.

## Training and validation

```bash
python train.py -c configs/upg_detr_m_visdrone.yml --device cuda --seed 42
torchrun --standalone --nproc_per_node=2 train.py -c configs/upg_detr_m_visdrone.yml --device cuda --seed 42
python train.py -c configs/upg_detr_m_coco.yml --device cuda --seed 42
python train.py -c configs/upg_detr_m_visdrone.yml --resume path/to/checkpoint.pth --device cuda
python train.py -c configs/upg_detr_m_visdrone.yml --test-only --resume path/to/checkpoint.pth --device cuda
```

The validation subset is the subset specified by `val_dataloader`; keep the checkpoint-selection split and final test protocol explicit when reporting results. Checkpoints and logs are generated only when you run training or validation.

## Code locations

| Component | File |
| --- | --- |
| Model wrapper | `src/zoo/dfine/dfine.py` |
| LUPE and PMRA | `src/zoo/dfine/hybrid_encoder.py` |
| LUPE feature extractor | `src/zoo/dfine/defe.py` |
| UGCA | `src/zoo/dfine/umap_conv_atttention.py` |
| Distributional decoder | `src/zoo/dfine/dfine_decoder.py` |
| Detection and uncertainty losses | `src/zoo/dfine/dfine_criterion.py` |
| Gaussian map generation | `src/zoo/dfine/map_generater.py` |
| Training and validation loop | `src/solver/det_engine.py` |
| Checkpointing and training schedule | `src/solver/det_solver.py` |

## Attribution

This implementation is based on [D-FINE](https://github.com/Peterande/D-FINE). Upstream copyright and attribution notices are preserved in `NOTICE`; the original Apache-2.0 license is included in `LICENSE`.
