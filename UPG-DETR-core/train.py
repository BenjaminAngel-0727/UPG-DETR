import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))

import argparse
import json
from pathlib import Path

from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS
from pprint import pprint

debug = False

if debug:
    def custom_repr(self):
        return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

def configure_categories(cfg, test_only=False):
    from src.data.dataset import coco_dataset

    key = "val_dataloader" if test_only else "train_dataloader"
    annotation = Path(cfg.yaml_cfg[key]["dataset"]["ann_file"])
    with annotation.open(encoding="utf-8") as handle:
        categories = json.load(handle)["categories"]
    categories = sorted(categories, key=lambda item: int(item["id"]))
    if len(categories) != int(cfg.yaml_cfg["num_classes"]):
        raise ValueError(
            f"num_classes={cfg.yaml_cfg['num_classes']} does not match "
            f"the {len(categories)} categories in {annotation}"
        )
    if not cfg.yaml_cfg.get("remap_mscoco_category", False):
        expected = list(range(len(categories)))
        if [int(item["id"]) for item in categories] != expected:
            raise ValueError("Non-contiguous category IDs require remap_mscoco_category: true")
    coco_dataset.mscoco_category2name.clear()
    coco_dataset.mscoco_category2name.update({int(item["id"]): item["name"] for item in categories})
    coco_dataset.mscoco_category2label.clear()
    coco_dataset.mscoco_category2label.update({int(item["id"]): i for i, item in enumerate(categories)})
    coco_dataset.mscoco_label2category.clear()
    coco_dataset.mscoco_label2category.update({i: int(item["id"]) for i, item in enumerate(categories)})

def main(args) -> None:

    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all(
        [args.tuning, args.resume]
    ), "Only support from_scrach or resume or tuning at one time"

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update(
        {
            k: v
            for k, v in args.__dict__.items()
            if k
            not in [
                "update",
            ]
            and v is not None
        }
    )

    cfg = YAMLConfig(args.config, **update_dict)
    configure_categories(cfg, args.test_only)

    if args.resume or args.tuning:
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if safe_get_rank() == 0:
        print("cfg: ")
        pprint(cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=False,
                        default="configs/upg_detr_m_visdrone.yml",)
    parser.add_argument("-r", "--resume", type=str, help="resume from checkpoint",

                        default=None
                        )
    parser.add_argument("-t", "--tuning", type=str, help="tuning from checkpoint")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        help="device",
        default="cuda"
    )
    parser.add_argument("--seed", type=int, help="exp reproducibility")
    parser.add_argument("--use-amp", action="store_true", default=None, help="auto mixed precision training")
    parser.add_argument("--output-dir", type=str, help="output directoy")
    parser.add_argument("--summary-dir", type=str, help="tensorboard summry")
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=False,
    )

    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")

    parser.add_argument("--print-method", type=str, default="builtin", help="print method")
    parser.add_argument("--print-rank", type=int, default=0, help="print rank id")

    parser.add_argument("--local-rank", type=int, help="local rank id")
    args = parser.parse_args()

    main(args)
