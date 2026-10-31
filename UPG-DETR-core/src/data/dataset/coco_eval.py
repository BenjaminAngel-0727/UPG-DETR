from faster_coco_eval.utils.pytorch import FasterCocoEvaluator

from ...core import register

__all__ = [
    "CocoEvaluator",
]

@register()
class CocoEvaluator(FasterCocoEvaluator):
    pass
