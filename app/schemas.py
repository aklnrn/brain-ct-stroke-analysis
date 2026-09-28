from dataclasses import dataclass
from typing import Optional


@dataclass
class CascadeResult:
    final_class_id: int
    final_class_name_ru: str
    lesion_detected: bool
    stroke_probability_percent: float
    confidence_percent: float
    route_taken: str
    route_description_ru: str
    original_image_url: str = ""
    prediction_overlay_url: str = ""

    has_mask_comparison: bool = False
    dice: Optional[float] = None
    iou: Optional[float] = None
    gt_pixels: Optional[int] = None
    pred_pixels: Optional[int] = None
    comparison_status: str = ""
    gt_overlay_url: str = ""
    comparison_overlay_url: str = ""
