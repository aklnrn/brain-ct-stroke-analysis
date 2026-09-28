from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.config import CLASS_LABELS_RU, ROUTE_LABELS_RU
from app.schemas import CascadeResult
from app.services.model_loader import get_model_registry
from app.services.visualization import (
    build_comparison_overlay,
    build_overlay,
    compute_dice,
    compute_iou,
    count_positive_pixels,
    crop_mask,
    ensure_same_shape,
    load_mask_png,
    resize_binary_mask,
    save_image_array,
)


class CascadeService:
    def __init__(self):
        self.registry = None

    def load_models(self):
        if self.registry is None:
            self.registry = get_model_registry()

    @staticmethod
    def _model_device(model: torch.nn.Module) -> torch.device:
        return next(model.parameters()).device

    @staticmethod
    def _sigmoid_scalar(tensor: torch.Tensor) -> float:
        return float(torch.sigmoid(tensor).view(-1)[0].item())

    @staticmethod
    def _softmax_vector(tensor: torch.Tensor) -> np.ndarray:
        probs = torch.softmax(tensor, dim=1).detach().cpu().numpy()[0]
        return probs.astype(np.float32)

    @staticmethod
    def _load_png_rgb(png_path: Path) -> np.ndarray:
        image = Image.open(png_path).convert("RGB")
        return np.array(image, dtype=np.uint8)

    def _load_binary_route_threshold(self) -> float:
        thresholds = self.registry.binary_thresholds or {}
        return float(thresholds.get("route_threshold", 0.5))

    def _load_bleeding_threshold(self) -> float:
        thresholds = self.registry.bleeding_thresholds or {}
        return float(thresholds.get("best_threshold", 0.5))

    def _load_stage1_thresholds(self) -> tuple[float, int]:
        thresholds = self.registry.stage1_thresholds or {}
        seg_threshold = float(thresholds.get("seg_threshold", 0.3))
        min_positive_pixels = int(thresholds.get("min_positive_pixels", 48))
        return seg_threshold, min_positive_pixels

    def _load_stage2_thresholds(self) -> tuple[float, float, int]:
        thresholds = self.registry.stage2_thresholds or {}
        seg_threshold = float(thresholds.get("seg_threshold", 0.35))
        presence_threshold = float(thresholds.get("presence_threshold", 0.45))
        min_positive_pixels = int(thresholds.get("min_positive_pixels", 24))
        return seg_threshold, presence_threshold, min_positive_pixels

    def _predict_binary_stroke_prob(self, png_path: Path) -> float:
        module = self.registry.binary_module
        model = self.registry.binary_model
        device = self._model_device(model)

        image = Image.open(png_path).convert("RGB")
        tensor = module.val_transform(image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            logits = model(tensor)

        return self._sigmoid_scalar(logits)

    def _predict_type_probs(self, png_path: Path) -> np.ndarray:
        module = self.registry.type_module
        model = self.registry.type_model
        device = self._model_device(model)

        image = Image.open(png_path).convert("RGB")
        tensor = module.val_transform(image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            logits = model(tensor)

        return self._softmax_vector(logits)

    def _predict_bleeding_mask(self, png_path: Path) -> tuple[np.ndarray, float]:
        module = self.registry.bleeding_module
        model = self.registry.bleeding_model
        device = self._model_device(model)
        threshold = self._load_bleeding_threshold()

        image = Image.open(png_path).convert("RGB")
        tensor = module.image_to_tensor(image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        pred_mask = (probs >= threshold).astype(np.uint8)
        max_prob = float(np.max(probs)) if probs.size else 0.0
        return pred_mask, max_prob

    def _predict_stage1_coarse_mask(
        self,
        full_hu: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        module = self.registry.stage1_module
        model = self.registry.stage1_model
        device = self._model_device(model)

        seg_threshold, min_positive_pixels = self._load_stage1_thresholds()

        brain_bbox = module.compute_brain_bbox_from_hu(full_hu)
        hu_crop = module.crop_hu(full_hu, brain_bbox)

        image_pil = module.model_rgb_to_pil(module.hu_to_model_rgb(hu_crop))
        tensor = module.image_to_tensor(image_pil).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(pixel_values=tensor)
            logits = outputs.logits
            logits = F.interpolate(
                logits,
                size=(module.IMG_SIZE, module.IMG_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        pred_mask = (probs >= seg_threshold).astype(np.uint8)
        pred_mask = module.clean_pred_mask(pred_mask, min_positive_pixels)
        return pred_mask, hu_crop, brain_bbox

    def _stage1_mask_to_roi_bbox(self, coarse_mask: np.ndarray, hu_crop: np.ndarray):
        stage2_module = self.registry.stage2_module

        coarse_bbox = stage2_module.mask_to_bbox(coarse_mask)
        if coarse_bbox is None:
            return None

        mask_h, mask_w = coarse_mask.shape
        hu_h, hu_w = hu_crop.shape

        x0, y0, x1, y1 = coarse_bbox

        scale_x = hu_w / float(mask_w)
        scale_y = hu_h / float(mask_h)

        roi_bbox = (
            int(round(x0 * scale_x)),
            int(round(y0 * scale_y)),
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
        )

        return stage2_module.expand_bbox(roi_bbox, hu_w, hu_h)

    @staticmethod
    def _paste_roi_mask_back(full_shape: tuple[int, int], roi_bbox, roi_mask: np.ndarray) -> np.ndarray:
        full_h, full_w = full_shape
        x0, y0, x1, y1 = roi_bbox

        roi_h = max(y1 - y0, 1)
        roi_w = max(x1 - x0, 1)

        resized_roi_mask = resize_binary_mask(roi_mask, (roi_h, roi_w))

        full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = resized_roi_mask[: y1 - y0, : x1 - x0]
        return full_mask

    def _predict_ischemia_mask(
        self,
        full_hu: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray, tuple[int, int, int, int]]:
        stage2_module = self.registry.stage2_module
        stage2_model = self.registry.stage2_model
        device = self._model_device(stage2_model)

        coarse_mask, hu_crop, brain_bbox = self._predict_stage1_coarse_mask(full_hu)
        roi_bbox = self._stage1_mask_to_roi_bbox(coarse_mask, hu_crop)

        if roi_bbox is None:
            empty = np.zeros(hu_crop.shape, dtype=np.uint8)
            return empty, 0.0, hu_crop, brain_bbox

        seg_threshold, presence_threshold, min_positive_pixels = self._load_stage2_thresholds()

        roi_hu = stage2_module.crop_hu(hu_crop, roi_bbox)
        image_pil = stage2_module.model_rgb_to_pil(stage2_module.hu_to_model_rgb(roi_hu))
        tensor = stage2_module.image_to_tensor(image_pil).unsqueeze(0).to(device)

        stage2_model.eval()
        with torch.no_grad():
            output = stage2_model(tensor)
            seg_logits, cls_logits = stage2_module.unpack_model_output(output)
            seg_probs = torch.sigmoid(seg_logits)[0, 0].detach().cpu().numpy()

            if cls_logits is not None:
                presence_prob = self._sigmoid_scalar(cls_logits)
            else:
                presence_prob = 1.0

        pred_roi_mask = (seg_probs >= seg_threshold).astype(np.uint8)

        if presence_prob < presence_threshold:
            pred_roi_mask = np.zeros_like(pred_roi_mask, dtype=np.uint8)

        pred_roi_mask = stage2_module.clean_pred_mask(pred_roi_mask, min_positive_pixels)
        pred_full_mask = self._paste_roi_mask_back(hu_crop.shape, roi_bbox, pred_roi_mask)

        return pred_full_mask, float(presence_prob), hu_crop, brain_bbox

    def _prepare_ischemia_display(self, hu_crop: np.ndarray) -> np.ndarray:
        module = self.registry.stage1_module
        window = module.window_hu(hu_crop, 40, 80)
        return np.stack([window, window, window], axis=-1).astype(np.uint8)

    @staticmethod
    def _comparison_status(dice: float, gt_pixels: int, pred_pixels: int) -> str:
        if gt_pixels == 0 and pred_pixels == 0:
            return "очаг отсутствует в обеих масках"

        if dice >= 0.80:
            return "высокая"

        if dice >= 0.50:
            return "умеренная"

        return "низкая"


    @staticmethod
    def _build_confidence(
        final_class_id: int,
        route_taken: str,
        binary_prob: float,
        type_probs: np.ndarray | None = None,
        branch_prob: float | None = None,
    ) -> float:
        if final_class_id == 0:
            if route_taken == "stopped_by_binary":
                return max(50.0, (1.0 - binary_prob) * 100.0)

            if route_taken == "type_normal" and type_probs is not None:
                return max(50.0, float(type_probs[0]) * 100.0)

            if branch_prob is not None:
                return max(50.0, (1.0 - branch_prob) * 100.0)

            return 50.0

        if final_class_id == 1:
            candidates = []
            if type_probs is not None:
                candidates.append(float(type_probs[1]))
            if branch_prob is not None:
                candidates.append(float(branch_prob))
            return max(50.0, max(candidates or [0.5]) * 100.0)

        if final_class_id == 2:
            candidates = []
            if type_probs is not None:
                candidates.append(float(type_probs[2]))
            if branch_prob is not None:
                candidates.append(float(branch_prob))
            return max(50.0, max(candidates or [0.5]) * 100.0)

        return 50.0

    def run_analysis(
        self,
        png_path: Path,
        dicom_path: Path,
        mask_path: Path | None = None,
    ) -> CascadeResult:
        self.load_models()

        stage1_module = self.registry.stage1_module

        png_display = self._load_png_rgb(png_path)
        full_hu = stage1_module.load_dicom_hu(str(dicom_path))

        ensure_same_shape(
            reference=full_hu,
            candidate=png_display,
            reference_name="DICOM-файла",
            candidate_name="PNG-изображения",
        )

        gt_full_mask = None
        if mask_path is not None:
            gt_full_mask = load_mask_png(mask_path)
            ensure_same_shape(
                reference=full_hu,
                candidate=gt_full_mask,
                reference_name="DICOM-файла",
                candidate_name="эталонной маски",
            )

        binary_prob = self._predict_binary_stroke_prob(png_path)
        route_threshold = self._load_binary_route_threshold()

        final_class_id = 0
        lesion_detected = False
        route_taken = "stopped_by_binary"
        type_probs = None
        branch_prob = None

        base_display = png_display
        pred_mask = np.zeros(png_display.shape[:2], dtype=np.uint8)
        gt_mask_for_comparison = gt_full_mask.copy() if gt_full_mask is not None else None

        if binary_prob >= route_threshold:
            type_probs = self._predict_type_probs(png_path)
            type_idx = int(np.argmax(type_probs))

            if type_idx == 0:
                final_class_id = 0
                lesion_detected = False
                route_taken = "type_normal"
                pred_mask = np.zeros(png_display.shape[:2], dtype=np.uint8)

            elif type_idx == 1:
                pred_mask_raw, bleeding_peak_prob = self._predict_bleeding_mask(png_path)
                branch_prob = bleeding_peak_prob

                pred_mask = resize_binary_mask(pred_mask_raw, png_display.shape[:2])
                base_display = png_display

                if pred_mask.sum() > 0:
                    final_class_id = 1
                    lesion_detected = True
                    route_taken = "bleeding_segmenter_positive"
                else:
                    final_class_id = 0
                    lesion_detected = False
                    route_taken = "bleeding_segmenter_negative"

            else:
                pred_mask, stage2_presence_prob, hu_crop, brain_bbox = self._predict_ischemia_mask(full_hu)
                branch_prob = stage2_presence_prob
                base_display = self._prepare_ischemia_display(hu_crop)

                if gt_full_mask is not None:
                    gt_mask_for_comparison = crop_mask(gt_full_mask, brain_bbox)

                if pred_mask.sum() > 0:
                    final_class_id = 2
                    lesion_detected = True
                    route_taken = "ischemia_stage1_stage2_positive"
                else:
                    final_class_id = 0
                    lesion_detected = False
                    route_taken = "ischemia_stage1_stage2_negative"

        confidence = self._build_confidence(
            final_class_id=final_class_id,
            route_taken=route_taken,
            binary_prob=binary_prob,
            type_probs=type_probs,
            branch_prob=branch_prob,
        )

        original_image_url = save_image_array(base_display)

        if pred_mask.shape != base_display.shape[:2]:
            raise ValueError(
                "Размер предсказанной маски не совпадает с размером изображения для отображения. "
                "Наложение невозможно."
            )

        if pred_mask.sum() > 0:
            prediction_overlay = build_overlay(base_display, pred_mask, color=(220, 53, 69), alpha=0.35)
        else:
            prediction_overlay = base_display

        prediction_overlay_url = save_image_array(prediction_overlay)

        has_mask_comparison = False
        dice = None
        iou = None
        gt_pixels = None
        pred_pixels = None
        comparison_status = ""
        gt_overlay_url = ""
        comparison_overlay_url = ""

        if gt_mask_for_comparison is not None:
            ensure_same_shape(
                reference=base_display,
                candidate=gt_mask_for_comparison,
                reference_name="изображения для отображения",
                candidate_name="эталонной маски",
            )

            ensure_same_shape(
                reference=gt_mask_for_comparison,
                candidate=pred_mask,
                reference_name="эталонной маски",
                candidate_name="предсказанной маски",
            )

            dice = compute_dice(pred_mask, gt_mask_for_comparison)
            iou = compute_iou(pred_mask, gt_mask_for_comparison)
            gt_pixels = count_positive_pixels(gt_mask_for_comparison)
            pred_pixels = count_positive_pixels(pred_mask)
            comparison_status = self._comparison_status(dice, gt_pixels, pred_pixels)

            has_mask_comparison = True

            gt_overlay = build_overlay(
                base_display,
                gt_mask_for_comparison,
                color=(34, 139, 34),
                alpha=0.35,
            )
            comparison_overlay = build_comparison_overlay(
                base_display,
                gt_mask_for_comparison,
                pred_mask,
                alpha=0.40,
            )

            gt_overlay_url = save_image_array(gt_overlay)
            comparison_overlay_url = save_image_array(comparison_overlay)

        return CascadeResult(
            final_class_id=final_class_id,
            final_class_name_ru=CLASS_LABELS_RU[final_class_id],
            lesion_detected=lesion_detected,
            stroke_probability_percent=binary_prob * 100.0,
            confidence_percent=confidence,
            route_taken=route_taken,
            route_description_ru=ROUTE_LABELS_RU[route_taken],
            original_image_url=original_image_url,
            prediction_overlay_url=prediction_overlay_url,
            has_mask_comparison=has_mask_comparison,
            dice=dice,
            iou=iou,
            gt_pixels=gt_pixels,
            pred_pixels=pred_pixels,
            comparison_status=comparison_status,
            gt_overlay_url=gt_overlay_url,
            comparison_overlay_url=comparison_overlay_url,
        )
