from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from app.config import GENERATED_DIR


def _to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)

    if arr.size == 0:
        return arr.astype(np.uint8)

    if arr.max() <= 1.0:
        arr = arr * 255.0

    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    arr = _to_uint8(image)

    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)

    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)

    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3]

    raise ValueError("Неподдерживаемый формат изображения для визуализации")


def save_image_array(image: np.ndarray) -> str:
    rgb = _ensure_rgb(image)
    file_name = f"{uuid4().hex}.png"
    file_path = GENERATED_DIR / file_name
    Image.fromarray(rgb).save(file_path)
    return f"/static/generated/{file_name}"


def resize_rgb_image(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    rgb = _ensure_rgb(image)
    pil = Image.fromarray(rgb)
    pil = pil.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)


def resize_binary_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    mask_uint8 = (_to_uint8(mask) > 0).astype(np.uint8) * 255
    pil = Image.fromarray(mask_uint8)
    pil = pil.resize((target_w, target_h), resample=Image.NEAREST)
    return (np.array(pil) > 127).astype(np.uint8)


def load_mask_png(mask_path: Path) -> np.ndarray:
    image = Image.open(mask_path)

    if image.mode not in ("L", "RGB", "RGBA"):
        image = image.convert("RGB")

    arr = np.array(image)

    if arr.ndim == 3:
        arr = arr[:, :, :3].max(axis=2)

    return (arr > 0).astype(np.uint8)


def ensure_same_shape(
    reference: np.ndarray,
    candidate: np.ndarray,
    reference_name: str,
    candidate_name: str,
) -> None:
    if reference.shape[:2] != candidate.shape[:2]:
        raise ValueError(
            f"Размеры {candidate_name} и {reference_name} не совпадают. "
            f"{reference_name}: {reference.shape[:2]}, {candidate_name}: {candidate.shape[:2]}. "
            "Сравнение невозможно."
        )


def crop_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return mask[y0:y1, x0:x1].astype(np.uint8)


def build_overlay(
    base_image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (220, 53, 69),
    alpha: float = 0.35,
) -> np.ndarray:
    base = _ensure_rgb(base_image).astype(np.float32)
    mask_bin = (mask > 0).astype(np.float32)

    if mask_bin.shape != base.shape[:2]:
        raise ValueError("Нельзя наложить маску: размеры изображения и маски не совпадают.")

    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    overlay = base.copy()

    overlay = np.where(
        mask_bin[..., None] > 0,
        (1.0 - alpha) * overlay + alpha * color_arr,
        overlay,
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def build_comparison_overlay(
    base_image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.40,
) -> np.ndarray:
    base = _ensure_rgb(base_image).astype(np.float32)

    if gt_mask.shape != base.shape[:2] or pred_mask.shape != base.shape[:2]:
        raise ValueError("Нельзя построить сравнение: размеры изображения и масок не совпадают.")

    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)

    gt_only = gt & (~pred)
    pred_only = pred & (~gt)
    overlap = gt & pred

    overlay = base.copy()

    gt_color = np.array([34, 139, 34], dtype=np.float32).reshape(1, 1, 3)
    pred_color = np.array([220, 53, 69], dtype=np.float32).reshape(1, 1, 3)
    overlap_color = np.array([255, 193, 7], dtype=np.float32).reshape(1, 1, 3)

    overlay = np.where(gt_only[..., None], (1.0 - alpha) * overlay + alpha * gt_color, overlay)
    overlay = np.where(pred_only[..., None], (1.0 - alpha) * overlay + alpha * pred_color, overlay)
    overlay = np.where(overlap[..., None], (1.0 - alpha) * overlay + alpha * overlap_color, overlay)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    pred = (pred_mask > 0).astype(np.float32)
    gt = (gt_mask > 0).astype(np.float32)

    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())

    if pred_sum == 0.0 and gt_sum == 0.0:
        return 1.0

    if pred_sum == 0.0 or gt_sum == 0.0:
        return 0.0

    intersection = float((pred * gt).sum())
    return float((2.0 * intersection + eps) / (pred_sum + gt_sum + eps))


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    pred = (pred_mask > 0).astype(np.float32)
    gt = (gt_mask > 0).astype(np.float32)

    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())

    if pred_sum == 0.0 and gt_sum == 0.0:
        return 1.0

    if pred_sum == 0.0 or gt_sum == 0.0:
        return 0.0

    intersection = float((pred * gt).sum())
    union = float(pred.sum() + gt.sum() - intersection)
    return float((intersection + eps) / (union + eps))


def count_positive_pixels(mask: np.ndarray) -> int:
    return int((mask > 0).sum())
