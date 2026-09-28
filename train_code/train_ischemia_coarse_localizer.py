import os
import json
import random
from pathlib import Path

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.cuda.amp import autocast, GradScaler

try:
    import pydicom
except ImportError as e:
    raise ImportError("Установить pydicom: pip install pydicom") from e

try:
    from transformers import SegformerForSemanticSegmentation
except ImportError as e:
    raise ImportError("Установить transformers: pip install transformers") from e



ROOT_DIR = "Brain_Stroke_CT_Dataset"

ISCHEMIA_DICOM_DIR = os.path.join(ROOT_DIR, "Ischemia", "DICOM")
ISCHEMIA_MASK_DIR = os.path.join(ROOT_DIR, "Ischemia", "MASKS_BIN")

NORMAL_DICOM_DIR = os.path.join(ROOT_DIR, "Normal", "DICOM")
NORMAL_MASK_DIR = os.path.join(ROOT_DIR, "Normal", "MASKS")

BLEEDING_DICOM_DIR = os.path.join(ROOT_DIR, "Bleeding", "DICOM")
BLEEDING_EMPTY_MASK_DIR = os.path.join(ROOT_DIR, "Bleeding", "MASKS_EMPTY")

RESULTS_DIR = "results_ischemia_coarse_localizer"
CHECKPOINT_DIR = "checkpoints_ischemia_coarse_localizer"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
THRESHOLDS_PATH = os.path.join(RESULTS_DIR, "thresholds.json")
VAL_METRICS_PATH = os.path.join(RESULTS_DIR, "val_metrics.json")

LOCAL_SEGFORMER_DIR = Path(__file__).resolve().parent.parent / "models" / "segformer_stage1_base"

MODEL_NAME = (
    str(LOCAL_SEGFORMER_DIR)
    if LOCAL_SEGFORMER_DIR.exists()
    else "nvidia/segformer-b3-finetuned-ade-512-512"
)


IMG_SIZE = 448
BATCH_SIZE = 2
EPOCHS = 140
LR = 8e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 35
NUM_WORKERS = 0

# Stage 1 учится на расширенной coarse-цели
COARSE_DILATION_KERNEL = 15

DEFAULT_SEG_THRESHOLD = 0.30
DEFAULT_MIN_POSITIVE_PIXELS = 48

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True



def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def dicom_stem(path):
    return os.path.splitext(os.path.basename(path))[0]

def list_files_flat(directory):
    files = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            files.append(path)
    return files

def load_dicom_hu(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array.astype(np.float32)

    if arr.ndim > 2:
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            arr = arr[0]

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept

    return hu

def window_hu(hu, level, width):
    lower = level - width / 2.0
    upper = level + width / 2.0
    hu = np.clip(hu, lower, upper)
    hu = (hu - lower) / (upper - lower + 1e-8)
    hu = (hu * 255.0).astype(np.uint8)
    return hu

def hu_to_model_rgb(hu):
    # Несколько окон под ишемию
    ch1 = window_hu(hu, 40, 80)
    ch2 = window_hu(hu, 35, 30)
    ch3 = window_hu(hu, 50, 130)
    return np.stack([ch1, ch2, ch3], axis=-1)

def compute_brain_bbox_from_hu(hu, threshold=10, margin=8):
    brain = window_hu(hu, 40, 80)
    mask = brain > threshold

    rows = mask.any(axis=1)
    cols = mask.any(axis=0)

    if not rows.any() or not cols.any():
        h, w = brain.shape
        return (0, 0, w, h)

    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]

    y0 = max(0, int(y_idx[0]) - margin)
    y1 = min(brain.shape[0], int(y_idx[-1]) + 1 + margin)
    x0 = max(0, int(x_idx[0]) - margin)
    x1 = min(brain.shape[1], int(x_idx[-1]) + 1 + margin)

    return (x0, y0, x1, y1)

def crop_hu(hu, bbox):
    x0, y0, x1, y1 = bbox
    return hu[y0:y1, x0:x1]

def crop_pil(img, bbox):
    return img.crop(bbox)

def model_rgb_to_pil(rgb):
    return Image.fromarray(rgb.astype(np.uint8))

def image_to_tensor(image_pil):
    image_pil = TF.resize(
        image_pil,
        [IMG_SIZE, IMG_SIZE],
        interpolation=InterpolationMode.BILINEAR
    )
    tensor = TF.to_tensor(image_pil)
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor

def load_mask_or_empty(mask_path, image_size):
    if mask_path is not None and os.path.exists(mask_path):
        return Image.open(mask_path).convert("L")
    w, h = image_size
    return Image.new("L", (w, h), 0)

def build_coarse_mask_pil(mask_pil):
    mask_pil = TF.resize(
        mask_pil,
        [IMG_SIZE, IMG_SIZE],
        interpolation=InterpolationMode.NEAREST
    )
    if mask_pil.getbbox() is not None:
        mask_pil = mask_pil.filter(ImageFilter.MaxFilter(COARSE_DILATION_KERNEL))
    return mask_pil

def fine_mask_to_tensor(mask_pil):
    mask_pil = TF.resize(
        mask_pil,
        [IMG_SIZE, IMG_SIZE],
        interpolation=InterpolationMode.NEAREST
    )
    mask = TF.to_tensor(mask_pil)
    return (mask > 0.5).float()

def coarse_mask_to_tensor(mask_pil):
    mask_pil = build_coarse_mask_pil(mask_pil)
    mask = TF.to_tensor(mask_pil)
    return (mask > 0.5).float()

def dilate_binary_mask_np(mask_np, kernel_size=COARSE_DILATION_KERNEL):
    if mask_np.sum() == 0:
        return mask_np.astype(np.uint8)

    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.MaxFilter(kernel_size))
    mask_np = (np.array(mask_img) > 127).astype(np.uint8)
    return mask_np

def resize_binary_mask(mask_np, size):
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_img = mask_img.resize((size, size), resample=Image.NEAREST)
    return (np.array(mask_img) > 127).astype(np.uint8)

def clean_pred_mask(pred_mask, min_positive_pixels):
    if pred_mask.sum() < min_positive_pixels:
        return np.zeros_like(pred_mask, dtype=np.uint8)
    return pred_mask.astype(np.uint8)

def dice_score(pred_mask, true_mask, eps=1e-7):
    pred = pred_mask.float().view(-1)
    true = true_mask.float().view(-1)

    pred_sum = pred.sum()
    true_sum = true.sum()

    if true_sum == 0 and pred_sum == 0:
        return 1.0
    if true_sum == 0 and pred_sum > 0:
        return 0.0

    inter = (pred * true).sum()
    dice = (2.0 * inter + eps) / (pred_sum + true_sum + eps)
    return float(dice.item())

def iou_score(pred_mask, true_mask, eps=1e-7):
    pred = pred_mask.float().view(-1)
    true = true_mask.float().view(-1)

    pred_sum = pred.sum()
    true_sum = true.sum()

    if true_sum == 0 and pred_sum == 0:
        return 1.0
    if true_sum == 0 and pred_sum > 0:
        return 0.0

    inter = (pred * true).sum()
    union = pred.sum() + true.sum() - inter
    iou = (inter + eps) / (union + eps)
    return float(iou.item())

def build_mask_path(mask_dir, dicom_path):
    return os.path.join(mask_dir, f"{dicom_stem(dicom_path)}.png")

def build_model(pretrained=True):
    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME,
            num_labels=1,
            ignore_mismatched_sizes=True
        )
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME,
            num_labels=1,
            ignore_mismatched_sizes=True
        )
    return model



def collect_items():
    items = []

    for dicom_path in list_files_flat(ISCHEMIA_DICOM_DIR):
        mask_path = build_mask_path(ISCHEMIA_MASK_DIR, dicom_path)
        if os.path.exists(mask_path):
            items.append({
                "dicom_path": dicom_path,
                "mask_path": mask_path,
                "label": 1,
                "group": 2,
                "source": "ischemia"
            })

    for dicom_path in list_files_flat(NORMAL_DICOM_DIR):
        mask_path = build_mask_path(NORMAL_MASK_DIR, dicom_path)
        if not os.path.exists(mask_path):
            mask_path = None

        items.append({
            "dicom_path": dicom_path,
            "mask_path": mask_path,
            "label": 0,
            "group": 0,
            "source": "normal"
        })

    for dicom_path in list_files_flat(BLEEDING_DICOM_DIR):
        mask_path = build_mask_path(BLEEDING_EMPTY_MASK_DIR, dicom_path)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Не найдена пустая маска: {mask_path}")

        items.append({
            "dicom_path": dicom_path,
            "mask_path": mask_path,
            "label": 0,
            "group": 1,
            "source": "bleeding_negative"
        })

    return items



class IschemiaCoarseDataset(Dataset):
    def __init__(self, items, train=False):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def apply_train_transforms(self, image, mask):
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        angle = random.uniform(-5, 5)
        image = TF.rotate(
            image,
            angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0
        )
        mask = TF.rotate(
            mask,
            angle,
            interpolation=InterpolationMode.NEAREST,
            fill=0
        )

        if random.random() < 0.25:
            image = TF.adjust_brightness(image, random.uniform(0.95, 1.05))
        if random.random() < 0.30:
            image = TF.adjust_contrast(image, random.uniform(0.95, 1.10))

        return image, mask

    def __getitem__(self, idx):
        item = self.items[idx]

        hu = load_dicom_hu(item["dicom_path"])
        bbox = compute_brain_bbox_from_hu(hu)
        hu = crop_hu(hu, bbox)

        mask = load_mask_or_empty(item["mask_path"], (hu.shape[1], hu.shape[0]))
        mask = crop_pil(mask, bbox)

        image = model_rgb_to_pil(hu_to_model_rgb(hu))

        if self.train:
            image, mask = self.apply_train_transforms(image, mask)

        image_tensor = image_to_tensor(image)
        coarse_mask_tensor = coarse_mask_to_tensor(mask)
        label_tensor = torch.tensor(item["label"], dtype=torch.float32)

        return image_tensor, coarse_mask_tensor, label_tensor



class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.25, beta=0.75, gamma=1.33, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)

        tversky = (tp + self.eps) / (
            tp + self.alpha * fp + self.beta * fn + self.eps
        )

        return ((1 - tversky) ** self.gamma).mean()

def compute_loss(logits, targets, bce_loss, focal_tversky_loss):
    loss_bce = bce_loss(logits, targets)
    loss_ft = focal_tversky_loss(logits, targets)
    total_loss = 0.45 * loss_bce + 0.55 * loss_ft
    return total_loss



def evaluate_model(model, loader, seg_threshold=0.30, min_positive_pixels=48):
    model.eval()

    bce_loss = nn.BCEWithLogitsLoss()
    focal_tversky_loss = FocalTverskyLoss(alpha=0.25, beta=0.75, gamma=1.33)

    total_loss = 0.0
    positive_dices = []
    positive_ious = []

    tp_img = 0
    fn_img = 0
    fp_img = 0
    tn_img = 0

    with torch.no_grad():
        for imgs, masks, labels in loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(pixel_values=imgs)
            logits = outputs.logits
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            loss = compute_loss(logits, masks, bce_loss, focal_tversky_loss)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs >= seg_threshold).float()

            for pred_mask, true_mask in zip(preds, masks):
                pred_mask_np = pred_mask[0].cpu().numpy().astype(np.uint8)
                pred_mask_np = clean_pred_mask(pred_mask_np, min_positive_pixels)
                pred_mask_t = torch.from_numpy(pred_mask_np).float()

                gt_positive = true_mask.sum().item() > 0
                pred_positive = pred_mask_t.sum().item() > 0

                if gt_positive:
                    positive_dices.append(dice_score(pred_mask_t, true_mask.cpu()))
                    positive_ious.append(iou_score(pred_mask_t, true_mask.cpu()))

                    if pred_positive:
                        tp_img += 1
                    else:
                        fn_img += 1
                else:
                    if pred_positive:
                        fp_img += 1
                    else:
                        tn_img += 1

    avg_loss = total_loss / len(loader)
    positive_dice = float(np.mean(positive_dices)) if positive_dices else 0.0
    positive_iou = float(np.mean(positive_ious)) if positive_ious else 0.0

    negative_total = fp_img + tn_img
    positive_total = tp_img + fn_img

    negative_fp_rate = fp_img / negative_total if negative_total > 0 else 0.0
    ischemia_recall = tp_img / positive_total if positive_total > 0 else 0.0
    ischemia_precision = tp_img / (tp_img + fp_img) if (tp_img + fp_img) > 0 else 0.0

    score = (
        positive_dice +
        0.20 * ischemia_recall +
        0.05 * ischemia_precision -
        0.15 * negative_fp_rate
    )

    return {
        "loss": avg_loss,
        "positive_dice": positive_dice,
        "positive_iou": positive_iou,
        "negative_fp_rate": negative_fp_rate,
        "ischemia_recall": ischemia_recall,
        "ischemia_precision": ischemia_precision,
        "tp_img": tp_img,
        "fn_img": fn_img,
        "fp_img": fp_img,
        "tn_img": tn_img,
        "score": score
    }

def search_best_thresholds(model, dataset):
    search_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    thresholds = np.linspace(0.20, 0.55, 8)
    min_pixels_candidates = [16, 24, 32, 48, 64, 96]

    best_metrics = None

    for threshold in thresholds:
        for min_pixels in min_pixels_candidates:
            metrics = evaluate_model(
                model,
                search_loader,
                seg_threshold=float(threshold),
                min_positive_pixels=int(min_pixels)
            )
            metrics["seg_threshold"] = float(threshold)
            metrics["min_positive_pixels"] = int(min_pixels)

            if best_metrics is None or metrics["score"] > best_metrics["score"]:
                best_metrics = metrics

    return best_metrics



def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Device:", DEVICE)

    items = collect_items()

    labels = [item["label"] for item in items]
    groups = [item["group"] for item in items]

    print("Всего изображений:", len(items))
    print("Распределение по presence-классам [negative, positive]:", np.bincount(labels))
    print("Распределение по группам [Normal, BleedingNeg, Ischemia]:", np.bincount(groups))

    train_items, val_items = train_test_split(
        items,
        test_size=0.2,
        stratify=groups,
        random_state=SEED
    )

    train_dataset = IschemiaCoarseDataset(train_items, train=True)
    val_dataset = IschemiaCoarseDataset(val_items, train=False)

    train_groups = np.array([item["group"] for item in train_items])
    group_counts = np.bincount(train_groups)
    group_weights = 1.0 / group_counts
    sample_weights = [group_weights[item["group"]] for item in train_items]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    model = build_model(pretrained=True).to(DEVICE)

    bce_loss = nn.BCEWithLogitsLoss()
    focal_tversky_loss = FocalTverskyLoss(alpha=0.25, beta=0.75, gamma=1.33)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=6
    )

    scaler = GradScaler(enabled=USE_AMP)

    best_score = -1.0
    best_epoch = 0
    early_stop_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_positive_dice": [],
        "val_positive_iou": [],
        "val_negative_fp_rate": [],
        "val_ischemia_recall": [],
        "val_ischemia_precision": [],
        "val_score": []
    }

    save_json(
        THRESHOLDS_PATH,
        {
            "seg_threshold": DEFAULT_SEG_THRESHOLD,
            "min_positive_pixels": DEFAULT_MIN_POSITIVE_PIXELS,
            "coarse_dilation_kernel": COARSE_DILATION_KERNEL,
            "note": "fallback thresholds"
        }
    )

    for epoch in range(EPOCHS):
        model.train()
        train_loss_total = 0.0

        for imgs, masks, _ in train_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=USE_AMP):
                outputs = model(pixel_values=imgs)
                logits = outputs.logits
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                loss = compute_loss(logits, masks, bce_loss, focal_tversky_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item()

        train_loss = train_loss_total / len(train_loader)

        val_metrics = evaluate_model(
            model,
            val_loader,
            seg_threshold=DEFAULT_SEG_THRESHOLD,
            min_positive_pixels=DEFAULT_MIN_POSITIVE_PIXELS
        )
        val_score = val_metrics["score"]

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_score)
        new_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_positive_dice"].append(val_metrics["positive_dice"])
        history["val_positive_iou"].append(val_metrics["positive_iou"])
        history["val_negative_fp_rate"].append(val_metrics["negative_fp_rate"])
        history["val_ischemia_recall"].append(val_metrics["ischemia_recall"])
        history["val_ischemia_precision"].append(val_metrics["ischemia_precision"])
        history["val_score"].append(val_score)

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Dice(Coarse Ischemia): {val_metrics['positive_dice']:.4f} | "
            f"IoU(Coarse Ischemia): {val_metrics['positive_iou']:.4f}"
        )
        print(
            f"Recall(Ischemia image-level): {val_metrics['ischemia_recall']:.4f} | "
            f"Precision(Ischemia image-level): {val_metrics['ischemia_precision']:.4f} | "
            f"FP rate on negatives: {val_metrics['negative_fp_rate']:.4f}"
        )
        print(f"Val Score: {val_score:.4f}")
        print(f"Текущий LR: {new_lr:.6f}")

        if new_lr != old_lr:
            print(f"Learning rate уменьшен с {old_lr:.6f} до {new_lr:.6f}")

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            early_stop_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Лучшая модель сохранена в эпоху {best_epoch}")
        else:
            early_stop_counter += 1
            print(f"Никаких улучшений {early_stop_counter} эпохи")

        if early_stop_counter >= PATIENCE:
            print("\nEarly stopping сработал")
            print(f"Лучшая эпоха: {best_epoch}")
            break

    print("\nОбучение завершено")

    pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, "history.csv"), index=False)

    plt.figure()
    plt.plot(history["train_loss"])
    plt.plot(history["val_loss"])
    plt.title("Loss")
    plt.legend(["Train", "Val"])
    plt.savefig(os.path.join(RESULTS_DIR, "loss.png"))
    plt.close()

    plt.figure()
    plt.plot(history["val_positive_dice"])
    plt.title("Validation Coarse Dice")
    plt.savefig(os.path.join(RESULTS_DIR, "val_coarse_dice.png"))
    plt.close()

    plt.figure()
    plt.plot(history["val_negative_fp_rate"])
    plt.title("Validation FP Rate on Negatives")
    plt.savefig(os.path.join(RESULTS_DIR, "val_fp_rate_negatives.png"))
    plt.close()

    best_model = build_model(pretrained=False).to(DEVICE)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    best_model.eval()

    fallback_metrics = evaluate_model(
        best_model,
        val_loader,
        seg_threshold=DEFAULT_SEG_THRESHOLD,
        min_positive_pixels=DEFAULT_MIN_POSITIVE_PIXELS
    )

    try:
        best_threshold_metrics = search_best_thresholds(best_model, val_dataset)
        note = "threshold search completed successfully"
    except Exception as e:
        best_threshold_metrics = {
            "seg_threshold": DEFAULT_SEG_THRESHOLD,
            "min_positive_pixels": DEFAULT_MIN_POSITIVE_PIXELS,
            **fallback_metrics
        }
        note = f"threshold search failed, fallback used: {str(e)}"

    thresholds_data = {
        "seg_threshold": float(best_threshold_metrics["seg_threshold"]),
        "min_positive_pixels": int(best_threshold_metrics["min_positive_pixels"]),
        "positive_dice": float(best_threshold_metrics["positive_dice"]),
        "positive_iou": float(best_threshold_metrics["positive_iou"]),
        "negative_fp_rate": float(best_threshold_metrics["negative_fp_rate"]),
        "ischemia_recall": float(best_threshold_metrics["ischemia_recall"]),
        "ischemia_precision": float(best_threshold_metrics["ischemia_precision"]),
        "coarse_dilation_kernel": int(COARSE_DILATION_KERNEL),
        "note": note
    }
    save_json(THRESHOLDS_PATH, thresholds_data)
    save_json(VAL_METRICS_PATH, thresholds_data)

    print("\n  Лучшие thresholds на валидации  ")
    print(f"Seg threshold: {best_threshold_metrics['seg_threshold']:.2f}")
    print(f"Min positive pixels: {best_threshold_metrics['min_positive_pixels']}")
    print(f"Dice(Coarse Ischemia): {best_threshold_metrics['positive_dice']:.4f}")
    print(f"IoU(Coarse Ischemia): {best_threshold_metrics['positive_iou']:.4f}")
    print(f"Recall(Ischemia image-level): {best_threshold_metrics['ischemia_recall']:.4f}")
    print(f"Precision(Ischemia image-level): {best_threshold_metrics['ischemia_precision']:.4f}")
    print(f"FP rate on negatives: {best_threshold_metrics['negative_fp_rate']:.4f}")
    print(f"Note: {note}")

    print("\nРезультаты обучения сохранены в:", RESULTS_DIR)
    print("Порог сохранен в:", THRESHOLDS_PATH)

if __name__ == "__main__":
    main()