import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

try:
    import pydicom
except ImportError as e:
    raise ImportError("Установить pydicom: pip install pydicom") from e

try:
    import segmentation_models_pytorch as smp
except ImportError as e:
    raise ImportError(
        "Установить segmentation-models-pytorch: pip install segmentation-models-pytorch"
    ) from e



ROOT_DIR = "Brain_Stroke_CT_Dataset"

ISCHEMIA_DICOM_DIR = os.path.join(ROOT_DIR, "Ischemia", "DICOM")
ISCHEMIA_MASK_DIR = os.path.join(ROOT_DIR, "Ischemia", "MASKS_BIN")

NORMAL_DICOM_DIR = os.path.join(ROOT_DIR, "Normal", "DICOM")
NORMAL_MASK_DIR = os.path.join(ROOT_DIR, "Normal", "MASKS")

BLEEDING_DICOM_DIR = os.path.join(ROOT_DIR, "Bleeding", "DICOM")
BLEEDING_EMPTY_MASK_DIR = os.path.join(ROOT_DIR, "Bleeding", "MASKS_EMPTY")

STAGE1_RESULTS_DIR = "results_ischemia_coarse_localizer"
STAGE1_THRESHOLDS_PATH = os.path.join(STAGE1_RESULTS_DIR, "thresholds.json")

RESULTS_DIR = "results_ischemia_refinement"
CHECKPOINT_DIR = "checkpoints_ischemia_refinement"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
THRESHOLDS_PATH = os.path.join(RESULTS_DIR, "thresholds.json")
VAL_METRICS_PATH = os.path.join(RESULTS_DIR, "val_metrics.json")

ROI_SIZE = 320
BATCH_SIZE = 6
EPOCHS = 120
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 25
NUM_WORKERS = 0

DEFAULT_SEG_THRESHOLD = 0.35
DEFAULT_PRESENCE_THRESHOLD = 0.45
DEFAULT_MIN_POSITIVE_PIXELS = 24

ROI_MARGIN_RATIO = 0.25
NEGATIVE_RANDOM_CROP_MIN_RATIO = 0.28
NEGATIVE_RANDOM_CROP_MAX_RATIO = 0.50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        [ROI_SIZE, ROI_SIZE],
        interpolation=InterpolationMode.BILINEAR
    )
    tensor = TF.to_tensor(image_pil)
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor

def mask_to_tensor(mask_pil):
    mask_pil = TF.resize(
        mask_pil,
        [ROI_SIZE, ROI_SIZE],
        interpolation=InterpolationMode.NEAREST
    )
    mask = TF.to_tensor(mask_pil)
    return (mask > 0.5).float()

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

def load_mask_or_empty(mask_path, image_size):
    if mask_path is not None and os.path.exists(mask_path):
        return Image.open(mask_path).convert("L")
    w, h = image_size
    return Image.new("L", (w, h), 0)

def build_mask_path(mask_dir, dicom_path):
    return os.path.join(mask_dir, f"{dicom_stem(dicom_path)}.png")

def mask_to_bbox(mask_np):
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return (x0, y0, x1, y1)

def expand_bbox(bbox, width, height, margin_ratio=ROI_MARGIN_RATIO):
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0

    mx = max(8, int(bw * margin_ratio))
    my = max(8, int(bh * margin_ratio))

    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(width, x1 + mx)
    y1 = min(height, y1 + my)

    return (x0, y0, x1, y1)

def jitter_bbox(bbox, width, height, jitter_ratio=0.10):
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0

    dx = int(bw * jitter_ratio)
    dy = int(bh * jitter_ratio)

    shift_x0 = random.randint(-dx, dx) if dx > 0 else 0
    shift_y0 = random.randint(-dy, dy) if dy > 0 else 0
    shift_x1 = random.randint(-dx, dx) if dx > 0 else 0
    shift_y1 = random.randint(-dy, dy) if dy > 0 else 0

    x0 = max(0, x0 + shift_x0)
    y0 = max(0, y0 + shift_y0)
    x1 = min(width, x1 + shift_x1)
    y1 = min(height, y1 + shift_y1)

    if x1 <= x0 + 4:
        x1 = min(width, x0 + 4)
    if y1 <= y0 + 4:
        y1 = min(height, y0 + 4)

    return (x0, y0, x1, y1)

def random_negative_roi_bbox(width, height):
    crop_w = int(width * random.uniform(NEGATIVE_RANDOM_CROP_MIN_RATIO, NEGATIVE_RANDOM_CROP_MAX_RATIO))
    crop_h = int(height * random.uniform(NEGATIVE_RANDOM_CROP_MIN_RATIO, NEGATIVE_RANDOM_CROP_MAX_RATIO))

    crop_w = max(64, min(crop_w, width))
    crop_h = max(64, min(crop_h, height))

    if width == crop_w:
        x0 = 0
    else:
        x0 = random.randint(0, width - crop_w)

    if height == crop_h:
        y0 = 0
    else:
        y0 = random.randint(0, height - crop_h)

    return (x0, y0, x0 + crop_w, y0 + crop_h)

def build_model(pretrained=True):
    encoder_weights = "imagenet" if pretrained else None
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        decoder_attention_type="scse",
        aux_params={
            "pooling": "avg",
            "dropout": 0.2,
            "activation": None,
            "classes": 1
        }
    )
    return model

def unpack_model_output(output):
    if isinstance(output, (tuple, list)):
        if len(output) == 2:
            return output[0], output[1]
        return output[0], None
    return output, None



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



class IschemiaRefinementDataset(Dataset):
    def __init__(self, items, train=False):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def apply_train_transforms(self, image, mask):
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        angle = random.uniform(-7, 7)
        image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        if random.random() < 0.30:
            image = TF.adjust_brightness(image, random.uniform(0.95, 1.05))
        if random.random() < 0.35:
            image = TF.adjust_contrast(image, random.uniform(0.95, 1.12))

        return image, mask

    def __getitem__(self, idx):
        item = self.items[idx]

        hu = load_dicom_hu(item["dicom_path"])
        brain_bbox = compute_brain_bbox_from_hu(hu)
        hu = crop_hu(hu, brain_bbox)

        mask = load_mask_or_empty(item["mask_path"], (hu.shape[1], hu.shape[0]))
        mask = crop_pil(mask, brain_bbox)

        mask_np = (np.array(mask) > 127).astype(np.uint8)
        h, w = mask_np.shape

        if item["label"] == 1:
            lesion_bbox = mask_to_bbox(mask_np)
            if lesion_bbox is None:
                roi_bbox = random_negative_roi_bbox(w, h)
                presence_label = 0.0
            else:
                roi_bbox = expand_bbox(lesion_bbox, w, h, margin_ratio=ROI_MARGIN_RATIO)
                if self.train:
                    roi_bbox = jitter_bbox(roi_bbox, w, h, jitter_ratio=0.12)
                presence_label = 1.0
        else:
            roi_bbox = random_negative_roi_bbox(w, h)
            presence_label = 0.0

        hu_roi = crop_hu(hu, roi_bbox)
        mask_roi = crop_pil(mask, roi_bbox)

        image = model_rgb_to_pil(hu_to_model_rgb(hu_roi))

        if self.train:
            image, mask_roi = self.apply_train_transforms(image, mask_roi)

        image_tensor = image_to_tensor(image)
        mask_tensor = mask_to_tensor(mask_roi)
        label_tensor = torch.tensor(presence_label, dtype=torch.float32)

        return image_tensor, mask_tensor, label_tensor



class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.20, beta=0.80, gamma=1.33, eps=1e-7):
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

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def edges(self, x):
        gx = F.conv2d(x, self.sobel_x.to(x.device), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(x.device), padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pred_edges = self.edges(probs)
        true_edges = self.edges(targets)
        return F.l1_loss(pred_edges, true_edges)

def compute_loss(
    seg_logits,
    seg_targets,
    cls_logits,
    cls_targets,
    seg_bce_loss,
    focal_tversky_loss,
    boundary_loss,
    aux_bce_loss
):
    loss_bce = seg_bce_loss(seg_logits, seg_targets)
    loss_ft = focal_tversky_loss(seg_logits, seg_targets)
    loss_boundary = boundary_loss(seg_logits, seg_targets)

    if cls_logits is not None:
        cls_logits = cls_logits.view(-1, 1)
        cls_targets = cls_targets.view(-1, 1)
        loss_aux = aux_bce_loss(cls_logits, cls_targets)
    else:
        loss_aux = torch.tensor(0.0, device=seg_logits.device)

    total_loss = (
        0.30 * loss_bce +
        0.40 * loss_ft +
        0.15 * loss_boundary +
        0.15 * loss_aux
    )

    return total_loss



def evaluate_model(model, loader, seg_threshold=0.35, presence_threshold=0.45, min_positive_pixels=24):
    model.eval()

    seg_bce_loss = nn.BCEWithLogitsLoss()
    focal_tversky_loss = FocalTverskyLoss(alpha=0.20, beta=0.80, gamma=1.33)
    boundary_loss = BoundaryLoss().to(DEVICE)
    aux_bce_loss = nn.BCEWithLogitsLoss()

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
            labels = labels.to(DEVICE)

            output = model(imgs)
            seg_logits, cls_logits = unpack_model_output(output)

            loss = compute_loss(
                seg_logits,
                masks,
                cls_logits,
                labels,
                seg_bce_loss,
                focal_tversky_loss,
                boundary_loss,
                aux_bce_loss
            )
            total_loss += loss.item()

            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()
            if cls_logits is not None:
                presence_probs = torch.sigmoid(cls_logits.view(-1)).cpu().numpy()
            else:
                presence_probs = np.ones(len(seg_probs), dtype=np.float32)

            for i in range(len(seg_probs)):
                prob_map = seg_probs[i, 0]
                pred_mask = (prob_map >= seg_threshold).astype(np.uint8)

                if presence_probs[i] < presence_threshold:
                    pred_mask = np.zeros_like(pred_mask, dtype=np.uint8)

                pred_mask = clean_pred_mask(pred_mask, min_positive_pixels)
                pred_mask_t = torch.from_numpy(pred_mask).float()

                true_mask = masks[i].cpu()
                gt_positive = true_mask.sum().item() > 0
                pred_positive = pred_mask_t.sum().item() > 0

                if gt_positive:
                    positive_dices.append(dice_score(pred_mask_t, true_mask))
                    positive_ious.append(iou_score(pred_mask_t, true_mask))

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
        0.15 * ischemia_recall +
        0.10 * ischemia_precision -
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

    seg_thresholds = np.linspace(0.25, 0.60, 8)
    presence_thresholds = [0.35, 0.45, 0.55, 0.65]
    min_pixels_candidates = [8, 16, 24, 32, 48]

    best_metrics = None

    for seg_threshold in seg_thresholds:
        for presence_threshold in presence_thresholds:
            for min_pixels in min_pixels_candidates:
                metrics = evaluate_model(
                    model,
                    search_loader,
                    seg_threshold=float(seg_threshold),
                    presence_threshold=float(presence_threshold),
                    min_positive_pixels=int(min_pixels)
                )
                metrics["seg_threshold"] = float(seg_threshold)
                metrics["presence_threshold"] = float(presence_threshold)
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

    train_dataset = IschemiaRefinementDataset(train_items, train=True)
    val_dataset = IschemiaRefinementDataset(val_items, train=False)

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

    seg_bce_loss = nn.BCEWithLogitsLoss()
    focal_tversky_loss = FocalTverskyLoss(alpha=0.20, beta=0.80, gamma=1.33)
    boundary_loss = BoundaryLoss().to(DEVICE)

    train_labels = np.array([item["label"] for item in train_items])
    num_pos = int((train_labels == 1).sum())
    num_neg = int((train_labels == 0).sum())
    aux_pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32).to(DEVICE)
    aux_bce_loss = nn.BCEWithLogitsLoss(pos_weight=aux_pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5
    )

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
            "presence_threshold": DEFAULT_PRESENCE_THRESHOLD,
            "min_positive_pixels": DEFAULT_MIN_POSITIVE_PIXELS,
            "note": "fallback thresholds"
        }
    )

    for epoch in range(EPOCHS):
        model.train()
        train_loss_total = 0.0

        for imgs, masks, labels_batch in train_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            optimizer.zero_grad()

            output = model(imgs)
            seg_logits, cls_logits = unpack_model_output(output)

            loss = compute_loss(
                seg_logits,
                masks,
                cls_logits,
                labels_batch,
                seg_bce_loss,
                focal_tversky_loss,
                boundary_loss,
                aux_bce_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item()

        train_loss = train_loss_total / len(train_loader)

        val_metrics = evaluate_model(
            model,
            val_loader,
            seg_threshold=DEFAULT_SEG_THRESHOLD,
            presence_threshold=DEFAULT_PRESENCE_THRESHOLD,
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
            f"Dice(Refinement): {val_metrics['positive_dice']:.4f} | "
            f"IoU(Refinement): {val_metrics['positive_iou']:.4f}"
        )
        print(
            f"Recall(Ischemia ROI-level): {val_metrics['ischemia_recall']:.4f} | "
            f"Precision(Ischemia ROI-level): {val_metrics['ischemia_precision']:.4f} | "
            f"FP rate on negative ROIs: {val_metrics['negative_fp_rate']:.4f}"
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
    plt.title("Validation Dice Refinement")
    plt.savefig(os.path.join(RESULTS_DIR, "val_dice_refinement.png"))
    plt.close()

    best_model = build_model(pretrained=False).to(DEVICE)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    best_model.eval()

    fallback_metrics = evaluate_model(
        best_model,
        val_loader,
        seg_threshold=DEFAULT_SEG_THRESHOLD,
        presence_threshold=DEFAULT_PRESENCE_THRESHOLD,
        min_positive_pixels=DEFAULT_MIN_POSITIVE_PIXELS
    )

    try:
        best_threshold_metrics = search_best_thresholds(best_model, val_dataset)
        note = "threshold search completed successfully"
    except Exception as e:
        best_threshold_metrics = {
            "seg_threshold": DEFAULT_SEG_THRESHOLD,
            "presence_threshold": DEFAULT_PRESENCE_THRESHOLD,
            "min_positive_pixels": DEFAULT_MIN_POSITIVE_PIXELS,
            **fallback_metrics
        }
        note = f"threshold search failed, fallback used: {str(e)}"

    thresholds_data = {
        "seg_threshold": float(best_threshold_metrics["seg_threshold"]),
        "presence_threshold": float(best_threshold_metrics["presence_threshold"]),
        "min_positive_pixels": int(best_threshold_metrics["min_positive_pixels"]),
        "positive_dice": float(best_threshold_metrics["positive_dice"]),
        "positive_iou": float(best_threshold_metrics["positive_iou"]),
        "negative_fp_rate": float(best_threshold_metrics["negative_fp_rate"]),
        "ischemia_recall": float(best_threshold_metrics["ischemia_recall"]),
        "ischemia_precision": float(best_threshold_metrics["ischemia_precision"]),
        "note": note
    }
    save_json(THRESHOLDS_PATH, thresholds_data)
    save_json(VAL_METRICS_PATH, thresholds_data)

    print("\n  Лучшие thresholds на валидации  ")
    print(f"Seg threshold: {best_threshold_metrics['seg_threshold']:.2f}")
    print(f"Presence threshold: {best_threshold_metrics['presence_threshold']:.2f}")
    print(f"Min positive pixels: {best_threshold_metrics['min_positive_pixels']}")
    print(f"Dice(Refinement): {best_threshold_metrics['positive_dice']:.4f}")
    print(f"IoU(Refinement): {best_threshold_metrics['positive_iou']:.4f}")
    print(f"Recall(Ischemia ROI-level): {best_threshold_metrics['ischemia_recall']:.4f}")
    print(f"Precision(Ischemia ROI-level): {best_threshold_metrics['ischemia_precision']:.4f}")
    print(f"FP rate on negative ROIs: {best_threshold_metrics['negative_fp_rate']:.4f}")
    print(f"Note: {note}")

    print("\nРезультаты обучения сохранены в:", RESULTS_DIR)
    print("Порог сохранен в:", THRESHOLDS_PATH)

if __name__ == "__main__":
    main()