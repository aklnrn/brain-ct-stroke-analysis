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
    import segmentation_models_pytorch as smp
except ImportError as e:
    raise ImportError(
        "Не найден пакет segmentation_models_pytorch. "
        "Установить его командой: pip install segmentation-models-pytorch"
    ) from e



ROOT_DIR = "Brain_Stroke_CT_Dataset"

BLEEDING_IMG_DIR = os.path.join(ROOT_DIR, "Bleeding", "PNG")
BLEEDING_MASK_DIR = os.path.join(ROOT_DIR, "Bleeding", "MASKS_BIN")

NORMAL_IMG_DIR = os.path.join(ROOT_DIR, "Normal", "PNG")
NORMAL_MASK_DIR = os.path.join(ROOT_DIR, "Normal", "MASKS")

RESULTS_DIR = "results_bleeding_segmenter"
CHECKPOINT_DIR = "checkpoints_bleeding_segmenter"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
THRESHOLDS_PATH = os.path.join(RESULTS_DIR, "thresholds.json")

IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 60
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def image_to_tensor(image):
    image = TF.resize(
        image,
        [IMG_SIZE, IMG_SIZE],
        interpolation=InterpolationMode.BILINEAR
    )
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor

def mask_to_tensor(mask):
    mask = TF.resize(
        mask,
        [IMG_SIZE, IMG_SIZE],
        interpolation=InterpolationMode.NEAREST
    )
    mask = TF.to_tensor(mask)
    mask = (mask > 0.5).float()
    return mask

def load_mask_or_empty(mask_path, image_size):
    if mask_path is not None and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L")
    else:
        mask = Image.new("L", image_size, 0)
    return mask

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

def build_model(pretrained=True):
    encoder_weights = "imagenet" if pretrained else None
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1
    )
    return model



def collect_items():
    items = []

    for img_name in sorted(os.listdir(BLEEDING_IMG_DIR)):
        if img_name.lower().endswith(".png"):
            image_path = os.path.join(BLEEDING_IMG_DIR, img_name)
            mask_path = os.path.join(BLEEDING_MASK_DIR, img_name)
            if os.path.exists(mask_path):
                items.append({
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "label": 1
                })

    for img_name in sorted(os.listdir(NORMAL_IMG_DIR)):
        if img_name.lower().endswith(".png"):
            image_path = os.path.join(NORMAL_IMG_DIR, img_name)
            mask_path = os.path.join(NORMAL_MASK_DIR, img_name)
            if not os.path.exists(mask_path):
                mask_path = None

            items.append({
                "image_path": image_path,
                "mask_path": mask_path,
                "label": 0
            })

    return items



class BleedingSegmentationDataset(Dataset):
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

        if random.random() < 0.3:
            image = TF.adjust_brightness(image, random.uniform(0.95, 1.05))
        if random.random() < 0.3:
            image = TF.adjust_contrast(image, random.uniform(0.95, 1.10))

        return image, mask

    def __getitem__(self, idx):
        item = self.items[idx]

        image = Image.open(item["image_path"]).convert("RGB")
        mask = load_mask_or_empty(item["mask_path"], image.size)

        if self.train:
            image, mask = self.apply_train_transforms(image, mask)

        image_tensor = image_to_tensor(image)
        mask_tensor = mask_to_tensor(mask)
        label_tensor = torch.tensor(item["label"], dtype=torch.long)

        return image_tensor, mask_tensor, label_tensor



class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
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
        return 1 - tversky.mean()

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
        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)

        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)

        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pred_edges = self.edges(probs)
        true_edges = self.edges(targets)
        return F.l1_loss(pred_edges, true_edges)

def compute_loss(logits, targets, bce_loss, tversky_loss, boundary_loss):
    loss_bce = bce_loss(logits, targets)
    loss_tversky = tversky_loss(logits, targets)
    loss_boundary = boundary_loss(logits, targets)

    total_loss = (
        0.50 * loss_bce +
        0.35 * loss_tversky +
        0.15 * loss_boundary
    )
    return total_loss



def evaluate_model(model, loader, threshold=0.5):
    model.eval()

    bce_loss = nn.BCEWithLogitsLoss()
    tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)
    boundary_loss = BoundaryLoss().to(DEVICE)

    total_loss = 0.0
    positive_dices = []
    positive_ious = []

    tp_img = 0
    fn_img = 0
    fp_img = 0
    tn_img = 0

    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(imgs)
            loss = compute_loss(
                logits,
                masks,
                bce_loss,
                tversky_loss,
                boundary_loss
            )
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            for pred_mask, true_mask in zip(preds, masks):
                gt_positive = true_mask.sum().item() > 0
                pred_positive = pred_mask.sum().item() > 0

                if gt_positive:
                    positive_dices.append(dice_score(pred_mask.cpu(), true_mask.cpu()))
                    positive_ious.append(iou_score(pred_mask.cpu(), true_mask.cpu()))

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
    bleeding_total = tp_img + fn_img

    negative_fp_rate = fp_img / negative_total if negative_total > 0 else 0.0
    bleeding_recall = tp_img / bleeding_total if bleeding_total > 0 else 0.0

    score = positive_dice - 0.10 * negative_fp_rate

    return {
        "loss": avg_loss,
        "positive_dice": positive_dice,
        "positive_iou": positive_iou,
        "negative_fp_rate": negative_fp_rate,
        "bleeding_recall": bleeding_recall,
        "tp_img": tp_img,
        "fn_img": fn_img,
        "fp_img": fp_img,
        "tn_img": tn_img,
        "score": score
    }

def search_best_threshold(model, loader):
    thresholds = np.linspace(0.30, 0.70, 9)

    best_metrics = None

    for threshold in thresholds:
        metrics = evaluate_model(model, loader, threshold=threshold)
        metrics["threshold"] = float(threshold)

        if best_metrics is None or metrics["score"] > best_metrics["score"]:
            best_metrics = metrics

    return best_metrics



def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Device:", DEVICE)

    items = collect_items()

    labels = [item["label"] for item in items]
    print("Всего изображений:", len(items))
    print("Распределение по классам [Normal, Bleeding]:", np.bincount(labels))

    train_items, val_items = train_test_split(
        items,
        test_size=0.2,
        stratify=labels,
        random_state=SEED
    )

    train_dataset = BleedingSegmentationDataset(train_items, train=True)
    val_dataset = BleedingSegmentationDataset(val_items, train=False)

    train_labels = np.array([item["label"] for item in train_items])
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[item["label"]] for item in train_items]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = build_model(pretrained=True).to(DEVICE)

    bce_loss = nn.BCEWithLogitsLoss()
    tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)
    boundary_loss = BoundaryLoss().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4
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
        "val_bleeding_recall": [],
        "val_score": []
    }

    for epoch in range(EPOCHS):
        model.train()
        train_loss_total = 0.0

        for imgs, masks, _ in train_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(imgs)
            loss = compute_loss(
                logits,
                masks,
                bce_loss,
                tversky_loss,
                boundary_loss
            )

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        train_loss = train_loss_total / len(train_loader)

        val_metrics = evaluate_model(model, val_loader, threshold=0.5)
        val_score = val_metrics["score"]

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_score)
        new_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_positive_dice"].append(val_metrics["positive_dice"])
        history["val_positive_iou"].append(val_metrics["positive_iou"])
        history["val_negative_fp_rate"].append(val_metrics["negative_fp_rate"])
        history["val_bleeding_recall"].append(val_metrics["bleeding_recall"])
        history["val_score"].append(val_score)

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Dice(Bleeding): {val_metrics['positive_dice']:.4f} | "
            f"IoU(Bleeding): {val_metrics['positive_iou']:.4f}"
        )
        print(
            f"Recall(Bleeding image-level): {val_metrics['bleeding_recall']:.4f} | "
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

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(RESULTS_DIR, "history.csv"), index=False)

    plt.figure()
    plt.plot(history["train_loss"])
    plt.plot(history["val_loss"])
    plt.title("Loss")
    plt.legend(["Train", "Val"])
    plt.savefig(os.path.join(RESULTS_DIR, "loss.png"))
    plt.close()

    plt.figure()
    plt.plot(history["val_positive_dice"])
    plt.title("Validation Dice (Bleeding only)")
    plt.savefig(os.path.join(RESULTS_DIR, "val_dice_bleeding.png"))
    plt.close()

    plt.figure()
    plt.plot(history["val_negative_fp_rate"])
    plt.title("Validation FP Rate on Negatives")
    plt.savefig(os.path.join(RESULTS_DIR, "val_fp_rate_negatives.png"))
    plt.close()

    best_model = build_model(pretrained=False).to(DEVICE)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    best_model.eval()

    best_threshold_metrics = search_best_threshold(best_model, val_loader)

    save_json(
        THRESHOLDS_PATH,
        {
            "best_threshold": float(best_threshold_metrics["threshold"]),
            "positive_dice": float(best_threshold_metrics["positive_dice"]),
            "positive_iou": float(best_threshold_metrics["positive_iou"]),
            "negative_fp_rate": float(best_threshold_metrics["negative_fp_rate"]),
            "bleeding_recall": float(best_threshold_metrics["bleeding_recall"])
        }
    )

    print("\n  Лучший threshold на валидации  ")
    print(f"Threshold: {best_threshold_metrics['threshold']:.2f}")
    print(f"Dice(Bleeding): {best_threshold_metrics['positive_dice']:.4f}")
    print(f"IoU(Bleeding): {best_threshold_metrics['positive_iou']:.4f}")
    print(f"Recall(Bleeding image-level): {best_threshold_metrics['bleeding_recall']:.4f}")
    print(f"FP rate on negatives: {best_threshold_metrics['negative_fp_rate']:.4f}")

    print("\nРезультаты обучения сохранены в:", RESULTS_DIR)
    print("Порог сохранен в:", THRESHOLDS_PATH)


if __name__ == "__main__":
    main()