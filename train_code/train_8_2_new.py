import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


ROOT_DIR = "Brain_Stroke_CT_Dataset"

BLEEDING_DIR = os.path.join(ROOT_DIR, "Bleeding", "PNG")
ISCHEMIA_DIR = os.path.join(ROOT_DIR, "Ischemia", "PNG")
NORMAL_DIR = os.path.join(ROOT_DIR, "Normal", "PNG")

RESULTS_DIR = "results_classification3_new"
CHECKPOINT_DIR = "checkpoints3_new"

THRESHOLDS_PATH = os.path.join(RESULTS_DIR, "thresholds.json")
VAL_PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "val_predictions.csv")
VAL_METRICS_PATH = os.path.join(RESULTS_DIR, "val_metrics.json")

IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4
PATIENCE = 5

# Порог для маршрутизации в сегментатор
TARGET_ROUTE_RECALL = 0.99

# Порог для отчетной бинарной классификации
MIN_REPORT_SPECIFICITY = 0.90
MIN_REPORT_PRECISION = 0.80

# Минимальный зазор между route_threshold и report_threshold, чтобы появилась "серая зона"
MIN_THRESHOLD_GAP = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



def collect_image_paths():
    image_paths = []
    labels = []

    for img in sorted(os.listdir(NORMAL_DIR)):
        if img.lower().endswith(".png"):
            image_paths.append(os.path.join(NORMAL_DIR, img))
            labels.append(0)

    for img in sorted(os.listdir(BLEEDING_DIR)):
        if img.lower().endswith(".png"):
            image_paths.append(os.path.join(BLEEDING_DIR, img))
            labels.append(1)

    for img in sorted(os.listdir(ISCHEMIA_DIR)):
        if img.lower().endswith(".png"):
            image_paths.append(os.path.join(ISCHEMIA_DIR, img))
            labels.append(1)

    return image_paths, labels



class StrokeDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.labels[idx], dtype=torch.float32)



train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])



def build_model(pretrained=True):
    if pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
        except AttributeError:
            weights = "DEFAULT"
    else:
        weights = None

    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model



def run_inference(model, loader):
    all_probs = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for imgs, labels_batch in loader:
            imgs = imgs.to(DEVICE)

            outputs = model(imgs)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_labels.extend(labels_batch.numpy().tolist())

    return np.array(all_probs), np.array(all_labels, dtype=int)

def compute_metrics(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "accuracy": float(accuracy),
        "f1": float(f1)
    }

def select_report_threshold(
    y_true,
    probs,
    route_threshold,
    min_specificity=0.90,
    min_precision=0.80,
    min_gap=0.10
):
    all_thresholds = np.linspace(0.001, 0.999, 999)

    start_threshold = route_threshold + min_gap
    candidate_thresholds = all_thresholds[all_thresholds >= start_threshold]

    # Если из-за большого зазора кандидатов не осталось, просто поиск threshold не ниже route_threshold
    if len(candidate_thresholds) == 0:
        candidate_thresholds = all_thresholds[all_thresholds >= route_threshold]

    all_metrics = []

    for t in candidate_thresholds:
        metrics = compute_metrics(y_true, probs, t)
        metrics["balanced_accuracy"] = 0.5 * (
            metrics["recall"] + metrics["specificity"]
        )
        all_metrics.append(metrics)

    candidates = [
        m for m in all_metrics
        if m["specificity"] >= min_specificity
        and m["precision"] >= min_precision
    ]

    if len(candidates) == 0:
        candidates = [
            m for m in all_metrics
            if m["specificity"] >= min_specificity
        ]

    if len(candidates) == 0:
        candidates = all_metrics

    best = max(
        candidates,
        key=lambda m: (
            m["balanced_accuracy"],
            m["specificity"],
            m["precision"],
            m["threshold"]
        )
    )

    best.pop("balanced_accuracy", None)
    return best

def select_route_threshold(y_true, probs, target_recall=0.99):
    thresholds = np.linspace(0.001, 0.999, 999)
    all_metrics = []

    for t in thresholds:
        metrics = compute_metrics(y_true, probs, t)
        all_metrics.append(metrics)

    candidates = [
        m for m in all_metrics
        if m["recall"] >= target_recall
    ]

    if len(candidates) > 0:
        # максимально высокий порог, который все еще держит нужный recall
        best = max(
            candidates,
            key=lambda m: (m["threshold"], m["specificity"])
        )
    else:
        # Если нужный recall недостижим, то брать максимально возможный recall и среди них максимально высокий threshold
        best = max(
            all_metrics,
            key=lambda m: (m["recall"], m["specificity"], m["threshold"])
        )

    return best

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Device:", DEVICE)

    paths, labels = collect_image_paths()

    print("Общее количество изображений:", len(paths))
    print("Распределение по классам:", np.bincount(labels))

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=SEED
    )

    train_dataset = StrokeDataset(train_paths, train_labels, train_transform)
    val_dataset = StrokeDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    train_class_counts = np.bincount(train_labels)
    pos_weight_value = train_class_counts[0] /train_class_counts[1]

    print("С использованием pos_weight:", round(pos_weight_value, 3))

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels_batch in train_loader:
            imgs = imgs.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total

        model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels_batch in val_loader:
                imgs = imgs.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE).unsqueeze(1)

                outputs = model(imgs)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct += (preds == labels_batch).sum().item()
                total += labels_batch.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"Текущий LR: {new_lr:.6f}")

        if new_lr != old_lr:
            print(f"Learning rate уменьшен с {old_lr:.6f} до {new_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            early_stop_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, "best_model.pth")
            )

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
    history_df.to_csv(
        os.path.join(RESULTS_DIR, "history.csv"),
        index=False
    )

    plt.figure()
    plt.plot(history["train_loss"])
    plt.plot(history["val_loss"])
    plt.title("Loss")
    plt.legend(["Train", "Val"])
    plt.savefig(os.path.join(RESULTS_DIR, "loss.png"))
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"])
    plt.plot(history["val_acc"])
    plt.title("Accuracy")
    plt.legend(["Train", "Val"])
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy.png"))
    plt.close()



    best_model = build_model(pretrained=False).to(DEVICE)
    best_model.load_state_dict(torch.load(
            os.path.join(CHECKPOINT_DIR, "best_model.pth"),
            map_location=DEVICE
        )
    )
    best_model.eval()

    val_probs, val_targets = run_inference(best_model, val_loader)
    val_auc = roc_auc_score(val_targets, val_probs)

    route_metrics = select_route_threshold(
        val_targets,
        val_probs,
        target_recall=TARGET_ROUTE_RECALL
    )
    route_threshold = route_metrics["threshold"]

    report_metrics = select_report_threshold(
        val_targets,
        val_probs,
        route_threshold=route_threshold,
        min_specificity=MIN_REPORT_SPECIFICITY,
        min_precision=MIN_REPORT_PRECISION,
        min_gap=MIN_THRESHOLD_GAP
    )
    report_threshold = report_metrics["threshold"]

    report_metrics = compute_metrics(val_targets, val_probs, report_threshold)
    route_metrics = compute_metrics(val_targets, val_probs, route_threshold)

    print("\n Валидационные пороги ")
    print(f"Report threshold: {report_threshold:.4f}")
    print(
        f"  Recall={report_metrics['recall']:.4f} | "
        f"Specificity={report_metrics['specificity']:.4f} | "
        f"FN={report_metrics['fn']} | FP={report_metrics['fp']}"
    )

    print(f"Route threshold:  {route_threshold:.4f}")
    print(
        f"  Recall={route_metrics['recall']:.4f} | "
        f"Specificity={route_metrics['specificity']:.4f} | "
        f"FN={route_metrics['fn']} | FP={route_metrics['fp']}"
    )

    thresholds_data = {
        "report_threshold": float(report_threshold),
        "route_threshold": float(route_threshold),
        "target_route_recall": float(TARGET_ROUTE_RECALL),
        "min_report_specificity": float(MIN_REPORT_SPECIFICITY),
        "validation_auc": float(val_auc)
    }
    save_json(THRESHOLDS_PATH, thresholds_data)

    route_decision = []
    for p in val_probs:
        if p < route_threshold:
            route_decision.append("skip_segmentation")
        elif p < report_threshold:
            route_decision.append("run_segmentation_gray_zone")
        else:
            route_decision.append("run_segmentation_stroke")

    val_predictions_df = pd.DataFrame({
        "path": val_paths,
        "true_label": val_labels,
        "prob_stroke": val_probs,
        "pred_report": (val_probs >= report_threshold).astype(int),
        "pred_route": (val_probs >= route_threshold).astype(int),
        "route_decision": route_decision
    })
    val_predictions_df.to_csv(VAL_PREDICTIONS_PATH, index=False)

    save_json(VAL_METRICS_PATH, {
        "validation_auc": float(val_auc),
        "report_metrics": report_metrics,
        "route_metrics": route_metrics
    })

    print("\nРезультаты обучения сохранены в:", RESULTS_DIR)
    print("Пороги сохранены в:", THRESHOLDS_PATH)

if __name__ == "__main__":
    main()