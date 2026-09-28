import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

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

RESULTS_DIR = "results_type_classifier"
CHECKPOINT_DIR = "checkpoints_type_classifier"

IMG_SIZE = 384
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    0: "Normal",
    1: "Bleeding",
    2: "Ischemia"
}

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
            labels.append(2)

    return image_paths, labels



class StrokeTypeDataset(Dataset):
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

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label



train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



def build_model(pretrained=True):
    if pretrained:
        try:
            weights = models.ResNet34_Weights.DEFAULT
        except AttributeError:
            weights = "DEFAULT"
    else:
        weights = None

    model = models.resnet34(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model



def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def run_inference(model, loader):
    all_probs = []
    all_preds = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)

            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            labels = labels.cpu().numpy()

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels)

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_probs, all_preds, all_labels



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

    train_dataset = StrokeTypeDataset(train_paths, train_labels, train_transform)
    val_dataset = StrokeTypeDataset(val_paths, val_labels, val_transform)

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

    class_counts = np.bincount(train_labels)
    class_weights = len(train_labels) / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    print("Class weights:", class_weights.cpu().numpy())

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    best_val_macro_f1 = 0.0
    best_epoch = 0
    early_stop_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": []
    }

    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels_batch in train_loader:
            imgs = imgs.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            optimizer.zero_grad()

            logits = model(imgs)
            loss = criterion(logits, labels_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels_batch).sum().item()
            train_total += labels_batch.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        model.eval()

        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for imgs, labels_batch in val_loader:
                imgs = imgs.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE)

                logits = model(imgs)
                loss = criterion(logits, labels_batch)

                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_val_preds.extend(preds.cpu().numpy().tolist())
                all_val_labels.extend(labels_batch.cpu().numpy().tolist())

        val_loss /= len(val_loader)
        val_acc = np.mean(np.array(all_val_preds) == np.array(all_val_labels))
        val_macro_f1 = f1_score(
            all_val_labels,
            all_val_preds,
            average="macro"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_macro_f1)
        new_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Acc:   {val_acc:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f}"
        )
        print(f"Текущий LR: {new_lr:.6f}")

        if new_lr != old_lr:
            print(f"Learning rate уменьшен с {old_lr:.6f} до {new_lr:.6f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
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
    history_df.to_csv(os.path.join(RESULTS_DIR, "history.csv"), index=False)

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

    plt.figure()
    plt.plot(history["val_macro_f1"])
    plt.title("Validation Macro-F1")
    plt.savefig(os.path.join(RESULTS_DIR, "macro_f1.png"))
    plt.close()

    best_model = build_model(pretrained=False).to(DEVICE)
    best_model.load_state_dict(
        torch.load(
            os.path.join(CHECKPOINT_DIR, "best_model.pth"),
            map_location=DEVICE
        )
    )

    val_probs, val_preds, val_targets = run_inference(best_model, val_loader)

    print("Длина val_paths:", len(val_paths))
    print("Длина val_labels:", len(val_labels))
    print("Длина val_targets:", len(val_targets))
    print("Длина val_preds:", len(val_preds))
    print("Длина val_probs:", len(val_probs))

    assert len(val_paths) == len(val_targets) == len(val_preds) == len(val_probs), \
        "Несовпадение длин массивов в финальной валидации"

    print("\n Валидационные результаты ")

    print(classification_report(
        val_targets,
        val_preds,
        target_names=[CLASS_NAMES[i] for i in range(3)],
        digits=4,
        zero_division=0
    ))

    cm = confusion_matrix(val_targets, val_preds, labels=[0, 1, 2])
    print("Confusion matrix:")
    print(cm)

    val_pred_df = pd.DataFrame({
        "path": val_paths,
        "true_label": val_targets,
        "pred_label": val_preds,
        "prob_normal": val_probs[:, 0],
        "prob_bleeding": val_probs[:, 1],
        "prob_ischemia": val_probs[:, 2]
    })
    val_pred_df.to_csv(
        os.path.join(RESULTS_DIR, "val_predictions.csv"),
        index=False
    )

    save_json(
        os.path.join(RESULTS_DIR, "class_mapping.json"),
        CLASS_NAMES
    )

    save_json(
        os.path.join(RESULTS_DIR, "best_metrics.json"),
        {
            "best_epoch": best_epoch,
            "best_val_macro_f1": float(best_val_macro_f1)
        }
    )

    print("\nРезультаты обучения сохранены в:", RESULTS_DIR)

if __name__ == "__main__":
    main()



