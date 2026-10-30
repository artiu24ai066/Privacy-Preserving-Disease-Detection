
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from .config import (
    OUTPUT_DIR,
    MODEL_DIR,
    GRAPH_DIR,
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    RANDOM_SEED
)

from .dataset import (
    HAMDataset,
    train_transform,
    test_transform
)

from .model import get_model


# ==========================================
# 1. REPRODUCIBILITY
# ==========================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data():

    train_df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "train.csv")
    )

    val_df = pd.read_csv(
        os.path.join(OUTPUT_DIR, "validation.csv")
    )

    train_dataset = HAMDataset(
        train_df,
        transform=train_transform
    )

    val_dataset = HAMDataset(
        val_df,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    return train_loader, val_loader, train_df


# ==========================================
# 3. CLASS WEIGHTS
# ==========================================


def calculate_class_weights(train_df):

    num_classes = 7

    counts = (
        train_df["label"]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
    )

    total = len(train_df)

    # Square-root inverse-frequency weights
    weights = np.sqrt(
        total / (num_classes * counts.values)
    )

    # Normalize weights so their average is 1
    weights = weights / weights.mean()

    print("\nModerate class weights:")

    for i, weight in enumerate(weights):
        print(f"Class {i}: {weight:.4f}")

    return torch.tensor(
        weights,
        dtype=torch.float32
    ).to(DEVICE)


# ==========================================
# 4. TRAIN ONE EPOCH
# ==========================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    for images, labels in tqdm(
        loader,
        desc="Training",
        leave=False
    ):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    epoch_loss = running_loss / len(loader.dataset)

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    epoch_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return epoch_loss, epoch_accuracy, epoch_f1


# ==========================================
# 5. VALIDATION
# ==========================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in tqdm(
            loader,
            desc="Validation",
            leave=False
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    val_loss = running_loss / len(loader.dataset)

    val_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    val_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return val_loss, val_accuracy, val_f1


# ==========================================
# 6. PLOT TRAINING HISTORY
# ==========================================

def plot_history(history):

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(GRAPH_DIR, "loss_curve.png")
    )

    plt.close()

    # Accuracy graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_accuracy"],
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        marker="o",
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(GRAPH_DIR, "accuracy_curve.png")
    )

    plt.close()


# ==========================================
# 7. MAIN CENTRALIZED TRAINING
# ==========================================

def train_centralized(use_class_weights=False):

    set_seed(RANDOM_SEED)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)

    print("\nDevice:", DEVICE)

    # Load datasets
    train_loader, val_loader, train_df = load_data()

    # Initialize model
    model = get_model(num_classes=7)

    model = model.to(DEVICE)

    # Loss function
    if use_class_weights:

        weights = calculate_class_weights(train_df)

        criterion = nn.CrossEntropyLoss(
            weight=weights
        )

        print("\nUsing class-weighted CrossEntropyLoss")

    else:

        criterion = nn.CrossEntropyLoss()

        print("\nUsing ordinary CrossEntropyLoss")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1": [],
        "val_f1": []
    }

    best_val_f1 = -1.0

    model_name = (
        "centralized_weighted.pth"
        if use_class_weights
        else "centralized_model.pth"
    )

    model_path = os.path.join(
        MODEL_DIR,
        model_name
    )

    # Epoch loop
    for epoch in range(EPOCHS):

        print(f"\n========== Epoch {epoch + 1}/{EPOCHS} ==========")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc, val_f1 = validate(
            model,
            val_loader,
            nn.CrossEntropyLoss()
        )

        # Store metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        # Print results
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Train Macro F1: {train_f1:.4f}")

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation Macro F1: {val_f1:.4f}")

        # Save best model using validation macro F1
        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            torch.save(
                model.state_dict(),
                model_path
            )

            print("Best model saved!")

    # Plot graphs
    plot_history(history)

    print("\nTraining completed!")

    print("Best Validation Macro F1:", best_val_f1)

    print("Model saved at:", model_path)


if __name__ == "__main__":

    train_centralized(use_class_weights=False)

