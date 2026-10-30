
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from opacus import PrivacyEngine

from .config import (
    OUTPUT_DIR,
    MODEL_DIR,
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    RANDOM_SEED,
    NOISE_MULTIPLIER,
    MAX_GRAD_NORM,
    DELTA
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
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


# ==========================================
# 3. VALIDATION
# ==========================================

def validate(model, loader, criterion):

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

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

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
# 4. DIFFERENTIALLY PRIVATE TRAINING
# ==========================================

def train_dp_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    for images, labels in tqdm(
        loader,
        desc="DP Training",
        leave=False
    ):

        # Opacus uses Poisson sampling.
        # An empty batch can occasionally occur.
        if images.size(0) == 0:
            continue

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        # Opacus performs per-sample gradient clipping
        # and calibrated Gaussian noise addition here.
        optimizer.step()


# ==========================================
# 5. MAIN DP TRAINING
# ==========================================

def train_dp_centralized():

    set_seed(RANDOM_SEED)

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n======================================")
    print("CENTRALIZED DIFFERENTIAL PRIVACY")
    print("======================================")

    print("Device:", DEVICE)
    print("Noise multiplier:", NOISE_MULTIPLIER)
    print("Maximum gradient norm:", MAX_GRAD_NORM)
    print("Delta:", DELTA)

    # Load datasets
    train_loader, val_loader = load_data()

    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))

    # Initialize model
    model = get_model(num_classes=7)

    # Disable inplace ReLU for Opacus compatibility
    for module in model.modules():

        if isinstance(module, nn.ReLU):
            module.inplace = False

    model = model.to(DEVICE)

    # Ordinary CrossEntropyLoss for baseline DP
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ======================================
    # Initialize Privacy Engine
    # ======================================

    privacy_engine = PrivacyEngine(
        accountant="rdp"
    )

    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM,
        grad_sample_mode="functorch"
    )

    print("\nOpacus Privacy Engine initialized!")
    print("DP-SGD is enabled.")

    # Training history
    history = {
        "epoch": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "epsilon": []
    }

    best_val_f1 = -1.0

    model_path = os.path.join(
        MODEL_DIR,
        "centralized_dp_model.pth"
    )

    # ======================================
    # Epoch Loop
    # ======================================

    for epoch in range(EPOCHS):

        print(
            f"\n========== DP Epoch {epoch + 1}/{EPOCHS} =========="
        )

        train_dp_one_epoch(
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

        # Calculate cumulative privacy budget
        epsilon = privacy_engine.get_epsilon(
            delta=DELTA
        )

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation Macro F1: {val_f1:.4f}")

        print(f"Privacy Epsilon: {epsilon:.4f}")
        print(f"Privacy Delta: {DELTA}")

        history["epoch"].append(epoch + 1)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["epsilon"].append(epsilon)

        # Save the best model using validation Macro F1
        # Save the underlying standard PyTorch model,
        # not the Opacus wrapper.
        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            torch.save(
                model._module.state_dict(),
                model_path
            )

            print("Best DP model saved!")

    # Save training history
    history_path = os.path.join(
        OUTPUT_DIR,
        "centralized_dp_history.csv"
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False
    )

    # Final privacy budget
    final_epsilon = privacy_engine.get_epsilon(
        delta=DELTA
    )

    print("\n======================================")
    print("DP TRAINING COMPLETED")
    print("======================================")

    print("Best Validation Macro F1:", best_val_f1)
    print("Final Privacy Epsilon:", final_epsilon)
    print("Privacy Delta:", DELTA)

    print("Model saved at:", model_path)
    print("History saved at:", history_path)


if __name__ == "__main__":

    train_dp_centralized()
