
import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from .config import (
    OUTPUT_DIR,
    MODEL_DIR,
    DEVICE,
    BATCH_SIZE,
    FED_ROUNDS,
    NUM_CLIENTS,
    RANDOM_SEED
)

from .dataset import HAMDataset, test_transform
from .model import get_model
from .client import train_client
from .aggregation import fedavg


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_global_model(model, validation_loader):

    model.eval()

    actual_labels = []
    predicted_labels = []

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            actual_labels.extend(labels.numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels
    )

    macro_f1 = f1_score(
        actual_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    return accuracy, macro_f1


def run_federated_training(partition_name="iid"):

    set_seed(RANDOM_SEED)

    if partition_name not in ["iid", "non_iid"]:
        raise ValueError("partition_name must be 'iid' or 'non_iid'")

    print("\n======================================")
    print("FEDERATED LEARNING")
    print("Partition:", partition_name.upper())
    print("======================================")

    print("Device:", DEVICE)
    print("Number of hospitals:", NUM_CLIENTS)
    print("Communication rounds:", FED_ROUNDS)

    # ---------------------------------------------
    # 1. Load validation dataset
    # ---------------------------------------------

    validation_path = os.path.join(
        OUTPUT_DIR,
        "validation.csv"
    )

    validation_df = pd.read_csv(validation_path)

    validation_dataset = HAMDataset(
        validation_df,
        transform=test_transform
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Validation images:", len(validation_dataset))

    # ---------------------------------------------
    # 2. Initialize global model
    # ---------------------------------------------

    global_model = get_model(num_classes=7).to(DEVICE)

    history = []

    best_macro_f1 = -1.0

    model_path = os.path.join(
        MODEL_DIR,
        f"federated_{partition_name}_model.pth"
    )

    # ---------------------------------------------
    # 3. Communication rounds
    # ---------------------------------------------

    for round_num in range(FED_ROUNDS):

        print(f"\n========== ROUND {round_num + 1}/{FED_ROUNDS} ==========")

        global_state = copy.deepcopy(
            global_model.state_dict()
        )

        client_states = []
        client_sizes = []

        # -----------------------------------------
        # 4. Train each hospital locally
        # -----------------------------------------

        for client_id in range(NUM_CLIENTS):

            local_state, num_samples = train_client(
                client_id=client_id,
                global_model_state=global_state,
                partition_name=partition_name
            )

            client_states.append(local_state)
            client_sizes.append(num_samples)

        # -----------------------------------------
        # 5. Aggregate local models using FedAvg
        # -----------------------------------------

        new_global_state = fedavg(
            client_states,
            client_sizes
        )

        global_model.load_state_dict(new_global_state)

        # -----------------------------------------
        # 6. Evaluate updated global model
        # -----------------------------------------

        accuracy, macro_f1 = evaluate_global_model(
            global_model,
            validation_loader
        )

        print("\nGlobal Model Evaluation")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Macro F1: {macro_f1:.4f}")

        history.append({
            "round": round_num + 1,
            "accuracy": accuracy,
            "macro_f1": macro_f1
        })

        # -----------------------------------------
        # 7. Save best global model
        # -----------------------------------------

        if macro_f1 > best_macro_f1:

            best_macro_f1 = macro_f1

            torch.save(
                global_model.state_dict(),
                model_path
            )

            print("Best global model saved!")

    # ---------------------------------------------
    # 8. Save training history
    # ---------------------------------------------

    history_df = pd.DataFrame(history)

    history_path = os.path.join(
        OUTPUT_DIR,
        f"federated_{partition_name}_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print("\n======================================")
    print("FEDERATED TRAINING COMPLETED")
    print("======================================")

    print("Partition:", partition_name)
    print(f"Best Validation Macro F1: {best_macro_f1:.4f}")
    print("Model saved at:", model_path)
    print("History saved at:", history_path)

    return global_model, history_df


if __name__ == "__main__":

    run_federated_training(partition_name="non_iid")
