
import os
import copy
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from opacus.accountants import RDPAccountant

from .config import (
    OUTPUT_DIR,
    MODEL_DIR,
    DEVICE,
    BATCH_SIZE,
    FED_ROUNDS,
    NUM_CLIENTS,
    RANDOM_SEED,
    DELTA
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


def get_client_epsilon(client_id, partition_name, dp_run_id):

    accountant_path = os.path.join(
        OUTPUT_DIR,
        "privacy",
        "federated_dp",
        dp_run_id,
        partition_name,
        f"client_{client_id}_accountant.pt"
    )

    if not os.path.exists(accountant_path):
        return None

    accountant_state = torch.load(
        accountant_path,
        map_location="cpu",
        weights_only=True
    )

    accountant = RDPAccountant()
    accountant.load_state_dict(accountant_state)

    return accountant.get_epsilon(delta=DELTA)


def run_federated_training(
    partition_name="iid",
    dp=False,
    dp_run_id=None,
    num_rounds=None
):

    set_seed(RANDOM_SEED)

    if partition_name not in ["iid", "non_iid"]:
        raise ValueError(
            "partition_name must be 'iid' or 'non_iid'"
        )

    if dp and not dp_run_id:
        raise ValueError(
            "Provide a unique dp_run_id for DP experiments."
        )

    rounds = num_rounds if num_rounds is not None else FED_ROUNDS

    if rounds <= 0:
        raise ValueError("num_rounds must be positive.")

    print("\n======================================")
    print("FEDERATED LEARNING")
    print("Partition:", partition_name.upper())
    print("Differential Privacy:", dp)
    print("======================================")

    print("Device:", DEVICE)
    print("Number of hospitals:", NUM_CLIENTS)
    print("Communication rounds:", rounds)

    if dp:
        print("DP run ID:", dp_run_id)

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

    suffix = "_dp" if dp else ""

    model_path = os.path.join(
        MODEL_DIR,
        f"federated_{partition_name}{suffix}_model.pth"
    )

    # ---------------------------------------------
    # 3. Communication rounds
    # ---------------------------------------------

    for round_num in range(rounds):

        print(
            f"\n========== ROUND {round_num + 1}/{rounds} =========="
        )

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
                partition_name=partition_name,
                dp=dp,
                dp_run_id=dp_run_id if dp else "presentation_run"
            )

            client_states.append(local_state)
            client_sizes.append(num_samples)

        # -----------------------------------------
        # 5. Aggregate using FedAvg
        # -----------------------------------------

        new_global_state = fedavg(
            client_states,
            client_sizes
        )

        global_model.load_state_dict(new_global_state)

        # -----------------------------------------
        # 6. Evaluate global model
        # -----------------------------------------

        accuracy, macro_f1 = evaluate_global_model(
            global_model,
            validation_loader
        )

        print("\nGlobal Model Evaluation")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Macro F1: {macro_f1:.4f}")

        round_result = {
            "round": round_num + 1,
            "accuracy": accuracy,
            "macro_f1": macro_f1
        }

        # -----------------------------------------
        # 7. Record privacy accounting
        # -----------------------------------------

        if dp:

            client_epsilons = []

            print("\nCumulative Privacy Accounting")

            for client_id in range(NUM_CLIENTS):

                epsilon = get_client_epsilon(
                    client_id,
                    partition_name,
                    dp_run_id
                )

                client_epsilons.append(epsilon)

                print(
                    f"Hospital {client_id + 1} epsilon: "
                    f"{epsilon:.4f}"
                )

                round_result[
                    f"client_{client_id}_epsilon"
                ] = epsilon

            # Conservative summary: maximum client epsilon.
            overall_epsilon = max(client_epsilons)

            round_result["max_client_epsilon"] = overall_epsilon

            print(
                f"Maximum client epsilon: {overall_epsilon:.4f}"
            )

        history.append(round_result)

        # -----------------------------------------
        # 8. Save best global model
        # -----------------------------------------

        if macro_f1 > best_macro_f1:

            best_macro_f1 = macro_f1

            torch.save(
                global_model.state_dict(),
                model_path
            )

            print("Best global model saved!")

    # ---------------------------------------------
    # 9. Save training history
    # ---------------------------------------------

    history_df = pd.DataFrame(history)

    history_path = os.path.join(
        OUTPUT_DIR,
        f"federated_{partition_name}{suffix}_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print("\n======================================")
    print("FEDERATED TRAINING COMPLETED")
    print("======================================")

    print("Partition:", partition_name)
    print("Differential Privacy:", dp)
    print(f"Best Validation Macro F1: {best_macro_f1:.4f}")
    print("Model saved at:", model_path)
    print("History saved at:", history_path)

    return global_model, history_df


if __name__ == "__main__":

    # One-round DP smoke test.
    # Use a fresh run ID for each independent experiment.

    run_federated_training(
        partition_name="non_iid",
        dp=True,
        dp_run_id="non_iid_dp_smoke_01",
        num_rounds=5
    )
