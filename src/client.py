
import os
import copy
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from opacus import PrivacyEngine

from .config import (
    OUTPUT_DIR,
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    NOISE_MULTIPLIER,
    MAX_GRAD_NORM,
    DELTA
)

from .dataset import HAMDataset, train_transform
from .model import get_model


def train_client(
    client_id,
    global_model_state,
    partition_name="iid",
    dp=False,
    dp_run_id="presentation_run"
):
    """
    Train one hospital's local model.

    dp=False: Ordinary local training.
    dp=True: Local DP-SGD using Opacus.

    Privacy accounting is retained across communication rounds
    for each client and experiment run.
    """

    # ---------------------------------------------
    # 1. Load hospital's local dataset
    # ---------------------------------------------

    csv_path = os.path.join(
        OUTPUT_DIR,
        "clients",
        partition_name,
        f"client_{client_id}.csv"
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Client dataset not found: {csv_path}"
        )

    client_df = pd.read_csv(csv_path)

    local_dataset = HAMDataset(
        client_df,
        transform=train_transform
    )

    local_loader = DataLoader(
        local_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    num_samples = len(local_dataset)

    print(f"\nHospital {client_id + 1}")
    print("Training images:", num_samples)
    print("Differential Privacy:", dp)

    # ---------------------------------------------
    # 2. Create local copy of global model
    # ---------------------------------------------

    local_model = get_model(num_classes=7).to(DEVICE)
    local_model.load_state_dict(global_model_state)

    # Disable inplace ReLU for Opacus compatibility
    for module in local_model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

            
    # ---------------------------------------------
    # 3. Define loss and optimizer
    # ---------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=LEARNING_RATE
    )

    privacy_engine = None
    accountant_path = None

    # ---------------------------------------------
    # 4. Enable DP-SGD if requested
    # ---------------------------------------------

    if dp:

        privacy_dir = os.path.join(
            OUTPUT_DIR,
            "privacy",
            "federated_dp",
            dp_run_id,
            partition_name
        )

        os.makedirs(privacy_dir, exist_ok=True)

        accountant_path = os.path.join(
            privacy_dir,
            f"client_{client_id}_accountant.pt"
        )

        privacy_engine = PrivacyEngine(
            accountant="rdp"
        )

        # Restore previous privacy accounting history
        # so privacy loss accumulates across rounds.
        if os.path.exists(accountant_path):

            accountant_state = torch.load(
                accountant_path,
                map_location="cpu",
                weights_only=True
            )

            privacy_engine.accountant.load_state_dict(
                accountant_state
            )

            print("Previous privacy accountant restored.")

        (
            local_model,
            optimizer,
            local_loader
        ) = privacy_engine.make_private(
            module=local_model,
            optimizer=optimizer,
            data_loader=local_loader,
            noise_multiplier=NOISE_MULTIPLIER,
            max_grad_norm=MAX_GRAD_NORM,
            grad_sample_mode="functorch"
        )

        print("Opacus DP-SGD enabled!")

    # ---------------------------------------------
    # 5. Local training
    # ---------------------------------------------

    for epoch in range(LOCAL_EPOCHS):

        local_model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in local_loader:

            if images.size(0) == 0:
                continue

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = local_model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            # Do not publish local training metrics
            # from private client data in the DP experiment.
            if not dp:

                batch_size = labels.size(0)

                running_loss += loss.item() * batch_size

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                correct += (
                    predictions == labels
                ).sum().item()

                total += batch_size

        if dp:

            epsilon = privacy_engine.accountant.get_epsilon(
                delta=DELTA
            )

            print(
                f"DP Local Epoch {epoch + 1}/{LOCAL_EPOCHS} | "
                f"Epsilon: {epsilon:.4f} | "
                f"Delta: {DELTA}"
            )

        else:

            if total > 0:

                epoch_loss = running_loss / total
                epoch_accuracy = correct / total

                print(
                    f"Local Epoch {epoch + 1}/{LOCAL_EPOCHS} | "
                    f"Loss: {epoch_loss:.4f} | "
                    f"Accuracy: {epoch_accuracy:.4f}"
                )

    # ---------------------------------------------
    # 6. Save cumulative privacy accountant
    # ---------------------------------------------

    if dp:

        torch.save(
            privacy_engine.accountant.state_dict(),
            accountant_path
        )

        final_epsilon = privacy_engine.accountant.get_epsilon(
            delta=DELTA
        )

        print(
            f"Hospital {client_id + 1} cumulative epsilon: "
            f"{final_epsilon:.4f}"
        )

    # ---------------------------------------------
    # 7. Return updated model parameters
    # ---------------------------------------------

    # Opacus wraps the model. Extract the original model.
    if dp and hasattr(local_model, "_module"):
        model_to_save = local_model._module
    else:
        model_to_save = local_model

    local_model_state = {
        key: value.detach().cpu().clone()
        for key, value in model_to_save.state_dict().items()
    }

    return local_model_state, num_samples


if __name__ == "__main__":

    initial_model = get_model(num_classes=7)

    initial_state = copy.deepcopy(
        initial_model.state_dict()
    )

    # Ordinary training smoke test
    updated_state, sample_count = train_client(
        client_id=0,
        global_model_state=initial_state,
        partition_name="iid",
        dp=False
    )

    print("\nLocal client test completed!")
    print("Returned samples:", sample_count)
    print("Number of state entries:", len(updated_state))
