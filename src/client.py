
import os
import copy
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from .config import (
    OUTPUT_DIR,
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    LOCAL_EPOCHS
)

from .dataset import HAMDataset, train_transform
from .model import get_model


def train_client(
    client_id,
    global_model_state,
    partition_name="iid"
):
    """
    Train one hospital's local model.

    Parameters
    ----------
    client_id : int
        Hospital/client number: 0, 1, or 2.

    global_model_state : dict
        Current global model parameters received from server.

    partition_name : str
        Either "iid" or "non_iid".

    Returns
    -------
    local_model_state : dict
        Updated local model parameters.

    num_samples : int
        Number of training images at this client.
    """

    # ---------------------------------------------
    # 1. Load this hospital's local dataset
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

    # ---------------------------------------------
    # 2. Create local copy of global model
    # ---------------------------------------------

    local_model = get_model(num_classes=7).to(DEVICE)

    local_model.load_state_dict(global_model_state)

    # ---------------------------------------------
    # 3. Define loss and optimizer
    # ---------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=LEARNING_RATE
    )

    # ---------------------------------------------
    # 4. Local training
    # ---------------------------------------------

    for epoch in range(LOCAL_EPOCHS):

        local_model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in local_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = local_model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

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

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        print(
            f"Local Epoch {epoch + 1}/{LOCAL_EPOCHS} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: {epoch_accuracy:.4f}"
        )

    # ---------------------------------------------
    # 5. Return updated model parameters
    # ---------------------------------------------

    local_model_state = {
        key: value.detach().cpu().clone()
        for key, value in local_model.state_dict().items()
    }

    return local_model_state, num_samples


if __name__ == "__main__":

    # Simple standalone test using Hospital 1's IID data

    initial_model = get_model(num_classes=7)

    initial_state = copy.deepcopy(
        initial_model.state_dict()
    )

    updated_state, sample_count = train_client(
        client_id=0,
        global_model_state=initial_state,
        partition_name="iid"
    )

    print("\nLocal client test completed!")
    print("Returned samples:", sample_count)
    print("Number of model parameters:",
          len(updated_state))
