
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from .config import (
    OUTPUT_DIR,
    MODEL_DIR,
    GRAPH_DIR,
    DEVICE,
    BATCH_SIZE,
    DELTA
)

from .dataset import HAMDataset, test_transform
from .model import get_model


CLASSES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


def evaluate_model():

    print("\n======================================")
    print("FEDERATED DP MODEL EVALUATION")
    print("======================================")

    print("Device:", DEVICE)

    os.makedirs(GRAPH_DIR, exist_ok=True)

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

    print("Validation dataset size:", len(validation_dataset))

    # ---------------------------------------------
    # 2. Load federated DP model
    # ---------------------------------------------

    model = get_model(num_classes=7).to(DEVICE)

    model_path = os.path.join(
        MODEL_DIR,
        "federated_non_iid_dp_model.pth"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    print("Federated DP model loaded successfully!")

    # ---------------------------------------------
    # 3. Generate predictions
    # ---------------------------------------------

    actual_labels = []
    predicted_labels = []

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            actual_labels.extend(labels.numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    actual_labels = np.array(actual_labels)
    predicted_labels = np.array(predicted_labels)

    # ---------------------------------------------
    # 4. Overall metrics
    # ---------------------------------------------

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

    print("\n========== OVERALL RESULTS ==========")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Macro F1: {macro_f1:.4f}")

    # ---------------------------------------------
    # 5. Classification report
    # ---------------------------------------------

    report_dict = classification_report(
        actual_labels,
        predicted_labels,
        labels=list(range(7)),
        target_names=CLASSES,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()

    print("\n========== CLASSIFICATION REPORT ==========")
    print(report_df)

    report_path = os.path.join(
        OUTPUT_DIR,
        "federated_non_iid_dp_classification_report.csv"
    )

    report_df.to_csv(report_path)

    # ---------------------------------------------
    # 6. Confusion matrix
    # ---------------------------------------------

    cm = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=list(range(7))
    )

    print("\n========== CONFUSION MATRIX ==========")
    print(cm)

    fig, ax = plt.subplots(figsize=(10, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45
    )

    plt.title("Non-IID Federated DP - Confusion Matrix")
    plt.tight_layout()

    confusion_path = os.path.join(
        GRAPH_DIR,
        "federated_non_iid_dp_confusion_matrix.png"
    )

    plt.savefig(confusion_path, dpi=300)
    plt.close()

    # ---------------------------------------------
    # 7. Predicted class distribution
    # ---------------------------------------------

    print("\n========== PREDICTED CLASS COUNTS ==========")

    for class_index, class_name in enumerate(CLASSES):

        count = np.sum(predicted_labels == class_index)

        print(f"{class_name}: {count}")

    # ---------------------------------------------
    # 8. Privacy accounting summary
    # ---------------------------------------------

    accountant_dir = os.path.join(
        OUTPUT_DIR,
        "privacy",
        "federated_dp",
        "non_iid_dp_smoke_01",
        "non_iid"
    )

    epsilon_values = []

    for client_id in range(3):

        accountant_path = os.path.join(
            accountant_dir,
            f"client_{client_id}_accountant.pt"
        )

        if not os.path.exists(accountant_path):
            raise FileNotFoundError(
                f"Accountant not found: {accountant_path}"
            )

        from opacus.accountants import RDPAccountant

        accountant_state = torch.load(
            accountant_path,
            map_location="cpu",
            weights_only=True
        )

        accountant = RDPAccountant()
        accountant.load_state_dict(accountant_state)

        epsilon = accountant.get_epsilon(delta=DELTA)
        epsilon_values.append(epsilon)

        print(
            f"Hospital {client_id + 1} epsilon: {epsilon:.4f}"
        )

    max_epsilon = max(epsilon_values)

    print(f"Maximum client epsilon: {max_epsilon:.4f}")

    # ---------------------------------------------
    # 9. Save summary
    # ---------------------------------------------

    summary = pd.DataFrame([{
        "experiment": "Non-IID Federated DP",
        "communication_rounds": 1,
        "local_epochs": 2,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "epsilon_hospital_1": epsilon_values[0],
        "epsilon_hospital_2": epsilon_values[1],
        "epsilon_hospital_3": epsilon_values[2],
        "max_client_epsilon": max_epsilon,
        "delta": DELTA
    }])

    summary_path = os.path.join(
        OUTPUT_DIR,
        "federated_non_iid_dp_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    print("\n======================================")
    print("FEDERATED DP EVALUATION COMPLETED")
    print("======================================")

    print("Classification report:", report_path)
    print("Confusion matrix:", confusion_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    evaluate_model()
