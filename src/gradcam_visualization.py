
import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from .config import (
    MODEL_DIR,
    GRAPH_DIR,
    GRADCAM_DIR,
    DEVICE,
    NUM_CLASSES,
    IMAGE_SIZE
)
from .model import get_model
from .dataset import test_transform


# Disease class names
CLASS_NAMES = [
    "akiec", "bcc", "bkl", "df",
    "mel", "nv", "vasc"
]


def load_trained_model():

    model = get_model(NUM_CLASSES)

    model_path = os.path.join(
        MODEL_DIR,
        "centralized_moderate_weighted.pth"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    print("Trained model loaded successfully!")

    return model


def generate_gradcam(model, image_path):

    # Load original image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Convert image to numpy array in [0, 1]
    rgb_image = np.array(image).astype(np.float32) / 255.0

    # Apply the same preprocessing used during evaluation
    input_tensor = test_transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)

    # Last convolutional block of ResNet18
    target_layers = [model.layer4[-1]]


    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    # Initialize Grad-CAM
    with GradCAM(
        model=model,
        target_layers=target_layers
    ) as cam:

        # First identify the predicted class
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        # Tell Grad-CAM which class to explain
        targets = [ClassifierOutputTarget(predicted_class)]

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0, :]

    # Overlay heatmap on original image
    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    # Get model prediction
    with torch.no_grad():
        probabilities = torch.softmax(output, dim=1)
        confidence = probabilities[0, predicted_class].item()
    return (
        rgb_image,
        visualization,
        predicted_class,
        confidence
    )


def main():

    os.makedirs(GRADCAM_DIR, exist_ok=True)

    # Load validation data
    validation_path = os.path.join(
        os.path.dirname(GRADCAM_DIR),
        "validation.csv"
    )

    df = pd.read_csv(validation_path)

    # Load model
    model = load_trained_model()

    # Select one image per available class
    selected_images = []

    random.seed(42)

    for class_id in range(NUM_CLASSES):

        class_df = df[df["label"] == class_id]

        if len(class_df) == 0:
            continue

        row = class_df.sample(
            n=1,
            random_state=42
        ).iloc[0]

        selected_images.append(row)

    print(f"\nGenerating Grad-CAM for {len(selected_images)} images...")

    for index, row in enumerate(selected_images):

        image_path = row["image_path"]
        true_label = int(row["label"])

        (
            original,
            heatmap,
            predicted_class,
            confidence
        ) = generate_gradcam(model, image_path)

        # Plot original image and Grad-CAM
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))

        axes[0].imshow(original)
        axes[0].set_title(
            f"Original\nTrue: {CLASS_NAMES[true_label]}"
        )
        axes[0].axis("off")

        axes[1].imshow(heatmap)
        axes[1].set_title(
            f"Grad-CAM\nPredicted: {CLASS_NAMES[predicted_class]}\n"
            f"Confidence: {confidence:.2%}"
        )
        axes[1].axis("off")

        plt.tight_layout()

        output_path = os.path.join(
            GRADCAM_DIR,
            f"gradcam_{CLASS_NAMES[true_label]}.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"{index + 1}. True: {CLASS_NAMES[true_label]} | "
            f"Predicted: {CLASS_NAMES[predicted_class]} | "
            f"Confidence: {confidence:.2%}"
        )

        print(f"   Saved: {output_path}")

    print("\nGrad-CAM generation completed!")


if __name__ == "__main__":
    main()
