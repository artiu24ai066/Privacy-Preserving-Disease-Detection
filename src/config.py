import torch
import os

# Project paths
ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")

GRADCAM_DIR = os.path.join(OUTPUT_DIR, "gradcam")

CSV_PATH = os.path.join(
    DATASET_DIR,
    "HAM10000_metadata.csv"
)

# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Dataset
NUM_CLASSES = 7
IMAGE_SIZE = 128

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5

# Federated learning
NUM_CLIENTS = 3
FED_ROUNDS = 5
LOCAL_EPOCHS = 2

# Differential privacy
NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM = 1.0
DELTA = 1e-5

# Reproducibility
RANDOM_SEED = 42
