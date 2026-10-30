import pandas as pd
import os

from .config import CSV_PATH, DATASET_DIR, OUTPUT_DIR
from sklearn.model_selection import GroupShuffleSplit


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(CSV_PATH)

print("Dataset preview:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nOriginal class distribution:")
print(df["dx"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())


# ==========================================
# 2. FIND IMAGE PATHS
# ==========================================

IMAGE_DIRS = [
    os.path.join(DATASET_DIR, "HAM10000_images_part_1"),
    os.path.join(DATASET_DIR, "HAM10000_images_part_2")
]


def get_image_path(image_id):

    for folder in IMAGE_DIRS:

        path = os.path.join(
            folder,
            image_id + ".jpg"
        )

        if os.path.exists(path):
            return path

    return None


df["image_path"] = df["image_id"].apply(
    get_image_path
)

missing_images = df["image_path"].isnull().sum()

print("\nMissing images:", missing_images)

df = df.dropna(subset=["image_path"]).copy()

print("Available images:", len(df))


# ==========================================
# 3. CHECK DATASET
# ==========================================

assert df["lesion_id"].notnull().all(), \
    "Some lesion IDs are missing!"

assert df["dx"].notnull().all(), \
    "Some class labels are missing!"

classes = sorted(df["dx"].unique())

print("\nClasses found:", classes)

assert len(classes) == 7, \
    "Expected seven disease classes"


# ==========================================
# 4. ENCODE LABELS
# ==========================================

class_to_idx = {
    name: idx
    for idx, name in enumerate(classes)
}

df["label"] = df["dx"].map(class_to_idx)

print("\nClass encoding:")
print(class_to_idx)


# ==========================================
# 5. SPLIT DATA BY LESION ID
# ==========================================

split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, temp_idx = next(
    split1.split(
        df,
        groups=df["lesion_id"]
    )
)

train_df = df.iloc[train_idx].copy()
temp_df = df.iloc[temp_idx].copy()


# Split remaining 20% into validation and test
split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_idx, test_idx = next(
    split2.split(
        temp_df,
        groups=temp_df["lesion_id"]
    )
)

val_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()


# ==========================================
# 6. CHECK SPLIT SIZES
# ==========================================

print("\nDataset split sizes:")

print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))


# ==========================================
# 7. VERIFY NO LESION LEAKAGE
# ==========================================

assert set(train_df["lesion_id"]).isdisjoint(
    set(val_df["lesion_id"])
)

assert set(train_df["lesion_id"]).isdisjoint(
    set(test_df["lesion_id"])
)

assert set(val_df["lesion_id"]).isdisjoint(
    set(test_df["lesion_id"])
)

print("\nNo lesion leakage detected!")


# ==========================================
# 8. CHECK CLASS DISTRIBUTIONS
# ==========================================

print("\nTraining distribution:")
print(
    train_df["dx"].value_counts().reindex(
        classes, fill_value=0
    )
)

print("\nValidation distribution:")
print(
    val_df["dx"].value_counts().reindex(
        classes, fill_value=0
    )
)

print("\nTesting distribution:")
print(
    test_df["dx"].value_counts().reindex(
        classes, fill_value=0
    )
)


# ==========================================
# 9. SAVE SPLITS
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False
)

val_df.to_csv(
    os.path.join(OUTPUT_DIR, "validation.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"),
    index=False
)

print("\nAll CSV files saved successfully!")
