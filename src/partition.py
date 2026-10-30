
import os
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, NUM_CLIENTS, RANDOM_SEED


# --------------------------------------------------
# 1. IID partitioning
# --------------------------------------------------

def create_iid_partition(df, num_clients, seed=42):

    rng = np.random.default_rng(seed)

    # Keep all images of the same lesion together
    lesion_ids = df["lesion_id"].unique().to_numpy().copy()

    rng.shuffle(lesion_ids)

    # Divide lesion IDs approximately equally
    client_lesions = np.array_split(lesion_ids, num_clients)

    client_dfs = []

    for lesion_group in client_lesions:

        client_df = df[
            df["lesion_id"].isin(lesion_group)
        ].copy()

        client_dfs.append(
            client_df.reset_index(drop=True)
        )

    return client_dfs


# --------------------------------------------------
# 2. Non-IID partitioning using Dirichlet distribution
# --------------------------------------------------

def create_non_iid_partition(
    df,
    num_clients,
    alpha=0.5,
    seed=42
):

    rng = np.random.default_rng(seed)

    # Each lesion should belong to exactly one disease class
    lesion_labels = (
        df.groupby("lesion_id")["label"]
        .nunique()
    )

    if (lesion_labels > 1).any():
        raise ValueError(
            "A lesion_id is associated with multiple labels."
        )

    lesion_table = (
        df[["lesion_id", "label"]]
        .drop_duplicates("lesion_id")
    )

    client_lesions = [[] for _ in range(num_clients)]

    # Retry if one client receives too few images
    average_images = len(df) / num_clients

    minimum_images = int(0.70 * average_images)
    maximum_images = int(1.30 * average_images)

    for attempt in range(5000):

        client_lesions = [[] for _ in range(num_clients)]

        # Distribute each disease class separately
        for class_label in sorted(df["label"].unique()):

            class_lesions = lesion_table[
                lesion_table["label"] == class_label
            ]["lesion_id"].to_numpy().copy()

            rng.shuffle(class_lesions)

            # Generate different class proportions
            proportions = rng.dirichlet(
                np.repeat(alpha, num_clients)
            )

            # Convert proportions into split positions
            split_points = (
                np.cumsum(proportions)[:-1]
                * len(class_lesions)
            ).astype(int)

            client_groups = np.split(
                class_lesions,
                split_points
            )

            for client_id, groups in enumerate(client_groups):
                client_lesions[client_id].extend(
                    groups.tolist()
                )

        # Count actual images, not just lesion IDs
        client_sizes = [
            df["lesion_id"].isin(groups).sum()
            for groups in client_lesions
        ]

        if(
            min(client_sizes) >= minimum_images and max(client_sizes) <= maximum_images
        ):
            break

    else:
        raise RuntimeError(
            "Could not generate balanced client sizes. "
            "Try a larger alpha value."
        )

    client_dfs = []

    for lesion_group in client_lesions:

        client_df = df[
            df["lesion_id"].isin(lesion_group)
        ].copy()

        client_dfs.append(
            client_df.reset_index(drop=True)
        )

    return client_dfs


# --------------------------------------------------
# 3. Print client distribution
# --------------------------------------------------

def display_distribution(client_dfs, partition_name):

    class_names = [
        "akiec", "bcc", "bkl", "df",
        "mel", "nv", "vasc"
    ]

    print(f"\n========== {partition_name} PARTITION ==========")

    for client_id, client_df in enumerate(client_dfs):

        print(f"\nHospital {client_id + 1}")
        print("Number of images:", len(client_df))
        print("Number of lesions:", client_df["lesion_id"].nunique())

        counts = (
            client_df["label"]
            .value_counts()
            .reindex(range(7), fill_value=0)
        )

        for label, count in counts.items():
            print(
                f"{class_names[label]}: {count}"
            )


# --------------------------------------------------
# 4. Save partitions
# --------------------------------------------------

def save_partitions(client_dfs, partition_name):

    partition_dir = os.path.join(
        OUTPUT_DIR, "clients", partition_name
    )

    os.makedirs(partition_dir, exist_ok=True)

    for client_id, client_df in enumerate(client_dfs):

        file_path = os.path.join(
            partition_dir,
            f"client_{client_id}.csv"
        )

        client_df.to_csv(file_path, index=False)

        print("Saved:", file_path)


# --------------------------------------------------
# 5. Main execution
# --------------------------------------------------

def main():

    train_path = os.path.join(
        OUTPUT_DIR, "train.csv"
    )

    train_df = pd.read_csv(train_path)

    print("Total training images:", len(train_df))
    print(
        "Total unique lesions:",
        train_df["lesion_id"].nunique()
    )

    # Create IID partitions
    iid_clients = create_iid_partition(
        train_df,
        NUM_CLIENTS,
        seed=RANDOM_SEED
    )

    display_distribution(iid_clients, "IID")
    save_partitions(iid_clients, "iid")

    # Create non-IID partitions
    non_iid_clients = create_non_iid_partition(
        train_df,
        NUM_CLIENTS,
        alpha=0.5,
        seed=RANDOM_SEED
    )

    display_distribution(non_iid_clients, "NON-IID")
    save_partitions(non_iid_clients, "non_iid")

    print("\nAll partitions created successfully!")


if __name__ == "__main__":
    main()
