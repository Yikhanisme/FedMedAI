"""
Tạo partition files cho FedSLD reproduction.
Chạy 1 lần trước khi train:
    python -m datasets.partition_fedsld
"""

from datasets.medmnist_code import get_pathmnist_datasets, get_organmnist_datasets
from datasets.partition import (practical_noniid_partition,
                                pathological_noniid_partition,
                                save_fedsld_partition)

NUM_CLIENTS = 12
SEED = 42

datasets_info = [
    ("pathmnist",        get_pathmnist_datasets),
    ("organmnist_axial", get_organmnist_datasets),
]

if __name__ == "__main__":
    for ds_name, get_fn in datasets_info:
        print(f"\n{'='*50}")
        print(f"Dataset: {ds_name.upper()}")
        train_ds, _, _, num_classes = get_fn()
        print(f"  Train samples: {len(train_ds)} | Classes: {num_classes}")

        print("  Generating Practical Non-IID partition...")
        part_practical = practical_noniid_partition(train_ds, NUM_CLIENTS, SEED)
        save_fedsld_partition(part_practical, train_ds, ds_name,
                              "practical_noniid", NUM_CLIENTS, SEED)

        print("  Generating Pathological Non-IID partition...")
        part_patho = pathological_noniid_partition(train_ds, NUM_CLIENTS,
                                                   num_classes_per_client=2, seed=SEED)
        save_fedsld_partition(part_patho, train_ds, ds_name,
                              "pathological_noniid", NUM_CLIENTS, SEED)

    print("\nDone! All partition files saved to data/partitions/")

