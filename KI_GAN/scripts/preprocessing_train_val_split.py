import os
import shutil
import pandas as pd
import random
import numpy as np
from enum import IntEnum


class AgentType(IntEnum):
    pedestrian = 0
    animal = 0
    car = 1
    truck = 2
    bus = 3
    motorcycle = 4
    tricycle = 5
    bicycle = 6


# Seed set for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def check_csv_lengths(input_path):
    combined_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
    for f in combined_files:
        file_path = os.path.join(input_path, f)
        df = pd.read_csv(file_path, header=None)
        print(f"{f}: {len(df)} rows")


def split_dataset(input_path, train_path, val_path, test_path, train_ratio=0.7, val_ratio=0.2, filter_test_classes=None,
                  set_all_red=False):
    # Get all CSV files in the input directory - train_val_no_split
    combined_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]

    # Calculate the number of files for each set
    total_files = len(combined_files)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)
    test_size = total_files - train_size - val_size

    # Split the files into train, validation, and test sets without shuffling
    train_files = combined_files[:train_size]
    val_files = combined_files[train_size:train_size + val_size]
    test_files = combined_files[train_size + val_size:]

    # Copy files to the train directory
    for f in train_files:
        shutil.copy(os.path.join(input_path, f), os.path.join(train_path, f))

    # Copy files to the validation directory
    for f in val_files:
        shutil.copy(os.path.join(input_path, f), os.path.join(val_path, f))

    # Copy files to the test directory with filtering if required
    for f in test_files:
        src_path = os.path.join(input_path, f)
        dst_path = os.path.join(test_path, f)
        df = pd.read_csv(src_path, header=None)

        # Process traffic signals for Tianjin_2 to set all to red
        if set_all_red:
            df[11] = 0  # Set all traffic signals to red

        if filter_test_classes:
            print(f"Processing file: {src_path}")
            print("Columns:", df.columns.tolist())  # Print column names for debugging
            print("Unique agent_type values before filtering:", df[9].unique())  # Print unique values for debugging
            if 9 in df.columns:
                # Convert the agent_type column to integers for comparison
                df[9] = df[9].astype(int)
                filtered_df = df[df[9].isin(filter_test_classes)]
                print("Unique agent_type values after filtering:",
                      filtered_df[9].unique())  # Print unique values for debugging
                filtered_df.to_csv(dst_path, index=False, header=False)
            else:
                print(f"Error: Column index 9 not found in {src_path}")
                raise KeyError(f"Column index 9 not found in {src_path}")
        else:
            df.to_csv(dst_path, index=False, header=False)

    print(f"Total files: {total_files}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")


def create_datasets(base_input_path, base_output_path, datasets):
    for dataset in datasets:
        input_path = base_input_path
        train_path = os.path.join(base_output_path, dataset['name'], 'train')
        val_path = os.path.join(base_output_path, dataset['name'], 'val')
        test_path = os.path.join(base_output_path, dataset['name'], 'test')
        train_ratio = dataset['train_ratio']
        val_ratio = dataset['val_ratio']
        seed = dataset['seed']
        filter_test_classes = dataset.get('filter_test_classes', None)
        set_all_red = dataset.get('set_all_red', False)

        # Set the random seed for reproducibility
        set_seed(seed)

        # Ensure output directories exist, remove if they already exist
        if os.path.exists(train_path):
            shutil.rmtree(train_path)
        if os.path.exists(val_path):
            shutil.rmtree(val_path)
        if os.path.exists(test_path):
            shutil.rmtree(test_path)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        # Check the length of each CSV file
        print(f"Checking CSV lengths for dataset: {dataset['name']}")
        check_csv_lengths(input_path)

        # Split the dataset
        print(f"Splitting dataset: {dataset['name']}")
        split_dataset(input_path, train_path, val_path, test_path, train_ratio, val_ratio, filter_test_classes,
                      set_all_red)


def main():
    base_input_path = '../datasets/train_val_no_split'
    base_output_path = '../datasets'

    datasets = [
        # Baseline
        {'name': 'Tianjin', 'train_ratio': 0.7, 'val_ratio': 0.2, 'seed': 42},
        # Vulnerable traffic participants
        {'name': 'Tianjin_1', 'train_ratio': 0.7, 'val_ratio': 0.2, 'seed': 42, 'filter_test_classes': [0, 4, 5, 6]},
        # No traffic encoder
        {'name': 'Tianjin_2', 'train_ratio': 0.7, 'val_ratio': 0.2, 'seed': 42, 'set_all_red': True},
        # No traffic encoder with vulnerable traffic participants
        {'name': 'Tianjin_3', 'train_ratio': 0.7, 'val_ratio': 0.2, 'seed': 42, 'filter_test_classes': [0, 4, 5, 6],
         'set_all_red': True},
        # Vehicle participants (non-vulnerable)
        {'name': 'Tianjin_4', 'train_ratio': 0.7, 'val_ratio': 0.2, 'seed': 42, 'filter_test_classes': [1, 2, 3]},
        # Vehicle participants (non-vulnerable) with no traffic encoder
        {'name': 'Tianjin_5', 'train_ratio': 0.7, 'val_ratio': 0.2, 'seed': 42, 'filter_test_classes': [1, 2, 3],
         'set_all_red': True},
        # No Motorcyclists to test traffic encoder - base set
        {'name': 'Tianjin_6', 'train_ratio': 0.7, 'val_ratio': 0.2, 'seed': 42, 'filter_test_classes': [0, 1, 2, 3, 5, 6]},
        # No Motorcyclists to test traffic encoder - no traffic encoder
        {'name': 'Tianjin_7', 'train_ratio': 0.7, 'val_ratio': 0.2, 'seed': 42, 'filter_test_classes': [0, 1, 2, 3, 5, 6],
         'set_all_red': True},
        # Motorcyclists only base set
        {'name': 'Tianjin_8', 'train_ratio': 0.7, 'val_ratio': 0.2, 'seed': 42,
         'filter_test_classes': [4]},
        # Motorcyclists only no traffic encoder
        {'name': 'Tianjin_9', 'train_ratio': 0.7, 'val_ratio': 0.2, 'seed': 42,
         'filter_test_classes': [4], 'set_all_red': True},
    ]

    create_datasets(base_input_path, base_output_path, datasets)


if __name__ == "__main__":
    main()
