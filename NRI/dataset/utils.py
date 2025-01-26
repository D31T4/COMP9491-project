TrainValidTestTuple = tuple[
    list[str],
    list[str],
    list[str]
]

def split_dataset(
    all_files: list[str], 
    train_ratio: float, 
    valid_ratio: float
) -> TrainValidTestTuple:
    '''
    partition data files into 3 subsets: train, validation, test

    Args:
    ---
    - all_files: list of files to be partitioned
    - train_ratio: ratio of training set
    - valid_ratio: ratio of validation set

    Returns:
    ---
    - training set
    - validation set
    - test set
    '''
    # Calculate the number of files for each set
    total_files = len(all_files)
    train_size = int(total_files * train_ratio)
    valid_size = int(total_files * valid_ratio)
    test_size = total_files - train_size - valid_size

    # Split the files into train, validation, and test sets without shuffling
    train_files = all_files[:train_size]
    valid_files = all_files[train_size:train_size + valid_size]
    test_files = all_files[train_size + valid_size:]

    return train_files, valid_files, test_files