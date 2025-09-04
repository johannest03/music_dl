from glob import glob
import os


def split_files(folder_path, train_size=0.8, test_size=0.2):
    assert train_size + test_size == 1.0, "Sizes must sum to 1."

    all_files = glob(os.path.join(folder_path, "**/*.mid"), recursive=True)
    train_files = all_files[:int(len(all_files) * train_size)]
    test_files = all_files[int(len(all_files) * train_size):]
    return train_files, test_files