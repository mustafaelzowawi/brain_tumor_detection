"""
PyTorch Dataset classes for loading brain tumor data.
"""
import os
import glob
import torch
# import numpy as np # No longer directly used here
# import pandas as pd # No longer directly used here
from torch.utils.data import Dataset
from monai.data import PersistentDataset, CacheDataset # Using MONAI's caching datasets
import logging

from . import config

# Note: The notebook originally used a custom Dataset class that seemed to replicate
# some MONAI functionality. It's generally recommended to use MONAI's built-in
# dataset types like CacheDataset or PersistentDataset when using MONAI transforms
# as they offer performance benefits (caching, multi-processing).

# We will define a function to create MONAI datasets directly.

def get_monai_datasets(train_data_list, val_data_list, test_data_list,
                       train_transforms, val_test_transforms):
    """
    Creates MONAI CacheDatasets for train, validation, and test sets.
    CacheDataset loads and caches transformed data in memory for faster access.
    PersistentDataset caches to disk, useful for larger datasets or limited RAM.
    Choose based on available resources.
    """
    train_ds = CacheDataset(
        data=train_data_list,
        transform=train_transforms,
        cache_rate=1.0, # Cache all items
        num_workers=config.NUM_WORKERS
    )
    val_ds = CacheDataset(
        data=val_data_list,
        transform=val_test_transforms,
        cache_rate=1.0,
        num_workers=config.NUM_WORKERS
    )
    test_ds = CacheDataset(
        data=test_data_list,
        transform=val_test_transforms,
        cache_rate=1.0,
        num_workers=config.NUM_WORKERS
    )
    return train_ds, val_ds, test_ds


# --- Dataset for loading preprocessed .pt files (from notebook) ---
# This dataset assumes you have already run a preprocessing script
# that applied transforms and saved each sample as a dictionary {'image': tensor, 'label': tensor}
# using torch.save().
class PreprocessedDataset(Dataset):
    """
    Loads data preprocessed and saved as torch tensor files (.pt).
    Each .pt file should contain a dictionary {'image': tensor, 'label': tensor}.
    """
    def __init__(self, data_dir):
        # Find all .pt files in the specified directory
        self.data_files = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
        if not self.data_files:
            raise FileNotFoundError(f"No .pt files found in directory: {data_dir}. "
                                    f"Ensure preprocessing was run and files saved correctly.")
        print(f"Found {len(self.data_files)} preprocessed files in {data_dir}")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # Load the saved dictionary containing the image and label tensors
        try:
            # Note: Loading arbitrary pickles can be unsafe.
            # Consider using weights_only=True if only loading tensors,
            # but here we load a dict, so it's likely fine if you trust the source.
            data = torch.load(self.data_files[idx])
            image = data['image']
            label = data['label']

            # Ensure label is in the correct shape (e.g., [1])
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.float32)
            label = label.view(1) # Reshape to [1] if necessary

            return image, label
        except Exception as e:
            print(f"Error loading file {self.data_files[idx]}: {e}")
            # Return None or raise error? Returning None might be problematic for DataLoader.
            # Best to ensure data integrity beforehand.
            raise

def get_preprocessed_datasets(train_dir=config.PREPROCESSED_TRAIN_DIR,
                                val_dir=config.PREPROCESSED_VAL_DIR,
                                test_dir=config.PREPROCESSED_TEST_DIR,
                                load_test=True):
    """Creates PreprocessedDataset instances for train, val, and optionally test sets."""
    try:
        train_ds = PreprocessedDataset(train_dir)
        val_ds = PreprocessedDataset(val_dir)
        test_ds = None
        if load_test:
            # Try to load test_ds, but don't fail if it's not there (e.g., for local testing)
            if os.path.exists(test_dir) and os.listdir(test_dir): # Check if dir exists and is not empty
                try:
        test_ds = PreprocessedDataset(test_dir)
                except FileNotFoundError:
                    logging.warning(f"Test preprocessed directory {test_dir} exists but contains no .pt files. Test set will be None.")
                    test_ds = None # Explicitly set to None
            else:
                logging.warning(f"Test preprocessed directory {test_dir} not found or is empty. Test set will be None.")
                test_ds = None # Explicitly set to None

        return train_ds, val_ds, test_ds
    except FileNotFoundError as e: # This will catch if train_dir or val_dir is missing
        print(f"Error creating preprocessed datasets: {e}")
        print("Please ensure the preprocessing script has been run and the paths in config.py are correct.")
        raise 