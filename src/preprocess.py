"""
Script to preprocess the raw NIfTI data and save it as PyTorch tensors (.pt files).
This allows for faster loading during training/evaluation if I/O is a bottleneck.
"""
import os
import time
import torch
import logging
import argparse
from tqdm import tqdm # For progress bar
from sklearn.model_selection import train_test_split

# Import project modules
from . import config
from . import utils
from . import transforms

# Setup logging
log_file = os.path.join(config.LOG_DIR, 'preprocessing.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def preprocess_and_save(data_list, transform, save_dir, set_name):
    """Applies transforms to each item in data_list and saves the output."""
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Preprocessing {set_name} set ({len(data_list)} samples) and saving to {save_dir}...")
    num_errors = 0
 
    for i, data_item in enumerate(tqdm(data_list, desc=f"Processing {set_name}")):
        try:
            # Apply the MONAI transforms
            processed_data = transform(data_item)
            image_tensor = processed_data["image"] # Output of ConcatItemsd
            label_tensor = processed_data[config.TARGET_LABEL]

            # Verify tensor shapes and types (optional but recommended)
            if not isinstance(image_tensor, torch.Tensor) or not isinstance(label_tensor, torch.Tensor):
                 raise TypeError("Transform did not return tensors.")

            # Create save path
            subject_id = data_item['SubjectID'] # Assumes SubjectID is in data_item from prepare_data_list
            save_path = os.path.join(save_dir, f"{subject_id}.pt")

            # Save the dictionary containing image and label tensors
            torch.save({"image": image_tensor, "label": label_tensor}, save_path)

        except Exception as e:
            num_errors += 1
            logging.error(f"Error processing item {i} (Subject: {data_item.get('SubjectID', 'Unknown')}): {e}")
            # Optionally: save problematic IDs to a file

    if num_errors > 0:
        logging.warning(f"Completed preprocessing {set_name} with {num_errors} errors.")
    else:
        logging.info(f"Successfully preprocessed and saved {len(data_list)} samples for {set_name} set.")

def main(args):
    logging.info("--- Starting Preprocessing Script ---")
    torch.manual_seed(config.RANDOM_SEED)

    # ----- 1. Load and Prepare Data -----
    logging.info("Loading labels...")
    try:
        data_df = utils.load_labels(config.NAME_MAPPING_CSV)
        logging.info(f"Loaded {len(data_df)} subjects from {config.NAME_MAPPING_CSV}")
    except Exception as e:
        logging.error(f"Failed to load data: {e}. Exiting.")
        return

    logging.info("Adding image paths...")
    data_df = utils.add_image_paths(data_df, config.MODALITIES, config.RAW_DATA_DIR)

    logging.info("Checking file paths...")
    if not utils.check_file_paths(data_df, config.MODALITIES):
        logging.error("Missing files detected. Please check data directory and paths. Exiting.")
        return

    # ----- 2. Split Data -----
    logging.info("Performing stratified train/validation/test split...")
    data_df = data_df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    
    # For N=16 total samples (8 HGG, 8 LGG):
    # Aim for ~60% train, ~20% val, ~20% test
    # Test set: 3 samples (18.75% of 16). This should give ~1-2 of each class.
    # Remaining for train/val: 13 samples
    # Validation set: 3 samples from the 13 (approx 23%). This should give ~1-2 of each class.
    # Training set: 10 samples remaining.

    # First, split into (train+validation) and test
    # test_size for 3 out of 16 is 3/16 = 0.1875
    main_df, test_df = train_test_split(
        data_df, test_size=0.1875, random_state=config.RANDOM_SEED, stratify=data_df[config.TARGET_LABEL]
    )

    # Next, split main_df into train and validation
    # val_size for 3 out of the remaining 13 is 3/13 approx 0.23
    train_df, val_df = train_test_split(
        main_df, test_size=0.23, random_state=config.RANDOM_SEED, stratify=main_df[config.TARGET_LABEL]
    )

    utils.print_class_distribution(train_df, 'Training post-split')
    utils.print_class_distribution(val_df, 'Validation post-split')
    utils.print_class_distribution(test_df, 'Test post-split')

    # ----- 3. Get Transforms and Prepare Data Lists -----
    logging.info("Getting preprocessing transforms...")
    train_transforms, val_test_transforms = transforms.get_preprocessing_transforms(config.MODALITIES)

    logging.info("Preparing data lists...")
    train_data_list = utils.prepare_data_list(train_df, config.MODALITIES)
    val_data_list = utils.prepare_data_list(val_df, config.MODALITIES)
    test_data_list = utils.prepare_data_list(test_df, config.MODALITIES)

    # Add SubjectID to the dictionary for saving filename
    # Ensure SubjectID column exists in train_df, val_df, and test_df, which it should from load_labels via data_df
    for i, item in enumerate(train_data_list): item['SubjectID'] = train_df.iloc[i]['SubjectID']
    for i, item in enumerate(val_data_list): item['SubjectID'] = val_df.iloc[i]['SubjectID']
    for i, item in enumerate(test_data_list): item['SubjectID'] = test_df.iloc[i]['SubjectID']

    # ----- 4. Run Preprocessing and Saving -----
    start_time = time.time()
    preprocess_and_save(train_data_list, train_transforms, config.PREPROCESSED_TRAIN_DIR, "training")
    preprocess_and_save(val_data_list, val_test_transforms, config.PREPROCESSED_VAL_DIR, "validation")
    preprocess_and_save(test_data_list, val_test_transforms, config.PREPROCESSED_TEST_DIR, "test")
    end_time = time.time()

    logging.info(f"--- Preprocessing Finished in {end_time - start_time:.2f} seconds ---")

if __name__ == "__main__":
    # Currently no arguments, runs based on config.py
    # Could add args to specify output directories if needed
    parser = argparse.ArgumentParser(description="Preprocess Brain Tumor Data")
    # parser.add_argument(...) # Add arguments if needed
    args = parser.parse_args() # Parse arguments if any are defined
    main(args) 