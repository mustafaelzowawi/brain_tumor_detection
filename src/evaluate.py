"""
Script for evaluating the trained Brain Tumor Classification model on the test set.
"""
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


from . import config
from . import utils
from . import transforms
from . import dataset
from . import model as model_definition

# Setup logging (evaluation specific)
log_file = os.path.join(config.LOG_DIR, 'evaluation.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def plot_confusion_matrix(cm, class_names, plot_dir):
    """Plots and saves the confusion matrix."""
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    
    plot_path = os.path.join(plot_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Confusion matrix plot saved to {plot_path}")

def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test set."""
    model.eval()
    all_preds = []
    all_labels = []

    logging.info(f"Starting evaluation on {len(test_loader.dataset)} test samples...")
    with torch.no_grad():
        for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
            batch_data = batch_data.to(device, dtype=torch.float)
            batch_labels = batch_labels.to(device, dtype=torch.float)

            with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                outputs = model(batch_data)

            # Apply sigmoid and threshold for predictions
            preds = torch.sigmoid(outputs) >= 0.5

            all_preds.extend(preds.cpu().numpy().flatten()) # Flatten predictions
            all_labels.extend(batch_labels.cpu().numpy().flatten()) # Flatten labels

            if (batch_idx + 1) % 20 == 0:
                logging.info(f"  Evaluated batch {batch_idx+1}/{len(test_loader)}")

    # Ensure labels are integers for classification_report
    all_labels = np.array(all_labels).astype(int)
    all_preds = np.array(all_preds).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                     target_names=[k for k,v in sorted(config.LABEL_MAPPING.items(), key=lambda item: item[1])])
    cm = confusion_matrix(all_labels, all_preds)

    logging.info("--- Evaluation Results ---")
    logging.info(f"Test Accuracy: {accuracy:.4f}")
    logging.info("\nClassification Report:")
    print(report) # Print report directly for better formatting
    logging.info("\nConfusion Matrix:")
    print(cm) # Print CM directly

    logging.info("Classification Report (logged):\n" + report)
    logging.info("Confusion Matrix (logged):\n" + str(cm))

    # Plot and save confusion matrix
    class_names = [k for k,v in sorted(config.LABEL_MAPPING.items(), key=lambda item: item[1])]
    plot_confusion_matrix(cm, class_names, config.PLOTS_DIR)

    return accuracy, report, cm

def main(args):
    # Setup evaluation-specific logging
    logging.info("--- Starting Evaluation Process ---")

    # Use args or config values
    device = torch.device(args.device) if args.device else config.DEVICE
    load_preprocessed = args.load_preprocessed
    model_path_arg = args.model_path
    num_workers = args.num_workers if args.num_workers is not None else config.NUM_WORKERS

    logging.info(f"Using device: {device}")
    logging.info(f"Load preprocessed data: {load_preprocessed}")

    # Load Test Data Information
    test_csv_path = os.path.join(config.DATA_DIR, 'test_set.csv')
    if not os.path.exists(test_csv_path):
        logging.error(f"Test set CSV not found at {test_csv_path}. Run train.py first. Exiting.")
        return

    logging.info(f"Loading test set information from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)

    # Check if file paths need to be reconstructed
    # Assuming the CSV contains the SubjectID
    if config.MODALITIES[0] not in test_df.columns:
        logging.info("Reconstructing image paths for test set...")
        test_df = utils.add_image_paths(test_df, config.MODALITIES, config.RAW_DATA_DIR)
        # Optionally re-check paths
        if not utils.check_file_paths(test_df, config.MODALITIES):
             logging.error("Missing files detected in test set. Exiting.")
             return

    # Create Test Dataset and DataLoader
    if load_preprocessed:
        logging.info("Creating test dataset from preprocessed files...")
        try:
            _, _, test_ds = dataset.get_preprocessed_datasets()
        except FileNotFoundError as e:
            logging.error(f"Failed to load preprocessed test data: {e}. Exiting.")
            return
        except Exception as e:
             logging.error(f"An error occurred while loading preprocessed test data: {e}. Exiting.")
             return
    else:
        logging.info("Getting preprocessing transforms (validation/test mode)...")
        _, val_test_transforms = transforms.get_preprocessing_transforms(config.MODALITIES)

        logging.info("Preparing test data list...")
        test_data_list = utils.prepare_data_list(test_df, config.MODALITIES)

        logging.info("Creating MONAI test dataset (using CacheDataset)...")
        test_ds = dataset.CacheDataset(
            data=test_data_list,
            transform=val_test_transforms,
            cache_rate=1.0,
            num_workers=num_workers
        )

    logging.info("Creating Test DataLoader...")
    test_loader = DataLoader(test_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # Load trained model
    # Determine model path: argument > config default best > config default final
    if model_path_arg:
        model_path = model_path_arg
        if not os.path.exists(model_path):
            logging.error(f"Specified model path does not exist: {model_path}. Exiting.")
            return
    else:
        #Modified Path
        model_base_dir = os.path.join(config.MODEL_DIR, "classification")
        model_path = os.path.join(model_base_dir, f"{config.MODEL_NAME}_best.pth")
        if not os.path.exists(model_path):
            logging.warning(f"Default best model ({model_path}) not found in {model_base_dir}. Trying final model...")
            model_path = os.path.join(model_base_dir, f"{config.MODEL_NAME}_final.pth")
            if not os.path.exists(model_path):
                logging.error(f"No trained model found in {model_base_dir}. Run train.py first or specify --model_path. Exiting.")
                return

    logging.info(f"Loading trained model from {model_path}...")
    model = model_definition.get_model() # Initialize model structure
    # Load state dict
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info("Model state_dict loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model state_dict: {e}")
        # Attempt to load assuming DataParallel wrapper if it fails
        try:
            logging.info("Attempting to load model assuming DataParallel wrapper...")
            state_dict = torch.load(model_path, map_location=device)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.` prefix
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            logging.info("Model state_dict loaded successfully after removing 'module.' prefix.")
        except Exception as inner_e:
             logging.error(f"Failed to load model state_dict even after handling DataParallel: {inner_e}. Exiting.")
             return

    model = model.to(device)

    # Run Evaluation
    evaluate_model(model, test_loader, device)

    logging.info("--- Evaluation Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor Classification Model")
    parser.add_argument('--model_path', type=str, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument('--device', type=str, help=f"Device to use (e.g., 'cuda', 'cpu') (default: from config)")
    parser.add_argument('--num_workers', type=int, help=f"Number of DataLoader workers (default: {config.NUM_WORKERS})")
    parser.add_argument('--load_preprocessed', action='store_true', help="Load data from preprocessed .pt files instead of raw NIfTI files.")

    args = parser.parse_args()
    main(args) 