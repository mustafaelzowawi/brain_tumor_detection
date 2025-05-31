"""
Main script for training the Brain Tumor Classification model.
"""
import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from . import config
from . import utils
from . import transforms
from . import dataset
from . import model as model_definition # Use alias to avoid conflict with torch.nn.Module

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log_file = os.path.join(config.LOG_DIR, 'training.log')
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)

def plot_training_history(history, plot_dir):
    """Plots training and validation loss and accuracy."""
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Training history plot saved to {plot_path}")

def train_one_epoch(model, train_loader, loss_function, optimizer, device, scaler):
    """Runs a single training epoch."""
    model.train()
    epoch_loss = 0
    epoch_correct = 0
    total_samples = 0

    start_time = time.time()
    for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
        batch_data = batch_data.to(device, dtype=torch.float)
        batch_labels = batch_labels.to(device, dtype=torch.float) # BCEWithLogitsLoss expects float

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            outputs = model(batch_data)
            loss = loss_function(outputs, batch_labels)

        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"NaN or Inf loss detected at batch {batch_idx}. Skipping batch.")
            continue

        # Scale loss and backward pass
        scaler.scale(loss).backward()

        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_MAX_NORM)

        # Optimizer step
        scaler.step(optimizer)

        # Update scaler
        scaler.update()

        # Accumulate loss and accuracy
        epoch_loss += loss.item() * batch_data.size(0)
        preds = torch.sigmoid(outputs) >= 0.5
        epoch_correct += torch.sum(preds == batch_labels.bool()).item()
        total_samples += batch_data.size(0)

        if (batch_idx + 1) % 50 == 0: # Log progress every 50 batches
            logging.info(f"  Batch {batch_idx+1}/{len(train_loader)} - Current Loss: {loss.item():.4f}")

    epoch_loss /= total_samples
    epoch_accuracy = epoch_correct / total_samples
    epoch_time = time.time() - start_time
    logging.info(f"Train Epoch Time: {epoch_time:.2f}s")
    return epoch_loss, epoch_accuracy

def validate_one_epoch(model, val_loader, loss_function, device):
    """Runs a single validation epoch."""
    model.eval()
    val_loss = 0
    val_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(device, dtype=torch.float)
            batch_labels = batch_labels.to(device, dtype=torch.float)

            with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                outputs = model(batch_data)
                loss = loss_function(outputs, batch_labels)

            val_loss += loss.item() * batch_data.size(0)
            preds = torch.sigmoid(outputs) >= 0.5
            val_correct += torch.sum(preds == batch_labels.bool()).item()
            total_samples += batch_data.size(0)

    val_loss /= total_samples
    val_accuracy = val_correct / total_samples
    return val_loss, val_accuracy

def main(args):
    logging.info("--- Starting Training Process ---")
    device = torch.device(args.device) if args.device else config.DEVICE
    use_amp = args.amp if args.amp is not None else config.USE_AMP
    num_epochs = args.epochs if args.epochs else config.NUM_EPOCHS
    batch_size = args.batch_size if args.batch_size else config.BATCH_SIZE
    learning_rate = args.lr if args.lr else config.LEARNING_RATE
    num_workers = args.num_workers if args.num_workers is not None else config.NUM_WORKERS
    load_preprocessed = args.load_preprocessed

    logging.info(f"Using device: {device}")
    logging.info(f"Using Automatic Mixed Precision: {use_amp}")
    logging.info(f"Number of workers: {num_workers}")
    logging.info(f"Load preprocessed data: {load_preprocessed}")
    torch.manual_seed(config.RANDOM_SEED)
    if device == torch.device("cuda"): torch.cuda.manual_seed(config.RANDOM_SEED)

    train_df_for_weights = None # Initialize

    if load_preprocessed:
        logging.info("Creating datasets from preprocessed files...")
        try:
            # Pass load_test=False as train.py only needs train_ds and val_ds for its loop.
            # The test set is handled by evaluate.py
            train_ds, val_ds, _ = dataset.get_preprocessed_datasets(load_test=False)
            # For pos_weight calculation, we will iterate train_ds directly later.
            logging.info(f"Loaded {len(train_ds)} training samples and {len(val_ds)} validation samples from preprocessed files.")
            # No need to create test_set.csv here, as it should be handled by preprocess.py or full run
        except FileNotFoundError as e:
             logging.error(f"Failed to load preprocessed data: {e}. Exiting.")
             return
        except Exception as e:
             logging.error(f"An error occurred while loading preprocessed data: {e}. Exiting.")
             return
    else:
        # Load and Prepare Data (Only if not loading preprocessed)
        logging.info("Loading labels...")
        try:
            data_df = utils.load_labels(config.NAME_MAPPING_CSV)
        except Exception as e:
            logging.error(f"Failed to load data: {e}. Exiting.")
            return

        logging.info("Adding image paths...")
        data_df = utils.add_image_paths(data_df, config.MODALITIES, config.RAW_DATA_DIR)

        logging.info("Checking file paths...")
        if not utils.check_file_paths(data_df, config.MODALITIES):
            logging.error("Missing files detected. Please check data directory and paths. Exiting.")
            return

        logging.info("Performing stratified train/validation/test split...")
        # Shuffle before splitting
        data_df = data_df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)

        # Split: ~62.6% train, ~18.7% validation, ~18.7% test (to match preprocess.py)
        # This split is now only for the case where load_preprocessed is False.
        main_df, test_df = train_test_split(
            data_df, test_size=0.1875, random_state=config.RANDOM_SEED, stratify=data_df[config.TARGET_LABEL]
        )
        train_df, val_df = train_test_split(
            main_df, test_size=0.23, random_state=config.RANDOM_SEED, stratify=main_df[config.TARGET_LABEL]
        )
        train_df_for_weights = train_df # Used for pos_weight if not load_preprocessed

        utils.print_class_distribution(train_df, 'Training (raw data path)')
        utils.print_class_distribution(val_df, 'Validation')
        utils.print_class_distribution(test_df, 'Test')

        # Save test set for later evaluation
        test_csv_path = os.path.join(config.DATA_DIR, 'test_set.csv')
        test_df.to_csv(test_csv_path, index=False)
        logging.info(f"Test set saved to {test_csv_path}")

        # Create Datasets and DataLoaders (Only if not loading preprocessed)
        logging.info("Getting preprocessing transforms...")
        train_transforms, val_test_transforms = transforms.get_preprocessing_transforms(config.MODALITIES)

        logging.info("Preparing data lists for MONAI datasets...")
        train_data_list = utils.prepare_data_list(train_df, config.MODALITIES)
        val_data_list = utils.prepare_data_list(val_df, config.MODALITIES)

        logging.info("Creating MONAI datasets (using CacheDataset)...")
        train_ds, val_ds, _ = dataset.get_monai_datasets(
            train_data_list, val_data_list, [], # Pass empty list for test_data_list for now
            train_transforms, val_test_transforms
        )

    logging.info("Creating DataLoaders...")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False, num_workers=num_workers) # Keep eval batch size from config for now

    # Initialize Model, Loss, Optimizer 
    logging.info("Initializing model...")
    model = model_definition.get_model().to(device)

    # Handle DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    logging.info("Setting up loss function and optimizer...")
    
    if load_preprocessed:
        # Iterate through the train_ds to calculate class weights if using preprocessed data
        logging.info("Calculating class weights from preprocessed training dataset...")
        num_neg = 0
        num_pos = 0
        # Create a temporary DataLoader to iterate through train_ds once
        # This is inefficient for large datasets but fine for calculating pos_weight once.
        temp_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False) # No shuffle needed
        for _, batch_labels in temp_loader:
            num_pos += torch.sum(batch_labels == 1).item()
            num_neg += torch.sum(batch_labels == 0).item()
        del temp_loader # free memory
        if num_pos == 0 and num_neg == 0: # Should not happen if train_ds is not empty
            logging.warning("Preprocessed training dataset appears empty or contains no labels. Using default pos_weight=1.0")
            pos_weight_value = 1.0
        elif num_pos == 0: # Avoid division by zero if no positive samples
            logging.warning("No positive samples found in preprocessed training dataset. Using default pos_weight=1.0 for BCEWithLogitsLoss.")
            pos_weight_value = 1.0
        else:
            pos_weight_value = num_neg / num_pos
    else:
        # This part will only be effective if load_preprocessed is False
        # and data_df is large enough for the 3-way split.
        if train_df_for_weights is None or train_df_for_weights.empty:
            logging.error("train_df_for_weights is None or empty when trying to calculate pos_weight. This should not happen if load_preprocessed is False and data loading was successful.")
            # Fallback or raise error
            pos_weight_value = 1.0
        else:
            num_neg = len(train_df_for_weights[train_df_for_weights[config.TARGET_LABEL] == 0])
            num_pos = len(train_df_for_weights[train_df_for_weights[config.TARGET_LABEL] == 1])
            pos_weight_value = num_neg / num_pos if num_pos > 0 else 1.0

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    logging.info(f"Calculated pos_weight for BCEWithLogitsLoss: {pos_weight.item():.4f}")
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE, verbose=True
    )

    # Initialize GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Training Loop 
    logging.info("--- Starting Training Loop ---")
    best_val_loss = float('inf')
    training_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    epochs_no_improve = 0
    early_stopping_patience = 5 # Stop after 5 epochs with no improvement in val_loss

    for epoch in range(num_epochs):
        logging.info(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        start_epoch_time = time.time()

        train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_function, optimizer, device, scaler)
        logging.info(f"Epoch {epoch+1} Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        val_loss, val_accuracy = validate_one_epoch(model, val_loader, loss_function, device)
        logging.info(f"Epoch {epoch+1} Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_accuracy)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_accuracy)

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model_save_dir = os.path.join(config.MODEL_DIR, "classification")
            os.makedirs(model_save_dir, exist_ok=True)
            model_save_path = os.path.join(model_save_dir, f"{config.MODEL_NAME}_best.pth")
            # Save model state_dict 
            # Save module's state_dict if using DataParallel
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), model_save_path)
            else:
                torch.save(model.state_dict(), model_save_path)
            logging.info(f"Validation loss improved. Saved best model to {model_save_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"Validation loss did not improve. ({epochs_no_improve}/{early_stopping_patience})")

        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        epoch_duration = time.time() - start_epoch_time
        logging.info(f"Epoch {epoch+1} duration: {epoch_duration:.2f}s")

    # Save the final model
    final_model_save_dir = os.path.join(config.MODEL_DIR, "classification")
    os.makedirs(final_model_save_dir, exist_ok=True)
    final_model_path = os.path.join(final_model_save_dir, f"{config.MODEL_NAME}_final.pth")
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)
    logging.info(f"Saved final model to {final_model_path}")

    # Plot training history
    plot_training_history(training_history, config.PLOTS_DIR)

    logging.info("--- Training Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brain Tumor Classification Model")
    parser.add_argument('--epochs', type=int, help=f"Number of training epochs (default: {config.NUM_EPOCHS})")
    parser.add_argument('--batch_size', type=int, help=f"Training batch size (default: {config.BATCH_SIZE})")
    parser.add_argument('--lr', type=float, help=f"Learning rate (default: {config.LEARNING_RATE})")
    parser.add_argument('--device', type=str, help=f"Device to use (e.g., 'cuda', 'cpu') (default: from config)")
    parser.add_argument('--amp', type=bool, help=f"Use Automatic Mixed Precision (default: {config.USE_AMP})")
    parser.add_argument('--num_workers', type=int, help=f"Number of DataLoader workers (default: {config.NUM_WORKERS})")
    parser.add_argument('--load_preprocessed', action='store_true', help="Load data from preprocessed .pt files instead of raw NIfTI files.")

    args = parser.parse_args()
    main(args) 