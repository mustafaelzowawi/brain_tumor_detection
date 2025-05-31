"""
Configuration settings for the Brain Tumor Classification project.
"""
import os
import torch

# --- Project Structure ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root directory
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# --- Data Paths ---
# NIfTI files are organized like: data/raw_data/BraTS20_Training_XXX/BraTS20_Training_XXX_modality.nii
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
NAME_MAPPING_CSV = os.path.join(RAW_DATA_DIR, 'name_mapping.csv')

# Directory for preprocessed data (if using the preprocess script)
PREPROCESSED_DIR = os.path.join(DATA_DIR, 'processed_data')
PREPROCESSED_TRAIN_DIR = os.path.join(PREPROCESSED_DIR, 'train')
PREPROCESSED_VAL_DIR = os.path.join(PREPROCESSED_DIR, 'val')
PREPROCESSED_TEST_DIR = os.path.join(PREPROCESSED_DIR, 'test')


# --- Data Settings ---
MODALITIES = ['flair', 't1', 't1ce', 't2']
LABEL_MAPPING = {'LGG': 0, 'HGG': 1}
TARGET_LABEL = 'Label' # Column name for the numeric label in the DataFrame
SUBJECT_ID_COL = 'BraTS_2020_subject_ID' # Column name for subject ID in name_mapping.csv
GRADE_COL = 'Grade' # Column name for the grade (LGG/HGG) in name_mapping.csv


# --- Preprocessing/Transform Settings ---
RESIZED_SHAPE = (128, 128, 128) # Default size is (240,240,155)
INTENSITY_SCALE_MIN_A = -57
INTENSITY_SCALE_MAX_A = 164
INTENSITY_SCALE_MIN_B = 0.0
INTENSITY_SCALE_MAX_B = 1.0
INTENSITY_CLIP = True
CROP_FOREGROUND_SOURCE_KEY = 'flair' # Modality to use for CropForegroundd
ORIENTATION_AXCODES = "RAS"

# --- Training Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10
BATCH_SIZE = 2 # Adjust based on GPU memory
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 2
GRAD_CLIP_MAX_NORM = 1.0
USE_AMP = True # Use Automatic Mixed Precision

# --- Model Settings ---
MODEL_NAME = "resnet50_classifier"
NUM_CLASSES = 1 # Binary classification (LGG vs HGG)
NUM_INPUT_CHANNELS = len(MODALITIES) # Should match the number of modalities
PRETRAINED_RESNET50 = True # Set to True to use pretrained weights
PRETRAINED_WEIGHTS_PATH = os.path.join(MODEL_DIR, "pretrained_resnet50_medicalnet.pth") # Path to downloaded MedicalNet weights
NUM_CLASSES_PRETRAINED = 10 # Number of classes MedicalNet ResNet50 was trained on

# --- Model Architecture ---
SPATIAL_DIMS = 3 # For 3D convolutions in ResNet50

# --- Plot Settings ---
PLOTS_DIR = os.path.join(BASE_DIR, 'plots') # Directory for saving plots

# --- Evaluation Settings ---
EVAL_BATCH_SIZE = 4

# --- Other ---
RANDOM_SEED = 42
NUM_WORKERS = 0 # For DataLoader (increase if I/O is bottleneck)

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_TRAIN_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_VAL_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_TEST_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)