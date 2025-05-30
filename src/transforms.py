"""
MONAI transforms for data preprocessing and augmentation.
"""
import torch # Import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    RandFlipd,
    RandRotate90d,
    ConcatItemsd,
    EnsureTyped
)

from . import config

def get_preprocessing_transforms(modalities):
    """Returns MONAI transforms for training and validation/testing."""

    # Training transforms
    train_transforms = Compose([
        LoadImaged(keys=modalities),
        EnsureChannelFirstd(keys=modalities),
        Orientationd(keys=modalities, axcodes=config.ORIENTATION_AXCODES),
        ScaleIntensityRanged(
            keys=modalities,
            a_min=config.INTENSITY_SCALE_MIN_A,
            a_max=config.INTENSITY_SCALE_MAX_A,
            b_min=config.INTENSITY_SCALE_MIN_B,
            b_max=config.INTENSITY_SCALE_MAX_B,
            clip=config.INTENSITY_CLIP
        ),
        CropForegroundd(keys=modalities, source_key=config.CROP_FOREGROUND_SOURCE_KEY, allow_smaller=True),
        Resized(keys=modalities, spatial_size=config.RESIZED_SHAPE),
        # --- Augmentations ---
        RandFlipd(keys=modalities, spatial_axis=[0], prob=0.5),
        RandRotate90d(keys=modalities, prob=0.5, max_k=3, spatial_axes=(0, 1)),
        # -------------------
        ConcatItemsd(keys=modalities, name="image"), # Concatenate modalities into a single multi-channel image
        EnsureTyped(keys=["image", config.TARGET_LABEL], dtype=torch.float32) # Ensure image and label are float tensors
    ])

    # Validation and Test transforms (no augmentation)
    val_test_transforms = Compose([
        LoadImaged(keys=modalities),
        EnsureChannelFirstd(keys=modalities),
        Orientationd(keys=modalities, axcodes=config.ORIENTATION_AXCODES),
        ScaleIntensityRanged(
            keys=modalities,
            a_min=config.INTENSITY_SCALE_MIN_A,
            a_max=config.INTENSITY_SCALE_MAX_A,
            b_min=config.INTENSITY_SCALE_MIN_B,
            b_max=config.INTENSITY_SCALE_MAX_B,
            clip=config.INTENSITY_CLIP
        ),
        CropForegroundd(keys=modalities, source_key=config.CROP_FOREGROUND_SOURCE_KEY, allow_smaller=True),
        Resized(keys=modalities, spatial_size=config.RESIZED_SHAPE),
        ConcatItemsd(keys=modalities, name="image"),
        EnsureTyped(keys=["image", config.TARGET_LABEL], dtype=torch.float32)
    ])

    return train_transforms, val_test_transforms 