"""
Defines the 3D ResNet classifier model.
"""
import torch
import torch.nn as nn
import logging
from monai.networks.nets import resnet50
import os
from collections import OrderedDict

from . import config

# Setup logging for this module if not already configured globally
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_base_resnet50_model():
    """
    Creates and returns a MONAI ResNet50 model instance.
    If config.PRETRAINED_RESNET50 is True, it attempts to load MONAI's MedicalNet weights (which might fail).
    If False, or if pretraining fails, it returns a ResNet50 with random weights.
    The returned model's first conv layer and FC layer will be replaced by ResNet50Classifier.
    """
    logger.info(f"Attempting to load MONAI ResNet50 (config.PRETRAINED_RESNET50: {config.PRETRAINED_RESNET50})...")

    if config.PRETRAINED_RESNET50:
        try:
            # Initialize the model structure first, without MONAI's `pretrained=True`
            # as we'll load weights manually.
            # n_input_channels should be what MedicalNet expects (typically 3 for RGB-like, or 1 if single modality).
            # We will use 3 as a common case for MedicalNet, this might need adjustment
            # based on the specific MedicalNet weights being used. The ResNet50Classifier
            # will later adapt the very first layer to config.NUM_INPUT_CHANNELS.
            # num_classes should also be what MedicalNet was trained on.
            logger.info(f"Initializing base ResNet50 structure for pretrained weights: SPATIAL_DIMS={config.SPATIAL_DIMS}, n_input_channels=3 (expected by MedicalNet), num_classes={config.NUM_CLASSES_PRETRAINED}")
            model = resnet50(
                spatial_dims=config.SPATIAL_DIMS,
                n_input_channels=3, # Or 1, depending on MedicalNet version
                num_classes=config.NUM_CLASSES_PRETRAINED,
                pretrained=False, # We load weights manually
                feed_forward=True,
                progress=True
            )
            
            weights_path = config.PRETRAINED_WEIGHTS_PATH
            if os.path.exists(weights_path):
                logger.info(f"Loading pretrained weights from: {weights_path}")
                # MONAI's MedicalNet weights might be a dict with a 'state_dict' key
                # or just the state_dict itself.
                # Common practice for MONAI's MedicalNet is that it's a full model checkpoint.
                # We need to load the state_dict.
                checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
                
                # Heuristic to find the state_dict:
                # It might be directly the checkpoint, or under a 'state_dict' key,
                # or even 'model' or 'net'.
                state_dict_to_load = None
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict_to_load = checkpoint['state_dict']
                        logger.info("Loaded state_dict from 'state_dict' key in checkpoint.")
                    elif 'model' in checkpoint: # Common in some training scripts
                        state_dict_to_load = checkpoint['model']
                        logger.info("Loaded state_dict from 'model' key in checkpoint.")
                    elif 'net' in checkpoint: # Another common key
                        state_dict_to_load = checkpoint['net']
                        logger.info("Loaded state_dict from 'net' key in checkpoint.")
                    else: # Check if the checkpoint itself is a state_dict
                        # A common check: does it have typical layer keys?
                        if any(k.endswith('.weight') or k.endswith('.bias') for k in checkpoint.keys()):
                            state_dict_to_load = checkpoint
                            logger.info("Checkpoint itself appears to be a state_dict.")
                        else:
                            logger.error(f"Could not find state_dict in checkpoint dict from {weights_path}. Keys: {list(checkpoint.keys())}")
                            raise RuntimeError(f"Unrecognized checkpoint format in {weights_path}")
                elif isinstance(checkpoint, OrderedDict) or isinstance(checkpoint, dict): # Direct state_dict
                     state_dict_to_load = checkpoint
                     logger.info("Checkpoint loaded directly as a state_dict.")
                else:
                    logger.error(f"Pretrained weights file {weights_path} is not a recognized format (dict or state_dict). Type: {type(checkpoint)}")
                    raise RuntimeError(f"Unrecognized format for pretrained weights at {weights_path}")

                if state_dict_to_load:
                    # MONAI's MedicalNet weights might have a 'module.' prefix if saved from DataParallel
                    # We need to remove this prefix if our model is not wrapped in DataParallel at this stage.
                    # Also, the ResNet architecture in MONAI can sometimes have keys like 'model.conv1.weight'
                    # if it's nested. The resnet50() function returns the core model directly.
                    
                    # Standardize keys: remove "module." prefix if present
                    new_state_dict = OrderedDict()
                    has_module_prefix = any(key.startswith("module.") for key in state_dict_to_load.keys())
                    
                    for k, v in state_dict_to_load.items():
                        name = k
                        if has_module_prefix and k.startswith("module."):
                            name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    
                    # Check for 'model.' prefix which can occur if the saved model was `self.model = resnet50(...)`
                    # and `torch.save(model.state_dict())` was called instead of `torch.save(model.model.state_dict())`.
                    # Or if the MedicalNet weights themselves are structured this way.
                    # The `resnet50` function from MONAI returns the network directly, so keys should not have 'model.'
                    # However, if the pretrained file *does* have it, we need to strip it.
                    # Let's check a few typical keys.
                    if not any(k.startswith("conv1.") or k.startswith("layer1.") for k in new_state_dict.keys()):
                        # Possible nesting like 'model.conv1.weight'
                        if any(k.startswith("model.conv1.") or k.startswith("model.layer1.") for k in new_state_dict.keys()):
                            logger.info("Detected 'model.' prefix in state_dict keys. Attempting to strip it.")
                            stripped_state_dict = OrderedDict()
                            for k, v in new_state_dict.items():
                                if k.startswith("model."):
                                    stripped_state_dict[k[6:]] = v # remove 'model.'
                                else: # Should not happen if consistent
                                    stripped_state_dict[k] = v
                            new_state_dict = stripped_state_dict

                    # The ResNet50Classifier will replace conv1 and fc, so we don't strictly need to load them,
                    # but it's cleaner if the state_dict matches.
                    # If MedicalNet has a different number of input channels for its conv1
                    # or a different number of output classes for its fc, `load_state_dict` might complain
                    # about mismatched shapes for these two layers if `strict=True`.
                    # The ResNet50Classifier replaces these anyway.
                    # We can load with strict=False to ignore mismatches for layers that will be replaced.
                    try:
                        model.load_state_dict(new_state_dict, strict=False)
                        logger.info("Successfully loaded pretrained weights into the base ResNet50 model (strict=False).")
                        logger.info("Layers conv1 and fc will be reinitialized by ResNet50Classifier.")
                    except RuntimeError as e:
                        logger.error(f"RuntimeError loading state_dict with strict=False: {e}")
                        logger.error("This might indicate a significant architecture mismatch beyond conv1/fc, or incorrect key names.")
                        logger.error(f"Model keys: {[k for k, _ in model.named_parameters()][:10]}")
                        logger.error(f"Loaded state_dict keys: {list(new_state_dict.keys())[:10]}")
                        raise
                else: # Should have been caught by earlier checks
                    raise RuntimeError("State dict to load was not populated.")

            else:
                logger.error(f"Pretrained weights file not found at: {weights_path}")
                logger.warning("Falling back to ResNet50 with random weights because weights file was not found.")
                # Initialize with random weights matching our target channels/classes directly
                model = resnet50(
                    spatial_dims=config.SPATIAL_DIMS,
                    n_input_channels=config.NUM_INPUT_CHANNELS,
                    num_classes=1000, # Placeholder, will be replaced
                    pretrained=False
                )
            return model
        except NotImplementedError as nie: # This was the original concern with pretrained=True
            logger.error(f"MONAI's pretrained=True for ResNet50 failed (NotImplementedError): {nie}")
            logger.warning("Falling back to ResNet50 with random weights because PRETRAINED_RESNET50=True strategy encountered an error.")
        except FileNotFoundError as fnfe:
            logger.error(f"Pretrained weights file not found: {fnfe}")
            logger.warning("Falling back to ResNet50 with random weights.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while trying to load pretrained ResNet50: {e}")
            logger.warning("Falling back to ResNet50 with random weights due to an unexpected error.")
    
    # Fallback or if PRETRAINED_RESNET50 is False: Initialize with random weights.
    # The ResNet50Classifier will adapt the input channels and output classes.
    # For the base model here, num_classes for the internal FC layer doesn't strictly matter
    # as it will be replaced, but let's use a common default like 1000 (ImageNet size) or config.NUM_CLASSES.
    # n_input_channels here is also less critical as it's replaced, but good to be explicit.
    logger.info("Initializing MONAI ResNet50 with random weights.")
    model = resnet50(
        spatial_dims=config.SPATIAL_DIMS,
        n_input_channels=config.NUM_INPUT_CHANNELS, # Will be adapted, but set for base model
        num_classes=1000, # Standard for many ResNets, will be replaced by ResNet50Classifier
        pretrained=False # Explicitly False for random weights
    )
    logger.info("MONAI ResNet50 with random weights initialized.")
    return model

class ResNet50Classifier(nn.Module):
    """
    A 3D ResNet50 classifier adapted for multi-modal input.
    Takes a pretrained ResNet50, modifies the first conv layer for specified input channels,
    and adds a final classification layer.
    """
    def __init__(self, base_model, num_input_channels_target, num_classes_target):
        super(ResNet50Classifier, self).__init__()
        self.features = base_model

        # 1. Adapt the first convolutional layer for the desired number of input channels
        # Original first conv layer in ResNet50: self.features.conv1
        # We need to check its attributes to correctly create a new one.
        original_conv1 = self.features.conv1
        
        logger.info(f"Adapting ResNet50's first conv layer from {original_conv1.in_channels} to {num_input_channels_target} input channels...")
        
        self.features.conv1 = nn.Conv3d(
            in_channels=num_input_channels_target,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=(original_conv1.bias is not None)
        )
        # Initialize new conv1 weights (e.g., Kaiming normal)
        nn.init.kaiming_normal_(self.features.conv1.weight, mode='fan_out', nonlinearity='relu')
        if self.features.conv1.bias is not None:
            nn.init.constant_(self.features.conv1.bias, 0)
        logger.info("New conv1 layer created and weights initialized.")

        # 2. Replace the final fully connected layer (classifier head)
        # The MONAI ResNet50 should have an 'fc' attribute for its classifier.
        if not hasattr(self.features, 'fc'):
            raise AttributeError("The provided ResNet50 base model does not have an 'fc' (fully connected) attribute as expected.")
        
        num_ftrs = self.features.fc.in_features
        self.features.fc = nn.Linear(num_ftrs, num_classes_target)
        logger.info(f"Replaced ResNet50's fc layer. Input features: {num_ftrs}, Output classes: {num_classes_target}")
        # Weights of the new fc layer are initialized by default by nn.Linear

    def forward(self, x):
        return self.features(x)

def get_model():
    """
    Instantiates the ResNet50Classifier model.
    It first gets a base ResNet50 (either with attempted pretraining or random weights)
    and then adapts it using ResNet50Classifier.
    """
    logger.info("Creating model instance...")
    
    # Get the base ResNet50 model
    base_resnet_model = get_base_resnet50_model()

    if base_resnet_model is None:
        logger.error("Failed to obtain a base ResNet50 model. Cannot proceed.")
        raise RuntimeError("Failed to get a base ResNet50 model.")

    # Create the classifier by adapting the base model
    model = ResNet50Classifier(
        base_model=base_resnet_model,
        num_input_channels_target=config.NUM_INPUT_CHANNELS,
        num_classes_target=config.NUM_CLASSES
    )
    logger.info("ResNet50Classifier model created successfully.")
    return model 

# Example of how you might add a config for MedicalNet's original number of classes if needed:
# In config.py:
# NUM_CLASSES_PRETRAINED = 10 # Or whatever MedicalNet was trained on for ResNet50
# If not defined, we can use a default or handle it.
if not hasattr(config, 'NUM_CLASSES_PRETRAINED'):
    logger.warning("config.NUM_CLASSES_PRETRAINED not found, using default of 10 for base model if pretraining.")
    config.NUM_CLASSES_PRETRAINED = 10 # Default if not in config 