import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from skimage import io # For .tif files
from skimage.transform import resize
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 # Required for Albumentations' border_mode

# Import your UNet model from model.py
from model import UNet

def load_image_for_prediction(image_path, n_channels=3, target_size=(512, 512)):
    """
    Loads an image, explicitly resizes it, and returns it as a NumPy array (HWC float32, 0-255 range).
    Ensures consistent HWC format.
    """
    try:
        if image_path.lower().endswith(('.tif', '.tiff')):
            img_np = io.imread(image_path).astype(np.float32)
        else:
            img = Image.open(image_path).convert('RGB' if n_channels == 3 else 'L')
            img_np = np.array(img).astype(np.float32)

        # Ensure image is float32 and in 0-255 range for Albumentations Normalize
        if img_np.max() <= 1.0 and img_np.max() > 0:
            img_np = (img_np * 255.0)
        
        # Handle channel dimensions for image (HWC format)
        if n_channels == 1 and img_np.ndim == 3:
             img_np = np.mean(img_np, axis=2, keepdims=True)
        elif n_channels == 3 and img_np.ndim == 2:
             img_np = np.stack([img_np, img_np, img_np], axis=-1)
             
        if img_np.ndim == 2:
            img_np = img_np[..., np.newaxis]

        # Explicitly resize here to ensure consistent dimensions
        if img_np.shape[:2] != target_size:
            img_resized = resize(img_np, (*target_size, n_channels), anti_aliasing=True, preserve_range=True)
            return img_resized.astype(np.float32)
        return img_np
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def predict_mask_ensemble(model_paths, image_path, device, transform, n_channels, image_size, output_path=None):
    """
    Loads multiple models, makes predictions, averages probabilities, and optionally saves the binary mask.

    Args:
        model_paths (list): List of paths to the trained UNet model state_dicts.
        image_path (str): Path to the input image.
        device (torch.device): The device (cuda or cpu) to run inference on.
        transform (A.Compose): Albumentations transform for preprocessing.
        n_channels (int): Number of channels in the input image (1 or 3).
        image_size (tuple): (height, width) of the input images/model output.
        output_path (str, optional): Path to save the predicted mask. If None, mask is not saved.
    
    Returns:
        numpy.ndarray: The ensembled predicted binary mask (H, W).
    """
    # Load and preprocess the image once for all models
    image_np = load_image_for_prediction(image_path, n_channels, image_size)
    if image_np is None:
        print(f"Could not load image: {image_path}. Skipping prediction.")
        return None

    # Apply transformations to get a tensor
    if transform:
        augmented = transform(image=image_np)
        image_tensor = augmented['image']
    else:
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

    image_tensor = image_tensor.unsqueeze(0).to(device) # Add batch dimension

    all_probabilities = []

    for model_path in model_paths:
        # print(f"  Loading model: {os.path.basename(model_path)}") # Commented out to reduce verbosity in loop
        model = UNet(n_channels=n_channels, n_classes=1, bilinear=True, 
                     simclr_encoder_path=None, # Weights are in .pt file
                     freeze_encoder=False).to(device)
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"    Error loading model state dict from {os.path.basename(model_path)}: {e}. Skipping this model.")
            continue # Skip to the next model if loading fails

        model.eval() # Set model to evaluation mode
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.sigmoid(outputs)
            all_probabilities.append(probabilities.squeeze().cpu().numpy()) # Squeeze batch and channel dims

    if not all_probabilities:
        print("No models were successfully loaded for ensemble prediction for this image.")
        return None

    # Average the probabilities across all models
    averaged_probabilities = np.mean(np.stack(all_probabilities), axis=0) # Stack and then mean over the first axis

    # Convert averaged probabilities to binary mask using a threshold (e.g., 0.5)
    final_mask_np = (averaged_probabilities > 0.95).astype(np.uint8)

    # Save the predicted mask if output_path is provided
    if output_path:
        mask_to_save = (final_mask_np * 255).astype(np.uint8)
        try:
            Image.fromarray(mask_to_save, mode='L').save(output_path)
            # print(f"Ensembled predicted mask saved to: {output_path}") # Commented out for loop verbosity
        except Exception as e:
            print(f"Error saving ensembled predicted mask to {output_path}: {e}")

    return final_mask_np


if __name__ == "__main__":
    # --- Configuration ---
    MODEL_SAVE_DIR = "path_to_model_save"

    INPUT_IMAGE_DIR = "path_to_input_image_dir" 

    PREDICTION_OUTPUT_DIR = "path_to_save_masks"
    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

    IMAGE_SIZE = (512, 512)
    N_CHANNELS = 3 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Preprocessing Transform (MUST match validation transform used during training) ---
    predict_transform = A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406) if N_CHANNELS == 3 else (0.5,),
            std=(0.229, 0.224, 0.225) if N_CHANNELS == 3 else (0.5,),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    # --- Find all trained models for ensembling ---
    # Get a list of all .pt files in your model save directory
    all_model_files = [os.path.join(MODEL_SAVE_DIR, f) for f in os.listdir(MODEL_SAVE_DIR) if f.endswith('.pt')]
    
    if not all_model_files:
        print(f"Error: No trained models found in {MODEL_SAVE_DIR}. Please run training first.")
        exit()
    else:
        print(f"Found {len(all_model_files)} models for ensembling.")
        for model_file in all_model_files:
            print(f"  - {os.path.basename(model_file)}")

    # --- Iterate through images in the input directory and make ensemble predictions ---
    print(f"\nStarting ensemble prediction for images in {INPUT_IMAGE_DIR}...")
    image_files_to_predict = [f for f in os.listdir(INPUT_IMAGE_DIR) if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg'))]
    
    if not image_files_to_predict:
        print(f"No image files found in {INPUT_IMAGE_DIR}. Please check the directory path and file types.")
    else:
        for image_filename in sorted(image_files_to_predict):
            current_image_path = os.path.join(INPUT_IMAGE_DIR, image_filename)
            print(f"\nProcessing: {image_filename}")
            
            output_filename = os.path.basename(current_image_path).rsplit('.', 1)[0] + "_ensembled_mask.png"
            output_mask_path = os.path.join(PREDICTION_OUTPUT_DIR, output_filename)

            ensembled_mask = predict_mask_ensemble(all_model_files, current_image_path, device, predict_transform, N_CHANNELS, IMAGE_SIZE, output_mask_path)

            if ensembled_mask is not None:
                print(f"  Predicted mask saved to: {output_mask_path}")
            else:
                print(f"  Prediction failed for {image_filename}.")
        print("\nAll ensemble predictions complete.")

