import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
import natsort

from torch.utils.data import DataLoader, Dataset
from model import get_data

from model import UNet


def load_and_preprocess_image(image_path, size=(512, 512), n_channels=3):
    """
    Loads an image, resizes, normalizes, and prepares it for the UNet model.
    Assumes image needs to be float32, normalized, and have shape [C, H, W].
    """
    img = io.imread(image_path)
    img = np.array(img, np.float32)

    # Convert to grayscale if n_channels is 1 and image is RGB
    if n_channels == 1 and img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, 0] # Take one channel, or use skimage.color.rgb2gray if desired
    
    # Handle potentially grayscale input that's missing channel dimension
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1) # Add channel dimension at the end (H, W, C)

    img = resize(img, size, anti_aliasing=True)

    # Normalize pixel values
    pixels = img.astype('float32')
    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / std

    # Move channel to the first dimension for PyTorch: [C, H, W]
    pixels = np.moveaxis(pixels, -1, 0) # Assumes input is [H, W, C]

    # Convert to PyTorch tensor and add batch dimension
    tensor_img = torch.from_numpy(pixels)
    return tensor_img


def save_predicted_masks(model, dataloader, save_dir, threshold=0.20, device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    idx = 0
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Saving predicted masks"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)

            binary_masks = (outputs > threshold).float().cpu()

            for mask in binary_masks:
                mask_img = mask.squeeze().numpy() * 255
                mask_img = mask_img.astype(np.uint8)

                out_path = os.path.join(save_dir, f"pred_mask_{idx:03d}.png")
                Image.fromarray(mask_img).save(out_path)
                idx += 1

if __name__ == "__main__":
    # --- Configuration ---
    # Path to your trained UNet model weights
    model_weights_path = "/home/htic/Desktop/raisa/astrocytes/astrocyte-detection/U-net/unet_model_weights.pt"
    
    # Path to the directory containing images you want to predict on
    # You can set this to your training-media/image-new for testing,
    # or a completely new test image directory.
    test_image_dir = "/home/htic/Desktop/raisa/astrocytes/astrocyte-detection/test-media/final-test"
    
    # Directory to save the predicted masks
    output_masks_dir = "/home/htic/Desktop/raisa/astrocytes/astrocyte-detection/test-media/final-test-mask"
    
    # Image loading and model parameters - MUST match training setup
    image_size = (512, 512)
    n_channels = 3 # Input channels of your UNet
    n_classes = 1  # Output classes of your UNet (1 for binary segmentation)
    bilinear = True # UNet's bilinear upsampling setting

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Instantiate the UNet model ---
    # Crucially, ensure the UNet architecture matches the one that was trained.
    # If your UNet uses a ResNet18 backbone, you would instantiate it like this:
    # model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear, simclr_weights_path=None).to(device)
    # The `simclr_weights_path=None` ensures it just builds the architecture, not trying to load SimCLR again.
    # Otherwise, for your original UNet:
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear).to(device)


    # --- 2. Load the saved model weights ---
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        print(f"Model weights loaded successfully from {model_weights_path}")
    else:
        print(f"Error: Model weights file not found at {model_weights_path}")
        print("Please check the 'model_weights_path' variable.")
        exit()

    # --- 3. Set model to evaluation mode ---
    model.eval() # Essential for inference

    # --- 4. Prepare data for prediction ---
    # We'll use your get_data function to load images (and dummy masks if needed)
    # This assumes get_data can handle just images, or you can create a simple
    # custom dataset/dataloader for prediction if your `test_image_dir` is different.

    # Option A: Load images using a simplified get_data for prediction (no masks needed)
    # For now, let's use a simpler loading for single images or a list of image paths
    
    # Get all image paths from the test directory
    image_filenames = natsort.natsorted(os.listdir(test_image_dir))
    image_paths = [os.path.join(test_image_dir, f) for f in image_filenames if f.endswith(('.tif', '.png', '.jpg', '.jpeg'))]

    if not image_paths:
        print(f"No images found in {test_image_dir}. Exiting.")
        exit()

    print(f"\nFound {len(image_paths)} images for prediction.")

    # Create a dummy list for masks, as save_predicted_masks expects a dataloader of (inputs, targets)
    # We'll just pass None for targets since we don't have them for prediction.
    # This might require a slight modification to `save_predicted_masks` or how you create the DataLoader.
    
    # A cleaner approach for prediction-only: Create a custom Dataset for just images.
    class PredictionDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, size, n_channels):
            self.image_paths = image_paths
            self.size = size
            self.n_channels = n_channels

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            # Use your load_and_preprocess_image function
            image_tensor = load_and_preprocess_image(img_path, self.size, self.n_channels)
            return image_tensor, torch.tensor(0) # Return dummy target (0) to satisfy DataLoader's expectation

    prediction_dataset = PredictionDataset(image_paths, image_size, n_channels)
    # Use a batch size that fits your GPU memory for prediction (can be larger than training batch size)
    prediction_dataloader = DataLoader(prediction_dataset, batch_size=4, shuffle=False) # No need to shuffle for prediction

    # --- 5. Run inference and save all predicted masks ---
    print(f"\nStarting prediction and saving masks to: {output_masks_dir}")
    save_predicted_masks(model, prediction_dataloader, output_masks_dir, device=device)
    print("All predicted masks saved successfully.")

    # --- 6. Visualize a sample prediction (optional, pick one image) ---
    print("\nVisualizing a sample prediction (first image):")
    sample_image_path = image_paths[0]
    sample_input_tensor = load_and_preprocess_image(sample_image_path, size=image_size, n_channels=n_channels)
    sample_input_tensor=sample_input_tensor.unsqueeze(0)  # Add batch dimension
    model.eval() # Ensure model is still in eval mode
    with torch.no_grad():
        sample_output = model(sample_input_tensor.to(device))
        sample_predicted_mask_probs = torch.sigmoid(sample_output)
        sample_predicted_mask_binary = (sample_predicted_mask_probs > 0.5).float()

    # view_mask(targets=torch.zeros_like(sample_predicted_mask_binary), # Dummy target for visualization
    #           output=sample_predicted_mask_binary.cpu(), 
    #           n=1, cmap='gray')

    print("\nPrediction script finished.")