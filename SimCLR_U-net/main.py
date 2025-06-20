import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image
from skimage import io # For .tif files
from skimage.transform import resize # Re-added for explicit resizing
from tqdm import tqdm
import natsort
import cv2 # Required for Albumentations' border_mode

# Scikit-learn for K-Fold cross-validation
from sklearn.model_selection import KFold
import collections

# Albumentations for preprocessing
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import UNet

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, n_channels, size=(512, 512), transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.n_channels = n_channels
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image_np = None
        mask_np = None

        try:
            # --- Load Image ---
            if image_path.lower().endswith(('.tif', '.tiff')):
                image_np = io.imread(image_path).astype(np.float32)
            else:
                img = Image.open(image_path).convert('RGB' if self.n_channels == 3 else 'L')
                image_np = np.array(img).astype(np.float32)

            # Ensure image is float32 and in 0-255 range for Albumentations Normalize
            if image_np.max() <= 1.0 and image_np.max() > 0:
                image_np = (image_np * 255.0)
            
            # Handle channel dimensions for image (HWC format)
            if self.n_channels == 1 and image_np.ndim == 3:
                 image_np = np.mean(image_np, axis=2, keepdims=True) # Convert RGB to grayscale, keep channel dim
            elif self.n_channels == 3 and image_np.ndim == 2:
                 image_np = np.stack([image_np, image_np, image_np], axis=-1) # Duplicate grayscale to 3 channels
                 
            if image_np.ndim == 2:
                image_np = image_np[..., np.newaxis] # Ensure HWC (e.g., H,W,1 for grayscale)

            # --- Load Mask ---
            if mask_path.lower().endswith(('.tif', '.tiff')):
                mask_np = io.imread(mask_path).astype(np.float32)
            else:
                mask_img = Image.open(mask_path).convert('L') # Load as grayscale
                mask_np = np.array(mask_img).astype(np.float32)

            # Binarize and explicitly ensure 2D (H,W) then add channel dim (H,W,1)
            mask_binary = (mask_np > 0).astype(np.float32)
            if mask_binary.ndim > 2: # Squeeze out extra dims if present (e.g., (H,W,1,1) -> (H,W))
                mask_binary = np.squeeze(mask_binary)
            if mask_binary.ndim == 2: # Ensure it's (H,W,1) for Albumentations
                mask_binary = mask_binary[..., np.newaxis]

            # --- EXPLICIT RESIZING TO TARGET SIZE ---
            # This is the crucial step to ensure consistent input dimensions for Albumentations.
            if image_np.shape[:2] != self.size:
                image_np = resize(image_np, (*self.size, self.n_channels), anti_aliasing=True, preserve_range=True).astype(np.float32)
            if mask_binary.shape[:2] != self.size:
                # For masks, the last dimension should be 1 after resizing, as it's a single channel mask.
                mask_binary = resize(mask_binary, (*self.size, 1), order=0, anti_aliasing=False, preserve_range=True).astype(np.float32)


        except Exception as e:
            print(f"Error loading {image_path} or {mask_path}: {e}")
            # Return dummy tensors of the expected size if an image/mask fails to load
            return torch.zeros(self.n_channels, self.size[0], self.size[1]), torch.zeros(1, self.size[0], self.size[1])

        # Apply transformations using Albumentations (now only normalization and ToTensor)
        if self.transform:
            augmented = self.transform(image=image_np, mask=mask_binary)
            image = augmented['image'] # Will be a PyTorch tensor (C, H, W)
            mask = augmented['mask']   # Will be a PyTorch tensor (1, H, W)
        else:
            # If no transform, convert directly to tensor and ensure C, H, W order
            image = torch.from_numpy(image_np).permute(2, 0, 1) # HWC to CHW
            mask = torch.from_numpy(mask_binary).permute(2, 0, 1) # HWC to CHW

        # --- Final robust check for mask tensor shape ---
        # Squeeze out any unwanted singleton dimensions, then ensure [1, H, W]
        mask = mask.squeeze() # Removes all singleton dimensions (e.g., 1,512,1 -> 512)
        if mask.dim() == 2: # If it became 2D (H,W), add the channel dimension back
            mask = mask.unsqueeze(0) # Result: [1, H, W]
        elif mask.dim() == 3 and mask.shape[0] != 1:
            # Should ideally not happen if previous steps are correct, but as a safeguard
            mask = mask[0, :, :].unsqueeze(0)


        return image, mask


# --- Training Function (Modified to return best validation loss) ---
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, model_save_path):
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)", leave=False)
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_pbar.set_postfix({'loss': running_loss / ((train_pbar.n + 1) * images.size(0))})

        epoch_train_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Train Loss: {epoch_train_loss:.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)", leave=False)
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                val_pbar.set_postfix({'val_loss': val_loss / ((val_pbar.n + 1) * images.size(0))})
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1} Val Loss: {epoch_val_loss:.4f}")

        # Save the model if validation loss improves
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model for this fold saved to {model_save_path} with improved validation loss: {best_loss:.4f}")

    print("Training finished for this fold!")
    return best_loss # Return the best validation loss achieved for this fold


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    ALL_IMAGE_DIR = "path_to_image_dir"
    ALL_MASK_DIR = "path_to_mask_dir"
    SIMCLR_ENCODER_PATH = "path_to_trained_simclr"

    # Directory to save the models for each fold
    MODEL_SAVE_DIR = "path_to_save_model"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Create the directory if it doesn't exist

    IMAGE_SIZE = (512, 512)
    N_CHANNELS = 3 # Adjust to 1 if your images are grayscale
    N_CLASSES = 1 # For binary segmentation (astrocyte vs. background)
    BILINEAR = True
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 50 # Number of epochs per fold. Adjust as needed.
    N_FOLDS = 5 # Number of folds for cross-validation (e.g., 5 or 10)
    RANDOM_SEED = 42 # For reproducibility of K-Fold splits

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Transforms (No random augmentations, A.Resize removed as it's now explicit in __getitem__) ---
    train_transform = A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406) if N_CHANNELS == 3 else (0.5,),
            std=(0.229, 0.224, 0.225) if N_CHANNELS == 3 else (0.5,),
            max_pixel_value=255.0, # Assumes input to Normalize is 0-255
        ),
        ToTensorV2(), # Converts to PyTorch tensor and moves channel to first dim (HWC to CHW)
    ])

    val_transform = A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406) if N_CHANNELS == 3 else (0.5,),
            std=(0.229, 0.224, 0.225) if N_CHANNELS == 3 else (0.5,),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


    # --- 1. Load all image and mask file paths ---
    all_image_paths = natsort.natsorted([os.path.join(ALL_IMAGE_DIR, f) for f in os.listdir(ALL_IMAGE_DIR) if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg'))])
    all_mask_paths = natsort.natsorted([os.path.join(ALL_MASK_DIR, f) for f in os.listdir(ALL_MASK_DIR) if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg'))])

    # Filter to ensure matching pairs
    initial_image_basenames = {os.path.basename(p) for p in all_image_paths}
    initial_mask_basenames = {os.path.basename(p) for p in all_mask_paths}
    common_basenames = sorted(list(initial_image_basenames.intersection(initial_mask_basenames)))

    all_image_paths = [os.path.join(ALL_IMAGE_DIR, bn) for bn in common_basenames]
    all_mask_paths = [os.path.join(ALL_MASK_DIR, bn) for bn in common_basenames]

    # Dynamic check for loaded image/mask pairs
    if len(all_image_paths) != len(common_basenames):
        print(f"Warning: Discrepancy in loaded image paths after matching. Expected {len(common_basenames)}, found {len(all_image_paths)}.")
    
    # Check if any data was loaded
    if len(all_image_paths) == 0:
        print("Error: No image-mask pairs found. Please check ALL_IMAGE_DIR and ALL_MASK_DIR paths and contents.")
        exit()

    print(f"Loaded {len(all_image_paths)} image-mask pairs for K-Fold Cross-Validation.")

    indices = np.arange(len(all_image_paths)) # Array of indices to split


    # --- 2. Initialize K-Fold Cross-Validation ---
    # Using KFold; consider StratifiedKFold if you have very few positive images AND
    # some folds might end up with no positive images at all. For pixel-level, KFold usually
    # works well with shuffling.
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # List to store validation losses from each fold
    fold_val_losses = []

    # --- 3. Start K-Fold Cross-Validation Loop ---
    print("\nStarting K-Fold Cross-Validation...")
    for fold, (train_indices, val_indices) in enumerate(kf.split(indices)):
        print(f"\n--- Starting Fold {fold + 1}/{N_FOLDS} ---")
        current_train_image_paths = [all_image_paths[i] for i in train_indices]
        current_train_mask_paths = [all_mask_paths[i] for i in train_indices]
        current_val_image_paths = [all_image_paths[i] for i in val_indices]
        current_val_mask_paths = [all_mask_paths[i] for i in val_indices]

        print(f"Fold {fold + 1}: Train samples: {len(current_train_image_paths)}, Val samples: {len(current_val_image_paths)}")

        train_dataset = SegmentationDataset(current_train_image_paths, current_train_mask_paths, N_CHANNELS, size=IMAGE_SIZE, transform=train_transform)
        val_dataset = SegmentationDataset(current_val_image_paths, current_val_mask_paths, N_CHANNELS, size=IMAGE_SIZE, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        print("Instantiating UNet model with SimCLR ResNet18 encoder for current fold...")
        model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=BILINEAR,
                     simclr_encoder_path=SIMCLR_ENCODER_PATH,
                     freeze_encoder=False).to(device)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        pos_weight_value = torch.tensor(10.0).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)

        fold_model_save_path = os.path.join(MODEL_SAVE_DIR, f"unet_resnet_simclr_finetuned_fold_{fold+1}.pt")

        best_val_loss_this_fold = train_model(model, train_loader, val_loader, optimizer, criterion, EPOCHS, device, fold_model_save_path)
        fold_val_losses.append(best_val_loss_this_fold)

    # --- After K-Fold Cross-Validation Loop ---
    print("\n--- K-Fold Cross-Validation Completed ---")
    print(f"Validation Losses for each fold: {fold_val_losses}")
    print(f"Average Validation Loss across {N_FOLDS} folds: {np.mean(fold_val_losses):.4f}")

    print("All fold models saved in:", MODEL_SAVE_DIR)
    print("\nTraining script finished.")
