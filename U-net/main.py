import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchsummary import summary
import torch.optim as optim
from torch.autograd import Variable
from time import time
from PIL import Image
from skimage.io import imread
from skimage.color import rgb2gray
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import torchvision.datasets as datasets
import skimage.io as io
from skimage.transform import resize
import natsort
from natsort import natsorted
from numpy import asarray
from tqdm.notebook import tqdm

from model import get_data
from model import UNet
from model import train
plt.ion()
def focal_loss(y_pred, y_real, gamma=2):
    y_pred = torch.clamp(F.sigmoid(y_pred), 1e-8, 1-1e-8)
    # gamma = 2
    return -torch.mean(((1-y_pred)**gamma)*y_real*torch.log(y_pred) + (1-y_real)*torch.log(1-y_pred))

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainpath ="/home/htic/Desktop/raisa/astrocytes/astrocyte-detection/training-media"

    X, Y = get_data(path=trainpath, start=0, end=80, size=(512,512))

    nr = 1
    X_ex = np.moveaxis(X[nr], 0, -1)
    Y_ex = Y[nr]
    print(X_ex.shape)
    print(Y_ex.shape)
    plt.imshow(X_ex)
    plt.imshow(Y[nr])

    plt.subplot(1, 2, 1)
    plt.imshow(X_ex)
    plt.axis('off')
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(Y[nr], cmap='gray')
    plt.axis('off')
    plt.title('Mask')

    plt.tight_layout()
    plt.show()
    plt.close()



    batch_size = 1
    dataloader2 = DataLoader(list(zip(X, Y)), batch_size=batch_size, shuffle=True)

    # Predefine input size 
    # x, y = next(iter(dataloader2))
    # input_size = x.shape[1]
    # del x, y
    # print(input_size)
    n_channels = 3
    n_classes = 2
    # Next creat an instance of the UNet model
    modelUnet = UNet(n_channels=3, n_classes=1).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    # Now define the optimizer
    optimizerUnet = optim.Adam(modelUnet.parameters(), lr = 0.001, weight_decay=1e-2)

    # And finally lets train the model
    # model_out = train(modelUnet, optimizerUnet, focal_loss, 100, dataloader2, print_status=True)
    model_out = train(modelUnet, optimizerUnet, criterion, 80, dataloader2, print_status=True,device=device)


    model_save_dir = "/home/htic/Desktop/raisa/astrocytes/astrocyte-detection/U-net"
    model_filename = "unet_model_weights.pt" # Using .pt is a good choice
    model_save_path = os.path.join(model_save_dir, model_filename)
    torch.save(model_out.state_dict(), model_save_path)
    print(f"Model saved successfully to {model_save_path}")
    # X_batch, Y_batch = next(iter(dataloader2))
    # # X = X[0].to(device)
    # # X = Variable(torch.from_numpy(X[0]))
    # model_out.eval()  # testing mode
    # Y_hat = F.sigmoid(model_out(X_batch.to(device))).detach().cpu()
    # Y_batch[5:] = 0
    # Y_hat[5:] = 0

# def view_mask(targets, output, n=2, cmap='gray'):
#     # Determine the actual number of samples to plot,
#     # which is the minimum of desired 'n' and available batch size.
#     num_samples_to_plot = min(n, targets.shape[0])

#     if num_samples_to_plot == 0:
#         print("No samples available in the batch to display.")
#         return

#     # Create a figure with dynamic subplot layout: 2 rows (target, output) and num_samples_to_plot columns
#     figure, axes = plt.subplots(2, num_samples_to_plot, figsize=(4 * num_samples_to_plot, 8))
    
#     # If only one sample is plotted, axes might not be a 2D array, so flatten it
#     if num_samples_to_plot == 1:
#         axes = np.array([[axes[0]], [axes[1]]]) # Make it a 2x1 array for consistent indexing

#     figure.suptitle(f'Visualizing {num_samples_to_plot} Sample(s)', fontsize=16)

#     for i in range(num_samples_to_plot):
#         # --- Plot Target (True) Masks ---
#         target_im = targets[i].cpu().detach().numpy().squeeze() # Use .squeeze() to remove single-dim channels
#         target_im[target_im > 0.5] = 1 # Ensure binary values
#         target_im[target_im < 0.5] = 0
        
#         axes[0, i].imshow(target_im, cmap=cmap)
#         axes[0, i].set_title(f'Real Mask {i+1}')
#         axes[0, i].axis('off')

#         # --- Plot Output (Predicted) Masks ---
#         # output[i] is already a 1-channel image (e.g., [1, H, W]), so just squeeze
#         output_im = output[i].cpu().detach().numpy().squeeze()
#         output_im[output_im > 0.5] = 1 # Apply threshold for binary output
#         output_im[output_im < 0.5] = 0
        
#         axes[1, i].imshow(output_im, cmap=cmap) # Use the specified cmap
#         axes[1, i].set_title(f'Predicted {i+1}')
#         axes[1, i].axis('off')
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
#     plt.show()

# # Example usage (ensure Y_batch and Y_hat are available from your training loop)
# # This will now safely plot up to 2 samples, or fewer if the batch size is smaller.
# view_mask(targets=Y_batch, output=Y_hat, n=2)

# def save_predicted_masks(model, dataloader, save_dir, threshold=0.5, device='cuda'):
#     os.makedirs(save_dir, exist_ok=True)
#     model.eval()

#     idx = 0  # for naming files
#     with torch.no_grad():
#         for inputs, _ in tqdm(dataloader, desc="Saving predicted masks"):
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             outputs = torch.sigmoid(outputs)  # ensure range 0–1

#             binary_masks = (outputs > threshold).float().cpu()

#             for mask in binary_masks:
#                 # If mask is [1, H, W], convert to [H, W]
#                 mask_img = mask.squeeze().numpy() * 255  # Convert 0/1 to 0/255
#                 mask_img = mask_img.astype(np.uint8)

#                 out_path = os.path.join(save_dir, f"pred_mask_{idx:03d}.png")
#                 Image.fromarray(mask_img).save(out_path)
#                 idx += 1
                
# save_path = "/home/htic/Desktop/raisa/astrocytes/astrocyte-detection/training-media/test-media/mask"
# save_predicted_masks(model_out, dataloader2, save_path, device=device)