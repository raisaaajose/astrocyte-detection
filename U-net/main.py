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

    trainpath ="path_to_training_data"

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

    n_channels = 3
    n_classes = 2
    modelUnet = UNet(n_channels=3, n_classes=1).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizerUnet = optim.Adam(modelUnet.parameters(), lr = 0.001, weight_decay=1e-2)

    # model_out = train(modelUnet, optimizerUnet, focal_loss, 100, dataloader2, print_status=True)
    model_out = train(modelUnet, optimizerUnet, criterion, 80, dataloader2, print_status=True,device=device)


    model_save_dir = "path_to_save_model"
    model_filename = "unet_model_weights.pt" # Using .pt is a good choice
    model_save_path = os.path.join(model_save_dir, model_filename)
    torch.save(model_out.state_dict(), model_save_path)
    print(f"Model saved successfully to {model_save_path}")
