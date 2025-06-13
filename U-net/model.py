""" Full assembly of the parts to form the complete network """
from parts import DoubleConv, Down, Up, OutConv
from IPython.display import clear_output
import os
import numpy as np
from torch.autograd import Variable
from skimage import io
from skimage.transform import resize
from natsort import natsorted
from numpy import asarray
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
def get_data(path, start, end, size=(128,128)):
  """ 
  function to load image data of cells and normalize the pictures for dataloader.
  """
  images = []
  annotations = []
  image_names = []
  mask_names = []
  
  # First get names of all images to read organized
  image_names = [ f.name for f in os.scandir(path+'/image-new')][start:end]
  image_names = natsorted(image_names) # Sort so that we get correct number matching between images and annotations
  mask_names = [ f.name for f in os.scandir(path+'/mask-new')][start:end]
  mask_names = natsorted(mask_names)
  # print('sorted_names', image_names)
  # print(mask_names)
  
  # Load images
  for image_name in image_names:
    im = io.imread(os.path.join(path+'/image-new', image_name))
    im = np.array(im, np.float32)
    im = resize(im, size, anti_aliasing=True)
    pixels = asarray(im)
    pixels = pixels.astype('float32')
    mean, std = pixels.mean(), pixels.std()
    # print('Before normalization', 'Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # global standardization of pixels
    pixels = (pixels - mean) / std
    mean2, std2 = pixels.mean(), pixels.std()
    pixels = np.moveaxis(pixels, 2, -3) # move channels to last i.e: [C,W,H]
    # print('images', pixels.shape)
    images.append(pixels)

  # Load masks
  for image_name in mask_names:
        an = io.imread(os.path.join(path+'/mask-new', image_name))
        an = np.array(an, np.float32)
        an = resize(an, size, anti_aliasing=True)
        pixels = asarray(an)
        pixels = pixels.astype('float32') / 255.0 # Normalize to 0-1 if they are 0-255
        annotations.append(pixels)

  X = images
  Y = annotations
  del images
  del annotations
  print(image_names)
  return (X, Y)

def train(model, opt, loss_fn, epochs, data_loader, print_status,device):

  loss_ls = []
  epoch_ls = []
  for epoch in range(epochs):
      avg_loss = 0
      model.train() 

      b=0
      for X_batch, Y_batch in data_loader:
          
          X_batch = X_batch.to(device)
          Y_batch = Y_batch.to(device)
        
          # set parameter gradients to zero
          opt.zero_grad()
          # print(input_size)
          # forward pass
          Y_pred = model(X_batch)
          
          """
          if (epoch % 10 ==0):
            plt.figure(figsize=(5,5))
            plt.imshow(Y_pred[-1,0,:,:].detach().numpy( ))
          """
          # print('Y_pred shape', Y_pred.shape)
          # print('Y_batch shape before', Y_batch.shape)
          Y_batch = Y_batch.unsqueeze(1)
          # Y_batch[1] = Y_pred[1]
          loss = loss_fn(Y_pred, Y_batch)  # compute loss
          loss.backward()  # backward-pass to compute gradients
          opt.step()  # update weights

          # Compute average epoch loss
          avg_loss += loss.item() / len(data_loader)
          #print(b)
          b=b+1
          # print(loss)
      
      """
      if print_status:
          print(f"Loss in epoch {epoch} was {avg_loss}")
      """
      loss_ls.append(avg_loss)
      epoch_ls.append(epoch)
      print(avg_loss,epoch)
      # Delete unnecessary tensors
      # Y_batch[5:] = 0
      # show intermediate results
      model.eval()  # testing mode
      Y_hat = F.sigmoid(model(X_batch.to(device))).detach().cpu()
      # del X_batch
      # Y_hat[5:, 0] = 0
      clear_output(wait=True)

  # plt.subplots_adjust(bottom=1, top=2, hspace=0.2)
  # for k in range(4):
  #     plt.subplot(3, 4, k+1)
  #     Y_batch2 = Variable(Y_batch[k,0,:,:], requires_grad=False)
  #     plt.imshow(Y_batch2.cpu().numpy(), cmap='Greys')
  #     # plt.imshow(X_batch[k,0,:,:].cpu().numpy( ))
  #     # plt.imshow(Y_batch[k].cpu().numpy( ))
  #     plt.title('Real')
  #     plt.axis('off')

  #     plt.subplot(3, 4, k+5)
  #     plt.imshow(Y_hat[k, 0], cmap='Greys')
  #     # plt.imshow(Y_hat[k, 0])
  #     plt.title('Output')
  #     plt.axis('off')
        
  # plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
  # plt.show()
  # plt.plot(epoch_ls, loss_ls, label='traning loss')
  # plt.plot(epoch_ls, [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in loss_ls], label='training loss')
  # plt.legend()
  # plt.xlabel('Epoch'), plt.ylabel('Loss')
  # plt.show()

  return model