# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 
import collections 

# Assuming parts.py contains DoubleConv, Up, OutConv
from parts import DoubleConv, Up, OutConv


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, simclr_encoder_path=None, freeze_encoder=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # --- ENCODER: Use ResNet18 as the backbone ---
        self.resnet = models.resnet18(pretrained=False)

        if n_channels != 3:
            self.resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet.fc = nn.Identity() 

        # --- Load SimCLR weights into the ResNet encoder ---
        if simclr_encoder_path:
            print(f"Loading SimCLR ResNet18 encoder weights from {simclr_encoder_path}...")
            simclr_state_dict = torch.load(simclr_encoder_path, map_location='cpu') 

            new_resnet_state_dict = collections.OrderedDict()
            for k, v in simclr_state_dict.items():
                if 'projection_head' in k or 'mlp' in k or 'fc' in k:
                    continue
                if k.startswith('module.'):
                    k = k[len('module.'):] 

                if k.startswith('encoder.'): 
                    name = k[len('encoder.'):]
                elif k.startswith('backbone.'):
                    name = k[len('backbone.'):]
                else:
                    name = k
                
                if 'projector' in name or 'head' in name: 
                     continue

                if name in self.resnet.state_dict() and self.resnet.state_dict()[name].shape == v.shape:
                    new_resnet_state_dict[name] = v
                else:
                    print(f"Skipping parameter '{k}' (maps to '{name}'). Mismatch or not found in torchvision ResNet18. Shape: {v.shape} vs expected {self.resnet.state_dict().get(name, torch.empty(0)).shape}")

            self.resnet.load_state_dict(new_resnet_state_dict, strict=False)
            print(f"Successfully loaded {len(new_resnet_state_dict)} parameters into ResNet18 encoder.")

            if freeze_encoder:
                print("Freezing ResNet18 encoder layers...")
                for param in self.resnet.parameters():
                    param.requires_grad = False
        else:
            print("No SimCLR encoder path provided. ResNet18 encoder will be randomly initialized.")

        # --- DECODER: Adapt to ResNet18's feature map sizes ---
        factor = 2 if bilinear else 1

        # ResNet18 Output Channels (at skip connection points, starting from deepest to shallowest):
        # x4 (layer4): 512 channels (bottleneck) - 1/32 original size (16x16 for 512x512 input)
        # x3 (layer3): 256 channels - 1/16 original size (32x32)
        # x2 (layer2): 128 channels - 1/8 original size (64x64)
        # x1 (layer1): 64 channels - 1/4 original size (128x128)
        # x0 (after initial conv/bn/relu, before maxpool): 64 channels - 1/2 original size (256x256)

        # UP1: Takes upsampled x4 (512 ch, 16x16) and skip x3 (256 ch, 32x32). Concatenated: 512+256 = 768
        # Output will be 256 // factor channels at 1/16 original size (32x32)
        self.up1 = Up(512 + 256, 256 // factor, bilinear)

        # UP2: Takes upsampled output from up1 (256 // factor ch, 32x32) and skip x2 (128 ch, 64x64).
        # Concatenated: (256 // factor) + 128 channels
        # Output will be 128 // factor channels at 1/8 original size (64x64)
        self.up2 = Up((256 // factor) + 128, 128 // factor, bilinear)

        # UP3: Takes upsampled output from up2 (128 // factor ch, 64x64) and skip x1 (64 ch, 128x128).
        # Concatenated: (128 // factor) + 64 channels
        # Output will be 64 // factor channels at 1/4 original size (128x128)
        self.up3 = Up((128 // factor) + 64, 64 // factor, bilinear)

        # UP4: Takes upsampled output from up3 (64 // factor ch, 128x128) and skip x0 (64 ch, 256x256).
        # Concatenated: (64 // factor) + 64 channels
        # Output will be 64 channels at 1/2 original size (256x256)
        # Note: The output channels are not further reduced by 'factor' here, common for final stages.
        self.up4 = Up((64 // factor) + 64, 64, bilinear) 
        
        # NEW UP5: To bring resolution from 1/2 (256x256) to full (512x512)
        # This layer does NOT have a direct ResNet skip connection at the original input resolution.
        # It takes the output of up4 (64 channels, 256x256) and upsamples it.
        # The 'in_channels' to this Up module is 64 (from up4's output).
        # The 'out_channels' should ideally be consistent with the final output of the UNet's DoubleConv before OutConv (e.g., 64).
        self.up5 = Up(64, 64, bilinear) # In_channels is 64 (from up4), out_channels for DoubleConv is 64.

        self.outc = OutConv(64, n_classes) # Final output from up5 has 64 channels

    def forward(self, x):
        # ENCODER PATH (ResNet18)
        # Store intermediate features for skip connections
        # The input x is 512x512
        
        # x0 (after initial conv/bn/relu, before maxpool)
        x0_pre = self.resnet.conv1(x) # -> 256x256 spatial
        x0_pre = self.resnet.bn1(x0_pre)
        x0 = self.resnet.relu(x0_pre) # Shape: [B, 64, H/2, W/2] (e.g., 256x256)

        x1_pool = self.resnet.maxpool(x0) # -> 128x128 spatial
        x1 = self.resnet.layer1(x1_pool)  # Shape: [B, 64, H/4, W/4] (e.g., 128x128)

        x2 = self.resnet.layer2(x1)       # Shape: [B, 128, H/8, W/8] (e.g., 64x64)
        x3 = self.resnet.layer3(x2)       # Shape: [B, 256, H/16, W/16] (e.g., 32x32)
        x4 = self.resnet.layer4(x3)       # Shape: [B, 512, H/32, W/32] (e.g., 16x16) - Bottleneck

        # DECODER PATH (your Up modules with skip connections)
        # x_upsampled, x_skip_connection
        x = self.up1(x4, x3)  # Current x: (256//factor) ch, 32x32
        x = self.up2(x, x2)   # Current x: (128//factor) ch, 64x64
        x = self.up3(x, x1)   # Current x: (64//factor) ch, 128x128
        x = self.up4(x, x0)   # Current x: 64 ch, 256x256

        # NEW UP5: Upsample from 256x256 to 512x512
        # No direct ResNet skip connection at this highest resolution. Pass None for x2.
        x = self.up5(x, None) # Current x: 64 ch, 512x512

        logits = self.outc(x) # Final 1x1 convolution
        return logits
