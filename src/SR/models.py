import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSR(nn.Module):
    def __init__(self, scale_factor=2, in_channels=3):
        super(SimpleSR, self).__init__()
        
        # Feature Extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
        )
        
        # Reconstruction & Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(64, in_channels * (scale_factor ** 2), kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(scale_factor)
        )
        
        # Final Adjustment
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding='same')

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        x = self.final_conv(x)
        return x

class ResidualSR(nn.Module):
    def __init__(self, scale_factor=2, in_channels=3, num_channels=64):
        super(ResidualSR, self).__init__()

        self.scale_factor = scale_factor
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(num_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding='same'),
            nn.PixelShuffle(scale_factor) 
        )

    def forward(self, x):
        feat = self.features(x)
        out = self.upsample(feat)
        
        base = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        
        return out + base