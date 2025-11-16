import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLLE(nn.Module):
    def __init__(self):
        super(SimpleLLE, self).__init__()
    
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding="same")
        self.conv_reconstruction = nn.Conv2d(16, 3, kernel_size=3, padding="same")
        self.final_conv = nn.Conv2d(3, 3, kernel_size=3, padding="same")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv_reconstruction(x))
        x = self.final_conv(x)
        
        return x