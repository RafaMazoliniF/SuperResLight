import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSR(nn.Module):
    def __init__(self, scale_factor=2):
        super(SimpleSR, self).__init__()
        
        self.scale_factor = scale_factor
        
        # 1. Feature Extraction (Sem Pooling)
        # O padding="same" garante que a dimensão HxW (625x625) seja mantida
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding="same")
        
        # 2. Reconstrução de Imagem
        # A última camada convolucional deve ter C * (scale_factor^2) canais de saída.
        # 16 canais de entrada -> 3 * (2^2) = 12 canais de saída.
        self.conv_reconstruction = nn.Conv2d(
            in_channels=16, 
            out_channels=3 * (self.scale_factor ** 2), # 3 * 4 = 12
            kernel_size=3, 
            padding="same"
        )
        
        # 3. Módulo de Upsampling
        self.upsampler = nn.PixelShuffle(self.scale_factor) # Aumenta a resolução em 2x
        
        # 4. Camada final para ajuste de cor (opcional)
        self.final_conv = nn.Conv2d(3, 3, kernel_size=3, padding="same")

    def forward(self, x):
        # x de entrada: (B, 3, 625, 625)
        
        # 1. Feature Extraction (H x W permanece 625 x 625)
        x = F.relu(self.conv1(x)) # Saída: (B, 8, 625, 625)
        x = F.relu(self.conv2(x)) # Saída: (B, 16, 625, 625)
        
        # 2. Preparação para o Upsampling
        x = F.relu(self.conv_reconstruction(x)) # Saída: (B, 12, 625, 625)
        
        # 3. Upsampling (PixelShuffle)
        # Transforma (B, 12, 625, 625) em (B, 3, 1250, 1250)
        x = self.upsampler(x)
        
        # 4. Ajuste final
        x = self.final_conv(x)
        
        return x