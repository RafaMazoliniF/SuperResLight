import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLLE(nn.Module):
    def __init__(self):
        super(SimpleLLE, self).__init__()
    
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(16, 3, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(3, 3, kernel_size=3, padding="same")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        
        return x

class SimpleLLE2(nn.Module):
    def __init__(self):
        super(SimpleLLE2, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 128, kernel_size=3, padding="same")
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 16, kernel_size=3, padding="same")
        self.bn5 = nn.BatchNorm2d(16)

        self.conv6 = nn.Conv2d(16, 3, kernel_size=3, padding="same")
        self.bn6 = nn.BatchNorm2d(3)

        self.conv7 = nn.Conv2d(3, 3, kernel_size=3, padding="same")

    def forward(self, x):
        indentity = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.conv7(x)
        
        x += indentity

        return x

import torchvision.models as models
class ResNetLLE_Light(nn.Module):
    def __init__(self):
        super(ResNetLLE_Light, self).__init__()
        
        # --- ENCODER (Mais Raso) ---
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Paramos em [:-3] (que inclui até a layer3), em vez de [:-2] (layer4)
        # O output aqui é [B, 256, H/16, W/16]
        modules = list(resnet18.children())[:-3] 
        self.encoder = nn.Sequential(*modules)

        for param in self.encoder.parameters():
            param.requires_grad = False

        # --- DECODER (Mais Leve) ---
        # 4 camadas de ConvTranspose2d para reverter o downsampling de 4x
        
        # 256x(H/16) -> 128x(H/8)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)

        # 128x(H/8) -> 64x(H/4)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        # 64x(H/4) -> 32x(H/2)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        
        # 32x(H/2) -> 16x(H)
        self.upconv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(16)

        # Camada final para voltar a 3 canais (RGB)
        # 16xH -> 3xH
        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, padding="same")

    def forward(self, x):
        # Guardar a imagem original para a conexão residual
        identity = x

        # --- Encoder ---
        x = self.encoder(x) # [B, 256, H/16, W/16]

        # --- Decoder ---
        x = F.relu(self.bn1(self.upconv1(x))) # [B, 128, H/8, W/8]
        x = F.relu(self.bn2(self.upconv2(x))) # [B, 64, H/4, W/4]
        x = F.relu(self.bn3(self.upconv3(x))) # [B, 32, H/2, W/2]
        x = F.relu(self.bn4(self.upconv4(x))) # [B, 16, H, W]

        x = self.final_conv(x)               # [B, 3, H, W]

        # --- Conexão Residual Global ---
        output = identity + x
        
        return output