import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ssim

# --- Custom Losses & Metrics ---

class SSIMLoss(nn.Module):
    def __init__(self, alpha=0.8, data_range=1.0):
        super(SSIMLoss, self).__init__()
        self.data_range = data_range
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        ssim_val = ssim(pred, target, data_range=self.data_range, size_average=True)
        l1_loss = self.l1(pred, target)
        # Loss = weighted sum of L1 and (1 - SSIM)
        return (self.alpha * l1_loss) + ((1 - self.alpha) * (1 - ssim_val))


class PSNRMetric(nn.Module):
    """Calculates PSNR value (not negative loss)."""
    def __init__(self, data_range=1.0):
        super(PSNRMetric, self).__init__()
        self.data_range = data_range

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        if mse.item() == 0:
            return torch.tensor(100.0, device=pred.device)
        
        return 10 * torch.log10((self.data_range ** 2) / mse)


# --- Visualization Utils ---

def visualize_results(model, dataloader, device, num_samples=1):
    model.eval()
    with torch.no_grad():
        lr_images, hr_images = next(iter(dataloader))
        
        current_batch_size = lr_images.size(0)
        num_to_show = min(num_samples, current_batch_size)
        
        lr_images = lr_images.to(device)[:num_to_show]
        hr_images = hr_images.to(device)[:num_to_show]
        sr_images = model(lr_images)

        def tensor_to_img(tensor):
            img = tensor.cpu().numpy().transpose(1, 2, 0)
            return np.clip(img, 0, 1)

        fig, axes = plt.subplots(num_to_show, 3, figsize=(15, 5 * num_to_show))
        titles = ['Low Resolution', 'Super Resolution', 'Ground Truth']
        
        if num_to_show == 1:
            axes = np.array([axes])

        for i in range(num_to_show):
            imgs = [lr_images[i], sr_images[i], hr_images[i]]
            for j, img in enumerate(imgs):
                axes[i, j].imshow(tensor_to_img(img))
                axes[i, j].set_title(titles[j])
                axes[i, j].axis('off')

        plt.tight_layout()
        plt.show()


# --- Training Engine ---

def training_loop(model, criterion, optimizer, n_epochs, train_dl, val_dl, device):
    # Instantiate metric once
    psnr_metric = PSNRMetric().to(device)
    
    for epoch in range(n_epochs):
        # Train
        model.train()
        running_loss = 0.0
        
        for lr_images, hr_images in train_dl:
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            optimizer.zero_grad()
            sr_images = model(lr_images)
            loss = criterion(sr_images, hr_images)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * lr_images.size(0)

        epoch_train_loss = running_loss / len(train_dl.dataset)

        # Validation
        model.eval()
        running_val_loss = 0.0
        running_psnr = 0.0
        
        with torch.no_grad():
            for lr_images, hr_images in val_dl:
                lr_images, hr_images = lr_images.to(device), hr_images.to(device)

                sr_images = model(lr_images)
                
                loss = criterion(sr_images, hr_images)
                psnr = psnr_metric(sr_images, hr_images)
                
                running_val_loss += loss.item() * lr_images.size(0)
                running_psnr += psnr.item() * lr_images.size(0)

        epoch_val_loss = running_val_loss / len(val_dl.dataset)
        epoch_psnr = running_psnr / len(val_dl.dataset)
        
        print(f"Epoch [{epoch+1}/{n_epochs}] | "
              f"Train Loss: {epoch_train_loss:.6f} | "
              f"Val Loss: {epoch_val_loss:.6f} | "
              f"PSNR: {epoch_psnr:.2f} dB")