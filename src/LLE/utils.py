import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def training_loop(model, criterion, optimizer, n_epochs, train_dl, device):
    for epoch in range(n_epochs):
        model.train() 
        running_loss = 0.0
        
        for lr_images, hr_images in train_dl:
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            optimizer.zero_grad()

            
            sr_images = model(lr_images)
            loss = criterion(sr_images, hr_images)    

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * lr_images.size(0)

        epoch_loss = running_loss / len(train_dl.dataset)
        
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.6f}")

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
            img = np.clip(img, 0, 1)
            return img

        fig, axes = plt.subplots(num_to_show, 3, figsize=(15, 5 * num_to_show))
        
        titles = ['Low Resolution (LR)', 'Super Resolution (SR)', 'Ground Truth (HR)']
        
        if num_to_show == 1:
            axes = np.array([axes])

        for i in range(num_to_show):
            axes[i, 0].imshow(tensor_to_img(lr_images[i]))
            axes[i, 0].set_title(titles[0])
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(tensor_to_img(sr_images[i]))
            axes[i, 1].set_title(titles[1])
            axes[i, 1].axis('off')

            axes[i, 2].imshow(tensor_to_img(hr_images[i]))
            axes[i, 2].set_title(titles[2])
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.show()

class PSNRLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super(PSNRLoss, self).__init__()
        self.data_range = data_range

    def forward(self, pred, target):

        mse = torch.mean((pred - target) ** 2)

        if mse.item() == 0:
            return torch.tensor(-100.0, device=pred.device)

        psnr = 10 * torch.log10((self.data_range ** 2) / mse)
        return -psnr