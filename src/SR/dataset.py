import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random


class CustomSRDataset(Dataset):
    """Dataset for Super Resolution tasks with optional patching."""
    
    def __init__(self, root_dir, lr_folder, hr_folder, transform=None, patch_size=64, scale_factor=2, use_patches=True):
        self.lr_path = os.path.join(root_dir, lr_folder)
        self.hr_path = os.path.join(root_dir, hr_folder)
        self.transform = transform
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.use_patches = use_patches
        
        self.image_filenames = [
            f for f in os.listdir(self.lr_path) 
            if f.endswith('.png')
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        
        lr_img_path = os.path.join(self.lr_path, filename)
        hr_img_path = os.path.join(self.hr_path, filename)
        
        lr_image = Image.open(lr_img_path).convert('RGB')
        hr_image = Image.open(hr_img_path).convert('RGB')
        
        if self.use_patches:
            # Extract random patch
            il, jl, hl, wl = transforms.RandomCrop.get_params(
                lr_image, output_size=(self.patch_size, self.patch_size)
            )

            ih, jh, hh, wh = transforms.RandomCrop.get_params(
                hr_image, output_size=(self.patch_size * self.scale_factor, self.patch_size * self.scale_factor)
            )
            
            lr_image = F.crop(lr_image, il, jl, hl, wl)
            hr_image = F.crop(hr_image, ih, jh, hh, wh)
            
            # Apply random horizontal flip
            if random.random() > 0.5:
                lr_image = F.hflip(lr_image)
                hr_image = F.hflip(hr_image)
        
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        
        return lr_image, hr_image


def get_SR_datasets(base_dir='../../RELLISUR-Dataset', patch_size=64, lr=1, hr=4):
    """Create SR train, validation and test datasets."""
    
    sr_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    # Train dataset uses patches
    train_dataset = CustomSRDataset(
        root_dir=os.path.join(base_dir, "Train"),
        lr_folder=os.path.join('NLHR', f'X{lr}'),
        hr_folder=os.path.join('NLHR', f'X{hr}'),
        transform=sr_transforms,
        patch_size=patch_size,
        scale_factor=hr // lr,
        use_patches=True
    )

    # Val/Test usually process full images
    val_dataset = CustomSRDataset(
        root_dir=os.path.join(base_dir, "Val"),
        lr_folder=os.path.join('NLHR', f'X{lr}'),
        hr_folder=os.path.join('NLHR', f'X{hr}'),
        transform=sr_transforms,
        use_patches=False
    )

    test_dataset = CustomSRDataset(
        root_dir=os.path.join(base_dir, "Test"),
        lr_folder=os.path.join('NLHR', f'X{lr}'),
        hr_folder=os.path.join('NLHR', f'X{hr}'),
        transform=sr_transforms,
        use_patches=False
    )

    return train_dataset, val_dataset, test_dataset

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_sizes):
    """Create dataloaders from datasets."""
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_sizes["train"], 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_sizes["val"], 
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_sizes["test"], 
        shuffle=False
    )

    return train_loader, val_loader, test_loader