import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random


class CustomSRDataset(Dataset):
    """Dataset for Super Resolution tasks with optional patching."""
    
    def __init__(self, root_dir, lr_folder, hr_folder, transform=None, patch_size=128, use_patches=True):
        self.lr_path = os.path.join(root_dir, lr_folder)
        self.hr_path = os.path.join(root_dir, hr_folder)
        self.transform = transform
        self.patch_size = patch_size
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
            i, j, h, w = transforms.RandomCrop.get_params(
                hr_image, output_size=(self.patch_size, self.patch_size)
            )
            
            lr_image = F.crop(lr_image, i, j, h, w)
            hr_image = F.crop(hr_image, i, j, h, w)
            
            # Apply random horizontal flip
            if random.random() > 0.5:
                lr_image = F.hflip(lr_image)
                hr_image = F.hflip(hr_image)
        
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        
        return lr_image, hr_image


def get_SR_datasets(base_dir='../../RELLISUR-Dataset', patch_size=128):
    """Create SR train, validation and test datasets."""
    
    sr_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    # Train dataset uses patches
    train_dataset = CustomSRDataset(
        root_dir=os.path.join(base_dir, "Train"),
        lr_folder='LLLR',
        hr_folder=os.path.join('NLHR-Duplicates', 'X1'),
        transform=sr_transforms,
        patch_size=patch_size,
        use_patches=True
    )

    # Val/Test usually process full images
    val_dataset = CustomSRDataset(
        root_dir=os.path.join(base_dir, "Val"),
        lr_folder='LLLR',
        hr_folder=os.path.join('NLHR-Duplicates', 'X1'),
        transform=sr_transforms,
        use_patches=False
    )

    test_dataset = CustomSRDataset(
        root_dir=os.path.join(base_dir, "Test"),
        lr_folder='LLLR',
        hr_folder=os.path.join('NLHR-Duplicates', 'X1'),
        transform=sr_transforms,
        use_patches=False
    )

    return train_dataset, val_dataset, test_dataset


class CustomLLEDataset(Dataset):
    """Dataset for Low-Light Enhancement tasks with optional patching."""
    
    def __init__(self, root_dir, low_light_folder, normal_light_folder, transforms=None, 
                 patch_size=128, use_patches=True):
        self.low_light_path = os.path.join(root_dir, low_light_folder)
        self.normal_light_path = os.path.join(root_dir, normal_light_folder)
        self.transforms = transforms
        self.patch_size = patch_size
        self.use_patches = use_patches
        
        self.image_filenames = [
            f for f in os.listdir(self.low_light_path) 
            if f.endswith('.png')
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        
        low_light_img_path = os.path.join(self.low_light_path, filename)
        normal_light_img_path = os.path.join(self.normal_light_path, filename)
        
        low_light_image = Image.open(low_light_img_path).convert('RGB')
        normal_light_image = Image.open(normal_light_img_path).convert('RGB')
        
        if self.use_patches:
            # Extract random patch from both images
            i, j, h, w = transforms.RandomCrop.get_params(
                normal_light_image, output_size=(self.patch_size, self.patch_size)
            )
            
            low_light_patch = F.crop(low_light_image, i, j, h, w)
            normal_light_patch = F.crop(normal_light_image, i, j, h, w)
            
            # Apply random horizontal flip
            if random.random() > 0.5:
                low_light_patch = F.hflip(low_light_patch)
                normal_light_patch = F.hflip(normal_light_patch)
        else:
            low_light_patch = low_light_image
            normal_light_patch = normal_light_image
        
        # Apply transforms
        if self.transforms:
            low_light_patch = self.transforms[0](low_light_patch)
            normal_light_patch = self.transforms[1](normal_light_patch)
            
        return low_light_patch, normal_light_patch


def get_LLE_datasets(base_dir='../../RELLISUR-Dataset', patch_size=64):
    """Create LLE train, validation and test datasets with padding."""

    # Training transforms
    input_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    label_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = CustomLLEDataset(
        root_dir=os.path.join(base_dir, "Train"),
        lr_folder='LLLR',
        hr_folder=os.path.join('NLHR-Duplicates', 'X1'),
        transforms=[input_transforms, label_transforms],
        patch_size=patch_size,
        use_patches=True
    )

    val_dataset = CustomLLEDataset(
        root_dir=os.path.join(base_dir, "Val"),
        lr_folder='LLLR',
        hr_folder=os.path.join('NLHR-Duplicates', 'X1'),
        transforms=[input_transforms, label_transforms],
        patch_size=patch_size,
        use_patches=False
    )

    test_dataset = CustomLLEDataset(
        root_dir=os.path.join(base_dir, "Test"),
        lr_folder='LLLR',
        hr_folder=os.path.join('NLHR-Duplicates', 'X1'),
        transforms=[input_transforms, label_transforms],
        patch_size=patch_size,
        use_patches=False
    )

    print(f"Training samples: {len(train_dataset)}")

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