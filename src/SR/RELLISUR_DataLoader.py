import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms

class CustomSRDataset(Dataset):
    def __init__(self, root_dir, lr_folder, hr_folder, transform=None):
        """
        Inicializa o Dataset.

        Args:
            root_dir (str): Diretório base (ex: 'RELLISUR-Dataset/Train').
            lr_folder (str): Subdiretório das imagens de Baixa Resolução (ex: 'NLHR/X2').
            hr_folder (str): Subdiretório das imagens de Alta Resolução (ex: 'NLHR').
            transform (callable, optional): Transformações opcionais.
        """
        self.lr_path = os.path.join(root_dir, lr_folder)
        self.hr_path = os.path.join(root_dir, hr_folder)
        self.transform = transform
        
        self.image_filenames = os.listdir(self.lr_path)
        self.image_filenames = [f for f in self.image_filenames if f.endswith(('.png'))]

    def __len__(self):
        """Retorna o número total de pares de amostras."""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """Carrega e retorna o par (LR, HR)."""
        # Nome do arquivo é o mesmo para LR e HR (ex: 'image001.png')
        filename = self.image_filenames[idx]
        
        lr_img_path = os.path.join(self.lr_path, filename)
        hr_img_path = os.path.join(self.hr_path, filename)
        
        lr_image = Image.open(lr_img_path).convert('RGB')
        hr_image = Image.open(hr_img_path).convert('RGB')
        
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        
        return lr_image, hr_image

def get_SR_datasets(lr, hr):
    base_dir = '../../RELLISUR-Dataset'

    sr_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = CustomSRDataset(
        root_dir=os.path.join(base_dir, "Train"),
        lr_folder=os.path.join('NLHR', f"X{lr}"),
        hr_folder=os.path.join('NLHR', f"X{hr}"),
        transform=sr_transforms
    )

    val_dataset = CustomSRDataset(
        root_dir=os.path.join(base_dir, "Val"),
        lr_folder=os.path.join('NLHR', f"X{lr}"),
        hr_folder=os.path.join('NLHR', f"X{hr}"),
        transform=sr_transforms
    )

    test_dataset = CustomSRDataset(
        root_dir=os.path.join(base_dir, "Test"),
        lr_folder=os.path.join('NLHR', f"X{lr}"),
        hr_folder=os.path.join('NLHR', f"X{hr}"),
        transform=sr_transforms
    )

    return train_dataset, val_dataset, test_dataset

def get_SR_dataloaders(train, val, test, batch_size):
    batch_size = batch_size

    train_dl = DataLoader(train, batch_size, shuffle=True)
    val_dl = DataLoader(val, batch_size, shuffle=False)
    test_dl = DataLoader(test, batch_size,shuffle=False)

    return train_dl, val_dl, test_dl