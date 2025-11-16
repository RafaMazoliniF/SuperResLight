import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms

class CustomSRDataset(Dataset):
    def __init__(self, root_dir, ll_folder, gt_folder, transform=None):
        """
        Inicializa o Dataset.

        Args:
            root_dir (str): Diretório base (ex: 'RELLISUR-Dataset/Train').
            ll_folder (str): Subdiretório das imagens de Baixa Resolução (ex: 'NLHR/X2').
            gt_folder (str): Subdiretório das imagens de Alta Resolução (ex: 'NLHR').
            transform (callable, optional): Transformações opcionais.
        """
        self.ll_path = os.path.join(root_dir, ll_folder)
        self.gt_path = os.path.join(root_dir, gt_folder)
        self.transform = transform
        
        self.image_filenames = os.listdir(self.ll_path)
        self.image_filenames = [f for f in self.image_filenames if f.endswith(('.png'))]

    def __len__(self):
        """Retorna o número total de pares de amostras."""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """Carrega e retorna o par (LR, HR)."""
        # Nome do arquivo é o mesmo para LR e HR (ex: 'image001.png')
        filename = self.image_filenames[idx]
        
        ll_img_path = os.path.join(self.ll_path, filename)
        gt_img_path = os.path.join(self.gt_path, filename)
        
        ll_image = Image.open(ll_img_path).convert('RGB')
        gt_image = Image.open(gt_img_path).convert('RGB')
        
        if self.transform:
            ll_image = self.transform(ll_image)
            gt_image = self.transform(gt_image)
        
        return ll_image, gt_image

def get_SR_datasets():
    base_dir = '../../RELLISUR-Dataset'

    sr_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = CustomSRDataset(
        root_dir=os.path.join(base_dir, "Train"),
        ll_folder=os.path.join('LLLR'),
        gt_folder=os.path.join('NLHR-Duplicates', f"X1"),
        transform=sr_transforms
    )

    val_dataset = CustomSRDataset(
        root_dir=os.path.join(base_dir, "Val"),
        ll_folder=os.path.join('LLLR'),
        gt_folder=os.path.join('NLHR-Duplicates', f"X1"),
        transform=sr_transforms
    )

    test_dataset = CustomSRDataset(
        root_dir=os.path.join(base_dir, "Test"),
        ll_folder=os.path.join('LLLR'),
        gt_folder=os.path.join('NLHR-Duplicates', f"X1"),
        transform=sr_transforms
    )

    return train_dataset, val_dataset, test_dataset

# ======================================================================================================== #

class CustomLLEDataset(Dataset):
    def __init__(self, root_dir, ll_folder, gt_folder, transforms=None):
        """
        Inicializa o Dataset.

        Args:
            root_dir (str): Diretório base (ex: 'RELLISUR-Dataset/Train').
            ll_folder (str): Subdiretório das imagens de Baixa Resolução (ex: 'NLHR/X2').
            gt_folder (str): Subdiretório das imagens de Alta Resolução (ex: 'NLHR').
            transform (callable, optional): Transformações opcionais.
        """
        self.ll_path = os.path.join(root_dir, ll_folder)
        self.gt_path = os.path.join(root_dir, gt_folder)
        self.transforms = transforms
        
        self.image_filenames = os.listdir(self.ll_path)
        self.image_filenames = [f for f in self.image_filenames if f.endswith(('.png'))]

    def __len__(self):
        """Retorna o número total de pares de amostras."""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """Carrega e retorna o par (LR, HR)."""
        # Nome do arquivo é o mesmo para LR e HR (ex: 'image001.png')
        filename = self.image_filenames[idx]
        
        ll_img_path = os.path.join(self.ll_path, filename)
        gt_img_path = os.path.join(self.gt_path, filename)
        
        ll_image = Image.open(ll_img_path).convert('RGB')
        gt_image = Image.open(gt_img_path).convert('RGB')
        
        if self.transforms:
            ll_image = self.transforms[0](ll_image)
            gt_image = self.transforms[1](gt_image)
        
        return ll_image, gt_image


def get_LLE_datasets_with_hist():
    base_dir = '../../RELLISUR-Dataset'
    
    # Transformação para a imagem de entrada (LLLR)
    input_transforms = transforms.Compose([
        transforms.Lambda(F.adjust_gamma(equalized_tensor, 0.8)),
        transforms.Lambda(F.adjust_brightness(equalized_tensor, 1.2)),
        transforms.Lambda(F.adjust_contrast(equalized_tensor, 1.2)),
        transforms.ToTensor()
    ])

    # Transformação para a imagem de label (NLHR)
    label_transforms = transforms.Compose([
        transforms.ToTensor() # Apenas converte para tensor
    ])

    train_dataset = CustomLLEDataset(
        root_dir=os.path.join(base_dir, "Train"),
        ll_folder=os.path.join('LLLR'),
        gt_folder=os.path.join('NLHR-Duplicates', f"X1"),
        transforms=[input_transforms, label_transforms]
    )

    val_dataset = CustomLLEDataset(
        root_dir=os.path.join(base_dir, "Val"),
        ll_folder=os.path.join('LLLR'),
        gt_folder=os.path.join('NLHR-Duplicates', f"X1"),
        transforms=[input_transforms, label_transforms]
    )

    test_dataset = CustomLLEDataset(
        root_dir=os.path.join(base_dir, "Test"),
        ll_folder=os.path.join('LLLR'),
        gt_folder=os.path.join('NLHR-Duplicates', f"X1"),
        transforms=[input_transforms, label_transforms]
    )

    return train_dataset, val_dataset, test_dataset

def get_SR_dataloaders(train, val, test, batch_size):
    batch_size = batch_size

    train_dl = DataLoader(train, batch_size, shuffle=True)
    val_dl = DataLoader(val, batch_size, shuffle=False)
    test_dl = DataLoader(test, batch_size,shuffle=False)

    return train_dl, val_dl, test_dl