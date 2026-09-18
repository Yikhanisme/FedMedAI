"""Chịu trách nhiệm tải, tiền xử lý (preprocess) và cung cấp data loader cho các dataset MedMNIST (như BloodMNIST)."""


import medmnist
from medmnist import info
import torch
import torchvision.transforms as transforms
from medmnist import INFO
from torch.utils.data import DataLoader


def get_bloodmnist_datasets(batch_size: int = 32, download: bool = True) :
    """
    Hàm khởi tạo DataLoaders cho bộ dữ liệu BloodMNIST.
    
    Args:
        batch_size (int): Kích thước batch.
        download (bool): Có tự động tải dữ liệu nếu chưa có hay không.
        
    Returns:
        train_dataset, val_dataset, test_dataset, num_classes
    """

    # khai bao dataset
    data_flag = 'bloodmnist'
    info = INFO[data_flag]
    num_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    #Transform data
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load dataset
    train_dataset = DataClass(split = 'train', transform = data_transform, download = download)
    val_dataset = DataClass(split = 'val', transform = data_transform, download = download)
    test_dataset = DataClass(split = 'test', transform = data_transform, download = download)

    return train_dataset, val_dataset, test_dataset, num_classes



def get_bloodmnist_dataloaders(batch_size: int = 32, download: bool = True):
    """
    Trả về DataLoaders phục vụ cho Centralized Baseline (Train trực tiếp toàn bộ dữ liệu).
    """
    train_dataset, val_dataset, test_dataset, num_classes = get_bloodmnist_datasets(download)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader, num_classes
