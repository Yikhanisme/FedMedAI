"""Chịu trách nhiệm tải, tiền xử lý (preprocess) và cung cấp data loader cho các dataset MedMNIST (như BloodMNIST)."""


import medmnist
from medmnist import info
import torch
import torchvision.transforms as transforms
from medmnist import INFO
from torch.utils.data import DataLoader


def blood_mnist_dataloader(batch_size: int = 32, download: bool = True) :
    """
    Hàm khởi tạo DataLoaders cho bộ dữ liệu BloodMNIST.
    
    Args:
        batch_size (int): Kích thước batch.
        download (bool): Có tự động tải dữ liệu nếu chưa có hay không.
        
    Returns:
        train_loader, val_loader, test_loader, num_classes
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
