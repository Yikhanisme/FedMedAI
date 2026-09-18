import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np

    
class ConvBlock(nn.Module):
    """
    Khối Convolutional cơ bản:
    Conv2d (stride=1, padding=1) -> BatchNorm2d -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class SimpleCNN(nn.Module):
    """
    Kiến trúc mạng CNN tối ưu cho bài toán phân loại ảnh y tế (MedMNIST - BloodMNIST 28x28x3):
    
    Input (3, 28, 28)
           ↓
       Conv Block 1 (3 -> 16, 28x28)
           ↓
       Conv Block 2 (16 -> 32, 28x28)
           ↓
       Pooling (MaxPool2d 2x2 -> 32, 14x14)
           ↓
       Global Average Pooling (GAP -> 32, 1x1)
           ↓
       Linear (32 -> num_classes)
           ↓
       Output (num_classes = 8)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 8):
        super(SimpleCNN, self).__init__()
        # 2 khối Conv liên tiếp
        self.conv_block1 = ConvBlock(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv_block2 = ConvBlock(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Lớp Pooling giảm kích thước không gian (28x28 -> 14x14)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling (nén 14x14 về 1x1 cho mỗi channel)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 1 tầng phân loại tuyến tính duy nhất
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)  # [B, 16, 28, 28]
        x = self.conv_block2(x)  # [B, 32, 28, 28]
        x = self.pool(x)         # [B, 32, 14, 14]
        x = self.gap(x)          # [B, 32, 1, 1]
        x = torch.flatten(x, 1)  # [B, 32]
        x = self.fc(x)           # [B, num_classes]
        return x


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Trích xuất trọng số mô hình PyTorch thành danh sách các mảng NumPy."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Nạp danh sách trọng số NumPy vào mô hình PyTorch."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


if __name__ == "__main__":
    # Chạy kiểm thử kiến trúc mô hình
    model = SimpleCNN(in_channels=3, num_classes=8)
    dummy_input = torch.randn(4, 3, 28, 28)
    output = model(dummy_input)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=== MODEL TEST: SimpleCNN ===")
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total trainable parameters: {total_params:,}")
    assert output.shape == (4, 8), "Error: Output shape is not (4, 8)!"
    print("Test passed successfully!")
