"""
数据集处理文件
包含数据加载、预处理和可视化功能
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import warnings

# 忽略PIL的UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

from config import *

class FoodDataset:
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(data_dir, transform=transform)
        self.class_names = self.dataset.classes
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def get_transforms():
    """获取数据增强变换"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_dataloaders():
    """创建数据加载器"""
    train_transform, val_transform = get_transforms()
    
    # 创建数据集
    train_dataset = FoodDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = FoodDataset(VAL_DIR, transform=val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.class_names

def visualize_samples(dataloader, num_samples=16, save_path=None):
    """可视化数据样本"""
    images, labels = next(iter(dataloader))
    
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    
    # 创建网格
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {labels[i]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_dataset():
    """分析数据集统计信息"""
    train_transform, _ = get_transforms()
    train_dataset = FoodDataset(TRAIN_DIR, transform=train_transform)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"类别数量: {len(train_dataset.class_names)}")
    print(f"类别名称: {train_dataset.class_names}")
    
    # 统计每个类别的样本数量
    class_counts = {}
    for _, label in train_dataset.dataset.samples:
        class_name = train_dataset.class_names[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\n各类别样本数量:")
    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name}: {count}")
    
    return class_counts 