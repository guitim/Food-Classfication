"""
模型定义文件
包含ResNet50、CBAM注意力机制和Swin Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from config import NUM_CLASSES, CHECKPOINT_DIR

class ChannelAttention(nn.Module):
    """通道注意力模块 (CAM)"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    """空间注意力模块 (SAM)"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

class CBAM(nn.Module):
    """CBAM注意力模块"""
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
        
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class ResNet50WithCBAM(nn.Module):
    """ResNet50 + CBAM注意力机制"""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(ResNet50WithCBAM, self).__init__()
        
        # 加载预训练的ResNet50
        if pretrained:
            self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = torchvision.models.resnet50(weights=None)
        
        # 移除最后的分类层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # 保留到倒数第二层
        
        # 获取特征维度 (ResNet50的最后一层是2048)
        feature_dim = 2048
        
        # 添加CBAM注意力模块
        self.cbam = CBAM(feature_dim)
        
        # 分类头
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        # 提取特征
        features = self.backbone(x)
        
        # 应用CBAM注意力
        features = self.cbam(features)
        features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        # 分类
        output = self.classifier(features)
        return output

class ResNet50Baseline(nn.Module):
    """ResNet50基线模型"""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(ResNet50Baseline, self).__init__()
        if pretrained:
            self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.model = torchvision.models.resnet50(weights=None)
        
        # 修改最后的分类层
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)

class SwinTransformer(nn.Module):
    """Swin Transformer模型"""
    def __init__(self, model_name='swin_tiny_patch4_window7_224', num_classes=NUM_CLASSES, pretrained=True):
        super(SwinTransformer, self).__init__()
        
        # 将简化的模型类型映射到完整的模型名称
        model_name_map = {
            'swin_t': 'swin_tiny_patch4_window7_224',
            'swin_s': 'swin_small_patch4_window7_224', 
            'swin_b': 'swin_base_patch4_window7_224'
        }
        
        # 如果传入的是简化名称，转换为完整名称
        if model_name in model_name_map:
            model_name = model_name_map[model_name]
        
        # 创建模型（不加载预训练权重）
        if model_name == 'swin_tiny_patch4_window7_224':
            self.model = torchvision.models.swin_t(weights=None)
        elif model_name == 'swin_small_patch4_window7_224':
            self.model = torchvision.models.swin_s(weights=None)
        elif model_name == 'swin_base_patch4_window7_224':
            self.model = torchvision.models.swin_b(weights=None)
        else:
            raise ValueError(f"不支持的Swin模型: {model_name}")
        
        # 修改最后的分类层
        num_ftrs = self.model.head.in_features
        self.model.head = nn.Linear(num_ftrs, num_classes)
        
        # 如果指定加载预训练权重
        if pretrained:
            self.load_pretrained_weights(model_name)
    
    def load_pretrained_weights(self, model_name):
        """加载预训练权重"""
        weight_file = os.path.join(CHECKPOINT_DIR, f"{model_name}.pth")
        if os.path.exists(weight_file):
            print(f"加载预训练权重: {weight_file}")
            checkpoint = torch.load(weight_file, map_location='cpu')
            
            # 处理权重加载
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 移除分类头的权重
            state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
            
            # 加载权重
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"缺失的键: {missing_keys}")
            if unexpected_keys:
                print(f"意外的键: {unexpected_keys}")
        else:
            print(f"警告: 预训练权重文件不存在: {weight_file}")
    
    def forward(self, x):
        return self.model(x)

def create_swin_transformer(model_type='swin_t', num_classes=NUM_CLASSES, pretrained=True):
    """创建Swin Transformer模型"""
    # 将简化的模型类型映射到完整的模型名称
    model_name_map = {
        'swin_t': 'swin_tiny_patch4_window7_224',
        'swin_s': 'swin_small_patch4_window7_224', 
        'swin_b': 'swin_base_patch4_window7_224'
    }
    
    if model_type not in model_name_map:
        raise ValueError(f"不支持的Swin模型类型: {model_type}")
    
    model_name = model_name_map[model_type]
    return SwinTransformer(model_name, num_classes, pretrained)

def get_model(model_type):
    """获取模型"""
    if model_type == 'resnet50':
        return ResNet50Baseline()
    elif model_type == 'resnet50_cbam':
        return ResNet50WithCBAM()
    elif model_type in ['swin_t', 'swin_s', 'swin_b']:
        return create_swin_transformer(model_type)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def load_pretrained_weights(model, model_type):
    """从本地加载预训练权重"""
    pretrained_dir = os.path.join(CHECKPOINT_DIR, 'pretrained_weights')
    
    if model_type == 'resnet50':
        weight_path = os.path.join(pretrained_dir, 'resnet50_pretrained.pth')
        if os.path.exists(weight_path):
            print(f"加载ResNet50预训练权重: {weight_path}")
            state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
            # 移除分类层的权重
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)
            model.model.load_state_dict(state_dict, strict=False)
            print("ResNet50预训练权重加载成功")
        else:
            print(f"警告: 预训练权重文件不存在: {weight_path}")
    
    elif model_type == 'resnet50_cbam':
        weight_path = os.path.join(pretrained_dir, 'resnet50_pretrained.pth')
        if os.path.exists(weight_path):
            print(f"加载ResNet50预训练权重: {weight_path}")
            state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
            # 移除分类层的权重
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)
            model.backbone.load_state_dict(state_dict, strict=False)
            print("ResNet50+CBAM预训练权重加载成功")
        else:
            print(f"警告: 预训练权重文件不存在: {weight_path}")
    
    elif model_type in ['swin_t', 'swin_s', 'swin_b']:
        # 根据模型类型确定权重文件名
        if model_type == 'swin_t':
            weight_file = 'swin_tiny_patch4_window7_224.pth'
        elif model_type == 'swin_s':
            weight_file = 'swin_small_patch4_window7_224.pth'
        else:  # swin_b
            weight_file = 'swin_base_patch4_window7_224.pth'
        
        weight_path = os.path.join(pretrained_dir, weight_file)
        if os.path.exists(weight_path):
            print(f"加载Swin Transformer预训练权重: {weight_path}")
            state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
            # 移除分类头的权重
            state_dict.pop('head.weight', None)
            state_dict.pop('head.bias', None)
            model.load_state_dict(state_dict, strict=False)
            print(f"{model_type.upper()}预训练权重加载成功")
        else:
            print(f"警告: 预训练权重文件不存在: {weight_path}")
    
    return model

def freeze_backbone(model):
    """冻结backbone参数，但保持分类头可训练"""
    if hasattr(model, 'backbone'):
        # 对于ResNet50WithCBAM，冻结backbone但保持classifier可训练
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, 'model') and hasattr(model.model, 'fc'):
        # 对于ResNet50Baseline，冻结backbone但保持fc层可训练
        for name, param in model.model.named_parameters():
            if 'fc' not in name:  # 冻结除了fc层之外的所有层
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif hasattr(model, 'model') and hasattr(model.model, 'head'):
        # 对于Swin Transformer，冻结backbone但保持head可训练
        for name, param in model.model.named_parameters():
            if 'head' not in name:  # 冻结除了head之外的所有层
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        # 对于其他模型，冻结除了最后一层之外的所有层
        for name, param in model.named_parameters():
            if 'head' not in name and 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    print("Backbone已冻结，分类头保持可训练")

def unfreeze_backbone(model):
    """解冻backbone参数"""
    for param in model.parameters():
        param.requires_grad = True
    print("Backbone已解冻")

def get_trainable_parameters(model):
    """获取可训练参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_total_parameters(model):
    """获取总参数数量"""
    return sum(p.numel() for p in model.parameters()) 