"""
可视化工具文件
包含Grad-CAM、注意力热图等可视化功能
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import seaborn as sns
import warnings

# 忽略PIL的UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

from torchvision import transforms
from config import CLASS_NAMES, IMAGE_SIZE

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class GradCAM:
    """Grad-CAM可视化类 - 正确实现"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.hooks = []
        self.register_hooks()
    
    def register_hooks(self):
        """注册前向和反向钩子"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # 注册钩子
        target = self.target_layer
        self.hooks.append(target.register_forward_hook(forward_hook))
        self.hooks.append(target.register_backward_hook(backward_hook))
    
    def remove_hooks(self):
        """移除钩子"""
        for hook in self.hooks:
            hook.remove()
    
    def generate_cam(self, input_image, target_class=None):
        """生成Grad-CAM"""
        # 前向传播
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # 反向传播
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # 检查梯度是否成功获取
        if self.gradients is None:
            print("警告: 未能获取梯度信息")
            return None
        
        # 获取梯度和激活
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # 计算权重 - 全局平均池化
        weights = torch.mean(gradients, dim=[2, 3])
        
        # 生成CAM
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        # 应用ReLU
        cam = F.relu(cam)
        
        # 归一化
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam.numpy()

def visualize_gradcam(model, image_path, target_layer, save_path=None):
    """可视化Grad-CAM"""
    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # 确保输入张量和模型在同一设备上
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # 创建GradCAM
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_cam(input_tensor)
    grad_cam.remove_hooks()
    
    # 检查CAM是否成功生成
    if cam is None:
        print("错误: 无法生成Grad-CAM")
        return
    
    # 获取模型预测
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()
    
    # 可视化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    original_image = np.array(image)
    ax1.imshow(original_image)
    ax1.set_title(f'原始图像\n预测: {CLASS_NAMES[predicted_class]} ({confidence:.2%})')
    ax1.axis('off')
    
    # Grad-CAM热图
    im2 = ax2.imshow(cam, cmap='jet')
    ax2.set_title('Grad-CAM热图')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 叠加图像 - 确保尺寸匹配
    # 将热图调整到原始图像大小
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    cam_colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    
    # 叠加
    alpha = 0.6
    overlay = cv2.addWeighted(original_image, 1-alpha, cam_colored, alpha, 0)
    ax3.imshow(overlay)
    ax3.set_title('Grad-CAM叠加')
    ax3.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM可视化已保存: {save_path}")
    plt.show()
    
    # 打印调试信息
    print(f"预测类别: {CLASS_NAMES[predicted_class]} (置信度: {confidence:.2%})")
    print(f"热图统计: 最小值={cam.min():.4f}, 最大值={cam.max():.4f}, 均值={cam.mean():.4f}")
    print(f"热图尺寸: {cam.shape}, 原始图像尺寸: {original_image.shape}")
    print(f"调整后热图尺寸: {cam_resized.shape}")

def visualize_cbam_attention(model, image_path, save_path=None):
    """可视化CBAM注意力"""
    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # 确保输入张量和模型在同一设备上
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # 获取CBAM注意力权重
    model.eval()
    with torch.no_grad():
        # 提取特征
        features = model.backbone(input_tensor)
        
        # 获取通道注意力
        channel_attention = model.cbam.channel_attention(features)
        spatial_attention = model.cbam.spatial_attention(features * channel_attention)
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图像
    original_image = np.array(image)
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 通道注意力
    channel_att = channel_attention.squeeze().cpu().numpy()
    axes[0, 1].imshow(channel_att, cmap='viridis')
    axes[0, 1].set_title('通道注意力')
    axes[0, 1].axis('off')
    
    # 空间注意力
    spatial_att = spatial_attention.squeeze().cpu().numpy()
    axes[0, 2].imshow(spatial_att, cmap='viridis')
    axes[0, 2].set_title('空间注意力')
    axes[0, 2].axis('off')
    
    # 特征图可视化
    features_np = features.squeeze().cpu().numpy()
    for i in range(3):
        feature_map = features_np[i*8]  # 选择第i*8个通道
        axes[1, i].imshow(feature_map, cmap='viridis')
        axes[1, i].set_title(f'特征图 {i*8}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_distribution(class_counts, save_path=None):
    """绘制类别分布"""
    plt.figure(figsize=(15, 8))
    
    # 创建柱状图
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.bar(range(len(classes)), counts)
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Training Set')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    
    # 添加数值标签
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts) * 0.01, str(count), 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_comparison(results, save_path=None):
    """绘制模型对比结果"""
    models = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.2f}%', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_ablation_study(results, save_path=None):
    """绘制消融实验结果"""
    experiments = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(experiments, accuracies, color='lightblue')
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.2f}%', ha='center', va='bottom')
    
    plt.xlabel('Experiment')
    plt.ylabel('Accuracy (%)')
    plt.title('CBAM Ablation Study Results')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show() 

def simple_gradcam(model, input_image, target_layer, target_class=None):
    """简单的Grad-CAM实现"""
    # 设置模型为评估模式
    model.eval()
    
    # 前向传播
    output = model(input_image)
    
    if target_class is None:
        target_class = output.argmax(dim=1)
    
    # 反向传播
    model.zero_grad()
    output[0, target_class].backward()
    
    # 获取目标层的梯度
    gradients = target_layer.weight.grad.clone()
    activations = target_layer.weight.clone()
    
    # 计算权重
    weights = torch.mean(gradients, dim=[2, 3])
    
    # 生成CAM
    cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
    for i, w in enumerate(weights[0]):
        cam += w * activations[0, i, :, :]
    
    # 应用ReLU
    cam = F.relu(cam)
    
    # 归一化
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    
    return cam.detach().cpu().numpy()

def visualize_gradcam_simple(model, image_path, target_layer, save_path=None):
    """使用简单Grad-CAM进行可视化"""
    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # 确保输入张量和模型在同一设备上
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # 生成Grad-CAM
    cam = simple_gradcam(model, input_tensor, target_layer)
    
    # 获取模型预测
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()
    
    # 可视化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    original_image = np.array(image)
    ax1.imshow(original_image)
    ax1.set_title(f'原始图像\n预测: {CLASS_NAMES[predicted_class]} ({confidence:.2%})')
    ax1.axis('off')
    
    # Grad-CAM热图
    im2 = ax2.imshow(cam, cmap='jet')
    ax2.set_title('Grad-CAM热图')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 叠加图像
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    cam_colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    
    alpha = 0.6
    overlay = cv2.addWeighted(original_image, 1-alpha, cam_colored, alpha, 0)
    ax3.imshow(overlay)
    ax3.set_title('Grad-CAM叠加')
    ax3.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM可视化已保存: {save_path}")
    plt.show()
    
    # 打印调试信息
    print(f"预测类别: {CLASS_NAMES[predicted_class]} (置信度: {confidence:.2%})")
    print(f"热图统计: 最小值={cam.min():.4f}, 最大值={cam.max():.4f}, 均值={cam.mean():.4f}")
    print(f"热图尺寸: {cam.shape}, 原始图像尺寸: {original_image.shape}") 

def correct_gradcam(model, input_image, target_layer, target_class=None):
    """正确的Grad-CAM实现"""
    # 存储梯度和激活
    gradients = None
    activations = None
    
    def save_gradient(grad):
        nonlocal gradients
        gradients = grad
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
        # 注册梯度钩子
        output.register_hook(save_gradient)
    
    # 注册前向钩子
    hook = target_layer.register_forward_hook(forward_hook)
    
    # 前向传播
    output = model(input_image)
    
    if target_class is None:
        target_class = output.argmax(dim=1)
    
    # 反向传播
    model.zero_grad()
    output[0, target_class].backward()
    
    # 移除钩子
    hook.remove()
    
    # 检查是否成功获取梯度
    if gradients is None or activations is None:
        print("警告: 未能获取梯度或激活信息")
        return None
    
    # 计算权重
    gradients = gradients.detach().cpu()
    activations = activations.detach().cpu()
    
    # 全局平均池化梯度
    weights = torch.mean(gradients, dim=[2, 3])
    
    # 生成CAM
    cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
    for i, w in enumerate(weights[0]):
        cam += w * activations[0, i, :, :]
    
    # 应用ReLU
    cam = F.relu(cam)
    
    # 归一化
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    
    return cam.numpy()

def visualize_gradcam_correct(model, image_path, target_layer, save_path=None, show_plot=False):
    """使用正确的Grad-CAM进行可视化"""
    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # 确保输入张量和模型在同一设备上
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # 生成Grad-CAM
    cam = correct_gradcam(model, input_tensor, target_layer)
    
    if cam is None:
        print("错误: 无法生成Grad-CAM")
        return
    
    # 获取模型预测
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()
    
    # 可视化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    original_image = np.array(image)
    ax1.imshow(original_image)
    ax1.set_title(f'原始图像\n预测: {CLASS_NAMES[predicted_class]} ({confidence:.2%})')
    ax1.axis('off')
    
    # Grad-CAM热图
    im2 = ax2.imshow(cam, cmap='jet')
    ax2.set_title('Grad-CAM热图')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 叠加图像
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    cam_colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    
    alpha = 0.6
    overlay = cv2.addWeighted(original_image, 1-alpha, cam_colored, alpha, 0)
    ax3.imshow(overlay)
    ax3.set_title('Grad-CAM叠加')
    ax3.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM可视化已保存: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)  # 关闭图形以释放内存
    
    # 打印调试信息
    print(f"预测类别: {CLASS_NAMES[predicted_class]} (置信度: {confidence:.2%})")
    print(f"热图统计: 最小值={cam.min():.4f}, 最大值={cam.max():.4f}, 均值={cam.mean():.4f}")
    print(f"热图尺寸: {cam.shape}, 原始图像尺寸: {original_image.shape}") 