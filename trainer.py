"""
训练器文件
包含两阶段训练策略和训练循环
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
TENSORBOARD_AVAILABLE = True
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from config import *
from models import get_model, freeze_backbone, unfreeze_backbone, get_trainable_parameters, get_total_parameters, load_pretrained_weights

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class Trainer:
    def __init__(self, model_type='resnet50', device=DEVICE):
        self.device = device
        self.model_type = model_type
        self.model = get_model(model_type).to(device)
        
        # 加载预训练权重
        from models import load_pretrained_weights
        self.model = load_pretrained_weights(self.model, model_type)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        self.writer = None
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def setup_training(self, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
        """设置训练参数"""
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=EPOCHS
        )
        
        # 创建TensorBoard writer
        if TENSORBOARD_AVAILABLE:
            log_dir = os.path.join(RESULTS_DIR, f"{self.model_type}_logs")
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 改进信息显示，显示更清晰的batch信息
        total_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f"训练中 (共{total_batches}个batch)")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 改进进度条显示
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Batch': f'{batch_idx+1}/{total_batches}',
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
            
            # 记录到TensorBoard（每个batch）
            if self.writer:
                self.writer.add_scalar('Batch/Train_Loss', loss.item(), 
                                     self.global_step if hasattr(self, 'global_step') else batch_idx)
                self.writer.add_scalar('Batch/Train_Acc', current_acc, 
                                     self.global_step if hasattr(self, 'global_step') else batch_idx)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        total_batches = len(val_loader)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(val_loader, 
                                                       desc=f"验证中 (共{total_batches}个batch)")):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_preds, all_targets
    
    def train_two_stage(self, train_loader, val_loader, continue_from_checkpoint=None, additional_epochs=50):
        """两阶段训练"""
        print(f"开始训练 {self.model_type} 模型")
        print(f"总参数数量: {get_total_parameters(self.model):,}")
        print(f"训练集大小: {len(train_loader.dataset)} 样本")
        print(f"验证集大小: {len(val_loader.dataset)} 样本")
        print(f"训练批次数: {len(train_loader)} (batch_size={BATCH_SIZE})")
        print(f"验证批次数: {len(val_loader)} (batch_size={BATCH_SIZE})")
        
        # 如果指定了继续训练的检查点，先加载
        start_epoch = 0
        if continue_from_checkpoint:
            print(f"从检查点继续训练: {continue_from_checkpoint}")
            self.load_checkpoint(continue_from_checkpoint)
            start_epoch = len(self.train_losses)  # 从之前的epoch数开始
            print(f"继续训练，从第 {start_epoch + 1} 轮开始")
        
        # 阶段1: 冻结backbone，只训练分类头
        print("\n=== 阶段1: 训练分类头 ===")
        freeze_backbone(self.model)
        print(f"可训练参数数量: {get_trainable_parameters(self.model):,}")
        
        self.setup_training(learning_rate=LEARNING_RATE)
        
        best_val_acc = max(self.val_accs) if self.val_accs else 0
        
        # 计算还需要训练的轮数
        remaining_stage1_epochs = max(0, STAGE1_EPOCHS - start_epoch)
        if remaining_stage1_epochs > 0:
            for epoch in range(remaining_stage1_epochs):
                current_epoch = start_epoch + epoch
                print(f"\nEpoch {current_epoch+1}/{STAGE1_EPOCHS}")
                
                train_loss, train_acc = self.train_epoch(train_loader)
                val_loss, val_acc, _, _ = self.validate_epoch(val_loader)
                
                self.scheduler.step()
                
                # 记录历史
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accs.append(train_acc)
                self.val_accs.append(val_acc)
                
                # 增强TensorBoard日志
                if self.writer:
                    self.writer.add_scalar('Stage1/Train_Loss', train_loss, current_epoch)
                    self.writer.add_scalar('Stage1/Val_Loss', val_loss, current_epoch)
                    self.writer.add_scalar('Stage1/Train_Acc', train_acc, current_epoch)
                    self.writer.add_scalar('Stage1/Val_Acc', val_acc, current_epoch)
                    self.writer.add_scalar('Stage1/Learning_Rate', self.optimizer.param_groups[0]['lr'], current_epoch)
                    
                    # 添加更多指标
                    self.writer.add_scalar('Stage1/Loss_Diff', train_loss - val_loss, current_epoch)
                    self.writer.add_scalar('Stage1/Acc_Diff', val_acc - train_acc, current_epoch)
                
                print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
                print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
                print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_checkpoint(f"{self.model_type}_stage1_best.pth")
                    print(f"✓ 新的最佳验证准确率: {best_val_acc:.2f}%")
        
        # 阶段2: 解冻backbone，整体微调
        print("\n=== 阶段2: 整体微调 ===")
        unfreeze_backbone(self.model)
        print(f"可训练参数数量: {get_trainable_parameters(self.model):,}")
        
        # 重新设置优化器，使用较小的学习率
        self.setup_training(learning_rate=STAGE2_LR)
        
        # 计算阶段2的起始epoch
        stage2_start_epoch = max(0, len(self.train_losses) - STAGE2_EPOCHS)
        remaining_stage2_epochs = max(0, STAGE2_EPOCHS - (len(self.train_losses) - stage2_start_epoch))
        
        # 如果指定了额外训练轮数，使用指定的轮数
        if additional_epochs > 0:
            remaining_stage2_epochs = additional_epochs
        
        if remaining_stage2_epochs > 0:
            for epoch in range(remaining_stage2_epochs):
                current_epoch = len(self.train_losses)
                print(f"\nEpoch {current_epoch+1} (额外训练第 {epoch+1}/{remaining_stage2_epochs} 轮)")
                
                train_loss, train_acc = self.train_epoch(train_loader)
                val_loss, val_acc, _, _ = self.validate_epoch(val_loader)
                
                self.scheduler.step()
                
                # 记录历史
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accs.append(train_acc)
                self.val_accs.append(val_acc)
                
                # 增强TensorBoard日志
                if self.writer:
                    self.writer.add_scalar('Stage2/Train_Loss', train_loss, current_epoch)
                    self.writer.add_scalar('Stage2/Val_Loss', val_loss, current_epoch)
                    self.writer.add_scalar('Stage2/Train_Acc', train_acc, current_epoch)
                    self.writer.add_scalar('Stage2/Val_Acc', val_acc, current_epoch)
                    self.writer.add_scalar('Stage2/Learning_Rate', self.optimizer.param_groups[0]['lr'], current_epoch)
                    
                    # 添加更多指标
                    self.writer.add_scalar('Stage2/Loss_Diff', train_loss - val_loss, current_epoch)
                    self.writer.add_scalar('Stage2/Acc_Diff', val_acc - train_acc, current_epoch)
                
                print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
                print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
                print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_checkpoint(f"{self.model_type}_stage2_best.pth")
                    print(f"✓ 新的最佳验证准确率: {best_val_acc:.2f}%")
        
        print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
        if self.writer:
            self.writer.close()
        
        return best_val_acc
    
    def save_checkpoint(self, filename):
        """保存模型检查点"""
        trained_models_dir = os.path.join(CHECKPOINT_DIR, 'trained_models')
        os.makedirs(trained_models_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_type': self.model_type,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        torch.save(checkpoint, os.path.join(trained_models_dir, filename))
        print(f"模型已保存: {trained_models_dir}/{filename}")
    
    def load_checkpoint(self, filename):
        """加载模型检查点"""
        # 首先尝试从trained_models目录加载
        trained_models_dir = os.path.join(CHECKPOINT_DIR, 'trained_models')
        checkpoint_path = os.path.join(trained_models_dir, filename)
        
        if not os.path.exists(checkpoint_path):
            # 如果trained_models中没有，尝试从根目录加载（兼容旧版本）
            checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)
            
        if not os.path.exists(checkpoint_path):
            # 如果还是不存在，尝试直接使用提供的路径
            checkpoint_path = filename
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"找不到检查点文件: {filename}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']
        print(f"模型已加载: {checkpoint_path}")
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, test_loader, class_names):
        """评估模型性能"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_targets, all_preds)
        report = classification_report(all_targets, all_preds, target_names=class_names)
        cm = confusion_matrix(all_targets, all_preds)
        
        print(f"测试准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(report)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        return accuracy, report, cm 