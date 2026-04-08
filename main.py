"""
食物图像分类项目主程序
"""
import os
import torch
import argparse
import warnings

# 忽略PIL的UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

from dataset import create_dataloaders, analyze_dataset, visualize_samples
from trainer import Trainer
from visualization import *
from config import *

def main():
    parser = argparse.ArgumentParser(description='食物图像分类项目')
    parser.add_argument('--model', type=str, default='resnet50', 
                       choices=['resnet50', 'resnet50_cbam', 'swin_t', 'swin_s', 'swin_b'],
                       help='选择模型类型')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'visualize', 'ablation'],
                       help='运行模式')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径')
    parser.add_argument('--continue_training', action='store_true',
                       help='是否继续训练')
    parser.add_argument('--additional_epochs', type=int, default=50,
                       help='额外训练的轮数')
    args = parser.parse_args()
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建必要的目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    if args.mode == 'train':
        # 训练模式
        print("=== 开始训练 ===")
        
        # 分析数据集
        print("\n1. 分析数据集...")
        class_counts = analyze_dataset()
        
        # 创建数据加载器
        print("\n2. 创建数据加载器...")
        train_loader, val_loader, class_names = create_dataloaders()
        
        # 训练模型
        print(f"\n3. 训练 {args.model} 模型...")
        trainer = Trainer(model_type=args.model, device=device)
        
        # 如果指定了继续训练
        if args.continue_training and args.checkpoint:
            print(f"继续训练模式，从检查点: {args.checkpoint}")
            best_acc = trainer.train_two_stage(
                train_loader, 
                val_loader, 
                continue_from_checkpoint=args.checkpoint,
                additional_epochs=args.additional_epochs
            )
        else:
            best_acc = trainer.train_two_stage(train_loader, val_loader)
        
        # 绘制训练历史
        print("\n4. 绘制训练历史...")
        trainer.plot_training_history(save_path=os.path.join(VISUALIZATIONS_DIR, f'{args.model}_training_history.png'))
        
        print(f"\n训练完成！最佳准确率: {best_acc:.2f}%")
        
    elif args.mode == 'evaluate':
        # 评估模式
        print("=== 模型评估 ===")
        
        if args.checkpoint is None:
            print("错误: 评估模式需要指定检查点路径")
            return
        
        # 创建数据加载器
        train_loader, val_loader, class_names = create_dataloaders()
        
        # 加载模型
        trainer = Trainer(model_type=args.model, device=device)
        trainer.load_checkpoint(args.checkpoint)
        
        # 评估模型
        accuracy, report, cm = trainer.evaluate_model(val_loader, class_names)
        
        # 保存结果
        with open(os.path.join(RESULTS_DIR, f'{args.model}_evaluation.txt'), 'w') as f:
            f.write(f"模型: {args.model}\n")
            f.write(f"准确率: {accuracy:.4f}\n\n")
            f.write("分类报告:\n")
            f.write(report)
        
        print(f"评估结果已保存到 {RESULTS_DIR}")
        
    elif args.mode == 'visualize':
        # 可视化模式
        print("=== 模型可视化 ===")
        
        if args.checkpoint is None:
            print("错误: 可视化模式需要指定检查点路径")
            return
        
        # 加载模型
        trainer = Trainer(model_type=args.model, device=device)
        trainer.load_checkpoint(args.checkpoint)
        
        # 创建Grad-CAM结果文件夹
        gradcam_dir = os.path.join(VISUALIZATIONS_DIR, 'gradcam_results')
        os.makedirs(gradcam_dir, exist_ok=True)
        
        # 为每个类别生成Grad-CAM可视化
        print("开始为每个类别生成Grad-CAM可视化...")
        
        for class_idx, class_name in enumerate(CLASS_NAMES):
            print(f"\n处理类别 {class_idx+1}/{len(CLASS_NAMES)}: {class_name}")
            
            # 查找该类别的验证图像
            class_dir = os.path.join("data/validation", class_name)
            if not os.path.exists(class_dir):
                print(f"警告: 类别 {class_name} 的验证目录不存在")
                continue
            
            # 获取该类别的前3张图像
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                print(f"警告: 类别 {class_name} 没有找到图像文件")
                continue
            
            # 选择前3张图像进行可视化
            selected_images = image_files[:3]
            
            for img_idx, img_file in enumerate(selected_images):
                image_path = os.path.join(class_dir, img_file)
                print(f"  处理图像 {img_idx+1}: {img_file}")
                
                # 确定目标层
                if args.model == 'resnet50':
                    target_layer = trainer.model.model.layer4[-1]
                elif args.model == 'resnet50_cbam':
                    target_layer = trainer.model.backbone.layer4[-1]
                else:
                    # Swin Transformer的最后一层
                    target_layer = trainer.model.layers[-1]
                
                # 生成Grad-CAM可视化
                from visualization import visualize_gradcam_correct
                save_filename = f"{class_name}_{img_idx+1}_{args.model}_gradcam.png"
                save_path = os.path.join(gradcam_dir, save_filename)
                
                try:
                    visualize_gradcam_correct(
                        trainer.model, 
                        image_path, 
                        target_layer,
                        save_path=save_path
                    )
                    print(f"    ✓ 已保存: {save_filename}")
                except Exception as e:
                    print(f"    ✗ 处理失败: {e}")
        
        print(f"\nGrad-CAM可视化完成！结果保存在: {gradcam_dir}")
        
        # CBAM注意力可视化（仅对CBAM模型）
        if args.model == 'resnet50_cbam':
            print("\n开始生成CBAM注意力可视化...")
            cbam_dir = os.path.join(VISUALIZATIONS_DIR, 'cbam_attention_results')
            os.makedirs(cbam_dir, exist_ok=True)
            
            # 为前5个类别生成CBAM可视化
            for class_idx, class_name in enumerate(CLASS_NAMES[:5]):
                print(f"处理CBAM注意力: {class_name}")
                
                class_dir = os.path.join("data/validation", class_name)
                if not os.path.exists(class_dir):
                    continue
                
                image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    image_path = os.path.join(class_dir, image_files[0])
                    save_filename = f"{class_name}_{args.model}_cbam_attention.png"
                    save_path = os.path.join(cbam_dir, save_filename)
                    
                    try:
                        visualize_cbam_attention(
                            trainer.model,
                            image_path,
                            save_path=save_path
                        )
                        print(f"  ✓ 已保存: {save_filename}")
                    except Exception as e:
                        print(f"  ✗ 处理失败: {e}")
            
            print(f"CBAM注意力可视化完成！结果保存在: {cbam_dir}")
        
    elif args.mode == 'ablation':
        # 消融实验模式
        print("=== CBAM消融实验 ===")
        
        # 创建数据加载器
        train_loader, val_loader, class_names = create_dataloaders()
        
        # 消融实验配置
        ablation_configs = [
            ('resnet50', 'ResNet50 (基线)'),
            ('resnet50_cbam', 'ResNet50 + CBAM (完整)')
        ]
        
        results = {}
        
        for model_type, description in ablation_configs:
            print(f"\n训练 {description}...")
            trainer = Trainer(model_type=model_type, device=device)
            best_acc = trainer.train_two_stage(train_loader, val_loader)
            results[description] = best_acc
        
        # 绘制消融实验结果
        print("\n绘制消融实验结果...")
        plot_ablation_study(results, save_path=os.path.join(VISUALIZATIONS_DIR, 'ablation_study.png'))
        
        # 保存结果
        with open(os.path.join(RESULTS_DIR, 'ablation_study.txt'), 'w') as f:
            f.write("CBAM消融实验结果:\n")
            for description, acc in results.items():
                f.write(f"{description}: {acc:.2f}%\n")
        
        print("消融实验完成！")

if __name__ == '__main__':
    main() 