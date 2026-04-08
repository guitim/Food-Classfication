"""
独立的数据可视化脚本
生成5张包含36个类别的图片，每张图片显示所有类别的样本
"""
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import warnings

# 忽略PIL的UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

from config import TRAIN_DIR, CLASS_NAMES, VISUALIZATIONS_DIR

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def create_class_overview_images(num_images=5):
    """创建多张类别概览图片"""
    print(f"开始生成 {num_images} 张类别概览图片...")
    
    # 确保输出目录存在
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    for img_idx in range(num_images):
        print(f"生成第 {img_idx + 1} 张图片...")
        
        # 创建6x6的子图布局
        fig, axes = plt.subplots(6, 6, figsize=(20, 20))
        fig.suptitle(f'食物图像数据集 - 36个类别概览 (第{img_idx + 1}张)', 
                    fontsize=16, fontweight='bold')
        
        # 为每个类别选择一张图片
        for i, class_name in enumerate(CLASS_NAMES):
            row = i // 6
            col = i % 6
            ax = axes[row, col]
            
            # 获取该类别的图片列表
            class_dir = os.path.join(TRAIN_DIR, class_name)
            if os.path.exists(class_dir):
                image_files = [f for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if image_files:
                    # 随机选择一张图片
                    random.seed(img_idx * 100 + i)  # 确保每次运行结果一致
                    selected_image = random.choice(image_files)
                    image_path = os.path.join(class_dir, selected_image)
                    
                    try:
                        # 加载和显示图片
                        img = Image.open(image_path)
                        img = img.convert('RGB')
                        
                        # 调整图片大小
                        img.thumbnail((150, 150), Image.Resampling.LANCZOS)
                        
                        ax.imshow(img)
                        ax.set_title(f'{class_name}\n({selected_image})', 
                                   fontsize=10, fontweight='bold')
                        ax.axis('off')
                        
                        # 添加边框
                        rect = patches.Rectangle((0, 0), 1, 1, 
                                               linewidth=2, edgecolor='black', 
                                               facecolor='none', transform=ax.transAxes)
                        ax.add_patch(rect)
                        
                    except Exception as e:
                        print(f"加载图片失败 {image_path}: {e}")
                        ax.text(0.5, 0.5, f'Error\n{class_name}', 
                               ha='center', va='center', transform=ax.transAxes,
                               fontsize=10, color='red')
                        ax.axis('off')
                else:
                    ax.text(0.5, 0.5, f'No Images\n{class_name}', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10, color='gray')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'No Directory\n{class_name}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='red')
                ax.axis('off')
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(VISUALIZATIONS_DIR, f'class_overview_{img_idx + 1}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存: {output_path}")
        
        plt.close()
    
    print(f"完成！生成了 {num_images} 张类别概览图片")

def create_class_distribution_analysis():
    """创建类别分布分析"""
    print("开始分析类别分布...")
    
    class_counts = {}
    total_images = 0
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if os.path.exists(class_dir):
            image_files = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            count = len(image_files)
            class_counts[class_name] = count
            total_images += count
        else:
            class_counts[class_name] = 0
    
    # 创建分布图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # 柱状图
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = ax1.bar(range(len(classes)), counts, color='skyblue', edgecolor='black')
    ax1.set_title('各类别图片数量分布', fontsize=14, fontweight='bold')
    ax1.set_xlabel('类别', fontsize=12)
    ax1.set_ylabel('图片数量', fontsize=12)
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}', ha='center', va='bottom', fontsize=8)
    
    # 饼图
    # 只显示前10个类别，其他归为"其他"
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    top_10_classes = sorted_counts[:10]
    other_count = sum(count for _, count in sorted_counts[10:])
    
    labels = [name for name, _ in top_10_classes]
    if other_count > 0:
        labels.append('其他')
    
    sizes = [count for _, count in top_10_classes]
    if other_count > 0:
        sizes.append(other_count)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
    
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.set_title('前10个类别占比', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(VISUALIZATIONS_DIR, 'class_distribution_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"分布分析图已保存: {output_path}")
    
    # 打印统计信息
    print(f"\n数据集统计信息:")
    print(f"总图片数量: {total_images}")
    print(f"类别数量: {len(CLASS_NAMES)}")
    print(f"平均每类图片数量: {total_images / len(CLASS_NAMES):.1f}")
    print(f"最多图片的类别: {max(class_counts.items(), key=lambda x: x[1])}")
    print(f"最少图片的类别: {min(class_counts.items(), key=lambda x: x[1])}")
    
    plt.close()

def main():
    """主函数"""
    print("=== 食物图像数据集可视化工具 ===")
    
    # 检查数据目录是否存在
    if not os.path.exists(TRAIN_DIR):
        print(f"错误: 训练数据目录不存在: {TRAIN_DIR}")
        return
    
    # 创建类别概览图片
    create_class_overview_images(num_images=5)
    
    # 创建类别分布分析
    create_class_distribution_analysis()
    
    print("\n=== 可视化完成 ===")
    print(f"所有图片已保存到: {VISUALIZATIONS_DIR}")

if __name__ == "__main__":
    main() 