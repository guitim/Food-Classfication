"""
食物图像分类项目配置文件
"""
import os

# 数据配置
DATA_ROOT = "data"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "validation")
TEST_DIR = os.path.join(DATA_ROOT, "test")

# 模型配置
NUM_CLASSES = 36
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# 训练配置
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda"

# 两阶段训练配置
STAGE1_EPOCHS = 10  # 冻结backbone，训练分类头
STAGE2_EPOCHS = 40  # 解冻backbone，整体微调
STAGE2_LR = 1e-4    # 第二阶段使用较小学习率

# 模型保存配置
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
VISUALIZATIONS_DIR = "visualizations"

# 数据增强配置
AUGMENTATION_CONFIG = {
    "train": {
        "RandomResizedCrop": {"size": IMAGE_SIZE, "scale": (0.8, 1.0)},
        "RandomHorizontalFlip": {"p": 0.5},
        "ColorJitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
        "RandomRotation": {"degrees": 15},
        "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    },
    "val": {
        "Resize": {"size": IMAGE_SIZE},
        "CenterCrop": {"size": IMAGE_SIZE},
        "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    }
}

# 类别名称映射
CLASS_NAMES = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
] 