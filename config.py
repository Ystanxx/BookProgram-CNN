"""
配置文件
定义所有超参数、模型配置和路径配置
"""
import os

# ==================== 数据配置 ====================
# 输入数据维度
INPUT_DIM = 1015  # 拉曼谱图特征点数量

# 分类任务配置
NUM_CLASSES = 10  # 发酵阶段数量
CLASS_NAMES = [
    "阶段1", "阶段2", "阶段3", "阶段4", "阶段5",
    "阶段6", "阶段7", "阶段8", "阶段9", "阶段10"
]

# 回归任务配置
NUM_TARGETS = 10  # 化学成分数量
TARGET_NAMES = [
    "儿茶素C", "EGCG", "咖啡因", "茶氨酸", "表儿茶素",
    "表没食子儿茶素", "没食子酸", "茶黄素", "茶红素", "可溶性糖"
]

# ==================== 模型配置 ====================
# 1D-CNN 配置
CNN_CONFIG = {
    'conv_channels': [32, 64, 128, 256],  # 各卷积层通道数
    'kernel_sizes': [7, 5, 3, 3],  # 各卷积层卷积核大小
    'pool_sizes': [2, 2, 2, 2],  # 各池化层大小
    'dropout_rate': 0.5,  # Dropout比率
    'fc_hidden_dim': 512,  # 全连接层隐藏层维度
}

# 1D-ResNet18 配置
RESNET_CONFIG = {
    'initial_channels': 64,  # 初始卷积层通道数
    'block_channels': [64, 128, 256, 512],  # 各残差层组通道数
    'num_blocks': [2, 2, 2, 2],  # 各残差层组的块数量
    'dropout_rate': 0.5,
}

# ==================== 训练配置 ====================
# 基本训练参数
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# 早停策略
EARLY_STOPPING_PATIENCE = 15  # 早停耐心值

# 学习率调度
LR_SCHEDULER = {
    'type': 'ReduceLROnPlateau',  # 学习率调度器类型
    'factor': 0.5,  # 衰减因子
    'patience': 10,  # 耐心值
    'min_lr': 1e-6,  # 最小学习率
}

# 损失函数权重
LOSS_WEIGHTS = {
    'classification': 1.0,  # 分类损失权重 α
    'regression': 1.0,  # 回归损失权重 β
}

# 交叉验证
K_FOLDS = 5  # 五折交叉验证

# ==================== 数据集配置 ====================
# 模拟数据生成参数
DATA_CONFIG = {
    'num_samples': 1000,  # 总样本数
    'noise_level': 0.05,  # 噪声水平
    'peak_intensity_range': (0.1, 1.0),  # 峰强度范围
    'concentration_range': (0.0, 100.0),  # 浓度范围（mg/L）
    'random_seed': 42,  # 随机种子
}

# 数据划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ==================== 路径配置 ====================
# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

# 创建必要的目录
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, RESULT_DIR]:
    os.makedirs(directory, exist_ok=True)

# 模型保存路径
MODEL_SAVE_PATH = {
    'cnn': os.path.join(MODEL_DIR, 'cnn_1d_fold_{}.pth'),
    'resnet': os.path.join(MODEL_DIR, 'resnet_1d_fold_{}.pth'),
}

# ==================== 设备配置 ====================
import torch

# 设备选择配置
# 重要提示：如果您的GPU不兼容当前PyTorch版本，请将此项设置为 'cpu'
# 选项: 'auto' - 自动检测, 'cuda' - 强制GPU, 'cpu' - 强制CPU
DEVICE_MODE = 'cpu'  # 默认使用CPU以确保兼容性

def get_device():
    """智能检测并返回可用设备"""
    global DEVICE_MODE
    
    # 检查环境变量
    env_force_cpu = os.environ.get('FORCE_CPU', '').lower() in ('1', 'true', 'yes')
    if env_force_cpu:
        print("💡 环境变量FORCE_CPU已设置，使用CPU模式")
        return torch.device('cpu')
    
    # 强制CPU模式
    if DEVICE_MODE == 'cpu':
        print("💡 配置设定为CPU模式")
        return torch.device('cpu')
    
    # 强制CUDA模式
    if DEVICE_MODE == 'cuda':
        if torch.cuda.is_available():
            print("💡 配置设定为CUDA模式")
            return torch.device('cuda')
        else:
            print("⚠️  警告: CUDA不可用，切换到CPU模式")
            return torch.device('cpu')
    
    # 自动检测模式
    if torch.cuda.is_available():
        try:
            # 尝试在CUDA上创建一个小张量来测试兼容性
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("✅ CUDA兼容性测试通过，使用GPU加速")
            return torch.device('cuda')
        except Exception as e:
            print(f"\n⚠️  警告: CUDA设备不兼容")
            print(f"   原因: {str(e)[:80]}...")
            print("   自动切换到CPU模式")
            print("   提示: 如需使用GPU，请确保PyTorch版本支持您的显卡")
            print(f"   或在config.py中设置 DEVICE_MODE = 'cpu'")
            return torch.device('cpu')
    else:
        print("💡 CUDA不可用，使用CPU模式")
        return torch.device('cpu')

DEVICE = get_device()

# ==================== 可视化配置 ====================
PLOT_CONFIG = {
    'dpi': 150,
    'figsize': (12, 8),
    'save_format': 'png',
}

# ==================== 日志配置 ====================
LOG_CONFIG = {
    'log_interval': 10,  # 每隔多少个batch打印一次日志
    'save_interval': 1,  # 每隔多少个epoch保存一次模型
}

print(f"配置加载完成，使用设备: {DEVICE}")
if DEVICE.type == 'cpu':
    print("💡 提示: 如果您有兼容的GPU，可以在config.py中修改 DEVICE_MODE = 'auto' 来启用GPU加速")

