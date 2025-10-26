"""
1D-CNN模型定义
用于拉曼谱图的多任务学习（分类+回归）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CNN1D(nn.Module):
    """一维卷积神经网络"""
    
    def __init__(self, 
                 input_dim=1015,
                 num_classes=10,
                 num_targets=10,
                 conv_channels=[32, 64, 128, 256],
                 kernel_sizes=[7, 5, 3, 3],
                 pool_sizes=[2, 2, 2, 2],
                 dropout_rate=0.5,
                 fc_hidden_dim=512):
        """
        初始化1D-CNN模型
        
        Args:
            input_dim: 输入维度
            num_classes: 分类类别数
            num_targets: 回归目标数
            conv_channels: 各卷积层的通道数列表
            kernel_sizes: 各卷积层的卷积核大小列表
            pool_sizes: 各池化层的大小列表
            dropout_rate: Dropout比率
            fc_hidden_dim: 全连接层隐藏维度
        """
        super(CNN1D, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_targets = num_targets
        
        # 构建卷积层
        self.conv_layers = nn.ModuleList()
        in_channels = 1  # 输入通道数（单通道谱图）
        current_length = input_dim
        
        for i, (out_channels, kernel_size, pool_size) in enumerate(
            zip(conv_channels, kernel_sizes, pool_sizes)
        ):
            # 卷积层 + BatchNorm + ReLU + MaxPooling
            conv_block = nn.Sequential(
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # 保持尺寸
                    bias=False
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
            )
            self.conv_layers.append(conv_block)
            
            in_channels = out_channels
            current_length = current_length // pool_size
        
        # 计算展平后的特征维度
        self.flatten_dim = conv_channels[-1] * current_length
        
        # 共享的全连接层
        self.fc_shared = nn.Sequential(
            nn.Linear(self.flatten_dim, fc_hidden_dim),
            nn.BatchNorm1d(fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim // 2, num_classes)
        )
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim // 2, num_targets)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, 1, input_dim)
        
        Returns:
            class_output: 分类输出 (batch_size, num_classes)
            reg_output: 回归输出 (batch_size, num_targets)
        """
        # 卷积特征提取
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 共享特征
        x = self.fc_shared(x)
        
        # 分类和回归输出
        class_output = self.classifier(x)
        reg_output = self.regressor(x)
        
        return class_output, reg_output
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_name': '1D-CNN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'num_targets': self.num_targets,
            'flatten_dim': self.flatten_dim,
        }
        return info


def create_cnn_model(config_dict=None):
    """
    创建1D-CNN模型的工厂函数
    
    Args:
        config_dict: 配置字典，如果为None则使用默认配置
    
    Returns:
        model: 1D-CNN模型实例
    """
    if config_dict is None:
        config_dict = config.CNN_CONFIG
    
    model = CNN1D(
        input_dim=config.INPUT_DIM,
        num_classes=config.NUM_CLASSES,
        num_targets=config.NUM_TARGETS,
        conv_channels=config_dict['conv_channels'],
        kernel_sizes=config_dict['kernel_sizes'],
        pool_sizes=config_dict['pool_sizes'],
        dropout_rate=config_dict['dropout_rate'],
        fc_hidden_dim=config_dict['fc_hidden_dim']
    )
    
    return model


if __name__ == '__main__':
    # 测试模型
    print("测试1D-CNN模型...")
    
    # 创建模型
    model = create_cnn_model()
    model.eval()
    
    # 打印模型信息
    info = model.get_model_info()
    print("\n模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试前向传播
    batch_size = 4
    test_input = torch.randn(batch_size, 1, config.INPUT_DIM)
    
    with torch.no_grad():
        class_output, reg_output = model(test_input)
    
    print(f"\n前向传播测试:")
    print(f"  输入形状: {test_input.shape}")
    print(f"  分类输出形状: {class_output.shape}")
    print(f"  回归输出形状: {reg_output.shape}")
    
    print("\n模型结构:")
    print(model)

