"""
1D-ResNet18模型定义
基于ResNet18架构的一维版本，用于拉曼谱图多任务学习
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BasicBlock1D(nn.Module):
    """一维基础残差块"""
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        初始化残差块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长
            downsample: 下采样层（用于匹配维度）
        """
        super(BasicBlock1D, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=3,
            stride=stride, 
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv1d(
            out_channels, 
            out_channels, 
            kernel_size=3,
            stride=1, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        """前向传播"""
        identity = x
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet1D(nn.Module):
    """一维ResNet-18网络"""
    
    def __init__(self,
                 input_dim=1015,
                 num_classes=10,
                 num_targets=10,
                 initial_channels=64,
                 block_channels=[64, 128, 256, 512],
                 num_blocks=[2, 2, 2, 2],
                 dropout_rate=0.5):
        """
        初始化1D-ResNet18模型
        
        Args:
            input_dim: 输入维度
            num_classes: 分类类别数
            num_targets: 回归目标数
            initial_channels: 初始卷积层通道数
            block_channels: 各残差层组的通道数
            num_blocks: 各残差层组的块数
            dropout_rate: Dropout比率
        """
        super(ResNet1D, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_targets = num_targets
        self.in_channels = initial_channels
        
        # 初始卷积层
        self.conv1 = nn.Conv1d(
            1, 
            initial_channels, 
            kernel_size=7,
            stride=2, 
            padding=3, 
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(initial_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 残差层组
        self.layer1 = self._make_layer(BasicBlock1D, block_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(BasicBlock1D, block_channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, block_channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, block_channels[3], num_blocks[3], stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc_hidden_dim = block_channels[-1] * BasicBlock1D.expansion
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim // 2),
            nn.BatchNorm1d(self.fc_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(self.fc_hidden_dim // 2, num_classes)
        )
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim // 2),
            nn.BatchNorm1d(self.fc_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(self.fc_hidden_dim // 2, num_targets)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        """
        构建残差层组
        
        Args:
            block: 残差块类型
            out_channels: 输出通道数
            num_blocks: 块的数量
            stride: 第一个块的步长
        
        Returns:
            layers: 残差层组
        """
        downsample = None
        
        # 如果步长不为1或通道数改变，需要下采样
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels * block.expansion)
            )
        
        layers = []
        # 第一个块可能需要下采样
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # 后续块
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
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
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 分类和回归输出
        class_output = self.classifier(x)
        reg_output = self.regressor(x)
        
        return class_output, reg_output
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_name': '1D-ResNet18',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'num_targets': self.num_targets,
            'fc_hidden_dim': self.fc_hidden_dim,
        }
        return info


def create_resnet_model(config_dict=None):
    """
    创建1D-ResNet18模型的工厂函数
    
    Args:
        config_dict: 配置字典，如果为None则使用默认配置
    
    Returns:
        model: 1D-ResNet18模型实例
    """
    if config_dict is None:
        config_dict = config.RESNET_CONFIG
    
    model = ResNet1D(
        input_dim=config.INPUT_DIM,
        num_classes=config.NUM_CLASSES,
        num_targets=config.NUM_TARGETS,
        initial_channels=config_dict['initial_channels'],
        block_channels=config_dict['block_channels'],
        num_blocks=config_dict['num_blocks'],
        dropout_rate=config_dict['dropout_rate']
    )
    
    return model


if __name__ == '__main__':
    # 测试模型
    print("测试1D-ResNet18模型...")
    
    # 创建模型
    model = create_resnet_model()
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

