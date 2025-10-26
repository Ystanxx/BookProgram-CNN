"""
模型包
包含1D-CNN和1D-ResNet18模型定义
"""

from .cnn_1d import CNN1D, create_cnn_model
from .resnet_1d import ResNet1D, BasicBlock1D, create_resnet_model

__all__ = [
    'CNN1D',
    'create_cnn_model',
    'ResNet1D',
    'BasicBlock1D',
    'create_resnet_model',
]

