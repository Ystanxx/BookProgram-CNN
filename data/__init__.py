"""
数据包
包含数据生成和加载功能
"""

from .data_generator import RamanDataGenerator, load_or_generate_data

__all__ = [
    'RamanDataGenerator',
    'load_or_generate_data',
]

