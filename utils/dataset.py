"""
PyTorch数据集类定义
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class RamanDataset(Dataset):
    """拉曼谱图数据集类"""
    
    def __init__(self, spectra, stages, concentrations, transform=None):
        """
        初始化数据集
        
        Args:
            spectra: 拉曼谱图数据 (N, input_dim)
            stages: 发酵阶段标签 (N,)
            concentrations: 化学成分浓度 (N, num_targets)
            transform: 数据变换函数
        """
        self.spectra = torch.FloatTensor(spectra)
        self.stages = torch.LongTensor(stages)
        self.concentrations = torch.FloatTensor(concentrations)
        self.transform = transform
        
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            spectrum: (1, input_dim) - 添加通道维度
            stage: 发酵阶段标签
            concentration: (num_targets,) - 浓度向量
        """
        spectrum = self.spectra[idx]
        stage = self.stages[idx]
        concentration = self.concentrations[idx]
        
        # 为1D卷积添加通道维度
        spectrum = spectrum.unsqueeze(0)  # (input_dim,) -> (1, input_dim)
        
        if self.transform:
            spectrum = self.transform(spectrum)
        
        return spectrum, stage, concentration


class RamanDataModule:
    """数据模块，管理数据加载和预处理"""
    
    def __init__(self, data_dir, batch_size=32, num_workers=0):
        """
        初始化数据模块
        
        Args:
            data_dir: 数据目录
            batch_size: 批次大小
            num_workers: 数据加载工作进程数
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scaler = StandardScaler()
        
        # 加载数据
        self.load_data()
        
    def load_data(self):
        """加载数据文件"""
        from data.data_generator import load_or_generate_data
        
        self.spectra, self.stages, self.concentrations = load_or_generate_data(
            self.data_dir,
            num_samples=config.DATA_CONFIG['num_samples'],
            regenerate=False
        )
        
        print(f"\n数据加载完成:")
        print(f"  谱图形状: {self.spectra.shape}")
        print(f"  阶段标签形状: {self.stages.shape}")
        print(f"  浓度标签形状: {self.concentrations.shape}")
    
    def normalize_spectra(self, train_spectra, val_spectra=None):
        """
        标准化谱图数据
        
        Args:
            train_spectra: 训练集谱图
            val_spectra: 验证集谱图（可选）
        
        Returns:
            标准化后的训练集和验证集
        """
        # 在训练集上拟合scaler
        train_spectra_normalized = self.scaler.fit_transform(train_spectra)
        
        if val_spectra is not None:
            val_spectra_normalized = self.scaler.transform(val_spectra)
            return train_spectra_normalized, val_spectra_normalized
        
        return train_spectra_normalized
    
    def get_k_fold_dataloaders(self, k_folds=5, random_state=42):
        """
        生成K折交叉验证的数据加载器
        
        Args:
            k_folds: 折数
            random_state: 随机种子
        
        Yields:
            fold: 折数索引
            train_loader: 训练集加载器
            val_loader: 验证集加载器
        """
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.spectra)):
            # 划分训练集和验证集
            train_spectra = self.spectra[train_idx]
            val_spectra = self.spectra[val_idx]
            train_stages = self.stages[train_idx]
            val_stages = self.stages[val_idx]
            train_concentrations = self.concentrations[train_idx]
            val_concentrations = self.concentrations[val_idx]
            
            # 标准化数据
            train_spectra_norm, val_spectra_norm = self.normalize_spectra(
                train_spectra, val_spectra
            )
            
            # 创建数据集
            train_dataset = RamanDataset(
                train_spectra_norm, 
                train_stages, 
                train_concentrations
            )
            val_dataset = RamanDataset(
                val_spectra_norm, 
                val_stages, 
                val_concentrations
            )
            
            # 创建数据加载器
            # 只有在实际使用GPU时才启用pin_memory
            use_pin_memory = (config.DEVICE.type == 'cuda')
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=use_pin_memory
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=use_pin_memory
            )
            
            print(f"\n折 {fold + 1}/{k_folds}:")
            print(f"  训练集大小: {len(train_dataset)}")
            print(f"  验证集大小: {len(val_dataset)}")
            
            yield fold, train_loader, val_loader
    
    def get_train_val_test_dataloaders(self, train_ratio=0.8, val_ratio=0.1, random_state=42):
        """
        获取训练集、验证集和测试集的数据加载器
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            random_state: 随机种子
        
        Returns:
            train_loader, val_loader, test_loader
        """
        np.random.seed(random_state)
        n_samples = len(self.spectra)
        indices = np.random.permutation(n_samples)
        
        # 计算划分点
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        # 划分数据
        train_spectra = self.spectra[train_idx]
        val_spectra = self.spectra[val_idx]
        test_spectra = self.spectra[test_idx]
        
        train_stages = self.stages[train_idx]
        val_stages = self.stages[val_idx]
        test_stages = self.stages[test_idx]
        
        train_concentrations = self.concentrations[train_idx]
        val_concentrations = self.concentrations[val_idx]
        test_concentrations = self.concentrations[test_idx]
        
        # 标准化
        train_spectra_norm = self.scaler.fit_transform(train_spectra)
        val_spectra_norm = self.scaler.transform(val_spectra)
        test_spectra_norm = self.scaler.transform(test_spectra)
        
        # 创建数据集
        train_dataset = RamanDataset(train_spectra_norm, train_stages, train_concentrations)
        val_dataset = RamanDataset(val_spectra_norm, val_stages, val_concentrations)
        test_dataset = RamanDataset(test_spectra_norm, test_stages, test_concentrations)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers
        )
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  验证集: {len(val_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        
        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试数据集类
    data_module = RamanDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE
    )
    
    # 测试单次划分
    train_loader, val_loader, test_loader = data_module.get_train_val_test_dataloaders()
    
    # 测试一个batch
    for spectra, stages, concentrations in train_loader:
        print(f"\nBatch测试:")
        print(f"  谱图形状: {spectra.shape}")
        print(f"  阶段标签形状: {stages.shape}")
        print(f"  浓度形状: {concentrations.shape}")
        break

