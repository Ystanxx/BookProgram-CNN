"""
训练脚本
实现五折交叉验证的多任务学习训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题
import matplotlib.pyplot as plt
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import os
import time
import json
from datetime import datetime

# 导入项目模块
import config
from models.cnn_1d import create_cnn_model
from models.resnet_1d import create_resnet_model
from utils.dataset import RamanDataModule
from utils.metrics import MetricsCalculator, calculate_loss


class Trainer:
    """训练器类"""
    
    def __init__(self, model, model_name, device=None):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            model_name: 模型名称 ('cnn' 或 'resnet')
            device: 设备
        """
        self.model = model
        self.model_name = model_name
        self.device = device if device else config.DEVICE
        self.model.to(self.device)
        
        # 损失函数
        self.class_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.SmoothL1Loss()  # 对异常值更鲁棒
        
        # 损失权重
        self.class_weight = config.LOSS_WEIGHTS['classification']
        self.reg_weight = config.LOSS_WEIGHTS['regression']
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_class_loss': [],
            'train_reg_loss': [],
            'val_loss': [],
            'val_class_loss': [],
            'val_reg_loss': [],
            'val_accuracy': [],
            'val_r2': [],
        }
        
        # 最佳模型追踪
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self, train_loader, optimizer):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
        
        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0.0
        total_class_loss = 0.0
        total_reg_loss = 0.0
        num_batches = 0
        
        for batch_idx, (spectra, stages, concentrations) in enumerate(train_loader):
            # 数据移到设备
            spectra = spectra.to(self.device)
            stages = stages.to(self.device)
            concentrations = concentrations.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            class_output, reg_output = self.model(spectra)
            
            # 计算损失
            loss, class_loss, reg_loss = calculate_loss(
                class_output, reg_output, stages, concentrations,
                self.class_criterion, self.reg_criterion,
                self.class_weight, self.reg_weight
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 累积损失
            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_reg_loss += reg_loss.item()
            num_batches += 1
            
            # 打印日志
            if (batch_idx + 1) % config.LOG_CONFIG['log_interval'] == 0:
                print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] | "
                      f"Loss: {loss.item():.4f} | "
                      f"Class: {class_loss.item():.4f} | "
                      f"Reg: {reg_loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        avg_class_loss = total_class_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches
        
        return avg_loss, avg_class_loss, avg_reg_loss
    
    def validate_epoch(self, val_loader):
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            平均损失和指标
        """
        self.model.eval()
        total_loss = 0.0
        total_class_loss = 0.0
        total_reg_loss = 0.0
        num_batches = 0
        
        # 指标计算器
        calculator = MetricsCalculator(config.NUM_CLASSES, config.NUM_TARGETS)
        
        with torch.no_grad():
            for spectra, stages, concentrations in val_loader:
                # 数据移到设备
                spectra = spectra.to(self.device)
                stages = stages.to(self.device)
                concentrations = concentrations.to(self.device)
                
                # 前向传播
                class_output, reg_output = self.model(spectra)
                
                # 计算损失
                loss, class_loss, reg_loss = calculate_loss(
                    class_output, reg_output, stages, concentrations,
                    self.class_criterion, self.reg_criterion,
                    self.class_weight, self.reg_weight
                )
                
                # 累积损失
                total_loss += loss.item()
                total_class_loss += class_loss.item()
                total_reg_loss += reg_loss.item()
                num_batches += 1
                
                # 更新指标
                calculator.update(class_output, stages, reg_output, concentrations)
        
        avg_loss = total_loss / num_batches
        avg_class_loss = total_class_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches
        
        # 计算指标
        metrics = calculator.compute_all_metrics()
        
        return avg_loss, avg_class_loss, avg_reg_loss, metrics
    
    def train(self, train_loader, val_loader, epochs, fold=0):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            fold: 折数索引
        
        Returns:
            训练历史
        """
        print(f"\n{'=' * 80}")
        print(f"开始训练 {self.model_name.upper()} 模型 - 折 {fold + 1}")
        print(f"{'=' * 80}")
        
        # 优化器
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.LR_SCHEDULER['factor'],
            patience=config.LR_SCHEDULER['patience'],
            min_lr=config.LR_SCHEDULER['min_lr'],
            verbose=True
        )
        
        # 重置历史和最佳模型追踪
        self.history = {key: [] for key in self.history.keys()}
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            print(f"\nEpoch [{epoch + 1}/{epochs}]")
            print("-" * 80)
            
            # 训练
            train_loss, train_class_loss, train_reg_loss = self.train_epoch(
                train_loader, optimizer
            )
            
            # 验证
            val_loss, val_class_loss, val_reg_loss, val_metrics = self.validate_epoch(
                val_loader
            )
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_class_loss'].append(train_class_loss)
            self.history['train_reg_loss'].append(train_reg_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_class_loss'].append(val_class_loss)
            self.history['val_reg_loss'].append(val_reg_loss)
            self.history['val_accuracy'].append(val_metrics['classification']['accuracy'])
            self.history['val_r2'].append(val_metrics['regression']['r2'])
            
            epoch_time = time.time() - epoch_start_time
            
            # 打印结果
            print(f"\n训练 | Loss: {train_loss:.4f} | "
                  f"Class: {train_class_loss:.4f} | Reg: {train_reg_loss:.4f}")
            print(f"验证 | Loss: {val_loss:.4f} | "
                  f"Class: {val_class_loss:.4f} | Reg: {val_reg_loss:.4f}")
            print(f"验证 | Accuracy: {val_metrics['classification']['accuracy']:.4f} | "
                  f"R²: {val_metrics['regression']['r2']:.4f}")
            print(f"时间: {epoch_time:.2f}s | "
                  f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # 保存模型
                model_path = config.MODEL_SAVE_PATH[self.model_name].format(fold)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'history': self.history,
                }, model_path)
                
                print(f"✓ 保存最佳模型到: {model_path}")
            else:
                self.patience_counter += 1
                print(f"✗ 验证损失未改善 ({self.patience_counter}/{config.EARLY_STOPPING_PATIENCE})")
            
            # 早停
            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n早停！最佳epoch: {self.best_epoch + 1}")
                break
        
        total_time = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"训练完成！总时间: {total_time / 60:.2f} 分钟")
        print(f"最佳验证损失: {self.best_val_loss:.4f} (Epoch {self.best_epoch + 1})")
        print(f"{'=' * 80}")
        
        return self.history


def train_with_k_fold(model_name='cnn', k_folds=5):
    """
    使用K折交叉验证训练模型
    
    Args:
        model_name: 模型名称 ('cnn' 或 'resnet')
        k_folds: 折数
    """
    print(f"\n开始 {k_folds} 折交叉验证训练 - {model_name.upper()} 模型")
    print(f"=" * 100)
    
    # 准备数据
    data_module = RamanDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE
    )
    
    # 存储所有折的历史
    all_fold_histories = []
    all_fold_metrics = []
    
    # K折交叉验证
    for fold, train_loader, val_loader in data_module.get_k_fold_dataloaders(k_folds):
        # 创建模型
        if model_name == 'cnn':
            model = create_cnn_model()
        elif model_name == 'resnet':
            model = create_resnet_model()
        else:
            raise ValueError(f"未知的模型名称: {model_name}")
        
        # 打印模型信息
        if fold == 0:
            model_info = model.get_model_info()
            print(f"\n模型信息:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
        
        # 创建训练器
        trainer = Trainer(model, model_name)
        
        # 训练
        history = trainer.train(train_loader, val_loader, config.EPOCHS, fold)
        
        # 加载最佳模型进行最终验证
        model_path = config.MODEL_SAVE_PATH[model_name].format(fold)
        checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 最终验证
        trainer.model = model
        val_loss, val_class_loss, val_reg_loss, val_metrics = trainer.validate_epoch(val_loader)
        
        print(f"\n折 {fold + 1} 最终验证结果:")
        print(f"验证损失: {val_loss:.4f}")
        print(f"分类准确率: {val_metrics['classification']['accuracy']:.4f}")
        print(f"回归R²: {val_metrics['regression']['r2']:.4f}")
        
        all_fold_histories.append(history)
        all_fold_metrics.append(val_metrics)
    
    # 计算平均性能
    print(f"\n{'=' * 100}")
    print(f"{k_folds}折交叉验证平均结果:")
    print(f"{'=' * 100}")
    
    avg_accuracy = np.mean([m['classification']['accuracy'] for m in all_fold_metrics])
    avg_f1 = np.mean([m['classification']['f1_macro'] for m in all_fold_metrics])
    avg_r2 = np.mean([m['regression']['r2'] for m in all_fold_metrics])
    avg_mae = np.mean([m['regression']['mae'] for m in all_fold_metrics])
    avg_rmse = np.mean([m['regression']['rmse'] for m in all_fold_metrics])
    
    std_accuracy = np.std([m['classification']['accuracy'] for m in all_fold_metrics])
    std_r2 = np.std([m['regression']['r2'] for m in all_fold_metrics])
    
    print(f"\n分类指标:")
    print(f"  准确率: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  F1分数: {avg_f1:.4f}")
    
    print(f"\n回归指标:")
    print(f"  R²: {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"  MAE: {avg_mae:.4f}")
    print(f"  RMSE: {avg_rmse:.4f}")
    
    # 保存结果
    results = {
        'model_name': model_name,
        'k_folds': k_folds,
        'avg_metrics': {
            'accuracy': float(avg_accuracy),
            'f1_macro': float(avg_f1),
            'r2': float(avg_r2),
            'mae': float(avg_mae),
            'rmse': float(avg_rmse),
        },
        'std_metrics': {
            'accuracy': float(std_accuracy),
            'r2': float(std_r2),
        },
        'fold_metrics': [
            {
                'accuracy': float(m['classification']['accuracy']),
                'f1_macro': float(m['classification']['f1_macro']),
                'r2': float(m['regression']['r2']),
                'mae': float(m['regression']['mae']),
                'rmse': float(m['regression']['rmse']),
            }
            for m in all_fold_metrics
        ],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 保存结果到JSON
    result_path = os.path.join(config.RESULT_DIR, f'{model_name}_kfold_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n结果已保存到: {result_path}")
    
    # 绘制学习曲线
    plot_learning_curves(all_fold_histories, model_name)
    
    return results


def plot_learning_curves(histories, model_name):
    """
    绘制学习曲线
    
    Args:
        histories: 所有折的训练历史列表
        model_name: 模型名称
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 损失曲线
    ax = axes[0, 0]
    for i, history in enumerate(histories):
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], label=f'Fold {i+1} Train', alpha=0.3)
        ax.plot(epochs, history['val_loss'], label=f'Fold {i+1} Val', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('总损失')
    ax.legend()
    ax.grid(True)
    
    # 分类损失
    ax = axes[0, 1]
    for i, history in enumerate(histories):
        epochs = range(1, len(history['train_class_loss']) + 1)
        ax.plot(epochs, history['train_class_loss'], label=f'Fold {i+1} Train', alpha=0.3)
        ax.plot(epochs, history['val_class_loss'], label=f'Fold {i+1} Val', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('分类损失')
    ax.legend()
    ax.grid(True)
    
    # 回归损失
    ax = axes[1, 0]
    for i, history in enumerate(histories):
        epochs = range(1, len(history['train_reg_loss']) + 1)
        ax.plot(epochs, history['train_reg_loss'], label=f'Fold {i+1} Train', alpha=0.3)
        ax.plot(epochs, history['val_reg_loss'], label=f'Fold {i+1} Val', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('回归损失')
    ax.legend()
    ax.grid(True)
    
    # 准确率和R²
    ax = axes[1, 1]
    for i, history in enumerate(histories):
        epochs = range(1, len(history['val_accuracy']) + 1)
        ax.plot(epochs, history['val_accuracy'], label=f'Fold {i+1} Acc', alpha=0.5)
        ax.plot(epochs, history['val_r2'], label=f'Fold {i+1} R²', alpha=0.5, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric')
    ax.set_title('验证准确率 & R²')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(config.RESULT_DIR, f'{model_name}_learning_curves.png')
    plt.savefig(plot_path, dpi=config.PLOT_CONFIG['dpi'])
    print(f"学习曲线已保存到: {plot_path}")
    plt.close()


if __name__ == '__main__':
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("=" * 100)
    print("拉曼谱图多任务学习 - 训练程序")
    print("=" * 100)
    
    # 训练CNN模型
    print("\n开始训练 1D-CNN 模型...")
    cnn_results = train_with_k_fold(model_name='cnn', k_folds=config.K_FOLDS)
    
    # 训练ResNet模型
    print("\n开始训练 1D-ResNet18 模型...")
    resnet_results = train_with_k_fold(model_name='resnet', k_folds=config.K_FOLDS)
    
    print("\n" + "=" * 100)
    print("所有模型训练完成！")
    print("=" * 100)

