"""
评估指标计算
包含分类和回归任务的各种评估指标
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, num_classes=10, num_targets=10):
        """
        初始化指标计算器
        
        Args:
            num_classes: 分类类别数
            num_targets: 回归目标数
        """
        self.num_classes = num_classes
        self.num_targets = num_targets
        self.reset()
    
    def reset(self):
        """重置所有累积的预测和标签"""
        self.all_class_preds = []
        self.all_class_labels = []
        self.all_reg_preds = []
        self.all_reg_labels = []
    
    def update(self, class_preds, class_labels, reg_preds, reg_labels):
        """
        更新预测和标签
        
        Args:
            class_preds: 分类预测 (batch_size, num_classes) 或 (batch_size,)
            class_labels: 分类标签 (batch_size,)
            reg_preds: 回归预测 (batch_size, num_targets)
            reg_labels: 回归标签 (batch_size, num_targets)
        """
        # 转换为numpy数组
        if torch.is_tensor(class_preds):
            class_preds = class_preds.cpu().numpy()
        if torch.is_tensor(class_labels):
            class_labels = class_labels.cpu().numpy()
        if torch.is_tensor(reg_preds):
            reg_preds = reg_preds.cpu().numpy()
        if torch.is_tensor(reg_labels):
            reg_labels = reg_labels.cpu().numpy()
        
        # 如果分类预测是logits，转换为类别标签
        if class_preds.ndim > 1:
            class_preds = np.argmax(class_preds, axis=1)
        
        self.all_class_preds.extend(class_preds)
        self.all_class_labels.extend(class_labels)
        self.all_reg_preds.append(reg_preds)
        self.all_reg_labels.append(reg_labels)
    
    def compute_classification_metrics(self):
        """
        计算分类指标
        
        Returns:
            metrics: 包含各种分类指标的字典
        """
        preds = np.array(self.all_class_preds)
        labels = np.array(self.all_class_labels)
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
            'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
            'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
            'precision_weighted': precision_score(labels, preds, average='weighted', zero_division=0),
            'recall_weighted': recall_score(labels, preds, average='weighted', zero_division=0),
            'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
        }
        
        # 混淆矩阵
        cm = confusion_matrix(labels, preds)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def compute_regression_metrics(self):
        """
        计算回归指标
        
        Returns:
            metrics: 包含各种回归指标的字典
        """
        preds = np.vstack(self.all_reg_preds)
        labels = np.vstack(self.all_reg_labels)
        
        # 计算整体指标
        mae = mean_absolute_error(labels, preds)
        mse = mean_squared_error(labels, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(labels, preds)
        
        # 计算每个目标的指标
        mae_per_target = []
        rmse_per_target = []
        r2_per_target = []
        
        for i in range(self.num_targets):
            mae_i = mean_absolute_error(labels[:, i], preds[:, i])
            mse_i = mean_squared_error(labels[:, i], preds[:, i])
            rmse_i = np.sqrt(mse_i)
            r2_i = r2_score(labels[:, i], preds[:, i])
            
            mae_per_target.append(mae_i)
            rmse_per_target.append(rmse_i)
            r2_per_target.append(r2_i)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae_per_target': np.array(mae_per_target),
            'rmse_per_target': np.array(rmse_per_target),
            'r2_per_target': np.array(r2_per_target),
        }
        
        return metrics
    
    def compute_all_metrics(self):
        """
        计算所有指标（分类+回归）
        
        Returns:
            metrics: 包含所有指标的字典
        """
        class_metrics = self.compute_classification_metrics()
        reg_metrics = self.compute_regression_metrics()
        
        metrics = {
            'classification': class_metrics,
            'regression': reg_metrics
        }
        
        return metrics
    
    def print_metrics(self, metrics=None):
        """
        打印指标
        
        Args:
            metrics: 指标字典，如果为None则计算并打印
        """
        if metrics is None:
            metrics = self.compute_all_metrics()
        
        print("\n" + "=" * 60)
        print("分类指标:")
        print("=" * 60)
        class_metrics = metrics['classification']
        print(f"准确率 (Accuracy):          {class_metrics['accuracy']:.4f}")
        print(f"精确率 (Precision-Macro):   {class_metrics['precision_macro']:.4f}")
        print(f"召回率 (Recall-Macro):      {class_metrics['recall_macro']:.4f}")
        print(f"F1分数 (F1-Macro):          {class_metrics['f1_macro']:.4f}")
        print(f"精确率 (Precision-Weighted): {class_metrics['precision_weighted']:.4f}")
        print(f"召回率 (Recall-Weighted):   {class_metrics['recall_weighted']:.4f}")
        print(f"F1分数 (F1-Weighted):       {class_metrics['f1_weighted']:.4f}")
        
        print("\n" + "=" * 60)
        print("回归指标:")
        print("=" * 60)
        reg_metrics = metrics['regression']
        print(f"平均绝对误差 (MAE):   {reg_metrics['mae']:.4f}")
        print(f"均方误差 (MSE):        {reg_metrics['mse']:.4f}")
        print(f"均方根误差 (RMSE):     {reg_metrics['rmse']:.4f}")
        print(f"决定系数 (R²):         {reg_metrics['r2']:.4f}")
        
        print("\n每个化学成分的指标:")
        print("-" * 60)
        for i, target_name in enumerate(config.TARGET_NAMES):
            print(f"{target_name:20s} | MAE: {reg_metrics['mae_per_target'][i]:.3f} | "
                  f"RMSE: {reg_metrics['rmse_per_target'][i]:.3f} | "
                  f"R²: {reg_metrics['r2_per_target'][i]:.3f}")
        print("=" * 60)


def calculate_loss(class_output, reg_output, class_labels, reg_labels, 
                   class_criterion, reg_criterion,
                   class_weight=1.0, reg_weight=1.0):
    """
    计算多任务损失
    
    Args:
        class_output: 分类输出
        reg_output: 回归输出
        class_labels: 分类标签
        reg_labels: 回归标签
        class_criterion: 分类损失函数
        reg_criterion: 回归损失函数
        class_weight: 分类损失权重
        reg_weight: 回归损失权重
    
    Returns:
        total_loss: 总损失
        class_loss: 分类损失
        reg_loss: 回归损失
    """
    class_loss = class_criterion(class_output, class_labels)
    reg_loss = reg_criterion(reg_output, reg_labels)
    
    total_loss = class_weight * class_loss + reg_weight * reg_loss
    
    return total_loss, class_loss, reg_loss


if __name__ == '__main__':
    # 测试指标计算器
    print("测试指标计算器...")
    
    calculator = MetricsCalculator(num_classes=10, num_targets=10)
    
    # 模拟一些预测和标签
    np.random.seed(42)
    for _ in range(5):
        batch_size = 32
        class_preds = torch.randn(batch_size, 10)
        class_labels = torch.randint(0, 10, (batch_size,))
        reg_preds = torch.randn(batch_size, 10) * 50 + 50
        reg_labels = torch.randn(batch_size, 10) * 50 + 50
        
        calculator.update(class_preds, class_labels, reg_preds, reg_labels)
    
    # 计算并打印指标
    metrics = calculator.compute_all_metrics()
    calculator.print_metrics(metrics)

