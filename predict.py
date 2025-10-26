"""
预测脚本
加载训练好的模型进行预测和推理
"""
import torch
import numpy as np
import os
import sys
import json
from datetime import datetime

# 导入项目模块
import config
from models.cnn_1d import create_cnn_model
from models.resnet_1d import create_resnet_model
from utils.dataset import RamanDataModule
from utils.metrics import MetricsCalculator


class Predictor:
    """预测器类"""
    
    def __init__(self, model_name='cnn', fold=0, device=None):
        """
        初始化预测器
        
        Args:
            model_name: 模型名称 ('cnn' 或 'resnet')
            fold: 使用哪一折的模型 (0-4)
            device: 设备
        """
        self.model_name = model_name
        self.fold = fold
        self.device = device if device else config.DEVICE
        
        # 创建模型
        if model_name == 'cnn':
            self.model = create_cnn_model()
        elif model_name == 'resnet':
            self.model = create_resnet_model()
        else:
            raise ValueError(f"未知的模型名称: {model_name}")
        
        # 加载模型权重
        self.load_model()
        self.model.to(self.device)
        self.model.eval()
    
    def load_model(self):
        """加载模型权重"""
        model_path = config.MODEL_SAVE_PATH[self.model_name].format(self.fold)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"成功加载模型: {model_path}")
        print(f"模型训练轮数: {checkpoint['epoch'] + 1}")
        print(f"验证损失: {checkpoint['val_loss']:.4f}")
        
        # 打印模型性能
        val_metrics = checkpoint['val_metrics']
        print(f"分类准确率: {val_metrics['classification']['accuracy']:.4f}")
        print(f"回归R²: {val_metrics['regression']['r2']:.4f}")
    
    def predict_single(self, spectrum):
        """
        预测单个谱图
        
        Args:
            spectrum: 拉曼谱图 (input_dim,) 或 (1, input_dim)
        
        Returns:
            stage_pred: 预测的发酵阶段
            stage_prob: 各阶段的概率分布
            concentrations_pred: 预测的化学成分浓度
        """
        # 确保输入形状正确
        if isinstance(spectrum, np.ndarray):
            spectrum = torch.FloatTensor(spectrum)
        
        if spectrum.dim() == 1:
            spectrum = spectrum.unsqueeze(0).unsqueeze(0)  # (input_dim,) -> (1, 1, input_dim)
        elif spectrum.dim() == 2:
            spectrum = spectrum.unsqueeze(0)  # (1, input_dim) -> (1, 1, input_dim)
        
        spectrum = spectrum.to(self.device)
        
        with torch.no_grad():
            class_output, reg_output = self.model(spectrum)
            
            # 分类预测
            stage_prob = torch.softmax(class_output, dim=1)
            stage_pred = torch.argmax(stage_prob, dim=1)
            
            # 回归预测
            concentrations_pred = reg_output
        
        # 转换为numpy
        stage_pred = stage_pred.cpu().numpy()[0]
        stage_prob = stage_prob.cpu().numpy()[0]
        concentrations_pred = concentrations_pred.cpu().numpy()[0]
        
        return stage_pred, stage_prob, concentrations_pred
    
    def predict_batch(self, spectra):
        """
        批量预测
        
        Args:
            spectra: 拉曼谱图批次 (batch_size, input_dim) 或 (batch_size, 1, input_dim)
        
        Returns:
            stage_preds: 预测的发酵阶段
            stage_probs: 各阶段的概率分布
            concentrations_preds: 预测的化学成分浓度
        """
        if isinstance(spectra, np.ndarray):
            spectra = torch.FloatTensor(spectra)
        
        if spectra.dim() == 2:
            spectra = spectra.unsqueeze(1)  # (batch_size, input_dim) -> (batch_size, 1, input_dim)
        
        spectra = spectra.to(self.device)
        
        with torch.no_grad():
            class_output, reg_output = self.model(spectra)
            
            # 分类预测
            stage_probs = torch.softmax(class_output, dim=1)
            stage_preds = torch.argmax(stage_probs, dim=1)
            
            # 回归预测
            concentrations_preds = reg_output
        
        # 转换为numpy
        stage_preds = stage_preds.cpu().numpy()
        stage_probs = stage_probs.cpu().numpy()
        concentrations_preds = concentrations_preds.cpu().numpy()
        
        return stage_preds, stage_probs, concentrations_preds
    
    def print_prediction(self, stage_pred, stage_prob, concentrations_pred):
        """
        打印预测结果
        
        Args:
            stage_pred: 预测的阶段
            stage_prob: 阶段概率
            concentrations_pred: 浓度预测
        """
        print("\n" + "=" * 80)
        print("预测结果")
        print("=" * 80)
        
        # 分类结果
        print(f"\n发酵阶段预测: {config.CLASS_NAMES[stage_pred]} (置信度: {stage_prob[stage_pred]:.2%})")
        
        print("\n各阶段概率分布:")
        for i, (class_name, prob) in enumerate(zip(config.CLASS_NAMES, stage_prob)):
            bar = '█' * int(prob * 50)
            print(f"  {class_name:10s} | {bar:50s} {prob:.2%}")
        
        # 回归结果
        print("\n化学成分浓度预测 (mg/L):")
        print("-" * 80)
        for target_name, concentration in zip(config.TARGET_NAMES, concentrations_pred):
            print(f"  {target_name:20s}: {concentration:6.2f} mg/L")
        print("=" * 80)


def ensemble_predict(model_names=['cnn', 'resnet'], spectrum=None, k_folds=5):
    """
    集成预测：使用多个模型和多折进行预测
    
    Args:
        model_names: 模型名称列表
        spectrum: 输入谱图
        k_folds: 折数
    
    Returns:
        ensemble_stage: 集成阶段预测
        ensemble_concentrations: 集成浓度预测
    """
    if spectrum is None:
        # 生成一个示例谱图
        print("未提供输入，使用模拟数据进行演示...")
        from data.data_generator import RamanDataGenerator
        generator = RamanDataGenerator(num_samples=1)
        spectra, _, _ = generator.generate_dataset()
        spectrum = spectra[0]
    
    all_stage_probs = []
    all_concentrations = []
    
    print(f"\n使用 {len(model_names)} 个模型和 {k_folds} 折进行集成预测...")
    
    for model_name in model_names:
        for fold in range(k_folds):
            try:
                predictor = Predictor(model_name=model_name, fold=fold)
                stage_pred, stage_prob, concentrations_pred = predictor.predict_single(spectrum)
                
                all_stage_probs.append(stage_prob)
                all_concentrations.append(concentrations_pred)
                
                print(f"✓ {model_name.upper()} Fold {fold + 1}: 阶段 {stage_pred}, 置信度 {stage_prob[stage_pred]:.2%}")
            except FileNotFoundError:
                print(f"✗ {model_name.upper()} Fold {fold + 1}: 模型文件未找到，跳过")
    
    if not all_stage_probs:
        raise ValueError("没有可用的模型进行预测")
    
    # 集成预测
    # 转换为numpy数组以确保类型一致
    all_stage_probs = np.array(all_stage_probs)
    all_concentrations = np.array(all_concentrations)
    
    avg_stage_prob = np.mean(all_stage_probs, axis=0)
    ensemble_stage = np.argmax(avg_stage_prob)
    ensemble_concentrations = np.mean(all_concentrations, axis=0)
    
    # 打印集成结果
    print("\n" + "=" * 80)
    print("集成预测结果")
    print("=" * 80)
    print(f"\n发酵阶段: {config.CLASS_NAMES[ensemble_stage]} (置信度: {avg_stage_prob[ensemble_stage]:.2%})")
    
    print("\n化学成分浓度 (mg/L):")
    print("-" * 80)
    for i, (target_name, concentration) in enumerate(zip(config.TARGET_NAMES, ensemble_concentrations)):
        # 计算每个化学成分的标准差
        std = np.std(all_concentrations[:, i])
        print(f"  {target_name:20s}: {concentration:6.2f} ± {std:5.2f} mg/L")
    print("=" * 80)
    
    return ensemble_stage, ensemble_concentrations


def evaluate_test_set(model_name='cnn', fold=0):
    """
    在测试集上评估模型性能
    
    Args:
        model_name: 模型名称
        fold: 使用哪一折的模型
    """
    print(f"\n在测试集上评估 {model_name.upper()} 模型 (Fold {fold})...")
    
    # 加载预测器
    predictor = Predictor(model_name=model_name, fold=fold)
    
    # 准备数据
    data_module = RamanDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE
    )
    
    # 获取测试集
    _, _, test_loader = data_module.get_train_val_test_dataloaders()
    
    # 评估
    calculator = MetricsCalculator(config.NUM_CLASSES, config.NUM_TARGETS)
    
    print("正在预测测试集...")
    for spectra, stages, concentrations in test_loader:
        stage_preds, _, concentration_preds = predictor.predict_batch(spectra)
        calculator.update(stage_preds, stages, concentration_preds, concentrations)
    
    # 计算并打印指标
    metrics = calculator.compute_all_metrics()
    calculator.print_metrics(metrics)
    
    return metrics


if __name__ == '__main__':
    print("=" * 80)
    print("拉曼谱图多任务学习 - 预测程序")
    print("=" * 80)
    
    # 示例1: 单个样本预测
    print("\n" + "=" * 80)
    print("示例 1: 单个样本预测")
    print("=" * 80)
    
    # 生成测试样本
    from data.data_generator import RamanDataGenerator
    generator = RamanDataGenerator(num_samples=1)
    test_spectra, test_stages, test_concentrations = generator.generate_dataset()
    test_spectrum = test_spectra[0]
    
    print(f"\n真实标签:")
    print(f"  发酵阶段: {config.CLASS_NAMES[test_stages[0]]}")
    print(f"  化学成分浓度: {test_concentrations[0].round(2)}")
    
    # 使用CNN模型预测
    print("\n使用 1D-CNN 模型预测:")
    try:
        predictor_cnn = Predictor(model_name='cnn', fold=0)
        stage_pred, stage_prob, conc_pred = predictor_cnn.predict_single(test_spectrum)
        predictor_cnn.print_prediction(stage_pred, stage_prob, conc_pred)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 train.py 训练模型")
    
    # 示例2: 集成预测
    print("\n" + "=" * 80)
    print("示例 2: 集成预测 (多模型多折)")
    print("=" * 80)
    
    try:
        ensemble_predict(model_names=['cnn', 'resnet'], spectrum=test_spectrum, k_folds=5)
    except Exception as e:
        print(f"错误: {e}")
        print("某些模型文件可能未找到")
    
    # 示例3: 测试集评估
    print("\n" + "=" * 80)
    print("示例 3: 测试集评估")
    print("=" * 80)
    
    try:
        evaluate_test_set(model_name='cnn', fold=0)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 train.py 训练模型")

