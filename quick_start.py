"""
快速开始示例脚本
演示如何使用本项目进行拉曼谱图分析
"""
import torch
import numpy as np
import os

# 导入项目模块
import config
from data.data_generator import RamanDataGenerator
from models.cnn_1d import create_cnn_model
from models.resnet_1d import create_resnet_model


def demo_data_generation():
    """演示数据生成"""
    print("=" * 80)
    print("1. 数据生成演示")
    print("=" * 80)
    
    # 创建数据生成器
    generator = RamanDataGenerator(num_samples=100)
    
    # 生成数据
    spectra, stages, concentrations = generator.generate_dataset()
    
    print(f"\n生成的数据:")
    print(f"  谱图形状: {spectra.shape}")
    print(f"  阶段标签形状: {stages.shape}")
    print(f"  浓度标签形状: {concentrations.shape}")
    print(f"\n数据统计:")
    print(f"  谱图范围: [{spectra.min():.3f}, {spectra.max():.3f}]")
    print(f"  阶段分布: {np.bincount(stages)}")
    print(f"  浓度范围: [{concentrations.min():.2f}, {concentrations.max():.2f}]")
    
    return spectra, stages, concentrations


def demo_model_architecture():
    """演示模型架构"""
    print("\n" + "=" * 80)
    print("2. 模型架构演示")
    print("=" * 80)
    
    # 创建CNN模型
    print("\n1D-CNN 模型:")
    cnn_model = create_cnn_model()
    cnn_info = cnn_model.get_model_info()
    for key, value in cnn_info.items():
        print(f"  {key}: {value}")
    
    # 创建ResNet模型
    print("\n1D-ResNet18 模型:")
    resnet_model = create_resnet_model()
    resnet_info = resnet_model.get_model_info()
    for key, value in resnet_info.items():
        print(f"  {key}: {value}")
    
    return cnn_model, resnet_model


def demo_forward_pass(model, model_name="CNN"):
    """演示前向传播"""
    print("\n" + "=" * 80)
    print(f"3. {model_name} 前向传播演示")
    print("=" * 80)
    
    # 准备测试数据
    batch_size = 8
    test_input = torch.randn(batch_size, 1, config.INPUT_DIM)
    
    print(f"\n输入形状: {test_input.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        class_output, reg_output = model(test_input)
    
    print(f"分类输出形状: {class_output.shape}")
    print(f"回归输出形状: {reg_output.shape}")
    
    # 解析输出
    stage_probs = torch.softmax(class_output, dim=1)
    stage_preds = torch.argmax(stage_probs, dim=1)
    
    print(f"\n第一个样本的预测:")
    print(f"  预测阶段: {config.CLASS_NAMES[stage_preds[0]]}")
    print(f"  置信度: {stage_probs[0][stage_preds[0]]:.2%}")
    print(f"  化学成分浓度预测 (前5个):")
    for i in range(5):
        print(f"    {config.TARGET_NAMES[i]:20s}: {reg_output[0][i].item():6.2f} mg/L")


def demo_single_prediction():
    """演示单个样本预测"""
    print("\n" + "=" * 80)
    print("4. 单样本预测演示")
    print("=" * 80)
    
    # 生成测试样本
    generator = RamanDataGenerator(num_samples=1, random_seed=123)
    spectra, stages, concentrations = generator.generate_dataset()
    test_spectrum = spectra[0]
    
    print(f"\n真实标签:")
    print(f"  发酵阶段: {config.CLASS_NAMES[stages[0]]}")
    print(f"  化学成分浓度 (前5个):")
    for i in range(5):
        print(f"    {config.TARGET_NAMES[i]:20s}: {concentrations[0][i]:6.2f} mg/L")
    
    # 创建模型并预测
    model = create_cnn_model()
    model.eval()
    
    # 准备输入
    spectrum_tensor = torch.FloatTensor(test_spectrum).unsqueeze(0).unsqueeze(0)
    
    # 预测
    with torch.no_grad():
        class_output, reg_output = model(spectrum_tensor)
    
    stage_prob = torch.softmax(class_output, dim=1)
    stage_pred = torch.argmax(stage_prob, dim=1)
    
    print(f"\n预测结果 (未训练模型 - 随机预测):")
    print(f"  预测阶段: {config.CLASS_NAMES[stage_pred[0]]}")
    print(f"  置信度: {stage_prob[0][stage_pred[0]]:.2%}")
    print(f"  化学成分浓度预测 (前5个):")
    for i in range(5):
        print(f"    {config.TARGET_NAMES[i]:20s}: {reg_output[0][i].item():6.2f} mg/L")
    
    print("\n注意: 这是未训练模型的随机预测。运行 train.py 进行训练后，预测会更准确。")


def demo_training_workflow():
    """演示训练流程（不实际训练）"""
    print("\n" + "=" * 80)
    print("5. 训练流程说明")
    print("=" * 80)
    
    print("""
训练步骤:
    
1. 准备数据:
   - 运行数据生成器创建模拟数据，或加载真实数据
   - 数据自动划分为5折用于交叉验证
   
2. 训练模型:
   运行命令: python train.py
   
   训练过程包括:
   - 五折交叉验证
   - 每折训练 {} 个 epoch
   - 使用 Adam 优化器，学习率 {}
   - 自动保存最佳模型到 saved_models/ 目录
   - 生成学习曲线图到 results/ 目录
   
3. 评估结果:
   训练完成后会输出:
   - 每折的验证性能
   - 交叉验证的平均性能和标准差
   - 分类准确率、F1分数
   - 回归R²、MAE、RMSE等指标
   
4. 使用模型预测:
   运行命令: python predict.py
   
   预测功能:
   - 单样本预测
   - 批量预测
   - 集成预测（多模型多折）
   - 测试集评估

预计训练时间:
   - 1D-CNN: 约 15-20 分钟 (CPU)
   - 1D-ResNet18: 约 25-30 分钟 (CPU)
   - 使用 GPU 可大幅加速
    """.format(config.EPOCHS, config.LEARNING_RATE))


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("拉曼谱图一维卷积神经网络分析系统 - 快速开始")
    print("=" * 80)
    
    # 1. 数据生成演示
    spectra, stages, concentrations = demo_data_generation()
    
    # 2. 模型架构演示
    cnn_model, resnet_model = demo_model_architecture()
    
    # 3. 前向传播演示
    demo_forward_pass(cnn_model, "1D-CNN")
    demo_forward_pass(resnet_model, "1D-ResNet18")
    
    # 4. 单样本预测演示
    demo_single_prediction()
    
    # 5. 训练流程说明
    demo_training_workflow()
    
    print("\n" + "=" * 80)
    print("快速开始演示完成！")
    print("=" * 80)
    print("\n下一步:")
    print("  1. 运行 'python train.py' 开始训练模型")
    print("  2. 训练完成后运行 'python predict.py' 进行预测")
    print("  3. 查看 README.md 了解更多功能和使用说明")
    print("\n祝您使用愉快！")


if __name__ == '__main__':
    main()

