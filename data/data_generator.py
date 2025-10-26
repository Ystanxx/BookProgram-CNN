"""
模拟拉曼谱图数据生成器
生成用于训练和测试的模拟数据
"""
import numpy as np
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class RamanDataGenerator:
    """拉曼谱图数据生成器"""
    
    def __init__(self, num_samples=1000, random_seed=42):
        """
        初始化数据生成器
        
        Args:
            num_samples: 生成的样本数量
            random_seed: 随机种子
        """
        self.num_samples = num_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 从配置文件获取参数
        self.input_dim = config.INPUT_DIM
        self.num_classes = config.NUM_CLASSES
        self.num_targets = config.NUM_TARGETS
        self.noise_level = config.DATA_CONFIG['noise_level']
        
        # 定义特征峰位（不同化学成分对应的拉曼位移，单位：cm^-1）
        # 这些是模拟的峰位，实际应用中应根据真实数据调整
        self.peak_positions = {
            "儿茶素C": [100, 250, 450, 650],
            "EGCG": [120, 280, 480, 680],
            "咖啡因": [150, 320, 520, 720],
            "茶氨酸": [180, 350, 550, 750],
            "表儿茶素": [200, 380, 580, 780],
            "表没食子儿茶素": [220, 400, 600, 800],
            "没食子酸": [240, 420, 620, 820],
            "茶黄素": [260, 440, 640, 840],
            "茶红素": [280, 460, 660, 860],
            "可溶性糖": [300, 480, 680, 880],
        }
        
    def _generate_baseline(self):
        """生成基线信号"""
        x = np.linspace(0, self.input_dim - 1, self.input_dim)
        # 使用多项式生成平滑的基线
        baseline = 0.1 + 0.0001 * x + 0.0000001 * x**2
        return baseline
    
    def _generate_peak(self, position, intensity, width=20):
        """
        生成高斯峰
        
        Args:
            position: 峰位置
            intensity: 峰强度
            width: 峰宽度
        """
        x = np.arange(self.input_dim)
        peak = intensity * np.exp(-((x - position) ** 2) / (2 * width ** 2))
        return peak
    
    def generate_spectrum(self, stage, concentrations):
        """
        生成单个拉曼谱图
        
        Args:
            stage: 发酵阶段 (0-9)
            concentrations: 10种化学成分的浓度数组
        
        Returns:
            spectrum: 生成的拉曼谱图
        """
        # 初始化谱图为基线
        spectrum = self._generate_baseline()
        
        # 根据浓度添加各化学成分的特征峰
        for idx, (component, positions) in enumerate(self.peak_positions.items()):
            concentration = concentrations[idx]
            # 浓度越高，峰强度越大
            intensity_factor = concentration / 100.0  # 归一化到0-1
            
            for pos in positions:
                # 添加发酵阶段的影响（不同阶段峰位略有偏移）
                pos_shift = pos + stage * 2  # 每个阶段偏移2个单位
                if 0 <= pos_shift < self.input_dim:
                    peak_intensity = intensity_factor * np.random.uniform(0.3, 0.8)
                    spectrum += self._generate_peak(pos_shift, peak_intensity)
        
        # 添加噪声
        noise = np.random.normal(0, self.noise_level, self.input_dim)
        spectrum += noise
        
        # 归一化到0-1范围，添加更严格的除零保护
        spec_min = spectrum.min()
        spec_max = spectrum.max()
        spec_range = spec_max - spec_min
        
        if spec_range > 1e-10:  # 只有范围足够大时才归一化
            spectrum = (spectrum - spec_min) / spec_range
        else:
            # 如果范围太小，设为常数0.5
            spectrum = np.full_like(spectrum, 0.5)
        
        return spectrum
    
    def generate_dataset(self):
        """
        生成完整数据集
        
        Returns:
            spectra: 拉曼谱图数组 (num_samples, input_dim)
            stages: 发酵阶段标签 (num_samples,)
            concentrations: 化学成分浓度 (num_samples, num_targets)
        """
        spectra = []
        stages = []
        concentrations_list = []
        
        for i in range(self.num_samples):
            # 随机选择发酵阶段
            stage = np.random.randint(0, self.num_classes)
            
            # 生成化学成分浓度
            # 不同发酵阶段有不同的浓度分布特征
            concentrations = np.zeros(self.num_targets)
            for j in range(self.num_targets):
                # 基础浓度 + 阶段相关变化
                base_conc = 50.0  # 基础浓度
                stage_effect = (stage / self.num_classes) * 30.0  # 阶段影响
                component_effect = (j / self.num_targets) * 20.0  # 成分差异
                random_variation = np.random.uniform(-10, 10)  # 随机变化
                
                concentrations[j] = np.clip(
                    base_conc + stage_effect + component_effect + random_variation,
                    0, 100
                )
            
            # 生成谱图
            spectrum = self.generate_spectrum(stage, concentrations)
            
            spectra.append(spectrum)
            stages.append(stage)
            concentrations_list.append(concentrations)
        
        spectra = np.array(spectra, dtype=np.float32)
        stages = np.array(stages, dtype=np.int64)
        concentrations_list = np.array(concentrations_list, dtype=np.float32)
        
        return spectra, stages, concentrations_list
    
    def save_dataset(self, save_dir):
        """
        生成并保存数据集
        
        Args:
            save_dir: 保存目录
        """
        print(f"正在生成 {self.num_samples} 个样本...")
        spectra, stages, concentrations = self.generate_dataset()
        
        # 保存为numpy文件
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'spectra.npy'), spectra)
        np.save(os.path.join(save_dir, 'stages.npy'), stages)
        np.save(os.path.join(save_dir, 'concentrations.npy'), concentrations)
        
        print(f"数据集已保存到 {save_dir}")
        print(f"谱图形状: {spectra.shape}")
        print(f"阶段标签形状: {stages.shape}")
        print(f"浓度标签形状: {concentrations.shape}")
        
        return spectra, stages, concentrations


def load_or_generate_data(data_dir, num_samples=1000, regenerate=False):
    """
    加载或生成数据集
    
    Args:
        data_dir: 数据目录
        num_samples: 样本数量
        regenerate: 是否重新生成数据
    
    Returns:
        spectra, stages, concentrations
    """
    spectra_path = os.path.join(data_dir, 'spectra.npy')
    stages_path = os.path.join(data_dir, 'stages.npy')
    concentrations_path = os.path.join(data_dir, 'concentrations.npy')
    
    # 检查数据是否已存在
    if not regenerate and os.path.exists(spectra_path):
        print("加载已有数据...")
        spectra = np.load(spectra_path)
        stages = np.load(stages_path)
        concentrations = np.load(concentrations_path)
        print(f"加载完成: {len(spectra)} 个样本")
    else:
        print("生成新数据...")
        generator = RamanDataGenerator(num_samples=num_samples)
        spectra, stages, concentrations = generator.save_dataset(data_dir)
    
    return spectra, stages, concentrations


if __name__ == '__main__':
    # 测试数据生成器
    generator = RamanDataGenerator(num_samples=100)
    spectra, stages, concentrations = generator.generate_dataset()
    
    print(f"\n数据统计:")
    print(f"谱图范围: [{spectra.min():.3f}, {spectra.max():.3f}]")
    print(f"阶段分布: {np.bincount(stages)}")
    print(f"浓度范围: [{concentrations.min():.2f}, {concentrations.max():.2f}]")
    print(f"浓度均值: {concentrations.mean(axis=0).round(2)}")

