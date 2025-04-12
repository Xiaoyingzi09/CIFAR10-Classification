import matplotlib.pyplot as plt
import numpy as np
import pickle

class TrainingVisualizer:
    def __init__(self, log_path, config, save_dir='/home/Hanano/gxx/DLHW/HM01/results/visualization/'):
        """
        封装可视化组件的生命周期
        :param log_path: 训练日志路径（.npz格式）
        :param save_dir: 结果保存目录
        """
        self.log_data = np.load(log_path)
        self.save_dir = save_dir
        self.config = config
        self.prefix = self._build_prefix_from_config()
        self.style_config = {
            'figure_size': (8, 5),
            'font_size': 16,
            'colors': {'train': 'blue', 'val': 'red', 'acc': 'green'},
            'dpi': 300
        }
    
    def plot_loss_curves(self, save_as=None):
        """封装损失曲线绘制逻辑"""
        plt.figure(figsize=self.style_config['figure_size'])
        plt.plot(self.log_data['train_loss'], label='Train Loss', 
                color=self.style_config['colors']['train'])
        plt.plot(self.log_data['val_loss'], label='Validation Loss',
                color=self.style_config['colors']['val'])
        
        plt.tick_params(axis='both', labelsize=self.style_config['font_size']) 
        plt.xlabel('Epochs', fontsize=self.style_config['font_size'])
        plt.ylabel('Loss', fontsize=self.style_config['font_size'])
        plt.title('Training Progress', fontsize=self.style_config['font_size']+2)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_as:
            plt.savefig(f"{self.save_dir}{save_as}", dpi=self.style_config['dpi'], bbox_inches='tight')
        plt.close()
    
    def plot_accuracy_curve(self, save_as=None):
        """封装准确率曲线绘制"""
        plt.figure(figsize=self.style_config['figure_size'])
        plt.plot(self.log_data['train_acc'], label='Train Accuracy',
                color=self.style_config['colors']['train'])
        plt.plot(self.log_data['val_acc'], label='Validation Accuracy',
                color=self.style_config['colors']['acc'])
        
        plt.tick_params(axis='both', labelsize=self.style_config['font_size']) 
        plt.xlabel('Epochs', fontsize=self.style_config['font_size'])
        plt.ylabel('Accuracy', fontsize=self.style_config['font_size'])
        plt.title('Classification Performance', fontsize=self.style_config['font_size']+2)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_as:
            plt.savefig(f"{self.save_dir}{save_as}", dpi=self.style_config['dpi'], bbox_inches='tight')
        plt.close()
    
    # def plot_all(self):
    #     """组合绘图入口方法"""
    #     self.plot_loss_curves(save_as='training_loss.pdf')
    #     self.plot_accuracy_curve(save_as='validation_acc.pdf')
    
    def plot_all(self):
        self.plot_loss_curves(save_as=f'{self.prefix}Loss.pdf')
        self.plot_accuracy_curve(save_as=f'{self.prefix}Acc.pdf')
            
    def _build_prefix_from_config(self):
        return f'[{self.config["lr_strategy"]}][lr={self.config["training_params"]["initial_lr"]}][lbd={self.config["training_params"]["reg_lambda"]}][hid_dim={self.config["layer_dims"][1]}]'


class WeightAnalyzer:
    def __init__(self, weight_path, config):
        """
        封装权重分析功能
        :param weight_path: 模型权重文件路径（.pkl格式）
        """
        self.config = config
        self.prefix = self._build_prefix_from_config()
        
        with open(weight_path, 'rb') as f:
            self.params = pickle.load(f)
    
    def plot_weight_distributions(self, save_dir='/home/Hanano/gxx/DLHW/HM01/results/visualization/', save_as='weight_analysis.pdf'):
        """可视化各层权重分布"""
        num_layers = len(self.params) // 2
        fig, axes = plt.subplots(num_layers, 2, 
                               figsize=(12, num_layers * 5),
                               tight_layout=True)
        
        for i in range(1, num_layers + 1):
            W = self.params[f'W{i}']
            
            # 权重矩阵可视化
            axes[i-1, 0].tick_params(axis='both', labelsize=14)  # 左边的图
            axes[i-1, 0].imshow(W, cmap='coolwarm', aspect='auto')
            axes[i-1, 0].set_title(f"Layer {i} Weight Matrix")
            
            # 权重直方图
            axes[i-1, 1].tick_params(axis='both', labelsize=14)  # 右边的图
            axes[i-1, 1].hist(W.flatten(), bins=50, color='blue', alpha=0.7)
            axes[i-1, 1].set_title(f"Layer {i} Weight Distribution")
            
            

            
        save_as = f'{self.prefix}Weights.pdf'
        plt.savefig(save_dir+save_as, bbox_inches='tight')
        plt.close()
        
    def _build_prefix_from_config(self):
        return f'[{self.config["lr_strategy"]}][lr={self.config["training_params"]["initial_lr"]}][lbd={self.config["training_params"]["reg_lambda"]}][hid_dim={self.config["layer_dims"][1]}]'
