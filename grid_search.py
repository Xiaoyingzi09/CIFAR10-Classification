import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from itertools import product
from train import NeuralNetworkTrainer
from model import *

class GridSearch:
    def __init__(self, data_loader, base_config):
        """
        独立封装的网格搜索类
        :param data_loader: 已初始化的数据加载器实例
        :param base_config: 基础配置文件内容
        """
        self.data_loader = data_loader
        self.base_config = base_config
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(self.__class__.__name__)
    
    def run(self, search_space):
        """执行网格搜索主流程"""
        results = []
        best_acc = 0
        best_params = {}
        
        for i, params in enumerate(tqdm(product(*search_space.values()))):
            current_params = dict(zip(search_space.keys(), params))
            self.logger.info(f"Running combination {i+1}: {current_params}")
            
            record_params = current_params.copy()
            
            # 创建实验目录
            # exp_dir = Path(self.base_config['base_output']) + f"\exp_{i+1}"
            # exp_dir = Path(self.base_config['base_output']) / f"exp_{i+1}"

            # exp_dir.mkdir(parents=True, exist_ok=True)
            
            # 执行单次训练
            trainer = self._create_trainer(current_params)
            _, _, _, val_accuracies = trainer.run_training()
            
            # 记录结果
            result = {
                **record_params,
                # 'final_val_acc': val_accuracies[-1],
                'final_val_acc': max(val_accuracies),
                # 'log_dir': str(exp_dir)
            }

            results.append(result)
            
            if result['final_val_acc'] > best_acc:
                best_acc = result['final_val_acc']
                best_params = current_params
                
        self._save_results(results, best_params)
        return best_params

    def _create_trainer(self, params):
        """根据参数创建训练器实例"""
        layer_dims = params.pop('layer_dims', self.base_config['layer_dims'])
        
        return NeuralNetworkTrainer(
            X_train=self.data_loader.X_train,
            y_train=self.data_loader.y_train,
            X_val=self.data_loader.X_val,
            y_val=self.data_loader.y_val,
            X_test=self.data_loader.X_test,
            y_test=self.data_loader.y_test,
            layer_dims=layer_dims,
            activation=ReLU,
            lr_strategy=self.base_config['lr_strategy'],
            **{**self.base_config['training_params'], **params}
        )

    def _save_results(self, results, best_params):
        """保存网格搜索结果"""
        df = pd.DataFrame(results)
        output_path = Path(self.base_config['base_output'])
        df.to_csv(output_path / 'grid_search_summary.csv', index=False)
        
        with open(output_path / 'best_params.json', 'w') as f:
            json.dump(best_params, f)
            