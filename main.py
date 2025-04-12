import numpy as np
from data_loader import DatasetLoader
from train import NeuralNetworkTrainer
from model import NeuralNetwork, ReLU
import logging
import json
from plot_results import TrainingVisualizer, WeightAnalyzer
from grid_search import GridSearch
from test import ModelTester

class Experiment:
    def __init__(self, config_path):
        """
        封装整个深度学习应用的生命周期
        :param config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.data_loader = None
        self.trainer = None
        self.tester = None  # 新增测试器实例
        
    def _load_config(self, path):
        """封装配置加载逻辑"""
        with open(path) as f:
            return json.load(f)
    
    def _setup_logger(self):
        """封装日志初始化"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(self.__class__.__name__)
    
    def initialize_components(self):
        """组件初始化方法"""
        self.logger.info("Initializing data loader...")
        self.data_loader = DatasetLoader(
            self.config['data_path'],
            batch_size=self.config['batch_size'],
            normalize=self.config['normalize']
        )
        
        self.logger.info("Building neural network...")
        self.trainer = NeuralNetworkTrainer(
            X_train=self.data_loader.X_train,
            y_train=self.data_loader.y_train,
            X_val=self.data_loader.X_val,
            y_val=self.data_loader.y_val,
            X_test=self.data_loader.X_test,
            y_test=self.data_loader.y_test,
            layer_dims=self.config['layer_dims'],
            activation=ReLU,
            lr_strategy=self.config['lr_strategy'],
            batch_size=self.config['batch_size'], 
            **self.config['training_params']
        )
        
        # 新增测试器初始化
        if self.config['test']:
            self.logger.info("Initializing tester...")
            self.tester = ModelTester(
                # model_path=self.config['model_path']
                model_path=f'/home/Hanano/gxx/DLHW/HM01/weights/[{self.config["lr_strategy"]}][lr={self.config["training_params"]["initial_lr"]}][lbd={self.config["training_params"]["reg_lambda"]}][hid_dim={self.config["layer_dims"][1]}]best_model.pkl'
            )
            
    
    def run(self):
        if self.config.get('train', False):
            self.logger.info("Starting training pipeline...")
            # best_model, losses, accuracies = self.trainer.run_training()
            
            best_model, losses, _, accuracies = self.trainer.run_training()
            
            # self._save_results(best_model, losses, accuracies)
            # self._save_results(best_model, losses, accuracies)
        
        if self.config.get('test', True):
            self.logger.info("Starting testing pipeline...")
            self._run_testing()
            
    def _run_testing(self):
        test_results = self.tester.evaluate(
            self.data_loader.X_test,
            self.data_loader.y_test,
            batch_size=self.config.get('test_batch_size', 10000)
        )
        self.logger.info(f"[Test Results] Loss: {test_results['loss']:.4f}, Acc: {test_results['accuracy']:.4f}")
        self.tester.print_report()
    
    def run_hyperparameter_search(self):
        """执行网格搜索的入口方法"""
        print("Start Search Param!!!")
        grid_search = GridSearch(
            data_loader=self.data_loader,
            base_config={
                'layer_dims': self.config['layer_dims'],
                'lr_strategy': self.config['lr_strategy'],
                'training_params': self.config['training_params'],
                'base_output': self.config['base_output']
            }
        )
        return grid_search.run(self.config['grid_search'])
    
    def _save_results(self, model, losses, accuracies):
        """封装结果保存逻辑"""
        np.savez(
            self.config['output_path']+'/training_history.npz',
            train_losses=losses,
            val_accuracies=accuracies
        )
        model.save(self.config['output_path'] / 'final_model.pkl')
        self.logger.info(f"Results saved to {self.config['output_path']}")
        
    def visualize_results(self):
        log_path = f'/home/Hanano/gxx/DLHW/HM01/logs/[{self.config["lr_strategy"]}][lr={self.config["training_params"]["initial_lr"]}][lbd={self.config["training_params"]["reg_lambda"]}][hid_dim={self.config["layer_dims"][1]}]training_log.npz'
        visualizer = TrainingVisualizer(log_path=log_path, config=self.config)
        visualizer.plot_all()

        weight_path = f'/home/Hanano/gxx/DLHW/HM01/weights/[{self.config["lr_strategy"]}][lr={self.config["training_params"]["initial_lr"]}][lbd={self.config["training_params"]["reg_lambda"]}][hid_dim={self.config["layer_dims"][1]}]best_model.pkl'
        analyzer = WeightAnalyzer(weight_path=weight_path, config=self.config)
        analyzer.plot_weight_distributions()

if __name__ == "__main__":
    # param_search = True
    param_search = False
    # visualize = False
    visualize = True
    
    if param_search:
        Ex = Experiment(config_path="/home/Hanano/gxx/DLHW/HM01/config.json")
        Ex.initialize_components()
        best_params = Ex.run_hyperparameter_search()
    else:
        Ex = Experiment(config_path="/home/Hanano/gxx/DLHW/HM01/config.json")
        Ex.initialize_components()
        Ex.run()
    
    if visualize:
        Ex.visualize_results()