import pickle
import numpy as np
from model import NeuralNetwork
from data_loader import DatasetLoader

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

class ModelTester:
    def __init__(self, model_path):
        """
        封装测试模块的完整生命周期
        :param model_path: 训练好的模型参数文件路径
        """
        print(f'Load Model: [{model_path}]')
        self.model = self._load_model(model_path)
        self.test_loss = None
        self.test_accuracy = None
        
    def _load_model(self, path):
        """封装模型加载逻辑"""
        with open(path, 'rb') as f:
            params = pickle.load(f)
            
        # 通过参数结构自动推断网络结构
        layer_dims = [params[f'W{i}'].shape[0] for i in range(1, len(params)//2 + 1)]
        network = NeuralNetwork(layer_dims=layer_dims)
        network.params = params
        return network
    
    def _batch_generator(self, X, y, batch_size):
        """生成批量测试数据"""
        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]
    
    def evaluate(self, X_test, y_test, batch_size=10000):
        """
        执行完整测试流程
        :return: 包含平均损失和准确率的字典
        """
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        all_preds = []
        all_labels = []

        
        for X_batch, y_batch in self._batch_generator(X_test, y_test, batch_size):
            pred = self.model.forward(X_batch)
            batch_loss = self._calculate_loss(y_batch, pred)
            batch_acc = self._calculate_accuracy(y_batch, pred)
            
            # 保存预测和真实标签
            all_preds.append(np.argmax(pred, axis=1))
            all_labels.append(np.argmax(y_batch, axis=1))
        
            total_loss += batch_loss * X_batch.shape[0]
            total_acc += batch_acc * X_batch.shape[0]
            num_batches += X_batch.shape[0]
        
        self.test_loss = total_loss / num_batches
        self.test_accuracy = total_acc / num_batches
        
        # 拼接所有批次的预测和真实标签
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # ==== 混淆矩阵绘制并保存 ====
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label', fontsize=16)
        plt.ylabel('True Label', fontsize=16)
        plt.title('Confusion Matrix', fontsize=18)
        plt.tick_params(axis='both', labelsize=14)

        # 保存图像
        save_dir = './results/visualization/'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'ConfusionMatrix.pdf'), dpi=300, bbox_inches='tight')
        plt.close()


        print(f'number of test data: {num_batches}')
        return {'loss': self.test_loss, 'accuracy': self.test_accuracy}
    
    def _calculate_loss(self, y_true, y_pred):
        """封装损失计算逻辑"""
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / m
    
    # def _calculate_accuracy(self, y_true, y_pred):
    #     """
    #     先逐个样本比较预测类别和真实类别是否一致，
    #     再统计正确分类的数量并计算准确率
    #     """
    #     # 如果 y_true 是 one-hot 编码，转换为类别索引
    #     if y_true.ndim > 1 and y_true.shape[1] > 1:
    #         y_true = np.argmax(y_true, axis=1)

    #     # 获取预测类别索引
    #     y_pred_labels = np.argmax(y_pred, axis=1)

    #     # 对比预测和真实标签
    #     correct = (y_pred_labels == y_true)

    #     # 统计准确数
    #     num_correct = np.sum(correct)
    #     total = len(y_true)

    #     # 返回准确率
    #     return num_correct / total

    
    def _calculate_accuracy(self, y_true, y_pred):
        """封装准确率计算逻辑"""
        print(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    
    def print_report(self):
        """生成格式化测试报告"""
        if self.test_loss is None:
            raise ValueError("请先执行evaluate方法进行测试")
            
        report = f"""
        ======= 模型测试报告 =======
        测试损失: {self.test_loss:.4f}
        测试准确率: {self.test_accuracy:.4f}
        ============================
        """
        print(report)
        
if __name__ == "__main__":
    # 初始化测试器
    tester = ModelTester('/home/Hanano/gxx/DLHW/HM01/weights/best_model.pkl')
    
    data_loader = DatasetLoader(
            "/home/Hanano/gxx/DLHW/HM01/dataset/cifar-10-batches-py",
            mode='test'
        )
    
    X_test, y_test = data_loader.X_test, data_loader.y_test

    # 执行测试
    results = tester.evaluate(X_test, y_test, batch_size=128)

    # 获取结果
    print(f"测试损失: {results['loss']:.4f}, 准确率: {results['accuracy']:.4f}")

    # 生成标准报告
    tester.print_report()