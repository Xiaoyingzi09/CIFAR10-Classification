import numpy as np
from sklearn.utils import shuffle
from model import NeuralNetwork  # 修改导入项
from utils import cross_entropy_loss, accuracy, cosine_annealing_with_warmup, TrainingLogger, step_decay, constant_lr
import copy
import pickle

class NeuralNetworkTrainer:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test,layer_dims, activation,
                 reg_lambda=0.01, epochs=1000, batch_size=64, 
                 initial_lr=0.01, min_lr=1e-6, warmup_epochs=10, 
                 lr_strategy='cosine_warmup',
                 log_path='/home/Hanano/gxx/DLHW/HM01/logs/training_log.npz'):
        # 封装训练状态
        self.model = NeuralNetwork(layer_dims, activation)  # 使用神经网络对象
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        # 封装超参数
        self.reg_lambda = reg_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.lr_strategy = lr_strategy
        
        # 训练状态跟踪
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_model = None
        self.save_path = f'/home/Hanano/gxx/DLHW/HM01/weights/[{self.lr_strategy}][lr={self.initial_lr}][lbd={self.reg_lambda}][hid_dim={self.model.layer_dims[1]}]best_model.pkl'
        
        log_path = f'/home/Hanano/gxx/DLHW/HM01/logs/[{self.lr_strategy}][lr={self.initial_lr}][lbd={self.reg_lambda}][hid_dim={self.model.layer_dims[1]}]training_log.npz'
        self.logger = TrainingLogger(log_path)
        

    def train_epoch(self, epoch, learning_rate):
        """封装单epoch训练逻辑"""
        X_shuffled, y_shuffled = shuffle(self.X_train, self.y_train)
        
        for i in range(0, X_shuffled.shape[0], self.batch_size):
            X_batch = X_shuffled[i:i+self.batch_size]
            y_batch = y_shuffled[i:i+self.batch_size]
            
            # 使用对象方法代替独立函数
            self.model.forward(X_batch)
            grads = self.model.backward(y_batch, self.reg_lambda)
            self.model.update_parameters(grads, learning_rate)

    def validate(self):
        """封装验证逻辑"""
        # 训练集评估
        train_output = self.model.forward(self.X_train)
        train_loss = cross_entropy_loss(self.y_train, train_output, 
                                      self.model.params, self.reg_lambda)
        train_acc = accuracy(self.y_train, train_output) 
        
        # 验证集评估
        val_output = self.model.forward(self.X_val)
        val_loss = cross_entropy_loss(self.y_val, val_output, 
                                    self.model.params, self.reg_lambda)
        val_acc = accuracy(self.y_val, val_output)
        
        # 验证集评估
        test_output = self.model.forward(self.X_test)
        test_loss = cross_entropy_loss(self.y_test, test_output, 
                                    self.model.params, self.reg_lambda)
        test_acc = accuracy(self.y_test, test_output)
        print(f'test_acc: {test_acc}')
        
        return train_loss, train_acc, val_loss, val_acc 

    def run_training(self):
        """封装完整训练流程"""
        for epoch in range(self.epochs):
            
            # lr = cosine_annealing_with_warmup(epoch, self.epochs, 
            #                                 self.warmup_epochs, 
            #                                 self.initial_lr, self.min_lr)
            lr = self._get_learning_rate(epoch)
            
            self.train_epoch(epoch, lr)
            train_loss, train_acc, val_loss, val_acc = self.validate()  # 修改参数接收
            
            self._update_training_state(train_loss, val_loss, val_acc, epoch)
            self._save_best_model(val_acc)
            
            # 记录日志条目
            self.logger.add_entry(
                epoch=epoch+1,
                lr=lr,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc
            )
            
            # 修改打印语句
            print(f"Epoch {epoch+1}/{self.epochs}, LR: {lr:.6f}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            self.logger.save_log()  # 训练结束后保存日志
        
        return self.best_model, self.train_losses, self.val_losses, self.val_accuracies

    def _update_training_state(self, train_loss, val_loss, val_acc, epoch):
        """封装状态更新逻辑"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)

    def _save_best_model(self, current_acc):
        """封装模型保存逻辑"""
        if current_acc > self.best_val_acc:
            self.best_val_acc = current_acc
            self.best_model = copy.deepcopy(self.model)
            
            with open(self.save_path, 'wb') as f:
                pickle.dump(self.best_model.params, f)
                
    def _get_learning_rate(self, epoch):
        if self.lr_strategy == 'cosine_warmup':
            return cosine_annealing_with_warmup(
                epoch=epoch,
                total_epochs=self.epochs,
                warmup_epochs=self.warmup_epochs,
                initial_lr=self.initial_lr,
                min_lr=self.min_lr
            )
        elif self.lr_strategy == 'step':
            return step_decay(
                epoch=epoch,
                initial_lr=self.initial_lr,
                step_size=100,
                decay_rate=0.5
            )
        elif self.lr_strategy == 'constant':
            return constant_lr(
                epoch=epoch,
                initial_lr=self.initial_lr
            )
        else:
            raise ValueError(f"Unsupported learning rate strategy: {self.lr_strategy}")
