import numpy as np

# def cross_entropy_loss(y_true, y_pred):
#     m = y_true.shape[0]
#     return -np.sum(y_true * np.log(y_pred + 1e-9)) / m

def cross_entropy_loss(y_true, y_pred, params, reg_lambda, l2=True):
    m = y_true.shape[0]
    ce_loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # 交叉熵损失
    if not l2:
        return ce_loss
    l2_reg = (reg_lambda / 2) * (np.sum(params['W1']**2) + np.sum(params['W2']**2))  # L2 正则化项
    return ce_loss + l2_reg

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

def cosine_annealing_with_warmup(epoch, total_epochs, warmup_epochs, initial_lr, min_lr=1e-6):
    """
    计算某个 epoch 对应的学习率，使用 warmup + 余弦退火策略。

    参数:
    - epoch: 当前训练的 epoch 号
    - total_epochs: 训练的总 epoch 数
    - warmup_epochs: warmup 持续的 epoch 数
    - initial_lr: 预设的初始学习率
    - min_lr: 余弦退火最终的最小学习率

    返回:
    - 当前 epoch 的学习率
    """
    if epoch < warmup_epochs:
        # Warmup: 线性增长学习率
        return min_lr + (initial_lr - min_lr) * (epoch / warmup_epochs)
    
    # 余弦退火
    cosine_decay = 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    return min_lr + (initial_lr - min_lr) * cosine_decay

def constant_lr(epoch, **kwargs):
    return kwargs['initial_lr']

def step_decay(epoch, step_size=50, decay_rate=0.5, **kwargs):
    return kwargs['initial_lr'] * (decay_rate ** (epoch // step_size))


class TrainingLogger:
    """面向训练过程的日志记录器"""
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_data = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def add_entry(self, epoch, lr, train_loss, train_acc, val_loss, val_acc):
        """添加单条训练记录"""
        self.log_data['epoch'].append(epoch)
        self.log_data['lr'].append(lr)
        self.log_data['train_loss'].append(train_loss)
        self.log_data['train_acc'].append(train_acc)
        self.log_data['val_loss'].append(val_loss)
        self.log_data['val_acc'].append(val_acc)
    
    def save_log(self):
        """保存日志到文件"""
        np.savez(self.log_path, **self.log_data)
        print(f"Training log saved to {self.log_path}")