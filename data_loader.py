import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
import os

class DatasetLoader:
    def __init__(self, data_path, batch_size=64, shuffle=True, normalize=True, one_hot=True, train_ratio=0.8, mode='train'):
        """
        CIFAR-10 数据加载器
        :param data_path: CIFAR-10 数据集路径（包含 data_batch_1 至 data_batch_5）
        :param batch_size: 批处理大小
        :param shuffle: 是否打乱数据
        :param normalize: 是否归一化（标准化）
        :param one_hot: 是否进行 One-hot 编码
        :param train_ratio: 用于训练集的比例（剩余部分为验证集）
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.one_hot = one_hot
        self.train_ratio = train_ratio
        
        self.mode = mode

        self._load_raw_data()
        # self._preprocess()
        self._load_test_data()


        # self._load_raw_data()
        self._preprocess()
        self._current_idx = 0

    def _load_raw_data(self):
        """加载所有训练 batches 并合并"""
        all_images = []
        all_labels = []
        for i in range(1, 6):
            batch_file = os.path.join(self.data_path, f"data_batch_{i}")
            with open(batch_file, 'rb') as f:
                data_dict = pickle.load(f, encoding='bytes')
                images = data_dict[b'data']
                labels = np.array(data_dict[b'labels'])
                images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                all_images.append(images)
                all_labels.append(labels)

        self.raw_images = np.concatenate(all_images, axis=0)  # shape: (50000, 32, 32, 3)
        self.raw_labels = np.concatenate(all_labels, axis=0)  # shape: (50000,)
        
    def _load_test_data(self, test_path=None):
        """加载test_batch数据并处理为统一格式"""
        if test_path is None:
            test_path = f"{self.data_path}/test_batch"
        with open(test_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            test_images = data_dict[b'data']
            test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            test_labels = np.array(data_dict[b'labels'])
        self.test_images = test_images  # shape: (10000, 32, 32, 3)
        self.test_labels = test_labels  # shape: (10000,)

    def _preprocess(self):
        """标准化、打乱、划分训练/验证集并预处理"""
        images = self.raw_images.astype('float32')
        labels = self.raw_labels.astype('int32')

        
        # 标准化
        if self.normalize:
            mean = np.mean(images, axis=(0, 1, 2))
            std = np.std(images, axis=(0, 1, 2))
            images = (images - mean) / std
            print(f"Normalize: mean={mean}, std={std}")

        # 打乱数据
        if self.shuffle:
            perm = np.random.permutation(len(images))
            images = images[perm]
            labels = labels[perm]

    
        num_total = len(images)
        num_train = int(num_total * self.train_ratio)

        self.X_train = images[:num_train].reshape(num_train, -1)  # 展平
        self.y_train = labels[:num_train]
        self.X_val = images[num_train:].reshape(num_total - num_train, -1)
        self.y_val = labels[num_train:]
        
        enc = OneHotEncoder(sparse_output=False)
        
        self.y_train = enc.fit_transform(self.y_train.reshape(-1, 1))
        self.y_val = enc.transform(self.y_val.reshape(-1, 1))
        
        
        test_images = self.test_images.astype('float32')
        test_labels = self.test_labels.astype('int32')
        
        if self.normalize:
            t_mean = np.mean(test_images, axis=(0, 1, 2))
            t_std = np.std(test_images, axis=(0, 1, 2))
            test_images = (test_images - t_mean) / t_std
            print(f"Test Data Normalize: mean={t_mean}, std={t_std}")
            
        self.X_test = test_images.reshape(len(test_images), -1)
        self.y_test = test_labels
        self.y_test = enc.fit_transform(self.y_test.reshape(-1, 1))

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx >= len(self.X_train):
            raise StopIteration
        end_idx = self._current_idx + self.batch_size
        batch = (
            self.X_train[self._current_idx:end_idx],
            self.y_train[self._current_idx:end_idx]
        )
        self._current_idx = end_idx
        return batch

    @property
    def total_batches(self):
        return len(self.X_train) // self.batch_size

    def reload(self):
        """重新加载并重新预处理数据（可用于重新打乱或更改参数）"""
        self._load_raw_data()
        self._preprocess()

