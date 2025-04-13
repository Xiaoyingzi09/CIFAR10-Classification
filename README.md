# CIFAR10-Classification
深度学习第一次作业CIFAR10图像分类任务

## 数据集下载
下载[CIFAR10数据集](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)，并解压到dataset文件夹下。

## 训练模型
1. 调整config中的参数
2. 执行如下命令
```
python main.py
```
## 配置文件参数

## 参数说明表格
后续执行训练、测试、和可视化等，主要通过调整config的参数来实现，具体参数说明如下：
| 参数名称              | 类型        | 示例值                                             | 说明                                                         |
|-----------------------|-------------|----------------------------------------------------|--------------------------------------------------------------|
| `data_path`           | `str`       | `/home/.../cifar-10-batches-py`                   | CIFAR-10 数据集的存储路径                                     |
| `output_path`         | `str`       | `/home/.../results`                               | 输出结果的保存路径                                           |
| `model_path`          | `str`       | `/home/.../weights/best_model.pkl`                | 保存最优模型的路径                                           |
| `base_output`         | `str`       | `/home/.../results/param_search`                  | 网格搜索的结果基础路径                                       |
| `batch_size`          | `int`       | `256`                                             | 每个训练批次的样本数量                                       |
| `normalize`           | `bool`      | `true`                                            | 是否对输入数据进行标准化                                     |
| `layer_dims`          | `list[int]` | `[3072, 512, 512, 10]`                            | 网络结构定义：输入层+隐藏层+输出层                           |
| `train`               | `bool`      | `false`                                           | 是否进行训练阶段                                             |
| `test`                | `bool`      | `true`                                            | 是否在测试集上评估模型                                       |
| `activation`          | `str`      | `relu`                                            | 采用的激活函数                                   |
| `lr_strategy`         | `str`       | `"constant"`                                     | 学习率策略，支持指定固定学习率，warmup+余弦退火                               |
| `training_params.epochs`       | `int`       | `300`                                             | 最大训练轮数                                                 |
| `training_params.reg_lambda`   | `float`     | `0.001`                                           | L2 正则化强度                                                 |
| `training_params.initial_lr`   | `float`     | `0.1`                                             | 初始学习率                                                   |
| `training_params.warmup_epochs`| `int`       | `20`                                              | 预热轮数，当前策略中无效                                     |
| `grid_search.initial_lr`       | `list`      | `[0.1, 0.01, 0.001]`                             | 要尝试的学习率集合                                           |
| `grid_search.reg_lambda`       | `list`      | `[0.1, 0.01, 0.001]`                             | 要尝试的正则化系数集合                                       |
| `grid_search.layer_dims`       | `list[list]`| `[[3072,128,128,10],...]`                        | 要尝试的网络结构集合                                         |


## 网格参数搜索
1. 调整config中的参数：
```
"train": false
"test": false
"param_search": true
"visualize": false
```
2. 执行如下命令
```
python main.py
```

## 训练固定超参数的模型
1. 调整config中的参数：
```
"train": true
"test": false
"param_search": false
"visualize": false
```
然后调整学习率等超参数即可

2. 执行如下命令
```
python main.py
```

## 测试固定超参数训练出的模型
1. 调整config中的参数：
```
"train": false
"test": true
"param_search": false
"visualize": false
```
然后调整学习率等超参数即可

2. 执行如下命令
```
python main.py
```

## 可视化
1. 调整config中的参数：
```
"train": false
"test": true
"param_search": false
"visualize": true
```
2. 执行如下命令
```
python main.py
```