"""
案例:
    ANN(人工神经网络)案例: 手机价格分类案例.

背景:
    基于手机的20列特征 -> 预测手机的价格区间(4个区间), 可以用机器学习做, 也可以用 深度学习做(推荐)

ANN案例的实现步骤:
    1. 构建数据集.
    2. 搭建神经网络.
    3. 模型训练.
    4. 模型测试.
"""

# 导包
import torch                                    # PyTorch框架, 封装了张量的各种操作
from torch.utils.data import TensorDataset      # 数据集对象.   数据 -> Tensor -> 数据集 -> 数据加载器
from torch.utils.data import DataLoader         # 数据加载器.
import torch.nn as nn                           # neural network, 封装了神经网络的各种操作
import torch.optim as optim                     # 优化器
from sklearn.model_selection import train_test_split    # 训练集和测试集的划分
import matplotlib.pyplot as plt                 # 绘图
import numpy as np                              # 数组(矩阵)操作
import pandas as pd                             # 数据处理
import time                                     # 时间模块

# todo 1. 定义函数, 构建数据集.
def create_dataset():
    # 1. 加载csv文件数据集.
    data = pd.read_csv('./data/手机价格预测.csv')
    # print(f'data: {data.head()}')
    # print(f'data: {data.shape}')    # (2000, 21)

    # 2. 获取x特征列 和 y标签列.
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # print(f'x: {x.head()}, {x.shape}')  # (2000, 20)
    # print(f'y: {y.head()}, {y.shape}')  # (2000, )

    # 3. 把特征列转成浮点型.
    x = x.astype(np.float32)
    # print(f'x: {x.head()}, {x.shape}')   # (2000, 20)

    # 4. 切分训练集和测试集.
    # 参1: 特征, 参2: 标签, 参3: 测试集所占比例, 参4: 随机种子, 参5: 样本的分布(即: 参考y的类别进行抽取数据)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)

    # 5. 把数据集封装成 张量数据集.  思路: 数据 -> 张量Tensor -> 数据集TensorDataSet -> 数据加载器DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    # print(f'train_dataset: {train_dataset}, test_dataset: {test_dataset}')

    # 6. 返回结果                         20(充当 输入特征数)     4(充当 输出标签数)
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))


# todo 2. 搭建神经网络.
# todo 3. 模型训练.
# todo 4. 模型测试.


# todo 5. 测试
if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    print(f'训练集 数据集对象: {train_dataset}')
    print(f'测试集 数据集对象: {test_dataset}')
    print(f'输入特征数: {input_dim}')    # 20
    print(f'输出标签数: {output_dim}')   # 4