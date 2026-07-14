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
class ANN(nn.Module):
    """定义一个3层全连接神经网络."""
    def __init__(self, input_dim, output_dim):
        super(ANN, self).__init__()
        # 输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层
        self.fc1 = nn.Linear(input_dim, 128)   # 第一个全连接层: 输入20维 -> 输出128维
        self.relu1 = nn.ReLU()                  # 激活函数: ReLU
        self.fc2 = nn.Linear(128, 64)           # 第二个全连接层: 输入128维 -> 输出64维
        self.relu2 = nn.ReLU()                  # 激活函数: ReLU
        self.fc3 = nn.Linear(64, output_dim)    # 输出层: 输入64维 -> 输出4维(4分类)
        # 注意: 输出层不需要激活函数, CrossEntropyLoss会自己计算Softmax

    def forward(self, x):
        """前向传播: x -> fc1 -> relu -> fc2 -> relu -> fc3"""
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


# todo 3. 模型训练.
def train():
    """模型训练函数。"""
    # ========== 1. 获取数据集 ==========
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()

    # ========== 2. 创建模型 ==========
    model = ANN(input_dim, output_dim)
    print('=' * 60)
    print(f'模型结构: \n{model}')
    print('=' * 60)

    # ========== 3. 创建优化器和损失函数 ==========
    criterion = nn.CrossEntropyLoss()           # 交叉熵损失函数(内置Softmax, 适用于多分类)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器, 学习率0.001

    # ========== 4. 创建数据加载器 ==========
    batch_size = 32                             # 每批次32个样本
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # shuffle=True打乱数据
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ========== 5. 训练模型 ==========
    num_epochs = 20                             # 训练轮次
    train_losses = []                           # 保存每轮训练损失
    train_accs = []                             # 保存每轮训练准确率
    test_accs = []                              # 保存每轮测试准确率

    print(f'开始训练, 共{num_epochs}轮...')
    print('=' * 60)
    start_time = time.time()

    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()                           # 设置为训练模式(启用Dropout/BatchNorm等)
        running_loss = 0.0                      # 累计损失
        correct = 0                             # 正确预测数
        total = 0                               # 总样本数

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 注意: targets必须是long类型, CrossEntropyLoss要求标签为long类型
            targets = targets.long()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播 + 参数更新
            optimizer.zero_grad()               # 梯度清零(否则会累加)
            loss.backward()                     # 反向传播计算梯度
            optimizer.step()                    # 更新参数

            # 统计
            running_loss += loss.item()         # 累加损失
            _, predicted = torch.max(outputs, 1)  # 取出最大值的索引作为预测类别
            total += targets.size(0)            # 累加总样本数
            correct += (predicted == targets).sum().item()  # 累加正确预测数

        # 计算本轮训练的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # --- 测试阶段 ---
        model.eval()                            # 设置为评估模式(禁用Dropout等)
        correct_test = 0
        total_test = 0

        with torch.no_grad():                   # 禁用梯度计算(节省内存加速)
            for inputs, targets in test_loader:
                targets = targets.long()
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets).sum().item()

        epoch_test_acc = correct_test / total_test
        test_accs.append(epoch_test_acc)

        # 每10轮打印一次结果
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1:02d}/{num_epochs}] '
                  f'Loss: {epoch_loss:.4f} '
                  f'Acc: {epoch_acc:.4f} '
                  f'Test Acc: {epoch_test_acc:.4f}')

    end_time = time.time()
    print('=' * 60)
    print(f'训练完成! 耗时: {end_time - start_time:.2f}秒')
    print(f'最终训练准确率: {train_accs[-1]:.4f}')
    print(f'最终测试准确率: {test_accs[-1]:.4f}')

    # ========== 6. 绘制训练曲线 ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(train_losses, label='Train Loss', color='blue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Curve')
    axes[0].legend()
    axes[0].grid(True)

    # 准确率曲线
    axes[1].plot(train_accs, label='Train Acc', color='blue')
    axes[1].plot(test_accs, label='Test Acc', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Test Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('./training_curve.png', dpi=150, bbox_inches='tight')
    print(f'训练曲线已保存到: ./training_curve.png')

    return model


# todo 4. 模型测试.
def test(model):
    """使用训练好的模型进行预测。"""
    if model is None:
        print('请先训练模型!')
        return

    # 加载测试数据
    _, test_dataset, input_dim, output_dim = create_dataset()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 评估模式
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            targets = targets.long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    print('=' * 60)
    print(f'模型测试集准确率: {accuracy:.4f} ({correct}/{total})')
    print('=' * 60)


# todo 5. 测试
if __name__ == '__main__':
    print('=' * 60)
    print('ANN案例: 手机价格分类')
    print('=' * 60)

    # 训练模型
    model = train()

    # 测试模型
    test(model)

    print('\n演示预测示例:')
    print('=' * 60)
    # 取一条测试数据演示预测
    _, test_dataset, input_dim, output_dim = create_dataset()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.eval()
    price_range_map = {0: '低端(0)', 1: '中端(1)', 2: '高端(2)', 3: '旗舰(3)'}
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i >= 3:  # 只演示前3条
                break
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            print(f'  样本{i+1}: 真实价格区间={price_range_map[targets.item()]}, '
                  f'预测价格区间={price_range_map[predicted.item()]}')
    print('=' * 60)
    print('程序运行完成!')