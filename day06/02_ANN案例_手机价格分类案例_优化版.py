"""
ANN案例: 手机价格分类案例 - 优化版

优化思路（相比01版本的6个改进点）:
    ① 数据标准化: StandardScaler 将特征缩放到均值为0、标准差为1的分布
    ② 网络深度: 2层隐藏层 -> 4层隐藏层 (20→128→256→512→128→4)
    ③ 优化器: SGD -> Adam (自适应学习率，收敛更快)
    ④ 学习率: 0.001 -> 0.0001 (1e-4，更精细的参数更新)
    ⑤ 数据划分: 训练集/测试集 -> 训练集/验证集 (验证集调超参，测试集最终评估)
    ⑥ 随机种子: 固定 torch.manual_seed(0) 保证可复现性
"""

# ============================================================
# 第一部分: 导包
# ============================================================
import torch                                    # PyTorch框架，定义张量、神经网络等
import torch.nn as nn                           # neural network，封装了各种网络层和损失函数
import pandas as pd                             # 数据读取和表格处理
from sklearn.model_selection import train_test_split  # 划分训练集/验证集
from torch.utils.data import TensorDataset      # 将特征和标签打包成数据集
from torch.utils.data import DataLoader         # 数据加载器，按批次(batch)迭代数据
import torch.optim as optim                     # 优化器，如SGD、Adam等
import numpy as np                              # 数值计算库
import time                                     # 计时，用于观察训练耗时
from sklearn.preprocessing import StandardScaler # 数据标准化工具


# ============================================================
# 第二部分: 构建数据集
# ============================================================
def create_dataset():
    """
    构建数据集: 读取csv -> 特征/标签分离 -> 类型转换 -> 数据划分 -> 标准化 -> 封装

    数据流程:
        CSV文件 -> DataFrame -> x(特征)/y(标签) -> 划分训练/验证集 ->
        标准化 -> TensorDataset -> 返回
    """
    # --------------------------------------------------
    # 步骤1: 读取CSV文件
    # --------------------------------------------------
    # pd.read_csv 读取CSV文件，返回DataFrame表格对象
    # shape: (2000, 21)，2000条手机数据，每条21列（20个特征 + 1个标签）
    data = pd.read_csv('./data/手机价格预测.csv')

    # --------------------------------------------------
    # 步骤2: 分离特征x和标签y
    # --------------------------------------------------
    # data.iloc[:, :-1]: 取所有行，取除最后一列外的所有列 -> 20个特征
    # data.iloc[:, -1]:  取所有行，只取最后一列 -> price_range (0/1/2/3)
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # x.shape: (2000, 20)，y.shape: (2000,)

    # --------------------------------------------------
    # 步骤3: 类型转换
    # --------------------------------------------------
    # x转float32: PyTorch的Tensor默认是float32，必须保证数据类型一致
    x = x.astype(np.float32)

    # y转int64: CrossEntropyLoss要求标签是long类型(int64)
    # 标签格式是类别索引(0/1/2/3)，不是one-hot编码
    y = y.astype(np.int64)
    # y示例: tensor([0, 3, 1, 2, 1, 0, ...])，每个值代表一个价格区间

    # --------------------------------------------------
    # 步骤4: 划分训练集和验证集
    # --------------------------------------------------
    # train_size=0.8: 80%训练，20%验证（不再是test，改名叫valid更好区分）
    # random_state=88: 随机种子，保证每次划分结果一样（可复现）
    # stratify=y: 分层抽样，保持训练集和验证集中各类别比例与原始数据一致
    # 例如原始数据4类比例是25%各占1/4，划分后也保持25%各占1/4
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y,
        train_size=0.8,     # 训练集占80%
        random_state=88,    # 随机种子88
        stratify=y          # 分层采样
    )
    # x_train.shape: (1600, 20)，x_valid.shape: (400, 20)

    # --------------------------------------------------
    # 步骤5: 数据标准化（优化点①）
    # --------------------------------------------------
    # 为什么要标准化？
    #   原始数据各特征尺度差异大，例如:
    #   - battery_power: 500~2000（数值大）
    #   - m_dep: 0.1~1.0（数值小）
    #   这种差异会导致梯度下降慢、网络偏重某些特征
    #
    # StandardScaler 公式: x_scaled = (x - mean) / std
    # 标准化后每个特征均值≈0，标准差≈1，尺度统一
    transfer = StandardScaler()

    # fit_transform: 用训练集fit（计算均值和标准差），再transform
    # 注意：必须用训练集的均值和标准差来transform验证集，不能用验证集的
    x_train = transfer.fit_transform(x_train)    # 训练集：fit + transform
    x_valid = transfer.transform(x_valid)        # 验证集：只用训练集的统计量transform

    # --------------------------------------------------
    # 步骤6: 封装成TensorDataset
    # --------------------------------------------------
    # TensorDataset: 将特征张量和标签张量打包成一个数据集
    # 每次迭代返回一个(x, y)元组，即一条样本的特征和标签
    train_dataset = TensorDataset(
        torch.from_numpy(x_train),   # 训练集特征，numpy转Tensor
        torch.tensor(y_train.values) # 训练集标签，转成long型Tensor
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(x_valid),
        torch.tensor(y_valid.values)
    )
    # train_dataset[0] -> (特征tensor[20维], 标签tensor[1维])

    # --------------------------------------------------
    # 步骤7: 返回
    # --------------------------------------------------
    # x_train.shape[1]: 特征数量 = 20（输入维度）
    # len(np.unique(y)): 类别数量 = 4（输出维度：0=低端,1=中端,2=高端,3=旗舰）
    return train_dataset, valid_dataset, x_train.shape[1], len(np.unique(y))


# ============================================================
# 第三部分: 构建神经网络模型
# ============================================================
class PhonePriceModel(nn.Module):
    """
    手机价格分类模型: 4层隐藏层的全连接神经网络

    网络结构（优化点②）:
        输入(20) → Linear1(128) → ReLU →
                  → Linear2(256) → ReLU →
                  → Linear3(512) → ReLU →
                  → Linear4(128) → ReLU →
                  → Linear5(4)   → 输出

    层次设计思路:
        - 先逐层扩张（特征提取）：20→128→256→512，提取越来越抽象的特征
        - 再逐层收缩（映射到类别）：512→128→4，把高维特征压缩到4个类别
    """

    def __init__(self, input_dim, output_dim):
        """
        初始化网络层

        参数:
            input_dim:  输入特征数 = 20
            output_dim: 输出类别数 = 4
        """
        # 必须调用父类nn.Module的初始化，才能自动管理参数
        super(PhonePriceModel, self).__init__()

        # -------- 隐藏层1: 20 → 128 --------
        # Linear(in_features, out_features): y = xW^T + b
        # 权重矩阵W: [128, 20]，偏置b: [128]
        # 参数量: 20*128 + 128 = 2,688
        self.linear1 = nn.Linear(input_dim, 128)

        # -------- 隐藏层2: 128 → 256 --------
        # 权重: [256, 128]，偏置: [256]
        # 参数量: 128*256 + 256 = 33,024
        self.linear2 = nn.Linear(128, 256)

        # -------- 隐藏层3: 256 → 512 --------
        # 权重: [512, 256]，偏置: [512]
        # 参数量: 256*512 + 512 = 131,584
        self.linear3 = nn.Linear(256, 512)

        # -------- 隐藏层4: 512 → 128 --------
        # 权重: [128, 512]，偏置: [128]
        # 参数量: 512*128 + 128 = 65,664
        self.linear4 = nn.Linear(512, 128)

        # -------- 输出层: 128 → 4 --------
        # 权重: [4, 128]，偏置: [4]
        # 参数量: 128*4 + 4 = 516
        # 输出4个值，对应4个类别的logits（原始分数，未激活）
        self.linear5 = nn.Linear(128, output_dim)

        # 总参数量: 2,688 + 33,024 + 131,584 + 65,664 + 516 = 233,476

    def forward(self, x):
        """
        前向传播: 数据从输入到输出的计算过程

        参数:
            x: 输入张量，shape=[batch_size, 20]
        返回:
            output: 输出张量，shape=[batch_size, 4]，每个值是对应类别的logit
        """
        # -------- 隐藏层前向传播 --------
        # Linear变换 + ReLU激活
        # ReLU: f(x) = max(0, x)，将负值置零，保留正值
        # 作用: 引入非线性，否则多层线性叠加等价于一层线性变换
        x = torch.relu(self.linear1(x))  # [batch, 20] -> [batch, 128]
        x = torch.relu(self.linear2(x))  # [batch, 128] -> [batch, 256]
        x = torch.relu(self.linear3(x))  # [batch, 256] -> [batch, 512]
        x = torch.relu(self.linear4(x))  # [batch, 512] -> [batch, 128]

        # -------- 输出层 --------
        # 不加ReLU或Softmax，直接输出logits
        # 原因: CrossEntropyLoss内部会做Softmax
        # 如果这里加了Softmax，等于做了两次Softmax，数值会不正确
        output = self.linear5(x)          # [batch, 128] -> [batch, 4]

        return output


# ============================================================
# 第四部分: 训练函数
# ============================================================
def train(train_dataset, input_dim, class_num):
    """
    训练模型

    参数:
        train_dataset: 训练数据集（TensorDataset对象）
        input_dim:     输入特征数 = 20
        class_num:     类别数 = 4
    """
    # --------------------------------------------------
    # 步骤1: 固定随机种子（优化点⑥）
    # --------------------------------------------------
    # torch.manual_seed(0): 固定PyTorch的随机种子
    # 保证每次运行训练过程一致（权重初始化、数据shuffle顺序）
    # 注意: 如果用了GPU，还需要设置 torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)

    # --------------------------------------------------
    # 步骤2: 创建数据加载器
    # --------------------------------------------------
    # DataLoader: 按批次迭代数据集
    # batch_size=8: 每批8条样本
    # shuffle=True: 每个epoch前打乱数据顺序，避免模型学到样本顺序的依赖
    dataloader = DataLoader(
        train_dataset,
        shuffle=True,   # 打乱顺序（训练集要打乱，验证集不打乱）
        batch_size=8    # 每批8条，1600条样本 → 200个batch/epoch
    )

    # --------------------------------------------------
    # 步骤3: 创建模型
    # --------------------------------------------------
    model = PhonePriceModel(input_dim, class_num)

    # --------------------------------------------------
    # 步骤4: 定义损失函数
    # --------------------------------------------------
    # CrossEntropyLoss = LogSoftmax + NLLLoss
    # 输入: model输出的logits [batch, 4] + 真实标签 [batch]
    # 输出: 标量损失值（越小越好）
    # 标签格式: 类别索引(0/1/2/3)，不是one-hot
    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # 步骤5: 定义优化器（优化点③④）
    # --------------------------------------------------
    # optim.Adam: 自适应矩估计优化器
    # 原理: 每个参数有自己的学习率，根据梯度的一阶/二阶矩动态调整
    # 优点: 比SGD收敛快，调参更鲁棒（不需要手动调整学习率衰减）
    #
    # lr=1e-4 = 0.0001: 学习率
    # 学习率大(如0.1): 收敛快但可能震荡不收敛
    # 学习率小(如1e-4): 收敛慢但更稳定精细
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # 对比: SGD需要手动调学习率，Adam自动调，所以可以用更小的学习率

    # --------------------------------------------------
    # 步骤6: 训练循环
    # --------------------------------------------------
    num_epoch = 50  # 训练50轮
    for epoch_idx in range(num_epoch):
        start = time.time()  # 记录本轮开始时间

        # -------- 初始化统计量 --------
        total_loss = 0.0   # 累计总损失（加权）
        total_num = 0      # 累计样本总数

        # -------- 遍历每个batch --------
        for x, y in dataloader:
            # model.train(): 切换到训练模式
            # 效果: Dropout生效、BatchNorm用batch统计量
            model.train()

            # 前向传播: 输入x，输出预测logits
            output = model(x)  # output.shape: [8, 4]

            # 计算损失: 预测logits vs 真实标签y
            loss = criterion(output, y)  # 标量

            # -------- 反向传播 --------
            optimizer.zero_grad()  # 梯度清零（否则会累加）
            loss.backward()        # 反向传播，计算每个参数的梯度
            optimizer.step()       # 根据梯度更新参数

            # -------- 统计本batch --------
            # loss.item(): 取Python标量（去除梯度追踪）
            # total_loss用 加权累加（loss * batch_size），最后统一除以总样本数
            total_num += len(y)                # 累加样本数: 8
            total_loss += loss.item() * len(y) # 累加加权损失: loss*8

        # -------- 打印本轮结果 --------
        # total_loss / total_num: 平均损失（每个样本的平均损失）
        avg_loss = total_loss / total_num
        elapsed = time.time() - start
        print('epoch: %4s loss: %.2f, time: %.2fs' % (epoch_idx + 1, avg_loss, elapsed))

    # --------------------------------------------------
    # 步骤7: 保存模型
    # --------------------------------------------------
    # model.state_dict(): 字典，键是层名，值是权重矩阵
    # 包含: linear1.weight, linear1.bias, linear2.weight, ... 等所有参数
    # 后缀用 .pth、.pkl 均可
    torch.save(model.state_dict(), './model/phone-price-model2.pth')


# ============================================================
# 第五部分: 评估函数
# ============================================================
def evaluate(valid_dataset, input_dim, class_num):
    """
    评估模型在验证集上的准确率

    参数:
        valid_dataset: 验证数据集
        input_dim:     输入特征数
        class_num:     类别数
    """
    # -------- 创建模型 --------
    model = PhonePriceModel(input_dim, class_num)

    # -------- 加载训练好的权重 --------
    # torch.load('./model/...pth'): 加载保存的参数字典（dict）
    # load_state_dict(dict): 将字典中的权重应用到模型的每一层
    model.load_state_dict(torch.load('./model/phone-price-model2.pth'))

    # -------- 创建数据加载器 --------
    dataloader = DataLoader(
        valid_dataset,
        batch_size=8,   # 每批8条
        shuffle=False   # 不打乱（评估不需要打乱）
    )

    # -------- 评估模式 --------
    # model.eval(): 切换到评估模式
    # 效果: Dropout关闭（所有神经元参与）、BatchNorm用全局统计量
    model.eval()

    # -------- 遍历验证集 --------
    correct = 0  # 预测正确的样本总数
    for x, y in dataloader:
        # 前向传播得到logits
        output = model(x)  # output.shape: [8, 4]

        # torch.argmax(logits, dim=1): 沿dim=1方向取最大值索引
        # 即每行（每个样本）的4个logits中，取最大值的索引作为预测类别
        # 返回shape: [8]，每个值是0/1/2/3中的一个
        y_pred = torch.argmax(output, dim=1)

        # 比较预测类别和真实类别，统计正确数
        # (y_pred == y): 返回布尔tensor，True=1, False=0
        # .sum(): 求和得到正确预测的数量
        correct += (y_pred == y).sum()

    # -------- 计算并打印准确率 --------
    accuracy = correct / len(valid_dataset)  # 正确数 / 总数
    print('Acc: %.5f' % (accuracy))          # 保留5位小数，如 0.97250


# ============================================================
# 第六部分: 主函数入口
# ============================================================
if __name__ == '__main__':
    """
    程序入口
    执行流程: 构建数据集 → 训练模型 → 评估模型
    """
    # -------- 步骤1: 构建数据集 --------
    # 返回: 训练集对象、验证集对象、输入维度(20)、输出维度(4)
    train_dataset, valid_dataset, input_dim, class_num = create_dataset()

    # -------- 步骤2: 训练模型 --------
    # 训练完成后会保存模型到 ./model/phone-price-model2.pth
    train(train_dataset, input_dim, class_num)

    # -------- 步骤3: 评估模型 --------
    # 加载保存的模型，在验证集上评估准确率
    evaluate(valid_dataset, input_dim, class_num)
