import math
import time

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils import data
import random


# sgd 随机梯度下降
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 累加器
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# def get_data_iter(data, batch_size):
#     inputs, outputs = data.iloc[:, 1:785], data.iloc[:, 0]
#     X, y = torch.tensor(inputs.values).type(torch.float), torch.tensor(outputs.values)
#     num_examples = len(X)
#     indices = list(range(num_examples))
#     # 这些样本是随机读取的，没有特定的顺序
#     train_iter = get_data_iter_batch(indices, num_examples, batch_size, X, y)
#     batch_num = math.ceil(num_examples/batch_size)
#     for i in range(batch_num):
#         yield next(train_iter)

# 打乱数据，获取小批量数据
def get_data_iter(data, batch_size):
    inputs, outputs = data.iloc[:, 1:785], data.iloc[:, 0]
    X, y = torch.tensor(inputs.values).reshape(-1, 1, 28, 28).type(torch.float), torch.tensor(outputs.values)

    shape_aug = torchvision.transforms.RandomResizedCrop(
        (28, 28), scale=(0.5, 1), ratio=(0.5, 2))
    for img in X:
        shape_aug(img)

    num_examples = len(X)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    train_iter = get_data_iter_batch(indices, num_examples, batch_size, X, y)
    batch_num = math.ceil(num_examples/batch_size)
    for i in range(batch_num):
        yield next(train_iter)


def get_test_iter(data, batch_size):
    inputs, outputs = data.iloc[:, 1:785], data.iloc[:, 0]
    X, y = torch.tensor(inputs.values).reshape(-1, 1, 28, 28).type(torch.float), torch.tensor(outputs.values)
    num_examples = len(X)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    train_iter = get_data_iter_batch(indices, num_examples, batch_size, X, y)
    batch_num = math.ceil(num_examples/batch_size)
    for i in range(batch_num):
        yield next(train_iter)


def get_data_iter_batch(indices, num_examples, batch_size, X, y):
    while True:
        random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
            yield X[batch_indices], y[batch_indices]


# 训练模型
def train_ch3(net, train_data, test_data, batch_size, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        print('train epoch:', epoch)
        train_iter = get_data_iter(train_data, batch_size)
        test_iter = get_data_iter(test_data, batch_size)
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        print('train loss:', train_metrics[0], 'train accuracy:', train_metrics[1])
    print('finish training')
    print('test.py accuracy:', evaluate_accuracy(net, test_iter))


def train_epoch_ch3(net, train_iter, loss, updater):
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


# 评估当前模型在测试集中的精确度
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# y_hat 为预测结果矩阵
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # 获取每个预测中的最大值作为预测结果
    cmp = y_hat.type(y.dtype) == y  # 比较预测是否正确
    return float(cmp.type(y.dtype).sum())  # 计算正确预测数


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_ch6(net, train_data, test_data, batch_size, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = Timer(), len(train_data)
    for epoch in range(num_epochs):
        print("epoch:", epoch)
        train_iter = get_data_iter(train_data, batch_size)
        test_iter = get_test_iter(test_data, batch_size)
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        net.train()
        for X, y in train_iter:
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print("train_acc", train_acc, "test_acc", test_acc)
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test.py acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


# 平方损失
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 交叉熵
def corr2d(X, K):  # @save
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


#############################################################


# 设置并行下载流
def get_dataloader_workers():
    return 4


# 下载 Fashion_Mnist 数据集
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
