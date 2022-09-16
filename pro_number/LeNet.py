import torch
import pandas as pd
from torch import nn
from util import pro_number_helper as d2l

# 读取数据集
print('read data')
path1 = './mnist_dataset/mnist_train.csv'
path2 = './mnist_dataset/mnist_test.csv'
train_data = pd.read_csv(path1, header=None)
test_data = pd.read_csv(path1, header=None)
print('finish')

# 初始化参数
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))

batch_size = 256
num_epochs = 10
lr = 0.9

print('train data')
d2l.train_ch6(net, train_data, test_data, batch_size, num_epochs, lr, d2l.try_gpu())
print('finish')

torch.save(net.state_dict(), 'LeNet.params')
