import pandas as pd
import torch
from torch import nn
from util import d2lHelper as d2l

# 读取数据集
path = './mnist_dataset/mnist_train.csv'
train_data = pd.read_csv(path, header=None)

# 初始化参数
batch_size = 256
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
