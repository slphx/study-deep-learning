import pandas as pd
import torch
from torch import nn
from util import pro_number_helper as d2l


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


# 读取数据集
print('read data')
path1 = './mnist_dataset/mnist_train.csv'
path2 = './mnist_dataset/mnist_test.csv'
train_data = pd.read_csv(path1, header=None)
test_data = pd.read_csv(path1, header=None)
print('finish')

# 初始化参数
batch_size = 256
num_epochs = 10
net = nn.Sequential(nn.Linear(784, 10))
net.apply(init_weights)


loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

print('train data')
d2l.train_ch3(net, train_data, test_data, batch_size, loss, num_epochs, trainer)
print('finish')

torch.save(net.state_dict(), 'softmax.params')