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
    # 这里，我们使用一个11*11的更大窗口来捕捉对象。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=1),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=1),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(5*5*256, 1024), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1024, 1024), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(1024, 10))

batch_size = 256
num_epochs = 10
lr = 0.01

print('train data')
d2l.train_ch6(net, train_data, test_data, batch_size, num_epochs, lr, d2l.try_gpu())
print('finish')

torch.save(net.state_dict(), 'AlexNet.params')
