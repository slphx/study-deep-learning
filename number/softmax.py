import pandas as pd
import pytorch

path = './mnist_dataset/mnist_train.csv'
train_data = pd.read_csv(path, header=None) #取消第一行作表头
