import pandas as pd

path = './mnist_dataset/mnist_train.csv'
train_data = pd.read_csv(path, header=None) #取消第一行作表头

# pd.read_csv(path, name=['a','b']) 为各个字段取名
# pd.read_csv(path, name=['a','b'], index_col='a') 将某一字段设为索引
# pd.read_csv(path, sep=',') 设置分割符


# 通过 iloc(基于索引号) loc(基于标签) 选择数据