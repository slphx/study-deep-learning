# Pandas 的使用
使用：  `import pandas as pd`

## pd.read_csv
 - `train_data = pd.read_csv(path, header=None)`
    取消第一行作表头

 - `pd.read_csv(path, name=['a','b'])`
    为各个字段取名
 - `pd.read_csv(path, name=['a','b'], index_col='a')`
    将某一字段设为索引
 - `pd.read_csv(path, sep=',')`
    设置分割符


## data.iloc & data.loc
  通过 $iloc$ (基于索引号) $loc$ (基于标签) 选择数据

## 处理缺失值
  - 删除法，直接忽略缺失值
  - 插值法，`inputs = inputs.fillna(inputs.mean())`

## 转换为张量
  `X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)`
