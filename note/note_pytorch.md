# pytorch
参考资料： [$torch$官方文档](https://pytorch.org/docs/stable/torch.html)
使用： `import torch`

## torch 常用函数
### torch.arrange
  - 函数原型： `torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) `
  - 例：
  ```python
  >>> torch.arange(5)
  tensor([ 0,  1,  2,  3,  4])
  >>> torch.arange(1, 4)
  tensor([ 1,  2,  3])
  >>> torch.arange(1, 2.5, 0.5)
  tensor([ 1.0000,  1.5000,  2.0000])
  ```

### torch.normal
  - 函数原型： `normal(mean=0.0, std=1.0, *, generator=None, out=None)`
  - 例： `x = torch.normal(mean=0, std=1, size=(1, 5))`
  - 该函数返回独立服从正态分布 $N(mean, std)$ 的随机数张量
  - $size$ 为输出张量的 $shape$
  - $generator$ 为 $torch.generator$
  - $mean$ & $std$ 可以是 $tensor$ (当维数不一致时以 $mean$ 维数为准)
  - 例：
  ```python
  >>> torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
  tensor([  1.0425,   3.5672,   2.7969,   4.2925,   4.7229,   6.2134, 8.0505,   8.1408,   9.0563,  10.0566])
  >>> torch.normal(mean=0.5, std=torch.arange(1., 6.))
  tensor([-1.2793, -1.0732, -2.0687,  5.1177, -1.2303])
  >>> torch.normal(mean=torch.arange(1., 6.))
  tensor([ 1.1552,  2.6148,  2.6535,  5.8318,  4.2361])
  >>> torch.normal(2, 3, size=(1, 4))
  tensor([[-1.3987, -1.9544,  3.6048,  0.7909]])
  ```

### torch.zeros
  - 函数原型：`torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)`
  - 例：
  ```python
  >>> torch.zeros(2, 3)
  tensor([[ 0.,  0.,  0.],
          [ 0.,  0.,  0.]])

  >>> torch.zeros(5)
  tensor([ 0.,  0.,  0.,  0.,  0.])
  ```


## torch.tensor
### $requires\_grad$
  - 使用 `requires_grad=True` 创建的 $tensor$ 会被 `torch.autograd` 记录并计算自动微分

## torch.nn
  - 使用：  `from torch import nn`
  - 简介： `torch.nn`是`pytorch`中自带的一个函数库，里面包含了神经网络中使用的一些常用函数。
  ### nn.Conv2d

  ### nn.Sequential

  ### nn.Flatten

  ### nn.Linear
