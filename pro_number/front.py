import torch
from torch import nn
from pathlib import Path
import gradio as gr

# model = nn.Sequential(
#     nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.BatchNorm2d(6), nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     nn.Flatten(),
#     nn.Linear(16 * 5 * 5, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
#     nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
#     nn.Linear(84, 10))
# state_dict = torch.load('LeNet.params')

model = nn.Sequential(
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
state_dict = torch.load('AlexNet.params')

model.load_state_dict(state_dict, strict=False)
model.eval()
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def predict(img):
    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    values, indices = torch.topk(probabilities, 5)
    confidences = {LABELS[i]: v.item() for i, v in zip(indices, values)}
    return confidences


gr.Interface(fn=predict, inputs="sketchpad", outputs="label", live=True).launch()
