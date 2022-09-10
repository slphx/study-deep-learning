import torch
from torch import nn
from pathlib import Path
import gradio as gr

model = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

state_dict = torch.load('LeNet.params')
model.load_state_dict(state_dict, strict=False)
model.eval()
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def predict(img):
    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    print(x)
    with torch.no_grad():
        out = model(x)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    values, indices = torch.topk(probabilities, 5)
    confidences = {LABELS[i]: v.item() for i, v in zip(indices, values)}
    return confidences


gr.Interface(fn=predict, inputs="sketchpad", outputs="label", live=True).launch()
