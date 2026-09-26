import torch
from sympy.tensor.array.arrayop import Flatten
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            nn.Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        output = self.model1(input)
        return output

writer = SummaryWriter("model")

real_model = Model()
input = torch.ones((64,3,32,32))
output = real_model(input)

writer.add_graph(real_model,input)
print(output.shape)


writer.close()