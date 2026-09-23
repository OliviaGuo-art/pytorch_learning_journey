import torch
import torchvision
from torch import nn
from torch.ao.nn.quantized import Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("data(CIFAR10)", train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, 64)

class NLA_learning(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sigmoid1 = nn.Sigmoid()

    def forward(self, input):
        output = self.Sigmoid1(input)
        return output


idk3 = NLA_learning()

writer = SummaryWriter("NLA_learning")

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)

    output = idk3(imgs)
    writer.add_images("output", output, step)

    step += 1

writer.close()