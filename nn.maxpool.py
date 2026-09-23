import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("data(CIFAR10)", train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, 64)

class Maxpool_learning(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3)


    def forward(self, input):
        output = self.maxpool1(input)
        return output

idk2 = Maxpool_learning()

writer = SummaryWriter("maxpool_learning")

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)

    output = idk2(imgs)

    writer.add_images("output", output, step)

    step += 1


writer.close()

