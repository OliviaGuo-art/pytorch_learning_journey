import torchvision
from sympy.printing.pytorch import torch
from torch import nn

from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Dataloader import targets
from Dataset import train_dataset

dataset = torchvision.datasets.CIFAR10("data(CIFAR10)", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, 64)

class Conv_learning(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 6, 3, 1, 0)      #这里Conv2d一定不要写成小写，很容易错！

    def forward(self, x):
        y = self.conv1(x)
        return y


idk = Conv_learning()  #注意这个forward里的参数x，它是imgs也就是每个batch的张量，而不是整个dataloader得到的直接往里塞

writer = SummaryWriter("conv_learning")

step = 0
for data in dataloader:
    imgs, targets = data
    output = idk(imgs)

   #print(imgs.shape)
   #print(output.shape)

    writer.add_images("input", imgs, step)

    #output = torch.reshape(output, (-1, 3, 30, 30))
    #writer.add_images("output", output, step)

    step += 1