import torch.optim
import torchvision
from torch import nn, no_grad
from torch.nn import Sequential, Linear, MaxPool2d, Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



dataset_test = torchvision.datasets.CIFAR10("../data(CIFAR10)", train=False, transform=torchvision.transforms.ToTensor())
dataloader_test = DataLoader(dataset_test, 64)

dataset_train = torchvision.datasets.CIFAR10("../data(CIFAR10)", train=True, transform=torchvision.transforms.ToTensor())
dataloader_train = DataLoader(dataset_train, 64)

class Loss_network(nn.Module):
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


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in dataloader:
            outputs1 = model(imgs)
            _, predicted = torch.max(outputs1, 1)       #在outputs的第一维里找最大值，也就是概率值那行  _在这里是忽略掉的意思
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total



writer = SummaryWriter("../final")


idk7 = Loss_network()

loss = nn.CrossEntropyLoss()
optim1 = torch.optim.SGD(idk7.parameters(), lr=0.01)

for epoch in range(20):
    idk7.train()
    running_loss = 0.0
    for data in dataloader_train:
        imgs, targets = data
        outputs2 = idk7(imgs)
        result_loss = loss(outputs2, targets)
        optim1.zero_grad()
        result_loss.backward()
        optim1.step()   #参数更新
        running_loss = running_loss + result_loss

    acc = evaluate(idk7, dataloader_test)

    writer.add_scalar("loss_lr=0.01", running_loss.item(), epoch)
    writer.add_scalar("acc_lr=0.01", acc, epoch)



'''optim2 = torch.optim.SGD(idk7.parameters(), lr=0.001)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = idk7(imgs)
        result_loss = loss(outputs, targets)
        optim2.zero_grad()
        result_loss.backward()
        optim2.step()   #参数更新
        running_loss = running_loss + result_loss
    writer.add_scalar("loss_lr=0.001", running_loss.item(), epoch)'''


writer.close()