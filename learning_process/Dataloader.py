#本节中，试用torchvision中的数据集进行学习

import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

'''import os, hashlib

root = "data(CIFAR10)"
print("程序认为的路径:", os.path.abspath(root))

if os.path.exists(root):
    print("文件夹里现在有:", os.listdir(root))
else:
    print("这个文件夹根本不存在")

tar = os.path.join(root, "cifar-10-python.tar.gz")
if os.path.exists(tar):
    md5 = hashlib.md5(open(tar, "rb").read()).hexdigest()
    print("压缩包 MD5:", md5)
    print("正确应该是: c58f30108f718f92721af3b95e74349a")
    print("是否一致:", md5 == "c58f30108f718f92721af3b95e74349a")
else:
    print("压缩包不在这个路径下")'''




data_train = torchvision.datasets.CIFAR10("../data(CIFAR10)", train= True, transform= torchvision.transforms.ToTensor(), download= True)
data_test = torchvision.datasets.CIFAR10("../data(CIFAR10)", train= False, transform= torchvision.transforms.ToTensor())

#一般训练集打乱，测试集不用打乱
data_train_loader =  DataLoader(data_train, batch_size= 64, shuffle= True, num_workers=0, drop_last= True)
data_test_loader =  DataLoader(data_test, batch_size= 64, shuffle= False, num_workers=0, drop_last= True)

#img, target = data_train[0]   这里是说每一个训练对象输出是img和target


#这里仅以测试集为例
writer = SummaryWriter("../dataloader")
#想做到每一步读入64张图片
step = 0
for epoch in range(2):       #epoch就是对于一个数据集，模型要训练几遍
    for data in data_test_loader:
        imgs, targets = data
        writer.add_images("test_images", imgs, step)   #一次传入一batch的图片要用add_images哦
        step += 1

writer.close()

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator("../dataloader")
ea.Reload()

print("所有 tag:", ea.Tags())

for tag in ea.Tags().get("images", []):
    events = ea.Images(tag)
    steps = [e.step for e in events]
    print(f"\nTag: {tag}")
    print(f"step 数量: {len(steps)}")
    print(f"最小 step: {min(steps)}")
    print(f"最大 step: {max(steps)}")
    print(f"前 10 个 step: {steps[:10]}")