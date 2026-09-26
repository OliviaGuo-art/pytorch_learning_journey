#transforms就像一个工具箱，里面有各种各样处理图片的工具，我们通常使用这些工具达到我们的目的

from PIL import Image
from tensorboard.summary.v1 import image
from torch.utils.tensorboard import SummaryWriter   #一定要记得这个writer是用来可视化的，不是我们的读入图片集哦
from torchvision import transforms


image.path = "../练手数据集/Data/train/bees/16838648_415acd9e3f.jpg"
img = Image.open(image.path)

writer = SummaryWriter("../logs")

#把PIL或者numpy格式的图片都可以转为tensor形式  为什么python中要用tensor格式呢 事实上，查看tensor属性可以发现，它带有很多神经网络训练的参数
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img)

writer.add_image("img_Tensor", img_tensor)
writer.close()