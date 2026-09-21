#RensorBoard是用来可视化一些训练参数的，比如损失

from torch.utils.tensorboard import SummaryWriter   #可以在控制台用help查看SummaryWriter的属性
                                                    #python中可以按下ctrl查看
import numpy as np
from PIL import Image


writer = SummaryWriter("logs")   #就是随便起名字
image_path = "练手数据集/Data/train/ants/0013035.jpg"
img_PIL = Image.open(image_path)      #在控制台运行我们需要的行可以查看其属性，这里就发现它是PIL
img_array = np.array(img_PIL)

writer.add_image("train_ants", img_array, 1, dataformats='HWC') #怎么打开tensorboard  首先先选中logs，然后在终端点那个光标，打tensorboard --logdir(logdir=事件所在文件夹名）,弹出来一个链接打开即可
#这里不同的global_step   记录在训练多少多少步的时候，图像是怎么样的
#add_image其实还是有很多用法的，此处没有列出，实战用到的时候去查吧


#真实训练中肯定不是这样的，此处展示参数
for i in range(100):
    writer.add_scalar("y = 2x", 2*i, i )
writer.close()
