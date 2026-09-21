from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)   #把 self.path 这个目录下的所有文件和子目录的名字，列成一个列表返回。

    def __getitem__(self, idx):    #idx就是序号
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name) #具体到每一个的路径
        img = Image.open(img_item_path)
        label = self.label_dir   #这里比较特殊，因为是分类问题，所以文件名字就是标签名字   标签单独放在一个文件里的就不能这么写了
        return img, label

    def __len__(self):
        return len(self.img_path)



#构建实例
root_dir = "Dataset/hymenoptera_data/train"      #windows要改成/
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset




#思考，标签单独放在一个文件里的怎么写