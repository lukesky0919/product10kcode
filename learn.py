from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import os
import os.path

data_dir='train\\'
ratio=[0.8,0.2]
class myimagefolder(ImageFolder):
    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes = [int(x) for x in classes]
        classes.sort()
        classes = [str(x) for x in classes]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

dataset = myimagefolder(data_dir)  # data_dir精确到分类目录的上一级
character = [[] for i in range(len(dataset.classes))]
for x, y in dataset.samples:  # 将数据按类标存放  x 图片 y 类别
    character[y].append(x)
print(character[3])


train_inputs, test_inputs = [], []
train_labels, test_labels = [], []
for i, data in enumerate(character):  # data为一类图片
    print(i)
    num_sample_train = int(len(data) * ratio[0])
    # print(num_sample_train)
    # num_sample_test = int(len(data) * ratio[1])
    num = len(data)
    for x in data[:num_sample_train]:
        print(x)
        print(str(x))
        train_inputs.append(str(x))
        train_labels.append(i)
    for x in data[num_sample_train:num]:
        test_inputs.append(str(x))
        test_labels.append(i)
