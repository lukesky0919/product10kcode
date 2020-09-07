import csv
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from efficientnet_pytorch import EfficientNet
class_num=9691
# effmodel = EfficientNet.from_pretrained('efficientnet-b3')
# # 修改全连接层
# num_ftrs = effmodel._fc.in_features
# effmodel._fc = nn.Linear(num_ftrs, class_num)
modelft_file = 'model\\bestmodel.pth'
effmodel = torch.load(modelft_file).cuda()


data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_key(dct, value):

    return [k for (k, v) in dct.items() if v == value]


def predict(model, pre_image_name,mapping):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #载入图片
    img = Image.open(pre_image_name)
    inputs = data_transforms(img)
    inputs.unsqueeze_(0)
    model = model.to(device)
    model.eval()
    inputs = inputs.to(device)

    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)
    # print(preds.item())
    class_name = get_key(mapping,preds.item())

    return class_name[0]






if __name__ == '__main__':
    test_dir = 'test'
    data = datasets.ImageFolder('F:\\product10kdata\\train')
    mapping = data.class_to_idx
    picname= []
    with open('test.csv', "rt", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    preclass = []
    for i,a in enumerate(rows):
        print(i, a[0])
        picname.append(a[0])
        cls = predict(effmodel,test_dir + '\\' + a[0],mapping)
        # print(cls)
        preclass.append(cls)

    dataframe = pd.DataFrame({'name': picname, 'class': preclass})
    dataframe.to_csv("pre2.csv", index=False, sep=',')






