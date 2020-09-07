import torch
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms ,models
import os
import os.path
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
# 如果使用pip安装的Efficient的话这样导入
from efficientnet_pytorch import EfficientNet

# 超参数设置
batchsize= 32
num_epochs = 5
class_num = 9691
net_name = 'efficientnet-b3'
device = torch.device("cuda:0")
train_data_dir = 'F:\\product10kdata\\train'


# 加载模型
effmodel = EfficientNet.from_pretrained('efficientnet-b3')

# 修改全连接层
num_ftrs = effmodel._fc.in_features
effmodel._fc = nn.Linear(num_ftrs, class_num)
# 定义损失函数和优化器
effmodel = effmodel.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(effmodel.parameters())



# 定义transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer_ImageNet = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

test_transformer_ImageNet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


class MyDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]


def split_train_test_Data(data_dir, ratio):
    """ the sum of ratio must equal to 1"""
    dataset = ImageFolder(data_dir)     # data_dir精确到分类目录的上一级
    print(len(dataset))
    character = [[] for i in range(len(dataset.classes))]
    #print(dataset.class_to_idx)
    for x, y in dataset.samples:  # 将数据按类标存放  x 图片 y 类别
        character[y].append(x)
    train_inputs, test_inputs = [], []
    train_labels, test_labels = [], []
    for i, data in enumerate(character):  # data为一类图片
        num_sample_train = int(len(data) * ratio[0])
        # print(num_sample_train)
        # num_sample_test = int(len(data) * ratio[1])
        num = len(data)
        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:num]:
            test_inputs.append(str(x))
            test_labels.append(i)
    trainset = MyDataset(train_inputs, train_labels, train_transformer_ImageNet)
    trainsize = len(trainset)
    train_dataloader = DataLoader(trainset,batch_size=batchsize, shuffle=True)
    testset = MyDataset(test_inputs, test_labels, test_transformer_ImageNet)
    testsize = len(testset)
    test_dataloader = DataLoader(testset,batch_size=batchsize, shuffle=False)

    return train_dataloader, test_dataloader, trainsize, testsize

def train(model,train_data,criterion,optimizer,num_epochs,trainsize):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    record = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):  # 训练epochs轮
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, num_epochs))

        model.train()  # 训练
        train_loss = 0.0
        train_acc = 0.0
        best_model_wts = model.state_dict()
        best_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):
            print(i)
            inputs = inputs.to(device)
            labels = labels.to(device)
            print(labels)
            # print(labels)
            # 记得清零
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / trainsize
        avg_train_acc = train_acc / trainsize

        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100,
                epoch_end - epoch_start))

        if avg_train_acc > best_acc:
            best_acc = avg_train_acc
            best_model_wts = model.state_dict()
            best_epoch = epoch + 1

    # save best model
    save_dir = 'model'
    os.makedirs(save_dir, exist_ok=True)
    model.load_state_dict(best_model_wts)
    model_out_path = save_dir + "\\" + 'bestmodel' + '.pth'
    torch.save(model, model_out_path)

    print("Best Accuracy for test : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))



def test(model,test_data,criterion,optimizer,testsize):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        test_acc = 0
        for i, (inputs, labels) in enumerate(test_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            ret, predictions = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            test_acc += acc.item() * inputs.size(0)

        avg_test_loss = test_loss / testsize
        avg_test_acc = test_acc / testsize

        print('Loss: {:.4f} Acc: {:.4f}%'.format(avg_test_loss,avg_test_acc * 100))





if __name__ == '__main__':
    train_data, test_data,trainsize,testsize = split_train_test_Data(train_data_dir, [0.8, 0.2])
    print(testsize,trainsize)
    train(effmodel,train_data,criterion,optimizer,num_epochs,trainsize)
    test(effmodel,test_data,criterion,optimizer,testsize)


























