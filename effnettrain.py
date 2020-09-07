import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from efficientnet_pytorch import EfficientNet


# 需要分类的数目
num_classes = 20
# 批处理尺寸
batchsize = 64
# 训练多少个epoch
epoch = 1

device = torch.device("cpu")

train_transforms = transforms.Compose(
        [transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),#随机裁剪到256*256
        transforms.RandomRotation(degrees=15),#随机旋转
        transforms.RandomHorizontalFlip(),#随机水平翻转
        transforms.CenterCrop(size=224),#中心裁剪到224*224
        transforms.ToTensor(),#转化成张量
        transforms.Normalize([0.485, 0.456, 0.406],#归一化
                             [0.229, 0.224, 0.225])
])
train_directory= 'F:\\product10kproject\\strain'
train_datasets = datasets.ImageFolder(train_directory, transform=train_transforms)
train_data_size = len(train_datasets)
train_data = torch.utils.data.DataLoader(train_datasets, batch_size=batchsize, shuffle=True)
print(train_data_size)

test_transforms = transforms.Compose(        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
         [0.229, 0.224, 0.225])])
test_directory = 'F:\\product10kproject\\test'
test_datasets = datasets.ImageFolder(test_directory,transform=test_transforms)
test_data_size = len(test_datasets)
test_data = torch.utils.data.DataLoader(test_datasets, batch_size=batchsize, shuffle=True)





# EfficientNet的使用和微调方法
model = EfficientNet.from_name('efficientnet-b3')
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs,num_classes)


# model = model.to('cuda:0')
model = model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

def train_test(model, loss_function, optimizer, epochs):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 若有gpu可用则用gpu
    #device = torch.device("cpu")
    record = []
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()  # 训练
        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0


        for i, (inputs, labels) in enumerate(train_data):
            print(i)
            inputs = inputs.to(device)
            labels = labels.to(device)
            print(labels)
            # 记得清零
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()
            # 每训练1个batch打印一次loss和准确率
            # train_loss += loss.item()
            # _, predictions = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += predictions.eq(labels.data.view_as(predictions))
            # correct_sum += correct
            # print('[epoch:%d] Loss: %.03f | Acc: %.3f%% '
            #        % (epoch + 1,  sum_loss / (i + 1),
            #           100. * float(correct) / float(total)))

            train_loss += loss.item() * inputs.size(0)
            print('trainloss: ', loss.item())
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            #print(acc.item())
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()  # 验证
            for j, (inputs, labels) in enumerate(test_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_function(outputs, labels)
                print('testloss: ',loss.item())
                test_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                test_acc += acc.item() * inputs.size(0)



        # avg_train_loss = sum_loss / train_data_size
        # avg_train_acc = correct / train_data_size
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size
        avg_test_loss = test_loss / test_data_size
        avg_test_acc = test_acc / test_data_size

        record.append([avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc])
        if avg_test_acc > best_acc  :#记录最高准确性的模型
            best_acc = avg_test_acc
            best_epoch = epoch + 1

        epoch_end = time.time()
        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\ttest: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_test_loss, avg_test_acc * 100,
                epoch_end - epoch_start))
        print("Best Accuracy for test : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))


    return model, record




if __name__ == '__main__':
    trained_model, record = train_test(model, loss_func, optimizer, epoch)

    record = np.array(record)
    plt.plot(record[:, 0:2])
    plt.legend(['Train Loss', 'test Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 10)
    plt.savefig('loss.png')
    plt.show()

    plt.plot(record[:, 2:4])
    plt.legend(['Train Accuracy', 'test Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('accuracy.png')
    plt.show()






















