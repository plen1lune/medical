import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import glob
import torch.optim as optim
from model import rresnet18, resnet34, resnet50, resnet101
import torchvision.models.resnet  # ctrl+鼠标左键点击即可下载权重
import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([  # transforms.RandomResizedCrop(224),
        # transforms.RandomResizedCrop((512, 384)),
        # transforms.Resize((512, 512)),
        transforms.CenterCrop((440, 330)),
        transforms.Resize((512, 384)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.5),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        # transforms.ColorJitter(contrast=0.5, saturation=0.5, hue=0.5),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.067, 0.068, 0.072], [0.151, 0.156, 0.161])]),  # 和官网初始化方法保持一
    "test": transforms.Compose([  # transforms.Resize((224,224)),
        # transforms.CenterCrop(224),
        transforms.CenterCrop((440, 330)),
        transforms.Resize((512, 384)),
        # transforms.ColorJitter(brightness=(0.19, 0.2), contrast=(0.49, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.067, 0.068, 0.072], [0.151, 0.156, 0.161])])}
# transforms.Normalize([0.067, 0.068, 0.072], [0.153, 0.159, 0.164])])}

# data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
data_root = "/home/xjl/test_lky/HT100/video_png3"
image_path = data_root + "/"  # data set path

train_dataset = datasets.ImageFolder(root=image_path + "train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)
print(train_num)
# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "test",
                                        transform=data_transform["test"])
val_num = len(validate_dataset)
print(val_num)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

net = resnet34()  # 一开始不能设置全连接层的输出种类为自己想要的，必须先将模型参数载入，再修改全连接层
# print(net)
# 官方提供载入预训练模型的方法
model_weight_path = "./weight/resnet34.pth"  # 1权重路径:预训练权重512 结果为87
# model_weight_path = "/home/llc/acc80/path3/raw/resNet34_1_acc0.887.pth"  # 2，续1，第一次训练个epoch.2颜色变换 512 结果不行
# model_weight_path = "/home/llc/acc80/path3/raw/resNet34_1_acc0.812.pth"  # 3，续1，第二次训练20个epoch.2颜色变换 512 结果不行
# model_weight_path = "./resnet34-pre.pth"  # 4，权重路重预训练权重512_128
# model_weight_path = "/mnt/nas/llc/acc80/path3/raw/resNet18_1_acc0.909.pth"  # 5，权重路重预训练权重512 .2

# model_weight_path = "/mnt/nas/llc/acc80/path3/b/resNet18_1_acc0.949.pth"

missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)  # 载入模型权重
inchannel = net.fc.in_features

# net.fc = nn.Linear(inchannel, 4)  # 重新确定全连接层

net.fc = nn.Sequential(
    nn.Linear(inchannel, 512),
    nn.ReLU(),
    # nn.Dropout(0.8),
    # nn.Linear output layer
    # nn.Linear(512, 4),
    nn.Linear(512, 3),
    nn.LogSoftmax(dim=1)
)

net.to(device)

loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.0001)
best_acc = 0.0
learningrate = 0.0001
for epoch in range(100):
    # train

    decay = 0.95
    # if epoch > 10 and epoch % 20 == 0 and acc_val > 0.85:
    #     learningrate = learningrate * decay
    # optimizer = torch.optim.SGD(net.parameters(), lr=learningrate, momentum=0.5)
    optimizer = optim.Adam(net.parameters(), lr=learningrate)

    net.train()  # 控制BN层状态

    running_loss = 0.0
    acc = 0.0
    acc_m = np.zeros(15)
    # acc0 = acc1 = acc2 = acc3 = acc4 = acc5 = acc6 = acc7 = acc8 = acc9 = acc10 = acc11 = acc12 = acc13 = acc14 = 0.0
    num_m = np.zeros(15)
    # num0 = num1 = num2 = num3 = num4 = num5 = num6 = num7 = num8 = num9 = num10 = num11 = num12 = num13 = num14 = 0

    # acc_metrix = [[0--0, 0--1, 0--2, 0--3],
    #          [1--0, 1--1, 1--2,1--3],
    #          [2--0, 2--1, 2--2, 2--3],
    #          [4--0, 4--1, 4--2, 4--3]]

    """metrix = [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0, 0]]"""
    metrix = np.zeros((15, 15))
    metrix_val = np.zeros((15, 15))
    """metrix_val = [[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]]"""
    # outputs = 0
    for step, (images, labels) in enumerate(train_loader, start=0):
        # images, labels = data
        # print(data)
        optimizer.zero_grad()
        logits, _ = net(images.to(device))

        loss = loss_function(logits, labels.to(device))

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        outputs, _ = net(images.to(device))  # eval model only have last output layer
        # loss = loss_function(outputs, test_labels)
        predict_y = torch.max(outputs, dim=1)[1]
        # print(labels.to(device))
        for i in range(0, len(labels.to(device))):
            # sample_fname, _ = train_loader.dataset.samples[step * batchsize + i]
            # print(sample_fname)
            # print(labels.to(device)[i])
            metrix[labels.to(device)[i]][predict_y[i]] += 1
        for i in range(15):
            acc_m[i] = metrix[i][i]
        """acc0 = metrix[0][0]
        acc1 = metrix[1][1]
        acc2 = metrix[2][2]"""
        for i in range(15):
            num_m[i] = np.sum(metrix[i])
        """num0 = metrix[0][0] + metrix[0][1] + metrix[0][2]
        num1 = metrix[1][0] + metrix[1][1] + metrix[1][2]
        num2 = metrix[2][0] + metrix[2][1] + metrix[2][2]"""
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
    """acc = acc0 + acc1 + acc2"""
    acc = np.sum(acc_m)
    accurate = acc / train_num
    """print((num0 + num1 + num2 + num3))
    print(train_num)"""

    print("\n2D_2C: %.3f " % (acc_m[0] / num_m[0]))
    print("\n2D_3C: %.3f " % (acc_m[1] / num_m[1]))
    print("\n2D_4C: %.3f " % (acc_m[2] / num_m[2]))
    """print("\n2D_5C: %.3f " % (acc_m[3] / num_m[3]))
    print("\n2D_A: %.3f " % (acc_m[4] / num_m[4]))
    print("\n2D_BX: %.3f " % (acc_m[5] / num_m[5]))
    print("\n2D_P: %.3f " % (acc_m[6] / num_m[6]))
    print("\n2D_RVI: %.3f " % (acc_m[7] / num_m[7]))
    print("\n2D_SAA: %.3f " % (acc_m[8] / num_m[8]))
    print("\n2D_SAB: %.3f " % (acc_m[9] / num_m[9]))
    print("\n2D_SAM: %.3f " % (acc_m[10] / num_m[10]))
    print("\n2D_SAoA: %.3f " % (acc_m[11] / num_m[11]))
    print("\n2D_SSF: %.3f " % (acc_m[12] / num_m[12]))
    print("\n3D: %.3f " % (acc_m[13] / num_m[13]))
    print("\nCDFI: %.3f " % (acc_m[14] / num_m[14]))"""

    print("\ntrain_acc: %.3f " % accurate)
    print(metrix)
    print()

    # validate
    net.eval()  # 控制BN层状态
    acc_val = 0.0  # accumulate accurate number / epoch
    """acc_val0 = 0.0
    acc_val1 = 0.0
    acc_val2 = 0.0"""
    acc_val_m = np.zeros(15)

    """num_val0 = 0
    num_val1 = 0
    num_val2 = 0"""
    num_val_m = np.zeros(15)

    with torch.no_grad():
        for step2, val_data in enumerate(validate_loader, start=0):
            print(step2)
            val_images, val_labels = val_data
            outputs1, _ = net(val_images.to(device))  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs1, dim=1)[1]
            for i in range(0, len(val_images.to(device))):
                val_sample_fname, _ = validate_loader.dataset.samples[step2 * batch_size + i]
                # print(val_sample_fname)
                # print(val_labels.to(device)[i])
                metrix_val[val_labels.to(device)[i]][predict_y[i]] += 1
            """acc_val0 = metrix_val[0][0]
            acc_val1 = metrix_val[1][1]
            acc_val2 = metrix_val[2][2]"""
            for i in range(15):
                acc_val_m[i] = metrix_val[i][i]

            """num_val0 = metrix_val[0][0] + metrix_val[0][1] + metrix_val[0][2]
            num_val1 = metrix_val[1][0] + metrix_val[1][1] + metrix_val[1][2]
            num_val2 = metrix_val[2][0] + metrix_val[2][1] + metrix_val[2][2]"""
            for i in range(15):
                for j in range(15):
                    num_val_m[i] = np.sum(metrix_val[i])

        acc_val = np.sum(acc_val_m)
        val_accurate = acc_val / val_num
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

        """如果测试集的准确度比历史最佳还高，那么更新json文件，保存权"""
        if val_accurate > best_acc and val_accurate > 0.90:
            if epoch > 10:
                learningrate = learningrate * decay
            # with open(json_path, 'w') as f:
            # f.truncate()
            json_context = {}
            """json_path = os.path.join("./cl3/b", "clasify18_%.3f.json" % val_accurate)
            best_acc = val_accurate
            save_path = './path3/b/resNet18_acc%.3f.pth' % val_accurate
            model_path = './model3/b/resNet18_acc%.3f.pth' % val_accurate"""
            json_path = os.path.join("./cl/resnet34", "3classresnet34_3_crop_acc%.3f.json" % val_accurate)
            best_acc = val_accurate
            save_path = './path/resnet34/3classresnet34_3_crop_acc%.3f.pth' % val_accurate
            model_path = './model/resnet34/3classresnet34_3_crop_acc%.3f.pth' % val_accurate
            torch.save(net.state_dict(), save_path)
            torch.save(net, model_path)
            """for step2, val_data in enumerate(validate_loader, start=0):

                val_images, val_labels = val_data
                outputs1 = net(val_images.to(device))  # eval model only have last output layer
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs1, dim=1)[1]
                for i in range(0, len(val_images.to(device))):
                    #此处实验原理是逐个（i）取得文件名，真值和预测值，以文件名为索引，【真值，预测值】为标签，写入json
                    val_sample_fname, _ = validate_loader.dataset.samples[step2 * batch_size + i]
                    #print(val_sample_fname)  # 打印文件
                    # print(val_labels.to(device)[i])
                    # 此处的意思是以文件名为索引，【真值，预测值】为标签，写入json
                    # metrix_val[val_labels.to(device)[i]][predict_y[i]] += 1
                    json_context[val_sample_fname] = (val_labels.to(device)[i].item(), predict_y[i].item())
                    #print(json_context[val_sample_fname])
                with open(json_path, 'w') as f:
                    json_str = json.dumps(json_context, indent=4, ensure_ascii=False)
                    f.write(json_str)"""
        # 仅在历史最佳时提供混淆矩阵，各类精度和总体精度
        """print("\nlight: %.3f " % (acc_val_m[0] / num_val_m[0]))
        print("\nmedium: %.3f " % (acc_val_m[1] / num_val_m[1]))
        print("\nserious: %.3f " % (acc_val_m[3] / num_val_m[2]))"""
        print("\n2D_2C: %.3f " % (acc_val_m[0] / num_val_m[0]))
        print("\n2D_3C: %.3f " % (acc_val_m[1] / num_val_m[1]))
        print("\n2D_4C: %.3f " % (acc_val_m[2] / num_val_m[2]))
        # print("\nlight: %.3f " % (acc_val_m[3] / num_val_m[3]))
        """print("\n2D_5C: %.3f " % (acc_val_m[3] / num_val_m[3]))
        print("\n2D_A: %.3f " % (acc_val_m[4] / num_val_m[4]))
        print("\n2D_BX: %.3f " % (acc_val_m[5] / num_val_m[5]))
        print("\n2D_P: %.3f " % (acc_val_m[6] / num_val_m[6]))
        print("\n2D_RVI: %.3f " % (acc_val_m[7] / num_val_m[7]))
        print("\n2D_SAA: %.3f " % (acc_val_m[7] / num_val_m[8]))
        print("\n2D_SAB: %.3f " % (acc_val_m[8] / num_val_m[9]))
        print("\n2D_SAM: %.3f " % (acc_val_m[10] / num_val_m[10]))
        print("\n2D_SAoA: %.3f " % (acc_val_m[11] / num_val_m[11]))
        print("\n2D_SSF: %.3f " % (acc_val_m[12] / num_val_m[12]))
        print("\n3D: %.3f " % (acc_val_m[13] / num_val_m[13]))
        print("\nCDFI: %.3f " % (acc_val_m[14] / num_val_m[14]))"""

        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))
        # print(metrix_val)
print('Finished Training')
