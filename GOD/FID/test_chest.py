import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import glob
import torch.optim as optim
from model import rresnet18, resnet34, resnet101, resnet50
import torchvision.models.resnet  # ctrl+鼠标左键点击即可下载权重
import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.Resize((512, 512)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # 和官网初始化方法保持一致
    "valid": transforms.Compose([transforms.Resize((512, 512)),
                                 # transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.905, 0.759, 0.870], [0.115, 0.205, 0.120])]),
    "chest_resulttest": transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.Resize((512, 512)),
                                # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.905, 0.759, 0.870], [0.115, 0.205, 0.120])])
}

# data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
data_root = "/u/hvg7cb/UVA"
image_path = data_root + "/ddpm/"  # data set path

train_dataset = datasets.ImageFolder(root=image_path + "chest_resulttest",
                                     transform=data_transform["chest_resulttest"])
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "chest_resulttest",
                                        transform=data_transform["chest_resulttest"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

net = rresnet18()  # 一开始不能设置全连接层的输出种类为自己想要的，必须先将模型参数载入，再修改全连接层
inchannel = net.fc.in_features

# net.fc = nn.Linear(inchannel, 4)  # 重新确定全连接层

net.fc = nn.Sequential(
    nn.Linear(inchannel, 512),
    nn.ReLU(),
    # nn.Dropout(0.7),
    # nn.Linear output layer
    nn.Linear(512, 2),
    nn.LogSoftmax(dim=1)
)

# 官方提供载入预训练模型的方法
model_weight_path = "/u/hvg7cb/UVA/gan/path/chest_acc0.958.pth"  # 权重路径
# missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)  # 载入模型权重
net.load_state_dict(torch.load(model_weight_path), strict=False)  # 载入模型权重

net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
best_acc = 0.0

running_loss = 0.0

# validate
net.eval()  # 控制BN层状态
acc_val = 0.0  # accumulate accurate number / epoch
acc_val0 = 0.0
acc_val1 = 0.0


num_val0 = 0
num_val1 = 0


metrix_val = [[0, 0],
              [0, 0]
              ]
with torch.no_grad():
    for step2, val_data in enumerate(validate_loader, start=0):
        print(step2)
        val_images, val_labels = val_data
        outputs1 = net(val_images.to(device))  # eval model only have last output layer
        # loss = loss_function(outputs, test_labels)
        predict_y = torch.max(outputs1, dim=1)[1]

        for i in range(0, len(val_images.to(device))):
            val_sample_fname, _ = validate_loader.dataset.samples[step2 * batch_size + i]
            # print(val_sample_fname)
            # print(val_labels.to(device)[i])
            metrix_val[val_labels.to(device)[i]][predict_y[i]] += 1
        acc_val0 = metrix_val[0][0]
        acc_val1 = metrix_val[1][1]


        num_val0 = metrix_val[0][0] + metrix_val[0][1]
        num_val1 = metrix_val[1][0] + metrix_val[1][1]


    acc_val += acc_val0 + acc_val1
    val_accurate = acc_val / val_num
    print('test_accuracy: %.3f' % val_accurate)

    """如果测试集的准确度比历史最佳还高，那么更新json文件，保存权重"""
    if val_accurate > best_acc:
        # with open(json_path, 'w') as f:
        # f.truncate()
        json_context = {}
        json_path = os.path.join("/u/hvg7cb/UVA/gan/cl/chest", "valid18_1_acc0.949_%.3f.json" % val_accurate)
        best_acc = val_accurate

        for step2, val_data in enumerate(validate_loader, start=0):

            val_images, val_labels = val_data
            outputs1 = net(val_images.to(device))  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs1, dim=1)[1]
            prob = torch.softmax(outputs1, dim=1)
            for i in range(0, len(val_images.to(device))):
                """此处实验原理是逐个（i）取得文件名，真值和预测值，以文件名为索引，【真值，预测值】为标签，写入json中"""
                val_sample_fname, _ = validate_loader.dataset.samples[step2 * batch_size + i]
                print(val_sample_fname)  # 打印文件名
                # print(val_labels.to(device)[i])
                # 此处的意思是以文件名为索引，【真值，预测值】为标签，写入json中
                # metrix_val[val_labels.to(device)[i]][predict_y[i]] += 1
                if prob[i][val_labels.to(device)[i].item()].item() < 0.60:
                    json_context[val_sample_fname] = (val_labels.to(device)[i].item(), predict_y[i].item(),
                                                      prob[i][0].item(), prob[i][1].item()
                                                      )
                    print(json_context[val_sample_fname])
            with open(json_path, 'w') as f:
                json_str = json.dumps(json_context, indent=4, ensure_ascii=False)
                f.write(json_str)
    # 仅在历史最佳时提供混淆矩阵，各类精度和总体精度
    print("\nlight: %.3f " % (acc_val0 / num_val0))
    print("\nmedium: %.3f " % (acc_val1 / num_val1))

    print('test_accuracy: %.3f' % val_accurate)
    print(metrix_val)
print('Finished Testing')
