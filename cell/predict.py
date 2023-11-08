import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import glob
import torch.optim as optim
from vit_forvisual import ViT_v2
from model import resnet18, resnet34, resnet101, resnet50
import torchvision.models.resnet  # ctrl+鼠标左键点击即可下载权重
import numpy as np
import seaborn as sns
# from einops.layers.torch import Rearrange
from einops import rearrange
from scipy.interpolate import griddata

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # 和官网初始化方法保持一致
    "test_HE": transforms.Compose([transforms.Resize((512, 512)),
                                 # transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.905, 0.759, 0.870], [0.115, 0.205, 0.120])]),
    "test800": transforms.Compose([transforms.Resize((512, 512)),
                                # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.905, 0.759, 0.870], [0.115, 0.205, 0.120])])
}

# data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
data_root = "/home/xjl/llc/acc80"
image_path = data_root + "/data/"  # data set path

train_dataset = datasets.ImageFolder(root=image_path + "test_HE",
                                     transform=data_transform["test_HE"])
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 1
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "test_HE",
                                        transform=data_transform["test_HE"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

resnet = resnet18(include_top=False)  # 一开始不能设置全连接层的输出种类为自己想要的，必须先将模型参数载入，再修改全连接层
net = ViT_v2(feature_size=16, patch_size=4, num_classes=3, dim=256, depth=10, heads=4,
             mlp_dim=512, pool='cls', dim_head=64)
# 官方提供载入预训练模型的方法
model_weight_path = "/home/xjl/llc/acc80/path/b/resNet18_new1_color_crop_acc0.929.pth"  # "/mnt/nas/llc/acc80/path3/b/resNet18_1_acc0.949.pth"  # 权重路径
resnet.load_state_dict(torch.load(model_weight_path), strict=False)  # 载入模型权重, strict=False
vit_path = "/home/xjl/llc/acc80/result/path_moredata_ontest/ViT_adam_resnet_pretrainedv3_bs16_depth10_clstoken_acc0.960.pth"  # "/mnt/nas/llc/acc80/result/path/VIT_resnet_pretrained_bs16_head1_depth10_clstoken_acc0.937.pth"
net.load_state_dict(torch.load(vit_path))
resnet.to(device)
net.to(device)

# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.0001)
best_acc = 0.0

# running_loss = 0.0

# validate
resnet.eval()  # 控制BN层状态
net.eval
acc_val = 0.0  # accumulate accurate number / epoch
acc_val0 = 0.0
acc_val1 = 0.0
acc_val2 = 0.0

num_val0 = 0
num_val1 = 0
num_val2 = 0

metrix_val = [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              ]


def visual_att(mapp_list, filename):
    cgap = np.zeros([17, 2])
    depths = []
    for mapp in mapp_list:
        mapp = mapp.squeeze()
        mapp = mapp.cpu().numpy()
        # print (mapp.shape)
        out = np.zeros([17, 1])
        for h in range(mapp.shape[0]):
            m = mapp[h, :, :]
            out = np.concatenate((out, m), axis=1)
            if h != mapp.shape[0] - 1:
                out = np.concatenate((out, cgap), axis=1)
        cc = out.shape[1]
        depths.append(out)
    f, axes = plt.subplots(figsize=(35, 75), nrows=10)
    for i, res in enumerate(depths):
        # print (res.shape)
        res = res[:, 1:]
        fig = sns.heatmap(res, ax=axes[i], cmap='Blues')  # , cmap='RdBu_r',square=True,center=0
        axes[i].set_xticklabels([])
    matrix_fig = fig.get_figure()
    matrix_fig.savefig(filename)


def visual_att_patch(mapp_list, sample_fname, dir_name):
    # img = cv2.resize(map, (256,256), interpolation = cv2.INTER_CUBIC)
    for k, mapp in enumerate(mapp_list):
        mapp = mapp[0, :, :, 1:]
        # mapp = rearrange(mapp, 'h n (p1 p2) -> h n p1 p2', p1 = 4, p2 = 4)
        mapp = mapp.cpu().numpy()
        print(mapp.shape)  # 4*17*4*4
        for h in range(mapp.shape[0]):
            m = mapp[h, 0, :]  # 4*4
            x = np.arange(0, 4, 1.)
            x = x.repeat(4, 0)
            y = np.arange(0, 4, 1.)
            y = np.tile(y, 4)
            point = np.stack([x, y], axis=-1)
            value = m
            xx, yy = np.mgrid[0:3:4j, 0:3:4j]
            pic_resize_griddata = griddata(point, value, (xx, yy), method='cubic', fill_value=0, rescale=True)
            fig = plt.figure(figsize=(4, 4), dpi=100)
            plt.plot()
            plt.imshow(pic_resize_griddata, interpolation='nearest', cmap='Blues')  # , origin='upper'
            filename = dir_name + '/' + sample_fname.split('/')[-2] + '-' + sample_fname.split('/')[-1].split('.')[
                0] + '/layer{}_head{}'.format(k, h) + '.png'
            dirr = os.path.dirname(filename)
            if not os.path.exists(dirr):
                os.makedirs(dirr)
            plt.savefig(filename)


def visual_att_patch_ave(mapp_list, sample_fname, dir_name):
    # img = cv2.resize(map, (256,256), interpolation = cv2.INTER_CUBIC)
    for k, mapp in enumerate(mapp_list):
        if k == 0:
            m = mapp[0, :, 0, 1:]  # 4*16
        else:
            m = torch.cat((m, mapp[0, :, 0, 1:]), 0)
    print(m.shape)
    m = torch.mean(m, 0)
    # mapp = rearrange(mapp, 'h n (p1 p2) -> h n p1 p2', p1 = 4, p2 = 4)
    m = m.cpu().numpy()
    print(m.shape)  # 16

    x = np.arange(0, 4, 1.)
    x = x.repeat(4, 0)
    y = np.arange(0, 4, 1.)
    y = np.tile(y, 4)
    point = np.stack([x, y], axis=-1)
    value = m
    xx, yy = np.mgrid[0:3:4j, 0:3:4j]
    pic_resize_griddata = griddata(point, value, (xx, yy), method='cubic', fill_value=0, rescale=True)
    fig = plt.figure(figsize=(4, 4), dpi=100)
    plt.plot()
    plt.imshow(pic_resize_griddata, interpolation='nearest', cmap='Blues')  # , origin='upper'
    filename = dir_name + '/' + sample_fname.split('/')[-2] + '-' + sample_fname.split('/')[-1].split('.')[
        0] + '/ave_att' + '.png'
    dirr = os.path.dirname(filename)
    if not os.path.exists(dirr):
        os.makedirs(dirr)
    plt.savefig(filename)


with torch.no_grad():
    for step2, val_data in enumerate(validate_loader, start=0):
        # print(step2)
        val_images, val_labels = val_data
        _, feas1 = resnet(val_images.to(device))  # eval model only have last output layer
        outputs1, _ = net(feas1)
        # loss = loss_function(outputs, test_labels)
        predict_y = torch.max(outputs1, dim=1)[1]

        for i in range(0, len(val_images.to(device))):
            val_sample_fname, _ = validate_loader.dataset.samples[step2 * batch_size + i]
            # print(val_sample_fname)
            # print(val_labels.to(device)[i])
            metrix_val[val_labels.to(device)[i]][predict_y[i]] += 1
        acc_val0 = metrix_val[0][0]
        acc_val1 = metrix_val[1][1]
        acc_val2 = metrix_val[2][2]

        num_val0 = metrix_val[0][0] + metrix_val[0][1] + metrix_val[0][2]
        num_val1 = metrix_val[1][0] + metrix_val[1][1] + metrix_val[1][2]
        num_val2 = metrix_val[2][0] + metrix_val[2][1] + metrix_val[2][2]

    acc_val += acc_val0 + acc_val1 + acc_val2
    val_accurate = acc_val / val_num
    print('test_accuracy: %.3f' % val_accurate)
    pic_list = ["/mnt/nas/llc/acc80/image3_b_add3/test/light/4-17_020_u.jpeg",
                "/mnt/nas/llc/acc80/image3_b_add3/test/medium/4-13_010.jpeg",
                "/mnt/nas/llc/acc80/image3_b_add3/test/medium/4-13_011_d.jpeg",
                "/mnt/nas/llc/acc80/image3_b_add3/test/light/8_57_010.jpeg",
                "/mnt/nas/llc/acc80/image3_b_add3/test/medium/4-17_022_r.jpeg",
                "/mnt/nas/llc/acc80/image3_b_add3/test/medium/4-33_029.jpeg",
                "/mnt/nas/llc/acc80/image3_b_add3/test/serious/3_24_049.jpeg",
                "/mnt/nas/llc/acc80/image3_b_add3/test/serious/4-13_073.jpeg"]

    """如果测试集的准确度比历史最佳还高，那么更新json文件，保存权重"""
    if val_accurate > best_acc:
        # with open(json_path, 'w') as f:
        # f.truncate()
        json_context = {}
        json_path = os.path.join("/home/xjl/llc/acc80/data", "test_HE2_18_%.3f.json" % val_accurate)
        best_acc = val_accurate

        for step2, val_data in enumerate(validate_loader, start=0):

            val_images, val_labels = val_data
            _, feas1 = resnet(val_images.to(device))  # eval model only have last output layer
            outputs1, attn_list = net(feas1)  # b,h,n,n   list of 1*4*17*17
            predict_y = torch.max(outputs1, dim=1)[1]
            prob = torch.softmax(outputs1, dim=1)
            for i in range(0, len(val_images.to(device))):
                """此处实验原理是逐个（i）取得文件名，真值和预测值，以文件名为索引，【真值，预测值】为标签，写入json中"""
                val_sample_fname, _ = validate_loader.dataset.samples[step2 * batch_size + i]
                print(val_sample_fname)  # 打印文件名
                # print(val_labels.to(device)[i])
                # 此处的意思是以文件名为索引，【真值，预测值】为标签，写入json中
                # metrix_val[val_labels.to(device)[i]][predict_y[i]] += 1
                if prob[i][val_labels.to(device)[i].item()].item() < 2:
                    json_context[val_sample_fname] = (val_labels.to(device)[i].item(), predict_y[i].item(),
                                                      prob[i][0].item(), prob[i][1].item(), prob[i][2].item()
                                                      )
                    print(json_context[val_sample_fname])
            with open(json_path, 'w') as f:
                json_str = json.dumps(json_context, indent=4, ensure_ascii=False)
                f.write(json_str)
    # 仅在历史最佳时提供混淆矩阵，各类精度和总体精度
    #print("\nlight: %.3f " % (acc_val0 / num_val0))
    #print("\nmedium: %.3f " % (acc_val1 / num_val1))
    #print("\nserious: %.3f " % (acc_val2 / num_val2))

    #print('test_accuracy: %.3f' % val_accurate)
    #print(metrix_val)
print('Finished Testing')
