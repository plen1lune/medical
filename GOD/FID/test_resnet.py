import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import numpy as np
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from model import rresnet18
import torch.nn as nn
# 检查CUDA是否可用
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# 加载预训练的Inception模型
def get_inception_model():
    model = rresnet18()
    inchannel = model.fc.in_features

    # net.fc = nn.Linear(inchannel, 4)  # 重新确定全连接层

    model.fc = nn.Sequential(
        nn.Linear(inchannel, 512),
        nn.ReLU(),
        # nn.Dropout(0.7),
        # nn.Linear output layer
        #nn.Linear(512, 2),
        #nn.LogSoftmax(dim=1)
    )
    model_weight_path = "/u/hvg7cb/UVA/gan/path/chest_acc0.958.pth"
    model.load_state_dict(torch.load(model_weight_path), strict=False)
    model.to(device)
    model.eval()
    return model



def calculate_features(dataloader, model, dims=512):
    model.eval()
    features = np.zeros((len(dataloader.dataset), dims))

    # 使用tqdm包装enumerate(dataloader)
    for i, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating features"):
        images = images.to(device)

        with torch.no_grad():
            pred = model(images)

        # 注释掉的部分已经不再需要
        # if pred.size(2) != 1 or pred.size(3) != 1:
        #     pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))

        features[i * len(images):i * len(images) + len(images)] = pred.cpu().numpy().reshape(len(images), -1)

    return features


# 计算FID分数
def calculate_fid(real_features, generated_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# 设置图像转换
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.Resize((512, 512)),
                                # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.905, 0.759, 0.870], [0.115, 0.205, 0.120])])


# 加载数据集
def load_dataset(directory, batch_size=1):
    dataset = ImageFolder(root=directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader


# 计算FID
def main():
    model = get_inception_model()

    # 加载数据集，你需要替换这里的路径
    real_dataloader = load_dataset('/u/hvg7cb/UVA/fid/chest_93/1_1')
    generated_dataloader = load_dataset('/u/hvg7cb/UVA/fid/chest_93/0_1')

    real_features = calculate_features(real_dataloader, model)
    generated_features = calculate_features(generated_dataloader, model)

    fid_value = calculate_fid(real_features, generated_features)
    print(f'FID: {fid_value}')


if __name__ == '__main__':
    main()
