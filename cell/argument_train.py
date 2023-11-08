from PIL import Image,ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import math


def resize_img(img, size):  # 设置图片大小
    """
    强制缩放图片
    force to resize the img
    """
    img = img.resize((size[1], size[2]), Image.BILINEAR)
    return img


def random_crop(img, size, scale=[0.08, 1.0], ratio=[3. / 4, 4. / 3]):  # size CHW
    """
    随机剪裁
    """
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    bound = min((float(img.size[0]) / img.size[1]) / (aspect_ratio ** 2),
                (float(img.size[1]) / img.size[0]) * (aspect_ratio ** 2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)
    target_area = img.size[0] * img.size[1] * np.random.uniform(scale_min, scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * aspect_ratio)
    h = int(target_size / aspect_ratio)
    i = np.random.randint(0, img.size[0] - w + 1)
    j = np.random.randint(0, img.size[1] - h + 1)
    img = img.crop((i, j, i + w, j + h))
    img = img.resize((size[1], size[2]), Image.BILINEAR)
    return img


def random_crop_scale(img, size, scale=[0.8, 1.0], ratio=[3. / 4, 4. / 3]):  # size CHW
    """
    image randomly croped
    scale rate is the mean element
    First scale rate be generated, then based on the rate, ratio can be generated with limit
    the range of ratio should be large enough, otherwise in order to successfully crop, the ratio will be ignored
    valuable passed from Config.py
    size information
    """
    scale[1] = min(scale[1], 1.)
    scale_rate = np.random.uniform(*scale)
    target_area = img.size[0] * img.size[1] * scale_rate
    target_size = math.sqrt(target_area)
    bound_max = math.sqrt(float(img.size[0]) / img.size[1] / scale_rate)
    bound_min = math.sqrt(float(img.size[0]) / img.size[1] * scale_rate)
    aspect_ratio_max = min(ratio[1], bound_min)
    aspect_ratio_min = max(ratio[0], bound_max)
    if aspect_ratio_max < aspect_ratio_min:
        aspect_ratio = np.random.uniform(bound_min, bound_max)
    else:
        aspect_ratio = np.random.uniform(aspect_ratio_min, aspect_ratio_max)

    w = int(aspect_ratio * target_size)
    h = int(target_size / aspect_ratio)

    i = np.random.randint(0, img.size[0] - w + 1)
    j = np.random.randint(0, img.size[1] - h + 1)
    img = img.crop((i, j, i + w, j + h))
    img = img.resize((size[1], size[2]), Image.BILINEAR)
    return img


def rotate_img(img, angle=[-14, 15]):
    """
    图像增强，增加随机旋转角度
    """
    angle = np.random.randint(*angle)
    img = img.rotate(angle)
    return img


def random_brightness(img, prob, delta):
    """
    图像增强，亮度调整
    :param img:
    :return:
    """
    brightness_prob = np.random.uniform(0, 1)
    if brightness_prob < prob:
        brightness_delta = np.random.uniform(-delta, +delta) + 1
        img = ImageEnhance.Brightness(img).enhance(brightness_delta)
    return img


def random_contrast(img, prob, delta):
    """
    图像增强，对比度调整
    """
    contrast_prob = np.random.uniform(0, 1)
    if contrast_prob < prob:
        contrast_delta = np.random.uniform(-delta, +delta) + 1
        img = ImageEnhance.Contrast(img).enhance(contrast_delta)
    return img


def random_saturation(img, prob, delta):
    """
    图像增强，饱和度调整
    """
    saturation_prob = np.random.uniform(0, 1)
    if saturation_prob < prob:
        saturation_delta = np.random.uniform(-delta, +delta) + 1
        img = ImageEnhance.Color(img).enhance(saturation_delta)
    return img


def random_hue(img, prob, delta):
    """
    图像增强，色度调整
    """
    hue_prob = np.random.uniform(0, 1)
    if hue_prob < prob:
        hue_delta = np.random.uniform(-delta, +delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + hue_delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


def distort_color(img, params):
    """
    概率的图像增强
    """
    prob = np.random.uniform(0, 1)
    if prob < 0.35:
        img = random_brightness(img, params['brightness_prob'], params['brightness_delta'])
        img = random_contrast(img, params['contrast_prob'], params['contrast_delta'])
        img = random_saturation(img, params['saturation_prob'], params['saturation_delta'])
        img = random_hue(img, params['hue_prob'], params['hue_delta'])
    elif prob < 0.7:
        img = random_brightness(img, params['brightness_prob'], params['brightness_delta'])
        img = random_saturation(img, params['saturation_prob'], params['saturation_delta'])
        img = random_hue(img, params['hue_prob'], params['hue_delta'])
        img = random_contrast(img, params['contrast_prob'], params['contrast_delta'])
    return img

