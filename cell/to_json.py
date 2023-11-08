import json
import os
import glob
import shutil

import numpy as np

with open("/home/llc/acc80/clasify/raw/test34_0.775.json", 'r', encoding='utf-8') as load_f:
    load_dict = json.load(load_f)


def copy_img(imgName, n):
    if n == 0:
        shutil.copy(imgName, "G:/BaiduNetdiskDownload/HEpatch/drop")
    elif n == 1:
        shutil.copy(imgName, "G:/BaiduNetdiskDownload/HEpatch/light")
    elif n == 2:
        shutil.copy(imgName, "G:/BaiduNetdiskDownload/HEpatch/medium")
    elif n == 3:
        shutil.copy(imgName, "G:/BaiduNetdiskDownload/HEpatch/serious")


data = load_dict.items()
list = list(data)
arr = np.array(list)
# print(type(arr))
#print(arr[55][0])
#print(arr[55][1])
#print(arr[55][1][0])
#print(arr[55][1][1])
print(arr.shape[0])
# print(arr[1,0])
# print(arr[1,1])
for i in range(0, arr.shape[0]):
    truth = arr[i][1][0]
    predict = arr[i][1][1]
    addr = arr[i][0]
    shutil.copy(addr, '/home/llc/acc80/analysis_portion/%d/to%d' % (truth, predict))
"""a = [0, 1, 2, 3]
print(arr.shape[0])

sourrce_dir = "data"
img = os.listdir(sourrce_dir)
for fileNum in img:
    #if not os.path.isdir(fileNum):
    for i in range(0,735):
        addr = arr[i, 0]
        name = arr[i, 1]
        #print(arr[i, 0])
        #print(arr[i, 1])
        imgName = os.path.join("data/",fileNum)
        filename = "G:/BaiduNetdiskDownload/HEpatch/补充HEpatch/"+imgName
        #print(filename)
        #print(imgName)
        #print(addr)
        if addr == imgName:
            #print('equal')
            #print(name)
            if name == '0':
                shutil.copy(filename, "G:/BaiduNetdiskDownload/HEpatch/drop")
            elif name == '1':
                shutil.copy(filename, "G:/BaiduNetdiskDownload/HEpatch/light")
            elif name == '2':
                shutil.copy(filename, "G:/BaiduNetdiskDownload/HEpatch/medium")
            elif name == '3':
                shutil.copy(filename, "G:/BaiduNetdiskDownload/HEpatch/serious")
            #copy_img("G:/BaiduNetdiskDownload/HEpatch/补充HEpatch"+imgName,name)
        continue"""

"""for i in range(0, 735):
    addr = arr[i, 0]
    #print(addr)
    if"""
