import sys
import os
#导入需要用到的模块
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
parser = argparse.ArgumentParser(description='pass args')
parser.add_argument('--datafile', type=str, help='input data path')
args = parser.parse_args()
data = pd.read_excel(args.datafile)
index = np.arange(6)
index = ['AlexNet','InceptionV4','ResNet50','Resnet101','ResNet152','VGG16','VGG19']
width = 0.5  # the width of the bars
x = np.arange(len(index))  # the label locations
fig, ax = plt.subplots(figsize=(5,4))
bins=8
rects1 = ax.bar(x + 0*width/bins, data.loc[0].values[1:], width/bins, label='Conv')
rects2 = ax.bar(x + 1*width/bins, data.loc[1].values[1:], width/bins, label='FC')
rects3 = ax.bar(x + 2*width/bins, data.loc[2].values[1:], width/bins, label='DROPOUT')
rects4 = ax.bar(x + 3*width/bins, data.loc[3].values[1:], width/bins, label='SOFTMAX')
rects5 = ax.bar(x + 4*width/bins, data.loc[4].values[1:], width/bins, label='POOL')
rects6 = ax.bar(x + 5*width/bins, data.loc[5].values[1:], width/bins, label='ACT')
rects7 = ax.bar(x + 6*width/bins, data.loc[6].values[1:], width/bins, label='BN')
rects8 = ax.bar(x + 7*width/bins, data.loc[7].values[1:], width/bins, label='LRN')
ax.legend()
plt.savefig("figure5.4.png")
