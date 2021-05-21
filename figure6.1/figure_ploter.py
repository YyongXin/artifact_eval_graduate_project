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
print(data)
index = ['origin','autoswap','autoswap']
width = 0.5  # the width of the bars
x = np.arange(len(index))  # the label locations
fig, ax = plt.subplots(figsize=(5,4))
bins=4
rects1 = ax.bar(x + 0*width/bins, data.loc[0].values[1:], width/bins, label='ResNet50')
rects2 = ax.bar(x + 1*width/bins, data.loc[1].values[1:], width/bins, label='Vgg16')
rects3 = ax.bar(x + 2*width/bins, data.loc[2].values[1:], width/bins, label='Inceptionv4')
rects4 = ax.bar(x + 3*width/bins, data.loc[3].values[1:], width/bins, label='Seq2Seq')
ax.legend()
plt.savefig("figure6.1.png")
