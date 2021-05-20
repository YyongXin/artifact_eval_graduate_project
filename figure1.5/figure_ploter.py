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
figuredata = pd.read_excel(args.datafile)
print(figuredata)
print(figuredata.loc[0].values[1:])
index = np.arange(6)
index = ['AlexNet','VGG16','ResNet50','ResNet101','ResNet152','InceptionV4']
Colors = [(142/256,87/256,141/256,1/4),(142/256,87/256,141/256,2/4),(142/256,87/256,141/256,3/4),(142/256,87/256,141/256,4/4)]
# pd.to_numeric(figuredata.loc[0].values[1:], errors='raise', downcast=None)
plt.bar(
        index,figuredata.loc[0].values[1:],
        color=Colors[0]
        )
plt.bar(
        index,figuredata.loc[1].values[1:],
        bottom=figuredata.loc[0].values[1:], #通过bottom来设置这个柱子距离底部的高度
        color=Colors[1]
        )
plt.bar(
        index,figuredata.loc[2].values[1:],
        bottom=figuredata.loc[0].values[1:]+figuredata.loc[1].values[1:], #通过bottom来设置这个柱子距离底部的高度
        color=Colors[2]
        )
plt.bar(
        index,figuredata.loc[3].values[1:],
        bottom=figuredata.loc[0].values[1:]+figuredata.loc[1].values[1:]+figuredata.loc[2].values[1:], #通过bottom来设置这个柱子距离底部的高度
        color=Colors[3]
        )
plt.savefig("figure1.5.png")
