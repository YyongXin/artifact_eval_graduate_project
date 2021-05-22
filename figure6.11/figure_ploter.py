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
width = 9  # the width of the bars
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
bins=3
# InceptionV4
x_i = [32,64,128]  # the label locations
y = [data[32][0],data[64][0],data[128][0]]
ax1.bar(x_i , y, width/bins, label='InceptionV4_Origin')
x_=[xi+width/bins for xi in x_i]
y_ = [data[32][1],data[64][1],data[128][1]]
ax1.bar(x_, y_, width/bins, label='InceptionV4_Opt')
y1 = [data[32][2],data[64][2],data[128][2]]
ax1.plot(x_i,y1,label='reduction')
ax1.legend()
# Resnet
ax2 = fig.add_subplot(1,2,2)
x_i = [32,64,128]  # the label locations
y = [data[32][4],data[64][4],data[128][4]]
ax2.bar(x_i , y, width/bins, label='InceptionV4_Origin')
x_=[xi+width/bins for xi in x_i]
y_ = [data[32][5],data[64][5],data[128][5]]
ax2.bar(x_, y_, width/bins, label='InceptionV4_Opt')
y1 = [data[32][6],data[64][6],data[128][6]]
ax2.plot(x_i,y1,label='reduction')
ax2.legend()
plt.savefig("figure6.10.png")
