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
label = ['SuperNeurons','TF','Pytorch','MXNet','METS']
fig = plt.figure( dpi=1080)
fig.add_subplot(131)
x = np.arange(16,176,16)  # the label locations
plt.plot(x,data[label[0]].values[:10],label=label[0])
plt.plot(x,data[label[1]].values[:10],label=label[1])
plt.plot(x,data[label[2]].values[:10],label=label[2])
plt.plot(x,data[label[3]].values[:10],label=label[3])
plt.legend()
fig.add_subplot(132)
x = np.arange(16,160,16)  # the label locations
plt.plot(x,data[label[0]].values[11:20],label=label[0])
plt.plot(x,data[label[1]].values[11:20],label=label[1])
plt.plot(x,data[label[2]].values[11:20],label=label[2])
plt.plot(x,data[label[3]].values[11:20],label=label[3])
plt.legend()
fig.add_subplot(133)
x = np.arange(16,128,16)  # the label locations
plt.plot(x,data[label[0]].values[21:28],label=label[0])
plt.plot(x,data[label[1]].values[21:28],label=label[1])
plt.plot(x,data[label[2]].values[21:28],label=label[2])
plt.plot(x,data[label[3]].values[21:28],label=label[3])
plt.legend()
plt.savefig("figure6.2.png")