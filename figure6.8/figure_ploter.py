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
index = ['Origin','Opt','Reduction']
width = 5  # the width of the bars
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
bins=2
x = [32,64,128]  # the label locations
ax1.bar(x , data.loc[0].values[1:4], width, label='Origin')
x_=[xi+width for xi in x]
ax1.bar(x_, data.loc[1].values[1:4], width, label='Opt')
ax1.legend()
ax2 = fig.add_subplot(2,2,2)
bins=2
x = [32,64,128]  # the label locations
ax2.bar(x , data.loc[3].values[1:4], width, label='Origin')
x_=[xi+width for xi in x]
ax2.bar(x_, data.loc[4].values[1:4], width, label='Opt')
ax2.legend()
ax3 = fig.add_subplot(2,2,3)
bins=2
x = [32,64,128]  # the label locations
ax3.bar(x , data.loc[6].values[1:4], width, label='Origin')
x_=[xi+width for xi in x]
ax3.bar(x_, data.loc[7].values[1:4], width, label='Opt')
ax3.legend()
ax4 = fig.add_subplot(2,2,4)
bins=2
x = [32,64,128,256]  # the label locations
ax4.bar(x , data.loc[9].values[1:5], width, label='Origin')
x_=[xi+width for xi in x]
ax4.bar(x_, data.loc[10].values[1:5], width, label='Opt')
ax4.legend()
plt.savefig("figure6.8.png")
