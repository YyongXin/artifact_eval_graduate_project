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
label = ['Origin','Opt']
fig = plt.figure( dpi=1080)
x = np.arange(1,188)  # the label locations
plt.plot(x,data[label[0]].values[:187],label=label[0])
plt.plot(x,data[label[1]].values[:187],label=label[1])
plt.legend()
plt.savefig("figure6.7.png")