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
width = 9  # the width of the bars
fig = plt.figure()
bins=3
# Seq2Seq
x = [32,64,128,256]  # the label locations
y = [data[32][1],data[64][1],data[128][1],data[256][1]]
plt.bar(x , y, width/bins, label='Seq2Seq_Origin')
x_=[xi+width/bins for xi in x]
y_ = [data["Unnamed: 2"][1],data["Unnamed: 4"][1],data["Unnamed: 6"][1],data["Unnamed: 8"][1]]
plt.bar(x_, y_, width/bins, label='Seq2Seq_Opt')
# Bert
x=[xi+2*width/bins for xi in x_] # the label locations
y = [data[32][2],data[64][2],data[128][2],data[256][1]]
plt.bar(x , y, width/bins, label='Bert_Origin')
x_=[xi+width/bins for xi in x]
y_ = [data["Unnamed: 2"][2],data["Unnamed: 4"][2],data["Unnamed: 6"][2],data["Unnamed: 8"][2]]
plt.bar(x_, y_, width/bins, label='Bert_Opt')
plt.legend()
plt.savefig("figure6.10.png")
