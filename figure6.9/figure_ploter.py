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
width = 3  # the width of the bars
fig = plt.figure()
bins=3
# VGG
x = [32,64,128]  # the label locations
y = [data[32][1],data[64][1],data[128][1]]
plt.bar(x , y, width/bins, label='VGG16_Origin')
x_=[xi+width/bins for xi in x]
y_ = [data["Unnamed: 2"][1],data["Unnamed: 4"][1],data["Unnamed: 6"][1]]
plt.bar(x_, y_, width/bins, label='VGG16_Opt')
#ResNet50
x=[xi+width/bins for xi in x_] # the label locations
y = [data[32][2],data[64][2],data[128][2]]
plt.bar(x , y, width/bins, label='ResNet50_Origin')
x_=[xi+width/bins for xi in x]
y_ = [data["Unnamed: 2"][2],data["Unnamed: 4"][2],data["Unnamed: 6"][2]]
plt.bar(x_, y_, width/bins, label='ResNet50_Opt')
#InceptionV4
x=[xi+width/bins for xi in x_] # the label locations
y = [data[32][3],data[64][3],data[128][3]]
plt.bar(x , y, width/bins, label='InceptionV4_Origin')
x_=[xi+width/bins for xi in x]
y_ = [data["Unnamed: 2"][3],data["Unnamed: 4"][3],data["Unnamed: 6"][3]]
plt.bar(x_, y_, width/bins, label='InceptionV4_Opt')
plt.legend()
plt.savefig("figure6.9.png")
