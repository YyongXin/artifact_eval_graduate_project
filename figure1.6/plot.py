import sys
import os
import numpy as np 
import matplotlib.pyplot as plt 
import math
import random
import argparse
parser = argparse.ArgumentParser(description='pass args')
parser.add_argument('--datafile', type=str, help='input data path')
args = parser.parse_args()
f=open(args.datafile,"r")
lines=f.readlines()
memory_load_MB=0
memory_load_list=[]
address_size={}
for line in lines:
    sp=line.strip().split(' ')
    if sp[0].startswith("MALLOC"):
        memory_load_MB=memory_load_MB+int(sp[2])/1024/1024
        memory_load_list.append(memory_load_MB)
        address_size[sp[1]]=int(sp[2])/1024/1024
    elif sp[0].startswith("FREE"):
        memory_load_MB=memory_load_MB-address_size[sp[1]]
        memory_load_list.append(memory_load_MB)
    else:
        continue
plt.figure() 
x=[i for i in range(len(memory_load_list))]
memory_load_list_n=[]
for line in lines:
    sp=line.strip().split(' ')
    if sp[0].startswith("MALLOC"):
        size=int(sp[2])/1024/1024
        if size<500:
            memory_load_MB=memory_load_MB+int(sp[2])/1024/1024
            memory_load_list_n.append(memory_load_MB)
        else:
            memory_load_MB=memory_load_MB-size
            memory_load_list_n.append(memory_load_MB)
        address_size[sp[1]]=int(sp[2])/1024/1024
    elif sp[0].startswith("FREE"):
        if address_size[sp[1]]>500:
            memory_load_MB=memory_load_MB+address_size[sp[1]]
            memory_load_list_n.append(memory_load_MB)
        else:
            memory_load_MB=memory_load_MB-address_size[sp[1]]
            memory_load_list_n.append(memory_load_MB)
    else:
        continue
plt.plot(x,memory_load_list)
plt.savefig("figure1.6.jpg",dpi=720)