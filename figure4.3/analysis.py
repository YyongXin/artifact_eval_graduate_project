import sys
import os
import binascii
import matplotlib.pyplot as plt
import time
import statsmodels.api as sm
import numpy as np
def plot_gatt(block_list,start_address,first_timestamp,figurename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    block_list_sorted = sorted(block_list,key = lambda i: i['Malloc_time'])
    for i in range(10):
        for block in block_list_sorted:
            address_l = block['address_l']
            address_r = block['address_r']
            Malloc_time = block['Malloc_time']
            Free_time = block['Free_time']
            size = block['size']
            address = [(int(address_l,16)-int(start_address,16))/1024.0/1024.0,(int(address_r,16)-int(start_address,16))/1024.0/1024.0]
            address = [x/10240.0 for x in address]
            time = [(int(Malloc_time)-first_timestamp),(int(Free_time)-first_timestamp)]
            time = [x/40000000000.0 for x in time]
    #         if time[0]<0.8 or address[0]>0.8:
    #             continue
    #         else:
            rect_indx = [time[0],address[0],(time[1]-time[0]),(address[1]-address[0])]
            rect_indx[0] = rect_indx[0] - 0.8 
    #         print(rect_indx)
            rect = plt.Rectangle((rect_indx[0]+0.1*i-0.32,rect_indx[1]),rect_indx[2],rect_indx[3])
            ax.add_patch(rect)
#     fig = plt.gcf()
    fig.savefig("{}.png".format('figure1.7'),dpi=720)
def extract_block_dict():
    lines = []
    block_dict ={}
    idx=-1
    for line in sys.stdin:
        sp=line.strip().split(' ')
        op=sp[0][:-1]
        if op not in ["MALLOC","FREE"]:
            continue
        blockddr=sp[1]
        if op=="MALLOC":
            size=sp[2]
            time_stamp=sp[3]
        else:
            time_stamp=sp[2]
        if blockddr not in block_dict:
            tmp={}
            idx=idx+1
            tmp["id"]=idx
            tmp["size"]=size
            tmp["interval"]=[0,0]
            tmp["interval"][0]=time_stamp
            block_dict[blockddr]=tmp
        else:
            block_dict[blockddr]["interval"][1]=time_stamp
            if op=="MALLOC":
                print(line)
    # print(block_dict)
def find_one_iteration(work_dir):
    f=open("{}/mem-info.log".format(work_dir),'r')
    ops=["MALLOC","FREE","READ","WRITE"]
    lines=f.readlines()
    items={}
    for line in lines:
        sp=line.strip().split(' ')
        if len(sp)>4:
            continue
        op=sp[0][:-1]
        if op not in ops:
            continue
        blockddr=sp[1]
        if op!="FREE":
            size=sp[2]
            try:
                time_stamp=sp[3]
            except:
                print(sp)
                sys.exit()
        else:
            time_stamp=sp[2]
            size = None
        if blockddr not in items:
            # 进这个分支的都是Malloc
            tmp={}
            tmp['size']=[]
            tmp['size'].append(size)
            for op_t in ops:
                tmp[op_t]=[]
            tmp[op].append(time_stamp)
            items[blockddr]=tmp
        else:
            items[blockddr][op].append(time_stamp)
            if op=="MALLOC":
                items[blockddr]['size'].append(size)
#             if items[blockddr]['size']!=size and size:
#                 print("size diff!!!!!!!!!!!")
#                 print(blockddr)
#                 print(op)
    return items
def gen_block_type_dict(work_dir):
    f=open("{}/block-type-info.log".format(work_dir),'r')
    lines=f.readlines()
    block_type_dict={}
    for line in lines:
        if line=="%===----- BlkTypeMap -----===%":
            continue
        sp=line.strip().split(' ')
        blockddr=sp[0]
        block_type=sp[1]
        block_type_dict[blockddr]=block_type
    return block_type_dict
def find_last_timestamp(memory_blocks):
    latest_timestamp=0
    for block_addrs in memory_blocks.keys():
        for time in memory_blocks[block_addrs]['MALLOC']:
            if int(time)>latest_timestamp:
                latest_timestamp=int(time)
        for time in memory_blocks[block_addrs]['FREE']:
            if int(time)>latest_timestamp:
                latest_timestamp=int(time)
    return latest_timestamp
def find_first_timestamp(memory_blocks):
    first_timestamp=sys.maxsize
    if isinstance(memory_blocks,dict):
        for block_addrs in memory_blocks.keys():
            for time in memory_blocks[block_addrs]['MALLOC']:
                if int(time)<first_timestamp:
                    first_timestamp=int(time)
    if isinstance(memory_blocks,list):
        for block in memory_blocks:
            time=block['Malloc_time']
            if int(time)<first_timestamp:
                first_timestamp=int(time)
    return first_timestamp
def padding_Free_timestamp(memory_blocks,latest_timestamp):
    #给尚未free的block在latest_timestamp的基础上加上一个常数时间
    #过滤掉没有Malloc 也没有free的block样本
    padding_timestamp=latest_timestamp+50000
    padded_memory_blocks={}
    for block_addrs in memory_blocks.keys():
        if len(memory_blocks[block_addrs]['MALLOC'])==0:
            continue
        if len(memory_blocks[block_addrs]['MALLOC'])>len(memory_blocks[block_addrs]['FREE']):
            memory_blocks[block_addrs]['FREE'].append(str(padding_timestamp))
        padded_memory_blocks[block_addrs]=memory_blocks[block_addrs]
    return padded_memory_blocks
def gen_2D_block_list(memory_blocks):
    '''
    block={
        address_l;
        address_r;
        size;
        Malloc_time;
        Free_time;
        Write_time;
        Read_time;
    }
    '''
    block_list=[]
    for block_addrs in memory_blocks.keys():
        size = memory_blocks[block_addrs]['size']
#         print(len(memory_blocks[block_addrs]['MALLOC']),len(memory_blocks[block_addrs]['WRITE']),len(memory_blocks[block_addrs]['READ']))
#         if len(memory_blocks[block_addrs]['MALLOC'])==10:
        Malloc_time=memory_blocks[block_addrs]['MALLOC'][-1]
        Free_time=memory_blocks[block_addrs]['FREE'][-1]
        address_l=block_addrs
        temp_address_r = int(block_addrs,16)+int(size[-1])
        address_r=hex(temp_address_r)
        item = {}
        item['size']=size[-1]
        item['address_l']=address_l
        item['address_r']=address_r
        item['Malloc_time']=Malloc_time
        item['Free_time']=Free_time
        item['Lifetime']=int(Free_time)-int(Malloc_time)
        Write_time = []
        for write_time in memory_blocks[block_addrs]['WRITE']:
            if int(write_time)>int(Malloc_time) and int(write_time)<int(Free_time):
                Write_time.append(write_time)
        Read_time = []
        for read_time in memory_blocks[block_addrs]['READ']: 
            if int(read_time)>int(Malloc_time) and int(read_time)<int(Free_time):
                Read_time.append(read_time)
        item['Write_time']=Write_time
        item['Read_time']=Read_time
        block_list.append(item)
#         else:
#             for i in range(len(memory_blocks[block_addrs]['MALLOC'])):
#                 Malloc_time=memory_blocks[block_addrs]['MALLOC'][i]
#                 Free_time=memory_blocks[block_addrs]['FREE'][i]
#                 address_l=block_addrs
#                 temp_address_r = int(block_addrs,16)+int(size[i])
#                 address_r=hex(temp_address_r)
#                 item = {}
#                 item['size']=size[i]
#                 item['address_l']=address_l
#                 item['address_r']=address_r
#                 item['Malloc_time']=Malloc_time
#                 item['Free_time']=Free_time
#                 block_list.append(item)
    return block_list
def get_start_address_of_mem_pool(memory_blocks):
    start_address = sys.maxsize
    if isinstance(memory_blocks,dict):
        for block_addrs in memory_blocks.keys():
            block_addrs_int = int(block_addrs,16)
            if block_addrs_int<start_address:
                start_address=block_addrs_int
    if isinstance(memory_blocks,list):
        for block in memory_blocks.keys():
            block_addrs = block['address_l']
            if block_addrs<start_address:
                start_address=block_addrs
    return hex(start_address)
def cal_access_interval(work_dir):
    f=open("{}/mem-info.log".format(work_dir),'r')
    ops=["MALLOC","FREE","READ","WRITE"]
    lines=f.readlines()
    items={}
    access_interval=[]
    for line in lines:
        sp=line.strip().split(' ')
        if len(sp)>4:
            continue
        op=sp[0][:-1]
        if op not in ops:
            continue
        blockddr=sp[1]
        if op!="FREE":
            size=sp[2]
            try:
                time_stamp=sp[3]
            except:
                print(sp)
                sys.exit()
        else:
            time_stamp=sp[2]
            size = None
        if blockddr not in items:
            # 进这个分支的都是Malloc
            tmp={}
            tmp['size']=[]
            tmp['size'].append(size)
            tmp['lastaccess']=time_stamp
            items[blockddr]=tmp
        else:
            interval = int(time_stamp) - int(items[blockddr]['lastaccess'])
            items[blockddr]['lastaccess'] = time_stamp
            access_interval.append(interval)
    return access_interval
def plot_CDF_figure(access_interval):
    ecdf = sm.distributions.ECDF(access_interval)
    x = np.linspace(min(access_interval), max(access_interval))
    y = ecdf(x)
    plt.plot(x, y)
    plt.savefig("{}.png".format('figure4.3'))
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--workdir', type=str, help='input data dir')
    args = parser.parse_args()
    work_dir = args.workdir
    access_interval = cal_access_interval(work_dir)
    plot_CDF_figure(access_interval)