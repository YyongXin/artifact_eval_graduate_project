import sys
import os
import random
import docplex.mp.model as cpx
import pandas as pd
def test_cplex1():
    # 导入库
    from docplex.mp.model import Model
    # 创建模型
    model = Model()
    # 创建变量列表
    X = model.continuous_var_list([i for i in range(0, 2)], lb=0, name='X')
    # 设定目标函数
    model.minimize(2 * X[0] + 3* X[1])
    # 添加约束条件
    model.add_constraint(3 * X[0] + X[1] >= 30)
    model.add_constraint(X[0] - X[1] <= 10)
    model.add_constraint(X[1] >= 1)
    # 求解模型
    sol = model.solve()
    # 打印结果
    print(sol)
    # 打印详细结果
    print(sol.solve_details)
    opt_df = pd.DataFrame.from_dict(X)
    print(opt_df)
def test():
    print("start to test")
    n = 10
    m = 5
    set_I = range(1, n+1)
    set_J = range(1, m+1)
    c = {(i,j): random.normalvariate(0,1) for i in set_I for j in set_J}
    a = {(i,j): random.normalvariate(0,5) for i in set_I for j in set_J}
    l = {(i,j): random.randint(0,10) for i in set_I for j in set_J}
    u = {(i,j): random.randint(10,20) for i in set_I for j in set_J}
    b = {j: random.randint(0,30) for j in set_J}
    opt_model = cpx.Model(name="MIP Model")
    #定义决策变量
    #在Python字典(或panda系列)中存储决策变量是标准的，
    #其中字典键是决策变量，值是决策变量对象。
    #一个决策变量由三个主要属性定义:
    #它的类型(连续、二进制或整数)、它的下界(默认为0)和上界(默认为无穷大)。
    #对于上面的例子，我们可以将决策变量定义为:
    # if x is Continuous
    x_vars  = {(i,j): opt_model.continuous_var(lb=l[i,j], ub= u[i,j],name="x_{0}_{1}".format(i,j)) for i in set_I for j in set_J}
    # if x is Binary
    x_vars  = {(i,j): opt_model.binary_var(name="x_{0}_{1}".format(i,j)) for i in set_I for j in set_J}
    # if x is Integer
    x_vars  = {(i,j): opt_model.integer_var(lb=l[i,j], ub= u[i,j],name="x_{0}_{1}".format(i,j)) for i in set_I for j in set_J}
    #约束条件
    #在设置决策变量并将它们添加到我们的模型之后，
    #就到了设置约束的时候了。
    #任何约束都有三个部分:
    #左手边(通常是决策变量的线性组合)、
    #右手边(通常是数值)和意义(小于或等于、等于、大于或等于)。
    #要设置任何约束，我们需要设置每个部分:
    # <= constraints，小于等于
    constraints = {j : opt_model.add_constraint(ct=opt_model.sum(a[i,j] * x_vars[i,j] for i in set_I) <= b[j],\
        ctname="constraint_{0}".format(j)) for j in set_J}
    # >= constraints
    constraints = {j:opt_model.add_constraint(ct=opt_model.sum(a[i,j] * x_vars[i,j] for i in set_I) >= b[j],\
        ctname="constraint_{0}".format(j)) for j in set_J}
    # == constraints
    constraints = {j : opt_model.add_constraint(ct=opt_model.sum(a[i,j] * x_vars[i,j] for i in set_I) == b[j],\
        ctname="constraint_{0}".format(j)) for j in set_J}
    #目标函数
    #下一步是定义一个目标，它是一个线性表达式。我们可以这样定义目标:
    objective = opt_model.sum(x_vars[i,j] * c[i,j] for i in set_I for j in set_J)
    # for maximization             
    opt_model.maximize(objective)
    # for minimization
    opt_model.minimize(objective)
    #求解模型
    # solving with local cplex
    opt_model.solve()
    # solving with cplex cloud
    opt_model.solve(url="your_cplex_cloud_url", key="your_api_key")
    #获得结果
    #只需要得到结果并进行后期处理。
    #panda包是一个很好的数据处理库。
    #如果问题得到最优解，我们可以得到和处理结果如下:
    import pandas as pd
    opt_df = pd.DataFrame.from_dict(x_vars, orient="index", columns = ["variable_object"])
    opt_df.index = pd.MultiIndex.from_tuples(opt_df.index,names=["column_i", "column_j"])
    opt_df.reset_index(inplace=True)
    # CPLEX
    opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.solution_value)
    opt_df.drop(columns=["variable_object"], inplace=True)
    opt_df.to_csv("./optimization_solution.csv")
    #opt_df是一个包含每个决策变量Xij的最优值的panda dataframe。我们还可以将这些结果保存到CSV文件中
def mip_cplex_opt(block_list,start_address,first_timestamp):
    '''
    n ∈ Z: number of memory blocks.
    B = {1, . . . , n}: a set of IDs of memory blocks.
    W ∈ N: the available maximum memory size.
    wi ∈ N (i ∈ B): size of memory block i.
    yi_ ∈ N (i ∈ B): time when i is malloced.
    yi^ ∈ N (i ∈ B): time when i is freed.
    '''
    n=len(block_list)
    B=[x+1 for x in range(n)]
    max_address = 0
    for item in block_list:
        item['address_l']=int(item['address_l'],16)-int(start_address,16)
        item['address_r']=int(item['address_r'],16)-int(start_address,16)
        if item['address_r']>max_address:
            max_address = item['address_r']
        item['Malloc_time']=int(item['Malloc_time'])-int(first_timestamp)
        item['Free_time']=int(item['Free_time'])-int(first_timestamp)
    W=max_address
    w={i:int(block_list[i-1]['size']) for i in B}
    y_s={i:block_list[i-1]['Malloc_time'] for i in B}
    y_e={i:block_list[i-1]['Free_time'] for i in B}
    #define Opt model 
    opt_model = cpx.Model(name="MIP Model")
    #定义决策变量
    #u ∈ Z: the peak memory usage.
    # xi ∈ Z (i ∈ B): memory offset (or, starting address) of memory block i within the entire allocated memory.
    # zij ∈ {0, 1} ((i, j) ∈ E): 0 means that memory block i is located lower than block j (i.e., xi + wi ≤ xj )\
    # and 1 means that it is not (i.e., xj + wj ≤ xi).
    u=opt_model.integer_var(lb=0, ub=W,name="u")
    x = {i:opt_model.integer_var(lb=0,ub=W,name="x_{}".format(i)) for i in B}
    E = []
    for i in range(n):
        for j in range(i+1,n):
            interval1 = [block_list[i]['Malloc_time'],block_list[i]['Free_time']]
            interval2 = [block_list[j]['Malloc_time'],block_list[j]['Free_time']]
            if interval1[0]>=interval2[1] or interval2[0]>=interval1[1]:
                continue
            else:
                tmp = (i+1,j+1)
                E.append(tmp)
    z = {(i,j): opt_model.binary_var(name="z_{0}_{1}".format(i,j)) for (i,j) in E}
    #约束条件
#     import pdb;pdb.set_trace()
    constraints1 = {i:opt_model.add_constraint(x[i]+w[i]<=u,ctname="constraint1_{0}".format(i)) for i in B}
#     print()
    constraints2 = {(i,j):opt_model.add_constraint(x[i]+w[i]<=x[j]+z[(i,j)]*W,ctname="constraint2_{0}_{1}".format(i,j)) for (i,j) in E}
    constraints3 = {(i,j):opt_model.add_constraint(x[j]+w[j]<=x[i]+(1-z[(i,j)])*W,ctname="constraint3_{0}_{1}".format(i,j)) for (i,j) in E}
    constraints4 = {i:opt_model.add_constraint(x[i]>=0,ctname="constraint4_{0}".format(i)) for i in B}
#     objective = u
    opt_model.minimize(u)
    sol = opt_model.solve()
#     opt_df_u = pd.DataFrame.from_dict(u)
#     opt_df_x = pd.DataFrame.from_dict(x)
#     opt_df_z = pd.DataFrame.from_dict(z)
    print(sol)
    print(sol.solve_details)
    return block_list
def heuristic_opt_size_greedy(block_list,start_address,first_timestamp):
    max_address = 0
    for item in block_list:
        item['address_l']=int(item['address_l'],16)-int(start_address,16)
        item['address_r']=int(item['address_r'],16)-int(start_address,16)
        if item['address_r']>max_address:
            max_address = item['address_r']
        item['Malloc_time']=int(item['Malloc_time'])-int(first_timestamp)
        item['Free_time']=int(item['Free_time'])-int(first_timestamp)
    placed_blocks=[]
    used_blocks = []
    unused_blocks = [(0,max_address)]
    def findBestFitBlock_offset(block,placed_blocks):
        offset = 0
        overlapping_blocks = []
        for placed_block in placed_blocks:
            if block['Malloc_time']>=placed_block['Free_time'] or block['Free_time']<placed_block['Malloc_time']:
                continue
            else:
                overlapping_blocks.append(placed_block)
        used_intervals = []
        offsets = [0]
        sorted_overlapping_blocks = sorted(overlapping_blocks,key = lambda i: i['Malloc_time'])
#         import pdb;pdb.set_trace()
        #find offsets
        for i in range(len(sorted_overlapping_blocks)):
            offsets.append(sorted_overlapping_blocks[i]['address_r'])
        #filte offsets
        valid_offsets=[] 
        for offset in offsets:
            isvalid = True
            for block in sorted_overlapping_blocks:
                if offset>=block['address_l'] and offset<block['address_r']:
                    isvalid=False
                    break
            if isvalid:
                valid_offsets.append(offset)
        #build holes
        holes = []
        for offset in valid_offsets:
            upper_bound = 0
            for i in range(len(sorted_overlapping_blocks)):
                if offset>=sorted_overlapping_blocks[i]['address_r']:
                    continue
                elif offset<sorted_overlapping_blocks[i]['address_l']:
                    upper_bound = sorted_overlapping_blocks[i]['address_l']
                    break
            if upper_bound == 0:
                upper_bound = max_address
            tmp = [offset,upper_bound]
            holes.append(tmp)
        #find smallest qualified hole
#         import pdb;pdb.set_trace()
        min_hole_size=sys.maxsize
        min_hole = []
        for hole in holes:
            hole_size=hole[1]-hole[0]
            if hole_size>=int(block['size']):
                if hole_size<min_hole_size:
                    min_hole_size = hole_size
                    min_hole = hole
        return min_hole[0]
    block_list_sorted = sorted(block_list,key = lambda i: i['size'],reverse=True)
    tobe_removed=[]
    while len(block_list_sorted)!=0:
        for i in range(len(block_list_sorted)):
            offset = findBestFitBlock_offset(block_list_sorted[i],placed_blocks)
#             import pdb;pdb.set_trace()
            block_list_sorted[i]['address_l']=offset
            block_list_sorted[i]['address_r']=block_list_sorted[i]['address_l']+int(block_list_sorted[i]['size'])
            placed_blocks.append(block_list_sorted[i])
            tobe_removed.append(block_list_sorted[i])
        for block in tobe_removed:
            block_list_sorted.remove(block)
        tobe_removed=[]
    # reverse datatype
    for item in placed_blocks:
        item['address_l']=hex(item['address_l']+int(start_address,16))
        item['address_r']=hex(item['address_r']+int(start_address,16))
        item['Malloc_time']=str(int(item['Malloc_time'])+int(first_timestamp))
        item['Free_time']=str(int(item['Free_time'])+int(first_timestamp))
    return placed_blocks
def heuristic_opt_lifetime_greedy(block_list,start_address,first_timestamp):
    for item in block_list:
        item['address_l']=int(item['address_l'],16)-int(start_address,16)
        item['address_r']=int(item['address_r'],16)-int(start_address,16)
        item['Malloc_time']=int(item['Malloc_time'])-int(first_timestamp)
        item['Free_time']=int(item['Free_time'])-int(first_timestamp)
    placed_blocks=[]
    block_list_sorted = sorted(block_list,key = lambda i: i['Lifetime'])
    def canPlaced(block,offset):
        canPlaced=True
        for placed_block in placed_blocks:
            if placed_block['address_r']<=offset:
                continue
            if block['Malloc_time']>=placed_block['Free_time'] or block['Free_time']<placed_block['Malloc_time']:
                continue
            else:
                canPlaced=False
        return canPlaced
    tobe_removed=[]
    offsets=[0]
    while len(block_list_sorted)!=0:
        try:
            mem_started_level=min(offsets)
        except:
            sys.exit()
        for i in range(len(block_list_sorted)):
            if canPlaced(block_list_sorted[i],mem_started_level):
                block_list_sorted[i]['address_l']=mem_started_level
                block_list_sorted[i]['address_r']=block_list_sorted[i]['address_l']+int(block_list_sorted[i]['size'])
                placed_blocks.append(block_list_sorted[i])
                tobe_removed.append(block_list_sorted[i])
#                 if int(block_list_sorted[i]['size']) not in offsets:
                offsets.append(int(block_list_sorted[i]['address_r']))
        offsets.remove(mem_started_level)
        for block in tobe_removed:
            block_list_sorted.remove(block)
        tobe_removed=[]
    for item in placed_blocks:
        item['address_l']=hex(item['address_l']+int(start_address,16))
        item['address_r']=hex(item['address_r']+int(start_address,16))
        item['Malloc_time']=str(int(item['Malloc_time'])+int(first_timestamp))
        item['Free_time']=str(int(item['Free_time'])+int(first_timestamp))
    return placed_blocks
if __name__=="__main__":
    test_cplex1()