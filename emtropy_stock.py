import tushare as ts
import pandas as pd
import numpy as np
import math
from numpy import array

"""
code,代码name,名称industry,所属行业 area,地区 pe,市盈率 outstanding,流通股本(亿) totals,总股本(亿)
totalAssets,总资产(万) liquidAssets,流动资产 fixedAssets,固定资产 reserved,公积金 reservedPerShare,每股公积金
esp,每股收益 bvps,每股净资 pb,市净率 timeToMarket,上市日期
undp,未分利润 perundp, 每股未分配 rev,收入同比(%) profit,利润同比(%) gpr,毛利率(%) npr,净利润率(%)
holders,股东人数
"""

def cal_weight(x):
    '''熵值法计算变量的权重'''
    # 标准化
    x = x.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))
 
    # 求k
    rows = x.index.size  # 行
    cols = x.columns.size  # 列
    k = 1.0 / math.log(rows)
 
    lnf = [[None] * cols for i in range(rows)]
 
    # 矩阵计算--
    # 信息熵
    # p=array(p)
    x = array(x)
    lnf = [[None] * cols for i in range(rows)]
    lnf = array(lnf)
    for i in range(0, rows):
        for j in range(0, cols):
            if x[i][j] == 0:
                lnfij = 0.0
            else:
                p = x[i][j] / x.sum(axis=0)[j]
                lnfij = math.log(p) * p * (-k)
            lnf[i][j] = lnfij
    lnf = pd.DataFrame(lnf)
    E = lnf
 
    # 计算冗余度
    d = 1 - E.sum(axis=0)
    # 计算各指标的权重
    w = [[None] * 1 for i in range(cols)]
    for j in range(0, cols):
        wj = d[j] / sum(d)
        w[j] = wj
        # 计算各样本的综合得分,用最原始的数据
    
    w = pd.DataFrame(w)
    return w

def uniform(x):
    uniform_mat = x.index.copy()
    min_index = {column:min(uniform_mat[column]) for column in uniform_mat.columns}
    max_index = {column:max(uniform_mat[column]) for column in uniform_mat.columns}
    for i in range(len(uniform_mat)):
        for column in uniform_mat.columns:
            if column in x.negative:
                uniform_mat[column][i] = (uniform_mat[column][i] - min_index[column]) / (max_index[column] - min_index[column])
            else:
                uniform_mat[column][i] = (max_index[column] - uniform_mat[column][i]) / (max_index[column] - min_index[column])

    x.uniform_mat = uniform_mat
    return x.uniform_mat



pd.set_option('display.width',None)
pd.set_option('display.max_rows',None)


stock = ts.get_stock_basics()

stock_sort = stock.loc[(stock['industry']=='通信设备')]
stock_sort.to_csv('stock.csv')
stock_name = stock_sort['name']
stock_sort.drop(['name','industry','timeToMarket','area','holders'],axis=1,inplace=True)
stock_re=stock_sort.dropna().reset_index(drop=True)

w = cal_weight(stock_re)
w.index = stock_re.columns
w.columns = ['weight']
print(w)
print(stock_name)


score = pd.Series(
        [np.dot(np.array(stock_re[row:row+1])[0], np.array(w)) for row in range(len(stock_re))],
        index=stock_name, name='得分'
    ).sort_values(ascending=False)
print(score,type(score))



        
score.to_csv('score.csv')

# Positive = indexs
# Negative = []

# stock_code = stock_sort['code']
# index = stock_sort[indexs]
# em = EmtropyMethod(index, Negative, Positive, stock_codee)
# em.uniform()

# stock_sort.to_csv('stock_sort.csv')
# stock_re.to_csv('stock_re.csv')



# print(stock, type(stock))
# print(stock_sort)
