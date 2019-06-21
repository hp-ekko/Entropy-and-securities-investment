import tushare as ts
import pandas as pd
import numpy as np
import math
from numpy import array
pd.set_option('display.width',None)
pd.set_option('display.max_rows',None)

class EmtropyMethod:
    def __init__(self, index, positive, negative, row_name):
        if len(index) != len(row_name):
            raise Exception('数据指标行数与行名称数不符')
        if sorted(index.columns) != sorted(positive+negative):
            raise Exception('正项指标加负向指标不等于数据指标的条目数')
 
        self.index = index.copy().astype('float64')
        self.positive = positive
        self.negative = negative
        self.row_name = row_name
        
    def uniform(self):
        uniform_mat = self.index.copy()
        min_index = {column:min(uniform_mat[column]) for column in uniform_mat.columns}
        max_index = {column:max(uniform_mat[column]) for column in uniform_mat.columns}
        for i in range(len(uniform_mat)):
            for column in uniform_mat.columns:
                if column in self.negative:
                    uniform_mat[column][i] = (uniform_mat[column][i] - min_index[column]) / (max_index[column] - min_index[column])
                else:
                    uniform_mat[column][i] = (max_index[column] - uniform_mat[column][i]) / (max_index[column] - min_index[column])
 
        self.uniform_mat = uniform_mat
        return self.uniform_mat
        
    def calc_probability(self):
        try:
            p_mat = self.uniform_mat.copy()
        except AttributeError:
            raise Exception('你还没进行归一化处理，请先调用uniform方法')
        for column in p_mat.columns:
            sigma_x_1_n_j = sum(p_mat[column])
            p_mat[column] = p_mat[column].apply(lambda x_i_j: x_i_j / sigma_x_1_n_j if x_i_j / sigma_x_1_n_j != 0 else 1e-6)
 
        self.p_mat = p_mat
        return p_mat
                 
    def calc_emtropy(self):
        try:
            self.p_mat.head(0)
        except AttributeError:
            raise Exception('你还没计算比重，请先调用calc_probability方法')
 
        import numpy as np
        e_j  = -(1 / np.log(len(self.p_mat)+1)) * np.array([sum([pij*np.log(pij) for pij in self.p_mat[column]]) for column in self.p_mat.columns])
        ejs = pd.Series(e_j, index=self.p_mat.columns, name='指标的熵值')
 
        self.emtropy_series = ejs
        return self.emtropy_series
        
    def calc_emtropy_redundancy(self):
        try:
            self.d_series = 1 - self.emtropy_series
            self.d_series.name = '信息熵冗余度'
        except AttributeError:
            raise Exception('你还没计算信息熵，请先调用calc_emtropy方法')
 
        return self.d_series
            
    def calc_Weight(self):
        self.uniform()
        self.calc_probability()
        self.calc_emtropy()
        self.calc_emtropy_redundancy()
        self.Weight = self.d_series / sum(self.d_series)
        self.Weight.name = '权值'
        return self.Weight
            
    def calc_score(self):
        self.calc_Weight()
 
        import numpy as np
        self.score = pd.Series(
            [np.dot(np.array(self.index[row:row+1])[0], np.array(self.Weight)) for row in range(len(self.index))],
            index=self.row_name, name='得分'
        ).sort_values(ascending=False)
        return self.score

# 所有的股票列表与五种能力返回列表，选取通信行业所有的进行股票进行评价
stock = ts.get_stock_basics()  #股票列表
stock_1 = stock.loc[(stock['industry']=='通信设备')]
# stock_1.fillna(0)
# stock_1.to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/stock_1.csv')
stock_2 = ts.get_profit_data(2018,4) #盈利能力返回数据
stock_2.drop(['code'],axis=1,inplace=True)
# stock_2.fillna(0)
# stock_2.to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/stock_2.csv')
stock_3 = ts.get_operation_data(2018,4)
stock_3.drop(['code'],axis=1,inplace=True)#营运能力返回数据
# stock_3.fillna(0)
# stock_3.to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/stock_3.csv')
stock_4 = ts.get_growth_data(2018,4)
stock_4.drop(['code'],axis=1,inplace=True)#成长能力返回数据
# stock_4.fillna(0)
# stock_4.to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/stock_4.csv')
stock_5 = ts.get_debtpaying_data(2018,4)
stock_5.drop(['code'],axis=1,inplace=True)#偿债能力返回数据
# stock_5.fillna(0)
# stock_5.to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/stock_5.csv')
stock_6 = ts.get_cashflow_data(2018,4)
stock_6.drop(['code'],axis=1,inplace=True)#现金流返回数据
# stock_6.fillna(0)
# stock_6.to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/stock_6.csv')


print(stock_1.columns.values.tolist(),stock_2.columns.values.tolist(),stock_3.columns.values.tolist(),stock_4.columns.values.tolist(),stock_5.columns.values.tolist(),stock_6.columns.values.tolist()) 
# print(stock_1.columns.values.tolist(),stock_2.columns.values.tolist(),stock_3.columns.values.tolist()) 
#把六个数据库昂整合成一个
stock_all1 = pd.merge(stock_1 ,stock_2,on='name')
stock_all2 = pd.merge(stock_3 ,stock_4,on='name')
stock_all3 = pd.merge(stock_5 ,stock_6,on='name')
stock_all4 = pd.merge(stock_all1 ,stock_all2,on='name')
stock_all = pd.merge(stock_all3 ,stock_all4,on='name')
stock_name = stock_all['name']
# stock_all.drop(['name','industry','area'],axis=1,inplace=True)
# stock_all.to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/stock_all.csv')
stock_re=stock_all.dropna().reset_index(drop=True)
stock_name = stock_re['name']
stock_re.to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/stock_re.csv')

#选择代表盈利能力、运营、成长、偿债、现金的指标进行评分
profit_indexs = ['roe', 'net_profit_ratio', 'gross_profit_rate', 
            'net_profits', 'eps', 
            'business_income', 'bips']
operation_indexs = ['arturnover', 'arturndays', 'inventory_turnover', 'inventory_days', 'currentasset_turnover', 'currentasset_days']
growth_indexs =  ['mbrg', 'nprg', 'nav', 'targ', 'epsg', 'seg'] 
deb_indexs = ['currentratio', 'quickratio', 'cashratio', 'icratio', 'sheqratio', 'adratio'] 
cash_indexs = [ 'cf_sales', 'rateofreturn', 'cf_nm', 'cf_liabilities', 'cashflowratio']
#正向指标
Positive_profit = profit_indexs
Positive_op = operation_indexs
Positive_g = growth_indexs
Positive_deb = deb_indexs
Positive_cash = cash_indexs
#负向指标
Negative = [] 
#挑选出符合指标的数据
profit_index = stock_re[Positive_profit]
op_index = stock_re[Positive_op]
g_index = stock_re[Positive_g]
deb_index = stock_re[Positive_deb]
cash_index = stock_re[Positive_cash]
#五项能力权重及股票评分
stock_profit = EmtropyMethod(profit_index, Negative, Positive_profit, stock_name)
stock_op = EmtropyMethod(op_index, Negative, Positive_op, stock_name)
stock_g = EmtropyMethod(g_index, Negative, Positive_g, stock_name)
stock_deb= EmtropyMethod(deb_index, Negative, Positive_deb, stock_name)
stock_cash = EmtropyMethod(cash_index, Negative, Positive_cash, stock_name)

print(stock_profit.calc_Weight(),stock_profit.calc_score().head(10))
stock_profit.calc_Weight().to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/pro_w.csv')
stock_profit.calc_score().to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/pro_s.csv')

print(stock_op.calc_Weight(),stock_op.calc_score().head(10))
stock_op.calc_Weight().to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/op_w.csv')
stock_op.calc_score().to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/op_s.csv')

print(stock_g.calc_Weight(),stock_g.calc_score().head(10))
stock_g.calc_Weight().to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/g_w.csv')
stock_g.calc_score().to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/g_s.csv')


print(stock_deb.calc_Weight(),stock_deb.calc_score().head(10))
stock_deb.calc_Weight().to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/deb_w.csv')
stock_deb.calc_score().to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/deb_s.csv')


print(stock_cash.calc_Weight(),stock_cash.calc_score().head(10))
stock_cash.calc_Weight().to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/cash_w.csv')
stock_cash.calc_score().to_csv('/Users/ekko/Desktop/python/Entropy-and-securities-investment-master/cash_s.csv')
# print(stock_1,stock_2)

