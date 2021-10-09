# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:35:48 2021

@author: VONO
"""

import pandas as pd
import numpy as np
import math
import statsmodels.api as sm 

#导入数据
m_ret = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\CRSP_common_stock_monthly.csv")
beta  = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\Result_monthly_beta.csv")
FF_Rf = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\FF_monthly_Rf.csv")
BAB   = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\Result_BAB_factor.csv")
FF4    = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\FF_5factor\FF_4factor.csv")


#首先处理无风险数据
FF_Rf['year']=(FF_Rf['dateff']/10000).apply(int)
FF_Rf['month']=(FF_Rf['dateff']/100).apply(int)-FF_Rf['year']*100
FF_Rf.drop(columns=['dateff'],inplace=True)

#然后处理common stock数据，分离年份和月份
m_ret['year']=(m_ret['date']/10000).apply(int)
m_ret['month']=(m_ret['date']/100).apply(int)-m_ret['year']*100
m_ret.rename(columns={'PERMNO':'id'},inplace=True)
m_ret.drop(columns=['date','SHRCD'],inplace=True)
m_ret['RET'].replace(['B','C'],[np.nan,np.nan],inplace=True)
m_ret['RET']=m_ret['RET'].apply(float)

#合并m_ret和FF_Rf
stock=pd.merge(m_ret, FF_Rf, on=['year','month'], how='left')
stock=pd.merge(stock,beta,on=['id','year','month'])
del m_ret
del FF_Rf
del beta

#求stock的超额收益
stock['ret']=stock['RET']-stock['rf']
stock.drop(columns=['RET','rf'],inplace=True)

#计算BAB的超额收益
BAB['date']=BAB['year']*100+BAB['month']
BAB.drop(columns=['Unnamed: 0','year','month'],inplace=True)
BAB=BAB[BAB['date']>=192904]
Excess_BAB=BAB['BAB'].mean()
Excess_BAB_t=BAB['BAB'].mean()/(BAB['BAB'].std()/math.sqrt(BAB['date'].count()))


#计算分组的beta取值(EXCHCD要求数值是1，代表NYSE的上市公司)
stock['date']=stock['year']*100+stock['month']
stock.drop(columns=['year','month'],inplace=True)
rank=stock[stock['EXCHCD']==1]
for i in range(1,10):
    bp=rank.groupby(['date'])['beta'].quantile(q=0.1*i)
    bp=pd.DataFrame(bp)
    bp.rename(columns={'beta':'bp{}'.format(i)},inplace=True)
    stock=pd.merge(stock,bp,on=['date'],how='left')
    del bp
del rank
del i
stock['rank']=1
#为了计算，定义rank_beta函数
def rank_beta(x):
    y=1
    for i in range(1,10):
        if x['beta'] > x['bp{}'.format(i)]:
            y=y+1
        else:
            break
    return y
'''
def rank_betav2(x):
    breaks = np.array(x[["bp1","bp2","bp3","bp4","bp5","bp6","bp7","bp8","bp9"]])
    return (breaks < x["beta"]).sum()+1
stock.iloc[:10000,3:14].apply(lambda x:rank_betav2(x),axis=1)
'''
#注意这里应该先解决beta取值是nan的问题
stock['rank']=stock.apply(rank_beta,axis=1)


#Regression Part
#首先处理FF4的数据
FF4['date']=(FF4['dateff']/100).apply(int)
FF4.drop(columns=['dateff'],inplace=True)
stock=stock[['id','date','ret','rank']]
stock=pd.merge(stock, FF4, on=['date'],how='left')

#删除所有stock中ret为nan的值
stock=stock[stock['ret'].isnull()==False]
stock.groupby('rank')['ret'].mean()

def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()
    return result.params

stock.groupby('rank').apply(regress, 'ret', ['mktrf'])

