# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 19:07:05 2021

@author: VONO
"""

import pandas as pd
import numpy as np
import scipy.stats as st

def rank_data(x):
    '''
    得到同一日期内各个证券beta的排序
    '''
    x["rank"] = st.rankdata(x["weight_beta"])
    x = x.drop(["weight_beta"],axis = 1)
    return x

def rank_weight(x):
    '''
    得到同一日期内各个证券的投资权重
    '''
    z_bar = np.mean(x["rank"])
    k = 2 / (np.sum(np.abs(x["rank"] - z_bar)))
    w_H = k * np.maximum(x["rank"] - z_bar,0)
    w_L = k * -np.minimum(x["rank"] - z_bar,0)
    x["w_H"] = w_H
    x["w_L"] = w_L
    return x


stock = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\Result_monthly_beta.csv")
m_ret = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\CRSP_common_stock_monthly.csv")
FF_Rf = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\FF_monthly_Rf.csv")


# 提取出有效的行
stock.dropna(subset = ["beta"],inplace = True)
# 去除日期不在每个月最后一日的行

stock["date"] =  pd.to_datetime(stock["date"])
stock["last_day"] = stock.groupby(["year","month"])["date"].transform(lambda v: max(v))
stock["is_last_date"] = stock["date"] == stock["last_day"]
stock = stock[stock["is_last_date"]]
stock.drop(["last_day","is_last_date"],axis = 1,inplace = True)

#stock.dropna(inplace=True)

# 计算weight_beta
w = 0.6
beta_xs = 1
stock["weight_beta"] = w * stock["beta"] + (1 - w) * beta_xs
# 计算同一日期各个证券beta的排序
ranks = stock.groupby("date")["id","date","weight_beta"].apply(rank_data)
# 计算同一日期各个证券的投资权重
ranks = ranks.groupby("date").apply(rank_weight)
# 将投资权重merge进入stock，得到stock
stock = pd.merge(stock,ranks,on=['id','date'],how = "left")
stock.sort_values(by=['id','date'],inplace=True)
# 去除没有配置资产的股票
# stock = stock.dropna(subset = ["w_L","w_H"])


#处理月度收益率
m_ret['year']=(m_ret['date']/10000).apply(int)
m_ret['month']=(m_ret['date']/100).apply(int)-m_ret['year']*100
m_ret.rename(columns={'PERMNO':'id'},inplace=True)
m_ret.drop(columns=['date','SHRCD'],inplace=True)
m_ret['RET'].replace(['B','C'],[np.nan,np.nan],inplace=True)
m_ret['RET']=m_ret['RET'].apply(float)
#m_ret['RET']=(1+m_ret['RET']).apply(np.log)
#合并数据
stock = pd.merge(stock,m_ret,on=['id','year','month'],how = "left")
del m_ret
stock.drop(["EXCHCD"],axis = 1,inplace = True)


#处理无风险收益率
FF_Rf['year']=(FF_Rf['dateff']/10000).apply(int)
FF_Rf['month']=(FF_Rf['dateff']/100).apply(int)-FF_Rf['year']*100
#合并数据
stock = pd.merge(stock,FF_Rf,on=['year','month'],how = "left")
del FF_Rf
stock.drop(["dateff"],axis = 1, inplace = True)


# 将8000多个obs缺失的RET设置为0
'''
这里把RET缺失值都删除看一下
'''
#stock["RET"] = stock["RET"].fillna(value = 0)
stock = stock[stock["RET"].isnull()==False]

# 计算high/low beta portfolio的收益
stock["w_H1"] = stock.groupby("id")["w_H"].shift(1)
stock["w_L1"] = stock.groupby("id")["w_L"].shift(1)
stock["weight_beta1"] = stock.groupby('id')['weight_beta'].shift(1)
stock.dropna(subset = ["w_H1","w_L1","weight_beta1"],inplace = True)

#下面删除w_H和w_L缺失的行
#stock=stock.loc[(stock['w_H1'].isnull()==False) | (stock['w_L1'].isnull()==False)]
#stock=stock.loc[(stock['w_H'].isnull()==False) | (stock['w_L'].isnull()==False)]



# 计算BAB factor
r_L = stock.groupby(['year','month']).apply(lambda x: np.dot(x["RET"],x["w_L1"]))
r_H = stock.groupby(['year','month']).apply(lambda x: np.dot(x["RET"],x["w_H1"]))
beta_L = stock.groupby(['year','month']).apply(lambda x: np.dot(x["weight_beta1"],x["w_L1"]))
beta_H = stock.groupby(['year','month']).apply(lambda x: np.dot(x["weight_beta1"],x["w_H1"]))
beta_L.name = "beta_L1"
r_L.name = "r_L"
beta_H.name = "beta_H1"
r_H.name = "r_H"
BAB_factor_L = pd.merge(beta_L,r_L,how = "outer",on = ['year','month'])
BAB_factor_H = pd.merge(beta_H,r_H,how = "outer",on = ['year','month'])
BAB_factor = pd.merge(BAB_factor_L,BAB_factor_H, how = "outer",on = ['year','month'])
Rf = stock[['year','month','rf']]
Rf = Rf.drop_duplicates(subset=['year','month'])
BAB_factor = pd.merge(BAB_factor,Rf, how = "outer", on = ['year','month'])
#BAB_factor["beta_L1"] =  BAB_factor["beta_L"].shift(1)
#BAB_factor["beta_H1"] =  BAB_factor["beta_H"].shift(1)
BAB_factor["BAB"] = (BAB_factor["r_L"] - BAB_factor["rf"]) / BAB_factor["beta_L1"] - (BAB_factor["r_H"] - BAB_factor["rf"]) / BAB_factor["beta_H1"]


#导出数据
BAB_factor[['year','month','BAB']].to_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\Result_BAB_factor.csv")



