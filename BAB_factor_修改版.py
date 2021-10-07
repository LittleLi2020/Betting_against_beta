# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 09:26:31 2021

@author: VONO
"""

import pandas as pd
import numpy as np
import scipy.stats as st


# In[256]:


def rank_data(x):
    '''
    得到同一日期内各个证券beta的排序
    '''
    x["rank"] = st.rankdata(x["weight_beta"])
    x = x.drop(["weight_beta"],axis = 1)
    return x


# In[257]:


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


# In[259]:


stock = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\Result_monthly_beta.csv")
m_ret = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\CRSP_common_stock_monthly.csv")
FF_Rf = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\FF_monthly_Rf.csv")

# 计算weight_beta
w = 0.6
beta_xs = 1
stock["weight_beta"] = w * stock["beta"] + (1 - w) * beta_xs
# 提取出有效的行
stock['judge']=stock['beta'].isnull()
stock=stock.loc[stock['judge']==False]

#删除不需要的文件和数据
del beta_xs
del w
stock.drop(columns=['date','beta','judge'],inplace=True)

# 计算同一日期各个证券beta的排序
stock['rank']=stock.groupby(['year','month'])['weight_beta'].rank()
#ranks = stock.loc[pd.notna(stock["weight_beta"]),:].groupby(['year','month'])["id","year","month","weight_beta"].apply(rank_data)
# 计算同一日期各个证券的投资权重
#ranks=ranks.groupby(['year','month']).apply(rank_weight)
z_bar=stock.groupby(['year','month'])['rank'].mean()
z_bar=pd.DataFrame(z_bar)
z_bar.rename(columns={'rank':'z_bar'},inplace=True)
stock=pd.merge(stock,z_bar,on=['year','month'],how='left')
stock['z_minus']=np.abs(stock["rank"] - stock['z_bar'])
z_minus_sum=stock.groupby(['year','month'])['z_minus'].sum()
z_minus_sum=pd.DataFrame(z_minus_sum)
z_minus_sum.rename(columns={'z_minus':'z_minus_sum'},inplace=True)
stock=pd.merge(stock,z_minus_sum,on=['year','month'],how='left')
stock['k']=2/stock['z_minus_sum']
stock['w_H']=np.maximum(stock['k']*(stock["rank"] - stock['z_bar']),0)
stock['w_L']=-np.minimum(stock['k']*(stock["rank"] - stock['z_bar']),0)



# 将投资权重merge进入stock_val，得到stock_fin
#stock_fin = pd.merge(stock,ranks,on=['id','year','month'],how = "left")
#stock_fin.sort_values(by=['id','year','month'],inplace=True)
stock_fin=stock[['id','year','month','weight_beta','rank','w_H','w_L']]
#删除不需要的文件和数据
#del ranks
#del stock


#处理月度收益率
m_ret['year']=(m_ret['date']/10000).apply(int)
m_ret['month']=(m_ret['date']/100).apply(int)-m_ret['year']*100
m_ret.rename(columns={'PERMNO':'id'},inplace=True)
m_ret.drop(columns=['date','SHRCD'],inplace=True)
m_ret['RET'].replace(['B','C'],[np.nan,np.nan],inplace=True)
m_ret['RET']=m_ret['RET'].apply(float)
#合并数据
stock_fin = pd.merge(stock_fin,m_ret,on=['id','year','month'],how = "left")
del m_ret


#处理无风险收益率
FF_Rf['year']=(FF_Rf['dateff']/10000).apply(int)
FF_Rf['month']=(FF_Rf['dateff']/100).apply(int)-FF_Rf['year']*100
#合并数据
stock_fin = pd.merge(stock_fin,FF_Rf,on=['year','month'],how = "left")
del FF_Rf



# 将少量缺失的return设置为用0来填充
#stock_fin.loc[pd.isna(stock_fin["RET"]),"RET"] = 0
# 如果某天某证券的beta没法计算，则将他们的投资权重设置为0，即不投资他们
#stock_fin.loc[pd.isna(stock_fin["weight_beta"]),"w_H"] = 0
#stock_fin.loc[pd.isna(stock_fin["weight_beta"]),"w_L"] = 0
# 如果某天某证券的beta没法计算，则将他们的beta设置为1。因为该证券的投资权重为0，所以不管设置他们的beta为多少，都不会影响BAB Factor的计算
#stock_fin.loc[pd.isna(stock_fin["weight_beta"]),"weight_beta"] = 1

#删除所有return的缺失值
stock_fin=stock_fin.loc[stock_fin['RET'].isnull()==False]



# 计算high/low beta portfolio的收益
stock_fin["w_H1"] = stock_fin.groupby("id")["w_H"].shift(1)
stock_fin["w_L1"] = stock_fin.groupby("id")["w_L"].shift(1)
    #下面删除w_H和w_L缺失的行
stock_fin=stock_fin.loc[(stock_fin['w_H1'].isnull()==False) | (stock_fin['w_L1'].isnull()==False)]
stock_fin=stock_fin.loc[(stock_fin['w_H'].isnull()==False) | (stock_fin['w_L'].isnull()==False)]


r_L = stock_fin.groupby(['year','month']).apply(lambda x: np.dot(x["RET"],x["w_L1"]))
r_H = stock_fin.groupby(['year','month']).apply(lambda x: np.dot(x["RET"],x["w_H1"]))
# 计算high/low beta portfolio的beta
beta_L = stock_fin.groupby(['year','month']).apply(lambda x: np.dot(x["weight_beta"],x["w_L"]))
beta_H = stock_fin.groupby(['year','month']).apply(lambda x: np.dot(x["weight_beta"],x["w_H"]))
beta_L.name = "beta_L"
r_L.name = "r_L"
beta_H.name = "beta_H"
r_H.name = "r_H"
# 计算BAB factor
BAB_factor_L = pd.merge(beta_L,r_L,how = "outer",on = ['year','month'])
BAB_factor_H = pd.merge(beta_H,r_H,how = "outer",on = ['year','month'])
BAB_factor = pd.merge(BAB_factor_L,BAB_factor_H, how = "outer",on = ['year','month'])
Rf = stock_fin[['year','month','rf']]
Rf = Rf.drop_duplicates(subset=['year','month','rf'])

BAB_factor = pd.merge(BAB_factor,Rf, how = "outer", on = ['year','month'])
BAB_factor["beta_L1"] =  BAB_factor["beta_L"].shift(1)
BAB_factor["beta_H1"] =  BAB_factor["beta_H"].shift(1)
BAB_factor["rf1"] = BAB_factor["rf"].shift(1)
BAB_factor["BAB"] = (BAB_factor["r_L"] - BAB_factor["rf1"]) / BAB_factor["beta_L1"] - (BAB_factor["r_H"] - BAB_factor["rf1"]) / BAB_factor["beta_H1"]
#BAB_factor["BAB"]=BAB_factor["BAB_1"].shift(1)

#导出数据
BAB_factor[['year','month','BAB']].to_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\Result_BAB_factor.csv")






