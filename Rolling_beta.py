# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 23:54:22 2021

@author: VONO
"""

import pandas as pd
import numpy as np


#导入数据
stock=pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\CRSP_common_stock.csv")
stock.drop(columns=['SHRCD'],inplace=True)
rf=pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\FF_daily_Rf.csv")
stock=pd.merge(stock,rf,on='date',how='left')
del rf

#计算超额收益率并设置时间作为索引
stock['RET'].replace(['B','C'],[np.nan,np.nan],inplace=True)
stock['RET']=stock['RET'].apply(float)
stock['ret']=stock['RET']-stock['rf']
stock['mr']=stock['vwretd']-stock['rf']
stock.rename(columns={'PERMNO':'id'},inplace=True)
stock=stock[['id','date','ret','mr']]
stock.set_index(pd.to_datetime(stock['date'].astype(str)),inplace=True)
stock.drop(columns=['date'],inplace=True)

#计算个股的rolling volatility
stock.sort_values(by=['id','date'],inplace=True)
stock['lnret']=np.log(1+stock['ret'])
rstd=stock.groupby(stock['id'])['lnret'].rolling('365D',min_periods=120).std().reset_index()
rstd.rename(columns={'lnret':'rstd'},inplace=True)


#计算市场的rolling volatility
market=stock[['mr']]
market = market.reset_index()
market.sort_values(by=['date'],inplace=True)
market = market.drop_duplicates(subset=["date","mr"])
market = market.set_index("date")
mstd=np.log(1+market['mr']).rolling('365D',min_periods=120).std()
mstd=pd.DataFrame(mstd)
mstd.rename(columns={'mr':'mstd'},inplace=True)

#合并数据
stock=pd.merge(stock,mstd,on=['date'],how='left')
del market
del mstd
stock=stock.reset_index()
stock=pd.merge(stock,rstd,on=['id','date'],how='left')
del rstd


#计算相关系数
#首先计算个股加和回报率
stock.sort_values(by=['id','date'],inplace=True)
stock['lnret1']=stock['lnret']
stock.drop(columns=['lnret'],inplace=True)
stock['lnret']=stock.groupby(stock['id'])['lnret1'].apply(lambda x:x+x.shift(-1)+x.shift(-2))
stock.drop(columns=['lnret1','ret'],inplace=True)
#计算市场加和回报率
stock['lnmr1']=(stock['mr']+1).apply(np.log)
stock['lnmr']=stock.groupby(stock['id'])['lnmr1'].apply(lambda x:x+x.shift(-1)+x.shift(-2))
stock.drop(columns=['lnmr1','mr'],inplace=True)
#计算相关系数
stock=stock.set_index('date',drop=True)
corr=stock[['id','lnret','lnmr']].groupby('id').rolling('1826D',min_periods=750).corr()
corr=corr.loc[(slice(None),slice(None),'lnret'),'lnmr']
corr=pd.DataFrame(corr)
corr.rename(columns={'lnmr':'corr'},inplace=True)

#再次合并数据
stock=pd.merge(stock,corr,on=['id','date'],how='left')
del corr
stock['beta']=stock['corr']*stock['rstd']/stock['mstd']
beta=stock[['id','beta']]
del stock

#抽取月度数据
beta=beta.reset_index()
beta['year']=beta['date'].apply(lambda x: x.year)
beta['month']=beta['date'].apply(lambda x: x.month)
month_beta=beta.groupby([beta['id'],beta['year'],beta['month']]).max()

#输出月度数据
month_beta.to_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\Result_monthly_beta.csv")

