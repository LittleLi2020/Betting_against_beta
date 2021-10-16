# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:28:38 2021

@author: VONO
"""

import pandas as pd
import numpy as np
import math
import statsmodels.formula.api as sm


#导入数据
m_ret = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\CRSP_common_stock_monthly.csv")
beta  = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\Result_monthly_beta.csv")
FF_Rf = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\FF_monthly_Rf.csv")
BAB  = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\HW1\data\Renew_data\Result_BAB_factor.csv")
FF4  = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\FF_5factor\FF_4factor.csv")
Liquid_factor = pd.read_csv(r"D:\Courses\2021-2022_1\实证金融学\FF_5factor\liquidity factor.csv")


def get_alpha(df,y_name,beta_rank = None):
    if beta_rank != None:
        df = df.loc[df["position"] == beta_rank,:]
        y_new_name = "P" + f"_{beta_rank + 1}"
        df.rename(columns = {y_name:y_new_name},inplace = True)
        y_name = y_new_name
    Excess_return=df[y_name].mean()
    Volatility = (df[y_name]*12).var()
    SharpeRatio = (Excess_return*12) / Volatility
    tvalue = Excess_return/((math.sqrt(Volatility)/12)/math.sqrt(df.shape[0]))
    #print("{} portfolio's excess return is: {:.2f}%".format(y_name,Excess_return*100))
    #print("{} portfolio's volatility is: {:.2f}%".format(y_name,Volatility*10000))
    #print("{} portfolio's sharpe ratio is: {:.2f}".format(y_name,SharpeRatio))
    summary = {}
    summary["Excess_return"] = round(Excess_return*100,2)
    summary["Excess_return_tvalue"] = round(tvalue, 2) 
    # n-factor model, n = 1,3,4,5
    ## CPAM alpha
    result = sm.ols(formula = f"{y_name} ~ 1 + mktrf",data = df).fit()
    #print("{} portfolio's CAPM alpha is: {:.2f}".format(y_name,result.params["Intercept"]*100))
    summary["CAPM_alpha"] = round(result.params["Intercept"]*100,2)
    summary["CAPM_tvalue"] = round(result.tvalues["Intercept"],2)
    ## Three-factor alpha
    result = sm.ols(formula = f"{y_name} ~ 1 + mktrf+ smb+ hml",data = df).fit()
    #print("{} portfolio's three-factor alpha is: {:.2f}".format(y_name,result.params["Intercept"]*100))
    summary["three_factor_alpha"] = round(result.params["Intercept"]*100,2)
    summary["three_factor_tvalue"] = round(result.tvalues["Intercept"],2)
    ## Four-factor alpha
    result = sm.ols(formula = f"{y_name} ~ 1 + mktrf+ smb+ hml+ umd",data = df).fit()
    #print("{} portfolio's four-factor alpha is: {:.2f}".format(y_name,result.params["Intercept"]*100))
    summary["four_factor_alpha"] = round(result.params["Intercept"]*100,2)
    summary["four_factor_tvalue"] = round(result.tvalues["Intercept"],2)
    ## Five-factor alpha
    result = sm.ols(formula = f"{y_name} ~ 1 + mktrf+ smb+ hml+ umd+ vwf",data = df,missing='drop').fit()
    #print("{} portfolio's five-factor alpha is: {:.2f}".format(y_name,result.params["Intercept"]*100))
    summary["five_factor_alpha"] = round(result.params["Intercept"]*100,2)
    summary["five_factor_tvalue"] = round(result.tvalues["Intercept"],2)
    summary["Volatility"] = round(Volatility * 100,3)
    summary["SharpeRatio"] = round(SharpeRatio,2)
    summary = pd.DataFrame(summary,index = [y_name]).T
    return summary



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
m_ret=m_ret[m_ret['RET'].isnull()==False]


#合并m_ret和FF_Rf
stock=pd.merge(m_ret, FF_Rf, on=['year','month'], how='left')
stock=pd.merge(stock,beta,on=['id','year','month'],how='right')
del m_ret
del FF_Rf
del beta


#求stock的超额收益
stock['ret']=stock['RET']-stock['rf']
stock.drop(columns=['RET','rf'],inplace=True)



#计算BAB的超额收益
BAB['date']=BAB['year']*100+BAB['month']
BAB.drop(columns=['year','month'],inplace=True)
BAB=BAB[BAB['date']>=192904]


'''
Excess_BAB=BAB['BAB'].mean()
Volatility_BAB = (BAB["BAB"]*12).var()
SharpeRatio_BAB = (Excess_BAB*12) / Volatility_BAB #原文中的sharpe ratio是用方差做分母的
print("BAB portfolio's excess return is: {:.2f}%".format(Excess_BAB*100))
print("BAB portfolio's volatility is: {:.2f}%".format(Volatility_BAB*100))
print("BAB portfolio's sharpe ratio is: {:.2f}".format(SharpeRatio_BAB))
summary = {}
summary["Excess_return"] = round(Excess_BAB*100,2)
'''


# n-factor model, n = 1,3,4,5
## 首先处理FF4的数据
FF4['date']=(FF4['dateff']/100).apply(int)
FF4.drop(columns=['dateff'],inplace=True)
BAB = pd.merge(BAB,FF4,on = "date", how = "left")

# 处理流动性因子数据
Liquid_factor = Liquid_factor[["DATE","PS_VWF"]]
Liquid_factor = Liquid_factor.rename(columns = {"DATE":"date","PS_VWF":"vwf"})
Liquid_factor["date"] = (Liquid_factor['date']/100).apply(int)
BAB = pd.merge(BAB,Liquid_factor,on = "date", how = "left")



#计算分组的beta取值(EXCHCD要求数值是1，代表NYSE的上市公司)
stock['date']=stock['year']*100+stock['month']
stock.drop(columns=['year','month'],inplace=True)
stock.dropna(subset = ["beta"],inplace = True)
ranks=stock[stock['EXCHCD']==1]


# 计算各个日期的beta分位数
for i in range(1,10):
    bp=ranks.groupby(['date'])['beta'].quantile(q=0.1*i)
    bp=pd.DataFrame(bp)
    bp.rename(columns={'beta':'bp{}'.format(i)},inplace=True)
    stock=pd.merge(stock,bp,on=['date'],how='left')
    
# 计算各个股票在各个日期的beta排序
position = np.array(stock["beta"]).reshape(len(stock),1) > np.array(stock[["bp1","bp2","bp3","bp4","bp5","bp6","bp7","bp8","bp9"]])
position = position.sum(axis = 1)
position = pd.DataFrame(position,columns = ["rank"])
stock = pd.concat([stock,position],axis = 1)
stock.drop(["bp1","bp2","bp3","bp4","bp5","bp6","bp7","bp8","bp9"],axis = 1,inplace = True)
stock.sort_values(by=['id','date'],inplace=True)
stock['position']=stock.groupby('id')['rank'].shift(1)
stock=stock[stock['position'].isnull()==False]


#计算portfolio的收益率
portfolio=stock.groupby(['date','position'])['ret'].mean()
portfolio=pd.DataFrame(portfolio)
portfolio=portfolio.reset_index()
portfolio = pd.merge(portfolio,FF4,on = "date", how = "left")
portfolio = pd.merge(portfolio,Liquid_factor,on = "date", how = "left")
portfolio = portfolio[portfolio['date']>=192904]

summaries = pd.DataFrame()
for i in range(10):
    summary = get_alpha(portfolio,"ret",i)
    summaries = pd.concat([summaries,summary],axis = 1)
    
summary_BAB = get_alpha(BAB,"BAB")
summaries = pd.concat([summaries,summary_BAB],axis = 1) 


