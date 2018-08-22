# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:33:24 2018

@author: master
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import time
from sklearn import linear_model
from pandas.tseries.offsets import Day, MonthEnd
from matplotlib.font_manager import _rebuild
_rebuild()
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

from ShowProcess import Process

import cx_Oracle
dsn = cx_Oracle.makedsn('10.88.102.160','1521','gfdwdb1')
conn = cx_Oracle.connect('jcquery','83936666',dsn)



sql_stock_price = 'select S_INFO_WINDCODE, TRADE_DT, S_DQ_ADJCLOSE, S_DQ_PCTCHANGE from GFWIND.ashareeodprices'
sql_index_price = 'select S_INFO_WINDCODE, TRADE_DT, S_DQ_CLOSE, S_DQ_PCTCHANGE from GFWIND.aindexeodprices'
sql_stock_indicator = 'select S_INFO_WINDCODE,TRADE_DT,UP_DOWN_LIMIT_STATUS, LOWEST_HIGHEST_STATUS from GFWIND.ashareeodderivativeindicator'
sql_stock_flow = 'select S_INFO_WINDCODE,TRADE_DT,S_MFD_INFLOW_LARGE_ORDER from GFWIND.asharemoneyflow'

stock_price = pd.read_sql(sql_stock_price,conn)
index_price = pd.read_sql(sql_index_price,conn)
stock_indicator = pd.read_sql(sql_stock_indicator,conn)
stock_flow = pd.read_sql(sql_stock_flow,conn)


stock_price.to_pickle('stock_price.pkl')
index_price.to_pickle('index_price.pkl')
stock_indicator.to_pickle('stock_indicator.pkl')
stock_flow.to_pickle('stock_flow.pkl')



stock_price = pd.read_pickle('stock_price.pkl')
index_price = pd.read_pickle('index_price.pkl')
stock_indicator = pd.read_pickle('stock_indicator.pkl')

stock_price['TRADE_DT'] = pd.DatetimeIndex(stock_price['TRADE_DT'])
index_price['TRADE_DT'] = pd.DatetimeIndex(index_price['TRADE_DT'])


################################################################################
'''1. 情绪面指标'''
################################################################################

# 1-1 跑赢半年均线、30日均线股票数量占比
index_day_price = pd.pivot_table(index_price,values='S_DQ_CLOSE',columns='S_INFO_WINDCODE',index='TRADE_DT')

index_SH = index_day_price['000001.SH']
index_SZ = index_day_price['399001.SZ']

index_SH = index_SH.dropna()
index_SZ = index_SZ.dropna()

stock_day_price = pd.pivot_table(stock_price,values='S_DQ_ADJCLOSE',columns='S_INFO_WINDCODE',index='TRADE_DT')
stock_day_count = stock_day_price.count(axis=1)



#绘图函数（包含指数）
def StockIndex(data,index_1,index_2,label_name,file_name):
    c = index_1[-1]
    d = index_2[-1]
    fig = plt.figure(figsize=(20,5))
    ax1 = fig.add_subplot(111)
    ax1.grid(False)
    ax1.bar(data.index,data,width=2.5,linewidth=2,color='yellowgreen',label=label_name,zorder=1)
    ax1.set_ylabel(label_name)
    #ax1.set_ylim(-1000,1000)
    ax1.legend(loc='upper right')
    
    ax2 = ax1.twinx()
    ax2.grid(True)
    ax2.plot(index_1.index,index_1,color='red',linewidth=0.8,label='上证综指',zorder=5)
    ax2.plot(index_2.index,index_2*c/d,color='blue',linewidth=0.8,label='深证成指',zorder=6)
    ax2.set_ylabel('指数')
    #ax2.set_ylim(0,7000)
    ax2.legend(loc='upper left')
    ax2.set_xlabel('时间')
    plt.savefig(file_name+'.jpg',dpi=1000)


#1. 半年均线
halfyear_mean = stock_day_price.ewm(span=126,axis=0).mean()
halfyear_count = stock_day_price>halfyear_mean
halfyear_count = halfyear_count.sum(axis=1)
halfyear_pct = halfyear_count/stock_day_count

StockIndex(halfyear_pct,index_SH,index_SZ,label_name='超过半年均线占比',file_name='半年均线')
    
#2. 30日均线
month_mean = stock_day_price.ewm(span=20,axis=0).mean()
month_count = stock_day_price>month_mean
month_count = month_count.sum(axis=1)
month_pct = month_count/stock_day_count

StockIndex(month_pct,index_SH,index_SZ,label_name='超过月均线占比',file_name='月均线')


#3. 跑赢赢指数股票占比
index_day_pct = pd.pivot_table(index_price,values='S_DQ_PCTCHANGE',columns='S_INFO_WINDCODE',index='TRADE_DT')

index_pct_SH = index_day_pct['000001.SH']
index_pct_SZ = index_day_pct['399001.SZ']

index_pct_SH = index_pct_SH.dropna()
index_pct_SZ = index_pct_SZ.dropna()


stock_price['CODE'] = stock_price['S_INFO_WINDCODE'].apply(lambda x:x[-2:])
stock_SH_pct = pd.pivot_table(stock_price[stock_price['CODE'] == 'SH'],values='S_DQ_PCTCHANGE',columns='S_INFO_WINDCODE',index='TRADE_DT')
stock_SZ_pct = pd.pivot_table(stock_price[stock_price['CODE'] == 'SZ'],values='S_DQ_PCTCHANGE',columns='S_INFO_WINDCODE',index='TRADE_DT')


#筛选出上证股票池
excess_SH_count = pd.Series(index=index_pct_SH.index)

for d in range(index_SH.shape[0]):
    temp_1 = stock_SH_pct.iloc[d,:] > index_pct_SH[d]
    excess_SH_count[d] = temp_1.sum()

excess_SH_pct = excess_SH_count/stock_day_count

#筛选出深证股票池
stock_SZ_pct = pd.merge(index_pct_SZ.reset_index(),stock_SZ_pct.reset_index(),on='TRADE_DT',how='left')
stock_SZ_pct = stock_SZ_pct.drop('399001.SZ',axis=1)
stock_SZ_pct = stock_SZ_pct.set_index('TRADE_DT')
stock_SZ_pct = stock_SZ_pct.dropna(axis=0,how='all')

excess_SZ_count = pd.Series(index=index_pct_SZ.index)

for d in range(index_SZ.shape[0]):
    temp_2 = stock_SZ_pct.iloc[d,:] > index_pct_SZ[d]
    excess_SZ_count[d] = temp_2.sum()

excess_SZ_pct = excess_SZ_count/stock_day_count


#上证
StockIndex(excess_SH_pct,index_SH,index_SZ,label_name='跑赢上证指数占比',file_name='跑赢上证指数')
#深证
StockIndex(excess_SZ_pct,index_SH,index_SZ,label_name='跑赢深证指数占比',file_name='跑赢深证指数')


#4. 上证综指&深证成指涨跌数占比

up_SH = stock_SH_pct>0
up_SH = up_SH.sum(axis=1)
diff_SH = 2*up_SH - stock_SH_pct.count(axis=1)
diff_pct_SH = diff_SH/stock_SH_pct.count(axis=1)

up_SZ = stock_SZ_pct>0
up_SZ = up_SZ.sum(axis=1)
diff_SZ = 2*up_SZ - stock_SZ_pct.count(axis=1)
diff_pct_SZ = diff_SZ/stock_SZ_pct.count(axis=1)

#上证
StockIndex(diff_pct_SH,index_SH,index_SZ,label_name='上涨下跌股票数量差值百分比',file_name='上证涨跌差值百分比')
#深证
StockIndex(diff_pct_SZ,index_SH,index_SZ,label_name='上涨下跌股票数量差值百分比',file_name='深证涨跌差值百分比')


#5. 创新高数量(过去一年)

def PassHighLow(data_x):
    new_high = pd.Series(index=data_x.index)
    new_low = pd.Series(index=data_x.index)
    #max_steps = data_x.shape[0] #时间
    #process_bar = Process(max_steps)
    
    for d in range(data_x.shape[0]):
        #process_bar.show_process()
        if d<250:
            temp_max = np.max(data_x.iloc[:d+1,:],axis=0) <= data_x.iloc[d,:]
            temp_min = np.min(data_x.iloc[:d+1,:],axis=0) >= data_x.iloc[d,:]
            new_high[d] = temp_max.sum()
            new_low[d] = temp_min.sum()
        else:
            temp_max = np.max(data_x.iloc[(d+1-250):(d+1),:],axis=0) <= data_x.iloc[d,:]
            temp_min = np.min(data_x.iloc[(d+1-250):(d+1),:],axis=0) >= data_x.iloc[d,:]
            new_high[d] = temp_max.sum()
            new_low[d] = temp_min.sum()
    
    return new_high,new_low        
            

new_high,new_low = PassHighLow(stock_day_price)
new_high_pct = new_high/stock_day_count
new_low_pct = new_low/stock_day_count

#新高
StockIndex(new_high_pct,index_SH,index_SZ,label_name='创过去一年新高数量占比',file_name='创过去一年新高数量占比')
#新低
StockIndex(new_low_pct,index_SH,index_SZ,label_name='创过去一年新低数量占比',file_name='创过去一年新低数量占比')

#6. 涨跌停数量占比

high_low_statues = pd.pivot_table(stock_indicator,values='UP_DOWN_LIMIT_STATUS',columns='S_INFO_WINDCODE',index='TRADE_DT')

high_count = high_low_statues == 1
high_count = high_count.sum(axis=1)
high_pct = high_count/stock_day_count
high_pct = high_pct.dropna()

low_count = high_low_statues == -1
low_count = low_count.sum(axis=1)
low_pct = low_count/stock_day_count
low_pct = low_pct.dropna()

#涨停
StockIndex(high_pct,index_SH,index_SZ,label_name='涨停数量占比',file_name='涨停数量占比')
#跌停
StockIndex(low_pct,index_SH,index_SZ,label_name='跌停数量占比',file_name='跌停数量占比')



################################################################################
'''2. 资金面指标'''
################################################################################

#1.沪深融资净买入

capital_loan = pd.read_excel('融资规模分析.xlsx')
capital_loan = capital_loan.set_index('截止日')
StockIndex(capital_loan['期间净买入额(亿元)'],index_SH[-2100:],index_SZ[-2100:],label_name='融资净买入',file_name='融资净买入')

#2.沪深融券净买入

security_loan = pd.read_excel('融券规模分析.xlsx')
security_loan = security_loan.set_index('截止日')
StockIndex(security_loan['期间净卖出额(亿元)'],index_SH[-500:],index_SZ[-500:],label_name='融券净卖出',file_name='融券净卖出')

#3.沪港深净买入
A_HK_stat = pd.read_excel('沪港通统计.xls')
A_HK_stat = A_HK_stat.set_index('指标名称')

#沪股通
StockIndex(A_HK_stat['沪股通:当日资金净流入(人民币)'],index_SH[-950:],index_SZ[-950:],label_name='沪股通资金流入',file_name='沪股通资金流入')
#深股通
StockIndex(A_HK_stat['深股通:当日资金净流入(人民币)'],index_SH[-950:],index_SZ[-950:],label_name='深股通资金流入',file_name='深股通资金流入')


#4.主力净流入

stock_flow = pd.read_pickle('stock_flow.pkl')
stock_mainflow = pd.pivot_table(stock_flow,values='S_MFD_INFLOW_LARGE_ORDER',index='TRADE_DT',columns='S_INFO_WINDCODE')
stock_mainflow.index = pd.DatetimeIndex(stock_mainflow.index)

total_mainflow = stock_mainflow.sum(axis=1)

StockIndex(total_mainflow,index_SH[-2900:],index_SZ[-2900:],label_name='A股主力资金净流入(万元)',file_name='A股主力资金净流入')


################################################################################
'''3. 衍生品'''
################################################################################

#1.股指期货
index_futures = pd.read_excel('股指期货.xls') 

index_futures = index_futures.set_index('指标名称')
index_futures.index = pd.DatetimeIndex('index_futures.index')

basis_SS300 = index_futures['沪深300指数'] - index_futures['期货收盘价(连续):沪深300指数期货']
basis_SH50 = index_futures['上证50指数'] - index_futures['期货收盘价(连续):上证50股指期货']
basis_ZZ500 = index_futures['中证500指数'] - index_futures['期货收盘价(连续):中证500股指期货']

basis_SS300 = basis_SS300.dropna()
basis_SH50 = basis_SH50.dropna()
basis_ZZ500 = basis_ZZ500.dropna()

def IndexFutures(data,index_1,label_name_a,label_name_b,file_name):
    fig = plt.figure(figsize=(20,5))
    ax1 = fig.add_subplot(111)
    ax1.grid(False)
    ax1.bar(data.index,data,width=2.5,linewidth=2,color='yellowgreen',label=label_name_a,zorder=1)
    ax1.set_ylabel(label_name_a)
    #ax1.set_ylim(-1000,1000)
    ax1.legend(loc='upper right')
    
    ax2 = ax1.twinx()
    ax2.grid(True)
    ax2.plot(index_1.index,index_1,linewidth=1.5,label=label_name_b,zorder=5)
    ax2.set_ylabel(label_name_b)
    #ax2.set_ylim(0,7000)
    ax2.legend(loc='upper left')
    ax2.set_xlabel('时间')
    plt.savefig(file_name+'.jpg',dpi=1000)



IndexFutures(basis_SS300,index_futures['沪深300指数'][-2100:],label_name_a='IF基差',label_name_b='沪深300',file_name='沪深300')

IndexFutures(basis_SH50,index_futures['上证50指数'][-900:],label_name_a='IH基差',label_name_b='上证50',file_name='上证50')

IndexFutures(basis_ZZ500,index_futures['中证500指数'][-900:],label_name_a='IC基差',label_name_b='中证500',file_name='中证500')


#2.50ETF 期权PCR

option_day_50ETF = pd.read_excel('50ETF期权日成交统计.xlsx')
option_month_50ETF = pd.read_excel('50ETF期权月成交统计.xlsx')
option_50ETF = pd.read_excel('50ETF_option.xlsx')

option_50ETF = option_50ETF.set_index('日期')
option_50ETF.index = pd.DatetimeIndex(option_50ETF.index)
option_50ETF.sort_index(inplace=True)


option_day_50ETF = option_day_50ETF.set_index('日期')
option_day_50ETF.index = pd.DatetimeIndex(option_day_50ETF.index)
option_day_50ETF.sort_index(inplace=True)

option_month_50ETF = option_month_50ETF.set_index('月份')
option_month_50ETF.index = pd.to_datetime(option_month_50ETF.index,format='%Y%m')
option_month_50ETF.sort_index(inplace=True)



a = index_SH[-1]
b = option_50ETF['收盘价(元)'][-1]
fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(111)
ax1.grid(False)
ax1.bar(option_day_50ETF['日认沽/认购(%)'].index, option_day_50ETF['日认沽/认购(%)']/100,width=2.5,linewidth=2,color='yellowgreen',label='50ETF期权日成交量PCR',zorder=2)
ax1.bar(option_month_50ETF['月认沽/认购(%)'].index, option_month_50ETF['月认沽/认购(%)']/100,width=5,linewidth=2,color='orange',label='50ETF期权月成交量PCR',zorder=3)
ax1.set_ylabel('50ETF期权成交量PCR')
#ax1.set_ylim(-1000,1000)
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.grid(True)
ax2.plot(option_50ETF.index[-900:],option_50ETF['收盘价(元)'][-900:],linewidth=1.8,label='标的物日收盘价（元）',zorder=5)
ax2.plot(index_SH.index[-1000:],index_SH[-1000:]*b/a,color='purple',linewidth=1.8,label='上证综指相对值',zorder=5)
ax2.set_ylabel('50ETF期权日收盘价')
#ax2.set_ylim(0,7000)
ax2.legend(loc='upper left')
ax2.set_xlabel('时间')
plt.savefig('50ETF_PCR.jpg',dpi=1000)


################################################################################
'''4. 利率'''
################################################################################

shibor = pd.read_excel('一年期SHIBOR.xlsx')
frgc007 = pd.read_excel('FRGC007.xlsx')
nation_interest = pd.read_excel('十年国债收益率.xls')

shibor = shibor.set_index('日期')
shibor.index = pd.DatetimeIndex(shibor.index)

frgc007 = frgc007.set_index('日期')
frgc007.index = pd.DatetimeIndex(frgc007.index)

nation_interest = nation_interest.set_index('频率')
nation_interest.index = pd.DatetimeIndex(nation_interest.index)



c = index_SH[-1]
d = index_SZ[-1]
fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(111)
ax1.grid(False)
ax1.plot(nation_interest.index, nation_interest,linewidth=1.4,color='purple',label='十年国债收益率',zorder=5)
ax1.plot(shibor.index, shibor['价格'],linewidth=1.6,color='orange',label='一年期SHIBOR收益率',zorder=5)
ax1.plot(frgc007.index, frgc007['价格'],linewidth=1.0,color='yellowgreen',label='上交所回购定盘利率(7天)',zorder=5)
ax1.set_ylabel('利率')
#ax1.set_ylim(-1000,1000)
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.grid(True)
ax2.plot(index_SH.index[-4040:],index_SH[-4040:],color='red',linewidth=0.8,label='上证综指',zorder=5)
ax2.plot(index_SZ.index[-4040:],index_SZ[-4040:]*c/d,color='blue',linewidth=0.8,label='深证成指',zorder=6)
ax2.set_ylabel('指数')
#ax2.set_ylim(0,7000)
ax2.legend(loc='upper left')
ax2.set_xlabel('时间')
plt.savefig('利率.jpg',dpi=1000)




