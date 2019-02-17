'''
  Replace missing value in Daily Dataset
'''
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
daily = pd.read_excel(BASE_DIR + '/Orig_Data/daily.xlsx','DAILY').drop('Date',axis=1)

#Replace Ask and Bid price missing values
select1 = pd.to_numeric(daily['Ask Price'],errors='coerce').isna()
select2 = pd.to_numeric(daily['Bid Price'],errors='coerce').isna()
daily['Ask Price'] = daily['Ask Price'].replace('N.A.',np.nan)
askprice = daily['Ask Price'].diff().values
askprice = askprice[~np.isnan(askprice)]
daily['Bid Price'] = daily['Bid Price'].replace('N.A.',np.nan)
bidprice = daily['Bid Price'].diff().values
bidprice = bidprice[~np.isnan(bidprice)]
mean1, sd1 = float(np.mean(askprice)), float(np.std(askprice))
mean2, sd2 = float(np.mean(bidprice)), float(np.std(bidprice))

#replace by adding a difference value which is randomly sampled from a normal distribution 
#based on the mean and SD of the difference
for idx,row in daily[select1].iterrows():
  rand_diff = np.random.normal(loc=mean1, scale=sd1)
  new = rand_diff + daily.loc[idx-1,'Ask Price']
  if new > 0:
    daily.loc[idx,'Ask Price'] = new
  else:
    daily.loc[idx,'Ask Price'] = daily.loc[idx-1,'Ask Price']
for idx,row in daily[select2].iterrows():
  rand_diff = np.random.normal(loc=mean2, scale=sd2)
  new = rand_diff + daily.loc[idx-1,'Bid Price']
  if new > 0:
    daily.loc[idx,'Bid Price'] = new
  else:
    daily.loc[idx,'Bid Price'] = daily.loc[idx-1,'Bid Price']

daily=daily.iloc[::-1]
daily=daily.reset_index(drop=True)
daily.to_csv('cleaned_daily.csv',encoding='utf-8')