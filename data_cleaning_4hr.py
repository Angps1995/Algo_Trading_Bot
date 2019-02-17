'''
  Replace missing value in 4Hr Dataset
'''

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

full_4h = pd.read_excel(BASE_DIR + '/Orig_Data/SPY 4hr data.xlsx').drop('Time Interval',axis=1)


#Replace missing Volume Column
select = pd.to_numeric(full_4h['Volume'],errors='coerce').isna()
full_4h['Volume'] = full_4h['Volume'].replace('N.A.',np.nan)
values = full_4h['Volume'].diff().values
values = values[~np.isnan(values)]
mean, sd = float(np.mean(values)), float(np.std(values))

#replace by adding a difference value which is randomly sampled from a normal distribution 
#based on the mean and SD of the difference
np.random.seed(123) 
for idx,row in full_4h[select].iterrows():
  rand_diff = np.random.normal(loc=mean, scale=sd)
  new = rand_diff + full_4h.loc[idx-1,'Volume']
  if new > 0:
    full_4h.loc[idx,'Volume'] = new
  else:
    full_4h.loc[idx,'Volume'] = full_4h.loc[idx-1,'Volume']


full_4h.to_csv('cleaned_4hr.csv',encoding='utf-8')