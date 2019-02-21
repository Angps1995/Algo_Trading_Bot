'''
    Script for Backtesting using DL model on Daily Dataset
'''

import os
import sys
import argparse
#Keras Deep Learning
from keras.preprocessing import sequence
import keras.backend as K
from keras.layers import Lambda
from keras.models import Sequential, Model
from keras.layers import LSTM, BatchNormalization, Dropout, Dense, Activation, Flatten, Concatenate, Conv1D, Input, Conv2D, Permute,CuDNNGRU, concatenate, multiply
from keras.optimizers import Adam, RMSprop
from keras import metrics, regularizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

#Helper
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR,'DL_Model'))
sys.path.append(os.path.join(BASE_DIR,'Trading_Env'))
sys.path.append(os.path.join(BASE_DIR,'Saved_weights'))
from Trading_env import Trading_Environment
import train_model

def backtest(dataset,weights):
    env = Trading_Environment(dataset)
    model = train_model.MODEL(hist=10)
    model.load_weights(weights)
    max_days_to_test = len(dataset) -1
    INPUT_HEIGHT = 10 
    total_profit = 0
    steps = []
    acts = []
    period = 0  
    while period < max_days_to_test:
        if period < 10:   #start at period 10 (row start from 0) to use previous 10 values
            pact = 0
    else:
        if env.price_diff_perc <= -0.3 or env.price_diff_perc > 0.3:
            pact = 2

        else:
            x = np.expand_dims(env.data.iloc[period-INPUT_HEIGHT:period,:].values,0)
            past_10_mean = np.mean(env.data.iloc[period-INPUT_HEIGHT:period,0])
            past_10_std = np.std(env.data.iloc[period-INPUT_HEIGHT:period,0])
            pred = model.predict(x)[0][0]
            
            #DL Method
            if ((pred - env.data.iloc[period,0])/env.data.iloc[period,0]) >= 0.025:
                pact = 11

            elif ((pred - env.data.iloc[period,0])/env.data.iloc[period,0]) <= -0.04:
                if len(env.positions) == 0:
                    pact = 0
                else:
                    pact = 2

            #SD Method
            else:  
                if env.data.iloc[period,0] < (past_10_mean - 1.1*past_10_std) and len(env.positions)==0:
                    pact = 1
                elif env.data.iloc[period,0] < (past_10_mean - 2*past_10_std) and len(env.positions)==0:
                    pact = 11
                elif env.data.iloc[period,0] < (past_10_mean - 2.5*past_10_std) and len(env.positions)==0:
                    pact = 111
                else:
                    pact = 0
                        
        obs,reward,profit = env.step(pact)
        total_profit += profit
        steps.append(period)
        acts.append(pact)
        period += 1
        if (period+1) % 100:
            print("period " + str(period) + ' cumulated profit: ' + str(total_profit))
    print(total_profit)
    dic={}
    dic["time period"] = steps
    dic["Actions"] = acts
    DL_DF = pd.DataFrame(dic)
    DL_DF=DL_DF[["time period","Actions"]]
    DL_DF.to_csv(args.fn)
    print("Backtest CSV results saved")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Cleaned_Data/cleaned_daily.csv', help='Full Dataset to Train on')
    parser.add_argument('--fn', default='DL_Backtest.csv', help='Name of file to save the backtest results')
    parser.add_argument('--w', default='Saved_weights/weights.hdf5', help='Name of weights file to load' )
    args = parser.parse_args()
    DATASET = pd.read_csv(os.path.join(BASE_DIR,args.dataset))
    
    backtest(DATASET,os.path.join(BASE_DIR,args.w))
