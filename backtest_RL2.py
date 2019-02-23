import numpy as np
import math
import pandas as pd
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import argparse
import numpy as np
import random
from collections import deque
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(os.path.join(BASE_DIR,'RL_BOT'))
from Agent import Agent

###Functions used###

# prints formatted price
def formatPrice(n):
    return (("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))),abs(n)

# returns the vector containing stock data from a fixed file
def getStockDataVec(df):
	vec = df.iloc[:,0].tolist()

	return vec

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])

window_size = 10
model_name = '/RL_BOTRL_Model_ep36'
agent = Agent(state_size=window_size, is_eval=True, model_name=model_name)
steps = []
acts = []
price = []
units = []
curr_cash = []
data = pd.read_csv(os.path.join(BASE_DIR,'Cleaned_Data/cleaned_daily.csv'))
data = getStockDataVec(data)
l = len(data) 
state = getState(data, 0, window_size + 1)
total_profit = 0
for t in range(l-1):
    if t < 10:
        action = 0
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        qty = 0
        print('Hold')
    else:
        action = agent.act(state)
        next_state = getState(data, t + 1, window_size + 1)
        past_10_mean = np.mean(data[t-window_size:t])
        past_10_std = np.std(data[t-window_size:t])
        qty = max((agent.buy_perc*agent.cash)//data[t],0)
        
        if qty == 0 and action == 1:
            options = agent.model.predict(state)[0]
            action = np.argmax((options[:1]+options[2:]))
            if action >0:
                action = action + 1 
            else:
                action = 0
        if len(agent.inventory_qty) > 0:
            if action == 0:
                reward = 0
                qty = 0
                print('Hold')
            elif action == 1 and qty > 0: # buy
                agent.inventory_value.append(data[t])
                agent.inventory_qty.append(qty)
                agent.cash -= qty*data[t]*(1+agent.trans_cost)
                reward = 0
                print("Buy " + str(qty) + 'units for ' + formatPrice(data[t])[0] + ' unit price, total cost: ' + str(qty*formatPrice(data[t])[1]*(1+agent.trans_cost)))

            elif action == 2: # sell one (earliest first)
                bought_price = agent.inventory_value.pop(0)
                bought_qty = agent.inventory_qty.pop(0)
                reward = np.sign(data[t] - bought_price)
                agent.cash += bought_qty*data[t]*(1-agent.trans_cost)
                total_profit += (data[t] - bought_price)*bought_qty*(1-agent.trans_cost)
                print("Sell " + str(bought_qty) + "units at : " + formatPrice(data[t])[0] + " | Profit: " + formatPrice((data[t] - bought_price)*bought_qty*(1-agent.trans_cost))[0] + " | Total Profit: " + formatPrice(total_profit)[0])
                    
            elif action == 3: # sell one (latest first)
                bought_price = agent.inventory_value.pop(-1)
                bought_qty = agent.inventory_qty.pop(-1)
                reward = np.sign(data[t] - bought_price)
                agent.cash += bought_qty*data[t]*(1-agent.trans_cost)
                total_profit += (data[t] - bought_price)*bought_qty*(1-agent.trans_cost)
                print("Sell " + str(bought_qty) + "units at : " + formatPrice(data[t])[0] + " | Profit: " + formatPrice((data[t] - bought_price)*bought_qty*(1-agent.trans_cost))[0] + " | Total Profit: " + formatPrice(total_profit)[0])
                    
            elif action == 4: # sell all
                bought_qty = sum(agent.inventory_qty)
                agent.cash += bought_qty*data[t]*(1-agent.trans_cost)
                total_profit += ((bought_qty*data[t]) - 
                                (np.dot(agent.inventory_value,agent.inventory_qty)))*(1-agent.trans_cost)
                reward = np.sign(((bought_qty*data[t]) - 
                                (np.dot(agent.inventory_value,agent.inventory_qty)))*(1-agent.trans_cost))
                agent.inventory_value = []
                agent.inventory_qty = []
                print("Sell all units at : " + formatPrice(data[t])[0] + " | Profit: " + formatPrice(((bought_qty*data[t]) - 
                                    (np.dot(agent.inventory_value,agent.inventory_qty)))*(1-agent.trans_cost))[0] + " | Total Profit: " + formatPrice(total_profit)[0])
        elif len(agent.inventory_qty) == 0:
            pred_value = agent.model.predict(state)
            action = np.argmax(pred_value[0][:2])
            if qty == 0 or action==1:
                action = 0
                reward = 0
                print("Hold")
            elif action == 1 and qty > 0: # buy
                agent.inventory_value.append(data[t])
                agent.inventory_qty.append(qty)
                agent.cash -= qty*data[t]*(1+agent.trans_cost)
                reward = 0
                print("Buy " + str(qty) + 'units for ' + formatPrice(data[t])[0] + ' unit price, total cost: ' + str(qty*formatPrice(data[t])[1]*(1+agent.trans_cost)))
            else:
                action = 0
                reward = 0
                qty = 0
                print('Hold')
        if action == 0 and qty > 0  and data[t] < (past_10_mean - 1.1*past_10_std):
            action = 1
            agent.inventory_value.append(data[t])
            agent.inventory_qty.append(qty)
            agent.cash -= qty*data[t]*(1+agent.trans_cost)
            reward = 0
            print("Buy " + str(qty) + 'units for ' + formatPrice(data[t])[0] + ' unit price, total cost: ' + str(qty*formatPrice(data[t])[1]*(1+agent.trans_cost)))                
    agent.unr_profit = (np.dot(agent.inventory_value,agent.inventory_qty) - (sum(agent.inventory_qty)*data[t]))*(1-agent.trans_cost)
    done = True if t == l-2 else False
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state
    steps.append(t)
    acts.append(action)
    price.append(data[t])
    curr_cash.append(agent.cash)
    if action == 0:
        units.append(0) 
    elif action == 1:
        units.append(qty)
    else:
        units.append(bought_qty)
    if done:
        print("--------------------------------")
        print("Total Profit for episode: " + formatPrice(agent.cash - 10000000)[0])
        print("--------------------------------")
dic={}
dic["time period"] = steps
dic["Actions"] = acts
dic["Prices"] = price
dic["Units"] = units

RL_DF = pd.DataFrame(dic)
RL_DF=RL_DF[["time period","Actions","Prices","Units"]]
#RL_DF.to_csv(BASE_DIR + '/RL_DF.csv')
print(RL_DF["Actions"].value_counts())