'''
Author: Ang Peng Seng, Leon Tham, Ryan Heng
Date: 25/2/19

Script to use the bot to auto-trade.

1) Put the csv file containing the 'last prices' of the SPY stock in the same directory as this file. 
    -- Put the 'Last Price' of the stock on the FIRST COLUMN in the CSV file

2) Make sure the RL_BOTRL_Model_v2_ep17 model weight file is also in the same directory

3) Run this script python3 auto_trade.py --dataset <INSERT DATA FILE NAME>  

4) It will output a trade.csv file in which the action taken in each period is shown.
    --0: Hold
    --1: Buy stock quantity equivalent to 99% of the CURRENT cash on hand at that time
    --2: Sell the stock quantity that was bought earliest first 
    --3: Sell the stock quantity that was bought latest first
    --4: Sell ALL stocks
'''


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

##AGENT##
class Agent:
    def __init__(self, state_size, cash = 10000000, buy_perc = 0.99, trans_cost = 0.0002, is_eval=False, model_name=""):
        self.state_size = state_size # normalized previous days
        self.action_size = 5 # sit, buy, sell one from first, sell one from latest,sell all
        self.memory = deque(maxlen=1000)
        self.inventory_value = []
        self.inventory_qty = []
        self.unr_profit = 0
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.cash = cash
        self.buy_perc = buy_perc
        self.trans_cost = trans_cost
        self.model = load_model(BASE_DIR + model_name) if is_eval else self._model()

    def _model(self):
        model = Sequential()
        model.add(Dense(units=32, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=16, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        options = self.model.predict(state)
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=2, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 
            
    def reset(self):
        self.action_size = 5 #sit, buy, sell one from first, sell one from latest,sell all
        self.inventory_value = []
        self.inventory_qty = []
        self.unr_profit = 0
        self.cash = 10000000

def trade(fl,model_name):
    window_size = 10
    
    agent = Agent(state_size=window_size, is_eval=True, model_name=model_name)
    steps = []
    acts = []
    price = []
    units = []
    curr_cash = []
    
    data = getStockDataVec(fl)
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
    RL_DF.to_csv(BASE_DIR + '/trades.csv')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='INSERT DATASET FILE NAME TO TEST ON')
    parser.add_argument('--model', default='/RL_BOTRL_Model_v2_ep17', help='Model weights to be used')

    args = parser.parse_args()
    data = pd.read_csv(os.path.join(BASE_DIR,args.dataset))
    trade(fl=data,model_name=args.model)