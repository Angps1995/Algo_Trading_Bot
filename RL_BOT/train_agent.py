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
sys.path.append(BASE_DIR)

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

#######  TRAINING  ########

def train_agent(data,window_size,episode_count, batch_size):

    agent = Agent(state_size=window_size)
    data = getStockDataVec(data)
    l = len(data) - 1
    batch_size = batch_size

    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))
        state = getState(data, 0, window_size + 1)
        agent.reset()
        total_profit = 0

        for t in range(l):
            action = agent.act(state)

            # sit
            next_state = getState(data, t + 1, window_size + 1)
            reward = 0
            qty = max((agent.buy_perc*agent.cash)//formatPrice(data[t])[1],0)
            if (qty == 0 and action ==1) or action == 0:
                action = 0
                print("Hold")
            elif action == 1: # buy
                qty = max((agent.buy_perc*agent.cash)//formatPrice(data[t])[1],0)
                agent.inventory_value.append(data[t])
                agent.inventory_qty.append(qty)
                agent.cash -= qty*data[t]*(1+agent.trans_cost)
                reward = np.sign(agent.unr_profit//(sum(agent.inventory_qty)))
                print("Buy " + str(qty) + 'units for ' + formatPrice(data[t])[0] + ' unit price, total cost: ' + str(qty*formatPrice(data[t])[1]))

            elif action == 2 and len(agent.inventory_qty) > 0: # sell one (earliest first)
                bought_price = agent.inventory_value.pop(0)
                bought_qty = agent.inventory_qty.pop(0)
                reward = np.sign(data[t] - bought_price)
                agent.cash += bought_qty*data[t]*(1-agent.trans_cost)
                total_profit += (data[t] - bought_price)*bought_qty
                print("Sell " + str(bought_qty) + "units at : " + formatPrice(data[t])[0] + " | Profit: " + formatPrice((data[t] - bought_price)*bought_qty)[0] + " | Total Profit: " + formatPrice(total_profit)[0])
            
            elif action == 3 and len(agent.inventory_qty) > 0: # sell one (latest first)
                bought_price = agent.inventory_value.pop(-1)
                bought_qty = agent.inventory_qty.pop(-1)
                reward = np.sign(data[t] - bought_price)
                agent.cash += bought_qty*data[t]*(1-agent.trans_cost)
                total_profit += (data[t] - bought_price)*bought_qty
                print("Sell " + str(bought_qty) + "units at : " + formatPrice(data[t])[0] + " | Profit: " + formatPrice((data[t] - bought_price)*bought_qty)[0] + " | Total Profit: " + formatPrice(total_profit)[0])
            
            elif action == 4 and len(agent.inventory_qty) > 0: # sell all
                bought_qty = sum(agent.inventory_qty)
                agent.inventory_value = []
                agent.inventory_qty = []
                reward = np.sign(data[t] - bought_price)
                agent.cash += bought_qty*data[t]*(1-agent.trans_cost)
                total_profit += (data[t] - bought_price)*bought_qty*(1-agent.trans_cost)
                print("Sell all units at : " + formatPrice(data[t])[0] + " | Profit: " + formatPrice((data[t] - bought_price)*bought_qty)[0] + " | Total Profit: " + formatPrice(total_profit)[0])
            agent.unr_profit = np.dot(agent.inventory_value,agent.inventory_qty) - (sum(agent.inventory_qty)*data[t])
            done = True if t == l - 1 else False
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("--------------------------------")
                print("Total Profit for episode: " + formatPrice(total_profit + agent.unr_profit)[0])
                print("--------------------------------")
                
        
            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)

        if e % 3 == 0:
            agent.model.save(ROOT_DIR + "RL_Model_ep" + str(e))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Cleaned_Data/cleaned_4hr.csv', help='Full Dataset to Train on')
    parser.add_argument('--split', default=4096, help='Row number to split Trg/Test')
    parser.add_argument('--w', default=10, help='Past periods to take into account for state')
    parser.add_argument('--ep', default=320, help='Number of episodes to train the agent')
    parser.add_argument('--b', default=32, help='Number of batches')
    args = parser.parse_args()
    df = pd.read_csv(os.path.join(ROOT_DIR,args.dataset))
    train = df[:args.split]
    data, window_size, episode_count, batch = train, args.w, args.ep, args.b

    train_agent(data,window_size,episode_count, batch)