'''
    Script for Backtesting using RL Agent on 4hr Dataset
'''

import os
import sys
import time
import argparse
import copy
import numpy as np
import pandas as pd
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers

#Helper
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR,'RL_DQN'))
sys.path.append(os.path.join(BASE_DIR,'Trading_Env'))

from Trading_env import Trading_Environment
from train_dqn import Q_Network

def backtest(dataset,weights):
    env = Trading_Environment(dataset)
    Q = Q_Network(input_size=env.history_len+1,  output_size=3)  
    serializers.load_npz(weights, Q)
    Q_ast = copy.deepcopy(Q)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(Q)
    pobs = env.reset()
    step = 0
    step_max = len(dataset)-1 
    total_profit = 0
    steps = []
    acts = []
    while step < step_max:
        if env.price_diff_perc <= -0.3 or env.price_diff_perc > 0.2:
            pact = 2
        else:
            if step==0:
                pact = 1
            else:
                pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))
                pact = np.argmax(pact.data)
                if pact == 1 and len(env.positions) != 0:
                    pact = 0
                elif pact == 2 and len(env.positions) == 0:
                    pact = 0

        obs,reward,profit = env.step(pact)
        total_profit += profit
        pobs = obs
        steps.append(step)
        acts.append(pact)
        step += 1
        
        if (step+1) % 100:
            print("period " + str(step) + ' cumulated profit: ' + str(total_profit))
    print(total_profit)
    dic={}
    dic["time period"] = steps
    dic["Actions"] = acts
    RL_DF = pd.DataFrame(dic)
    RL_DF=RL_DF[["time period","Actions"]]
    RL_DF.to_csv(args.fn)
    print("Backtest CSV results saved")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Cleaned_Data/cleaned_4hr.csv', help='Full Dataset to Train on')
    parser.add_argument('--fn', default='RL_Backtest.csv', help='Name of file to save the backtest results')
    parser.add_argument('--w', default='Saved_weights/Qmodel.npz', help='Name of weights file to load' )
    args = parser.parse_args()
    DATASET = pd.read_csv(os.path.join(BASE_DIR,args.dataset))
    
    backtest(DATASET,os.path.join(BASE_DIR,args.w))