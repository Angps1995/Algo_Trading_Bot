import pandas as pd
import numpy as np
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(os.path.join(ROOT_DIR, 'Trading_Env'))

from Trading_env import Trading_Environment

class Q_Network(chainer.Chain):

    def __init__(self, input_size, output_size):  #input size is the history length, output size is the no. of actions
        super(Q_Network, self).__init__(
            fc1 = L.Linear(input_size, 128),
            fc2 = L.Linear(128, 64),
            fc3 = L.Linear(64,32),
            fc4 = L.Linear(32, output_size)   #output will be the respective values of each action during that time period
        )

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        y = self.fc4(h)
        return y

    def reset(self):
        self.zerograds()

def train_dqn(env):
    Q = Q_Network(input_size=env.history_len+1,  output_size=3)  
    Q_ast = copy.deepcopy(Q)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(Q)
    #training details
    epoch_num = 60
    step_max = len(env.data)-1  #max no. of steps the agent will run through
    memory_size = 320
    batch_size = 32
    epsilon = 1.0
    epsilon_decrease = 1e-3
    epsilon_min = 0.1
    start_reduce_epsilon = 200
    train_freq = 10
    update_q_freq = 20
    gamma = 0.98
    show_log_freq = 5
    
    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []
    total_profits = []
    
    start = time.time()
    
    for epoch in range(epoch_num):

        pobs = env.reset()  #reset obs every epoch
        step = 0
        #done = False
        total_reward = 0
        total_loss = 0
        total_profit = 0
        
        while step < step_max:
          # select act
            if env.price_diff_perc <= -0.3 or env.price_diff_perc > 0.2:
                pact = 2
            else:
                pact = np.random.randint(3)   #get random action
                if np.random.rand() > epsilon:   #get best action based on policy
                    pact = Q(np.array(pobs, dtype=np.float32).reshape(1, -1))
                    pact = np.argmax(pact.data)

            # get info after applying action
            obs, reward, profit = env.step(pact)

            # add memory
            memory.append((pobs, pact, reward, profit, obs))  #add into memory: past obs, past action, reward,profit, curr obs
            if len(memory) > memory_size:
                memory.pop(0)  #remove oldest memory if memory size exceeded

            # train or update q
            if len(memory) == memory_size:
              if total_step % train_freq == 0:
                    shuffled_memory = np.random.permutation(memory)
                    batches = np.vsplit(np.array(shuffled_memory),memory_size//batch_size)
                    for batch in batches:
                      
                      b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                      b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                      b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                      b_profit = np.array(batch[:, 3].tolist(), dtype=np.int32)
                      b_obs = np.array(batch[:, 4].tolist(), dtype=np.float32).reshape(batch_size, -1)
                      
                      q = Q(b_pobs)  
                      maxq = np.max(Q_ast(b_obs).data, axis=1)
                      target = copy.deepcopy(q.data)
                      for j in range(batch_size):
                          target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]   #update the target policy
                      Q.reset()
                      loss = F.mean_squared_error(q, target)  #get mse
                      total_loss += loss.data     #add loss
                      loss.backward()
                      optimizer.update()
                      if total_step % update_q_freq == 0:
                        Q_ast = copy.deepcopy(Q)

            # decrease epsilon after certain no. of steps
            if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                epsilon -= epsilon_decrease

            # next step
            total_reward += reward
            total_profit += profit
            pobs = obs
            step += 1
            total_step += 1

        total_rewards.append(total_reward)
        total_losses.append(total_loss)
        total_profits.append(total_profit)
        if (epoch+1) % show_log_freq == 0:
            log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
            log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
            ave_profit = sum(total_profits[((epoch+1)-show_log_freq):])/show_log_freq
            elapsed_time = time.time()-start
            print('\t'.join(map(str, [epoch+1, epsilon, total_step, log_reward, log_loss, ave_profit, elapsed_time])))
            start = time.time()
            
    return Q, total_losses, total_rewards, total_profits
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Cleaned_Data/cleaned_4hr.csv', help='Full Dataset to Train on')
    parser.add_argument('--split', default=4096, help='Row number to split Trg/Test')
    parser.add_argument('--Q', default='Qmodel.npz', help='Name of file to save the trained Q model' )
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(ROOT_DIR,args.dataset))
    train = df[:args.split]
    env = Trading_Environment(train)
    Q, total_losses, total_rewards, total_profits = train_dqn(env)

    #save Q
    serializers.save_npz(args.Q, Q)


