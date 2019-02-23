import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
sys.path.append(BASE_DIR)
class Agent:
    def __init__(self, state_size, cash = 10000000, buy_perc = 0.8, trans_cost = 0.0002, is_eval=False, model_name=""):
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
        self.model = load_model(ROOT_DIR + model_name) if is_eval else self._model()

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