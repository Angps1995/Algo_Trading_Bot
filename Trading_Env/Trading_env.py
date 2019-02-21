'''
    Trading Environment
'''
import numpy as np

class Trading_Environment:
	def __init__(self,data,history_len=90, starting_cash = 10000000, buying_perc = 0.8, trans_cost=0.02):
		self.data = data
		self.history_len = history_len #length of transaction to keep record
		self.t = 0                     #time
		self.profits = 0               #total profit
		self.positions = np.array([])  #price at buying of stock 
		self.position_value = 0        #price difference between current time and all holding price
		self.price_diff_perc = 0
		self.history = np.zeros(history_len) #this is the same as the net change column
		self.cash = starting_cash
		self.buying_perc = buying_perc
		self.stock_qty = 0
		self.trans_cost = trans_cost
	def reset(self):
		self.t = 0
		self.profits = 0 
		self.positions = np.array([])
		self.position_value = 0
		self.cash = 10000000
		self.history = np.zeros(self.history_len)
		return np.concatenate(([self.position_value],self.history))
	
	def step(self,act):
		# Possible actions
		# 0 : stay
		# 1 : buy
		# 2 : sell
		reward = 0
		profits = 0
		stock_price = self.data.iloc[self.t,0]
		#if buy, hold stock at price
		if act==1:
			if len(self.positions) != 0:
				reward = -1   #dont want model to buy when it already has stock
			else:
				buy_cap = (self.buying_perc-0.2)*self.cash
				self.stock_qty = buy_cap//stock_price
				self.positions = np.concatenate((self.positions,[self.stock_qty*stock_price]))
				self.cash -= self.stock_qty*stock_price*(1+self.trans_cost)
		elif act==11:
			if len(self.positions) != 0:
				reward = -1   #dont want model to buy when it already has stock
			else:
				buy_cap = self.buying_perc*self.cash
				self.stock_qty = buy_cap//stock_price
				self.positions = np.concatenate((self.positions,[self.stock_qty*stock_price]))
				self.cash -= self.stock_qty*stock_price*(1+self.trans_cost)
		elif act==111:
			if len(self.positions) != 0:
				reward = -1   #dont want model to buy when it already has stock
			else:
				buy_cap = (self.buying_perc+0.1)*self.cash
				self.stock_qty = buy_cap//stock_price
				self.positions = np.concatenate((self.positions,[self.stock_qty*stock_price]))
				self.cash -= self.stock_qty*stock_price*(1+self.trans_cost)
		#if sell, sell all stocks and sum the profit. reward is positive if profit > 0 else negative
		elif act==2:
			if len(self.positions)==0:
				reward = -1  #dont want model to sell when it has no stock
			else:
				
				profits = sum(self.stock_qty*stock_price - self.positions)
				self.profits += profits
				reward = np.sign(profits)
				self.positions = np.array([])
				self.stock_qty = 0
				self.cash += self.stock_qty*stock_price*(1-self.trans_cost)
		# Move to next time
		self.t += 1
		self.position_value = sum(self.stock_qty*stock_price - self.positions)
		self.price_diff_perc = (self.stock_qty*stock_price - self.positions)/self.positions
		self.history = np.concatenate((self.history[1:],[self.data.iloc[self.t,0]-self.data.iloc[self.t-1,0]]))
		
		return np.concatenate(([self.position_value],self.history)), reward, profits