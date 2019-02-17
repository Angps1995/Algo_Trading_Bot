'''
    Deep Learning Model to predict stock price of next period (Used to train DAILY Dataset)
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
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.stats import truncnorm
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

## MODEL ##
def MODEL(hist,var=19):
    INPUT_HEIGHT = hist
    INPUT_VARIABLES = var
    input_x = Input(shape=(INPUT_HEIGHT,INPUT_VARIABLES))
    conv_1 = Conv1D(filters=256,kernel_size=8,strides=1,padding="SAME",activation='relu')(input_x)
    conv_2 = Conv1D(filters=128,kernel_size=4,strides=1,padding="SAME",activation='relu')(conv_1)
    conv_3 = Conv1D(filters=64,kernel_size=3,strides=1,padding="SAME",activation='relu')(conv_2)
    conv_4 = Conv1D(filters=32,kernel_size=2,strides=1,padding="SAME",activation='relu')(conv_3)
    conv_5 = Conv1D(filters=16,kernel_size=2,strides=1,padding="SAME",activation='relu')(conv_4)
    flat = Flatten()(conv_4)
    fc_1 = Dense(128,name='fc1')(flat)
    fc_1_nor = BatchNormalization()(fc_1)
    fc_2 = Dense(64,name='fc2')(fc_1_nor)
    fc_2_nor = BatchNormalization()(fc_2)
    fc_3 = Dense(32,name='fc3')(fc_2_nor)
    fc_3_nor = BatchNormalization()(fc_3)
    fc_4 = Dense(32,name='fc4')(fc_3_nor)
    fc_4_nor = BatchNormalization()(fc_4)
    fc_5 = Dense(16,name='fc5')(fc_4_nor)
    fc_5_nor = BatchNormalization()(fc_5)
    Output = Dense(1,name='output')(fc_5_nor)
    model = Model(inputs=input_x,outputs=Output)
    return model 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Cleaned_Data/cleaned_daily.csv', help='Full Dataset to Train on')
    parser.add_argument('--split', default=3200, help='Row number to split Trg/Test')
    parser.add_argument('--pastperiods', default=10, help='Past periods to use for each time period')
    parser.add_argument('--w', default='weights.hdf5', help='Name of file to save the trained weights' )
    args = parser.parse_args()

    INPUT_HEIGHT = args.pastperiods
    INPUT_VARIABLES = 19
    DATASET = pd.read_csv(os.path.join(ROOT_DIR,args.dataset))
    TOTAL_PERIODS = len(DATASET)
    x = np.expand_dims(DATASET.iloc[:INPUT_HEIGHT,:].values,0)
    for i in tqdm(range(1,TOTAL_PERIODS-INPUT_HEIGHT)):
        x = np.vstack((x,np.expand_dims(DATASET[i:i+INPUT_HEIGHT].values,0)))
    y = DATASET['Last Price'].values.reshape(-1,1)[INPUT_HEIGHT:]

    x_train = x[:args.split]
    x_test = x[args.split:]
    y_train = y[:args.split]
    y_test = y[args.split:] 



    model = MODEL(hist = args.pastperiods)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=[metrics.mse])

    learning_rate_reduction = ReduceLROnPlateau(monitor='mean_squared_error', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.1, 
                                                min_lr=0.0001)

    # fit model
    batch_size = 64
    start = time.time()
    checkpoint = ModelCheckpoint(os.path.join(ROOT_DIR,args.w), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    hist = model.fit(x_train, y_train, 
                    epochs=400, verbose=1,
                    batch_size = batch_size,
                    validation_split=0.2,
                    shuffle=True,
                    callbacks=[learning_rate_reduction,checkpoint])
    end = time.time()
    print("TIME TOOK {:3.2f}MIN".format((end - start )/60))