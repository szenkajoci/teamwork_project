#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 09:50:50 2022

@author: szenkajozsef

@source: https://machinelearningknowledge.ai/keras-lstm-layer-explained-for-beginners-with-example/
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

heightCase = ['01_flat','02_low','03_middle','04_high']

profileCase = ['neutral','stable']

coriolisCase = ['corilois','non_coriolis']
coriolisDict = {
    "corilois" : True,
    "non_coriolis" : False
    }

heights = [0, 10, 20, 40]

points = {
    1 : '(point-04-000625)',
    2 : '(point-04-080000)',
    3 : '(point-14-080000)'
}

measuredData = pd.DataFrame()
calculatedData = pd.DataFrame()
calculatedData2 = pd.DataFrame()

for height in heightCase:
    
    for coriolis in coriolisCase:
        
        for profile in profileCase:
        
            sourcePath = 'data/' + height + '/' + coriolis + '/' + profile 
            files = pd.Series(os.listdir(sourcePath))
            files = files[files.str.endswith('.out')]
            files = files.reset_index(drop=True)
            
            
            if files.size>0:
                print(files[0])
                source = sourcePath + "/" + files[0]
                # df = pd.read_csv(source, sep=' ', skiprows = 3, header=None, names = ['TimeStep','vx1','vx2','vx3','vy1','vy2','vy3','vz1','vz2','vz3','flow-time'])

                try:
                    df = pd.read_csv(source, sep=' ', skiprows = 3, header=None, names = ['TimeStep','vx1','vx2','vx3','vy1','vy2','vy3','vz1','vz2','vz3','flow-time'])
                except:
                    print(source + ' cannot be found!\n')
                    break
                
                df.insert(0,'height',height)
                df.insert(1,'coriolis',coriolisDict.get(coriolis))
                df.insert(2,'profile',profile)
                
                df['u1'] = np.sqrt(np.power(df['vx1'], 2)+np.power(df['vy1'], 2)+np.power(df['vz1'], 2))
                df['u2'] = np.sqrt(np.power(df['vx2'], 2)+np.power(df['vy2'], 2)+np.power(df['vz2'], 2))
                df['u3'] = np.sqrt(np.power(df['vx3'], 2)+np.power(df['vy3'], 2)+np.power(df['vz3'], 2))
                df['part'] = [1 if i<len(df)/2 else 2 for i in range(len(df))]
                
                measuredData = pd.concat([measuredData,df])
            
            else:
                print('In ' + sourcePath + ' no data file can be found!\n')
                
training_set = measuredData[(measuredData['height']=='01_flat') & (measuredData['coriolis']==True) & (measuredData['profile']=='stable')][['u1']].values

#preprocessing

from sklearn.preprocessing import MinMaxScaler

train_size, val_size = 0.5, 0

num_time_steps = training_set.shape[0]
num_train, num_val = (
    int(num_time_steps * train_size),
    int(num_time_steps * val_size),
)

train_array_true = training_set[:num_train]
val_array_true = training_set[num_train : (num_train + num_val)]
test_array_true = training_set[(num_train + num_val) :]

sc = MinMaxScaler(feature_range = (-1, 1))
training_set_scaled = sc.fit_transform(training_set)

train_array = training_set_scaled[:num_train]
val_array = training_set_scaled[num_train : (num_train + num_val)]
test_array = training_set_scaled[(num_train + num_val) :]

X_train = []
y_train = []
for i in range(60, train_array.shape[0]):
    X_train.append(train_array[i-60:i, 0])
    y_train.append(train_array[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.25))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.25))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 2, batch_size = 32)

regressor.summary()

#test

dataset_total = np.concatenate((train_array, test_array), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test_array) - 60:]
X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


predicted = regressor.predict(X_test)
predicted_transformed = sc.inverse_transform(predicted)

plt.figure(figsize=(14,11))
plt.plot(test_array_true, color = 'black', label = 'Measured')
plt.plot(predicted_transformed, color = 'green', label = 'Predicted')
plt.title('Time series Prediction')
plt.xlabel('Time')
plt.ylabel(r'$u_1$')
plt.legend()
plt.show()


predicted2 = regressor.predict(X_train)
predicted_transformed2 = sc.inverse_transform(predicted2)

plt.plot(train_array_true, color = 'black', label = 'Measured')
plt.plot(predicted_transformed2, color = 'green', label = 'Predicted')
plt.title('Time series Prediction')
plt.xlabel('Time')
plt.ylabel(r'$u_1$')
plt.legend()
plt.show()


n = 1000
inputs = dataset_total[len(dataset_total) - 60:]
for j in range(n):
    X_test = []
    for i in range(60, inputs.shape[0]+1):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted = regressor.predict(X_test[-1])
    
    inputs = np.concatenate((inputs, predicted[0].reshape(1,1)), axis = 0)
    
    
inputs_transformed = sc.inverse_transform(inputs)