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
                
training_set = measuredData[(measuredData['height']=='01_flat') & (measuredData['coriolis']==True) & (measuredData['profile']=='stable')][['u1']]

training_set = training_set[training_set.index%4==1].values

# test heatmap

plt.pcolormesh(np.cov(training_set[:200],rowvar='true'), cmap = 'summer')
plt.title('Heatmap for the velocity magnitude')
plt.show()


# preprocessing

from sklearn.preprocessing import MinMaxScaler

train_size, val_size = 0.5, 0
windows_size = 50

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
for i in range(windows_size, train_array.shape[0]):
    X_train.append(train_array[i-windows_size:i, 0])
    y_train.append(train_array[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.4))

regressor.add(LSTM(units = 100))
regressor.add(Dropout(0.4))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 5, batch_size = 32)

regressor.summary()

#test

dataset_total = np.concatenate((train_array, test_array), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test_array) - windows_size:]
X_test = []
for i in range(windows_size, inputs.shape[0]):
    X_test.append(inputs[i-windows_size:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


predicted = regressor.predict(X_test)
predicted_transformed = sc.inverse_transform(predicted)

#plt.figure(figsize=(14,11))
plt.plot(test_array_true, color = 'black', label = 'Measured')
plt.plot(predicted_transformed, color = 'green', label = 'Predicted')
plt.title('Time series prediction - test data')
plt.xlabel('Time')
plt.ylabel(r'$u_1$')
plt.legend()
plt.show()
plt.savefig('LSTM_timedata/test_predicted.png', bbox_inches='tight')


predicted2 = regressor.predict(X_train)
predicted_transformed2 = sc.inverse_transform(predicted2)

plt.plot(train_array_true[windows_size:], color = 'black', label = 'Measured')
plt.plot(predicted_transformed2, color = 'green', label = 'Predicted')
plt.title('Time series prediction - train data')
plt.xlabel('Time')
plt.ylabel(r'$u_1$')
plt.legend()
plt.show()
plt.savefig('LSTM_timedata/train_predicted.png', bbox_inches='tight')


n = 1000
inputs = dataset_total[len(dataset_total) - windows_size:]
for j in range(n):
    X_test = []
    for i in range(windows_size, inputs.shape[0]+1):
        X_test.append(inputs[i-windows_size:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted = regressor.predict(X_test[-1].reshape(1,windows_size,1)) #dimension?
    
    inputs = np.concatenate((inputs, predicted), axis = 0)
    
pd.DataFrame(X_test.reshape(X_test.shape[0],X_test.shape[1])).to_csv('LSTM_timedata/predictionHU.csv',sep=';',decimal=',')
pd.DataFrame(X_test.reshape(X_test.shape[0],X_test.shape[1])).to_csv('LSTM_timedata/prediction.csv',sep=',',decimal='.')

inputs_transformed = sc.inverse_transform(inputs)

plt.plot(inputs_transformed, color = 'green')
plt.title('Time series prediction - recursive')
plt.xlabel('Time')
plt.ylabel(r'$u_1$')
plt.show()
plt.savefig('LSTM_timedata/recursive_prediction.png', bbox_inches='tight')