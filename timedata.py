#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mac 29 12:22:58 2022

@author: szenkajozsef
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

for height in heightCase:
    
    for coriolis in coriolisCase:
        
        for profile in profileCase:
        
            sourcePath = 'data/' + height + '/' + coriolis + '/' + profile 
            files = pd.Series(os.listdir(sourcePath))
            files = files[files.str.endswith('.out')]
            files = files.reset_index(drop=True)
            
            print(files[0])
            
            if files.size>0:
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
                
                measuredData = pd.concat([measuredData,df])
            
            else:
                print('In ' + sourcePath + ' no data file can be found!\n')

# u1
plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
    if dftemp1.shape[0]!=0:
        plt.plot(dftemp1['flow-time'], dftemp1['u1'], 'C9-', label = 'Neutral with Coriolis')
    dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
    if dftemp2.shape[0]!=0:
        plt.plot(dftemp2['flow-time'], dftemp2['u1'], 'C9:', label = 'Neutral without Coriolis')
    dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
    if dftemp3.shape[0]!=0:
        plt.plot(dftemp3['flow-time'], dftemp3['u1'], 'C3-', label = 'Stable with Coriolis')
    dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
    if dftemp4.shape[0]!=0:
        plt.plot(dftemp4['flow-time'], dftemp4['u1'], 'C3:', label = 'Stable without Coriolis')
    plt.title('1st point velocity magnitude')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\overline{u}_1$')
    plt.legend()

# u2
plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
    if dftemp1.shape[0]!=0:
        plt.plot(dftemp1['flow-time'], dftemp1['u2'], 'C9-', label = 'Neutral with Coriolis')
    dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
    if dftemp2.shape[0]!=0:
        plt.plot(dftemp2['flow-time'], dftemp2['u2'], 'C9:', label = 'Neutral without Coriolis')
    dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
    if dftemp3.shape[0]!=0:
        plt.plot(dftemp3['flow-time'], dftemp3['u2'], 'C3-', label = 'Stable with Coriolis')
    dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
    if dftemp4.shape[0]!=0:
        plt.plot(dftemp4['flow-time'], dftemp4['u2'], 'C3:', label = 'Stable without Coriolis')
    plt.title('2nd point velocity magnitude')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\overline{u}_2$')
    plt.legend()

# u3
plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
    if dftemp1.shape[0]!=0:
        plt.plot(dftemp1['flow-time'], dftemp1['u3'], 'C9-', label = 'Neutral with Coriolis')
    dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
    if dftemp2.shape[0]!=0:
        plt.plot(dftemp2['flow-time'], dftemp2['u3'], 'C9:', label = 'Neutral without Coriolis')
    dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
    if dftemp3.shape[0]!=0:
        plt.plot(dftemp3['flow-time'], dftemp3['u3'], 'C3-', label = 'Stable with Coriolis')
    dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
    if dftemp4.shape[0]!=0:
        plt.plot(dftemp4['flow-time'], dftemp4['u3'], 'C3:', label = 'Stable without Coriolis')
    plt.title('3rd point velocity magnitude')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\overline{u}_3$')
    plt.legend()

# calculated mean and std values
grouped = measuredData.groupby(['height','coriolis','profile'])

meanData = grouped.mean()[['vx1','vx2','vx3','vy1','vy2','vy3','vz1','vz2','vz3','u1', 'u2', 'u3']]
meanData.insert(0,'type','mean')
calculatedData = pd.concat([calculatedData, meanData])

stdData = grouped.std()[['vx1','vx2','vx3','vy1','vy2','vy3','vz1','vz2','vz3','u1', 'u2', 'u3']]
stdData.insert(0,'type','std')
calculatedData = pd.concat([calculatedData, stdData])

# u1 - hist
plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
    if dftemp1.shape[0]!=0:
        dftemp1['u1'].plot.hist(bins=25, label = 'Neutral with Coriolis', histtype='step', density=True)
    dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
    if dftemp2.shape[0]!=0:
        dftemp2['u1'].plot.hist(bins=25, label = 'Neutral without Coriolis', histtype='step', density=True)
    dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
    if dftemp3.shape[0]!=0:
        dftemp3['u1'].plot.hist(bins=25, label = 'Stable with Coriolis', histtype='step', density=True)
    dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
    if dftemp4.shape[0]!=0:
        dftemp4['u1'].plot.hist(bins=25, label = 'Stable without Coriolis', histtype='step', density=True)
    plt.title('1st point velocity histogram')
    plt.xlabel(r'$\overline{u}_1$')
    plt.ylabel(r'Frequency')
    plt.legend()
    
# u2 - hist
plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
    if dftemp1.shape[0]!=0:
        dftemp1['u2'].plot.hist(bins=25, label = 'Neutral with Coriolis', histtype='step', density=True)
    dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
    if dftemp2.shape[0]!=0:
        dftemp2['u2'].plot.hist(bins=25, label = 'Neutral without Coriolis', histtype='step', density=True)
    dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
    if dftemp3.shape[0]!=0:
        dftemp3['u2'].plot.hist(bins=25, label = 'Stable with Coriolis', histtype='step', density=True)
    dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
    if dftemp4.shape[0]!=0:
        dftemp4['u2'].plot.hist(bins=25, label = 'Stable without Coriolis', histtype='step', density=True)
    plt.title('2nd point velocity histogram')
    plt.xlabel(r'$\overline{u}_2$')
    plt.ylabel(r'Frequency')
    plt.legend()
    
# u3 - hist
plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
    if dftemp1.shape[0]!=0:
        dftemp1['u3'].plot.hist(bins=25, label = 'Neutral with Coriolis', histtype='step', density=True)
    dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
    if dftemp2.shape[0]!=0:
        dftemp2['u3'].plot.hist(bins=25, label = 'Neutral without Coriolis', histtype='step', density=True)
    dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
    if dftemp3.shape[0]!=0:
        dftemp3['u3'].plot.hist(bins=25, label = 'Stable with Coriolis', histtype='step', density=True)
    dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
    if dftemp4.shape[0]!=0:
        dftemp4['u3'].plot.hist(bins=25, label = 'Stable without Coriolis', histtype='step', density=True)
    plt.title('3rd point velocity histogram')
    plt.xlabel(r'$\overline{u}_3$')
    plt.ylabel(r'Frequency')
    plt.legend()
    
# u1 - fft
plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)

    if dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")].shape[0]!=0:
        dftemp1 = pd.DataFrame(np.fft.fft(dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]['u1']),columns=['magn'])
        # dftemp1['magn'] = abs(dftemp1['magn']).rolling(10,min_periods=1).mean()
        dftemp1['magn'] = abs(dftemp1['magn'])
        dftemp1 = dftemp1.iloc[0:int(len(dftemp1)/2)]
        dftemp1['freq'] = np.arange(0,len(dftemp1))/(len(dftemp1)*8*2)
        plt.loglog(dftemp1['freq'], dftemp1['magn'], 'C9-', label = 'Neutral with Coriolis')
    if dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")].shape[0]!=0:
        dftemp2 = pd.DataFrame(np.fft.fft(dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]['u1']),columns=['magn'])
        # dftemp2['magn'] = abs(dftemp2['magn']).rolling(10,min_periods=1).mean()
        dftemp2['magn'] = abs(dftemp2['magn'])
        dftemp2 = dftemp2.iloc[0:int(len(dftemp2)/2)]
        dftemp2['freq'] = np.arange(0,len(dftemp2))/(len(dftemp2)*8*2)
        plt.loglog(dftemp2['freq'], dftemp2['magn'], 'C9:', label = 'Neutral without Coriolis')
    if dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")].shape[0]!=0:
        dftemp3 = pd.DataFrame(np.fft.fft(dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]['u1']),columns=['magn'])
        # dftemp3['magn'] = abs(dftemp3['magn']).rolling(10,min_periods=1).mean()
        dftemp3['magn'] = abs(dftemp3['magn'])
        dftemp3 = dftemp3.iloc[0:int(len(dftemp3)/2)]
        dftemp3['freq'] = np.arange(0,len(dftemp3))/(len(dftemp3)*8*2)
        plt.loglog(dftemp3['freq'], dftemp3['magn'], 'C3-', label = 'Stable with Coriolis')
    if dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")].shape[0]!=0:
        dftemp4 = pd.DataFrame(np.fft.fft(dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]['u1']),columns=['magn'])
        # dftemp4['magn'] = abs(dftemp4['magn']).rolling(10,min_periods=1).mean()
        dftemp4['magn'] = abs(dftemp4['magn'])
        dftemp4 = dftemp4.iloc[0:int(len(dftemp4)/2)]
        dftemp4['freq'] = np.arange(0,len(dftemp4))/(len(dftemp4)*8*2)
        plt.loglog(dftemp4['freq'], dftemp4['magn'], 'C3:', label = 'Stable without Coriolis')
    plt.title('1st point velocity FFT')
    plt.xlabel(r'Frequency')
    plt.ylabel(r'PSD')
    plt.legend()
      
# u2 - fft
plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    if dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")].shape[0]!=0:
        dftemp1 = pd.DataFrame(np.fft.fft(dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]['u2']),columns=['magn'])
        dftemp1['magn'] = abs(dftemp1['magn']).rolling(10,min_periods=1).mean()
        dftemp1['freq'] = np.arange(0,len(dftemp1)*0.125,0.125)
        plt.plot(dftemp1['freq'], dftemp1['magn'], 'C9-', label = 'Neutral with Coriolis')
    if dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")].shape[0]!=0:    
        dftemp2 = pd.DataFrame(np.fft.fft(dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]['u2']),columns=['magn'])
        dftemp2['magn'] = abs(dftemp2['magn']).rolling(10,min_periods=1).mean()
        dftemp2['freq'] = np.arange(0,len(dftemp2)*0.125,0.125)
        plt.plot(dftemp2['freq'], dftemp2['magn'], 'C9:', label = 'Neutral without Coriolis')
    if dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")].shape[0]!=0:
        dftemp3 = pd.DataFrame(np.fft.fft(dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]['u2']),columns=['magn'])
        dftemp3['magn'] = abs(dftemp3['magn']).rolling(10,min_periods=1).mean()
        dftemp3['freq'] = np.arange(0,len(dftemp3)*0.125,0.125)
        plt.plot(dftemp3['freq'], dftemp3['magn'], 'C3-', label = 'Stable with Coriolis')
    if dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")].shape[0]!=0:
        dftemp4 = pd.DataFrame(np.fft.fft(dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]['u2']),columns=['magn'])
        dftemp4['magn'] = abs(dftemp4['magn']).rolling(10,min_periods=1).mean()
        dftemp4['freq'] = np.arange(0,len(dftemp4)*0.125,0.125)
        plt.plot(dftemp4['freq'], dftemp4['magn'], 'C3:', label = 'Stable without Coriolis')
    plt.title('2nd point velocity FFT')
    plt.xlabel(r'Frequency')
    plt.ylabel(r'PSD')
    plt.legend()
      
# u3 - fft
plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    if dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")].shape[0]!=0:
        dftemp1 = pd.DataFrame(np.fft.fft(dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]['u3']),columns=['magn'])
        dftemp1['magn'] = abs(dftemp1['magn']).rolling(10,min_periods=1).mean()
        dftemp1['freq'] = np.arange(0,len(dftemp1)*0.125,0.125)
        plt.loglog(dftemp1['freq'], dftemp1['magn'], 'C9-', label = 'Neutral with Coriolis')
    if dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")].shape[0]!=0:
        dftemp2 = pd.DataFrame(np.fft.fft(dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]['u3']),columns=['magn'])
        dftemp2['magn'] = abs(dftemp2['magn']).rolling(10,min_periods=1).mean()
        dftemp2['freq'] = np.arange(0,len(dftemp2)*0.125,0.125)
        plt.loglog(dftemp2['freq'], dftemp2['magn'], 'C9:', label = 'Neutral without Coriolis')
    if dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")].shape[0]!=0:
        dftemp3 = pd.DataFrame(np.fft.fft(dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]['u3']),columns=['magn'])
        dftemp3['magn'] = abs(dftemp3['magn']).rolling(10,min_periods=1).mean()
        dftemp3['freq'] = np.arange(0,len(dftemp3)*0.125,0.125)
        plt.loglog(dftemp3['freq'], dftemp3['magn'], 'C3-', label = 'Stable with Coriolis')
    if dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")].shape[0]!=0:
        dftemp4 = pd.DataFrame(np.fft.fft(dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]['u3']),columns=['magn'])
        dftemp4['magn'] = abs(dftemp4['magn']).rolling(10,min_periods=1).mean()
        dftemp4['freq'] = np.arange(0,len(dftemp4)*0.125,0.125)
        plt.loglog(dftemp4['freq'], dftemp4['magn'], 'C3:', label = 'Stable without Coriolis')
    plt.title('3rd point velocity FFT')
    plt.xlabel(r'Frequency')
    plt.ylabel(r'PSD')
    plt.legend()
      