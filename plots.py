#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: szenkajozsef
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

heightCase = ['01_flat','02_low','03_middle','04_high']

profileCase = ['neutral','stable']

coriolisCase = ['corilois','non_coriolis']
coriolisDict = {
    "corilois" : True,
    "non_coriolis" : False
    }

heights = [0, 10, 20, 40]

measuredData = pd.DataFrame()

for height in heightCase:
    
    for coriolis in coriolisCase:
        
        for profile in profileCase:
        
            source = 'data/' + height + '/' + coriolis + '/' + profile + '/VerticalProfiles_avg_3z.txt'
            
            try:
                df = pd.read_csv(source, sep='\t', skiprows = 1)
            except:
                print(source + ' cannot be found!\n')
                break
            
            df.insert(0,'height',height)
            df.insert(1,'coriolis',coriolisDict.get(coriolis))
            df.insert(2,'profile',profile)
            
            volAvg = df[' Vol'].mean()
            volMax = df[' Vol'].max()
            
            df['high'] = False
            df['low'] = False
            df[' UVmeanfilt'] = math.nan
            df[' VWmeanfilt'] = math.nan
            df[' UWmeanfilt'] = math.nan
            df['l'] = math.nan
            
            for index, row in df.iterrows():
                df.loc[index,'high']=True if (row[' Vol']>volAvg) * (row[' Vol']<volMax*0.99)  else False
                df.loc[index,'low']=True if row[' Vol']<volAvg*0.75 else False
            
            df['toDel'] = False
            for index, row in df.iterrows():
                if index!=df.shape[0]-1:
                    if ((df.loc[index,'high']==True)*(df.loc[index+1,'low']==True)) + ((df.loc[index,'low']==True)*(df.loc[index+1,'high']==True)):
                        V1 = df.loc[index,' Vol']
                        V2 = df.loc[index+1,' Vol']
                        df.loc[index,' Vol'] = V1 + V2 
                        df.loc[index,' Z'] = (df.loc[index,' Z']*V1 + df.loc[index+1,' Z']*V2)/(V1+V2)
                        df.loc[index,' Uavg'] = (df.loc[index,' Uavg']*V1 + df.loc[index+1,' Uavg']*V2)/(V1+V2)
                        df.loc[index,' Vavg'] = (df.loc[index,' Vavg']*V1 + df.loc[index+1,' Vavg']*V2)/(V1+V2)
                        df.loc[index,' Wavg'] = (df.loc[index,' Wavg']*V1 + df.loc[index+1,' Wavg']*V2)/(V1+V2)
                        df.loc[index,' Urms'] = np.sqrt((np.power(df.loc[index,' Urms'],2)*V1 + np.power(df.loc[index+1,' Urms'],2)*V2)/(V1+V2))
                        df.loc[index,' Vrms'] = np.sqrt((np.power(df.loc[index,' Vrms'],2)*V1 + np.power(df.loc[index+1,' Vrms'],2)*V2)/(V1+V2))
                        df.loc[index,' Wrms'] = np.sqrt((np.power(df.loc[index,' Wrms'],2)*V1 + np.power(df.loc[index+1,' Wrms'],2)*V2)/(V1+V2))
                        df.loc[index,' UVmean'] = (df.loc[index,' UVmean']*V1 + df.loc[index+1,' UVmean']*V2)/(V1+V2)
                        df.loc[index,' UWmean'] = (df.loc[index,' UWmean']*V1 + df.loc[index+1,' UWmean']*V2)/(V1+V2)
                        df.loc[index,' VWmean'] = (df.loc[index,' VWmean']*V1 + df.loc[index+1,' VWmean']*V2)/(V1+V2)
                        df.loc[index,'high'] = False
                        df.loc[index,'low'] = False
                        df.loc[index+1,'high'] = False
                        df.loc[index+1,'low'] = False
                        df.loc[index+1,'toDel'] = True
                    
            df = df[df['toDel']==False]
            
            df = df.reset_index(drop=True)
            
            df['udash']=np.sqrt(np.power(df[' Uavg'],2)+
                                np.power(df[' Vavg'],2)+
                                np.power(df[' Wavg'],2))
            
            df[' UVmeanfilt'] = df[' UVmean'].rolling(4,min_periods=1).mean()
            df[' UWmeanfilt'] = df[' UWmean'].rolling(4,min_periods=1).mean()
            df[' VWmeanfilt'] = df[' VWmean'].rolling(4,min_periods=1).mean()
            
            # length scale with the restriction du/dz > 0.005
            for index, row in df.iterrows():
                if index<df.shape[0]-1 and index>0:
                    df.loc[index,'dudz'] = ((df.loc[index+1,'udash']-df.loc[index,'udash'])/(df.loc[index+1,' Z']-df.loc[index,' Z']) + \
                        (df.loc[index,'udash']-df.loc[index-1,'udash'])/(df.loc[index,' Z']-df.loc[index-1,' Z'])) /2
            
            df['dudzfilt'] = df['dudz'].rolling(4,min_periods=1).mean()
            
            for index, row in df.iterrows():
                if index<df.shape[0]-1 and index>0:
                    if abs(df.loc[index,'dudzfilt']) >= 0.005:
                        df.loc[index,'l'] = np.power(np.power(df.loc[index,' UWmeanfilt'],2) + np.power(df.loc[index,' VWmeanfilt'],2),1/4)/ df.loc[index,'dudzfilt']
                            
            df['alfa']=list(map(math.atan2,df[' Vavg'],df[' Uavg']))
            df['alfa']=df['alfa']*180/math.pi
            
            plt.plot(df['udash'], df[' Z'], 
                      label=height+" "+ coriolis + " "+profile) 
            plt.title('Comparision')
            plt.xlabel(r'$\overline{u}$')
            plt.ylabel(r'$Z$')
            plt.legend()
            
            measuredData = pd.concat([measuredData,df])
          
# velocity magnitude  

plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
    if dftemp1.shape[0]!=0:
        plt.plot(dftemp1['udash'], dftemp1[' Z'], 'C9-', label = 'Neutral with Coriolis')
    dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
    if dftemp2.shape[0]!=0:
        plt.plot(dftemp2['udash'], dftemp2[' Z'], 'C9:', label = 'Neutral without Coriolis')
    dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
    if dftemp3.shape[0]!=0:
        plt.plot(dftemp3['udash'], dftemp3[' Z'], 'C3-', label = 'Stable with Coriolis')
    dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
    if dftemp4.shape[0]!=0:
        plt.plot(dftemp4['udash'], dftemp4[' Z'], 'C3:', label = 'Stable without Coriolis')
    plt.axhline(y=heights[i], color='0.5', linestyle='--', label = 'Object height')
    plt.title('Velocity magnitude')
    plt.xlabel(r'$\overline{u}$')
    plt.ylabel(r'$Z$')
    plt.legend()
plt.savefig('plots/velocity_magn.png', bbox_inches='tight')

# wind direction

plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
    if dftemp1.shape[0]!=0:
        plt.plot(dftemp1['alfa'], dftemp1[' Z'], 'C9-', label = 'Neutral with Coriolis')
    dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
    if dftemp2.shape[0]!=0:
        plt.plot(dftemp2['alfa'], dftemp2[' Z'], 'C9:', label = 'Neutral without Coriolis')
    dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
    if dftemp3.shape[0]!=0:
        plt.plot(dftemp3['alfa'], dftemp3[' Z'], 'C3-', label = 'Stable with Coriolis')
    dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
    if dftemp4.shape[0]!=0:
        plt.plot(dftemp4['alfa'], dftemp4[' Z'], 'C3:', label = 'Stable without Coriolis')
    plt.axhline(y=heights[i], color='0.5', linestyle='--', label = 'Object height')
    plt.title('Velocity angle')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$Z$')
    plt.legend()
plt.savefig('plots/direction.png', bbox_inches='tight')

# Ekman-spiral

plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
    if dftemp1.shape[0]!=0:
        plt.plot(dftemp1[' Uavg'], dftemp1[' Vavg'], 'C9-', label = 'Neutral with Coriolis')
    dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
    if dftemp2.shape[0]!=0:
        plt.plot(dftemp2[' Uavg'], dftemp2[' Vavg'], 'C9:', label = 'Neutral without Coriolis')
    dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
    if dftemp3.shape[0]!=0:
        plt.plot(dftemp3[' Uavg'], dftemp3[' Vavg'], 'C3-', label = 'Stable with Coriolis')
    dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
    if dftemp4.shape[0]!=0:
        plt.plot(dftemp4[' Uavg'], dftemp4[' Vavg'], 'C3:', label = 'Stable without Coriolis')
    plt.title('Velocity components')
    plt.xlabel(r'$\overline{u}$')
    plt.ylabel(r'$\overline{v}$')
    plt.legend()
plt.savefig('plots/eckman.png', bbox_inches='tight')

# RMS values

plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
    if dftemp1.shape[0]!=0:
        plt.plot(dftemp1[' Urms'], dftemp1[' Z'], 'C9-', label = 'Neutral with Coriolis')
    dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
    if dftemp2.shape[0]!=0:
        plt.plot(dftemp2[' Urms'], dftemp2[' Z'], 'C9:', label = 'Neutral without Coriolis')
    dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
    if dftemp3.shape[0]!=0:
        plt.plot(dftemp3[' Urms'], dftemp3[' Z'], 'C3-', label = 'Stable with Coriolis')
    dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
    if dftemp4.shape[0]!=0:
        plt.plot(dftemp4[' Urms'], dftemp4[' Z'], 'C3:', label = 'Stable without Coriolis')
    plt.axhline(y=heights[i], color='0.5', linestyle='--', label = 'Object height')
    plt.title('X velocity RMS')
    plt.xlabel(r'$\sigma_{u}$')
    plt.ylabel(r'$Z$')
    plt.legend()
plt.savefig('plots/x_rms.png', bbox_inches='tight')


plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
    if dftemp1.shape[0]!=0:
        plt.plot(dftemp1[' Vrms'], dftemp1[' Z'], 'C9-', label = 'Neutral with Coriolis')
    dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
    if dftemp2.shape[0]!=0:
        plt.plot(dftemp2[' Vrms'], dftemp2[' Z'], 'C9:', label = 'Neutral without Coriolis')
    dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
    if dftemp3.shape[0]!=0:
        plt.plot(dftemp3[' Vrms'], dftemp3[' Z'], 'C3-', label = 'Stable with Coriolis')
    dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
    if dftemp4.shape[0]!=0:
        plt.plot(dftemp4[' Vrms'], dftemp4[' Z'], 'C3:', label = 'Stable without Coriolis')
    plt.axhline(y=heights[i], color='0.5', linestyle='--', label = 'Object height')
    plt.title('Y velocity RMS')
    plt.xlabel(r'$\sigma_{v}$')
    plt.ylabel(r'$Z$')
    plt.legend()
plt.savefig('plots/y_rms.png', bbox_inches='tight')

plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
    if dftemp1.shape[0]!=0:
        plt.plot(dftemp1[' Wrms'], dftemp1[' Z'], 'C9-', label = 'Neutral with Coriolis')
    dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
    if dftemp2.shape[0]!=0:
        plt.plot(dftemp2[' Wrms'], dftemp2[' Z'], 'C9:', label = 'Neutral without Coriolis')
    dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
    if dftemp3.shape[0]!=0:
        plt.plot(dftemp3[' Wrms'], dftemp3[' Z'], 'C3-', label = 'Stable with Coriolis')
    dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
    if dftemp4.shape[0]!=0:
        plt.plot(dftemp4[' Wrms'], dftemp4[' Z'], 'C3:', label = 'Stable without Coriolis')
    plt.axhline(y=heights[i], color='0.5', linestyle='--', label = 'Object height')
    plt.title('Z velocity RMS')
    plt.xlabel(r'$\sigma_{w}$')
    plt.ylabel(r'$Z$')
    plt.legend()
plt.savefig('plots/z_rms.png', bbox_inches='tight')

# original momentum flux

# plt.figure(figsize=(14,11))
# for i in range(0,4,1):
#     dftemp = measuredData[measuredData['height']==heightCase[i]]

#     plt.subplot(2, 2, i+1)
    
#     dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
#     if dftemp1.shape[0]!=0:
#         plt.plot(dftemp1[' UVmean'], dftemp1[' Z'], 'C9-', label = 'Neutral with Coriolis')
#     dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
#     if dftemp2.shape[0]!=0:
#         plt.plot(dftemp2[' UVmean'], dftemp2[' Z'], 'C9:', label = 'Neutral without Coriolis')
#     dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
#     if dftemp3.shape[0]!=0:
#         plt.plot(dftemp3[' UVmean'], dftemp3[' Z'], 'C3-', label = 'Stable with Coriolis')
#     dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
#     if dftemp4.shape[0]!=0:
#         plt.plot(dftemp4[' UVmean'], dftemp4[' Z'], 'C3:', label = 'Stable without Coriolis')
#     plt.axhline(y=heights[i], color='0.5', linestyle='--', label = 'Object height')
#     plt.title('XY turbulence flux')
#     plt.xlabel(r"$\overline{u'v'}$")
#     plt.ylabel(r'$Z$')
#     plt.legend()
    
# plt.figure(figsize=(14,11))
# for i in range(0,4,1):
#     dftemp = measuredData[measuredData['height']==heightCase[i]]

#     plt.subplot(2, 2, i+1)
    
#     dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
#     if dftemp1.shape[0]!=0:
#         plt.plot(dftemp1[' UWmean'], dftemp1[' Z'], 'C9-', label = 'Neutral with Coriolis')
#     dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
#     if dftemp2.shape[0]!=0:
#         plt.plot(dftemp2[' UWmean'], dftemp2[' Z'], 'C9:', label = 'Neutral without Coriolis')
#     dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
#     if dftemp3.shape[0]!=0:
#         plt.plot(dftemp3[' UWmean'], dftemp3[' Z'], 'C3-', label = 'Stable with Coriolis')
#     dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
#     if dftemp4.shape[0]!=0:
#         plt.plot(dftemp4[' UWmean'], dftemp4[' Z'], 'C3:', label = 'Stable without Coriolis')
#     plt.axhline(y=heights[i], color='0.5', linestyle='--', label = 'Object height')
#     plt.title('XZ turbulence flux')
#     plt.xlabel(r"$\overline{u'w'}$")
#     plt.ylabel(r'$Z$')
#     plt.legend()
    
# plt.figure(figsize=(14,11))
# for i in range(0,4,1):
#     dftemp = measuredData[measuredData['height']==heightCase[i]]

#     plt.subplot(2, 2, i+1)
    
#     dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
#     if dftemp1.shape[0]!=0:
#         plt.plot(dftemp1[' VWmean'], dftemp1[' Z'], 'C9-', label = 'Neutral with Coriolis')
#     dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
#     if dftemp2.shape[0]!=0:
#         plt.plot(dftemp2[' VWmean'], dftemp2[' Z'], 'C9:', label = 'Neutral without Coriolis')
#     dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
#     if dftemp3.shape[0]!=0:
#         plt.plot(dftemp3[' VWmean'], dftemp3[' Z'], 'C3-', label = 'Stable with Coriolis')
#     dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
#     if dftemp4.shape[0]!=0:
#         plt.plot(dftemp4[' VWmean'], dftemp4[' Z'], 'C3:', label = 'Stable without Coriolis')
#     plt.axhline(y=heights[i], color='0.5', linestyle='--', label = 'Object height')
#     plt.title('YZ turbulence flux')
#     plt.xlabel(r"$\overline{v'w'}$")
#     plt.ylabel(r'$Z$')
#     plt.legend()

# filtered momentum flux

plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
    if dftemp1.shape[0]!=0:
        plt.plot(dftemp1[' UVmeanfilt'], dftemp1[' Z'], 'C9-', label = 'Neutral with Coriolis')
    dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
    if dftemp2.shape[0]!=0:
        plt.plot(dftemp2[' UVmeanfilt'], dftemp2[' Z'], 'C9:', label = 'Neutral without Coriolis')
    dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
    if dftemp3.shape[0]!=0:
        plt.plot(dftemp3[' UVmeanfilt'], dftemp3[' Z'], 'C3-', label = 'Stable with Coriolis')
    dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
    if dftemp4.shape[0]!=0:
        plt.plot(dftemp4[' UVmeanfilt'], dftemp4[' Z'], 'C3:', label = 'Stable without Coriolis')
    plt.axhline(y=heights[i], color='0.5', linestyle='--', label = 'Object height')
    plt.title('XY turbulence flux filtered')
    plt.xlabel(r"$\overline{u'v'}$")
    plt.ylabel(r'$Z$')
    plt.legend()
plt.savefig('plots/xy_flux.png', bbox_inches='tight')

plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
    if dftemp1.shape[0]!=0:
        plt.plot(dftemp1[' UWmeanfilt'], dftemp1[' Z'], 'C9-', label = 'Neutral with Coriolis')
    dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
    if dftemp2.shape[0]!=0:
        plt.plot(dftemp2[' UWmeanfilt'], dftemp2[' Z'], 'C9:', label = 'Neutral without Coriolis')
    dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
    if dftemp3.shape[0]!=0:
        plt.plot(dftemp3[' UWmeanfilt'], dftemp3[' Z'], 'C3-', label = 'Stable with Coriolis')
    dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
    if dftemp4.shape[0]!=0:
        plt.plot(dftemp4[' UWmeanfilt'], dftemp4[' Z'], 'C3:', label = 'Stable without Coriolis')
    plt.axhline(y=heights[i], color='0.5', linestyle='--', label = 'Object height')
    plt.title('XZ turbulence flux filtered')
    plt.xlabel(r"$\overline{u'w'}$")
    plt.ylabel(r'$Z$')
    plt.legend()
plt.savefig('plots/xz_flux.png', bbox_inches='tight')
    
plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
    if dftemp1.shape[0]!=0:
        plt.plot(dftemp1[' VWmeanfilt'], dftemp1[' Z'], 'C9-', label = 'Neutral with Coriolis')
    dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
    if dftemp2.shape[0]!=0:
        plt.plot(dftemp2[' VWmeanfilt'], dftemp2[' Z'], 'C9:', label = 'Neutral without Coriolis')
    dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
    if dftemp3.shape[0]!=0:
        plt.plot(dftemp3[' VWmeanfilt'], dftemp3[' Z'], 'C3-', label = 'Stable with Coriolis')
    dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
    if dftemp4.shape[0]!=0:
        plt.plot(dftemp4[' VWmeanfilt'], dftemp4[' Z'], 'C3:', label = 'Stable without Coriolis')
    plt.axhline(y=heights[i], color='0.5', linestyle='--', label = 'Object height')
    plt.title('YZ turbulence flux filtered')
    plt.xlabel(r"$\overline{v'w'}$")
    plt.ylabel(r'$Z$')
    plt.legend()
plt.savefig('plots/yz_flux.png', bbox_inches='tight')

# length scale
plt.figure(figsize=(14,11))
for i in range(0,4,1):
    dftemp = measuredData[measuredData['height']==heightCase[i]]

    plt.subplot(2, 2, i+1)
    
    dftemp1 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="neutral")]
    if dftemp1.shape[0]!=0:
        plt.plot(dftemp1['l'], dftemp1[' Z'], 'C9-', label = 'Neutral with Coriolis')
    dftemp2 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="neutral")]
    if dftemp2.shape[0]!=0:
        plt.plot(dftemp2['l'], dftemp2[' Z'], 'C9:', label = 'Neutral without Coriolis')
    dftemp3 = dftemp[(dftemp['coriolis']==True) & (dftemp['profile']=="stable")]
    if dftemp3.shape[0]!=0:
        plt.plot(dftemp3['l'], dftemp3[' Z'], 'C3-', label = 'Stable with Coriolis')
    dftemp4 = dftemp[(dftemp['coriolis']==False) & (dftemp['profile']=="stable")]
    if dftemp4.shape[0]!=0:
        plt.plot(dftemp4['l'], dftemp4[' Z'], 'C3:', label = 'Stable without Coriolis')
    plt.axhline(y=heights[i], color='0.5', linestyle='--', label = 'Object height')
    plt.title('Turbulence length scale')
    plt.xlabel(r"$l$")
    plt.ylabel(r'$Z$')
    plt.legend()
plt.savefig('plots/length_scale.png', bbox_inches='tight')
