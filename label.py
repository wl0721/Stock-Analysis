# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:13:43 2019

@author: Administrator
"""

import pandas as pd
import os
import re

path1 = '/home/wanglin/Documents/get_data/label/label/'
path2 = "/data/stock/newSystemData/rawdata/wind/index/"

def get_ret(path=path2, start='20160101', end='20160131'):
    n = 0
    new_data = pd.DataFrame(columns=('open','date'))
    for fpathe, _, fs in os.walk(path):
            for f in fs:
                tmp_path = os.path.join(fpathe,f)
                # print(tmp_path)
                pattern1 = re.compile(r'(2016\/[0-9]+\/[0-9]+\/)+zz500+(\.[a-z]+)$')
                date = re.search(pattern1, tmp_path)
                # print(date)
                if date != None:
                    df = pd.read_csv(tmp_path, header=None, delimiter=' ', dtype={'code':str})
                    df = df.iloc[:,[0]]
                    # print(df)
                    date_value = date.group().split('/zz500.csv')
                    # date_value = pd.to_datetime(date_value,format="%Y/%m/%d")
                    pattern2 = re.compile(r'\/')
                    new_date = re.sub(pattern2,'',date_value[0])
                    df['date'] = str(new_date)
                    data = df.loc[0].values.ravel()
                    new_data.loc[n] = data
                    n += 1
    new_data = new_data.set_index('date')	
    new_data = new_data.sort_index()
    ye_open = new_data.iloc[:-1,:]
    to_open = new_data.iloc[1:,:]
    ind = to_open.index
    ye_open = ye_open.reset_index(drop=True)
    to_open = to_open.reset_index(drop=True)
    ret = (to_open-ye_open) / ye_open
   
    ret = pd.Series(ret['open'].values,index=ind)
    ret = ret[start:end]
    # print(ret)
    return ret

def get_lab(path=path1, start='20160101', end='20160131'):
    re_data = []
    for fpathe, _, fs in os.walk(path):
        for f in fs:
            tmp_path = os.path.join(fpathe, f)
            pattern = re.compile(r'201[6-9][0-9]+(\.[a-z]+)$')
            date = re.search(pattern, tmp_path)
            if date != None:
                date_value = date.group().split('.csv')                      
                data = pd.read_csv(tmp_path, delimiter='\t', header=None, dtype={0:str,2:float})                   
                data = data.set_index(0)
                value = data[2]  
                value.name = date_value[0]                   
                re_data.append(value)

    re_value = pd.concat(re_data, join='outer', axis=1)
    re_value = re_value.sort_index(axis=1)
    re_value = re_value.loc[:,start:end]
    #print(re_value)
   
    ret = get_ret()

    label = re_value-ret
    label = label.reset_index()
    return label

label = get_lab()
label.to_csv('/home/wanglin/Documents/get_data/label/label.csv')
print(label.head())