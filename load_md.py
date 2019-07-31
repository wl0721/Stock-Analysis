# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:15:51 2019

@author: Administrator
"""

 
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:44:53 2019

@author: User
"""
import pandas as pd
import numpy as np
import os
from titk import ticks_to_ti

shape = (5120,4096)
data_path ='/data/remoteDir/server_200/mem_data' 
inx_path = os.path.join(data_path ,'.index/tick.csv')
col_path = os.path.join(data_path ,'.index/code.csv')

inx = pd.read_csv(inx_path).set_index('tick_time')['idx']
col = pd.read_csv(col_path,dtype={'stock_code':str}).set_index('stock_code')['idx']
#print(inx.values)
                  
def get_md_by_tick(day,tag,ticks=None,codes=None,dtype='float32',fillna=True,change_zero=True):
    year = str(day // 10000).zfill(4)
    month = str(day // 100 % 100).zfill(2)
    date= str(day% 100).zfill(2)
    path = os.path.join(data_path,year,month,date,tag)
    assert os.path.exists(path),f'{path} not exists'
    
    ticks = inx.index.tolist() if ticks is None else ticks
    codes = col.index.tolist() if codes is None else codes
    
    r = inx.reindex(ticks).values
    c = col.reindex(codes).values
    
    file = np.memmap(path,mode='r',shape=shape,dtype=dtype)
    df = pd.DataFrame(file[r][:,c] ,index = ticks ,columns= codes)
    if change_zero:
        df = df.replace(0,np.nan)
    if fillna:
        df = df.fillna(method='ffill')
    return df

def get_md_by_ti(day,tag,ti=None,codes=None,dtype='float32',fillna=True,change_zero=True):
    year = str(day // 10000).zfill(4)
    month = str(day // 100 % 100).zfill(2)
    date= str(day% 100).zfill(2)    
    path = os.path.join(data_path,year,month,date,tag)
    assert os.path.exists(path),f'{path} not exists'
    
    ti = ticks_to_ti(ticks=inx.index.tolist()) if ti is None else ti
    ticks = ticks_to_ti(ti = ti)
    codes = col.index.tolist() if codes is None else codes
    
    r = inx.reindex(ticks).values
    c = col.reindex(codes).values
    
    file = np.memmap(path,mode='r',shape=shape,dtype=dtype)
    df = pd.DataFrame(file[r][:,c] ,index = ti ,columns= codes)
    
    if change_zero:
        df = df.replace(0,np.nan)
    if fillna:
        df = df.fillna(method='ffill')
    return df