# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:14:15 2019

@author: Administrator
"""

import pandas as pd 
from label import get_lab

path1 = '/home/wanglin/Documents/get_data/label/'

def load_label(path = path1):
    file_path = path1 + '/' + 'label1' + '.csv'
    label = get_lab()
    label = label.fillna(method='ffill')
    label = label.set_index('index')
    date_list = label.columns.values[1:]
    codes_list = label.index.values


    f = open(file_path, 'w')

    line1 = 'raw' + '\n'
    f.writelines(line1)
    for day in date_list:
        for code in codes_list:
            line2 = str(day) + ' ' + str(code) + ' ' + str(label.loc[code, day]) + '\n'
            f.writelines(line2)

    f.close()

load_label()

df = pd.read_csv('/home/wanglin/Documents/get_data/label/label1.csv')
data = pd.DataFrame(columns=['code', 'earn_rate'])
data['code'] = df['raw'].apply(lambda x: x.split(' ')[1])
data['earn_rate'] = df['raw'].apply(lambda x: x.split(' ')[2])

data.index = df['raw'].apply(lambda x: x.split(' ')[0])
data.index.name = 'date'

data.to_csv('/home/wanglin/Documents/get_data/label/label_new.csv')
# print(data.head())