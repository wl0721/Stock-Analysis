#!/user/bin/env python3
# -*- coding: utf-8 -*-

from load_md import get_md_by_tick as gmt
from IndexData import get_deal_day_list_in_period
from label import get_lab
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tag_list = ['high', 'low', 'last', 'ask1', 'bid1', 'ask_vol1', 'bid_vol1', 'amount',
            'trade', 'avebid', 'aveoff', 'totbid', 'totoff', 'vol']
start_day = '20160101'
end_day = '20160131'
day_list = get_deal_day_list_in_period(start_day, end_day)
# code_list = list(pd.read_csv('/home/wanglin/Documents/get_data/label/code.csv', index_col=0).code.values)
# label = get_lab()
# label = label.set_index('index')
# code_list = label.index.values  # 得到所有code的列表
code_list = ['000001', '000002', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011']  # 选取10支股票

# 得到给定日期内所有股票的14个tag数据
def get_data():
    X = pd.DataFrame()
    for day in day_list:
        Y = pd.DataFrame()
        for code in code_list:
            Z = pd.DataFrame()
            for tag in tag_list:
                if tag == 'amount' or tag == 'totbid' or tag == 'totoff' or tag == 'vol':
                    df = gmt(int(day), tag, dtype='int64', codes=[code])
                else:
                    df = gmt(int(day), tag, dtype='float32', codes=[code])

                Z = pd.concat([df, Z], axis=1)  # 将所有的tag进行横向上的拼接
            Z['code'] = code  # 在Z中添加一列，存放所有的股票代码
            Y.append(Z)  # 将所有的股票数据进行列向上的拼接
        X.append(Y)  # 将所有的day（2016年1月份）进行列向上的拼接
        df_ = X.code  # 取出要操作的code列
        X = X.drop('code', axis=1)  # 在X中删除目标列
        X.insert(0, 'code', df_)  # 选择位置，再重新在X中插入code列
    # X.index = range(len(X))
    # print(X.head())
    X.to_csv('/home/wanglin/Documents/get_data/label/data_wl/data.csv')

path1 = '/home/wanglin/Documents/get_data/label/data_wl'
# 得到给定日期内所有股票（3267）每天的收益率
def get_label(path=path1):
    file_path = path1 + '/' + 'label_raw' + '.csv'
    label = get_lab()
    label = label.fillna(method='ffill')
    label = label.set_index('index')
    day_list = label.columns.values[1:]
    # code_list = label.index.values
    code_list = ['000001', '000002', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011']
    
    f = open(file_path, 'w')

    line1 = 'raw' + '\n'
    f.writelines(line1)
    for day in day_list:
        for code in code_list:
            line2 = str(day) + ' ' + str(code) + ' ' + str(label.loc[code, day]) + '\n'
            f.writelines(line2)
    f.close()

    df = pd.read_csv('/home/wanglin/Documents/get_data/label/data_wl/label_raw.csv')
    data = pd.DataFrame(columns=['code', 'earn_rate'])
    data['code'] = df['raw'].apply(lambda x: x.split(' ')[1])
    data['earn_rate'] = df['raw'].apply(lambda x: x.split(' ')[2])
    data.index = df['raw'].apply(lambda x: x.split(' ')[0])
    data.index.name = 'date'

    data.to_csv('/home/wanglin/Documents/get_data/label/data_wl/label.csv')

def load_data():
    data = pd.read_csv('/home/wanglin/Documents/get_data/label/data.csv', index_col=0)
    label = pd.read_csv('/home/wanglin/Documents/get_data/label/label.csv', index_col=0)

    data.drop('code', axis=1, inplace=True)
    label.drop('code', axis=1, inplace=True)

    label = label.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    x_train, y_train, x_test, y_test = [], [], [], []
    for i in range(data.shape[0] // 1620):  # i表示的code数，data.shape[0]表示的是所有的行数，每一天采样得到81个数据
        df = data[i*1620 : (i+1)*1620]
        if df.isnull().sum().max() >= 81:
            continus
        df = df.fillna(method='bfill').fillna(method='ffill')
        df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        for j in range(15):
            x_train.append(df[j*81:(j+5)*81].values)
            y_train.append(label['earn_rate'][10*i+j:10*i+j+1].values)
        x_test.append(df[-405:].values)
        y_test.append(label['earn_rate'][10*i+9:10*(i+1)].values)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test

def buile_model():
    model = Sequential()
    model.add(LSTM(15, input_size=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(10))
    model.add(Dense(1, activation='relu'))
    
    start = time.time()
    model.compile(loss='mse', optimizer='adam')
    print('Compilation Time: ', time.time() - start)
    return model
