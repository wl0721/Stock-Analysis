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

# 得到给定日期内所有10支股票的14个tag数据
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
				# df.index = range(4802)  # 将每个df的行索引值都改成range(4802)

                Z = pd.concat([df, Z], axis=1)  # 将所有的tag进行横向上的拼接
            Z['code'] = code  # 在Z中添加一列，存放所有的股票代码
            Y.append(Z)  # 将所有的股票数据进行列向上的拼接
        X.append(Y)  # 将所有的day（2016年1月份）进行列向上的拼接
        df_ = X.code  # 取出要操作的code列
        X = X.drop('code', axis=1)  # 在X中删除目标列
        X.insert(0, 'code', df_)  # 选择位置，再重新在X中插入code列
    X.index = range(len(X))
    # print(X.head())
    X.to_csv('/home/wanglin/Documents/get_data/label/data_wl/data.csv')  # data.csv文件中保存的是所有10支股票在给定日期内的所有tick数据

	#  对data.csv文件中的tick数据进行采样，每隔三分钟采样一次
	with open('/home/wanglin/Documents/get_data/label/data_wl/data.csv') as reader, open('/home/wanglin/Documents/get_data/label/data_new.csv', 'w') as writer:
		for index, line in enumerate(reader):
		  	if (index - 1) % 60 == 0:
				writer.write(line)
	new_data = pd.read_csv('/home/wanglin/Documents/get_data/label/data_new.csv', header=None)
	new_data.columns = ['index', 'high', 'low', 'last', 'ask1', 'bid1', 'ask_vol1', 'bid_vol1', 'amount', 'trade', 'avebid', 'aveoff', 'totbid', 'totoff', 'vol']
    new_data.drop('index', axis=1, inplace=True)
    new_data.to_csv('/home/wanglin/Documents/get_data/label/data_new_2.csv')

# 对得到的股票数据进行小波去噪
def wavelet_decomposition():
	x = [i for i in range(len(data))]
	# print(np.array(x).shape)  #(4801,)

	index_list = data[:-10]
	day_list1 = np.array(x)[:-10]
	# print(day_list1.shape)  #(4791,)

	index_for_predict = data[-10:]
	day_list2 = np.array(x)[-10:]
	# print(index_list.shape)  #(4791,1)
	# print(index_for_predict.shape)  #(10.1)

	A3, D3, D2, D1 = pywt.wavedec(index_list, 'db4', mode='sym', level=3) #wavelet decomposition
	coeff = [A3, D3, D2, D1]
	# print(coeff)

	order_A3 = sm.tsa.arma_order_select_ic(A3,ic='aic')['aic_min_order']
	order_D3 = sm.tsa.arma_order_select_ic(D3,ic='aic')['aic_min_order']
	order_D2 = sm.tsa.arma_order_select_ic(D2,ic='aic')['aic_min_order']
	order_D1 = sm.tsa.arma_order_select_ic(D1,ic='aic')['aic_min_order']

	print("order_A3, order_D3, order_D2, order_D1: ", order_A3, order_D3, order_D2, order_D1)
	# order_A3:(4,2); order_D3:(4,2); order_D2:(4,1); order_D1:(0,1)

	model_A3 = ARMA(A3, order=order_A3)
	model_D3 = ARMA(D3, order=order_D3)
	model_D2 = ARMA(D2, order=order_D2)
	model_D1 = ARMA(D1, order=order_D1)

	results_A3 = model_A3.fit()
	results_D3 = model_D3.fit()
	results_D2 = model_D2.fit()
	results_D1 = model_D1.fit()

	plt.figure()
	plt.subplot(411)
	plt.plot(A3, 'blue')
	plt.plot(results_A3.fittedvalues, 'red')
	plt.title('model_A3')

	plt.subplot(412)
	plt.plot(D3,'blue')
	plt.plot(results_D3.fittedvalues, 'red')
	plt.title('model_D3')

	plt.subplot(413)
	plt.plot(D2,'blue')
	plt.plot(results_D2.fittedvalues, 'red')
	plt.title('model_D2')

	plt.subplot(414)
	plt.plot(D1, 'blue')
	plt.plot(results_D1.fittedvalues, 'red')
	plt.title('model_D1')
	plt.tight_layout(h_pad = 2.0)
	plt.legend(loc='best')
	plt.show()
	# plt.savefig('figure_1.jpg')

	A3_all, D3_all, D2_all, D1_all = pywt.wavedec(data, 'db4', mode='sym', level=3)
	delta = [len(A3_all) - len(A3), len(D3_all) - len(D3), len(D2_all) - len(D2), len(D1_all) - len(D1)]
	print("delta: ", delta)  #delta:[1,1,2,5]

	pA3 = model_A3.predict(params=results_A3.params,start=1,end=len(A3)+delta[0])
	pD3 = model_D3.predict(params=results_D3.params,start=1,end=len(D3)+delta[1])
	pD2 = model_D2.predict(params=results_D2.params,start=1,end=len(D2)+delta[2])
	pD1 = model_D1.predict(params=results_D1.params,start=1,end=len(D1)+delta[3])

	coffe_new = [pA3, pD3, pD2, pD1]
	denoised_index = pywt.waverec(coffe_new, 'db4')

	plt.figure()
	plt.subplot(211)
	plt.plot(index_list, color='blue', label='index_list', linewidth=1)
	plt.legend(loc='best')

	plt.subplot(212)
	plt.plot(denoised_index, color='red', label='denoised_index', linewidth=1)
	plt.legend(loc='best')
	plt.show()
	# plt.savefig('figure_2.jpg')

	temp_data_wt = {'real_value':index_for_predict, 'pre_value_wt':denoised_index[-10:],
                'err_wt':denoised_index[-10:]-index_for_predict,
                'err_rate_wt/%':(denoised_index[-10:]-index_for_predict)/index_for_predict*100}
	predict_wt = pd.DataFrame(temp_data_wt,index=day_list2,columns=['real_value','pre_value_wt','err_wt','err_rate_wt/%'])
	print("predict_wt: \n", predict_wt)
	
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
