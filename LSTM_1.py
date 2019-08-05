# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:43:42 2019

@author: Administrator
"""

import pandas as pd 
import numpy as np 
from load_md import get_md_by_tick as gmt 
import tensorflow as tf
import matplotlib.pyplot as plt

X1 = []
X2 = []
X3 = []
X4 = []
X5 = []
tags = ['open', 'high', 'low', 'last']
for tag in tags:
    df1 = gmt(20190408, tag, dtype='float32', codes=['000001'])
    df2 = gmt(20190409, tag, dtype='float32', codes=['000001'])
    df3 = gmt(20190410, tag, dtype='float32', codes=['000001'])
    df4 = gmt(20190411, tag, dtype='float32', codes=['000001'])
    df5 = gmt(20190412, tag, dtype='float32', codes=['000001'])

    df1 = df1.rename(columns={'000001':tag})
    df2 = df2.rename(columns={'000001':tag})
    df3 = df3.rename(columns={'000001':tag})
    df4 = df4.rename(columns={'000001':tag})
    df5 = df5.rename(columns={'000001':tag})

    X1.append(df1)
    X2.append(df2)
    X3.append(df3)
    X4.append(df4)
    X5.append(df5)

    Y1 = pd.concat(X1, axis=1)
    Y2 = pd.concat(X2, axis=1)
    Y3 = pd.concat(X3, axis=1)
    Y4 = pd.concat(X4, axis=1)
    Y5 = pd.concat(X5, axis=1)

    df = pd.concat([Y1, Y2, Y3, Y4, Y5], axis=0)

data = df.iloc[:, 0:4].values
#print(len(data))  #24010

def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=19208): 
    batch_index = []
    data_train = data[train_begin : train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis=0))/np.std(data_train, axis=0)
    train_x, train_y = [], []
    for i in range(len(normalized_train_data) - time_step): 
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i+time_step, :4]
        y = normalized_train_data[i:i+time_step, 3, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y

def get_test_data(time_step=20, test_begin=19208):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean)/std
    size = (len(normalized_test_data) + time_step - 1)//time_step
    test_x, test_y = [], []
    for i in range(size-1):
        x = normalized_test_data[i*time_step : (i+1)*time_step, :4]
        y = normalized_test_data[i*time_step : (i+1)*time_step, 3]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:, :4]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:, 3]).tolist())
    return mean, std, test_x, test_y

rnn_unit = 3
lstm_layers = 2
input_size = 4
output_size = 1
lr = 0.0006

weights = {
        'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
        'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
        }
biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit,])),
        'out': tf.Variable(tf.constant(0.1, shape=[1,]))
        }
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

def lstmCell():
    basicLstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=keep_prob)
    return basicLstm

def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
    cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell() for i in range(lstm_layers)])
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit]) 
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states

def train_lstm(batch_size=60, time_step=20, train_begin=4802, train_end=9604):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred,[-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            for step in range(len(batch_index)-1):
                _, loss_ = sess.run([train_op,loss], feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]],keep_prob:0.5})
            print("Number of iterations:",i," loss:",loss_)
        print("model_save: ", saver.save(sess,'model_save2/modle.ckpt'))
        #I run the code on windows 10,so use  'model_save2\\modle.ckpt'
        #if you run it on Linux,please use  'model_save2/modle.ckpt'
        print("The train has finished")
        
train_lstm()

def prediction(time_step=20):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y = get_test_data(time_step)
    print(test_x)
    with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint('model_save2')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X:[test_x[step]], keep_prob:1})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[3] + mean[3]
        test_predict = np.array(test_predict) * std[3] + mean[3]
        print(test_predict)
        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)]) #脝芦虏卯鲁脤露脠
        print("The accuracy of this predict:", acc)
        
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b',)
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()
        
prediction()

"""
df.index = range(len(df))

newData = pd.DataFrame(index = range(0,len(df)), columns=['open', 'high', 'low', 'last'])

for tag in tags:
    for i in range(0, len(df)):
       newData[tag][i] = df[tag][i]

#print(newData)
newData.dropna(inplace=True)
plt.plot(newData.index, newData['open'], color='red', label='Open')
plt.plot(newData.index, newData['high'], color='green', label='High')
plt.plot(newData.index, newData['low'], color='orange', label='Low')
plt.plot(newData.index, newData['last'], color='blue', label='Open')
plt.legend()
plt.show()


Y1.index = range(len(Y1))
Y1.dropna(inplace=True)
plt.plot(Y1.index, Y1['open'], color='red', label='Open', linewidth=1)
plt.plot(Y1.index, Y1['high'], color='green', label='High', linewidth=1)
plt.plot(Y1.index, Y1['low'], color='orange', label='Low', linewidth=1)
plt.plot(Y1.index, Y1['last'], color='blue', label='Last', linewidth=1)
plt.legend()
plt.show()
"""
