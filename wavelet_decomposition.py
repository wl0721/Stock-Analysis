import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
import os
import math
from load_md import get_md_by_tick as gmt

df = gmt(20190430,'last',dtype='float32',fillna='ffill',codes=['000001'])
data = df.values.T[0]
#print(type(data))  #4802 <class 'numpy.ndarray'>

x = [i for i in range(len(data))]
#print(np.array(x).shape)  #(4801,)

index_list = data[:-10]
day_list1 = np.array(x)[:-10]
#print(day_list1.shape)  #(4791,)

index_for_predict = data[-10:]
day_list2 = np.array(x)[-10:]
#print(index_list.shape)  #(4791,1)
#print(index_for_predict.shape)  #(10.1)

A3, D3, D2, D1 = pywt.wavedec(index_list, 'db4', mode='sym', level=3) #wavelet decomposition
coeff = [A3, D3, D2, D1]
#print(coeff)

order_A3 = sm.tsa.arma_order_select_ic(A3,ic='aic')['aic_min_order']
order_D3 = sm.tsa.arma_order_select_ic(D3,ic='aic')['aic_min_order']
order_D2 = sm.tsa.arma_order_select_ic(D2,ic='aic')['aic_min_order']
order_D1 = sm.tsa.arma_order_select_ic(D1,ic='aic')['aic_min_order']

print("order_A3, order_D3, order_D2, order_D1: ", order_A3, order_D3, order_D2, order_D1)
#order_A3:(4,2); order_D3:(4,2); order_D2:(4,1); order_D1:(0,1)

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
#plt.savefig('figure_1.jpg')

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
#plt.savefig('figure_2.jpg')

temp_data_wt = {'real_value':index_for_predict, 'pre_value_wt':denoised_index[-10:],
                'err_wt':denoised_index[-10:]-index_for_predict,
                'err_rate_wt/%':(denoised_index[-10:]-index_for_predict)/index_for_predict*100}
predict_wt = pd.DataFrame(temp_data_wt,index=day_list2,columns=['real_value','pre_value_wt','err_wt','err_rate_wt/%'])
print("predict_wt: \n", predict_wt)
