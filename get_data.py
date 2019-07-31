from load_md import get_md_by_tick as gmt
import pandas as pd
from IndexData import get_deal_day_list_in_period

tag_list = ['high', 'low', 'last', 'ask1', 'bid1', 'ask_vol1', 'bid_vol1',
        'amount', 'trade', 'avebid', 'aveoff', 'totbid', 'totoff', 'vol']
start_day = '20160101'
end_day = '20160131'
day_list = get_deal_day_list_in_period(start_day, end_day)

# 得到000001股票在2016年1月所有股票交易日的tick数据，并将数据保存为.csv文件
def get_data():
    X = []
    for day in day_list:
        Y = []
        for tag in tag_list:
            if tag == 'amount' or tag == 'totbid' or tag == 'totoff' or tag == 'vol':
                df = gmt(int(day), tag, dtype='int64', codes=['000001'])
            else:
                df = gmt(int(day), tag, dtype='float32', codes=['000001'])
            df = df.rename(columns={'000001':tag})
            
            Y.append(df)
        D = pd.concat(Y, axis=1)
        X.append(D)
        data = pd.concat(X, axis=0)
    data.index = range(len(day_list) * 4802) # 将data的行索引值改成len(day_list)*4802
    print(data.head())
    data.to_csv('/home/wanglin/Documents/get_data/label/data_201601.csv')
    return data
