# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:16:13 2019

@author: Administrator
"""

import pandas as pd 
import os
import re

path = '/data/stock/newSystemData/rawdata/wind/stock_eod'

def get_deal_day_list_in_period(start, end):
    raw_date_list = []
    new_date_list = []

    for fpathe, _, fs in os.walk(path):
        for f in fs:
            tmp_path = os.path.join(fpathe,f)
            pattern = re.compile(r'20[0-1][0-9]+(\.[a-z]+)$')
            date = re.search(pattern,tmp_path)
            if date != None:
                date_value = date.group().split('.csv')
                date_value = date_value[0]
                raw_date_list.append(date_value)
    raw_date_list.sort()

    date_list = [x.strftime('%Y%m%d') for x in list(pd.date_range(start=start, end=end))]
    
    for day in date_list:
        if day in raw_date_list:
            new_date_list.append(day)

    return new_date_list

# print(get_deal_day_list_in_period('20160101', '20160131'))