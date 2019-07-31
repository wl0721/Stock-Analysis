# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:15:20 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:01:44 2019

@author: User
"""

tk_digit = 1000
ti_zero = 9_30_00_000
ss_gap = 3

def _tick_to_sec(tk):
    hhmmss = tk // tk_digit
    hh = hhmmss // 10000
    mm = hhmmss % 10000 // 100
    ss = hhmmss % 100
    return hh * 3600 + mm *60 + ss

def _sec_to_tick(sec):
    hh = sec // 3600
    mmss = sec % 3600
    mm = mmss // 60
    ss = mmss % 60
    hhmmss = hh * 10000 + mm * 100 + ss
    return int(hhmmss * tk_digit)

base_sec = _tick_to_sec(ti_zero)

_sec_to_ti = lambda sec: int((sec - base_sec)/ss_gap)

_ti_to_sec = lambda ti: base_sec + ti*ss_gap

def ticks_to_ti(ticks = None, ti=None):
    if ti is None:
        assert ticks is not None, 'one of ticks/ti must no be None'
        return [_sec_to_ti(_tick_to_sec(tk)) for tk in ticks]
    elif ticks is None:
        assert ti is not None, 'one of ticks/ti must no be None'
        return [_sec_to_tick(_ti_to_sec(t)) for t in ti]
