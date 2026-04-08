# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:10:39 2024

@author: hannah
"""
import matlab.engine
import time
import pandas as pd
en = matlab.engine.start_matlab()
ch=en.open_can_channel()
df=en.get_can_data()
i=0
try:
    while True:
        i+=1
        df= df.append(df,en.get_can_data())
        time.sleep(0.1)
        print(i,"\n", df)
except KeyboardInterrupt:
    print('interrupted!')
print(df)
en.close_can_channel()