# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:32:42 2019

@author: kokis
"""

import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import csv

# 2：Wineのデータセットを読み込む--------------------------------
df_sp_all=pd.read_csv('2017Omiyasp0_6.csv')
#品種(0列、1～3)と色（10列）とプロリンの量(13列)を使用する
df_sp=np.array(df_sp_all['Speed'])

bar_n=6*10
num_p=np.zeros(bar_n)

for row in range(len(df_sp)):
    for i in range(bar_n):
        if(df_sp[row]>i*0.1 and df_sp[row]<=(i+1)*0.1):
            num_p[i]+=1
            break
        
with open('for_hist_6.csv', 'w') as f:
    writer = csv.writer(f)
    num_p2 = num_p.reshape(-1,1)
    
    writer.writerows(num_p2)
    