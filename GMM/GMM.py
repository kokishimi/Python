# -*- coding: utf-8 -*-
"""
Created on Wed May  8 19:11:45 2019

@author: kokis
"""

# 1：ライブラリのインポート--------------------------------
import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn import mixture #機械学習用のライブラリを利用

# 2：Wineのデータセットを読み込む--------------------------------
df_sp_all=pd.read_csv('2017Omiyasp0_6.csv')
#品種(0列、1～3)と色（10列）とプロリンの量(13列)を使用する
df_sp=df_sp_all['Speed']

# 3：データの整形-------------------------------------------------------
#sp_mean=df_sp.mean()
#sp_std=df_sp.std()
#sp_norm=(df_sp-sp_mean)/sp_std
#sp_norm2=sp_norm.values.reshape(-1, 1)

sp=df_sp.values.reshape(-1, 1)

# 解説6：GMMを実施---------------------------------
n_max=8
#vaty=['spherical','tied','diag','full']
vaty=['spherical']
bar_num=n_max*len(vaty)
plt_n=bar_num*2+2

# 4：プロットしてみる------------------------------------------
plt.figure(figsize=[6.4, plt_n*2])
plt.subplot(plt_n, 1, 1)
#plt.hist(sp_norm,bins=1000)
plt.hist(sp,bins=1000)

for m in range(n_max):
    for n in range(len(vaty)):
        gmm=mixture.GaussianMixture(n_components=m+1,covariance_type=vaty[n])
        
        #z_gmm=gmm.fit(sp_norm2)
        #z_gmm=z_gmm.predict(sp_norm2)

        z_gmm=gmm.fit(sp)
        z_gmm=z_gmm.predict(sp)



        # 7: 結果をプロット-----------------------------------------------
        plt.subplot(plt_n, 1, 2*len(vaty)*m+2*n+2)
        #plt.scatter(sp_norm,z_gmm)
        plt.scatter(sp,z_gmm)

        # 結果を表示
        print('n,vartype:')
        print(m+1,vaty[n])
        
        print("*** weights")
        print(gmm.weights_)

        print("*** means")
        print(gmm.means_)

        print("*** covars")
        print(gmm.covariances_)

        # 各ガウス分布について

        gd_all=[]
        gd_buf=[]
        for k in range(m+1):
            if n==1:
                gd_buf = np.random.normal(gmm.means_[k][0], np.sqrt(gmm.covariances_.flatten()[0]), 1000)
            else:
                gd_buf = np.random.normal(gmm.means_[k][0], np.sqrt(gmm.covariances_.flatten()[k]), 1000)
            
            if k==0:
                gd=gd_buf
            else:
                gd=np.vstack((gd, gd_buf))
                
            if k==0:
                gd_all=gd.tolist()*int(gmm.weights_[k]*100)
            else:
                gd_all=gd_all+gd[k].tolist()*int(gmm.weights_[k]*100)
        

        plt.subplot(plt_n, 1, 2*len(vaty)*m+2*n+3)
        #plt.hist(gd_all,bins=1000,range=(sp_norm.min(),sp_norm.max()))

        plt.hist(gd_all,bins=1000,range=(sp.min(),sp.max()))
        # メッシュ上の各点での対数尤度の等高線を描画
        #print(gmm.bic(sp_norm2))

        print(gmm.bic(sp))
        if n==0:
            bic_r=gmm.bic(sp)
        else:
            bic_r=np.vstack((bic_r, gmm.bic(sp)))
        
        print('m,n:',m,',',n)

        #clsp_mean=gmm.means_+sp_mean
        #print(clsp_mean)
        #clsp_std=np.sqrt(gmm.covariances_)*sp_std
        #print(clsp_std)

        spmax=0.0
        min_mean=gmm.means_[0]
        min_num=0
        for i in range(m+1):
            if(min_mean>gmm.means_[i]):
                min_mean=gmm.means_[i]
                min_num=i
                
        for i in range(len(sp)):
            if(z_gmm[i]==min_num):
                if(sp[i]>spmax):
                    spmax=sp[i]
        
        
        if(m>0):
            sp2max=0.0
            if(min_num==0):
                min2_mean=gmm.means_[1]
                min2_num=1
            else:
                min2_mean=gmm.means_[0]
                min2_num=0
        
            for i in range(m+1):
                if(min_num!=i):
                    if(min2_mean>gmm.means_[i]):
                        min2_mean=gmm.means_[i]
                        min2_num=i
        
            for i in range(len(sp)):
                if(z_gmm[i]==min2_num):
                    if(sp[i]>sp2max):
                        sp2max=sp[i]
                
    if m==0:
        bic=bic_r
        spmax2=spmax
        sp2max2=0.0
    else:
        bic=np.vstack((bic,bic_r))
        spmax2=np.vstack((spmax2,spmax))
        sp2max2=np.vstack((sp2max2,sp2max))
        
plt.subplot(plt_n,1,2*len(vaty)*n_max+2)
for i in range(n_max):
    for j in range(len(vaty)):
        if j==0:
            name_r=vaty[0]+str(i+1)
        else:
            name_r=np.insert(name_r,j,vaty[j]+str(i+1))
    if i==0:
        name=name_r
    else:
        name=np.vstack((name,name_r))

name2=name.flatten()
bic2=bic.flatten()
    
plt.bar(name2,bic2)
plt.show()

print('bic')
print(bic)
print('maxsp')
print(spmax2)
print('second maxsp')
print(sp2max2)

    
    