import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/sn/Desktop/housing.csv")
data.dtypes

k={'ocean_proximity':{'NEAR BAY':1,'<1H OCEAN':2,'INLAND':3,'NEAR OCEAN':4,'ISLAND':5}}
data=data.replace(k)
data=data.replace(np.nan,0)
x=data.loc[:,data.columns!='median_income'].values
y=data['median_income'].values
x.shape

# Analitical solution
N=x.shape
x=x.reshape(N[0],9)
x1=np.concatenate((x,np.ones((N[0],1))),axis=1)
a=np.linalg.inv(np.dot(np.transpose(x1),x1))
b=np.dot(a,np.transpose(x1))
w=np.dot(b,y)
y_hat=np.dot(x1,w)
mse1=np.sum((y-y_hat)**2)/N[0]
print(mse1)
print(w)

# gradiend descent
w_0=np.random.rand(10,1)/1000
epoch=10
learning_rate=0.00000000001
cost=[]
for i in range(epoch):
  dj_dw=2/N[0]*np.matmul(np.transpose(x1),np.matmul(x1,w_0)-y)
  w_new=w_0-learning_rate*dj_dw
  y_hat=np.matmul(x1,w_new)
  mse=np.sum((y-y_hat)**2)/N[0]
  print(mse)
  cost.append(mse)
  w_0=w_new
print(mse)
print(w_new)

plt.plot(cost)