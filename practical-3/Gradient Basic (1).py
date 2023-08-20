#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:


import pandas as pd
import numpy as np
XX = [[100,1,1,1], [200, 2,2,1],[300,2,2,1]]
X = np.array(XX)
YY = [5,10,9]
Y = np.array(YY)
X[:,0]
X


# #### Prediction Yhat

# In[3]:


theta0 = 0.5 #Plot3 theta0 = 1, theta1 = 0
theta1 = 0.1
yhat=[]
for i in range(len(X)):
 yh=theta0 + (theta1*X[i][0])
 yhat.append(yh)
 i+=1
print(yhat)
plt.figure(figsize=(6, 6))
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.scatter(X[:,0], Y, color='blue')
plt.plot(X[:,0], yhat, '-r')
plt.xlabel("size")
plt.ylabel("price")
print ('Yhat: ', yhat)


# #### Mean square error

# In[4]:


#Mean Square Error
MSE=0
RSS1=0
Error=[]
for i in range(len(Y)):
 yhat=theta0 + (theta1*X[i][0])
 Error1= yhat - Y[i]
 Error.append(Error1)
 i+=1
 RSS1=RSS1+Error1**2 #Sum of squares of errors


print(Error)

MSE=RSS1/len(Y) #mean square error
print("Mean square error is", MSE)


# In[ ]:




