#!/usr/bin/env python
# coding: utf-8

# #### OR gate

# In[ ]:


# OR gate
import numpy as np
# Define input
X = np.array([[0,0],[0,1],[1,0],[1,1]])

#Define output features
Y = np.array([[0,1,1,1]])

# Reshape target into vector
Y = Y.reshape(4,1)

# Select random weights w1, w2
W = np.array([[10],[10]])
# bias b
b = -5

z = np.dot(X,W)+b
sig = 1/(1 + np.exp(-z))
print('Sigmoid \n:', sig)

yhat1 = 0
yhat = []
for i in range(len(Y)):
    if (sig[i] < 0.5):
        yhat1 = 0
    else:
        yhat1 = 1
        yhat.append(yhat1)
print('\n Yhat:', yhat)

