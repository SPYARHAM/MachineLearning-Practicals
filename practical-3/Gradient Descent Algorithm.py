#!/usr/bin/env python
# coding: utf-8

# ### Gradient Descent Algorithm

# Cost function can be mathematically defined as:
# 
# 𝑦=𝛽+θnXn, 
# where x is the parameters(can go from 1 to n), 𝛽 is the bias and θ is the weight

# #### Advertising Dataset

# This is a dataset that gives us the total sales for different products, after marketing them on Television, Radio and Newspaper. Using our algorithm, we can find out which medium performs the best for our sales and assign weights to all the mediums accordingly. This dataset can be downloaded from the link given below:
# 
# https://www.kaggle.com/sazid28/advertising.csv

# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[51]:


df=pd.read_csv('Advertising.csv')
df.head()


# #### Data Normalization

# In[52]:


X=df[['TV','radio','newspaper']]
Y=df['sales']
Y=np.array((Y-Y.mean())/Y.std())
X=X.apply(lambda rec:(rec-rec.mean())/rec.std(),axis=0)
X


# To implement a gradient descent algorithm we need to follow 4 steps:
# 
# 1. Randomly initialize the bias and the weight theta
# 2. Calculate predicted value of y that is Y given the bias and the weight
# 3. Calculate the cost function from predicted and actual values of Y
# 4. Calculate gradient and the weights

# In[53]:


# Function to initialize random values for bias and theta
import random
def initialize(dim):
    b=random.random()
    theta=np.random.rand(dim)
    return b,theta
b,theta=initialize(3)
print( "Bias:",b,"Weights:",theta)


# In[55]:


def predict_Y(b,theta,X):
    return b + np.dot(X,theta)
Y_hat=predict_Y(b,theta,X)
Y_hat[0:20]
#Y_hat


# In[49]:


import math
def get_cost(Y,Y_hat):
    Y_resd=Y-Y_hat
    return np.sum(np.dot(Y_resd.T,Y_resd))/(len(Y))
Y_hat=predict_Y(b,theta,X)
get_cost(Y,Y_hat)


# To get the updated bias and weights we use the gradient descent formula of:
# 
# The parameters passed to the function are
# 
# x,y : the input and output variable
# 
# y_hat: predicted value with current bias and weights
# 
# b_0,theta_0: current bias and weights
# 
# Learning rate: learning rate to adjust the update step

# In[63]:


def update_theta(x,y,y_hat,b_0,theta_o,learning_rate):
    db=(np.sum(y_hat-y)*2)/len(y)
    #dw=(np.dot((y_hat-y),x)*2)/len(y)
    dw=np.dot((y_hat-y),x)
    dw1=(np.sum(dw)*2)/len(y)
    b_1=b_0-learning_rate*db
    theta_1=theta_o-learning_rate*dw1
    return b_1,theta_1
print("After initialization -Bias: ",b,"theta: ",theta)
Y_hat=predict_Y(b,theta,X)
b,theta=update_theta(X,Y,Y_hat,b,theta,0.001)
print("After first update -Bias: ",b,"theta: ",theta)
get_cost(Y,Y_hat)


# In[70]:


def run_gradient_descent(X,Y,alpha,num_iterations):
    b,theta=initialize(X.shape[1])
    iter_num=0
    gd_iterations_df=pd.DataFrame(columns=['iteration','cost'])
    result_idx=0
    for each_iter in range(num_iterations):
        Y_hat=predict_Y(b,theta,X)
        this_cost=get_cost(Y,Y_hat)
        prev_b=b
        prev_theta=theta
        b,theta=update_theta(X,Y,Y_hat,prev_b,prev_theta,alpha)
        if(iter_num%10==0):
            gd_iterations_df.loc[result_idx]=[iter_num,this_cost]
        result_idx=result_idx+1
        iter_num +=1
    print("Final Estimate of b and theta :",b,theta)
    return gd_iterations_df,b,theta
gd_iterations_df,b,theta=run_gradient_descent(X,Y,alpha=0.001,num_iterations=400)




# In[71]:


gd_iterations_df[0:10]


# In[69]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(gd_iterations_df['iteration'],gd_iterations_df['cost'])
plt.xlabel("Number of iterations")
plt.ylabel("Cost or MSE")


# In[ ]:




