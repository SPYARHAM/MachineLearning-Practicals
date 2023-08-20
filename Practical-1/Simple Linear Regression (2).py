#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression

# ### Contents

# 
# 1. Understanding the data
# 2. Reading the data
# 3. Data Exploration
# 4. Simple Regression Model
# 5. Model Evaluation

# #### Importing the required packages

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ### Understanding the data

# FuelConsumption.csv :
# 
# We have downloaded a fuel consumption dataset, FuelConsumption.csv , which contains model-specific fuel consumption ratings and estimatedcarbon dioxide emissions for new light-duty vehicles for retail sale in Canada.
# 
# Data Source: https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64
# 
# MODELYEAR e.g. 2014
# 
# MAKE e.g. Acura
# 
# MODEL e.g. ILX
# 
# VEHICLE CLASS e.g. SUV
# 
# ENGINE SIZE e.g. 4.7
# 
# CYLINDERS e.g 6
# 
# TRANSMISSION e.g. A6
# 
# 

# #### Reading the Data

# In[2]:


df = pd.read_csv("FuelConsumption.csv")
# take a look at the dataset
df.head()


# Data Exploration

# In[3]:


# summarize the data
df.describe()


# In[4]:


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] 
cdf.head(10)


# Lets plot these features

# In[5]:


data1 = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']] 
data1.hist()
plt.show()


# Now, lets plot each of these features vs the Emission, to see how linear is their relation:

# In[6]:


plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue') 
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()


# In[7]:


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# Practice
# plot CYLINDER vs the Emission, to see how linear is their relation:

# In[8]:


#write your code here


# Creating train and test dataset

# Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set. This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that have been used to train the data. It is more realistic for real world problems.
# 
# This means that we know the outcome of each data point in this dataset, making it great to test with! And since this data has not been used to train the model, the model has no knowledge of the outcome of these data points. So, in essence, it is truly an out-of-sample testing.
# 
# Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% for testing. We create a mask to select random rows using np.random.rand() function:

# In[9]:


msk = np.random.rand(len(df))<0.8
train = cdf[msk]
test = cdf[~msk]


# Simple Regression Model

# Linear Regression fits a linear model with coefficients θ = (θ1, . . . , θn) to minimize the 'residual sum of squares' between the independent x in the dataset, and the dependent y by the linear approximation.

# To see training data distribution

# In[10]:


# write your code here


# In[11]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.array(train[['ENGINESIZE']])
train_y = np.array(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# As mentioned before, Coefficient and Intercept in the simple linear regression, are the parameters of the fit line. Given that it is a simple linear
# regression, with only 2 parameters, and knowing that the parameters are the intercept and slope of the line, sklearn can estimate them directly from
# our data. 
# Notice that all of the data must be available to traverse and calculate the parameters.

# Plot the outputs

# In[14]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue') 
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0],'-y') 
plt.xlabel("Engine size")
plt.ylabel("Emission")


# Model Evaluation

# we compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.
# 
# There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set:
# 
# Mean absolute error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.
# 
# Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean absolute error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
# 
# Root Mean Squared Error (RMSE): This is the square root of the Mean Square Error.
# 
# R-squared is not error, but is a popular metric for accuracy of your model. It represents how close the data are to the fitted regression line. The
# higher the R-squared, the better the model fits your data. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).

# In[15]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f"% np.mean(np.absolute(test_y_hat - test_y))) 
print("Residual sum of squares (MSE): %.2f"% np.mean((test_y_hat - test_y)**2)) 
print("R2-score: %.2f"% r2_score(test_y_hat , test_y))


# Example 2

# Define the problem

# In[27]:


import pandas as pd
import numpy as np
XX =[[100,1,1,1],[200,2,2,1],[300,2,2,1]]
X = np.array(XX)
YY =[5,10,9]
Y = np.array(YY)
X


# Prediction

# In[17]:


theta0 =1
theta1 =1
yhat1 =(theta0 +(theta1*X[0][0]))

yhat2 =(theta0 +(theta1*X[1][0]))

yhat3 =(theta0 +(theta1*X[2][0]))

yhat =[yhat1, yhat2, yhat3]

yhat


# Mean Squared Error

# In[18]:


Error1 = yhat1 - Y[0]

Error2 = yhat2 - Y[1]

Error3 = yhat3 - Y[2]

Error3


# In[19]:


Error =(Error1**2)+(Error2**2)+(Error3**2)
Errormean =1/3* Error

Errormean


# Plot 1: theta0 = 0, theta1 = 1 (Intercept = 0 and slope = 1)

# In[29]:


theta0 =0
theta1 =1
yhat1 =(theta0 +(theta1*X[0][0]))

yhat2 =(theta0 +(theta1*X[1][0]))

yhat3 =(theta0 +(theta1*X[2][0]))
yhat =[yhat1, yhat2, yhat3]
#yhat = [20, 40, 60]
plt.figure(figsize=(5,5))
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.scatter(X[:,0], Y, color='blue')
plt.plot(X[:,0], yhat,'-r')
plt.xlabel("size")
plt.ylabel("price")
print ('Yhat: ', yhat)


# Plot2 theta0 = 1, theta1 = 0

# In[22]:


theta0 =1
theta1 =0
yhat1 =(theta0 +(theta1*X[0][0]))

yhat2 =(theta0 +(theta1*X[1][0]))

yhat3 =(theta0 +(theta1*X[2][0]))
yhat =[yhat1, yhat2, yhat3]

plt.figure(figsize=(5,5))
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.scatter(X[:,0], Y, color='blue')
plt.plot(X[:,0], yhat,'-r')
plt.xlabel("size")
plt.ylabel("price")
print ('Yhat: ', yhat)


# Plot3: theta0 = 0.5, theta1= 0.1

# In[24]:


theta0 =0.5
theta1 =0.1
yhat1 =(theta0 +(theta1*X[0][0]))

yhat2 =(theta0 +(theta1*X[1][0]))

yhat3 =(theta0 +(theta1*X[2][0]))
yhat =[yhat1, yhat2, yhat3]

plt.figure(figsize=(5,5))
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.scatter(X[:,0], Y, color='blue')
plt.plot(X[:,0], yhat,'-r')
plt.xlabel("size")
plt.ylabel("price")
print ('Yhat: ', yhat)


# In[26]:


plt.figure(figsize=(6,6))
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.scatter(X[:,0], Y, color='blue')
plt.plot(X[:,0], yhat,'-r')
plt.xlabel("size")
plt.ylabel("price")


# #### Example 3: Linear regression on Benglore house price data

# In[43]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


# In[44]:


df = pd.read_csv("Bengaluru_House_Data.csv")

# take a look at the dataset

df.head()


# In[45]:


df.dropna()


# In[46]:


df.fillna(value='FILL VALUE')
df['bath'].fillna(value=df['bath'].mean())
#df['balcony'].fillna(value=df['balcony'].mean())
df.head()


# In[47]:


# summarize the data
df.describe()


# In[48]:


cdf = df[['size','bath','balcony','price']]
cdf.dropna(axis=1)
cdf.head(9)


# In[49]:


# Write your code here for scatter plot


# In[50]:


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# #### train data distribution

# In[51]:


# Write your code here


# ##### Dropping Nan
# 

# In[52]:


cdf = df.dropna()
cdf
plt.scatter(cdf.bath, cdf.price, color='blue')
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()


# In[54]:


msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# In[55]:


plt.scatter(train.bath, train.price, color='blue')
plt.xlabel("size")
plt.ylabel("price")
plt.show()


# In[56]:


# regression model
#Write your code here for size vs price
from sklearn import linear_model


# In[ ]:




