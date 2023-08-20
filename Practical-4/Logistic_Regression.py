#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv('iris-data.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


#Removing all null values row
df = df.dropna(subset=['petal_width_cm'])
df.info()


# In[7]:


#Plot


# In[8]:


sns.pairplot(df, hue='class', size=2.5)


# From the plots it can be observed that there is some abnormality in the class name. Let's explore further

# In[9]:


df['class'].value_counts()


# Two observations can be made from the above results
# - For 5 data points 'Iris-versicolor' has been specified as 'versicolor' 
# - For 1 data points, 'Iris-setosa' has been specified as 'Iris-setossa'

# In[10]:


df['class'].replace(["Iris-setossa","versicolor"], ["Iris-setosa","Iris-versicolor"], inplace=True)
df['class'].value_counts()


# # Simple Logistic Regression

# Consider only two class 'Iris-setosa' and 'Iris-versicolor'. Dropping all other class

# In[11]:


final_df = df[df['class'] != 'Iris-virginica']


# In[12]:


final_df.head()


# # Outlier Check

# In[13]:


sns.pairplot(final_df, hue='class', size=2.5)


# From the above plot, sepal_width and sepal_length seems to have outliers. To confirm let's plot them seperately

# SEPAL LENGTH

# In[14]:


final_df.hist(column = 'sepal_length_cm',bins=20, figsize=(10,5))


# It can be observed from the plot, that for 5 data points values are below 1 and they seem to be outliers. So, these data points
# are considered to be in 'm' and are converted to 'cm'.

# In[15]:


final_df.loc[final_df.sepal_length_cm < 1, ['sepal_length_cm']] = final_df['sepal_length_cm']*100
final_df.hist(column = 'sepal_length_cm',bins=20, figsize=(10,5))


# SEPAL WIDTH

# In[16]:


final_df = final_df.drop(final_df[(final_df['class'] == "Iris-setosa") & (final_df['sepal_width_cm'] < 2.5)].index)


# In[17]:


sns.pairplot(final_df, hue='class', size=2.5)


# Successfully removed outliers!!

# # Label Encoding

# In[18]:


final_df['class'].replace(["Iris-setosa","Iris-versicolor"], [1,0], inplace=True)


# In[19]:


final_df.head()


# # Model Construction

# In[20]:


inp_df = final_df.drop(final_df.columns[[4]], axis=1)
out_df = final_df.drop(final_df.columns[[0,1,2,3]], axis=1)
#
scaler = StandardScaler()
inp_df = scaler.fit_transform(inp_df)
#
X_train, X_test, y_train, y_test = train_test_split(inp_df, out_df, test_size=0.2, random_state=42)


# In[49]:


X_tr_arr = X_train
X_ts_arr = X_test
y_tr_arr = y_train.as_matrix()
y_ts_arr = y_test.as_matrix()
#print(y_tr_arr.shape)


# In[50]:


print('Input Shape', (X_tr_arr.shape))
print('Output Shape', X_ts_arr.shape)


# In[51]:


def weightInitialization(n_features):
    w = np.zeros((1,n_features))
    b = 0
    return w,b


# In[52]:


def sigmoid_activation(result):
    final_result = 1/(1+np.exp(-result))
    return final_result


# In[53]:


def model_optimize(w, b, X, Y):
    m = X.shape[0]
    
    #Prediction
    final_result = sigmoid_activation(np.dot(w,X.T)+b)
    Y_T = Y.T
    cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))
    #
    
    #Gradient calculation
    dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T))
    db = (1/m)*(np.sum(final_result-Y.T))
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost


# In[60]:


def model_predict(w, b, X, Y, learning_rate, no_iterations):
    costs = []
    for i in range(no_iterations):
        #
        grads, cost = model_optimize(w,b,X,Y)
        #
        dw = grads["dw"]
        db = grads["db"]
        #weight update
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)
        #
        
        if (i % 10 == 0):
            costs.append(cost)
            print("Cost after %i iteration is %f" %(i, cost))
    
    #final parameters
    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}
    
    return coeff, gradient, costs


# In[61]:


def predict(final_pred, m):
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1
    return y_pred


# In[66]:


#Get number of features
n_features = X_tr_arr.shape[1]
print('Number of Features', n_features)
w, b = weightInitialization(n_features)
#Gradient Descent
coeff, gradient, costs = model_predict(w, b, X_tr_arr, y_tr_arr, learning_rate=0.0001,no_iterations=4500)
#Final prediction
w = coeff["w"]
b = coeff["b"]
print('Optimized weights', w)
print('Optimized intercept',b)
#
final_train_pred = sigmoid_activation(np.dot(w,X_tr_arr.T)+b)
final_test_pred = sigmoid_activation(np.dot(w,X_ts_arr.T)+b)
#
m_tr =  X_tr_arr.shape[0]
m_ts =  X_ts_arr.shape[0]
#
y_tr_pred = predict(final_train_pred, m_tr)
print('Training Accuracy',accuracy_score(y_tr_pred.T, y_tr_arr))
#
y_ts_pred = predict(final_test_pred, m_ts)
print('Test Accuracy',accuracy_score(y_ts_pred.T, y_ts_arr))


# In[67]:


plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (every 10th)')
plt.title('Cost reduction over time')
plt.show()


# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


clf = LogisticRegression()


# In[32]:


clf.fit(X_tr_arr, y_tr_arr)


# In[33]:


print (clf.intercept_, clf.coef_)


# In[34]:


pred = clf.predict(X_ts_arr)


# In[35]:


print ('Accuracy from sk-learn: {0}'.format(clf.score(X_ts_arr, y_ts_arr)))


# In[ ]:




