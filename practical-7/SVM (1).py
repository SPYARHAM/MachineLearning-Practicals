#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


# In[28]:


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features. 
y = iris.target


# In[29]:


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=1,gamma=1).fit(X, y)


# In[30]:


expected = iris.target
predicted = svc.predict(X)


# In[5]:


from sklearn import metrics

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[33]:


x1, y1 = np.meshgrid(np.arange(2, 4, 0.2),
                     np.arange(3, 5, 0.2))
print(x1)
print(y1)
#print()


# In[31]:


# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
print(x_min)
print(x_max)
print(y_min)
print(y_max)
h = (x_max / x_min)/100
print(h)
print(np.arange(x_min, x_max, h))

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
print(xx)
print(yy)


# In[36]:


plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
print(Z)
plt.contourf(xx, yy, Z, alpha=1)


# In[10]:


plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('SVC with linear kernel')
plt.show()


# In[16]:


plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()


# ### Example 2

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x = np.linspace(-5, 5, 50)
print(x)
y = np.sqrt(50 - x**2)
print(y)
y = np.hstack([y,-y])
print(y)
x = np.hstack([x,-x])
x1 = np.linspace(-5, 5, 50)
y1 = np.sqrt(25 - x1**2)
y1 = np.hstack([y1,-y1])
x1 = np.hstack([x1,-x1])
plt.scatter(x,y)
plt.scatter(x1,y1)


# In[18]:


df1 = pd.DataFrame(np.vstack([y,x]).T, columns =['x1','x2'] )
df1['Y'] = 0
df2 = pd.DataFrame(np.vstack([y1,x1]).T, columns =['x1','x2'] )
df2['Y'] = 1
df = df1.append(df2)
df.head(5)


# In[19]:


X = df.iloc[:,:2]
y = df.Y
y


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.25, random_state=0)


# In[21]:


#Import svm model
from sklearn import svm
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[22]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# ## Polynomial Kernel

# In[23]:


#polynomial kernel
df['x1square']= df['x1']**2
df['x2square']= df['x2']**2
df['x1*x2']= df['x1']*df['x2']
df.head(5)


# In[24]:


X = df[['x1','x2','x1square','x2square','x1*x2']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.25, random_state=0)
X_train


# In[25]:


import plotly.express as px
fig = px.scatter_3d(df, x='x1', y = 'x2', z='x1*x2',color='Y')
fig.show()


# In[20]:


import plotly.express as px
fig = px.scatter_3d(df, x='x1square', y = 'x2square', z='x1*x2',color='Y')
fig.show()


# ## Assignment
# 

# #### Implement SVM using polynomial kernel of cubic root instead of square root and observe the difference fig.show()
# 

# In[ ]:




