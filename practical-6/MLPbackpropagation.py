#!/usr/bin/env python
# coding: utf-8

# Multilayer Perceptron using Backpropagation

# In[2]:


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)
clf = MLPClassifier(random_state=1, max_iter=300)
clf.fit(X_train, y_train)


# In[3]:


clf.predict_proba(X_test[:1])


# In[4]:


clf.predict(X_test[:5, :])


# In[5]:


clf.score(X_test, y_test)


# In[6]:


X1=[[0, 0],[0, 1], [1, 0], [1, 1]]


# In[7]:


print(X1)


# In[8]:


y1=[0,1,1,0]


# In[9]:


print(y1)


# In[10]:


clf = MLPClassifier(random_state=1, max_iter=100)


# In[12]:


clf.fit(X1,y1)


# In[13]:


clf.predict(X1)


# In[ ]:




