#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data = pd.read_csv('newdataset.csv',header=0,names=['Startup_Name','Industry_Vertical','R&D_Spend','Administration','Marketing_Spend','State','Amount_in_USD'])


# In[3]:


data


# In[4]:


data.drop(columns='Startup_Name',inplace=True)


# In[5]:


data


# In[6]:


data=data[['Industry_Vertical','State','R&D_Spend','Administration','Marketing_Spend','Amount_in_USD']]


# In[7]:


data


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


data.Industry_Vertical.nunique()


# In[10]:


'''Percentage of Missing records'''

data.isnull().sum()/len(data)


# In[11]:


data.Industry_Vertical.value_counts().sort_values(ascending=False).head(10)


# In[12]:


replace_values = {'eCommerce' : 'ecommerce', 'ECommerce':'ecommerce', 'E-Commerce' : 'ecommerce','ecommmerce':'ecommerce', 'Ecommerce':'ecommerce','Health Care': 'Healthcare', 'HealthCare': 'Healthcare', 'IT': 'Information Technology'}     


# In[13]:


data = data.replace({"Industry_Vertical": replace_values})      


# In[14]:


data.Industry_Vertical.nunique()


# In[15]:


industryVerticals = ['Consumer Internet', 'Technology', 'ecommerce', 'Healthcare', 'Finance', 'Logistics', 'Food and Beverage', 'Education', 'Ed-Tech', 'Information Technology']

data = data[data['Industry_Vertical'].isin(industryVerticals)]


# In[16]:


data


# In[17]:


data.dtypes


# In[18]:


data['Industry_Vertical'] = data['Industry_Vertical'].astype('category')


# In[19]:


data['State'] = data['State'].astype('category')


# In[20]:


data['R&D_Spend'] = data['R&D_Spend'].astype('int')


# In[21]:


data['Administration'] = data['Administration'].astype('int')


# In[22]:


data['Marketing_Spend'] = data['Marketing_Spend'].astype('int')


# In[23]:


data['Amount_in_USD'] = data['Amount_in_USD'].astype('int')


# In[24]:


data.dtypes


# In[25]:


X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# In[26]:


print(X)


# In[27]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), [0,1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[28]:


print(X)


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[30]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[31]:


y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[32]:


from sklearn import metrics


# In[33]:


print("MSE:",metrics.mean_squared_error(y_test,y_pred))


# In[34]:


print("RMSE",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[35]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


# In[36]:


df1 = df.head(50)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[37]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 


# In[38]:


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


# In[41]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[42]:


print(X_test)


# In[50]:


regressor.predict(np.array([[1.0 ,0.0 ,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,60000,38829,482947]]))


# In[53]:


regressor.predict(np.array([[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,131876,99814,362861]]))


# In[70]:


regressor.score(X_test,y_test)


# In[71]:


regressor.intercept_


# In[72]:


regressor.coef_


# In[73]:


print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[64]:


y_true = [108733, 110352, 96778, 155752]
y_predi = [109464.056345,115961.114306 ,96015.963218 ,163852.919093 ]


# In[66]:


print(metrics.mean_absolute_error(y_true, y_predi))
print(metrics.mean_squared_error(y_true, y_predi))
print(np.sqrt(metrics.mean_squared_error(y_true, y_predi)))


# In[69]:


print(regressor.score(X_test, y_test),r2_score(y_test,y_pred))


# In[68]:


from sklearn.metrics import r2_score


# In[ ]:




