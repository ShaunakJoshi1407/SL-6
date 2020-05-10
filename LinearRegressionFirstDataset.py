#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''Importing Libraries'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

pd.set_option('display.max_rows',3000)


# In[2]:



startup_data=pd.read_csv('startup_funding_modified.csv',header=0,encoding='utf8',names=['Sr_No','Startup_Date','Startup_Name','Industry_Vertical','SubVertical','City_Location','Investor_Name','Investment_Type','Amount_in_USD','Remarks'])

startup_data.drop(columns='Remarks',inplace=True)


# In[3]:


startup_data


# In[359]:


startup_data["Startup_Date"]= pd.to_datetime(startup_data["Startup_Date"])


# In[360]:


startup_data


# In[361]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


city_data=startup_data.City_Location

replacement_dictionary={"USA":np.NaN,"US":np.NaN,"US/India":np.NaN,"Dallas":np.NaN,"Boston":np.NaN,"New York":np.NaN,"London":np.NaN,"Nairobi":np.NaN,"SFO / Karnataka":np.NaN,"Seattle":np.NaN,"San Francisco":np.NaN,"Santa Monica":np.NaN,"Missourie":np.NaN,"India":np.NaN,"Palo Alto":np.NaN,"California":np.NaN,"SFO / Bangalore":np.NaN,"Burnsville":np.NaN,"Singapore":np.NaN,"Bangalore":"Karnataka","Belgaum":"Karnataka","Udupi":"Karnataka","New Delhi":"NCT of Delhi","Gurgaon":"NCT of Delhi","Noida":"NCT of Delhi","Delhi":"NCT of Delhi","Gurugram":"NCT of Delhi","Mumbai":"Maharashtra","Pune":"Maharashtra","Nagpur":"Maharashtra","Hyderabad":"Telangana","Faridabad":"Haryana","Surat":"Gujarat","Ahmedabad":"Gujarat","Ahemdabad":"Gujarat","Ahemadabad":"Gujarat","Vadodara":"Gujarat","Jaipur":"Rajasthan","Jodhpur":"Rajasthan","Udaipur":"Rajasthan","Kolkata":"West Bengal","Kolkatta":"West Bengal","Indore":"Madhya Pradesh","Bhopal":"Madhya Pradesh","Gwalior":"Madhya Pradesh","Coimbatore":"Tamil Nadu","Karur":"Tamil Nadu","Hubli":"Tamil Nadu","Panji":"Goa","Panaji":"Goa","Kozhikode":"Kerala","Trivandrum":"Kerala","Agra":"Uttar Pradesh","Kanpur":"Uttar Pradesh","Lucknow":"Uttar Pradesh","Varanasi":"Uttar Pradesh","Gaya":"Bihar","Bhubneswar":"Odisha","Bhubaneswar":"Odisha","New New Delhi":"NCT of Delhi","Nw Delhi":"NCT of Delhi","Kochi":"Kerala","Srinagar":"Jammu & Kashmir","Rourkela":"Bihar","Siliguri":"West Bengal"}

city_data=startup_data.City_Location.replace(replacement_dictionary)

startup_data['New_City_Location']=city_data


# In[9]:


# startup_data.New_City_Location.value_counts(dropna=False)
startup_data.dropna(subset=['New_City_Location'],inplace=True)
startup_data.isna().sum()
# new_startup_data=startup_data.loc[:,['New_City_Location']]


# In[28]:


startup_data.Amount_in_USD=startup_data.Amount_in_USD.replace(r'[,]','',regex=True)
# startup_data.Amount_in_USD=startup_data.Amount_in_USD.replace('\.\d*','',regex=True)
startup_data.Amount_in_USD=startup_data.Amount_in_USD.replace(r'\+*','',regex=True)

startup_data.drop(startup_data.loc[(startup_data.Amount_in_USD=='Undisclosed') | (startup_data.Amount_in_USD=='undisclosed')].index,inplace=True)

startup_data.Amount_in_USD.fillna(0,inplace=True)

startup_data.Amount_in_USD=pd.to_numeric(startup_data.Amount_in_USD)


# In[365]:


startup_data.Amount_in_USD.value_counts(dropna=False)


# In[366]:


startup_data


# In[7]:


startup_data.drop(columns='Investment_Type',inplace=True)


# In[229]:


startup_data


# In[16]:


startup_data


# In[4]:


replace_values = {'eCommerce' : 'ecommerce', 'ECommerce':'ecommerce', 'E-Commerce' : 'ecommerce','ecommmerce':'ecommerce', 'Ecommerce':'ecommerce','Health Care': 'Healthcare', 'HealthCare': 'Healthcare', 'IT': 'Information Technology'}     


# In[5]:


startup_data = startup_data.replace({"Industry_Vertical": replace_values})      


# In[6]:


startup_data.Industry_Vertical.nunique()


# In[23]:


startup_data


# In[32]:


plt.figure(figsize=(12,5))
iv=startup_data['SubVertical'].value_counts().head(10)
iv.plot.barh()

plt.title('Frequency  of  SubVertical ')
plt.ylabel('Frequency')
plt.xlabel('SubVertical')
plt.show()


# In[10]:


replace_values = {'Online lending platform' : 'Online Lending Platform','Online Lending':'Online Lending Platform','Online learning platform':'Online Learning Platform'}   


# In[11]:


startup_data = startup_data.replace({"SubVertical": replace_values})      


# In[12]:


startup_data


# In[13]:


'''Percentage of Missing records'''

startup_data.isnull().sum()/len(startup_data)


# In[14]:


startup_data.drop(columns='SubVertical',inplace=True)


# In[15]:


'''Percentage of Missing records'''

startup_data.isnull().sum()/len(startup_data)


# In[16]:


startup_data=startup_data.dropna(axis=0,how='any')


# In[377]:


startup_data


# In[17]:



startup_data.isnull().sum()/len(startup_data)


# In[18]:


startup_data.drop(columns='Startup_Date',inplace=True)


# In[380]:


startup_data


# In[19]:


startup_data.drop(columns='Investor_Name',inplace=True)


# In[20]:


startup_data.drop(columns='Sr_No',inplace=True)


# In[21]:


startup_data.drop(columns='Startup_Name',inplace=True)


# In[23]:


startup_data.drop(columns='City_Location',inplace=True)


# In[24]:


startup_data


# In[25]:


startup_data['New_City_Location'] = startup_data['New_City_Location'].astype('category')


# In[26]:


startup_data['Industry_Vertical'] = startup_data['Industry_Vertical'].astype('category')


# In[29]:


startup_data['Amount_in_USD'] = startup_data['Amount_in_USD'].astype('int')


# In[30]:


startup_data.dtypes


# In[31]:


replace_values = {'Chennai' : 'Tamil Nadu','NCT of Delhi':'Delhi'}   


# In[32]:


startup_data = startup_data.replace({"New_City_Location": replace_values})      


# In[33]:


startup_data


# In[34]:


startup_data=startup_data[['Industry_Vertical','New_City_Location','Amount_in_USD']]


# In[35]:


startup_data


# In[36]:


startup_data = startup_data.drop(startup_data[startup_data["Amount_in_USD"] == 0].index)


# In[37]:


for col in startup_data.columns:
    print(col,':',len(startup_data[col].unique()),'labels')


# In[38]:


pd.get_dummies(startup_data,drop_first=True).shape


# In[39]:


startup_data.Industry_Vertical.value_counts().sort_values(ascending=False).head(10)


# In[40]:


startup_data.New_City_Location.value_counts().sort_values(ascending=False).head(10)


# In[127]:


for col_name in startup_data:
    print(col_name)


# In[41]:


industryVerticals = ['Consumer Internet', 'Technology', 'ecommerce', 'Healthcare', 'Finance', 'Logistics', 'Food and Beverage', 'Education', 'Ed-Tech', 'Information Technology']

startup_data = startup_data[startup_data['Industry_Vertical'].isin(industryVerticals)]


# In[42]:


startup_data=startup_data[['Industry_Vertical','New_City_Location','Amount_in_USD']]


# In[43]:


startup_data


# In[44]:


states=['Karnataka','Delhi','Maharashtra','Telangana','Tamil Nadu']


# In[45]:



startup_data = startup_data[startup_data['New_City_Location'].isin(states)]


# In[46]:


X = startup_data.iloc[:, :-1].values
y = startup_data.iloc[:, -1].values


# In[47]:


print(X)


# In[48]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), [0,1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[49]:


print(X)


# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[51]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[52]:


y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[53]:


from sklearn import metrics


# In[54]:


print("MSE:",metrics.mean_squared_error(y_test,y_pred))


# In[55]:


print("RMSE",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[56]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


# In[60]:


df1 = df.head(30)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[58]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  


# In[61]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score


# In[62]:


from sklearn.metrics import r2_score


# In[68]:


print(y_test)


# In[69]:


print(y_train)


# In[70]:


print(X_train)


# In[71]:


print(X_test)


# In[74]:


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


# In[80]:


from sklearn.metrics import r2_score


# In[85]:


print(regressor.score(X_test, y_test),r2_score(y_test,y_pred))


# In[86]:


regressor.score(X_test,y_test)


# In[ ]:




