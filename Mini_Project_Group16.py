#!/usr/bin/env python
# coding: utf-8

# This dataset Contains Information about Indian Startups from 2015-2019 regarding information about startups such as their location,amount of funding they have received etc.This dataset can help us understand the startup landscape and help us in comparing with current startup landscape in india.

# In[2]:


'''Importing Libraries'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

pd.set_option('display.max_rows',3000)


# In[3]:


'''Load Data and rename columns to enable us to use dot notation and to parse Date as a datetime object which can 
give us much more insights with the dates.Header tells pandas to ignore the first line which is the header of the 
column

Having Some Issue with the Geomapping Code.Cleaned Up the Replace Method Now

Removing the Remarks Column Since there are around 85% values empty
'''

startup_data=pd.read_csv('startup_funding_modified.csv',header=0,encoding='utf8',names=['Sr_No','Startup_Date','Startup_Name','Industry_Vertical','SubVertical','City_Location','Investor_Name','Investment_Type','Amount_in_USD','Remarks'])

startup_data.drop(columns='Remarks',inplace=True)


# In[4]:


startup_data


# In[5]:


'''Information about the memory consumption of the dataset.This step is important as it can tell us if we need to
read the dataset in chunks while reading the data in whichever format we have available'''

startup_data.info(memory_usage='deep')


# In[6]:


'''Basic stats about the dataset'''
startup_data.describe(include=['object','int'])


# In[7]:


'''Fixed Erroneous records from the dataset file so that no issues in python file.'''

startup_data["Startup_Date"]= pd.to_datetime(startup_data["Startup_Date"])


# In[8]:


startup_data.head(10)


# In[9]:


startup_data.dtypes


# In[10]:


print(startup_data['Startup_Name'].nunique())


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


city_data=startup_data.City_Location

replacement_dictionary={"USA":np.NaN,"US":np.NaN,"US/India":np.NaN,"Dallas":np.NaN,"Boston":np.NaN,"New York":np.NaN,"London":np.NaN,"Nairobi":np.NaN,"SFO / Karnataka":np.NaN,"Seattle":np.NaN,"San Francisco":np.NaN,"Santa Monica":np.NaN,"Missourie":np.NaN,"India":np.NaN,"Palo Alto":np.NaN,"California":np.NaN,"SFO / Bangalore":np.NaN,"Burnsville":np.NaN,"Singapore":np.NaN,"Bangalore":"Karnataka","Belgaum":"Karnataka","Udupi":"Karnataka","New Delhi":"NCT of Delhi","Gurgaon":"NCT of Delhi","Noida":"NCT of Delhi","Delhi":"NCT of Delhi","Gurugram":"NCT of Delhi","Mumbai":"Maharashtra","Pune":"Maharashtra","Nagpur":"Maharashtra","Hyderabad":"Telangana","Faridabad":"Haryana","Surat":"Gujarat","Ahmedabad":"Gujarat","Ahemdabad":"Gujarat","Ahemadabad":"Gujarat","Vadodara":"Gujarat","Jaipur":"Rajasthan","Jodhpur":"Rajasthan","Udaipur":"Rajasthan","Kolkata":"West Bengal","Kolkatta":"West Bengal","Indore":"Madhya Pradesh","Bhopal":"Madhya Pradesh","Gwalior":"Madhya Pradesh","Coimbatore":"Tamil Nadu","Karur":"Tamil Nadu","Hubli":"Tamil Nadu","Panji":"Goa","Panaji":"Goa","Kozhikode":"Kerala","Trivandrum":"Kerala","Agra":"Uttar Pradesh","Kanpur":"Uttar Pradesh","Lucknow":"Uttar Pradesh","Varanasi":"Uttar Pradesh","Gaya":"Bihar","Bhubneswar":"Odisha","Bhubaneswar":"Odisha","New New Delhi":"NCT of Delhi","Nw Delhi":"NCT of Delhi","Kochi":"Kerala","Srinagar":"Jammu & Kashmir","Rourkela":"Bihar","Siliguri":"West Bengal"}

city_data=startup_data.City_Location.replace(replacement_dictionary)

startup_data['New_City_Location']=city_data


# In[13]:


city_data=startup_data.City_Location

replacement_dictionary={"USA":np.NaN,"US":np.NaN,"US/India":np.NaN,"Dallas":np.NaN,"Boston":np.NaN,"New York":np.NaN,"London":np.NaN,"Nairobi":np.NaN,"SFO / Karnataka":np.NaN,"Seattle":np.NaN,"San Francisco":np.NaN,"Santa Monica":np.NaN,"Missourie":np.NaN,"India":np.NaN,"Palo Alto":np.NaN,"California":np.NaN,"SFO / Bangalore":np.NaN,"Burnsville":np.NaN,"Singapore":np.NaN,"Bangalore":"Karnataka","Belgaum":"Karnataka","Udupi":"Karnataka","New Delhi":"NCT of Delhi","Gurgaon":"NCT of Delhi","Noida":"NCT of Delhi","Delhi":"NCT of Delhi","Gurugram":"NCT of Delhi","Mumbai":"Maharashtra","Pune":"Maharashtra","Nagpur":"Maharashtra","Hyderabad":"Telangana","Faridabad":"Haryana","Surat":"Gujarat","Ahmedabad":"Gujarat","Ahemdabad":"Gujarat","Ahemadabad":"Gujarat","Vadodara":"Gujarat","Jaipur":"Rajasthan","Jodhpur":"Rajasthan","Udaipur":"Rajasthan","Kolkata":"West Bengal","Kolkatta":"West Bengal","Indore":"Madhya Pradesh","Bhopal":"Madhya Pradesh","Gwalior":"Madhya Pradesh","Coimbatore":"Tamil Nadu","Karur":"Tamil Nadu","Hubli":"Tamil Nadu","Panji":"Goa","Panaji":"Goa","Kozhikode":"Kerala","Trivandrum":"Kerala","Agra":"Uttar Pradesh","Kanpur":"Uttar Pradesh","Lucknow":"Uttar Pradesh","Varanasi":"Uttar Pradesh","Gaya":"Bihar","Bhubneswar":"Odisha","Bhubaneswar":"Odisha","New New Delhi":"NCT of Delhi","Nw Delhi":"NCT of Delhi","Kochi":"Kerala","Srinagar":"Jammu & Kashmir","Rourkela":"Bihar","Siliguri":"West Bengal"}

city_data=startup_data.City_Location.replace(replacement_dictionary)

startup_data['New_City_Location']=city_data


# In[14]:


# startup_data.New_City_Location.value_counts(dropna=False)
startup_data.dropna(subset=['New_City_Location'],inplace=True)
startup_data.isna().sum()
# new_startup_data=startup_data.loc[:,['New_City_Location']]


# In[15]:


new_dataframe=pd.merge(left=geo_data,right=startup_data,left_on='ST_NM',right_on='New_City_Location')

# new_dataframe[new_dataframe.ST_NM.isna()]


# In[16]:


'''Percentage of Missing records'''

startup_data.isnull().sum()/len(startup_data)


# In[17]:


get_ipython().run_line_magic('pinfo', 'sns.scatterplot')


# In[18]:


'''Inference
The startup scene in india basically peaked at around 2016 and has been on a decline ever since.Maybe some governmental factors have come into picture or there are too many people trying to jump in this field.This has resulted in lesser amount of funding for everybody
'''
plt.figure(figsize=(15,10))
plt.xlabel("Year",fontsize=12)
plt.ylabel("Count of Startups year Wise",fontsize=12)
startup_data.Startup_Date.dt.year.value_counts().sort_index().plot(kind='bar')


# In[19]:


'''Startups which have received funding in the year 2019'''
startup_fund_2019=startup_data.loc[startup_data.Startup_Date.dt.year==2019]


# In[20]:


max(startup_fund_2019.Amount_in_USD)


# In[21]:


startup_data.Amount_in_USD=startup_data.Amount_in_USD.replace(r'[,]','',regex=True)
# startup_data.Amount_in_USD=startup_data.Amount_in_USD.replace('\.\d*','',regex=True)
startup_data.Amount_in_USD=startup_data.Amount_in_USD.replace(r'\+*','',regex=True)

startup_data.drop(startup_data.loc[(startup_data.Amount_in_USD=='Undisclosed') | (startup_data.Amount_in_USD=='undisclosed')].index,inplace=True)

startup_data.Amount_in_USD.fillna(0,inplace=True)

startup_data.Amount_in_USD=pd.to_numeric(startup_data.Amount_in_USD)


# In[22]:


startup_data.Amount_in_USD.value_counts(dropna=False)


# In[23]:


'''Inference
Karnataka has the Highest Number of Startups with the Most Value
Kerala has the least Startups in terms of value

Surprisingly,Value of Startups in Haryana is more than that of Maharashtra(Check Startups based on New City Localtion)
'''
plt.figure(figsize=(15,10))
sns.barplot(x='Amount_in_USD',y='New_City_Location',data=startup_data)


# In[24]:


max(startup_fund_2019.Amount_in_USD)


# In[25]:


min(startup_fund_2019.Amount_in_USD)


# In[28]:


startup_fund_2016=startup_data.loc[startup_data.Startup_Date.dt.year==2016]
print(max(startup_fund_2016.Amount_in_USD))

print(min(startup_fund_2016.Amount_in_USD))


# In[29]:


print(startup_fund_2016.Amount_in_USD.mean())


# In[31]:


startup_fund_2019


# In[32]:


startup_fund_2019.Amount_in_USD=startup_fund_2019.Amount_in_USD({"undisclosed":"0"})


# In[33]:


startup_fund_2019.replace(to_replace="Undisclosed",value="0")


# In[34]:


startup_fund_2019.replace(to_replace="undisclosed",value="0")


# In[35]:


startup_fund_2019['Amount_in_USD']=startup_fund_2019['Amount_in_USD'].str.lower()


# In[36]:


startup_data['Amount_in_USD']=startup_data['Amount_in_USD'].str.lower()


# In[37]:


replace_values = {'undisclosed' : 0, 'Undisclosed' : 0.0, '14,342,000+' : '14,342,000'}   


# In[38]:


startup_fund_2019 = startup_fund_2019.replace({"Amount_in_USD": replace_values})             


# In[39]:


startup_fund_2019


# In[40]:


max(startup_fund_2019.Amount_in_USD)


# In[41]:


print(max(startup_fund_2019.Amount_in_USD))


# In[42]:


int(max(startup_fund_2019.Amount_in_USD))


# In[43]:


startup_fund_2019.Amount_in_USD


# In[44]:


startup_fund_2019.Amount_in_USD.sort()


# In[45]:


startup_fund_2019.sort_values(by=['Amount_in_USD'], inplace=True, ascending=False)


# In[46]:


startup_fund_2019.dtypes


# In[47]:


startup_fund_2019=startup_fund_2019.sort_values(by='Amount_in_USD', inplace=True, ascending=False)


# In[48]:


def clean_amount(x):
    #x = ''.join([c for c in str(x) if c in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']])
    x = str(x).replace(",","").replace("+","")
    x = str(x).lower().replace("undisclosed","")
    x = str(x).lower().replace("n/a","")
    if x == '':
        x = '-999'
    return x

startup_data["CleanedAmount"] = startup_data["Amount_in_USD"].apply(lambda x: float(clean_amount(x)))
Amount_in_USD = startup_data["CleanedAmount"]
Amount_in_USD = Amount_in_USD[~np.isnan(Amount_in_USD)]
Amount_in_USD = Amount_in_USD[Amount_in_USD!=-999]
plt.figure(figsize=(8,6))
plt.scatter(range(len(Amount_in_USD)), np.sort(Amount_in_USD.values), color="#1E90FF")
plt.xlabel('index', fontsize=12)
plt.ylabel('Funding value in USD', fontsize=12)
plt.title("Distribution of funding value in USD")
plt.show()


# In[49]:


startup_data.iloc[startup_data["CleanedAmount"].sort_values(ascending=True).index[:5]]


# In[50]:


startup_data


# In[51]:


startup_data.sort_values("CleanedAmount", axis = 0, ascending = False, 
                 inplace = True, na_position ='last') 


# In[52]:


startup_data


# Here,we can see that Rapido Bike Taxi, Flipkart,Paytm ,True North,Snapdeal,Ola,Ola Cabs and BigBasket are some of the highest funded startups.

# In[53]:


startup_data.unique()


# In[54]:


startup_data.Startup_Name.nunique()


# We have 2277 unique names and the rest I believe are some Startups that would have received funding multiple times
# 

# In[56]:


name_data=startup_data.copy()


# In[57]:


len(name_data.Startup_Name.unique())


# In[58]:


plt.figure(figsize=(15,10))
plt.xlabel("Startup Name",fontsize=12)
plt.ylabel("No of times funding given to a startup by investor",fontsize=12)
name_data.Startup_Name.value_counts().head(20).plot(kind='bar')


# From the graph plotted, we can see that Ola cabs and Swiggy have received the most number of fundings i.e. 8 times.

# In[59]:


'''Top 10 Investors in the year 2019'''

list_investors_2019=list_investors.loc[list_investors.Startup_Date.dt.year==2019]
new_investors_2019=list_investors_2019.Investor_Name.apply(lambda x:x.split(','))
new_investors_separate_2019=[]
new_investors_separate_2019=[data.strip() for item in new_investors_2019 for data in item if data not in new_investors_separate_2019 or data != ' ']

new_investor_series_2019=pd.Series(data=new_investors_separate_2019,name='Name')
new_investor_series_2019=new_investor_series_2019.loc[new_investor_series_2019!='']

plt.figure(figsize=(15,10))
plt.xlabel("Investor_Name",fontsize=12)
plt.ylabel("No of times funding given to a startup by investor",fontsize=12)
new_investor_series_2019.value_counts().head(11).plot(kind='bar')


# In[60]:


'''Important Investors throughout the Years (Overall) and also investors for the year 2016 and 2019 for the comparisison'''

list_investors=startup_data.copy()


# In[61]:


list_startups_2019=startup_fund_2019.Startup_Name.unique().tolist()


# In[62]:


'''Very Few Startups have received funding twice in the same financial year(Which is pretty rare) while almost all the others have received only funding once in the entire year.'''

startup_fund_2019.Startup_Name.value_counts()


# In[63]:


list_startups_2016=startup_fund_2016.Startup_Name.unique().tolist()


# In[64]:


'''Here the difference in clear.The reason for more amount of funding received was due to more startups being launched and several of these startups received funding multiple times withing the same calender year(More than the number of startups in the year 2019)'''

startup_fund_2016.Startup_Name.value_counts()


# In[65]:


set(list_startups_2016).intersection(list_startups_2019)


# Thus,we can see that not a lot of commonn startups are funded in 2016 and 2019

# In[66]:


startup_data.Amount_in_USD.describe()


# In[67]:


startup_data.Investment_Type.unique()


# In[68]:


'''Clearing the Investment Type Column(Same as previous Columns)'''

copy_investment_type=startup_data.Investment_Type.copy()
copy_investment_type.fillna('Unknown',inplace=True)

replacement_dictionary_investment_type={"Private\\\\nEquity":"Private Equity","PrivateEquity":"Private Equity","Private Equity Round":"Private Equity","Private":"Private Funding","Seed Funding Round":"Seed Funding","Seed\\\\nFunding":"Seed Funding","Seed funding":"Seed Funding","Crowd funding":"Crowd Funding","Seed / Angel Funding":"Seed/Angel Funding","Seed / Angle Funding":"Seed/Angel Funding","Seed/ Angel Funding":"Seed/Angel Funding","Angel / Seed Funding":"Seed/Angel Funding","Series B (Extension)":"Series B","Debt":"Debt Funding","Angel Round":"Angel Funding","Seed":"Seed Funding","pre-series A":"Pre-Series A","pre-Series A":"Pre-Series A","Angel":"Angel Funding","Equity":"Equity Funding","Equity Based Funding":"Equity Funding"}

copy_investment_type=copy_investment_type.replace(replacement_dictionary_investment_type)
copy_investment_type.unique()


# In[69]:


startup_data.Investment_Type.fillna('Unknown',inplace=True)
startup_data.Investment_Type=startup_data.Investment_Type.replace(replacement_dictionary_investment_type)


# In[70]:


plt.figure(figsize=(20,10))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlabel("Count of Startups",fontsize=15)
plt.ylabel("Funding Type",fontsize=15)
startup_data.Investment_Type.value_counts().plot(kind='barh')


# In[71]:


'''Percentage of Fundings for Each Category'''

startup_data.Investment_Type.value_counts()/len(startup_data.Investment_Type)


# The major type of funding is received by seed funding,private equity and seed/angel funding.

# In[72]:


new_data=startup_data.groupby(startup_data.Startup_Date.dt.year)['Amount_in_USD'].agg({'Total_Sum':'sum',"Total_Avg":'mean'})


# In[73]:


fig,ax=plt.subplots(1,2,figsize=(15,10))
ax[0].ticklabel_format(style='plain', axis='y')
ax[1].ticklabel_format(style='plain', axis='y')
sns.barplot(x=new_data.index,y=new_data.Total_Sum,ax=ax[0])
sns.barplot(x=new_data.index,y=new_data.Total_Avg,ax=ax[1])


# Inference:Even Though the Year 2016 had the Most Amount of Startups out of all the years,neither the amount of money invested(Total/Mean) is among the highest.Surprisingly,the Average Highest Funding is received in the year 2019.

# In[74]:


plt.figure(figsize=(14,5))
iv=startup_data['Industry_Vertical'].value_counts().head(10)
iv.plot.bar()

plt.title('Frequency  of Industry Vertical ')
plt.ylabel('Frequency')
plt.xlabel('Industry Vertical')
plt.show()


# In[75]:


#####  In which sector there are most startups
d=startup_data['Industry_Vertical'].value_counts().head(5)
f=startup_data.groupby('Investment_Type').sum()['Amount_in_USD']
fig,ax=plt.subplots(nrows=1,ncols=1)
labels=[d.index,f.index]
size=[d.values,f.values]
colors = [['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','pink'],['green','pink','red','yellow']]
plt.axis('equal')
explode = ((0.1, 0, 0, 0,0),(-0.5,0.5,0.1,0.1))
plt.title('Percentage Distribution of Industry Vertical ')
plt.pie(size[0],explode=explode[0], labels=labels[0], colors=colors[0], autopct='%1.1f%%', shadow=True, startangle=140)
plt.show()


# In[76]:


from pandas import DataFrame as show


# In[77]:


## Most Number Of investors

cmi=show(data.groupby('StartupName')['NoOfInvestors'].count().sort_values(ascending=False))
fig=plt.figure(figsize=(10,5))
sns.barplot(y='NoOfInvestors',x='StartupName',data=cmi.reset_index().head())
plt.show()
cmi.head(10)


# In[78]:


## Top Investors
investors = startup_data['Investor_Name']
cinvestors=show(investors)['Investor_Name'].value_counts()[2:]
cinvestors.head(10)
print("Top Investors in Frequency ")
plt.figure(figsize = (12,5))
bar= sns.barplot(x=cinvestors.index[:20],y=cinvestors.values[:20])
bar.set_xticklabels(bar.get_xticklabels(),rotation=70)
bar.set_title("Top Investors by funding on multiple days ", fontsize=15)
bar.set_xlabel("", fontsize=10)
bar.set_ylabel("Frequency of Funding", fontsize=12)
plt.show()


# In[81]:


color = sns.color_palette()


# In[82]:


city = startup_data['City_Location'].value_counts().head(10)
print(city)
plt.figure(figsize=(15,8))
sns.barplot(city.index, city.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('city location of startups', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("city location of startups with number of funding", fontsize=16)
plt.show()


# In[83]:


from wordcloud import WordCloud

names = startup_data["Investor_Name"][~pd.isnull(funding_data["Investor_Name"])]
#print(names)
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Investor Names", fontsize=35)
plt.axis("off")
plt.show()


# In[91]:


investors = startup_data['Investor_Name'].value_counts().head(10)
print(investors)
plt.figure(figsize=(15,8))
sns.barplot(investors.index, investors.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('Investors Names', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Investors Names with number of funding", fontsize=16)
plt.show()


# From this we can see that India Angel Network,Ratan Tata,Kalaari Capital and Sequoia Capital made the maximun number of fundings.

# In[92]:


investment = startup_data['Investment_Type'].value_counts()
print(investment)
plt.figure(figsize=(15,8))
sns.barplot(investment.index, investment.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('Investment Type', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Investment Type with number of funding", fontsize=16)
plt.show()


# In[94]:


from wordcloud import WordCloud

names = startup_data["Investor_Name"][~pd.isnull(startup_data["Investor_Name"])]
#print(names)
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Investor Names", fontsize=35)
plt.axis("off")
plt.show()


# In[98]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go


# In[99]:


temp = startup_data["Investment_Type"].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='Types of investment funding with %')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="BorrowerGender")


# In[110]:


plt.figure(figsize=(8,5))
sns.distplot(startup_data['AmountInUSD_log'].dropna())
plt.xlabel('log of Amount in USD', fontsize=12)
plt.title("Log Histogram of investment in USD", fontsize=16)
plt.show()


# In[112]:


location = startup_data['City_Location'].value_counts()
print("Description count of Location")
print(location[:5])


g = sns.boxplot(x='City_Location', y="AmountInUSD_log",
                data=startup_data[startup_data.City_Location.isin(location[:15].index.values)])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("City Localization by Amount LogDistribuition", fontsize=20)
g.set_xlabel("Top 15 Cities", fontsize=15)
g.set_ylabel("Amount (USD) - Log", fontsize=15)

plt.subplots_adjust(hspace = 0.65,top = 0.9)

plt.show()

#Let's create a new feature that is a Amount with log to better see the values distribuitions
startup_data['AmountInUSD_log'] = np.log(startup_data["Amount_in_USD"] + 1)


# In[ ]:


plt.figure(figsize=(14,10))
plt.subplot(211)
sns.countplot(x='Date_month_year', data=df_startups)
plt.xticks(rotation=90)
plt.xlabel('', fontsize=12)
plt.ylabel('Date Counting', fontsize=12)
plt.title("Count frequency Investiments Date ", fontsize=16)

plt.subplot(212)
sns.pointplot(x='Date_month_year', y='AmountInUSD_log', data=df_startups)
plt.xticks(rotation=90)
plt.xlabel('Dates', fontsize=12)
plt.ylabel('Amount Distribuition Log', fontsize=12)
plt.title("Money Distribuition by Month-Year", fontsize=16)

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)

plt.show()


# In[115]:


startup_data.Startup_Date.replace((['12/05.2015', '13/04.2015','15/01.2015','22/01//2015']),                          ('12/05/2015','13/04/2015','15/01/2015','22/01/2015'), inplace=True)


# In[114]:


startup_data.head(10)


# In[116]:


startup_data['Date'] = pd.to_datetime(startup_data['Startup_Date'])

startup_data['Date_month_year'] = startup_data['Startup_Date'].dt.to_period("M")
startup_data['Date_year'] = startup_data['Startup_Date'].dt.to_period("A")


# In[117]:


plt.figure(figsize=(14,10))
plt.subplot(211)
sns.countplot(x='Date_month_year', data=startup_data)
plt.xticks(rotation=90)
plt.xlabel('', fontsize=12)
plt.ylabel('Date Counting', fontsize=12)
plt.title("Count frequency Investiments Date ", fontsize=16)

plt.subplot(212)
sns.pointplot(x='Date_month_year', y='AmountInUSD_log', data=startup_data)
plt.xticks(rotation=90)
plt.xlabel('Dates', fontsize=12)
plt.ylabel('Amount Distribuition Log', fontsize=12)
plt.title("Money Distribuition by Month-Year", fontsize=16)

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)

plt.show()


# In[119]:


from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS


# In[120]:


plt.figure(figsize = (15,15))

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=150,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(startup_data[startup_data['Industry_Vertical'] == 'Technology']['Investor_Name']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("WORD CLOUD - INVESTORS TECHNOLOGY")
plt.axis('off')
plt.show()


# Plotting Average Year-Start-wise graph for investment in period between January-2015 to Jan-2019 (Annual Starting)

# In[122]:


q=startup_data['Amount_in_USD'].resample('AS').mean()
q.plot(kind='bar')
plt.title('average funding amount')


# In[123]:


startup_data['year']=startup_data['Date'].dt.year
startup_data['month']=startup_data['Date'].dt.month


# In[125]:


fig =plt.figure(figsize=(20,7))
fig.set_figheight
ts_month = startup_data.groupby(['year', 'month']).agg({'Amount_in_USD':'sum'})['Amount_in_USD']
ts_month.plot(linewidth=4, color='crimson',marker="o", markersize=10, markerfacecolor='olive')
plt.ylabel('USD in Billions')
plt.xlabel('Month');
plt.title('Funding Variation Per Month from 2015-2019')


# Companies with most number of investors

# In[128]:


cmi=show(startup_data.groupby('Startup_Name')['numberofinvestors'].count().sort_values(ascending=False))
fig=plt.figure(figsize=(10,5))
sns.barplot(y='numberofinvestors',x='StartupName',data=cmi.reset_index().head())
plt.show()
cmi.head(10)


# In[129]:


def calculate_n_investors(x):#function to calculate record wise number of investors
    if  re.search(',',x) and x!='empty':
        return len(x.split(','))
    elif x!='empty':
        return 1
    else:
        return -1
startup_data['numberofinvestors']=startup_data['Investor_Name'].replace(np.NaN,'empty').apply(calculate_n_investors)#removing missing investors and replacing with 'empty'


# In[130]:


n_inv2=startup_data

n_inv=startup_data['Investor_Name']
n_inv.fillna(value='None',inplace=True)
listed_n_inv=n_inv.apply(lambda x: x.lower().strip().split(','))
investors=[]
for i in listed_n_inv:
    for j in i:
        if(i!='None' or i!=''):
            investors.append(j.strip())
unique_investors=list(set(investors))


# In[131]:


investors=pd.Series(investors)
unique_investors=pd.Series(unique_investors)


# In[132]:


investors=list(investors[investors!=''])
unique_investors=list(unique_investors[unique_investors!=''])


# In[135]:


cmi=show(startup_data.groupby('Startup_Name')['numberofinvestors'].count().sort_values(ascending=False))
fig=plt.figure(figsize=(10,5))
sns.barplot(y='numberofinvestors',x='Startup_Name',data=cmi.reset_index().head())
plt.show()
cmi.head(10)


# In[137]:


tp10fund=show(startup_data.groupby('Startup_Name')['Amount_in_USD'].sum().sort_values(ascending=False))
tp10fund.head(10)


# Does Funding depend on the number of investors?

# In[138]:


top10=tp10fund.join(cmi)
sns.heatmap(top10.corr(),annot=True)
plt.title('Corelation Matrix')
plt.show()


# Here,we can see that there is a 23% correlation between the number of investors and the Amount in USD.

# In[139]:


from sklearn.model_selection import train_test_split


# In[ ]:




