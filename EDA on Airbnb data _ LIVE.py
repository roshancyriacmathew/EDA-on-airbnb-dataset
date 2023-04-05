#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir(r'D:\The AI & DS Channel temp\xx_airBnb open data')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
style.use('ggplot')
import seaborn as sns
import missingno as msno
import plotly.express as px


# In[3]:


df = pd.read_csv("Airbnb_Open_Data.csv", low_memory=False)


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df.nunique()


# In[10]:


df.drop(['id','NAME','host id', 'host name','house_rules','license'], axis=1, inplace=True)


# In[11]:


df = df.drop_duplicates()


# In[12]:


df.shape


# In[13]:


df.info()


# In[14]:


percentage_null = df.isnull().sum()/df.shape[0]*100
percentage_null = pd.DataFrame({"columns":percentage_null.keys(), "%":percentage_null})
percentage_null.reset_index(drop=True, inplace=True)
percentage_null


# In[16]:


msno.bar(df)


# In[18]:


df.columns


# In[19]:


df.columns = df.columns.str.replace(' ','_')


# In[20]:


df.columns


# In[22]:


categorical_values = df.select_dtypes(include=[object])
print("Count of categorical features in the dataset:", categorical_values.shape[1])

numerical_values = df.select_dtypes(include=[np.float64, np.int64])
print("Count fo numerical features in the dataset:", numerical_values.shape[1])


# In[23]:


categorical_values.head()


# In[24]:


numerical_values.head()


# In[25]:


categorical_values.isnull().sum()


# In[26]:


sns.countplot(x='host_identity_verified', data=df)


# In[27]:


data = df['host_identity_verified'].value_counts()
data.plot(kind='pie', autopct='%0.1f%%')


# In[28]:


df['host_identity_verified'].value_counts()


# In[29]:


df['host_identity_verified'].fillna('unconfirmed', inplace=True)


# In[30]:


df['host_identity_verified'].isnull().sum()


# In[31]:


sns.countplot(y='neighbourhood_group', data=df)


# In[32]:


df['neighbourhood_group'].value_counts()


# In[33]:


df.replace({'neighbourhood_group': {'brookln':'Brooklyn', 'manhatan':'Manhattan'}}, inplace=True)


# In[34]:


df['neighbourhood_group'].nunique()


# In[35]:


df['neighbourhood_group'].fillna('Manhattan', inplace=True)


# In[36]:


df['neighbourhood'].value_counts()


# In[37]:


neighbourhood_count = df['neighbourhood'].value_counts()
top_15 = neighbourhood_count.head(15)


# In[38]:


top_15.plot(kind='bar', figsize=(10,5))
plt.xlabel('Neightbourhood')
plt.ylabel('Counts')
plt.title('Top 15 Neighbbourhood')
plt.show()


# In[39]:


df['country'].nunique()


# In[40]:


df['country'].value_counts()


# In[41]:


df['country'].fillna('United States', inplace=True)


# In[42]:


df['country_code'].isnull().sum()


# In[43]:


df['country_code'].value_counts()


# In[44]:


df['country_code'].fillna('US', inplace=True)


# In[45]:


data = df['instant_bookable'].value_counts()
data.plot(kind='pie', autopct='%0.1f%%')


# In[46]:


df['tmp'] =1
fig=px.pie(df, names='instant_bookable', values='tmp', hole=0.6, title="instant_bookable")
fig.update_traces(textposition='outside', textinfo='percent+label')
fig.update_layout(title_text='instant_bookable', annotations=[dict(text='instant_bookable', x=0.5, y=0.5, font_size=10, showarrow=False)])


# In[47]:


df['instant_bookable'].fillna('false', inplace=True)


# In[48]:


df['cancellation_policy'].nunique()


# In[49]:


sns.countplot(x='cancellation_policy', data=df)


# In[50]:


df['cancellation_policy'].value_counts()


# In[51]:


df['cancellation_policy'].fillna('moderate', inplace=True)


# In[52]:


sns.countplot(x='room_type', data=df)


# In[53]:


df['room_type'].value_counts()


# In[54]:


df['room_type'].fillna('Entire home/apt', inplace=True)


# In[55]:


def remove_dollar_sign(value):
    if pd.isna(value):
        return np.NaN
    else:
        return float(value.replace("$","").replace(",","").replace(" ",""))


# In[56]:


df['price'] = df['price'].apply(lambda x:remove_dollar_sign(x))
df['service_fee'] = df['service_fee'].apply(lambda x:remove_dollar_sign(x))


# In[57]:


df['price']


# In[58]:


plt.figure(figsize=(15,10))
plt.title("Relationship between price and service fee")
sns.scatterplot(x=df.price, y=df.service_fee, hue=df.room_type, s=30);


# In[59]:


df['last_review']


# In[60]:


def get_year(date):
    try:
        return str(date).split("/")[2]
    except:
        pass
    
df['last_review'] = df['last_review'].apply(get_year)


# In[61]:


df['last_review']


# In[63]:


fig, ax = plt.subplots(figsize=(12,8))
sns.countplot(y='last_review', data=df, ax=ax)


# In[64]:


df['last_review'].median()


# In[65]:


df['last_review'].fillna(2019, inplace=True)


# In[66]:


df['last_review'].isnull().sum()


# In[67]:


df.isnull().sum()


# In[68]:


df.columns


# In[69]:


year=df['Construction_year'].value_counts()
plt.figure(figsize=(20,8))
sns.pointplot(x=year.index, y=year.values)
plt.xlabel("Construction year")
plt.ylabel("Count")
plt.title("Consruction year")


# In[70]:


df['Construction_year'].isnull().sum()


# In[71]:


mode = df['Construction_year'].mode().iloc[0]
mode


# In[72]:


df['Construction_year'].fillna(2014, inplace=True)


# In[73]:


fig = px.histogram(df, x='availability_365')
fig.show()


# In[74]:


df['minimum_nights'].fillna(df['minimum_nights'].mode()[0], inplace=True)
df['number_of_reviews'].fillna(df['number_of_reviews'].mode()[0], inplace=True)
df['reviews_per_month'].fillna(df['reviews_per_month'].mode()[0], inplace=True)
df['review_rate_number'].fillna(df['review_rate_number'].mode()[0], inplace=True)
df['calculated_host_listings_count'].fillna(df['calculated_host_listings_count'].mode()[0], inplace=True)
df['availability_365'].fillna(df['availability_365'].mode()[0], inplace=True)


# In[75]:


numerical_values.isnull().sum()


# In[76]:


df.isnull().sum()


# In[77]:


df.head()


# In[78]:


df = df.dropna()


# In[79]:


df.isnull().sum()


# In[81]:


df.corr()


# In[82]:


df.drop('tmp', axis=1, inplace=True)


# In[83]:


plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation between the columns')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




