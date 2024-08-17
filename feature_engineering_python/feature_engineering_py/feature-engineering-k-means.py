#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
get_ipython().system('unzip /kaggle/input/expedia-personalized-sort/data.zip')
### ZipFile can't read this proprietary format 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


Train=pd.read_csv("train.csv")
Test=pd.read_csv("test.csv")


# In[3]:


print(Train.shape)
print(Test.shape)


# In[4]:


print(Train.isnull().sum())


# In[5]:


print(Test.isnull().sum())


# In[6]:


Train=Train.dropna(axis=1,how="any")


# In[7]:


Train.isnull().sum()


# In[8]:


Train.info()


# In[9]:


import seaborn as sns
df = Train.corr()
print(df)


# In[10]:


Train[["promotion_flag","position"]].groupby("promotion_flag").sum().position.plot(kind="pie",shadow=True,autopct="%1.1f%%",radius=1.2,startangle=120)
plt.title("Promotion Graph")
plt.show()


# In[11]:


Train=Train.drop(["srch_id","date_time","site_id","visitor_location_country_id","prop_country_id","prop_id","srch_destination_id"],axis=1)


# In[12]:


Test=Test.dropna(axis=1,how="any")


# In[13]:


Test.columns


# In[14]:


Test=Test.drop(["srch_id","date_time","site_id","visitor_location_country_id","prop_country_id","prop_id","srch_destination_id"],axis=1)


# In[15]:


Test.columns


# In[16]:


Train=Train[['prop_starrating', 'prop_brand_bool', 'prop_location_score1',
       'prop_log_historical_price', 'price_usd', 'promotion_flag',
       'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
       'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
       'random_bool']]


# In[17]:


from sklearn.cluster import KMeans
data = Train
n_cluster = range(1,11)

kmeans = [KMeans(n_clusters = i).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]


# In[18]:


plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), scores,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[20]:


from sklearn.cluster import KMeans
km = KMeans(n_clusters = 8)
kmean=km.fit(Train)
y_kmeans = km.predict(Train)


# In[22]:


from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = plt.figure(1, figsize = (7, 7))

ax = Axes3D(fig, rect = [0, 0, 0.95, 1], 
            elev = 48, azim = 134)

ax.scatter(Train.iloc[:, 4:5],
           Train.iloc[:, 7:8], 
           Train.iloc[:, 11:12],
           c = km.labels_.astype(np.float), edgecolor = 'm')

ax.set_xlabel('USD')
ax.set_ylabel('srch_booking_window')
ax.set_zlabel('srch_saturday_night_bool')

plt.title('K Means', fontsize = 10)


# In[24]:


km.predict(Test)


# In[ ]:




