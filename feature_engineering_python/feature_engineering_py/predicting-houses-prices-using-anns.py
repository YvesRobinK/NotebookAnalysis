#!/usr/bin/env python
# coding: utf-8

# # Solving a Regression Problem using ANN
# 
# In this notebook we are going to explore the House Sales in King County, USA dataset, preprocess it, and apply Artificial Neural Network model to predict the price of houses.

# ## The Data
# 
# We will be using data from a Kaggle data set:
# 
# https://www.kaggle.com/harlfoxem/housesalesprediction
# 
# #### Feature Columns
#     
# * id - Unique ID for each home sold
# * date - Date of the home sale
# * price - Price of each home sold
# * bedrooms - Number of bedrooms
# * bathrooms - Number of bathrooms, where .5 accounts for a room with a toilet but no shower
# * sqft_living - Square footage of the apartments interior living space
# * sqft_lot - Square footage of the land space
# * floors - Number of floors
# * waterfront - A dummy variable for whether the apartment was overlooking the waterfront or not
# * view - An index from 0 to 4 of how good the view of the property was
# * condition - An index from 1 to 5 on the condition of the apartment,
# * grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.
# * sqft_above - The square footage of the interior housing space that is above ground level
# * sqft_basement - The square footage of the interior housing space that is below ground level
# * yr_built - The year the house was initially built
# * yr_renovated - The year of the house’s last renovation
# * zipcode - What zipcode area the house is in
# * lat - Lattitude
# * long - Longitude
# * sqft_living15 - The square footage of interior housing living space for the nearest 15 neighbors
# * sqft_lot15 - The square footage of the land lots of the nearest 15 neighbors

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
df.head()


# # 1. Exploratory Data Analysis

# ### Check for missing values

# In[3]:


df.isnull().sum()


# In[4]:


df.describe().transpose()


# In[5]:


plt.figure(figsize=(12,8))
sns.displot(df['price'])


# In[6]:


sns.countplot(x=df['bedrooms'])


# In[7]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='sqft_living',data=df)


# In[8]:


plt.figure(figsize=(12,8))
sns.boxplot(x='bedrooms',y='price',data=df)


# ### Geographical Properties

# In[9]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='long',data=df)


# In[10]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='lat',data=df)


# In[11]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=df,hue='price')


# ### Handling Outliers

# In[12]:


df.sort_values('price',ascending=False).head(20)


# In[13]:


non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]


# In[14]:


plt.figure(figsize=(12,8))
sns.scatterplot(
    x='long',y='lat',
    data=non_top_1_perc,hue='price',
    palette='RdYlGn',
    edgecolor=None,
    alpha=0.2
)


# ### Other Features

# In[15]:


plt.figure(figsize=(12,8))
sns.boxplot(x='waterfront',y='price',data=df)


# ## Working with Feature Data

# In[16]:


df.head()


# In[17]:


df.info()


# In[18]:


df = df.drop('id',axis=1)


# ### Feature Engineering from Date

# In[19]:


df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)


# In[20]:


plt.figure(figsize=(12,10))

plt.subplot(2, 2, 1)
sns.boxplot(x='year',y='price',data=df)

plt.subplot(2, 2, 2)
sns.boxplot(x='month',y='price',data=df)


# In[21]:


plt.figure(figsize=(12,10))

plt.subplot(2, 2, 1)
df.groupby('month').mean()['price'].plot()

plt.subplot(2, 2, 2)
df.groupby('year').mean()['price'].plot()


# In[22]:


df = df.drop('date',axis=1)


# In[23]:


# https://i.pinimg.com/originals/4a/ab/31/4aab31ce95d5b8474fd2cc063f334178.jpg
# May be worth considering to remove this or feature engineer categories from it
df['zipcode'].value_counts()


# In[24]:


df = df.drop('zipcode',axis=1)


# In[25]:


# could make sense due to scaling, higher should correlate to more value
df['yr_renovated'].value_counts()


# In[26]:


df['sqft_basement'].value_counts()


# ## Scaling and Train Test Split

# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = df.drop('price',axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

scaler = MinMaxScaler()

X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)
print(X_test.shape)


# # 2. Model Building

# In[28]:


from sklearn import metrics

def print_evaluate(true, predicted, train=True):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    if train:
        print("========Training Result=======")
        print('MAE: ', mae)
        print('MSE: ', mse)
        print('RMSE: ', rmse)
        print('R2 Square: ', r2_square)
    elif not train:
        print("=========Testing Result=======")
        print('MAE: ', mae)
        print('MSE: ', mse)
        print('RMSE: ', rmse)
        print('R2 Square: ', r2_square)


# In[29]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam


# In[30]:


model = Sequential()

model.add(Dense(X_train.shape[1],activation='relu'))
model.add(Dense(32,activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer=Adam(0.001), loss='mse')


# ## Training the Model

# In[31]:


r = model.fit(
    X_train, y_train.values,
    validation_data=(X_test,y_test.values),
    batch_size=128,
    epochs=500
)


# In[32]:


plt.figure(figsize=(10, 6))

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()


# ## Evaluation on Test Data

# In[33]:


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print_evaluate(y_train, y_train_pred, train=True)
print_evaluate(y_test, y_test_pred, train=False)


# ## Is this a good result

# In[34]:


df['price'].mean()


# Mean Absolute Error: `94871` means that our prediction of a house price will be `+/-94871`.  

# ## Comparing with LinearRegression

# In[35]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)


# In[36]:


y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

print_evaluate(y_train, y_train_pred, train=True)
print_evaluate(y_test, y_test_pred, train=False)

