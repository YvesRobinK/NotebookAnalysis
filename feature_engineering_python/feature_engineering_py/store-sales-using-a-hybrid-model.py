#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from statsmodels.graphics.tsaplots import plot_pacf

from xgboost import XGBRegressor


# In[2]:


df = pd.read_csv("../input/store-sales-time-series-forecasting/train.csv", index_col="date", parse_dates=True)
X_test = pd.read_csv("../input/store-sales-time-series-forecasting/test.csv", index_col="date", parse_dates=True)


# In[3]:


df


# In[4]:


X_test


# In[5]:


family_unique = len(df.family.unique())


# In[6]:


plt.plot(df.groupby(df.index)["sales"].mean())


# In[7]:


# enc = OrdinalEncoder()


# df.family = enc.fit_transform( np.array(df.family).reshape( (-1, 1) ) )
# df.head()

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
dfOh = pd.DataFrame( ohe.fit_transform( np.array(df.family).reshape( (-1, 1) ) ) )
dfOh.columns = df.family.unique()
dfOh.index = df.index

df = pd.concat([df.drop("family", axis=1), dfOh], axis=1)
df.head()


# In[8]:


dfOh = pd.DataFrame( ohe.transform( np.array(X_test.family).reshape( (-1, 1) ) ) )
dfOh.columns = X_test.family.unique()
dfOh.index = X_test.index

X_test = pd.concat([X_test.drop("family", axis=1), dfOh], axis=1)
X_test.head()


# # Feature Engineering

# In[9]:


X = df.copy()
y = X.pop("sales")


# In[10]:


plt.plot(y)


# So, let's make trend!

# In[11]:


dp = DeterministicProcess(
    index=X.index,
    order=1,
    drop=True
)

dp.in_sample()


# In[12]:


X = pd.concat([X, dp.in_sample()], axis=1)


# In[13]:


X_test = pd.concat([X_test, dp.out_of_sample(steps=X_test.shape[0], forecast_index=X_test.index)], axis=1)


# In[14]:


X.head()


# In[15]:


X_test.head()


# In[16]:


X_1 = pd.DataFrame(X["trend"])
X_2 = X.drop("trend", axis=1)


# In[17]:


X_1


# In[18]:


_ = plot_pacf(y, lags=12)


# From the plot we see that 4, 5, 6 lags is a great idea!

# In[19]:


for i in range(4, 7):
    X_2[f"Lag_{i}"] = y.shift(i)


# In[20]:


for i in range(4, 7):
    X_test[f"Lag_{i}"] = np.zeros(X_test.shape[0])


# In[21]:


X_2 = X_2.fillna(0.0)


# In[22]:


X_1.head()


# In[23]:


X_2.head()


# # Modeling

# In[24]:


X_train, X_valid, y_train, y_valid = train_test_split(X_1, y, test_size=0.2, shuffle=False)


# For first we must learn a trend. LinearRegression is good to learn trend

# In[25]:


modelL = LinearRegression()
modelL.fit(X_train, y_train)


# In[26]:


y_pred = modelL.predict( X_train )
y_pred_valid = modelL.predict( X_valid )


# Next see on reduces

# In[27]:


y_train - y_pred


# In[28]:


mean_absolute_error(y_valid, modelL.predict(X_valid))


# In[29]:


X_train, X_valid, y_train, y_valid = train_test_split(X_2, y, test_size=0.2, shuffle=False)


# Next we must learn this, that LinearRegression didn't learning with trend. XGBRegressor or other machine learning models is good idea

# We will train XGBRegressor on LinearRegression's reduces

# In[30]:


modelX = XGBRegressor(n_estimators=20)
modelX.fit(X_train, y_train-y_pred)


# In[31]:


mean_absolute_error(y_valid, modelX.predict(X_valid))


# # Create a submission

# In[32]:


import warnings
warnings.filterwarnings('ignore')

preds = []

for i in range(1):
    y_pred = modelL.predict(pd.DataFrame(X_test.trend))

    y_pred += modelX.predict(X_test.drop("trend", axis=1))
    
    X_test["Lag_4"][i] = y[-4]
    X_test["Lag_5"][i] = y[-5]
    X_test["Lag_6"][i] = y[-6]
    
    preds.append(y_pred)


# In[33]:


preds


# In[34]:


df = pd.DataFrame({
    "id": X_test.trend.astype(np.int32)-1,
    "sales": preds[0]
})

df.to_csv("submission.csv", index=False)


# In[35]:


pd.read_csv("./submission.csv")


# In[ ]:




