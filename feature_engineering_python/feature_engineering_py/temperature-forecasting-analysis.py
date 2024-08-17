#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


train=pd.read_csv('/kaggle/input/smart-homes-temperature-time-series-forecasting/train.csv')
test=pd.read_csv('/kaggle/input/smart-homes-temperature-time-series-forecasting/test.csv')


# In[3]:


total_data=pd.concat([train.drop(['Indoor_temperature_room'],axis=1),test],ignore_index=True)


# # <h1 style='background:#9AB0BD; border:4; border-radius: 30px;height:60px; font-size:250%; font-weight: bold; color:black'><center>Temperature Forecasting Analysis</center></h1> 
# 
# ![Solar house sensors and actuators map.png](attachment:89597e52-3e7c-43d5-af5a-26791433ffad.png)
# 
# 
# <a id='top'></a>
# <div class="list-group" id="list-tab" role="tablist">
#     
# <h1 style='background:#9AB0BD; border:0;height:50px; border-radius: 10px; color:black'><center> TABLE OF CONTENTS </center></h1>
# 
# ### [**1. Importing Libraries**](#title-one)
# ### [**2. Data Analysis**](#title-two)
# ### [**3. Basic Prediction- before feature Engineering**](#title-three)
# ### [**4. Feature Eng and analysis**](#title-four)
# ### [**5. Removing some Rows**](#title-five)
# ### [**6. Linear Regression**](#title-six)
# ### [**7. XGBRegressor**](#title-seven)
# ### [**8. LinearBoostRegressor**](#title-eight)
# ### [**9. MLP**](#title-nine)
#     
# <a id="title-one"></a>
# <h1 style='background:#9AB0BD; border:4;height:40px; border-radius: 30px; color:black'><center>Importing Libraries</center></h1> 

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# <a id="title-two"></a>
# <h1 style='background:#9AB0BD; border:4;height:40px; border-radius: 30px; color:black'><center>Data Analysis</center></h1> 

# In[6]:


total_data.head()


# Observations:
# 
# > Time variable- values are noted for every 15minutes(can be changed into range of values).
# 
# > From observed 5 values, the Indoor_temperature_room increases with time(without zig-zag pattern).
# 
# > Time components are Date, Time, Day_of_the_week.

# In[7]:


total_data.info()


# In[8]:


total_data.describe()


# In[9]:


#No nan values in the dataset.
total_data.isnull().sum()


# In[10]:


plt.figure(figsize = (18,18))
sns.heatmap(total_data.corr(), annot = True, cmap = "RdYlGn")
plt.show()


# > Some of the variables are highly correlated with each other(Ex:-CO2_(dinning-room),CO2_room), which leads to multicollinearity.
# 
# 1. Remove some of the highly correlated independent variables.
# 2. Linearly combine the independent variables, such as adding them together.
# 3. Perform an analysis designed for highly correlated variables, such as principal components analysis or partial least squares regression.
# 4. LASSO and Ridge regression are advanced forms of regression analysis that can handle multicollinearity. If you know how to perform linear least squares regression, you’ll be able to handle these analyses with just a little additional study.
# 
# https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/

# In[11]:


train.columns


# From the Graph
# 
# > Output has high correlation between variables Relavtive_humidity_room,Meteo_Sun_light_in_west_facade, Outdoor_relative_humidity_Sensor.
# 
# > Least correlation with Lighting_room,Meteo_Sun_light_in_south_facade

# In[12]:


train.hist(bins=10, figsize=(15, 10))
plt.tight_layout()


# >  Relative_humidity_(dinning room) and Outdoor_relative_humidity_Sensor are left skewed(so can apply log function)

# <a id="title-three"></a>
# <h1 style='background:#9AB0BD; border:4;height:40px; border-radius: 30px; color:black'><center>Basic Prediction- before feature Engineering
# </center></h1> 

# In[13]:


X=train.drop(['Indoor_temperature_room','Id','Date','Time'],axis=1)
Y=train['Indoor_temperature_room']
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=40)


# In[14]:


from sklearn.linear_model import LinearRegression
lnr=LinearRegression(normalize=True,)
lnr.fit(x_train,y_train)
y_pred=lnr.predict(x_train)
print("Basic model prediction accuracy=",mean_squared_error(y_pred,y_train))


# In[15]:


y_pred=lnr.predict(x_val)
print("Basic model prediction accuracy=",mean_squared_error(y_pred,y_val))


# <a id="title-four"></a>
# <h1 style='background:#9AB0BD; border:4;height:40px; border-radius: 30px; color:black'><center>Feature Eng and analysis</center></h1> 

# In[16]:


total_data['Date']=pd.to_datetime(total_data['Date'],format="%d/%m/%Y")
total_data['Day']=total_data['Date'].dt.dayofyear.astype(float)
total_data['Time']=pd.DatetimeIndex(total_data['Time'])
total_data['Minutes']=total_data['Time'].apply(lambda x: x.hour *60 + x.minute).astype(float)
total_data['Day_of_the_week']=total_data['Day_of_the_week'].astype(int)


# In[17]:


X_1=pd.merge(total_data,train[['Id','Indoor_temperature_room']],on='Id')


# In[18]:


def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax
fig,ax0=plt.subplots(figsize=(20,15))
seasonal_plot(X_1, y="Indoor_temperature_room", period="Day", freq="Minutes", ax=ax0)


# > Clearly a seasonality curve with respect to hour each day
# 
# >Day 80 seems like an Outlier

# In[19]:


fig,ax=plt.subplots(figsize=(8,6))
ax.plot(X_1['Id'],(X_1['Meteo_Rain']),label='any')
ax.plot(X_1['Id'],X_1['Indoor_temperature_room'],label='Indoor_temperature_room')
ax.set_title('Find')
plt.legend()
plt.show()


# > Comparing with all other variables

# In[20]:


fig,ax=plt.subplots(figsize=(10,8))
sns.boxplot(data=X_1.drop(['Time','Id','Date','Minutes','Day','Day_of_the_week'],axis=1))
plt.xticks(rotation=90)


# > lot of values seems like not important(values nearly equall to zero, might be noise)

# In[21]:


fig,ax=plt.subplots(figsize=(20,15))
ax.plot(X_1['Id'],X_1['Indoor_temperature_room'])
ax.set_title('Find')
plt.show()


# > No long term trend(bcz Date is only from 13 to 11, nearly a month), but seasonality exist within a Day

# In[22]:


Q1=X_1['Indoor_temperature_room']
Q3=X_1['Lighting_room']
fig,ax=plt.subplots(figsize=(8,6))
ax.plot(X_1['Id'],Q1,label='Indoor_temperature_room')
ax.plot(X_1['Id'],Q3/4,label='another')
ax.set_title('Find')
plt.legend()
plt.show()


# <a id="title-five"></a>
# <h1 style='background:#9AB0BD; border:4;height:40px; border-radius: 30px; color:black'><center>Removing some rows</center></h1> 

# > Day 73 and 102 does not have all values(24hrs) in training dataset

# In[23]:


total_data=total_data[(total_data['Day']!=80)]
X_1=pd.merge(total_data,train[['Id','Indoor_temperature_room']],on='Id')


# In[24]:


Q2=X_1[['Indoor_temperature_room','Minutes']].groupby(['Minutes']).mean().reset_index()


# In[25]:


fig,ax=plt.subplots()
ax.plot(Q2['Minutes'],Q2['Indoor_temperature_room'])
plt.show()


# In[26]:


a1=np.arange(0,1400)
a2=-np.sin(2*np.pi*a1/1400)
fig,ax=plt.subplots()
ax.plot(a1,a2)
plt.show()


# * Clearly a sinosoidal wave with respect to each day
# * 18.5 at 0:00am and slowly decreasing to 16 at 7am, and started increasing to 21 approx at around 3pm and finally reached 19 by the end of the day.

# > Here, I have decided to subtract the average values(00:00 to 24:00) of all days from Output variable and predict for remaining part 

# In[27]:


Q2.rename(columns={'Indoor_temperature_room':'Out_avg'},inplace=True)


# In[28]:


total_data=pd.merge(total_data,Q2, on='Minutes').sort_values(['Id']).reset_index()
total_data=total_data.iloc[:,1:]


# > Algorithms that do not require normalization/scaling are the ones that rely on rules. They would not be affected by any monotonic transformations of the variables. Scaling is a monotonic transformation. Examples of algorithms in this category are all the tree-based algorithms — CART, Random Forests, Gradient Boosted Decision Trees. These algorithms utilize rules (series of inequalities) and do not require normalization.
# 
# https://towardsdatascience.com/all-about-feature-scaling-bcc0ad75cb35

# > By using Fourier Series we can create new variables to predict the seasonality within a interval

# In[29]:


#from statsmodels.tsa.deterministic import CalendarFourier
#cal_fourier_gen = CalendarFourier("D", 2)
#P=cal_fourier_gen.in_sample(total_data['Time'])


# In[30]:


#total_data=pd.concat([total_data.reset_index(),P.reset_index().drop(['Time'],axis=1)],axis=1)


# In[31]:


X_1=pd.merge(total_data,train[['Id','Indoor_temperature_room']],on='Id')
plt.figure(figsize = (18,18))
sns.heatmap(X_1.corr(), annot = True, cmap = "RdYlGn")
plt.show()


# In[32]:


total_data.columns


# In[33]:


#total_data['Relative_humidity_avg']=(total_data['Relative_humidity_(dinning-room)']).shift(1).bfill()
#total_data['CO2_avg']=(total_data['CO2_(dinning-room)']).shift(1).bfill()
#total_data['light_total']=(total_data['Lighting_(dinning-room)']).shift(1).bfill()
#total_data['Meteo_Rain_lg']=total_data['Meteo_Rain'].shift(1).bfill()
#total_data['Outdoor_relative_humidity_Sensor_lg']=total_data['Outdoor_relative_humidity_Sensor'].shift(1).bfill()
#total_data['Outdoor_relative_humidity_Sensor-lag']=total_data['Outdoor_relative_humidity_Sensor'].shift(1).bfill()
#required_features=['Id','Relative_humidity_avg','Meteo_Rain','Meteo_Wind','Meteo_Sun_light_in_west_facade','Meteo_Sun_light_in_east_facade','Outdoor_relative_humidity_Sensor','Out_avg']
ad=total_data[['Id','Out_avg']]
final_data=total_data.drop(['Minutes','Out_avg','Date','Time','Day_of_the_week'],axis=1)
final_data['Meteo_Rain']=final_data['Meteo_Rain']*10


# In[34]:


X_1=pd.merge(final_data,train[['Id','Indoor_temperature_room']],on='Id')
plt.figure(figsize = (18,18))
sns.heatmap(X_1.corr(), annot = True, cmap = "RdYlGn")
plt.show()


# In[35]:


X_1=pd.merge(final_data,train[['Id','Indoor_temperature_room']],on='Id')


# In[36]:


final_data2=final_data


# <a id="title-six"></a>
# <h1 style='background:#9AB0BD; border:4;height:40px; border-radius: 30px; color:black'><center>Linear Regression</center></h1> 

# In[37]:


X_1=pd.merge(final_data2,train[['Id','Indoor_temperature_room']],on='Id')
ad1=pd.merge(train['Id'],ad,on='Id')
X_1['Indoor_temperature_room']=X_1['Indoor_temperature_room']-ad1['Out_avg']


# In[38]:


X=X_1.drop(['Indoor_temperature_room','Id'],axis=1)
y=X_1['Indoor_temperature_room']
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2,shuffle=False)


# In[39]:


from sklearn.linear_model import LinearRegression
lnr=LinearRegression(normalize=True,)
lnr.fit(x_train,y_train)
y_pred=lnr.predict(x_train)
print("Basic model prediction accuracy=",mean_squared_error(y_pred,y_train))


# In[40]:


y_pred2=lnr.predict(x_val)
print("Basic model prediction accuracy=",mean_squared_error(y_pred2,y_val))


# In[41]:


some=pd.Series(y_pred)


# In[42]:


fig,ax=plt.subplots(figsize=(15,8))
ax.plot(some)
ax.plot(y_train)
plt.show()


# In[43]:


coeff=lnr.coef_
fig,ax=plt.subplots(figsize=(8,6))
ax.bar(x=X.columns,height=coeff)
plt.xticks(rotation=90)
plt.show()


# <a id="title-seven"></a>
# <h1 style='background:#9AB0BD; border:4;height:40px; border-radius: 30px; color:black'><center>XGBRegressor</center></h1> 

# # XGBRegressor

# In[44]:


X=X_1.drop(['Indoor_temperature_room','Id'],axis=1)
y=X_1['Indoor_temperature_room']
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=40)


# In[45]:


test_data=pd.merge(final_data2,test['Id'],on='Id')
ad2=pd.merge(test_data['Id'],ad,on='Id')


# In[46]:


from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


# In[47]:


#define model evaluation method
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#evaluate model
#scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)


# In[48]:


params={
    "n_estimators":[i for i in range(100,400,50)],
    "max_depth":[i for i in range(4,10,1)]
}


# In[49]:


#grid=XGBRegressor(learning_rate=0.05,n_estimators=2500)
#grid=GridSearchCV(xgr,param_grid=params,cv=8,verbose=1,n_jobs=-1,scoring='neg_mean_squared_error')
#grid_search=grid.fit(x_train,y_train)
#print(grid_search.best_score_ )
#grid.fit(x_train_y_train)


# In[50]:


#best_params=grid_search.best_params_


# In[51]:


model1=XGBRegressor(learning_rate=0.05,n_estimators=350)
model1.fit(x_train,y_train)
pred=model1.predict(x_val)
print(mean_squared_error(pred,y_val))
predicted_val=model1.predict(test_data.drop(['Id'],axis=1))


# In[52]:


pred=model1.predict(X_1.drop(['Id','Indoor_temperature_room'],axis=1))


# In[53]:


fig,ax=plt.subplots()
ax.bar(model1.feature_names_in_,model1.feature_importances_ *100)
plt.xticks(rotation=90)
plt.title("feature importance")
plt.show()


# In[54]:


fig,ax=plt.subplots(figsize=(8,6))
#ax.plot(test_data['Id'],predicted_val)
ax.plot(pred)
ax.plot(X_1['Indoor_temperature_room'])
plt.show()


# <a id="title-eight"></a>
# <h1 style='background:#9AB0BD; border:1;height:40px; border-radius: 30px; color:black'><center>LinearBoostRegressor</center></h1> 

# In[55]:


get_ipython().system('pip install -q --upgrade linear-tree')

from lineartree import LinearBoostRegressor
regressor = LinearBoostRegressor(base_estimator=LinearRegression(),
                                 n_estimators = 600,
                                 random_state = 42)


# In[56]:


#grid=GridSearchCV(regressor,param_grid=params,cv=8,verbose=1,n_jobs=-1,scoring='neg_mean_absolute_error')
#grid_search=grid.fit(x_train,y_train)
#print(grid_search.best_score_)
#best_params=grid_search.best_params_


# In[57]:


#model2=LinearBoostRegressor(base_estimator=LinearRegression(),random_state=42,**best_params)
model2=regressor
model2.fit(x_train,y_train)
predicted_val2=model2.predict(test_data.drop(['Id'],axis=1))
pred=model2.predict(x_train)
pred2=model2.predict(x_val)
print(mean_squared_error(y_train,pred))
print(mean_squared_error(y_val,pred2))


# <a id="title-nine"></a>
# <h1 style='background:#9AB0BD; border:4;height:40px; border-radius: 30px; color:black'><center>MLP</center></h1> 

# In[58]:


from sklearn.preprocessing import StandardScaler,MinMaxScaler
ss=StandardScaler()
ss.fit(X_1.drop(['Id','Indoor_temperature_room'],axis=1))
df=ss.transform(final_data.drop(['Id'],axis=1))
final_data2=pd.DataFrame(df,index=final_data.index,columns=final_data.columns[1:])
final_data2=pd.concat([final_data2,final_data['Id']],axis=1)


# In[59]:


X_1=pd.merge(final_data2,train[['Id','Indoor_temperature_room']],on='Id')
ad1=pd.merge(train['Id'],ad,on='Id')
X_1['Indoor_temperature_room']=X_1['Indoor_temperature_room']-ad1['Out_avg']


# In[60]:


X=X_1.drop(['Indoor_temperature_room','Id'],axis=1)
y=X_1['Indoor_temperature_room']
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=40)


# In[61]:


test_data=pd.merge(final_data2,test['Id'],on='Id')
ad2=pd.merge(test_data['Id'],ad,on='Id')


# In[62]:


from sklearn.neural_network import MLPRegressor


# In[63]:


from tensorflow import keras as k
from tensorflow.keras import layers
from tensorflow.keras import Sequential


# In[64]:


model3=Sequential()
model3.add(layers.Dense(64,activation='relu'))
model3.add(layers.Dropout(0.2))
model3.add(layers.Dense(32,activation='relu'))
model3.add(layers.Dropout(0.1))
model3.add(layers.Dense(1))
model3.compile(optimizer='adam', loss='mean_squared_error')
callback = k.callbacks.EarlyStopping(monitor='loss', patience=3)


# In[65]:


model3.fit(x_train,y_train, epochs=200,validation_data=(x_val,y_val),shuffle=True,callbacks=[callback])


# In[66]:


pred=model3.predict(X_1.drop(['Id','Indoor_temperature_room'],axis=1))
fig,ax=plt.subplots(figsize=(15,8))
ax.plot((X_1['Indoor_temperature_room']))
ax.plot(pred)
plt.show()


# In[67]:


#clf = MLPRegressor(random_state=42, max_iter=400,hidden_layer_sizes=(64,32,16),activation="tanh",learning_rate_init=0.05)
#clf.fit(x_train,y_train)
#pred=clf.predict(x_train)
#pred2=clf.predict(x_val)
#print(mean_squared_error(pred,y_train))
#print(mean_squared_error(pred2,y_val))


# In[68]:


predicted_val2=model3.predict(test_data.drop(['Id'],axis=1))


# In[69]:


my_submission=pd.DataFrame({'Id':test_data['Id'],'Indoor_temperature_room':predicted_val2.reshape(-1)})
my_submission['Indoor_temperature_room']=my_submission['Indoor_temperature_room']+ad2['Out_avg']


# In[70]:


fig,ax=plt.subplots(figsize=(15,8))
ax.plot((my_submission['Indoor_temperature_room']))
ax.plot(test_data['Outdoor_relative_humidity_Sensor'])
plt.show()


# In[71]:


my_submission.to_csv('submission.csv', index=False)


# <h1 style='background:#9AB0BD;text-align: center;height:40px; border:4; border-radius: 30px; color:black'>Thanks for reading. If u Like Please upvote:)</h1> 
