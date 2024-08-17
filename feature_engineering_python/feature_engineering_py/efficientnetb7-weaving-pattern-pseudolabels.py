#!/usr/bin/env python
# coding: utf-8

# Credits to [Alexander Ryzhkov](https://www.kaggle.com/alexryzhkov) for his [great lightAutoMl notebook](https://www.kaggle.com/alexryzhkov/tps-lightautoml-baseline-with-pseudolabels)!

# In this notebook, I have converted the tabular data into image data by feature engineering to be able to use pre-trained models. In this case I use an EfficientNetB7:
# 
# ![](https://1.bp.blogspot.com/-Cdtb97FtgdA/XO3BHsB7oEI/AAAAAAAAEKE/bmtkonwgs8cmWyI5esVo8wJPnhPLQ5bGQCLcBGAs/s640/image4.png)
# 
# The EfficientNetB7 performs a grid search to find the relationship between different scaling dimensions of the baseline network. The effectiveness of the scaling relies heavily on the baseline model. Therefore, an additional architecture search was developed. 

# In[1]:


import numpy as np
import pandas as pd 
pd.set_option('display.max_columns', 2000)
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose

from tensorflow.keras import optimizers, utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Flatten, Dropout, PReLU
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
import tensorflow as tf

from scipy.special import erfinv

import warnings
warnings.filterwarnings("ignore")

import gc


# In[2]:


train = pd.read_csv("../input/tabular-playground-series-jul-2021/train.csv")
targetCols=[i for i in train.columns if "target" in i]
test = pd.read_csv("../input/tabular-playground-series-jul-2021/test.csv")
samSub = pd.read_csv("../input/tabular-playground-series-jul-2021/sample_submission.csv")
pslb = pd.read_csv("../input/tps-lightautoml-baseline-with-pseudolabels/lightautoml_with_pseudolabelling_kernel_version_14.csv")


# In[3]:


train["date_time"] = train.date_time.astype("datetime64")
test["date_time"] = test.date_time.astype("datetime64")
pslb["date_time"] = pslb.date_time.astype("datetime64")


# In[4]:


test = test.merge(right=pslb, on = "date_time")
test[targetCols] = test[targetCols].round(1)


# In[5]:


targetCols.append("date_time")
yTr=train[targetCols]
yTe=test[targetCols]
targetCols=targetCols[:-1]


# In[6]:


def SaisonalComponents(DF1, DF2, periods=24*2):
    DF12 = pd.concat([DF1, DF2])
    DF = DF12.copy()
    for i in DF12.columns[1:]:
        result = seasonal_decompose(DF12[f"{i}"], model='additive', period=periods)
        DF12[f"S{i}"] = result.seasonal
        DF12[f"S{i}"] = DF12[f"S{i}"].fillna(DF12[f"S{i}"].mean())
        #result.plot()
    return DF, DF12


# In[7]:


gap=24*2
Orig, SeasAdj = SaisonalComponents(train, test, periods=gap)


# # Feature Engineering
# Here, so many features are created that the square root of the resulting columns results in an even number - after throwing out the variables that are not needed.

# In[8]:


def FeatEng(DF, lags=range(2,29)):
    
    DF=DF.copy()
    
    DF["weekday"] = np.sin(DF.date_time.dt.weekday / 7 * np.pi/2)
    DF["hour"] = np.sin(DF.date_time.dt.hour / 24 * np.pi/2)
    DF["day"] = DF.date_time.dt.day
    DF["dayOfYear"] = DF.date_time.dt.dayofyear
    DF["month"] = DF.date_time.dt.month
    DF["working_hours"] =  DF["hour"].isin(list(range(7, 22, 1))).astype("int")
    DF["SMC"] = np.log1p(DF["absolute_humidity"] * 100) - np.log1p(DF["relative_humidity"])
    DF["Elapsed"] = np.sin(DF.date_time.dt.dayofyear / 365 * np.pi / 2)
    
    DF["sensor_1sq"] = DF["sensor_1"]**2
    DF["sensor_3sq"] = DF["sensor_2"]**2
    DF["sensor_3sq"] = DF["sensor_3"]**2
    DF["sensor_4sq"] = DF["sensor_4"]**2
    DF["relative_humiditysq"] = DF["relative_humidity"]**2
    DF["absolute_humiditysq"] = DF["absolute_humidity"]**2
    DF["deg_Csq"] = DF["deg_C"]**2
    
    for l in lags:
        for v in DF.columns[1:12]:
            
            m=DF[f"{v}"].mean()
            s=DF[f"{v}"].std()
            mx=DF[f"{v}"].mean()+DF[f"{v}"].std()
            mi=DF[f"{v}"].mean()-DF[f"{v}"].std()

            DF["mean{0}L{1}".format(v,l)] = DF[f"{v}"].rolling(window=l, center=True).mean().fillna(m).round(3)
            DF["sd{0}L{1}".format(v,l)] = DF[f"{v}"].rolling(window=l, center=True).std().fillna(s).round(3)
            DF["max{0}L{1}".format(v,l)] = DF[f"{v}"].rolling(window=l, center=True).max().fillna(mx).round(3)
            DF["min{0}L{1}".format(v,l)] = DF[f"{v}"].rolling(window=l, center=True).min().fillna(mi).round(3)
            DF["lagDelta{0}L{1}".format(v,l)] = DF[f"{v}"] - DF[f"{v}"].rolling(window=l).mean().fillna(m).round(3)
            gc.collect()

    DF.dropna(inplace=True)

    return DF


# In[9]:


get_ipython().run_cell_magic('time', '', 'trainTest = FeatEng(Orig)\ntrainTest.rename(columns={"target_carbon_monoxide": "target_carbon_monoxideN", \n                          "target_benzene": "target_benzeneN", \n                          "target_nitrogen_oxides": "target_nitrogen_oxidesN"\n                         },\n                 inplace=True\n                )\ntrainTest.shape\n')


# In[10]:


length=train.shape[0]

train = pd.concat([trainTest.iloc[:length,:], SeasAdj.iloc[:length,12:]], axis=1)
test = pd.concat([trainTest.iloc[length:,:], SeasAdj.iloc[length:,12:]], axis=1)

train = pd.concat([train, test]).reset_index(drop=True)
Ys = pd.concat([yTr, yTe]).reset_index(drop=True)

train.shape


# # Test Validation Data
# 
# At this point it is important to set the gap between the training and validation data set as large as the largest delay - used at dhe feature engineering part.

# In[11]:


X = train.loc[train.date_time < datetime.datetime(2011,1,1,12,0,0)]
x = train.loc[train.date_time > datetime.datetime(2011,1,1,12,0,0) + datetime.timedelta(hours=gap)]

print("tr shape: " + str(X.shape))
print("val shape: " + str(x.shape))
imgSze = int(np.sqrt(X.shape[1]-1))
print("square root: " + str(imgSze))


# # Normalize

# In[12]:


def rg(DF1, DF2, e, Vars):
    
    DF1=DF1.copy()
    length = DF1.shape[0]
    DF2=DF2.copy()
    
    DF12 = pd.concat([DF1[Vars], DF2[Vars]])
    
    for i in Vars:
        r = DF12[i].rank()
        Range = (r/r.max()-0.5)*2
        Range = np.clip(Range, a_max = 1-e, a_min = -1+e)
        rg = erfinv(Range)
        rg = rg * 2**0.5
        DF1[i] = rg[:length]
        DF2[i] = rg[length:]
        
    return DF1, DF2


# In[13]:


X, x = rg(X, x, 0.000001, train.columns[1:])


# In[14]:


Y=Ys.loc[Ys.date_time < datetime.datetime(2011,1,1,12,0,0)].drop(columns=["date_time"]).values
y=Ys.loc[Ys.date_time > datetime.datetime(2011,1,1,12,0,0) + datetime.timedelta(hours=gap)].drop(columns=["date_time"]).values
print(y.shape)

X=np.reshape(X.drop(columns=["date_time"]).to_numpy(),(-1, imgSze, imgSze, 1))
x=np.reshape(x.drop(columns=["date_time"]).to_numpy(),(-1, imgSze, imgSze, 1))
print(x.shape)


# To use pre-trained models, it is important to create three channels. 

# In[15]:


XD = np.ndarray(shape=(X.shape[0], X.shape[1], X.shape[2], 3), dtype= np.uint8)
XD[:, :, :, 0] = X[:,:,:,0]
XD[:, :, :, 1] = X[:,:,:,0]
XD[:, :, :, 2] = X[:,:,:,0]

xD = np.ndarray(shape=(x.shape[0], x.shape[1], x.shape[2], 3), dtype= np.uint8)
xD[:, :, :, 0] = x[:,:,:,0]
xD[:, :, :, 1] = x[:,:,:,0]
xD[:, :, :, 2] = x[:,:,:,0]


# # Train Test Data

# In[16]:


train, test = rg(train, test, 0.000001, train.columns[1:])


# # Image Visualization

# In[17]:


plt.figure(figsize=(5,5), dpi= 100)
fig, p = plt.subplots(4, 6, figsize=(25,20))

r=0
c=0

for i in range(0, 288, 12):
 
    p[r, c].imshow(x[i], interpolation='nearest')
    
    p[r, c].set_xlabel(test.iloc[i,0])
    
    if c == 0:
        p[r, c].set_ylabel('heigth')
    
    if r == 0:
        p[r, c].set_title('width')
    
    if c < 5:
        c+=1
    else:
        c=0
        r+=1
        
plt.show()   


# In[18]:


Ys=Ys.drop(columns=["date_time"]).values

train=np.reshape(train.drop(columns=["date_time"]).to_numpy(),(-1, imgSze, imgSze, 1))
test=np.reshape(test.drop(columns=["date_time"]).to_numpy(),(-1, imgSze, imgSze, 1))
print(test.shape)


# In[19]:


trainD = np.ndarray(shape=(train.shape[0], train.shape[1], train.shape[2], 3), dtype= np.uint8)
trainD[:, :, :, 0] = train[:,:,:,0]
trainD[:, :, :, 1] = train[:,:,:,0]
trainD[:, :, :, 2] = train[:,:,:,0]

testD = np.ndarray(shape=(test.shape[0], test.shape[1], test.shape[2], 3), dtype= np.uint8)
testD[:, :, :, 0] = test[:,:,:,0]
testD[:, :, :, 1] = test[:,:,:,0]
testD[:, :, :, 2] = test[:,:,:,0]


# # The Model

# In[20]:


M = EfficientNetB1(
    include_top=False, 
    input_shape=(imgSze, imgSze, 3),
    weights='imagenet'
    )

model=Sequential()
model.add(M)
model.add(GlobalMaxPooling2D(name="pool"))
model.add(Dropout(rate=0.1))
model.add(Dense(3, PReLU()))

M.trainable = True


# In[21]:


def rmsle(y_pred, y_true):
    y_pred = tf.cast(y_pred, dtype="float32")
    y_true = tf.cast(y_true, dtype="float32")
    r = tf.sqrt(tf.keras.backend.mean(tf.square(tf.math.log(y_pred+1) - tf.math.log(y_true+1))))
    return r


# In[22]:


lrReducer = ReduceLROnPlateau (    
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0.000001,
    )


# In[23]:


def lr_schaker(epoch, lr):
    if epoch == 20:
        lr = lr*1.1
    elif epoch == 40:
        lr = lr*1.1
    return lr


# In[24]:


model.compile(
  optimizer=optimizers.SGD(
      lr=0.2,
      decay = 0.0001, 
      momentum = 0.5,
      nesterov = True,
      clipvalue=20
      ),
  loss=rmsle,
  metrics="mae",
)

model.summary()


# In[25]:


history = model.fit(
  trainD,
  Ys,
  batch_size=128,
  epochs=40,
  validation_data=(xD, y),
  verbose=1,
  use_multiprocessing=True,
  workers=4, 
  callbacks=[lrReducer, LearningRateScheduler(lr_schaker, verbose=0)] 
)


# In[26]:


predBx = model.predict(testD)
predBx = pd. DataFrame(np.reshape(predBx, (test.shape[0], 3))) 


# In[27]:


samSub[targetCols] = predBx.values
samSub.to_csv("Submission.csv",index=False)


# In[28]:


samSub.describe()

