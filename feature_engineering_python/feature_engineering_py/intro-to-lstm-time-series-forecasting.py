#!/usr/bin/env python
# coding: utf-8

# <div style="padding:16px;color:black;margin:0;font-size:240%;text-align:center;display:fill;border-radius:5px;background-color:#fcedae;overflow:hidden;font-weight:700;border: 5px solid #f28d27;"> 🔮 Detailed LSTM 🗓️ Time Series Forecasting 📆</div>

# # 📟 1. Time Series Analysis

# In mathematics, a time series is a series of data points indexed in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time. Thus it is a sequence of discrete-time data. [*Wikipedia*](https://en.wikipedia.org/wiki/Time_series)
# 
# Time series analysis is a specific way of analyzing a sequence of data points collected over an interval of time. In time series analysis, analysts record data points at consistent intervals over a set period of time rather than just recording the data points intermittently or randomly. [*Tableau*](https://www.tableau.com/learn/articles/time-series-analysis)

# A Time-Series is a sequence of data points collected at different timestamps. These are essentially successive measurements collected from the same data source at the same time interval. Further, we can use these chronologically gathered readings to monitor trends and changes over time. The time-series models can be univariate or multivariate. The univariate time series models are implemented when the dependent variable is a single time series, like room temperature measurement from a single sensor. On the other hand, a multivariate time series model can be used when there are multiple dependent variables, i.e., the output depends on more than one series. An example for the multivariate time-series model could be modelling the GDP, inflation, and unemployment together as these variables are linked to each other...
# 
# - A Time-Series represents a series of time-based orders. It would be Years, Months, Weeks, Days, Horus, Minutes, and Seconds
# - A time series is an observation from the sequence of discrete-time of successive intervals.
# - A time series is a running chart.
# - The time variable/feature is the independent variable and supports the target variable to predict the results.
# - Time Series Analysis (TSA) is used in different fields for time-based predictions – like Weather Forecasting, Financial, Signal processing, Engineering domain – Control Systems, Communications Systems.
# - Since TSA involves producing the set of information in a particular sequence, it makes a distinct from spatial and other analyses.
# - Using AR, MA, ARMA, and ARIMA models, we could predict the future.

# <div style="padding:10px;color:black;margin:0;font-size:200%;text-align:center;display:fill;border-radius:5px;background-color:#fcedae;overflow:hidden;font-weight:700;border: 5px solid #f28d27;">🧮 LSTM - Long Short Term Memory </div>

# # 🧮 2. LSTM - Long short-term memory

# **Long short-term memory (LSTM) is an artificial neural network used in the fields of artificial intelligence and deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. Such a recurrent neural network (RNN) can process not only single data points (such as images), but also entire sequences of data (such as speech or video).** For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition, speech recognition, machine translation, robot control, video games, and healthcare. LSTM has become the most cited neural network of the 20th century.
# 
# https://colah.github.io/posts/2015-08-Understanding-LSTMs/

# ### Recurrent Neural Networks
# 
# Humans don’t start their thinking from scratch every second. As you read this essay, you understand each word based on your understanding of previous words. You don’t throw everything away and start thinking from scratch again. Your thoughts have persistence.
# 
# Traditional neural networks can’t do this, and it seems like a major shortcoming. For example, imagine you want to classify what kind of event is happening at every point in a movie. It’s unclear how a traditional neural network could use its reasoning about previous events in the film to inform later ones.
# 
# Recurrent neural networks address this issue. They are networks with loops in them, allowing information to persist.

# The name of LSTM refers to the analogy that a standard RNN has both "long-term memory" and "short-term memory". The connection weights and biases in the network change once per episode of training, analogous to how physiological changes in synaptic strengths store long-term memories; the activation patterns in the network change once per time-step, analogous to how the moment-to-moment change in electric firing patterns in the brain store short-term memories. The LSTM architecture aims to provide a short-term memory for RNN that can last thousands of timesteps, thus "long short-term memory".
# 
# A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell.
# 
# LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series. LSTMs were developed to deal with the vanishing gradient problem that can be encountered when training traditional RNNs. Relative insensitivity to gap length is an advantage of LSTM over RNNs, hidden Markov models and other sequence learning methods in numerous applications

# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/LSTM_Cell.svg/1280px-LSTM_Cell.svg.png)

# ### LSTM
# It is special kind of recurrent neural network that is capable of learning long term dependencies in data. This is achieved because the recurring module of the model has a combination of four layers interacting with each other.
# 
# ![](https://www.tutorialspoint.com/time_series/images/neural_network.jpg)
# 
# #### Neural Network
# The picture above depicts four neural network layers in yellow boxes, point wise operators in green circles, input in yellow circles and cell state in blue circles. An LSTM module has a cell state and three gates which provides them with the power to selectively learn, unlearn or retain information from each of the units. The cell state in LSTM helps the information to flow through the units without being altered by allowing only a few linear interactions. Each unit has an input, output and a forget gate which can add or remove the information to the cell state. The forget gate decides which information from the previous cell state should be forgotten for which it uses a sigmoid function. The input gate controls the information flow to the current cell state using a point-wise multiplication operation of ‘sigmoid’ and ‘tanh’ respectively. Finally, the output gate decides which information should be passed on to the next hidden state

# <div style="padding:10px;color:black;margin:0;font-size:200%;text-align:center;display:fill;border-radius:5px;background-color:#fcedae;overflow:hidden;font-weight:700;border: 5px solid #f28d27;">📋 The Core Idea Behind LSTMs</div>

# # 📋 3. The Core Idea Behind LSTMs
# 
# The key to LSTMs is the cell state, the horizontal line running through the top of the diagram.
# 
# The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.
# 
# ![](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-C-line.png)
# 
# The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.
# 
# Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.
# 
# ![](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-gate.png)
# 
# The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”
# 
# An LSTM has three of these gates, to protect and control the cell state.

# ## LSTM vs RNN
# 
# Consider, you have the task of modifying certain information in a calendar. To do this, an RNN completely changes the existing data by applying a function. Whereas, LSTM makes small modifications on the data by simple addition or multiplication that flow through cell states. This is how LSTM forgets and remembers things selectively, which makes it an improvement over RNNs.
# 
# Now consider, you want to process data with periodic patterns in it, such as predicting the sales of colored powder that peaks at the time of Holi in India. A good strategy is to look back at the sales records of the previous year. So, you need to know what data needs to be forgotten and what needs to be stored for later reference. Else, you need to have a really good memory. Recurrent neural networks seem to be doing a good job at this, theoretically. However, they have two downsides, exploding gradient and vanishing gradient, that make them redundant.
# 
# Here, LSTM introduces memory units, called cell states, to solve this problem. The designed cells may be seen as differentiable memory.

# <div style="padding:10px;color:black;margin:0;font-size:200%;text-align:center;display:fill;border-radius:5px;background-color:#fcedae;overflow:hidden;font-weight:700;border: 5px solid #f28d27;">📒 Data and Modules</div>

# # 📒 4. Data and Modules

# In[1]:


import numpy as np 
import pandas as pd 

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

import os
train=pd.read_csv("/kaggle/input/tabular-playground-series-sep-2022/train.csv")
test=pd.read_csv("/kaggle/input/tabular-playground-series-sep-2022/test.csv")


# <div style="padding:10px;color:black;margin:0;font-size:200%;text-align:center;display:fill;border-radius:5px;background-color:#fcedae;overflow:hidden;font-weight:700;border: 5px solid #f28d27;">📑 Inspecting Data</div>

# # 📑 5. Inspecting Data

# In[2]:


train.head()


# In[3]:


sns.set(rc={'figure.figsize':(24,8)})
ax=sns.lineplot(data=train,x='date',y='num_sold',hue='country')
ax.axes.set_title("\nBasic Time Series of Sales\n",fontsize=20);


# In[4]:


sns.set(rc={'figure.figsize':(24,8)})
ax=sns.lineplot(data=train,x='date',y='num_sold',hue='product')
ax.axes.set_title("\nBasic Time Series of Sales\n",fontsize=20);


# In[5]:


sns.set(rc={'figure.figsize':(24,8)})
ax=sns.lineplot(data=train,x='date',y='num_sold',hue='store')
ax.axes.set_title("\nBasic Time Series of Sales\n",fontsize=20);


# In[6]:


get_ipython().system('pip install scalecast --upgrade')


# <div style="padding:10px;color:black;margin:0;font-size:200%;text-align:center;display:fill;border-radius:5px;background-color:#fcedae;overflow:hidden;font-weight:700;border: 5px solid #f28d27;">📖 Time Series Analysis</div>

# # 📖 6. Time Series Analysis

# In[7]:


import pickle
from scalecast.Forecaster import Forecaster
sns.set(rc={'figure.figsize':(25,8)})

df = train[(train['country']=='Belgium')&(train['product']=='Kaggle Advanced Techniques')&(train['store']=='KaggleMart')]


# ## Forecaster Class

# In[8]:


f = Forecaster(y=df.num_sold,current_dates=df.date)


# ## Partial Auto-corelation

# In[9]:


f.plot_pacf(lags=7)
plt.show()


# ## Seasonal Decompose

# In[10]:


f.seasonal_decompose (model='additive', extrapolate_trend='freq', period=1).plot()
plt.show()


# ## Stationarity test

# In[11]:


stat, p, _, _, _, _ = f.adf_test(full_res=True)
print( stat,p)


# <div style="padding:10px;color:black;margin:0;font-size:200%;text-align:center;display:fill;border-radius:5px;background-color:#fcedae;overflow:hidden;font-weight:700;border: 5px solid #f28d27;">🔮 LSTM Forecasting</div>

# # 🔮 7. LSTM Forecasting

# To model anything in scalecast, we need to complete the following three basic steps:
# 
# **1. Specify a test length** — all models are tested in scalecast with the same slice of data and at least one data point must be set aside to do so. There is no getting around this. The test length is a discrete number of the last observations in the full time series. You can pass a percentage of a discrete number to the set_test_length function.
# 
# **2. Generate future dates** —all models in scalecast produce a forecast in the same scale as the observed data. There is no getting around this. The number of dates you generate in this step will determine how long all models will be forecast out.
# 
# **3. Choose an estimator** — we will be using the “lstm” estimator, but there are a handful of others available.

# In[12]:


f.set_test_length(30)     
f.generate_future_dates(90)
f.set_estimator('lstm')


# ## Default Forecasting

# In[13]:


f.manual_forecast(call_me='lstm_default')
f.plot_test_set(ci=True)


# ## General Forecasting with 30 Lags

# In[14]:


f.manual_forecast(call_me='lstm_30lags',lags=30)
f.plot_test_set(ci=True)


# ## General Forecasting with 7 Lags, 5 Epochs

# In[15]:


f.manual_forecast(call_me='lstm_7lags_5epochs',
                  lags=24,
                  epochs=5,
                  validation_split=.2,
                  shuffle=True)
f.plot_test_set(ci=True)


# ## General Forecasting with EarlyStopping

# In[16]:


from tensorflow.keras.callbacks import EarlyStopping
f.manual_forecast(call_me='lstm_30lags_earlystop_8layers',
                  lags=30,
                  epochs=50,
                  validation_split=.2,
                  shuffle=True,
                  callbacks=EarlyStopping(monitor='val_loss',
                                          patience=8),
                  lstm_layer_sizes=(16,16,16),
                  dropout=(0,0,0))
f.plot_test_set(ci=True)


# ## Manual Forecasting

# In[17]:


f.manual_forecast(call_me='lstm_best',
                  lags=36,
                  batch_size=32,
                  epochs=15,
                  validation_split=.2,
                  shuffle=True,
                  activation='tanh',
                  optimizer='Adam',
                  learning_rate=0.001,
                  lstm_layer_sizes=(72,)*4,
                  dropout=(0,)*4,
                  plot_loss=True)
f.plot_test_set(order_by='LevelTestSetMAPE',models='top_2',ci=True)


# <div style="padding:10px;color:black;margin:0;font-size:200%;text-align:center;display:fill;border-radius:5px;background-color:#fcedae;overflow:hidden;font-weight:700;border: 5px solid #f28d27;">📉 Model Benchmarking</div>

# # 📉 8. Model Benchmarking

# In[18]:


f.set_estimator('mlr') # 1. choose the mlr estimator
f.add_ar_terms(7) # 2. add regressors (24 lagged terms)
f.add_seasonal_regressors('month','quarter',dummy=True) # 2.
f.add_seasonal_regressors('year') # 2.
f.add_time_trend() # 2.
f.diff() # 3. difference non-stationary data


# In[19]:


f.manual_forecast()
f.plot_test_set(order_by='LevelTestSetMAPE',models='top_2')


# In[20]:


f.plot(models=['mlr','lstm_best'],
       order_by='LevelTestSetMAPE',
       level=True)


# <div style="padding:10px;color:black;margin:0;font-size:200%;text-align:center;display:fill;border-radius:5px;background-color:#fcedae;overflow:hidden;font-weight:700;border: 5px solid #f28d27;">⏳ Comparing Models</div>

# # ⏳ 9. Comparing Models

# In[21]:


f.export('model_summaries',determine_best_by='LevelTestSetMAPE')[
    ['ModelNickname',
     'LevelTestSetMAPE',
     'LevelTestSetRMSE',
     'LevelTestSetR2',
     'best_model']
]


# <div style="padding:10px;color:black;margin:0;font-size:200%;text-align:center;display:fill;border-radius:5px;background-color:#fcedae;overflow:hidden;font-weight:700;border: 5px solid #f28d27;">📓 Resources</div>

# # 📓 10. Resources 

# - [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
# - [A Gentle Introduction to Long Short-Term Memory Networks by the Experts](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/)
# - [Introduction to Long Short Term Memory (LSTM)](https://www.analyticsvidhya.com/blog/2021/03/introduction-to-long-short-term-memory-lstm/)
# - [What is LSTM? Introduction to Long Short Term Memory](https://intellipaat.com/blog/what-is-lstm/)
# - [Illustrated Guide to LSTM’s and GRU’s: A step by step explanation](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
# - [Long Short-Term Memory](https://www.sciencedirect.com/topics/engineering/long-short-term-memory)
# - [Time Series - LSTM Model](https://www.sciencedirect.com/topics/engineering/long-short-term-memory)
# - [Exploring the LSTM Neural Network Model for Time Series](https://towardsdatascience.com/exploring-the-lstm-neural-network-model-for-time-series-8b7685aa8cf)

# <div style="padding:20px;color:white;margin:20;font-size:270%;text-align:center;display:fill;border-radius:5px;background-color:#cc1100;overflow:hidden;font-weight:700">Please <b>UPVOTE</b> if it helped!</div>
