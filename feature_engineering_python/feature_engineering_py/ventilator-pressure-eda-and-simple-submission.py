#!/usr/bin/env python
# coding: utf-8

# # [Ventilator Pressure Prediction](https://www.kaggle.com/c/ventilator-pressure-prediction): EDA and a simple submission
# 
# ### Summary
# In this competition we are provided with 75,450 non-contiguous cycles (each cycle is uniquely labelled with an individual `breath_id`) of the [PVP1 automated ventilator](https://www.peoplesvent.org/en/latest/) connected to a high-grade test lung ([Quicklung, Ingmar Medical](https://www.ingmarmed.com/product/quicklung/))  Three different values of the compliance (C) were tested [10,20,50] mL cm H<sub>2</sub>O in conjunction with three different values of resistance (R) [5,20,50] cm H<sub>2</sub>O/L/s, resulting in a total of 9 different lung settings.
# 
# A typical breath cycle has the following aspect 
# 
# ![](https://raw.githubusercontent.com/Carl-McBride-Ellis/images_for_kaggle/main/PVP1_typical_cycle.png)
# 
# A cycle lasts for up to 3 seconds. It is the inspiratory section (from 0-1 seconds) that we model in this competition.
# 
# When it comes to model evaluation we have to predict the `pressure` for 50,300 test cycles, of which 19% are assigned to the Public Leaderboard, and the remaining 81% to the Private Leaderboard. It is the mean absolute error (`mae`) between the predicted and actual pressures during the inspiratory phase of each breath that constitutes the evaluation metric in this competition.
# 
# ### Read in the data

# In[1]:


import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
plt.style.use('fivethirtyeight')
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


train_data = pd.read_csv('../input/ventilator-pressure-prediction/train.csv',index_col=0)
test_data  = pd.read_csv('../input/ventilator-pressure-prediction/test.csv', index_col=0)
sample     = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')


# Let us take a quick look at the training data

# In[3]:


train_data


# How many unique values do we have for each feature?

# In[4]:


train_data.nunique().to_frame()


# and the test data

# In[5]:


test_data.nunique().to_frame()


# We can see that we have over 6 million rows of training data, corresponding to 75,450 breaths, and 50,300 breaths in the test dataset. On average we have 80 time steps of data per breath. Let us check this for the training data

# In[6]:


train_data.groupby("breath_id")["time_step"].count().unique().item()


# and the test data

# In[7]:


test_data.groupby("breath_id")["time_step"].count().unique().item()   


# The next question is whether we have any missing data or not?

# In[8]:


train_data.isnull().sum(axis = 0).to_frame()


# Wonderful, it seems not!
# 
# # Time
# In this data the unit of time is seconds. How long does longest breath last?

# In[9]:


train_data.time_step.max()


# The longest breath is just under 3 seconds.

# What is the maximum time that the exploratory solenoid valve is set to 0?

# In[10]:


train_data.query('u_out == 0').time_step.max()


# The valve seems to be activated after 1 seccond.
# # The first breath
# 
# Let us select `breath_id=1` and take a look at the features

# In[11]:


breath_one = train_data.query('breath_id == 1').reset_index(drop = True)
breath_one


# Let us see how many unique values there are in each of these columns

# In[12]:


breath_one.nunique().to_frame()


# there is only one value for `R`, one value for `C` for the `breath_id`. 
# 
# Let us visualize `u_in`, `u_out` and `pressure` with respect to the `time_step`:

# In[13]:


breath_one.plot(x="time_step", y="u_in", kind='line',figsize=(12,3), lw=2, title="u_in");
breath_one.plot(x="time_step", y="u_out", kind='line',figsize=(12,3), lw=2, title="u_out");
breath_one.plot(x="time_step", y="pressure", kind='line',figsize=(12,3), lw=2, title="pressure");


# # All breaths
# What values do we have for `R`, which represents how restricted the airway is (in cmH<sub>2</sub>O/L/S).

# In[14]:


train_data.R.value_counts().to_frame()


# now for the values of `C`, the lung attribute indicating how compliant the lung is (in mL/cmH<sub>2</sub>O)

# In[15]:


train_data.C.value_counts().to_frame()


# thus we have nine combinations of `R` and `C`. Let us look at a count of each of these combinations in the training data (dividing by 80 to account for the number time steps in each breath)

# In[16]:


pd.crosstab(train_data["R"],train_data["C"]) /80


# and similarly for the test data

# In[17]:


pd.crosstab(test_data["R"],test_data["C"]) /80


# We also have `u_out`, the control input for the exploratory solenoid valve. Either 0 or 1.

# In[18]:


train_data.u_out.value_counts().to_frame()


# # Pressure
# And now we shall look at the `pressure`. The pressure is measured in cmH<sub>2</sub>0, where 1 cmH<sub>2</sub>0 is roughly equal to 98 Pascals. The global peak inspiratory pressure (PIP) in the training data is

# In[19]:


train_data.pressure.max()


# This value is safely below the point where pressure relief valve opens (at 70 cmH<sub>2</sub>0) in order to prevent excessive pressures in the lung, thus reducing any barotrauma risk.
# 
# The pressures in the training data have the following distribution

# In[20]:


plt.figure(figsize = (12,5))
ax = sns.distplot(train_data['pressure'], 
             bins=120, 
             kde_kws={"clip":(0,40)}, 
             hist_kws={"range":(0,40)},
             color='darkcyan', 
             kde=False);
values = np.array([rec.get_height() for rec in ax.patches])
norm = plt.Normalize(values.min(), values.max())
colors = plt.cm.jet(norm(values))
for rec, col in zip(ax.patches, colors):
    rec.set_color(col)
plt.xlabel("Histogram of pressures", size=14)
ax.set(yticklabels=[])
plt.show();


# with a median value of 

# In[21]:


train_data.pressure.median()


# Note however that in this competition the expiratory phase is not scored, so for practical purposes we are only really interested in the pressure for `u_out=0`, *i.e.* the first second of the experiments:

# In[22]:


u_out_is_zero = train_data.query("u_out == 0").reset_index(drop = True)
plt.figure(figsize = (12,5))
ax = sns.distplot(u_out_is_zero['pressure'], 
             bins=120, 
             kde_kws={"clip":(0,50)}, 
             hist_kws={"range":(0,50)},
             color='darkcyan', 
             kde=False);
values = np.array([rec.get_height() for rec in ax.patches])
norm = plt.Normalize(values.min(), values.max())
colors = plt.cm.jet(norm(values))
for rec, col in zip(ax.patches, colors):
    rec.set_color(col)
plt.xlabel("Histogram of pressures (u_out=0)", size=14)
ax.set(yticklabels=[])
plt.show();


# with a median value of 

# In[23]:


u_out_is_zero.pressure.median()


# We have nine combinations of experiments; `C` can be 10, 20 or 50, and `R` can be 5, 20 or 50. Lets take a quick look at an example of each

# In[24]:


breath_2 = train_data.query('breath_id == 2').reset_index(drop = True)
breath_3 = train_data.query('breath_id == 3').reset_index(drop = True)
breath_4 = train_data.query('breath_id == 4').reset_index(drop = True)
breath_5 = train_data.query('breath_id == 5').reset_index(drop = True)
breath_17 = train_data.query('breath_id == 17').reset_index(drop = True)
breath_18 = train_data.query('breath_id == 18').reset_index(drop = True)
breath_21 = train_data.query('breath_id == 21').reset_index(drop = True)
breath_39 = train_data.query('breath_id == 39').reset_index(drop = True)

fig, axes = plt.subplots(3,3,figsize=(15,15))
sns.lineplot(data=breath_39, x="time_step", y="pressure", lw=2, ax=axes[0,0])
axes[0,0].set_title ("R=5, C=10", fontsize=18)
axes[0,0].set(xlabel='')
#axes[0,0].set(ylim=(0, None))
sns.lineplot(data=breath_21, x="time_step", y="pressure",  lw=2, ax=axes[0,1])
axes[0,1].set_title ("R=20, C=10", fontsize=18)
axes[0,1].set(xlabel='')
axes[0,1].set(ylabel='')
#axes[0,1].set(ylim=(0, None))
sns.lineplot(data=breath_18, x="time_step", y="pressure",  lw=2,ax=axes[0,2])
axes[0,2].set_title ("R=50, C=10", fontsize=18)
axes[0,2].set(xlabel='')
axes[0,2].set(ylabel='')
#axes[0,2].set(ylim=(0, None))
sns.lineplot(data=breath_17, x="time_step", y="pressure",  lw=2,ax=axes[1,0])
axes[1,0].set_title ("R=5, C=20", fontsize=18)
axes[1,0].set(xlabel='')
#axes[1,0].set(ylim=(0, None))
sns.lineplot(data=breath_2, x="time_step", y="pressure",  lw=2,ax=axes[1,1])
axes[1,1].set_title ("R=20, C=20", fontsize=18)
axes[1,1].set(xlabel='')
axes[1,1].set(ylabel='')
#axes[1,1].set(ylim=(0, None))
sns.lineplot(data=breath_3, x="time_step", y="pressure",  lw=2,ax=axes[1,2])
axes[1,2].set_title ("R=50, C=20", fontsize=18)
axes[1,2].set(xlabel='')
axes[1,2].set(ylabel='')
#axes[1,2].set(ylim=(0, None))
sns.lineplot(data=breath_5, x="time_step", y="pressure",  lw=2,ax=axes[2,0])
axes[2,0].set_title ("R=5, C=50", fontsize=18)
#axes[2,0].set(ylim=(0, None))
sns.lineplot(data=breath_one, x="time_step", y="pressure",  lw=2,ax=axes[2,1])
axes[2,1].set_title ("R=20, C=50", fontsize=18)
axes[2,1].set(ylabel='')
#axes[2,1].set(ylim=(0, None))
sns.lineplot(data=breath_4, x="time_step", y="pressure",  lw=2,ax=axes[2,2])
axes[2,2].set_title ("R=50, C=50", fontsize=18)
axes[2,2].set(ylabel='')
#axes[2,2].set(ylim=(0, None))

plt.show();


# # Positive end-expiratory pressure (PEEP)
# It is worth noting that even before the experiments start (*i.e.* the `time_step=0` and `u_in=0`) there is a positive pressure in the airway. The system is maintained above atmospheric pressure to promote gas exchange to the lungs.

# In[25]:


zero_time = train_data.query("time_step < 0.000001 & u_in < 0.000001").reset_index(drop = True)
zero_time_5_10  = zero_time.query("R ==  5 & C == 10").reset_index(drop = True)
zero_time_5_20  = zero_time.query("R ==  5 & C == 20").reset_index(drop = True)
zero_time_5_50  = zero_time.query("R ==  5 & C == 50").reset_index(drop = True)
zero_time_20_10 = zero_time.query("R == 20 & C == 10").reset_index(drop = True)
zero_time_20_20 = zero_time.query("R == 20 & C == 20").reset_index(drop = True)
zero_time_20_50 = zero_time.query("R == 20 & C == 50").reset_index(drop = True)
zero_time_50_10 = zero_time.query("R == 50 & C == 10").reset_index(drop = True)
zero_time_50_20 = zero_time.query("R == 50 & C == 20").reset_index(drop = True)
zero_time_50_50 = zero_time.query("R == 50 & C == 50").reset_index(drop = True)

fig, axes = plt.subplots(9,1,figsize=(12,15))
sns.violinplot(x=zero_time_5_10["pressure"], linewidth=2, ax=axes[0], color="indianred")
axes[0].set_title ("R=5, C=10", fontsize=14)
axes[0].set(xlim=(3, 8))
sns.violinplot(x=zero_time_5_20["pressure"], linewidth=2, ax=axes[1], color="firebrick")
axes[1].set_title ("R=5, C=20", fontsize=14)
axes[1].set(xlim=(3, 8))
sns.violinplot(x=zero_time_5_50["pressure"], linewidth=2, ax=axes[2], color="darkred" )
axes[2].set_title ("R=5, C=50", fontsize=14)
axes[2].set(xlim=(3, 8))
sns.violinplot(x=zero_time_20_10["pressure"], linewidth=2, ax=axes[3], color="greenyellow")
axes[3].set_title ("R=20, C=10", fontsize=14)
axes[3].set(xlim=(3, 8))
sns.violinplot(x=zero_time_20_20["pressure"], linewidth=2, ax=axes[4], color="olivedrab")
axes[4].set_title ("R=20, C=20", fontsize=14)
axes[4].set(xlim=(3, 8))
sns.violinplot(x=zero_time_20_50["pressure"], linewidth=2, ax=axes[5], color="olive" )
axes[5].set_title ("R=20, C=50", fontsize=14)
axes[5].set(xlim=(3, 8))
sns.violinplot(x=zero_time_50_10["pressure"], linewidth=2, ax=axes[6], color="steelblue")
axes[6].set_title ("R=50, C=10", fontsize=14)
axes[6].set(xlim=(3, 8))
sns.violinplot(x=zero_time_50_20["pressure"], linewidth=2, ax=axes[7], color="cornflowerblue")
axes[7].set_title ("R=50, C=20", fontsize=14)
axes[7].set(xlim=(3, 8))
sns.violinplot(x=zero_time_50_50["pressure"], linewidth=2, ax=axes[8], color="midnightblue" )
axes[8].set_title ("R=50, C=50", fontsize=14)
axes[8].set(xlim=(3, 8));


# The average value of PEEP at the beginning of each cycle is

# In[26]:


zero_time["pressure"].mean()


# Note that not all cycles start with `u_in=0`, and a cycle can even start with the inspiratory solenoid valve set to the maximum value of 100.
# # Exploratory perturbation policies
# In the very interesting paper ["*Machine Learning for Mechanical Ventilation Control*"](https://arxiv.org/pdf/2102.06779.pdf) whose lead author is [Daniel Suo](https://www.kaggle.com/danielsuo), the host of this competition, they describe their experiments, examples of which can also be found in our dataset:

# In[27]:


breath_3034 = train_data.query('breath_id == 3034').reset_index(drop = True)
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.plot(breath_3034["time_step"],breath_3034["u_in"], lw=2, label='u_in')
ax.plot(breath_3034["time_step"],breath_3034["pressure"], lw=2, label='pressure')
#ax.set(xlim=(0,1))
ax.legend(loc="upper right")
ax.set_title("Boundary exploration policy", fontsize=14)
ax.set_xlabel("Time (s)", fontsize=14)
plt.show();


# In[28]:


breath_3101 = train_data.query('breath_id == 3101').reset_index(drop = True)
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.plot(breath_3101["time_step"],breath_3101["u_in"], lw=2, label='u_in')
ax.plot(breath_3101["time_step"],breath_3101["pressure"], lw=2, label='pressure')
#ax.set(xlim=(0,1))
ax.legend(loc="upper right")
ax.set_title("Triangular exploration policy", fontsize=14)
ax.set_xlabel("Time (s)", fontsize=14)
plt.show();


# # Negative pressure
# The minimum value for the pressure where `u_in=0` at `time_step=0` is

# In[29]:


zero_time[zero_time['pressure']==zero_time['pressure'].min()]


# Both of these breaths have a somewhat unusual aspect

# In[30]:


breath_542 = train_data.query('breath_id == 542').reset_index(drop = True)
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(breath_542["time_step"],breath_542["u_in"], lw=2, label='u_in')
ax.plot(breath_542["time_step"],breath_542["pressure"], lw=2, label='pressure')
#ax.set(xlim=(0,1))
ax.legend(loc="upper right")
ax.set_xlabel("time_id", fontsize=14)
ax.set_title("breath_id = 542", fontsize=14)
plt.show();

breath_119582 = train_data.query('breath_id == 119582').reset_index(drop = True)
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(breath_119582["time_step"],breath_119582["u_in"], lw=2, label='u_in')
ax.plot(breath_119582["time_step"],breath_119582["pressure"], lw=2, label='pressure')
#ax.set(xlim=(0,1))
ax.legend(loc="upper right")
ax.set_xlabel("time_id", fontsize=14)
ax.set_title("breath_id = 119582", fontsize=14)
plt.show();


# Note that all of the instances of negative pressure occur only in the `R=50` (high restriction) with `C=10` (thick latex) systems.
# # Simple feature engineering
# We shall add a new feature, which is the [cumulative sum](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.cumsum.html) of the `u_in` feature:

# In[31]:


train_data['u_in_cumsum'] = (train_data['u_in']).groupby(train_data['breath_id']).cumsum()
test_data['u_in_cumsum']  = (test_data['u_in']).groupby(test_data['breath_id']).cumsum()


# The thinking behind this feature is that it is reasonable to assume the pressure in the lungs is approximately proportional to how much air has actually been pumped into them. It goes almost without saying that this feature is not useful when breathing out, but given that the expiratory phase is not scored in this competition this should not be too much of a problem.
# 
# ### Shifting `u_in`
# Let us take a look at the first second of `breath_id=928`, which is an excellent example of an oscillatory experiment

# In[32]:


breath_928 = train_data.query('breath_id == 928').reset_index(drop = True)
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
ax.plot(breath_928["time_step"],breath_928["u_in"], lw=2, label='u_in')
ax.plot(breath_928["time_step"],breath_928["pressure"], lw=2, label='pressure')
ax.set(xlim=(0,1))
ax.legend(loc="upper right")
ax.set_xlabel("time_id", fontsize=14)
plt.show();


# It can be observed that there is a lag between `u_in` and the resulting `pressure` of around 0.1 seconds. I am sure it is with this in mind that [Chun Fu](https://www.kaggle.com/patrick0302) wrote his excellent notebook ["*Add lag u_in as new feat*"](https://www.kaggle.com/patrick0302/add-lag-u-in-as-new-feat/notebook), which introduces a new *shifted* `u_in` feature. Here we shall use a shift of 2 rather than his original shift of 1, which is now more in line with the delay seen:

# In[33]:


train_data['u_in_shifted'] = train_data.groupby('breath_id')['u_in'].shift(2).fillna(method="backfill")
test_data['u_in_shifted']  = test_data.groupby('breath_id')['u_in'].shift(2).fillna(method="backfill")


# ### Descriptive statistics of `u_in`
# Again inspired by the work of Chun Fu, this time in his notebook ["*Add last u_in as new feat*"](https://www.kaggle.com/patrick0302/add-last-u-in-as-new-feat/) it is found, at least with gradient boosting type models, that providing the estimator with some descriptive statistics regarding `u_in` for the cycle in question seems to help in improving the model. Here are a number of examples, some of which may (or may not) be useful:

# In[34]:


for df in (train_data, test_data):
    df['u_in_first']  = df.groupby('breath_id')['u_in'].transform('first')
    df['u_in_min']    = df.groupby('breath_id')['u_in'].transform('min')
    df['u_in_mean']   = df.groupby('breath_id')['u_in'].transform('mean')
    df['u_in_median'] = df.groupby('breath_id')['u_in'].transform('median')
    df['u_in_max']    = df.groupby('breath_id')['u_in'].transform('max')
    df['u_in_last']   = df.groupby('breath_id')['u_in'].transform('last')


# # A simple submission

# In[35]:


X_train = train_data.drop(['pressure'], axis=1)
y_train = train_data['pressure']
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble     import HistGradientBoostingRegressor
regressor  =  HistGradientBoostingRegressor(max_iter=100,
     loss="least_absolute_deviation",early_stopping=False)
regressor.fit(X_train, y_train)
sample["pressure"] = regressor.predict(test_data)
sample.to_csv('submission.csv',index=False)


# Another approach would be to treat the data as a time series and use a [Temporal Convolutional Network (TCN)](https://www.kaggle.com/carlmcbrideellis/temporal-convolutional-network-using-keras-tcn).
# # Related reading
# * [The People's Ventilator Project](https://www.peoplesvent.org/en/latest/) Home page
# * [The People's Ventilator Project](https://github.com/cohenlabprinceton/pvp) GitHub material
# * [Julienne LaChance, Tom J. Zajdel, Manuel Schottdorf, Jonny L. Saunders, Sophie Dvali, Chase Marshall, Lorenzo Seirup, Daniel A. Notterman, and Daniel J. Cohen "*PVP1–The People’s Ventilator Project: A fully open, low-cost, pressure-controlled ventilator*", medRxiv doi:10.1101/2020.10.02.20206037 October 5 (2020)](https://www.medrxiv.org/content/10.1101/2020.10.02.20206037v1.full.pdf)
# * [QuickLung ventilator](https://www.ingmarmed.com/product/quicklung/)
# * [Dean R. Hess "*Respiratory Mechanics in Mechanically Ventilated Patients*", Respiratory Care November **vol 59** pp. 1773-1794 (2014)](http://rc.rcjournal.com/content/59/11/1773)
# * [Daniel Suo *et al.* "*Machine Learning for Mechanical Ventilation Control*", arXiv:2102.06779
#  (2021)](https://arxiv.org/pdf/2102.06779.pdf)
