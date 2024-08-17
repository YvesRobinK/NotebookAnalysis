#!/usr/bin/env python
# coding: utf-8

# # ExtraTrees Regression for the March TPS
# 
# This model is based on my [EDA notebook](https://www.kaggle.com/ambrosm/tpsmar22-eda-which-makes-sense).
# 
# It uses an `ExtraTreesRegressor`. According to the documentation, `ExtraTreesRegressor(criterion=“absolute_error”)` optimizes for MAE, but with this criterion it is unacceptably slow. It is better to optimize the forest for MSE.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
from cycler import cycler
from IPython import display

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

oldcycler = plt.rcParams['axes.prop_cycle']
plt.rcParams['axes.facecolor'] = '#0057b8' # blue
plt.rcParams['axes.prop_cycle'] = cycler(color=['#ffd700'] +
                                         oldcycler.by_key()['color'][1:])


# In[2]:


train_orig = pd.read_csv('../input/tabular-playground-series-mar-2022/train.csv', index_col='row_id', parse_dates=['time'])
test_orig = pd.read_csv('../input/tabular-playground-series-mar-2022/test.csv', index_col='row_id', parse_dates=['time'])
train_orig.shape, test_orig.shape


# # Drop outliers

# In[3]:


# Memorial Day
train_orig = train_orig[(train_orig.time.dt.month != 5) | (train_orig.time.dt.day != 27)]

# July 4
train_orig = train_orig[(train_orig.time.dt.month != 7) | (train_orig.time.dt.day != 4)]

# Labor Day
train_orig = train_orig[(train_orig.time.dt.month != 9) | (train_orig.time.dt.day != 2)]

# Maybe drop some more ...


# # Feature engineering

# In[4]:


# Feature engineering
# Combine x, y and direction into a single categorical feature with 65 unique values
# which can be one-hot encoded
def place_dir(df):
    return df.apply(lambda row: f"{row.x}-{row.y}-{row.direction}", axis=1).values.reshape([-1, 1])

for df in [train_orig, test_orig]:
    df['place_dir'] = place_dir(df)
    


# In[5]:


ohe = OneHotEncoder(drop='first', sparse=False)
ohe.fit(train_orig[['place_dir']])

def engineer(df):
    """Return a new dataframe with the engineered features"""
    
    new_df = pd.DataFrame(ohe.transform(df[['place_dir']]),
                          columns=ohe.categories_[0][1:],
                          index=df.index)
    new_df['saturday'] = df.time.dt.weekday == 5
    new_df['sunday'] = df.time.dt.weekday == 6
    new_df['daytime'] = df.time.dt.hour * 60 + df.time.dt.minute
    new_df['dayofyear'] = df.time.dt.dayofyear # to model the trend
    return new_df


train = engineer(train_orig)
test = engineer(test_orig)

train['congestion'] = train_orig.congestion

features = list(test.columns)
print(list(features))


# # Validation
# 
# Currently I use a set of afternoons in August and September for validation (i.e. a single train-test split rather than cross-validation). This may change in the future.
# 
# I tried HuberRegressor with a low epsilon because this is one of the few linear models which can optimize mean absolute error, but RandomForestRegressor is better. As I said in the introduction, `RandomForestRegressor(criterion=“absolute_error”)` would optimize for MAE, but with this criterion it is unacceptably slow. It is better to optimize the random forest for MSE.
# 
# And `ExtraTreesRegressor` is even better than `RandomForestRegressor`! 
# 
# The bar chart shows that we shouldn't expect much improvement by increasing n_estimators above 1000.

# In[6]:


get_ipython().run_cell_magic('time', '', '# Split into train and test\n# Use all Monday-Wednesday afternoons in August and September for validation\nval_idx = ((train_orig.time.dt.month >= 8) & \n           (train_orig.time.dt.weekday <= 3) &\n           (train_orig.time.dt.hour >= 12)).values\ntrain_idx = ~val_idx\n\nX_tr, X_va = train.loc[train_idx, features], train.loc[val_idx, features]\ny_tr, y_va = train.loc[train_idx, \'congestion\'], train.loc[val_idx, \'congestion\']\n\n# Train and validate the regressor\n#pipe = make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.001, alpha=100))\n# pipe = RandomForestRegressor(n_estimators=0, max_samples=0.03,\n#                             n_jobs=-1, random_state=1)\npipe = ExtraTreesRegressor(n_estimators=0,\n                           #bootstrap=True, max_samples=0.20,\n                           min_samples_split=101,\n                           n_jobs=-1, random_state=1)\nestimators_list, mae_list = [], []\nn_estimators = 4\ninitialized = False\nwhile n_estimators < 256:\n    n_estimators *= 4\n    pipe.set_params(n_estimators=n_estimators,\n                    warm_start=initialized)\n    pipe.fit(X_tr, y_tr)\n    initialized = True\n\n    # Compute the (intermediate) validation score\n    y_va_pred = pipe.predict(X_va)\n    \n    estimators_list.append(pipe.get_params()[\'n_estimators\'])\n    mae_list.append(mean_absolute_error(y_va, y_va_pred))\n    print(f"{estimators_list[-1]:4} estimators:   "\n          f"Validation MAE = {mae_list[-1]:.5f}")\n')


# In[7]:


plt.figure(figsize=(12, 4))
plt.bar(range(len(estimators_list)), mae_list)
plt.xticks(range(len(estimators_list)), estimators_list)
plt.ylim(6.10, 6.40)
plt.ylabel('Validation MAE')
plt.xlabel('n_estimators')
plt.show()


# # Re-training and submission
# 
# We retrain the classifier on the complete training data, compute the test predictions and then postprocess two special cases (see the [EDA](https://www.kaggle.com/ambrosm/tpsmar22-eda-which-makes-sense) for the special congestion values).

# In[8]:


get_ipython().run_cell_magic('time', '', "# Retrain the classifier on the complete training data (except outliers)\npipe.set_params(warm_start=False)\npipe.fit(train[features], train.congestion)\ntest['congestion'] = pipe.predict(test[features]).round(0).astype(int)\nassert test.congestion.min() >= 0\nassert test.congestion.max() <= 100\nsub = test.reset_index()[['row_id', 'congestion']]\n\nsub.to_csv('submission_extratrees.csv', index=False)\nsub\n")


# In[9]:


# Plot the distribution of the test predictions
# compared to the Monday afternoons in August and September
plt.figure(figsize=(16,3))
plt.hist(train.congestion[((train_orig.time.dt.month >= 8) & 
                           (train_orig.time.dt.weekday == 0) &
                           (train_orig.time.dt.hour >= 12)).values],
         bins=np.linspace(-0.5, 100.5, 102),
         density=True, label='Validation',
         color='#ffd700')
plt.hist(sub['congestion'], np.linspace(-0.5, 100.5, 102),
         density=True, rwidth=0.5, label='Test predictions',
         color='r')
plt.xlabel('Congestion')
plt.ylabel('Frequency')
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
plt.legend()
plt.show()


# In[ ]:




