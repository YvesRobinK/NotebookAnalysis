#!/usr/bin/env python
# coding: utf-8

# # TPS-Jan22 🎉 | EDA + FE + Simple CatBoost

# ### Happy New Year!
# May this year bring good health, happiness and success at whatever you choose to accomplish!
# 
# ___
# 
# # 📌 Introduction
# 
# >Hello Kagglers,
# >
# >This notebook is a simple implementation of a CatBoost Regressor, as well as some EDA, feature engineering, cross-validation explanation and a SMAPE function.
# >
# >As a beginner and newcomer, making this first notebook as public is a milestone for me. I believe that there is no better way to improve than to share and report on your knowledge, investigations and achievements.
# >
# >May it be interesting and useful to you. Do not hesitate to provide feedback!
# 
# 
# # 📝 Agenda
# >1. [📚 Loading libraries and files](#Loading)
# >2. [🔍 Exploratory Data Analysis](#EDA)
# >3. [⚙️ Feature Engineering](#FeatureEngineering)
# >4. [✅ Cross-validation Method](#Validation)
# >5. [🏋️ Model Training & Inference](#TrainingInference)

# ___
# # <a name="Loading">📚 Loading libraries and files</a>

# In[1]:


import os
import warnings

import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

import math
from pathlib import Path

# Mute warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().system('tree ../input/')


# In[3]:


data_dir = Path('../input/tabular-playground-series-jan-2022')
holiday_dir = Path('../input/public-and-unofficial-holidays-nor-fin-swe-201519')
gdp_dir = Path('../input/gdp-20152019-finland-norway-and-sweden')

train = pd.read_csv(
    data_dir / 'train.csv',
    dtype={
        'country': 'category',
        'store': 'category',
        'product': 'category',
        'num_sold': 'float32',
    },
    index_col='row_id'
)

test = pd.read_csv(
    data_dir / "test.csv",
    dtype={
        'country': 'category',
        'store': 'category',
        'product': 'category',
    },
    index_col='row_id'
)

target_col = train.columns.difference(test.columns)[0]

holiday_data = pd.read_csv(holiday_dir / 'holidays.csv')

gdp = pd.read_csv(
    gdp_dir / 'GDP_data_2015_to_2019_Finland_Norway_Sweden.csv', index_col='year')


# ___
# # <a name="EDA">🔍 Exploratory Data Analysis</a>

# ### <a name="FeatureAnalysis">Feature Analysis</a>

# First, let's have a glance at some basic information about our data.

# In[4]:


train.info()


# Note that the <code>date</code> feature is originally <code>str</code>-typed, so we will convert it to <code>datetime</code> to make any further process easier with *pandas*.
# 
# However, before converting <code>date</code> values, let's see if all of the values are, ideally, following the same <code>month/day/four-digit year</code> format. We can get an idea of how widespread this issue is by checking the length of each entry in the <code>date</code> column.

# In[5]:


def len_data_count(column):
    return column.str.len().value_counts()

print(len_data_count(train.date))
print(len_data_count(test.date))


# It looks like all values are 10-characters long, which is good news. We can now convert our column.

# In[6]:


train['date'] = pd.to_datetime(train['date'])
test['date']  = pd.to_datetime(test['date'])


# Moreover, it looks like there is no missing values in any field.
# 
# Want to make sure about it? Alright.

# In[7]:


display(train.isnull().sum(), test.isnull().sum())


# We now have the confirmation.
# 
# Afterwards, let's look at the **cardinality** of each column.

# In[8]:


display(train.iloc[:,1:-1].nunique(), test.iloc[:,1:-1].nunique())


# Then, it would be relevant to count each of these values' occurrences.<br />
# At least for the categorical features, since the <code>date</code> column will be the subject of a later treatment.

# In[9]:


# Count for each unique values
categorical_cols = train.select_dtypes('category').columns.tolist()

for col in categorical_cols:
    display(pd.DataFrame(train[col].value_counts()))


# Well, the least that can be said is that **all features are balanced!**

# ___
# # <a name="FeatureEngineering">⚙️ Feature Engineering</a>

# 📌 This part has been updated and largely inspired by these two notebooks:
# > * [TPSJAN22-03 Linear Model](https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model) & [TPSJAN22-06 LightGBM Quickstart](https://www.kaggle.com/ambrosm/tpsjan22-06-lightgbm-quickstart) by [AmbrosM](https://www.kaggle.com/ambrosm)<br />
# > * [TPS Jan 22 - EDA + modelling](https://www.kaggle.com/samuelcortinhas/tps-jan-22-eda-modelling) by [Samuel Cortinhas](https://www.kaggle.com/samuelcortinhas)

# We are dealing with time-series data, therefore it is relevant to consider the impact of holidays, which naturally play a large role in business activities.

# In[10]:


import dateutil.easter as easter

def holiday_features(holiday_df, df):
    
    fin_holiday = holiday_df.loc[holiday_df.country == 'Finland']
    swe_holiday = holiday_df.loc[holiday_df.country == 'Sweden']
    nor_holiday = holiday_df.loc[holiday_df.country == 'Norway']
    
    df['fin holiday'] = df.date.isin(fin_holiday.date).astype(int)
    df['swe holiday'] = df.date.isin(swe_holiday.date).astype(int)
    df['nor holiday'] = df.date.isin(nor_holiday.date).astype(int)
    
    df['holiday'] = np.zeros(df.shape[0]).astype(int)
    
    df.loc[df.country == 'Finland', 'holiday'] = df.loc[df.country == 'Finland', 'fin holiday']
    df.loc[df.country == 'Sweden', 'holiday'] = df.loc[df.country == 'Sweden', 'swe holiday']
    df.loc[df.country == 'Norway', 'holiday'] = df.loc[df.country == 'Norway', 'nor holiday']
    
    df.drop(['fin holiday', 'swe holiday', 'nor holiday'], axis=1, inplace=True)
    
    gdp_exponent = 1.2121103201489674
    # c.f https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model
    
    # GDP features
    def get_gdp(row):
        """Return the GDP based on row.country and row.date.year"""
        country = 'GDP_' + row.country
        
        return gdp.loc[row.date.year, country] ** gdp_exponent
    
    df['gdp'] = pd.DataFrame(df.apply(get_gdp, axis=1))
    
    
    # Easter
    easter_date = df.date.apply(lambda date: pd.Timestamp(easter.easter(date.year)))
    df['days_from_easter'] = (df.date - easter_date).dt.days.clip(-5, 65)
    
    # Last Sunday of May (Mother's Day)
    sun_may_date = df.date.dt.year.map({
        2015: pd.Timestamp(('2015-5-31')),
        2016: pd.Timestamp(('2016-5-29')),
        2017: pd.Timestamp(('2017-5-28')),
        2018: pd.Timestamp(('2018-5-27')),
        2019: pd.Timestamp(('2019-5-26'))
    })
    #new_df['days_from_sun_may'] = (df.date - sun_may_date).dt.days.clip(-1, 9)
    
    # Last Wednesday of June
    wed_june_date = df.date.dt.year.map({
        2015: pd.Timestamp(('2015-06-24')),
        2016: pd.Timestamp(('2016-06-29')),
        2017: pd.Timestamp(('2017-06-28')),
        2018: pd.Timestamp(('2018-06-27')),
        2019: pd.Timestamp(('2019-06-26'))
    })
    df['days_from_wed_jun'] = (df.date - wed_june_date).dt.days.clip(-5, 5)
    
    # First Sunday of November (second Sunday is Father's Day)
    sun_nov_date = df.date.dt.year.map({
        2015: pd.Timestamp(('2015-11-1')),
        2016: pd.Timestamp(('2016-11-6')),
        2017: pd.Timestamp(('2017-11-5')),
        2018: pd.Timestamp(('2018-11-4')),
        2019: pd.Timestamp(('2019-11-3'))
    })
    df['days_from_sun_nov'] = (df.date - sun_nov_date).dt.days.clip(-1, 9)
    
    return df

train = holiday_features(holiday_data, train)
test  = holiday_features(holiday_data, test)


# Next, the cardinality of each categorical feature is quite low, and that we do not want to impose an ordinal order, **one-hot encoding** may be a good way to encode our categorical features.

# In[11]:


train = pd.get_dummies(train, columns=categorical_cols)
test  = pd.get_dummies(test, columns=categorical_cols)


# Since we have a <code>date</code>-typed feature here, and models are rarely able to use dates and times as they are, we would benefit from encoding it as categorical variables as this can often yield useful information about temporal patterns.
# 
# Furthermore, time-series data (such as product sales) often have distributions that differs from week days to week-ends for example, it is likely that using the day of the week as a new feature is a relevant option we have.

# In[12]:


# Nothing to see here!
# Copy 'date' feature for further visualization/explanation
date_copy = train.date


# In[13]:


def new_date_features(df):
    df['year'] = df.date.dt.year 
    df['quarter'] = df.date.dt.quarter
    df['month'] = df.date.dt.month  
    df['week'] = df.date.dt.week 
    df['day'] = df.date.dt.day  
    df['weekday'] = df.date.dt.weekday
    df['day_of_week'] = df.date.dt.dayofweek  
    df['day_of_year'] = df.date.dt.dayofyear  
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_month'] = df.date.dt.days_in_month  
    df['is_weekend'] = np.where((df['weekday'] == 5) | (df['weekday'] == 6), 1, 0)
    df['is_friday'] = np.where((df['weekday'] == 4), 1, 0)
    
    df.drop('date', axis=1, inplace=True)
    
    return df
    
train = new_date_features(train)
test  = new_date_features(test)


# Finally, here are our datasets, before moving to the cross-validation step.

# In[14]:


# Target transformation
y = np.log1p(train[target_col] / train.gdp)

train.drop(target_col, axis=1, inplace=True)
train


# ___
# # <a name="Validation">✅ Cross-validation method</a>

# As afore-mentionned, we are dealing with time-series data.<br />
# Thus, we do not want to use information about the future to train our model. We will therefore opt for <code>TimeSeriesSplit</code> as a our **cross-validation** technique.
# 
# 📌 According to the *scikit-learn* documentation:
# >[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) provides train/test indices to split time series data samples that are observed at fixed time intervals, in train/test sets. In each split, test indices must be higher than before, and thus shuffling in cross validator is inappropriate.

# In[15]:


# Function modified from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html
# Inspired by https://www.kaggle.com/tomwarrens/timeseriessplit-how-to-use-it/notebook

from matplotlib.patches import Patch

def plot_cv_indices(cv, X, y, n_splits, date_col=None):
    """Create a sample plot for indices of a cross-validation object."""
    
    fig, ax = plt.subplots(1, 1, figsize = (12, 8))

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=10,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
            zorder=2
        )

    # Formatting
    yticklabels = list(range(n_splits))
    
    if date_col is not None:
        tick_locations  = ax.get_xticks()
        tick_dates = [" "] + date_col.iloc[list(tick_locations[1:-1])].astype(str).tolist() + [" "]

        tick_locations_str = [str(int(i)) for i in tick_locations]
        new_labels = ['\n\n'.join(x) for x in zip(list(tick_locations_str), tick_dates)]
        
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(new_labels)
    
    # Custom visualization
    ax.set_facecolor('#fcfcfc')
    ax.grid(alpha=0.7, linewidth=1, zorder=0)
    
    ax.set_yticks(np.arange(n_splits) + .5)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel('CV iteration', fontsize=15, labelpad=10)
    ax.set_ylim([n_splits+0.2, -.2])
    ax.yaxis.set_tick_params(labelsize=12, pad=10, length=0)
    
    ax.set_xlabel('Sample index', fontsize=15, labelpad=10)
    ax.xaxis.set_tick_params(labelsize=12, pad=10, length=0)
    
    ax.legend(
        [
            Patch(color=cmap_cv(.8)), 
            Patch(color=cmap_cv(.02))
        ],
        [
            'Testing set', 
            'Training set'
        ],
        fontsize=12,
        loc=(1.02, .8)
    )
    
    ax.set_title(
        '{}'.format(type(cv).__name__),
        loc="left", 
        color="#000", 
        fontsize=20, 
        pad=5, 
        y=1, 
        zorder=3
    )
    
    return ax


# In[16]:


from sklearn.model_selection import TimeSeriesSplit

folds = TimeSeriesSplit(n_splits=4)

# Visualization
cmap_cv = plt.cm.bwr
plot_cv_indices(folds, train, y, folds.n_splits, date_col=date_copy);


# ___
# # <a name="TrainingInference">🏋️ Model Training & Inference</a>

# Submissions are evaluated on SMAPE between forecasts and actual values.

# ![SMAPE formula](https://media.geeksforgeeks.org/wp-content/uploads/20211120224204/smapeformula.png)

# In[17]:


def smape(actual, predicted):
    numerator = np.abs(predicted - actual)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    
    return np.mean(numerator / denominator)*100


# ### Training phase

# **Tip:** Since the SMAPE evaluation metric is asymmetric. In this case, underestimated values are much more penalized than overestimated values. Then, feel free to round your predictions **up** to the nearest value.<br />
# <br />
# 📌 You will find more by having a glance to these awesome notebooks: 
# > * [SMAPE Weirdness](https://www.kaggle.com/cpmpml/smape-weirdness) by [CPMP](https://www.kaggle.com/cpmpml)
# > * [TPS Jan 2022: A simple average model (no ML)](https://www.kaggle.com/carlmcbrideellis/tps-jan-2022-a-simple-average-model-no-ml) by [Carl McBride Ellis](https://www.kaggle.com/carlmcbrideellis).
# >
# >The last one being related to this very Tabular Playground.

# In[18]:


from catboost import CatBoostRegressor

y_pred = np.zeros(len(test))
scores = []

for fold, (train_id, test_id) in enumerate(folds.split(train, groups=date_copy.dt.year)):
    print("Fold: ", fold)
    
    # Splitting
    X_train, y_train = train.iloc[train_id], y.iloc[train_id]
    X_valid, y_valid = train.iloc[test_id], y.iloc[test_id]
    
    # Model with parameters
    params = {
        'iterations': 10000,
        'depth': 5, 
        'l2_leaf_reg': 12.06,
        'bootstrap_type': 'Bayesian',
        'boosting_type': 'Plain',
        'loss_function': 'MAE',
        'eval_metric': 'SMAPE',
        'od_type': 'Iter',       # type of overfitting detector
        'od_wait': 40,
        'has_time': True         # use the order of the data (ts), do not permute
    }
    
    model = CatBoostRegressor(**params)

    # Training
    model.fit(
        X_train, y_train, 
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=1000,
        verbose=1000
    )
    
    print('\n')
    
    # Evaluation
    valid_pred = model.predict(X_valid)
    
    valid_score = smape(
        np.expm1(y_valid) * X_valid.gdp.values, 
        np.ceil(np.expm1(valid_pred) * X_valid.gdp.values)
    )
    
    scores.append(valid_score)
    
    # Prediction for submission
    y_pred += (np.expm1(model.predict(test)) * test.gdp.values) / folds.n_splits


# ### Evaluation

# Next, we can evaluate the model thanks to our custom SMAPE function.

# In[19]:


score = np.array(scores).mean()
print('Mean SMAPE score: ', score)


# ### Submission

# In[20]:


submission = pd.read_csv('../input/tabular-playground-series-jan-2022/sample_submission.csv')
submission.num_sold = np.ceil(y_pred) # rounding up
submission


# In[21]:


submission.to_csv('submission.csv', index=False)

