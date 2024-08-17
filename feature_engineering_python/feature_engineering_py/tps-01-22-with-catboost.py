#!/usr/bin/env python
# coding: utf-8

# # TPS-01-22 with Catboost
# 
# ## Overview
# In this notebook I will build a Catboost Model for [Tabular Playground Series - Jan 2022 Competition](https://www.kaggle.com/c/tabular-playground-series-jan-2022). Before Modeling, I will also perform some Exploratory data analysis and feature engineering to find insights.

# ## Import datasets

# In[1]:


import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt


# In[2]:


class Config:
    input_path = "../input/tabular-playground-series-jan-2022"
    train_path = os.path.join(input_path, "train.csv")
    test_path = os.path.join(input_path, "test.csv")
    n_folds = 5
    submission_path = os.path.join(input_path, "sample_submission.csv")
config = Config()


# In[3]:


train = pd.read_csv(config.train_path)
train.head()


# In[4]:


test = pd.read_csv(config.test_path)
test.head()


# In[5]:


submission = pd.read_csv(config.submission_path)
submission.head()


# ## EDA & Preprocessing

# In[6]:


def visualize(df, column):
    df[column].value_counts().plot(kind="bar")
    plt.title("Distribution of %s"%(column))
    plt.show()
    df.groupby(column)["num_sold"].sum().plot(kind="bar")
    plt.title("Total Sale Data in different %s"%(column))
    plt.show()
    df.groupby(column)["num_sold"].mean().plot(kind="bar")
    plt.title("Average Sale Data in different %s"%(column))
    plt.show()


# For different countries, Norway has the highest Sale Data; For different products, Kaggle Hat has the highest Sale Data; For different stores, KaggleRama has the highest Sale Data.

# In[7]:


for column in ["country", "product", "store"]:
    visualize(train, column)


# ### Feature Engineering for datetime

# In[8]:


def day_of_year(date):
    daysInMonth = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    year = 0
    month = 0
    day = 0
    i = 0
    value = 0
    for c in date:
        value = ord(c) - 48
        if value >= 0:
            if i == 0:
                year = year * 10 + value
            elif i == 1:
                month = month * 10 + value
            else:
                day = day * 10 + value
        else:
            i += 1
    num_days = day + daysInMonth[month - 1]
    is_leap = year % 400 == 0 if year % 100 == 0 else year % 4 == 0
    if is_leap and month > 2:
        num_days += 1
    return num_days

def add_datetime_features(df):
    new_df = df.copy()
    years = []
    months = []
    days = []
    weekdays = []
    weekends = []
    seasons = []
    day_of_years = []
    for item in df["date"]:
        dt = time.strptime(item, '%Y-%m-%d')
        is_weekend = dt.tm_wday >= 5
        season = (dt.tm_mon - 3) // 3 % 4
        years.append(dt.tm_year)
        months.append(dt.tm_mon)
        days.append(dt.tm_mday)
        weekdays.append(dt.tm_wday)
        weekends.append(is_weekend)
        seasons.append(season)
        day_of_years.append(day_of_year(item))
    new_df["year"] = years
    new_df["month"] = months
    new_df["day"] = days
    new_df["weekday"] = weekdays
    new_df["weekend"] = weekends
    new_df["season"] = seasons
    new_df["day_of_year"] = day_of_years
    new_df["end_of_year"] = new_df["day_of_year"] >= 350
    new_df["end_of_year"] = new_df["end_of_year"].astype(int)
    new_df.pop("date")
    return new_df


# In[9]:


train_df = add_datetime_features(train)
train_df.head()


# In[10]:


test_df = add_datetime_features(test)
test_df.head()


# ### Drop Id columns
# 

# In[11]:


train_df.pop("row_id")
test_df.pop("row_id");


# ### More EDA
# As we can see that Sale Data is increasing with year, but it is greater in end of month, end of week and Spring and Winter. It has strong cyclicity.

# In[12]:


for column in ["year", "month", "day", "weekday", "season", "weekend", "end_of_year"]:
    visualize(train_df, column)


# ### Handle Categorical Features

# In[13]:


data = pd.concat([train_df, test_df])
categorical_columns = ['country', 'store', 'product', 'year', "month", 'weekday', 'season']
for column in categorical_columns:
    item = pd.get_dummies(data[column])
    item.columns = ["_".join([column, str(item)]) for item in item.columns]
    data = pd.concat([data, item], axis=1)
    data.pop(column)
train_df = data[0:len(train_df)]
test_df = data[len(train_df):]
test_df.pop("num_sold");


# ### Feature Correlation

# In[14]:


corr = train_df.corr()
corr


# As we can see that the most sinificant feature are product, store, country, whether it is weekend, weekday, seasons.

# In[15]:


corr["num_sold"].sort_values(key=lambda item: abs(item), ascending=False)[:30]


# In[16]:


import seaborn as sns
correlated_features = list(corr[corr["num_sold"].abs() > 0.08].index)
plt.figure(figsize=(15, 15))
sns.heatmap(train_df[correlated_features].corr(), annot=True)


# ## Modeling

# In[17]:


def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

def inference(models, X):
    y_preds = []
    for model in models:
        y_pred = model.predict(X)
        y_preds.append(y_pred)
    return np.mean(y_preds, axis=0)


# In[18]:


from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold
kfold =TimeSeriesSplit(config.n_folds)
#kfold =KFold(config.n_folds, shuffle=True, random_state=42)
cats = []
scores = []
best_score = 100
best_fold = 0
worst_score = 0
worst_fold = 0 
for fold, (train_indices, valid_indices) in enumerate(kfold.split(train_df)):
    print("Fold %d:"%(fold))
    X_train = train_df.iloc[train_indices]
    y_train = X_train.pop("num_sold")
    X_val = train_df.iloc[valid_indices]
    y_val = X_val.pop("num_sold")
    params = {
        'n_estimators': 10000, 
        #'od_wait': 1000, 
        'learning_rate': 0.03, 
        'depth': 7, 
        #'l2_leaf_reg': 5,
        'verbose' : 1000,
        "eval_metric": "SMAPE",
        "objective": "RMSE"
    }
    cat = CatBoostRegressor(**params)
    cat.fit(X_train, y_train, eval_set=(X_val, y_val))
    cats.append(cat)
    y_pred = cat.predict(X_val)
    score = smape(y_val, y_pred)
    scores.append(score)
    if score < best_score:
        best_score = score
        best_fold = fold
    if score > worst_score:
        worst_score = score
        worst_fold = fold
print("Average SMAPE: %.2f"%(np.mean(scores)))


# In[19]:


models = []
for fold in range(len(cats)):
    models.append(cats[fold])


# ## Submission

# In[20]:


y_pred = inference([models[-1]], test_df)
submission["num_sold"] = y_pred
submission.to_csv("submission.csv", index=False)
submission.head()

