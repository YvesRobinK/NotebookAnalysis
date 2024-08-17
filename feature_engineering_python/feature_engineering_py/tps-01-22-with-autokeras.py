#!/usr/bin/env python
# coding: utf-8

# # TPS-01-22 with AutoKeras
# 
# ## Overview
# In this notebook I will use AutoKeras to build models for [Tabular Playground Series - Jan 2022 Competition](https://www.kaggle.com/c/tabular-playground-series-jan-2022). I will explore using [AutoModel](https://autokeras.com/auto_model/#automodel-class) which is keras functional API style with more flexibility. Before Modeling, I will also perform some Exploratory data analysis and feature engineering to find insights.

# ## Imports

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
    batch_size = 128
    label_name = "num_sold"
    modes = ["train", "inference"]
    mode = modes[1]
    output_dataset_paths = ["../input/tps0122-with-autokeras-output-v1/"]
    submission_path = os.path.join(input_path, "sample_submission.csv")
config = Config()


# In[3]:


if config.mode == config.modes[0]:
    get_ipython().system('pip install autokeras')


# In[4]:


train = pd.read_csv(config.train_path)
train.head()


# In[5]:


test = pd.read_csv(config.test_path)
test.head()


# In[6]:


submission = pd.read_csv(config.submission_path)
submission.head()


# ## EDA & Preprocessing

# In[7]:


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

# In[8]:


for column in ["country", "product", "store"]:
    visualize(train, column)


# ### Feature Engineering for datetime

# In[9]:


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
        is_weekend = 1 if dt.tm_wday >= 5 else 0
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
    new_df["end_of_year"] = new_df["end_of_year"]
    new_df["end_of_year"] = new_df["end_of_year"].astype(int)
    new_df.pop("date")
    return new_df


# In[10]:


train_df = add_datetime_features(train)
train_df.head()


# In[11]:


test_df = add_datetime_features(test)
test_df.head()


# ### Drop Id columns
# 

# In[12]:


train_df.pop("row_id")
test_df.pop("row_id");


# ### More EDA
# As we can see that Sale Data is increasing with year, but it is greater in end of month, end of week and Spring and Winter. It has strong cyclicity.

# In[13]:


for column in ["year", "month", "day", "weekday", "season", "weekend", "end_of_year"]:
    visualize(train_df, column)


# In[14]:


train_df.head()


# In[15]:


train_df.head()


# In[16]:


data = pd.concat([train_df, test_df])
categorical_columns = ['country', 'store', 'product', 'year', "month", 'weekday', 'season']
for column in categorical_columns:
    item = pd.get_dummies(data[column])
    item.columns = ["_".join([column, "_".join(str(item).split(" "))]) for item in item.columns]
    data = pd.concat([data, item], axis=1)
    data.pop(column)
train_df = data[0:len(train_df)]
test_df = data[len(train_df):]
test_df.pop("num_sold");


# In[17]:


train_df.head()


# In[18]:


test_df.head()


# In[19]:


for data in [train_df, test_df]:
    for column in data.columns:
        data[column] =  data[column].astype(float)


# ## Train Validation Split

# In[20]:


from sklearn.model_selection import TimeSeriesSplit, KFold, train_test_split
X_train, X_val = train_test_split(train_df, random_state=42)
y_train = X_train.pop(config.label_name)
y_val = X_val.pop(config.label_name)


# In[21]:


X_train.head()


# In[22]:


import tensorflow as tf
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(256).batch(config.batch_size).prefetch(tf.data.AUTOTUNE).cache()
valid_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
valid_ds = valid_ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE).cache()


# ## Modeling

# In[23]:


def inference(models, X):
    y_preds = []
    for model in models:
        y_pred = model.predict(X)
        y_preds.append(y_pred)
    return np.mean(y_preds, axis=0)
def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


# In[24]:


if config.mode == config.modes[0]:
    import autokeras as ak
    inputs = ak.StructuredDataInput()
    x1 = ak.DenseBlock()(inputs)
    x2 = ak.DenseBlock()(inputs)
    x = ak.Merge()([x1, x2])
    output = ak.RegressionHead()(x)
    auto_model = ak.AutoModel(
        overwrite=True, inputs=inputs, outputs=output, max_trials=20
    )
    auto_model.fit(train_ds, validation_data=valid_ds, epochs=20)


# ### Save the Model

# In[25]:


if config.mode == config.modes[0]:
    tf_auto_model = auto_model.export_model()
    tf_auto_model.save("auto_model.tf")


# ### Load the Model

# In[26]:


models = []
if config.mode == config.modes[0]:
    model = tf.keras.models.load_model("auto_model.tf")
    models.append(model)
    tf.keras.utils.plot_model(model, show_shapes=True)
else:
    for path in config.output_dataset_paths:
        model = tf.keras.models.load_model(path + "auto_model.tf")
        models.append(model)


# In[27]:


for model in models:
    model.summary()


# ### Evaluation

# In[28]:


for model in models:
    y_pred = inference([model], valid_ds)
    print("SMAPE:", smape(y_val, y_pred.reshape(-1)))


# ## Submission

# In[29]:


test_ds = tf.data.Dataset.from_tensor_slices((test_df))
test_ds = test_ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)


# In[30]:


y_pred = inference(models, test_ds)
submission["num_sold"] = y_pred
submission.to_csv("submission.csv", index=False)
submission.head()

