#!/usr/bin/env python
# coding: utf-8

# ## Microbusiness Density Forecasting with KerasTuner
# In this notebook, I will create a single DNN model to predict Microbusiness Density in the fucture. I will do feature engineering by providing lag 1 target feature and calculate monthly change rate of different time series features. I will also add county embedding and state embedding in the DNN. I will search best parameters using KerasTuner. I use last 8 months of training data as validation set. CV score varies a lot for different parameters. Using KerasTuner can do more experiments to find an optimal result in short period of time and reduce human labor. Within 30 attempts, I get about 1.5 CV score and 13 to 1.5 LB score. Better DNN architecture, feature engineering, hyperpameter searching spaces and and post processing of the result can improve CV and LB score. Happy new year to everyone and happy kaggling.

# ## Configuration

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import math
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


class CFG:
    epochs = 200
    max_trials = 30
    tuning_epochs = 10
    batch_size = 512


# ## Utilities

# In[3]:


def diff_month(dt):
    return (dt.year - 2019) * 12 + dt.month - 8
def smape(y_true, y_pred):
    return 200.0  * tf.reduce_mean(tf.abs(y_true - y_pred) / (tf.abs(y_true) + tf.abs(y_pred)))

def get_cosine_decay_learning_rate_scheduler(epochs, lr_start=0.001, lr_end=1e-6):
    def cosine_decay(epoch):
        if epoch <= CFG.tuning_epochs:
            return lr_start
        if epochs > 1:
            w = (1 + math.cos(epoch / (epochs-1) * math.pi)) / 2
        else:
            w = 1
        return w * lr_start + (1 - w) * lr_end
    return LearningRateScheduler(cosine_decay, verbose=0)

def not_valid_number(num):
    return pd.isna(num) or num == 0 or num == np.inf

def preprocess_data(features):
    target = features.pop("microbusiness_density")
    for column in numeric_lookup_layers.keys():
        lookup = numeric_lookup_layers[column]
        features[column] = lookup[features["cfips"]]
    return features, target

def make_dataset(df, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices({
    "x": df["x"], 
    "year": df["year"], 
    "month": df["month"], 
    "cfips": df["cfips"], 
    "county_id": df["county_id"],
    "state_id": df["state_id"],
    "microbusiness_density": df["microbusiness_density"],
    "microbusiness_density_shift_1": df["microbusiness_density_shift_1"]
    }).map(preprocess_data)
    if shuffle:
        ds = ds.shuffle(CFG.batch_size * 4)
    ds = ds.batch(CFG.batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return ds

def cauclate_smape(item):
    cfips = item.iloc[0].cfips
    y_true = tf.constant(item["microbusiness_density"], dtype=tf.float64)
    y_pred = tf.constant(item["y_pred"], dtype=tf.float64)
    return smape(y_true, y_pred).numpy()

def plot_prediction(train, cfips, model):
    train_df = train[(train.cfips == cfips) & (train.x >= 1)].copy()
    train_df["prediction"] = model.predict(make_dataset(train_df, shuffle=False))
    plt.plot(train_df["x"], train_df["microbusiness_density"])
    plt.plot(train_df["x"], train_df["prediction"], linestyle='dashed')
    plt.legend(labels=['microbusiness_density', 'prediction'],  loc='lower right')
    plt.title(f"Microbusiness Density Forecasting about {train_df.iloc[0].county} has SMAPE {county_smapes[cfips]:.2f}")
    plt.show()
    
def sinx(x):
    return np.sin(np.pi / 6 * x)

def cosx(x):
    return np.cos(np.pi / 6 * x)


# ## Load data

# In[4]:


train = pd.read_csv("/kaggle/input/godaddy-microbusiness-density-forecasting/train.csv")
train.head()


# In[5]:


test = pd.read_csv("/kaggle/input/godaddy-microbusiness-density-forecasting/test.csv")
test.head()


# In[6]:


census = pd.read_csv("/kaggle/input/godaddy-microbusiness-density-forecasting/census_starter.csv")
census.head()


# ## Imputation

# In[7]:


df = census[census.pct_college_2018 == 0] 
pct_college_2017 = list(df["pct_college_2017"])
census.loc[census.pct_college_2018 == 0, "pct_college_2018"] = pct_college_2017
census.loc[census.pct_college_2019 == 0, "pct_college_2019"] = pct_college_2017
census.loc[census.pct_college_2020 == 0, "pct_college_2020"] = pct_college_2017
census.loc[census.pct_college_2021 == 0, "pct_college_2021"] = pct_college_2017
for cfips in train[train.microbusiness_density == 0].cfips.unique():
    df = train[train.cfips==cfips]
    targets = list(df["microbusiness_density"])
    last_none_zero_value = 1
    for i in range(len(targets)):
        target = targets[i]
        if target != 0:
            last_none_zero_value = target
        else:
             targets[i] = last_none_zero_value
    train.loc[train.cfips==cfips, "microbusiness_density"] = targets
train["active"].replace(0, 1, inplace=True)


# ## Feature Engineering

# In[8]:


get_ipython().run_cell_magic('time', '', 'county_lookup_layer = tf.keras.layers.IntegerLookup()\ncounty_lookup_layer.adapt(train.cfips.unique())\nstate_lookup_layer = tf.keras.layers.StringLookup()\nstate_lookup_layer.adapt(train.state)\ntrain[\'x\'] = pd.to_datetime(train[\'first_day_of_month\']).apply(diff_month)\ntest[\'x\'] = pd.to_datetime(test[\'first_day_of_month\']).apply(diff_month)\ntrain[\'year\'] = train[\'first_day_of_month\'].apply(lambda first_day_of_month: int(first_day_of_month.split("-")[0]) - 2019)\ntest[\'year\'] = test[\'first_day_of_month\'].apply(lambda first_day_of_month: int(first_day_of_month.split("-")[0]) - 2019)\ntrain[\'month\'] = train[\'first_day_of_month\'].apply(lambda first_day_of_month: int(first_day_of_month.split("-")[1]) - 1)\ntest[\'month\'] = test[\'first_day_of_month\'].apply(lambda first_day_of_month: int(first_day_of_month.split("-")[1]) - 1)\ntrain[\'sinx\'] = train[\'x\'].apply(sinx)\ntest[\'sinx\'] = train[\'x\'].apply(sinx)\ntrain[\'cosx\'] = train[\'x\'].apply(cosx)\ntest[\'cosx\'] = train[\'x\'].apply(cosx)\ntrain[\'county_id\'] = county_lookup_layer(tf.constant(train[\'cfips\'])).numpy().reshape(-1)\ntest[\'county_id\'] = county_lookup_layer(tf.constant(test[\'cfips\'])).numpy().reshape(-1)\ntrain[\'state_id\'] = state_lookup_layer(tf.constant(train[\'state\'])).numpy().reshape(-1)\ncounty_datas = []\nlast_value_dict = dict()\ncounty_state_dict = dict()\nfor i in range(len(census)):\n    item = census.iloc[i]\n    cfips = item.cfips\n    df = train[train.cfips == cfips]\n    if len(df) == 0:\n        continue\n    y_values = list(df["microbusiness_density"])\n    active_values = list(df["active"])\n    county_state_dict[item.cfips] = df.iloc[0].state\n    last_value = y_values[-1]\n    last_value_dict[item.cfips] = last_value\n    pct_bb_change_rate_48 = (item.pct_bb_2021 / item.pct_bb_2017) ** (1 / 48.0)\n    pct_college_change_rate_48 = (item.pct_college_2021 / item.pct_college_2017) ** (1 / 48.0)\n    median_hh_inc_change_rate_48 = (item.median_hh_inc_2021 / item.median_hh_inc_2017) ** (1 / 48.0)\n    \n    pct_bb_change_rate_12 = (item.pct_bb_2021 / item.pct_bb_2020) ** (1 / 12.0)\n    pct_college_change_rate_12 = (item.pct_college_2021 / item.pct_college_2020) ** (1 / 12.0)\n    median_hh_inc_change_rate_12 = (item.median_hh_inc_2021 / item.median_hh_inc_2020) ** (1 / 12.0)\n\n    y_change_rate_38 = (y_values[-1] / y_values[0]) ** (1 / 38.0)\n    y_change_rate_12 = (y_values[-1] / y_values[-13]) ** (1 / 12.0)\n    active_change_rate_38 = (active_values[-1] / active_values[0]) ** (1 / 38.0)\n    active_change_rate_12 = (active_values[-1] / active_values[-13]) ** (1 / 12.0)\n    mean_target = df["microbusiness_density"].mean()\n    data = {\n        "pct_bb_change_rate_48": pct_bb_change_rate_48,\n        "pct_college_change_rate_48": pct_college_change_rate_48,\n        "median_hh_inc_change_rate_48": median_hh_inc_change_rate_48,\n        "pct_bb_change_rate_12": pct_bb_change_rate_12,\n        "pct_college_change_rate_12": pct_college_change_rate_12,\n        "median_hh_inc_change_rate_12": median_hh_inc_change_rate_12,\n        "y_change_rate_38": y_change_rate_38,\n        "y_change_rate_12": y_change_rate_12,\n        "active_change_rate_38": active_change_rate_38,\n        "active_change_rate_12": active_change_rate_12,\n        "cfips": item.cfips,\n        "mean_target": mean_target\n    }\n    county_datas.append(data)\n    train.loc[train.cfips == item.cfips, "microbusiness_density_shift_1"] = df["microbusiness_density"].shift(1)\ntest[\'state_id\'] = state_lookup_layer(tf.constant(test[\'cfips\'].apply(lambda cfips: county_state_dict[cfips]))).numpy().reshape(-1)\ncounty_df = pd.DataFrame(county_datas)\ncounty_df.replace(np.nan, 1.0, inplace=True)\ncounty_df.head()\n')


# In[9]:


train.head()


# ## Correlated features
# 

# In[10]:


train_df = pd.merge(left=train, right=county_df, how='left',
               left_on='cfips', right_on='cfips')


# In[11]:


corr = train_df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr, vmin=0, vmax=1.0,  cmap="PiYG")
plt.show()


# In[12]:


corr["microbusiness_density"].sort_values(key=lambda x: abs(x), ascending=False)


# ## Build Numeric Feature Lookup Tables

# In[13]:


numeric_columns = list(county_df.columns)
numeric_columns.remove("cfips")
numeric_lookup_layers = dict()
for column in numeric_columns:
    keys = tf.constant(county_df["cfips"].astype(int))
    values = tf.constant(county_df[column])
    lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=1.0
    )
    numeric_lookup_layers[column] = lookup
numeric_lookup_layers


# ## Create Tensorflow Dataset

# In[14]:


train_features = train[(train.first_day_of_month >= "2019-09-01") & (train.first_day_of_month < "2022-03-01")]
valid_features = train[train.first_day_of_month >= "2022-03-01"]
train_ds = make_dataset(train_features)
valid_ds = make_dataset(valid_features, shuffle=False)


# ## Tuning Model

# In[15]:


def build_model(hp):
    use_dropout = hp.Choice("use_dropout", [True, False])
    dropout_value = hp.Float("dropout", min_value=0.1, max_value=0.3)
    learning_rate = hp.Float("learing_rate", min_value=1e-5, max_value=1e-3, sampling="log")
    depth = 4
    numeric_depth = 4
    units = list(reversed(sorted([hp.Int(f"unit_{i}", min_value=16, max_value=128, step=16) for i in range(depth)])))
    numeric_units = list(reversed(sorted([hp.Int(f"numeric_unit_{i}", min_value=16, max_value=128, step=16) for i in range(numeric_depth)])))
    county_units = list(reversed(sorted([hp.Int(f"county_unit_{i}", min_value=16, max_value=128, step=16) for i in range(3)])))
    state_units = list(reversed(sorted([hp.Int(f"state_unit_{i}", min_value=16, max_value=128, step=16) for i in range(3)])))
    activation = "relu"
    l2_factor = hp.Choice("l2", [1e-5, 3e-5, 5e-5, 1e-6, 5e-6])
    county_embed_size = hp.Choice("county_embed_size", [32, 64, 128])
    state_embed_size = hp.Choice("state_embed_size", [8, 16])
    x_inputs = tf.keras.Input(shape=(1), dtype=tf.float32, name="x")
    county_inputs = tf.keras.Input(shape=(1), dtype=tf.int64, name="county_id")
    state_inputs = tf.keras.Input(shape=(1), dtype=tf.int64, name="state_id")
    month_inputs = tf.keras.Input(shape=(1), dtype=tf.int64, name="month")
    #year_inputs = tf.keras.Input(shape=(1), dtype=tf.float32, name="year")
    microbusiness_density_shift_1_inputs = tf.keras.Input(shape=(1), dtype=tf.float32, name="microbusiness_density_shift_1")
    numeric_inputs = [tf.keras.Input(shape=(1), dtype=tf.float32, name=column) for column in numeric_columns] + [microbusiness_density_shift_1_inputs]
    numeric_features = numeric_inputs + [tf.sin(np.pi / 6 * x_inputs), tf.cos(np.pi / 6 * x_inputs)]
    inputs = [x_inputs, county_inputs, state_inputs, month_inputs] + numeric_inputs
    origin_numeric_vector = tf.keras.layers.Concatenate(axis=-1)(numeric_features)
    microbusiness_density_shift_1_vector = tf.stack([microbusiness_density_shift_1_inputs] * len(numeric_features), axis=-1)
    numeric_vector = tf.keras.layers.Multiply()([origin_numeric_vector, microbusiness_density_shift_1_vector])
    numeric_vector = tf.keras.layers.Flatten()(numeric_vector)
    for i in range(numeric_depth):
        kernel_regularizer = None if i != numeric_depth - 1 else tf.keras.regularizers.l2(l2_factor)
        numeric_vector = tf.keras.layers.Dense(numeric_units[i], activation=activation, kernel_regularizer=kernel_regularizer)(numeric_vector)
        
    county_vector = tf.keras.layers.Reshape((-1, 1))(county_inputs)
    county_vector = tf.keras.layers.Embedding(len(county_lookup_layer.get_vocabulary()), county_embed_size, input_length=1)(county_vector)
    county_vector = tf.keras.layers.Reshape((county_embed_size,))(county_vector)
    
    for i, unit in enumerate(county_units):
        kernel_regularizer = None if i != 2 else tf.keras.regularizers.l2(l2_factor)
        county_vector = tf.keras.layers.Dense(unit, activation=activation, kernel_regularizer=kernel_regularizer)(county_vector)
    if use_dropout:
        county_vector = tf.keras.layers.Dropout(dropout_value)(county_vector)

    state_vector = tf.keras.layers.Reshape((-1, 1))(state_inputs)
    state_vector = tf.keras.layers.Embedding(len(state_lookup_layer.get_vocabulary()), state_embed_size, input_length=1)(state_vector)
    state_vector = tf.keras.layers.Reshape((state_embed_size,))(state_vector)

    for i, unit in enumerate(state_units):
        kernel_regularizer = None if i != 2 else tf.keras.regularizers.l2(l2_factor)
        state_vector = tf.keras.layers.Dense(unit, activation=activation, kernel_regularizer=kernel_regularizer)(state_vector)
    if use_dropout:
        state_vector = tf.keras.layers.Dropout(dropout_value)(state_vector)
        
    month_vector = tf.one_hot(month_inputs, depth=12)
    month_vector = tf.keras.layers.Flatten()(month_vector)
    
    vector = tf.keras.layers.Concatenate(axis=-1)([numeric_vector, county_vector, state_vector, month_vector, origin_numeric_vector])
    vector = tf.keras.layers.Flatten()(vector)
    for i in range(depth):
        kernel_regularizer = None if i != depth - 1 else tf.keras.regularizers.l2(l2_factor)
        vector = tf.keras.layers.Dense(units[i], activation=activation, kernel_regularizer=kernel_regularizer)(vector)
    output = tf.keras.layers.Dense(1)(vector)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(loss="huber_loss", optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=[smape])
    return model


# In[16]:


tuner = kt.BayesianOptimization(
    build_model,
    objective=kt.Objective("val_smape", direction="min"),
    max_trials=CFG.max_trials,
    overwrite=True
)
tuner.search(train_ds, epochs=CFG.tuning_epochs, validation_data=valid_ds, verbose=2)


# In[17]:


tuner.results_summary()


# In[18]:


loaded_model = tuner.get_best_models()[0]
loaded_model.summary()


# ## Training Model

# In[19]:


best_hps = tuner.get_best_hyperparameters()
model = build_model(best_hps[0])


# In[20]:


checkpoints = tf.keras.callbacks.ModelCheckpoint(
    "model.tf", 
    monitor="val_smape", 
    mode="min", 
    save_best_only=True
)
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=30,
    monitor="val_loss",
    mode="min",
    restore_best_weights=True
)
epochs = CFG.epochs
learning_rate = model.optimizer.learning_rate.numpy()
scheduler = get_cosine_decay_learning_rate_scheduler(epochs=epochs, lr_start=learning_rate, lr_end=learning_rate * 0.01)
model.fit(train_ds, epochs=epochs, validation_data=valid_ds, callbacks=[checkpoints, early_stop, scheduler], verbose=2)
model = tf.keras.models.load_model("model.tf", custom_objects={"smape": smape})


# ## Model Evaluation

# In[21]:


metrics = model.evaluate(valid_ds)
smape_1 = metrics[1]
print(f"Validation Loss:{metrics[0]} Validation SMAPE: {metrics[1]}")

metrics = loaded_model.evaluate(valid_ds)
smape_2 = metrics[1]
print(f"Validation Loss:{metrics[0]} Validation SMAPE: {metrics[1]}")
best_model = None
if smape_1 < smape_2:
    print("Retrained Model is better than best KerasTuner Model. Use Retrained Model.")
    best_model = model
else:
    best_model = loaded_model
    print("Retrained Model is not better than best KerasTuner Model. Use best KerasTuner Model.")


# Calculate smape for different counties.

# In[22]:


valid_features["y_pred"] = best_model.predict(valid_ds)
county_smapes = valid_features.groupby("cfips").apply(cauclate_smape)
county_smapes.sort_values(ascending=True, inplace=True)


# Here are good prediction samples.

# In[23]:


county_smapes.head(30)


# In[24]:


for cfips in county_smapes.index[:30]:
    plot_prediction(train, cfips, best_model)


# Here are bad prediction samples.

# In[25]:


county_smapes.tail(30)


# In[26]:


for cfips in county_smapes.index[-31:]:
    plot_prediction(train, cfips, best_model)


# ## Create submission file

# In[27]:


test["microbusiness_density_shift_1"] = 0.0
test["microbusiness_density"] = 0.0
dates = sorted(test.first_day_of_month.unique())
for date in dates:
    df = test[test.first_day_of_month == date]
    df["microbusiness_density_shift_1"] = df["cfips"].apply(lambda cfips: last_value_dict[cfips])
    ds = make_dataset(df, shuffle=False)
    y_pred = model.predict(ds).reshape(-1)
    test.loc[test.first_day_of_month == date, "microbusiness_density"] = y_pred
    for cfips, val in zip(list(df["cfips"]), list(y_pred)):
        last_value_dict[cfips] = val    
submission = test[["row_id","microbusiness_density"]]
submission.to_csv("submission.csv", index=False)
submission.head()


# **If you find it helpful, please upvote.**
