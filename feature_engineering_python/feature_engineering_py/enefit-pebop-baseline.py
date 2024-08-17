#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import pickle
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import VotingRegressor
import lightgbm as lgb
import optuna


# In[2]:


class MonthlyKFold:
    def __init__(self, n_splits=3):
        self.n_splits = n_splits
        
    def split(self, X, y, groups=None):
        dates = 12 * X["year"] + X["month"]
        timesteps = sorted(dates.unique().tolist())
        X = X.reset_index()
        
        for t in timesteps[-self.n_splits:]:
            idx_train = X[dates.values < t].index
            idx_test = X[dates.values == t].index
            
            yield idx_train, idx_test
            
    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


# In[3]:


def feature_eng(df_data, df_client, df_gas, df_electricity, df_forecast, df_historical, df_location, df_target):
    
    df_data = (df_data.with_columns(pl.col("datetime").cast(pl.Date).alias("date"),))
    df_client = (df_client.with_columns((pl.col("date") + pl.duration(days=2)).cast(pl.Date)))
    df_gas = (df_gas.rename({"forecast_date": "date"}).with_columns((pl.col("date") + pl.duration(days=1)).cast(pl.Date)))
    df_electricity = (df_electricity.rename({"forecast_date": "datetime"}).with_columns(pl.col("datetime") + pl.duration(days=1)))
    df_location = (df_location.with_columns(pl.col("latitude").cast(pl.datatypes.Float32),pl.col("longitude").cast(pl.datatypes.Float32)))
    df_forecast = (
        df_forecast.rename({"forecast_datetime": "datetime"}).with_columns(pl.col("latitude").cast(pl.datatypes.Float32),pl.col("longitude").cast(pl.datatypes.Float32),pl.col('datetime').dt.convert_time_zone("Europe/Bucharest").dt.replace_time_zone(None).cast(pl.Datetime("us")),)
        .join(df_location, how="left", on=["longitude", "latitude"]).drop("longitude", "latitude")
    )
    df_historical = (
        df_historical.with_columns(pl.col("latitude").cast(pl.datatypes.Float32),pl.col("longitude").cast(pl.datatypes.Float32),pl.col("datetime") + pl.duration(hours=37)).join(df_location, how="left", on=["longitude", "latitude"]).drop("longitude", "latitude")
    )
    df_forecast_date = (
        df_forecast.group_by("datetime").mean().drop("county")
    )
    df_forecast_local = (df_forecast.filter(pl.col("county").is_not_null()).group_by("county", "datetime").mean())
    df_historical_date = (df_historical.group_by("datetime").mean().drop("county"))
    df_historical_local = (df_historical.filter(pl.col("county").is_not_null()).group_by("county", "datetime").mean())
    df_data = (df_data.join(df_gas, on="date", how="left").join(df_client, on=["county", "is_business", "product_type", "date"], how="left").join(df_electricity, on="datetime", how="left")
        .join(df_forecast_date, on="datetime", how="left", suffix="_fd").join(df_forecast_local, on=["county", "datetime"], how="left", suffix="_fl").join(df_historical_date, on="datetime", how="left", suffix="_hd").join(df_historical_local, on=["county", "datetime"], how="left", suffix="_hl")
        .join(df_forecast_date.with_columns(pl.col("datetime") + pl.duration(days=7)), on="datetime", how="left", suffix="_fdw").join(df_forecast_local.with_columns(pl.col("datetime") + pl.duration(days=7)), on=["county", "datetime"], how="left", suffix="_flw").join(df_historical_date.with_columns(pl.col("datetime") + pl.duration(days=7)), on="datetime", how="left", suffix="_hdw").join(df_historical_local.with_columns(pl.col("datetime") + pl.duration(days=7)), on=["county", "datetime"], how="left", suffix="_hlw")
        .join(df_target.with_columns(pl.col("datetime") + pl.duration(days=2)).rename({"target": "target_1"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left").join(df_target.with_columns(pl.col("datetime") + pl.duration(days=3)).rename({"target": "target_2"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left").join(df_target.with_columns(pl.col("datetime") + pl.duration(days=4)).rename({"target": "target_3"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left").join(df_target.with_columns(pl.col("datetime") + pl.duration(days=5)).rename({"target": "target_4"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left").join(df_target.with_columns(pl.col("datetime") + pl.duration(days=6)).rename({"target": "target_5"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left").join(df_target.with_columns(pl.col("datetime") + pl.duration(days=7)).rename({"target": "target_6"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left").join(df_target.with_columns(pl.col("datetime") + pl.duration(days=14)).rename({"target": "target_7"}), on=["county", "is_business", "product_type", "is_consumption", "datetime"], how="left")
        .with_columns(pl.col("datetime").dt.ordinal_day().alias("dayofyear"),pl.col("datetime").dt.hour().alias("hour"),pl.col("datetime").dt.day().alias("day"),pl.col("datetime").dt.weekday().alias("weekday"),pl.col("datetime").dt.month().alias("month"),pl.col("datetime").dt.year().alias("year"),).with_columns(pl.concat_str("county", "is_business", "product_type", "is_consumption", separator="_").alias("category_1"),)
        .with_columns((np.pi * pl.col("dayofyear") / 183).sin().alias("sin(dayofyear)"),(np.pi * pl.col("dayofyear") / 183).cos().alias("cos(dayofyear)"),(np.pi * pl.col("hour") / 12).sin().alias("sin(hour)"),(np.pi * pl.col("hour") / 12).cos().alias("cos(hour)"),)
        .with_columns(pl.col(pl.Float64).cast(pl.Float32),)
        .drop("date", "datetime", "hour", "dayofyear")
    )
    
    return df_data


# In[4]:


def to_pandas(X, y=None):
    cat_cols = ["county", "is_business", "product_type", "is_consumption", "category_1"]
    
    if y is not None:
        df = pd.concat([X.to_pandas(), y.to_pandas()], axis=1)
    else:
        df = X.to_pandas()    
    
    df = df.set_index("row_id")
    df[cat_cols] = df[cat_cols].astype("category")
    df["target_mean"] = df[[f"target_{i}" for i in range(1, 7)]].mean(1)
    df["target_std"] = df[[f"target_{i}" for i in range(1, 7)]].std(1)
    df["target_ratio"] = df["target_6"] / (df["target_7"] + 1e-3)
    
    return df


# In[5]:


def lgb_objective(trial):
    params = {
        'n_iter'           : 1000,
        'verbose'          : -1,
        'random_state'     : 42,
        'objective'        : 'l2',
        'learning_rate'    : trial.suggest_float('learning_rate', 0.005, 0.1),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'colsample_bynode' : trial.suggest_float('colsample_bynode', 0.5, 1.0),
        'lambda_l1'        : trial.suggest_float('lambda_l1', 1e-4, 10.0),
        'lambda_l2'        : trial.suggest_float('lambda_l2', 1e-4, 10.0),
        'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 4, 256),
        'max_depth'        : trial.suggest_int('max_depth', 5, 16),
        'max_bin'          : trial.suggest_int('max_bin', 32, 1024),
    }
    
    model  = lgb.LGBMRegressor(**params)
    X, y   = df_train.drop(columns=["target"]), df_train["target"]
    cv     = MonthlyKFold(1)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    
    return -1 * np.mean(scores)


# In[6]:


train = pd.read_csv("/kaggle/input/predict-energy-behavior-of-prosumers/train.csv")
pivot_train = train.pivot_table(index='datetime',columns=['county','product_type','is_business','is_consumption'], values='target', aggfunc='mean')
pivot_train.columns = ['county{}_productType{}_isBusiness{}_isConsumption{}'.format(*col) for col in pivot_train.columns.values]
pivot_train.index = pd.to_datetime(pivot_train.index)
pivot_train


# In[7]:


root = "/kaggle/input/predict-energy-behavior-of-prosumers"

data_cols        = ['target', 'county', 'is_business', 'product_type', 'is_consumption', 'datetime', 'row_id']
client_cols      = ['product_type', 'county', 'eic_count', 'installed_capacity', 'is_business', 'date']
gas_cols         = ['forecast_date', 'lowest_price_per_mwh', 'highest_price_per_mwh']
electricity_cols = ['forecast_date', 'euros_per_mwh']
forecast_cols    = ['latitude', 'longitude', 'hours_ahead', 'temperature', 'dewpoint', 'cloudcover_high', 'cloudcover_low', 'cloudcover_mid', 'cloudcover_total', '10_metre_u_wind_component', '10_metre_v_wind_component', 'forecast_datetime', 'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall', 'total_precipitation']
historical_cols  = ['datetime', 'temperature', 'dewpoint', 'rain', 'snowfall', 'surface_pressure','cloudcover_total','cloudcover_low','cloudcover_mid','cloudcover_high','windspeed_10m','winddirection_10m','shortwave_radiation','direct_solar_radiation','diffuse_radiation','latitude','longitude']
location_cols    = ['longitude', 'latitude', 'county']
target_cols      = ['target', 'county', 'is_business', 'product_type', 'is_consumption', 'datetime']

save_path = None
load_path = None


# In[8]:


df_data        = pl.read_csv(os.path.join(root, "train.csv"), columns=data_cols, try_parse_dates=True)
df_client      = pl.read_csv(os.path.join(root, "client.csv"), columns=client_cols, try_parse_dates=True)
df_gas         = pl.read_csv(os.path.join(root, "gas_prices.csv"), columns=gas_cols, try_parse_dates=True)
df_electricity = pl.read_csv(os.path.join(root, "electricity_prices.csv"), columns=electricity_cols, try_parse_dates=True)
df_forecast    = pl.read_csv(os.path.join(root, "forecast_weather.csv"), columns=forecast_cols, try_parse_dates=True)
df_historical  = pl.read_csv(os.path.join(root, "historical_weather.csv"), columns=historical_cols, try_parse_dates=True)
df_location    = pl.read_csv(os.path.join(root, "weather_station_to_county_mapping.csv"), columns=location_cols, try_parse_dates=True)
df_target      = df_data.select(target_cols)

schema_data        = df_data.schema
schema_client      = df_client.schema
schema_gas         = df_gas.schema
schema_electricity = df_electricity.schema
schema_forecast    = df_forecast.schema
schema_historical  = df_historical.schema
schema_target      = df_target.schema


# ### Feature Engineering

# In[9]:


X, y = df_data.drop("target"), df_data.select("target")
X = feature_eng(X, df_client, df_gas, df_electricity, df_forecast, df_historical, df_location, df_target)
df_train = to_pandas(X, y)


# In[10]:


df_train = df_train[df_train["target"].notnull() & df_train["year"].gt(2021)]


# ### HyperParam Optimization

# In[11]:


#study = optuna.create_study(direction='minimize', study_name='Regressor')
#study.optimize(lgb_objective, n_trials=100, show_progress_bar=True)


# In[12]:


p1={'n_iter': 1500,'verbose': -1,'objective': 'l2','learning_rate': 0.05670084478292278, 'colsample_bytree': 0.6440444070196796, 'colsample_bynode': 0.637635804565811, 'lambda_l1': 6.29090474401462, 'lambda_l2': 6.775341543233317, 'min_data_in_leaf': 95, 'max_depth': 9, 'max_bin': 630}
p2={'n_iter': 1500,'verbose': -1,'objective': 'l2','learning_rate': 0.0632167263149817, 'colsample_bytree': 0.6958033941948067, 'colsample_bynode': 0.6030801666196094, 'lambda_l1': 7.137580620471935, 'lambda_l2': 9.348169401713742, 'min_data_in_leaf': 74, 'max_depth': 11, 'max_bin': 530}
p3={'n_iter': 1500,'verbose': -1,'objective': 'l2','learning_rate': 0.061236402165228264, 'colsample_bytree': 0.81427095118471, 'colsample_bynode': 0.6097376843527067, 'lambda_l1': 6.360490880385201, 'lambda_l2': 9.954136008333839, 'min_data_in_leaf': 238, 'max_depth': 16, 'max_bin': 649}
p4={'n_iter': 1500,'verbose': -1,'objective': 'l2','learning_rate': 0.08753282378023663, 'colsample_bytree': 0.7508715107428325, 'colsample_bynode': 0.6831819500325418, 'lambda_l1': 8.679353563755722, 'lambda_l2': 6.105008696961338, 'min_data_in_leaf': 198, 'max_depth': 15, 'max_bin': 835}
p5={'n_iter': 1500,'verbose': -1,'objective': 'l2','learning_rate': 0.05929380742257108, 'colsample_bytree': 0.6101576947777211, 'colsample_bynode': 0.6052639518604396, 'lambda_l1': 8.087311995794915, 'lambda_l2': 6.067361158677095, 'min_data_in_leaf': 122, 'max_depth': 9, 'max_bin': 797}
p6={'n_iter': 1500,'verbose': -1,'objective': 'l2','learning_rate': 0.05689066836106983, 'colsample_bytree': 0.8915976762048253, 'colsample_bynode': 0.5942203285139224, 'lambda_l1': 7.6277555139102864, 'lambda_l2': 6.6591278779517808, 'min_data_in_leaf' : 156, 'max_depth': 11, 'max_bin': 813}
p7={'n_iter': 1500,'verbose': -1,'objective': 'l2','learning_rate': 0.06210133914728566, 'colsample_bytree': 0.9394149364406023, 'colsample_bynode': 0.6136449922460668, 'lambda_l1': 6.8170120783290963, 'lambda_l2': 6.9413925098162625, 'min_data_in_leaf': 100, 'max_depth': 12, 'max_bin': 749}


# In[13]:


if load_path is not None:
    model = pickle.load(open(load_path, "rb"))
else:
    model = VotingRegressor([
        ('lgb_1', lgb.LGBMRegressor(**p1, random_state=10)), 
        ('lgb_2', lgb.LGBMRegressor(**p2, random_state=10)), 
        ('lgb_3', lgb.LGBMRegressor(**p3, random_state=10)), 
        ('lgb_4', lgb.LGBMRegressor(**p4, random_state=10)), 
        ('lgb_5', lgb.LGBMRegressor(**p5, random_state=10)), 
        ('lgb_6', lgb.LGBMRegressor(**p6, random_state=10)),
        ('lgb_7', lgb.LGBMRegressor(**p7, random_state=10)),
    ],weights=[0.19,0.14,0.11,0.11,0.11,0.18,0.16])
    
    model_solar = VotingRegressor([
        ('lgb_11', lgb.LGBMRegressor(**p1, random_state=11)), 
        ('lgb_12', lgb.LGBMRegressor(**p2, random_state=11)), 
        ('lgb_13', lgb.LGBMRegressor(**p3, random_state=11)), 
        ('lgb_14', lgb.LGBMRegressor(**p4, random_state=11)), 
        ('lgb_15', lgb.LGBMRegressor(**p5, random_state=11)), 
        ('lgb_16', lgb.LGBMRegressor(**p6, random_state=11)),
        ('lgb_17', lgb.LGBMRegressor(**p7, random_state=11)),
    ],weights=[0.19,0.14,0.11,0.11,0.11,0.18,0.16])
    
    model.fit(
        X=df_train.drop(columns=["target"]),
        y=df_train["target"]
    )
    
    model_solar.fit(
        X=df_train[df_train['is_consumption']==0].drop(columns=["target"]),
        y=df_train[df_train['is_consumption']==0]["target"]
    )


# In[14]:


import enefit

env = enefit.make_env()
iter_test = env.iter_test()


# In[15]:


for (test, revealed_targets, client, historical_weather,
        forecast_weather, electricity_prices, gas_prices, sample_prediction) in iter_test:
    
    test = test.rename(columns={"prediction_datetime": "datetime"})
    
    df_test           = pl.from_pandas(test[data_cols[1:]], schema_overrides=schema_data)
    df_client         = pl.from_pandas(client[client_cols], schema_overrides=schema_client)
    df_gas            = pl.from_pandas(gas_prices[gas_cols], schema_overrides=schema_gas)
    df_electricity    = pl.from_pandas(electricity_prices[electricity_cols], schema_overrides=schema_electricity)
    df_new_forecast   = pl.from_pandas(forecast_weather[forecast_cols], schema_overrides=schema_forecast)
    df_new_historical = pl.from_pandas(historical_weather[historical_cols], schema_overrides=schema_historical)
    df_new_target     = pl.from_pandas(revealed_targets[target_cols], schema_overrides=schema_target)
    
    df_forecast       = pl.concat([df_forecast, df_new_forecast]).unique()
    df_historical     = pl.concat([df_historical, df_new_historical]).unique()
    df_target         = pl.concat([df_target, df_new_target]).unique()
    
    X_test = feature_eng(df_test, df_client, df_gas, df_electricity, df_forecast, df_historical, df_location, df_target)
    X_test = to_pandas(X_test)
    
    test['target'] = model.predict(X_test).clip(0)
    test['target_solar'] = model_solar.predict(X_test).clip(0)
    test.loc[test['is_consumption']==0, "target"] = test.loc[test['is_consumption']==0, "target_solar"]    
    
    sample_prediction["target"] = test['target']
    
    env.predict(sample_prediction)

