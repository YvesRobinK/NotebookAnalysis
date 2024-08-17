#!/usr/bin/env python
# coding: utf-8

# ## INGV Challenge - Volcanic Eruption Prediction
# https://www.kaggle.com/c/predict-volcanic-eruptions-ingv-oe  
# 
# ## Summary
# In part 1, we will read data and do some basic EDA.  
# In the part 2 of this notebook, we will in generate almost 900 hundreds features in less than 35 minutes using multiprocessing.  
# Features will be saved in a Kaggle dataset.
# 
# In part 3, we will fit a LGBM model, like in several notebooks. And we will see how to boost our CV MAE Score by removing many features.
# 
# But we will see that CV MAE Score is much better than the Public LB Score : test sets and train sets are really different as we will see in part 4. We have a big problem of overfitting.
# 
# Thanks to  
# https://www.kaggle.com/amanooo/ingv-volcanic-basic-solution-stft  
# https://www.kaggle.com/jesperdramsch/introduction-to-volcanology-seismograms-and-lgbm   
# https://www.kaggle.com/kyakovlev/m5-three-shades-of-dark-darker-magic  
# https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction  
# https://www.kaggle.com/tunguz/ms-malware-adversarial-validation  
# and many more...

# ## Constants and Libraries

# In[1]:


__seed = 666

__create_dataframes = True  # To create Train and Test datasets. If False, data will we read in "../input/ingvchallengefeatures/"
__dataframe_size = 5000     # to debug : use small value like 10 or 100, or use 5000 to create complete dataframes

__path_to_my_data = "../input/ingvchallengefeatures/"  # Path to read Train and Test dataframes, if we didn't create them

__n_folds = 5


# In[2]:


import numpy as np 
np.random.seed(__seed)

import scipy
import scipy.signal

import pandas as pd
pd.set_option('max_columns', 100)
pd.set_option('max_rows', 200)

import matplotlib.pyplot as plt
import seaborn as sns

import gc, pickle, os, itertools, datetime
from functools import partial

import psutil
__n_cores = psutil.cpu_count()     # Available CPU cores
from multiprocessing import Pool   # Multiprocess Runs

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import lightgbm as lgbm


# # Part 1 : Train and Sample_submission files

# In[3]:


y_train = pd.read_csv("../input/predict-volcanic-eruptions-ingv-oe/train.csv"
                      , index_col="segment_id"
                      , dtype={"time_to_eruption":np.int32}
                     , squeeze=True)   # to return a Series and not a DataFrame
print("Length of time_to_eruption serie : {}\n\nThe five first lines :".format(len(y_train)))
print(y_train.head().map('{:,.0f}'.format))


# In[4]:


print("Time to Eruption in Train\n-------------------------\nMin : {:,}\nMedian : {:,}\nMax : {:,}".format(
    y_train.min(), y_train.median(), y_train.max()))

# Convert 'time_to_eruption'to hours:minutes:seconds (Just for reference)
# Thanks to Amanooo in
#    https://www.kaggle.com/amanooo/ingv-volcanic-basic-solution-stft
_temp = y_train.apply(lambda x:datetime.timedelta(milliseconds = x))
print("\nTime to Eruption in Train\n-------------------------\nMin : {}\nMedian : {}\nMax : {}".format(
    _temp.min(), _temp.median(), _temp.max()))

del _temp


# In[5]:


# matplotlib histogram
plt.hist(y_train, color = 'blue', edgecolor = 'black', bins=150, alpha=0.5)

# Add labels
plt.title('Distribution of time to eruption in Train')
plt.xlabel('Time')
plt.ylabel('Frequency');


# In[6]:


print("Number of sequences\n-------------------\nIn Train : {} files\nIn Test : {} files".format(
    len(os.listdir('../input/predict-volcanic-eruptions-ingv-oe/train/'))
    , len(os.listdir('../input/predict-volcanic-eruptions-ingv-oe/test/'))))


# In[7]:


sample_submission = pd.read_csv("../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv")
y_pred = sample_submission.set_index("segment_id")["time_to_eruption"].astype(np.int32)
y_pred.head()


# Can we do something with Segment ID ? No, it's just a index.

# In[8]:


print("Quantile of segment Id in Train : \n{}\n".format(y_train.index.to_series().quantile([0, .1, .5, .9, 1]).map('{:,.0f}'.format)))
print("Quantile of segment Id in Test : \n{}\n".format(y_pred.index.to_series().quantile([0, .1, .5, .9, 1]).map('{:,.0f}'.format)))

plt.boxplot([y_train.index.to_series(), y_pred.index.to_series()])
plt.ylabel("Segment ID")
plt.xticks([1, 2], ['Train segment ID', 'Test segment ID']);


# In[9]:


def make_file_submission(pred, filename = "submission.csv", verbose = False):
    
    sample_submission["time_to_eruption"] = pred
    sample_submission.loc[sample_submission["time_to_eruption"]<0, "time_to_eruption"]=0
    
    sample_submission.to_csv(filename, index=False)
    
    if verbose:
        print(sample_submission.head())


# ## Show some sequences

# In[10]:


# Thanks to jesperdramsch in
#    https://www.kaggle.com/jesperdramsch/introduction-to-volcanology-seismograms-and-lgbm
def plot_sequence(segment_id="1000015382"):
    sequence = pd.read_csv("../input/predict-volcanic-eruptions-ingv-oe/train/"+segment_id+".csv", dtype="Int16")
    sequence.fillna(0).plot(subplots=True, figsize=(25, 10))
    plt.tight_layout()
    plt.show()
    
print(y_train.sort_values().head())

# Some sequences with low value of time to eruption
plot_sequence("1658693785")
#plot_sequence("1957235969")
#plot_sequence("442994108")


# In[11]:


# Some sequences with high value of time to eruption
print(y_train.sort_values().tail())
plot_sequence("1162128945")
#plot_sequence("1131527270")
#plot_sequence("356854390")


# # Part 2 : Features engineering
# As fast as possible with Pandas, pandas.agg and multiprocessing Pool.
# 
# About multiprocessing, look at :
# * https://www.kaggle.com/kyakovlev/m5-three-shades-of-dark-darker-magic  
# * https://sites.google.com/site/python3tutorial/multiprocessing_map/multiprocessing_partial_function_multiple_arguments 
# 
# 

# In[12]:


## Multiprocess Runs
# Thanks to Konstantin Yakovlev in :
#    https://www.kaggle.com/kyakovlev/m5-three-shades-of-dark-darker-magic
def df_parallelize_run(func, t_split):
    
    num_cores = np.min([__n_cores, len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=0)
    pool.close()
    pool.join()
    
    return df


# In[13]:


## Put all statistics for many columns on one unique row
## and put name to columns
def add_stats_on_the_row(df_stats, df_result = None):
    
    # Name of columns
    input_list = list(itertools.product(*[list(df_stats.columns), list(df_stats.index.values)]))
    columns = [f'{e[0]}_{str(e[1])}' for e in input_list]
    
    # Create dataframe with statistics, on a unique row
    res = pd.DataFrame(df_stats.values.reshape(1, df_stats.shape[0] * df_stats.shape[1], order="F"), columns = columns)
    
    # Add those statistics to a pre-existing data frame (or not)
    if df_result is None:
        return res
    else:
        return pd.concat([df_result, res], axis = 1)

# An example :
print("Little example, how I put statistics on a unique row :")
df_ex = pd.DataFrame({'a':[100, 1, 2], 'b':[10, 20, 3], 'c':[20, 60, 7]})

# Use the function and pandas.agg
df_stats = add_stats_on_the_row(df_ex[["a", "c"]].agg(["min", "mean", "max"]))

print("Let's create a litte data frame \n {}\n\nAnd use this function to compute statistics : \n{}".
      format(df_ex, df_stats))

del df_ex, df_stats     


# In[14]:


## Create lists of features names
def lfeat_with_suff(suffix=""):
    return [f"sensor_{i+1}{suffix}" for i in range (10)]

lfeat = lfeat_with_suff()             # = ["sensor_1", "sensor_2", ..., "sensor_10"]
lfeat_abs = lfeat_with_suff("_abs")   # = ["sensor_1_abs", "sensor_2_abs", ..., "sensor_10_abs"]
lfeat_fft_real = lfeat_with_suff("_fft_real")   
lfeat_fft_imag = lfeat_with_suff("_fft_imag")   

input_list = [range(10), ["min", "max"]]
list_result = list(itertools.product(*input_list))


# In[15]:


# https://www.kaggle.com/amanooo/ingv-volcanic-basic-solution-stft
# Thanks to Amanooo

# STFT Specifications
fs = 100                # sampling frequency 
#N = len(segment_df)     # data size
N = 60001               # data size
n = 256                 # FFT segment size
max_f = 20              # ～20Hz

delta_f = fs / n        # 0.39Hz
delta_t = n / fs / 2    # 1.28s


def STFT_Features(segment_df, segment_id):
    
    segment = [segment_id]
    
    for sensor in segment_df.columns:
        x = segment_df[sensor][:N]
        if x.isna().sum() > 1000:     ##########
            segment += ([np.NaN] * 10)
            continue
        f, t, Z = scipy.signal.stft(x.fillna(0), fs = fs, window = 'hann', nperseg = n)
        f = f[:round(max_f/delta_f)+1]
        Z = np.abs(Z[:round(max_f/delta_f)+1]).T    # ～max_f, row:time,col:freq

        th = Z.mean() * 1     ##########
        Z_pow = Z.copy()
        Z_pow[Z < th] = 0
        Z_num = Z_pow.copy()
        Z_num[Z >= th] = 1

        Z_pow_sum = Z_pow.sum(axis = 0)
        Z_num_sum = Z_num.sum(axis = 0)

        A_pow = Z_pow_sum[round(10/delta_f):].sum()
        A_num = Z_num_sum[round(10/delta_f):].sum()
        BH_pow = Z_pow_sum[round(5/delta_f):round(8/delta_f)].sum()
        BH_num = Z_num_sum[round(5/delta_f):round(8/delta_f)].sum()
        BL_pow = Z_pow_sum[round(1.5/delta_f):round(2.5/delta_f)].sum()
        BL_num = Z_num_sum[round(1.5/delta_f):round(2.5/delta_f)].sum()
        C_pow = Z_pow_sum[round(0.6/delta_f):round(1.2/delta_f)].sum()
        C_num = Z_num_sum[round(0.6/delta_f):round(1.2/delta_f)].sum()
        D_pow = Z_pow_sum[round(2/delta_f):round(4/delta_f)].sum()
        D_num = Z_num_sum[round(2/delta_f):round(4/delta_f)].sum()
        segment += [A_pow, A_num, BH_pow, BH_num, BL_pow, BL_num, C_pow, C_num, D_pow, D_num]

    features = [f"sensor_{i+1}_{f}" for i in range(10) for f in ["A_pow", "A_num", "BH_pow", "BH_num", "BL_pow", "BL_num"
                                                    , "C_pow", "C_num", "D_pow", "D_num"]]
    df = pd.DataFrame(np.array([segment]), columns = ["segment_id"] + features)
    df["segment_id"] = df["segment_id"].astype(np.int32)
    
    return df


# In[16]:


# Thanks to
# https://www.kaggle.com/isaienkov/ingv-volcanic-eruption-prediction-eda-modeling
# https://www.kaggle.com/ajcostarino/ingv-volcanic-eruption-prediction-lgbm-baseline
# https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction
def features_engineering_one_file(onefile, source):
    
    # Read a file
    sequence = pd.read_csv(f'/kaggle/input/predict-volcanic-eruptions-ingv-oe/{source}/{onefile}.csv')
    
    res = STFT_Features(sequence[lfeat], onefile)
    
    # Count NAN values for each sensor, and replace them by 0
    res = pd.concat([res, pd.DataFrame(sequence[lfeat].isna().sum().values.reshape(1, 10, order="F")
                   , columns=[f"sensor_{i+1}_nan" for i in range(10)])], axis=1)
    sequence.fillna(0, inplace = True)
    
    # Absolute value of each sensor
    sequence[lfeat_abs] = sequence[lfeat].abs()
    
    # Fast Fourier Transformation for all sensors
    ft = np.fft.fft(sequence[lfeat], axis=0)
    sequence[lfeat_fft_real] = np.real(ft)
    sequence[lfeat_fft_imag] = np.imag(ft)
    
    # Basic statistic on each sensor
    lfun = ["mean", "std", "min", "max", "mad", "skew", "kurtosis"]
    lquantiles = [.01, .05, .1, .25, .5, .75, .9, .95, .99]
    res = add_stats_on_the_row(sequence[lfeat + lfeat_abs + lfeat_fft_real + lfeat_fft_imag].agg(lfun), res)
    res = add_stats_on_the_row(sequence[lfeat + lfeat_abs + lfeat_fft_real + lfeat_fft_imag].quantile(lquantiles), res)
            
    # End
    
    return res


# In[17]:


get_ipython().run_cell_magic('time', '', 'def create_data_for_model(df, source="train"):\n    \n    # Create a new function which need only one parameter : the file\n    func_partial_fe = partial(features_engineering_one_file, source = source)\n    \n    # Read all sequences in parallel for feature engineering\n    df_set = list()\n    df_set.append(df_parallelize_run(func_partial_fe, df.index[:__dataframe_size]))\n            \n    # Transform to Pandas DataFrame\n    df_set = pd.concat(df_set)\n    df_set.reset_index(inplace = True, drop=True)\n    df_set.set_index("segment_id", inplace=True)\n    \n    return df_set\n    \n# Create dataframes or read them in my Data\nif __create_dataframes: \n    \n    print("Train dataset creation...")\n    train_set = create_data_for_model(df = y_train)\n    \n    if __dataframe_size > 100:\n    \n        print("Test dataset creation...")\n        test_set = create_data_for_model(df = y_pred, source = "test")\n    \nelse:\n    \n    train_set = pickle.load(open(__path_to_my_data + "ingv_train_set.pkl", "rb"))\n    test_set = pickle.load(open(__path_to_my_data + "ingv_test_set.pkl", "rb"))\n    \n# Write train set and test set\npickle.dump(train_set, open( "ingv_train_set.pkl", "wb" ) ) \nif __dataframe_size > 100:\n    pickle.dump(test_set, open( "ingv_test_set.pkl", "wb" ) ) \n')


# In[18]:


a=train_set.sum(axis=0, skipna=False)
print("Features with NAN values : \n{}".format(list(a[a.isna()].index)))


# In[19]:


# Some other feature engineering
def other_fe(df_e):
    
    df = df_e.copy()
    
    # Inter quantile ratio
    lfeat = ['abs_0.5', 'abs_0.75', 'abs_0.95', 'abs_0.99', 'abs_max']
    lfeat_new1 = ["q75_o_med", "q95_o_q75", "q99_o_q95", "max_o_q99"]
    lfeat_new2 = ["q95_o_med", "q99_o_med", "max_o_med"]
    
    for i in range(10):
        for feat1, num, denom in zip(lfeat_new1, lfeat[1:], lfeat[:-1]):
            df[f'sensor_{i+1}_{feat1}'] = df[f'sensor_{i+1}_{num}'] / df[f'sensor_{i+1}_{denom}']
            df.loc[ df[f'sensor_{i+1}_{denom}'] == 0, f'sensor_{i+1}_{feat1}'] = 0

        for feat2, num in zip(lfeat_new2, lfeat[2:]):
            df[f'sensor_{i+1}_{feat2}'] = df[f'sensor_{i+1}_{num}'] / df[f'sensor_{i+1}_abs_0.5']
            df.loc[ df[f'sensor_{i+1}_abs_0.5'] == 0, f'sensor_{i+1}_{feat2}'] = 0
            
        df[f'sensor_{i+1}_q3_q1_abs'] = df[f'sensor_{i+1}_abs_0.75'] - df[f'sensor_{i+1}_abs_0.25']
        df[f'sensor_{i+1}_q3_q1'] = df[f'sensor_{i+1}_0.75'] - df[f'sensor_{i+1}_0.25']
        df[f'sensor_{i+1}_d9_d1_abs'] = df[f'sensor_{i+1}_abs_0.9'] - df[f'sensor_{i+1}_abs_0.1']
        df[f'sensor_{i+1}_d9_d1'] = df[f'sensor_{i+1}_0.9'] - df[f'sensor_{i+1}_0.1']
        
    df[[f"sensor_{i+1}_empty" for i in range(10)]]=0
    for i in range(10):
        df.loc[df[f"sensor_{i+1}_abs_max"]==0, f"sensor_{i+1}_empty"]=1
        
    df.fillna(0, inplace=True)
        
    return df


# Some others features
train_set = other_fe(train_set)
if __dataframe_size > 100:
    test_set = other_fe(test_set)
train_set["time_to_eruption"] = y_train
train_set.head()


# Do have some features with only one value ? They are useless.

# In[20]:


a = train_set.nunique()
to_del = list(a[a<2].index)
print("Features deleted : \n{}".format(to_del))
train_set.drop(to_del, axis=1, inplace=True)
test_set.drop(to_del, axis=1, inplace=True)


# # Part 3 : LGBM

# In[21]:


def fit_and_predict_with_lgbm(df_train, params
            , seed=__seed, features=None, X_test=None, check_feat_importance = True, n_folds=__n_folds
            , early_stopping_rounds = 100):
    
    folds = KFold(n_splits = n_folds, shuffle = True, random_state = seed)
    list_mae, list_r2 = [], []
    predictions = None
    feature_importance_df = pd.DataFrame()
    feat_imp={}

    params["random_state"] = seed + 1
    params["bagging_seed"] = seed - 1
    
    # Keep only a list of features
    if not features is None:
        lfeat = features
    else:
        lfeat = [f for f in df_train.columns if f not in ["time_to_eruption"]]
    print("Train DF shape : {} rows and {} features".format(df_train.shape[0], len(lfeat)))
    
    if not X_test is None:
        X_test = X_test[lfeat]
        print("Test DF shape  : {} rows and {} features".format(X_test.shape[0], X_test.shape[1]))
        predictions = np.zeros(len(X_test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train)):
    
        print("Fold n°{}".format(fold_+1))
        
        X_tr, X_val = df_train[lfeat].iloc[trn_idx], df_train[lfeat].iloc[val_idx]
        y_tr, y_val = df_train['time_to_eruption'].iloc[trn_idx], df_train['time_to_eruption'].iloc[val_idx]
    
        model = lgbm.LGBMRegressor(**params, n_estimators = 3000, n_jobs = -1)
        model.fit(X_tr, y_tr, 
              eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',
              verbose=1000, early_stopping_rounds = early_stopping_rounds)
    
        y_val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
        _mae = mean_absolute_error(y_val_pred, y_val) ; list_mae.append(_mae)
        _r2 = r2_score(y_val_pred, y_val) ; list_r2.append(_r2)
        print("Scores on valid set for fold {} : MAE {:,.0f} & R2 {:.3f}".format(fold_+1, _mae, _r2))
        
        # Predictions
        if not X_test is None:
            predictions += model.predict(X_test[lfeat], num_iteration=model.best_iteration_) / folds.n_splits
    
        # Feature importance
        if check_feat_importance:
            
            # to do a nice and useless plot
            fold_importance_df = pd.DataFrame()
            fold_importance_df["Feature"] = X_tr.columns
            fold_importance_df["importance"] = model.feature_importances_[:len(X_tr.columns)]
            fold_importance_df["fold"] = fold_ + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
            # Evaluate each feature importance by random permutation of the feature values
            # More usefull
            # https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html
            # https://www.kaggle.com/kyakovlev/m5-three-shades-of-dark-darker-magic
            for feat in X_tr.columns:
                if fold_ == 0:
                    feat_imp[feat] = []
                temp_df = X_val.copy()
                temp_df[feat] = np.random.permutation(temp_df[feat])
                y_temp = model.predict(temp_df, num_iteration=model.best_iteration_)
                feat_imp[feat].append(mean_absolute_error(y_temp, y_val) - _mae)
        
    
    # Mean MAE
    _mae = np.array(list_mae).mean()
    _r2 = np.array(list_r2).mean()
    print("Mean scores on valid sets : MAE {:,.0f} & R2 {:.3f}".format(_mae, _r2))
    
    # Feature importance
    #   https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html :
    #   "eli5 provides a way to compute feature importances for any black-box estimator by measuring how score 
    #   decreases when a feature is not available; the method is also known as “permutation importance” 
    #   or “Mean Decrease Accuracy (MDA)”.
    # Here I count folds where feature permutation produce a score decrease
    # When a score in a fold decreases for a feature, we consider this feature can be removed.
    feat_imp_df = None
    if check_feat_importance:
        
        feat_imp_df = pd.DataFrame(feat_imp).transpose()

        lfeat = [i for i in range(__n_folds)]
        lfeat_sign = [f"sign_{i}" for i in range(n_folds)]

        feat_imp_df[lfeat_sign] = 0
        for mae_loss, mae_loss_sign in zip(lfeat, lfeat_sign):
            feat_imp_df.loc[feat_imp_df[mae_loss]>0, mae_loss_sign] = 1
    
        feat_imp_df["sum_sign"] = feat_imp_df[lfeat_sign].sum(axis=1)
    
    return {"pred":predictions, "feat_imp":feat_imp_df, "list_mae":list_mae, "mean_mae_valid":_mae
           , "list_r2":list_r2, "mean_r2_valid":_r2, "plot_feat_imp":feature_importance_df}


params = {
    'num_leaves': 28,          # small value to avoid overfitting
    'min_data_in_leaf': 10, 
#    'max_depth': 5,
    'learning_rate': 0.15,
    'max_bins': 50,            # small value to avoid overfitting
    "feature_fraction": 0.5,
    "bagging_freq": 1,
    "bagging_fraction": 0.8,
    "lambda_l1": 0.1,
    "boosting": "gbdt",
    'objective':'regression',
    "metric": 'mae',
    "verbosity": -1,
    "nthread": -1,
}


# In[22]:


get_ipython().run_cell_magic('time', '', 'result1 = fit_and_predict_with_lgbm(train_set, params)\n\n# Now we will keep features usefull in all folds\nfeat_imp_df = result1["feat_imp"]\nfeats = list(feat_imp_df.loc[feat_imp_df["sum_sign"] == __n_folds].index)\nprint("\\nThere are {} features with no loss of MAE on the {} folds.\\n".format(len(feats), __n_folds))\n')


# In[23]:


get_ipython().run_cell_magic('time', '', 'result2 = fit_and_predict_with_lgbm(train_set, params, seed=__seed+10, features=feats, X_test = test_set)\n\nprint("\\n\\nResults summary :")\nprint("CV n°1 results  - Mean MAE : {:,.0f}. Mean R2 : {:.3f}".format(result1["mean_mae_valid"], result1["mean_r2_valid"]))\nprint("CV n°2 results  - Mean MAE : {:,.0f}. Mean R2 : {:.3f}".format(result2["mean_mae_valid"], result2["mean_r2_valid"]))\n\nfeat_imp_df = result2["feat_imp"]\nfeats = list(feat_imp_df.loc[feat_imp_df["sum_sign"] == __n_folds].index)\nprint("\\nThere are {} features with no loss of MAE on the {} folds.\\n".format(len(feats), __n_folds))\n\nmake_file_submission(result2["pred"], "submission2.csv")\n')


# In[24]:


get_ipython().run_cell_magic('time', '', 'result3 = fit_and_predict_with_lgbm(train_set, params, seed=__seed+69, features=feats, X_test = test_set)\n\nprint("\\n\\nResults summary :")\nprint("CV n°1 results  - Mean MAE : {:,.0f}. Mean R2 : {:.3f}".format(result1["mean_mae_valid"], result1["mean_r2_valid"]))\nprint("CV n°2 results  - Mean MAE : {:,.0f}. Mean R2 : {:.3f}".format(result2["mean_mae_valid"], result2["mean_r2_valid"]))\nprint("CV n°3 results  - Mean MAE : {:,.0f}. Mean R2 : {:.3f}".format(result3["mean_mae_valid"], result3["mean_r2_valid"]))\n\nfeat_imp_df = result3["feat_imp"]\nfeats = list(feat_imp_df.loc[feat_imp_df["sum_sign"] == __n_folds].index)\nprint("\\nThere are {} features with no loss of MAE on the {} folds.\\n".format(len(feats), __n_folds))\n\nmake_file_submission(result3["pred"], "submission3.csv")\n')


# In[25]:


get_ipython().run_cell_magic('time', '', 'result4 = fit_and_predict_with_lgbm(train_set, params, seed=__seed-41, features=feats, X_test = test_set)\n\nprint("\\n\\nResults summary :")\nprint("CV n°1 results  - Mean MAE : {:,.0f}. Mean R2 : {:.3f}".format(result1["mean_mae_valid"], result1["mean_r2_valid"]))\nprint("CV n°2 results  - Mean MAE : {:,.0f}. Mean R2 : {:.3f}".format(result2["mean_mae_valid"], result2["mean_r2_valid"]))\nprint("CV n°3 results  - Mean MAE : {:,.0f}. Mean R2 : {:.3f}".format(result3["mean_mae_valid"], result3["mean_r2_valid"]))\nprint("CV n°4 results  - Mean MAE : {:,.0f}. Mean R2 : {:.3f}".format(result4["mean_mae_valid"], result4["mean_r2_valid"]))\n\nfeat_imp_df = result4["feat_imp"]\nfeats = list(feat_imp_df.loc[feat_imp_df["sum_sign"] == __n_folds].index)\nprint("\\nThere are {} features with no loss of MAE on the {} folds.\\n".format(len(feats), __n_folds))\n\nmake_file_submission(result4["pred"], "submission4.csv")\n')


# In[26]:


get_ipython().run_cell_magic('time', '', 'result5 = fit_and_predict_with_lgbm(train_set, params, seed=__seed-41, features=feats, X_test = test_set)\n\nprint("\\n\\nResults summary :")\nprint("CV n°1 results  - Mean MAE : {:,.0f}. Mean R2 : {:.3f}".format(result1["mean_mae_valid"], result1["mean_r2_valid"]))\nprint("CV n°2 results  - Mean MAE : {:,.0f}. Mean R2 : {:.3f}".format(result2["mean_mae_valid"], result2["mean_r2_valid"]))\nprint("CV n°3 results  - Mean MAE : {:,.0f}. Mean R2 : {:.3f}".format(result3["mean_mae_valid"], result3["mean_r2_valid"]))\nprint("CV n°4 results  - Mean MAE : {:,.0f}. Mean R2 : {:.3f}".format(result4["mean_mae_valid"], result4["mean_r2_valid"]))\nprint("CV n°4 results  - Mean MAE : {:,.0f}. Mean R2 : {:.3f}".format(result5["mean_mae_valid"], result5["mean_r2_valid"]))\n\nfeat_imp_df = result5["feat_imp"]\nfeats = list(feat_imp_df.loc[feat_imp_df["sum_sign"] == __n_folds].index)\nprint("\\nThere are {} features with no loss of MAE on the {} folds.\\n".format(len(feats), __n_folds))\n\nmake_file_submission(result5["pred"], "submission5.csv")\n')


# So I submit n°3.

# ## About features importances 

# In[27]:


# Features which could be removed...
feat_imp_df[feat_imp_df["sum_sign"] != __n_folds]


# In[28]:


feature_importance_df = result5["plot_feat_imp"]

cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:100].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,20))
sns.barplot(x="importance",
            y="Feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds) - The 100th most important features')
plt.tight_layout();


# # Part 4 : Adversarial Validation
# Train and test are really different.  
# Thnaks to Bojan Tunguz  
# https://www.kaggle.com/tunguz/ms-malware-adversarial-validation  
# or in https://www.kaggle.com/tunguz/ms-malware-adversarial-validation
# and in so many others challenges...
# 
# Thanks to https://www.kaggle.com/ajcostarino/ignv-adversarial-validation-cv-lb-differences
# 
# I think my MAE will grow on the Private LB...

# In[29]:


df = pd.concat([train_set.drop("time_to_eruption", axis=1), test_set], axis=0)
ts = train_set.shape
li = df.iloc[:ts[0]].index.values

df["is_test"] = 1
df.loc[li, "is_test"] = 0
print(df["is_test"].value_counts(), "\n")

# I tried many features combinations of features, I had always the same results : test AUC > 0.65 !
feat=[f"sensor_{i+1}_d9_d1" for i in range(10)] # Inter deciles (1st and 9th) differences for each sensor.

tr, te, tr_y, te_y  = train_test_split(df[feat], df["is_test"], test_size=0.33, random_state=__seed, shuffle=True)

param = {'num_leaves': 30,
         'min_data_in_leaf': 30, 
         'max_depth': -1,
         'learning_rate': 0.1,
         "min_child_samples": 20,
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": __seed,
         "boosting": "gbdt",
         'objective':'binary',
#         "metric": 'binary_logloss',
         "metric": 'auc',
         "verbosity": -1}

clf = lgbm.LGBMClassifier(**param, n_estimators = 1000, n_jobs = -1)
clf.fit(tr, tr_y, eval_set = [(tr, tr_y), (te, te_y)], verbose=50, early_stopping_rounds = 100)


feature_importances = pd.DataFrame(clf.feature_importances_
            , index = tr.columns, columns=['importance']).sort_values('importance', ascending=False)

feature_importances = feature_importances.reset_index()
feature_importances.columns = ['feature', 'importance']

fig, ax = plt.subplots(figsize = (18, 8))
sns.set()
plt.subplot(1, 1, 1);
sns.barplot(x="importance", y="feature", orient='h', data=feature_importances.head(50));
plt.title('Feature Importance to detect Train or Test obs');


# In[ ]:




