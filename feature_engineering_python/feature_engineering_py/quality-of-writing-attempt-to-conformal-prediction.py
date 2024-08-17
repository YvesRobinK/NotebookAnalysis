#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import lightgbm as lgb
from collections import Counter
import re
from scipy.stats import skew, kurtosis
from tqdm.notebook import tqdm
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor

plt.style.use('ggplot')


# In[2]:


train_logs = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv', low_memory=False,
                float_precision='round_trip')
train_scores = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv')


# In[3]:


train_logs.head()


# In[4]:


train_scores.head()


# In[5]:


class FeatureEngineer:
    
    def __init__(self, seed):
        self.seed = seed
        
        self.activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
        self.events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 
              'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
        self.text_changes = ['q', ' ', 'NoChange', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']
        self.punctuations = ['"', '.', ',', "'", '-', ';', ':', '?', '!', '<', '>', '/',
                        '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+']
        self.gaps = [1, 2, 3, 5, 10, 20, 50]
    
    def activity_counts(self, df):
        tmp_df = df.groupby('id').agg({'activity': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['activity'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.activities:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'activity_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols
        return ret


    def event_counts(self, df, colname):
        tmp_df = df.groupby('id').agg({colname: list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df[colname].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.events:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'{colname}_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols
        return ret


    def text_change_counts(self, df):
        tmp_df = df.groupby('id').agg({'text_change': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['text_change'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.text_changes:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'text_change_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols
        return ret

    def match_punctuations(self, df):
        tmp_df = df.groupby('id').agg({'down_event': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['down_event'].values):
            cnt = 0
            items = list(Counter(li).items())
            for item in items:
                k, v = item[0], item[1]
                if k in self.punctuations:
                    cnt += v
            ret.append(cnt)
        ret = pd.DataFrame({'punct_cnt': ret})
        return ret


    def get_input_words(self, df):
        tmp_df = df[(~df['text_change'].str.contains('=>'))&(df['text_change'] != 'NoChange')].reset_index(drop=True)
        tmp_df = tmp_df.groupby('id').agg({'text_change': list}).reset_index()
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: ''.join(x))
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: re.findall(r'q+', x))
        tmp_df['input_word_count'] = tmp_df['text_change'].apply(len)
        tmp_df['input_word_length_mean'] = tmp_df['text_change'].apply(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_max'] = tmp_df['text_change'].apply(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_std'] = tmp_df['text_change'].apply(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df.drop(['text_change'], axis=1, inplace=True)
        return tmp_df
    
    def engineer_features(self, df):
        print("Starting to engineer features")
        
        # initialize features dataframe
        feats = pd.DataFrame({'id': df['id'].unique().tolist()})
        
        # get shifted features
        # time shift
        print("Engineering time data")
        for gap in self.gaps:
            print(f"> for gap {gap}")
            df[f'up_time_shift{gap}'] = df.groupby('id')['up_time'].shift(gap)
            df[f'action_time_gap{gap}'] = df['down_time'] - df[f'up_time_shift{gap}']
        df.drop(columns=[f'up_time_shift{gap}' for gap in self.gaps], inplace=True)

        # cursor position shift
        print("Engineering cursor position data")
        for gap in self.gaps:
            print(f"> for gap {gap}")
            df[f'cursor_position_shift{gap}'] = df.groupby('id')['cursor_position'].shift(gap)
            df[f'cursor_position_change{gap}'] = df['cursor_position'] - df[f'cursor_position_shift{gap}']
            df[f'cursor_position_abs_change{gap}'] = np.abs(df[f'cursor_position_change{gap}'])
        df.drop(columns=[f'cursor_position_shift{gap}' for gap in self.gaps], inplace=True)

        # word count shift
        print("Engineering word count data")
        for gap in self.gaps:
            print(f"> for gap {gap}")
            df[f'word_count_shift{gap}'] = df.groupby('id')['word_count'].shift(gap)
            df[f'word_count_change{gap}'] = df['word_count'] - df[f'word_count_shift{gap}']
            df[f'word_count_abs_change{gap}'] = np.abs(df[f'word_count_change{gap}'])
        df.drop(columns=[f'word_count_shift{gap}' for gap in self.gaps], inplace=True)
        
        # get aggregate statistical features
        print("Engineering statistical summaries for features")
        # [(feature name, [ stat summaries to add ])]
        feats_stat = [
            ('event_id', ['max']),
            ('up_time', ['max']),
            ('action_time', ['sum', 'max', 'mean', 'std']),
            ('activity', ['nunique']),
            ('down_event', ['nunique']),
            ('up_event', ['nunique']),
            ('text_change', ['nunique']),
            ('cursor_position', ['nunique', 'max', 'mean']),
            ('word_count', ['nunique', 'max', 'mean'])]
        for gap in self.gaps:
            feats_stat.extend([
                (f'action_time_gap{gap}', ['max', 'min', 'mean', 'std', 'sum', skew, kurtosis]),
                (f'cursor_position_change{gap}', ['max', 'mean', 'std', 'sum', skew, kurtosis]),
                (f'word_count_change{gap}', ['max', 'mean', 'std', 'sum', skew, kurtosis])
            ])
        
        pbar = tqdm(feats_stat)
        for item in pbar:
            colname, methods = item[0], item[1]
            for method in methods:
                pbar.set_postfix()
                if isinstance(method, str):
                    method_name = method
                else:
                    method_name = method.__name__
                    
                pbar.set_postfix(column=colname, method=method_name)
                tmp_df = df.groupby(['id']).agg({colname: method}).reset_index().rename(columns={colname: f'{colname}_{method_name}'})
                feats = feats.merge(tmp_df, on='id', how='left')

        # counts
        print("Engineering activity counts data")
        tmp_df = self.activity_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering event counts data")
        tmp_df = self.event_counts(df, 'down_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        tmp_df = self.event_counts(df, 'up_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering text change counts data")
        tmp_df = self.text_change_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering punctuation counts data")
        tmp_df = self.match_punctuations(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        # input words
        print("Engineering input words data")
        tmp_df = self.get_input_words(df)
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        # compare feats
        print("Engineering ratios data")
        feats['word_time_ratio'] = feats['word_count_max'] / feats['up_time_max']
        feats['word_event_ratio'] = feats['word_count_max'] / feats['event_id_max']
        feats['event_time_ratio'] = feats['event_id_max']  / feats['up_time_max']
        feats['idle_time_ratio'] = feats['action_time_gap1_sum'] / feats['up_time_max']
        
        print("Done!")
        return feats


# In[6]:


fe = FeatureEngineer(seed=42)
train_features = fe.engineer_features(train_logs)


# In[7]:


train_features = train_features.merge(train_scores, on='id', how='left')


# In[8]:


def apply_logarithm(df):
    means = df.describe().T['mean']
    small_value = 1e-4  # small value to replace -inf
    for col in df.columns:
        if means[col] > 100:
            col_mean = np.nanmean(df[col])
            df[col] = np.log(df[col].fillna(col_mean))
            df[col] = df[col].replace(-np.inf, small_value)
    return df

train_features = apply_logarithm(train_features.drop('id',axis=1))


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = train_features.drop(['score'],axis=1)
y = train_features.score

X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=42)


# In[10]:


lgbm_params = {'learning_rate': 0.00944009169793956, 'n_estimators': 583, 'max_depth': 7, 
               'num_leaves': 15, 'subsample': 0.9298228799681219, 
               'colsample_bytree': 0.5170414791235199, 'reg_alpha': 0.35376691357998363, 
               'reg_lambda': 0.13882765431815042}
    
cat_params = {'learning_rate': 0.022279290621535674, 'iterations': 624, 'depth': 8, 'silent':True}
xgb_params = {'learning_rate': 0.019184386785980855, 'n_estimators': 1258, 
              'max_depth': 9, 'min_child_weight': 10, 'subsample': 0.5439725081570363, 
              'colsample_bytree': 0.5970984882868963, 'gamma': 3.049415070768245}
lgbm = LGBMRegressor(**lgbm_params)
cat = CatBoostRegressor(**cat_params)
xgb = XGBRegressor(**xgb_params)

model = VotingRegressor([('xgb', xgb),('cat', cat),('lgbm',lgbm)])
model.fit(X_train,y_train)


# In[11]:


residuals = y_valid - model.predict(X_valid)
## conformalized quantile regression
mad = np.median(np.abs(residuals - np.median(residuals)))
alpha = 0.01


# In[12]:


from yellowbrick.regressor import PredictionError

visualizer = PredictionError(model)
visualizer.score(X_valid,y_valid)
visualizer.show()


# In[13]:


## Visualization of error
fig, ax = plt.subplots(figsize=(10,7))
ax.hist(abs(residuals))
plt.axvline(abs(residuals).quantile(q=0.975),color='red',linestyle='dashed',linewidth=1)
plt.xlabel('Absolute Error')
plt.ylabel('Data points')
plt.title('Error Distribution')
plt.show()


# In[14]:


average_residual = np.mean(residuals)


# In[15]:


intervals = []
point_estimates = []
for i in range(len(X_valid)):
    x = X_valid.iloc[i]  
    y_pred = model.predict(np.array(x).reshape(1,-1))
    residuals = y_valid - model.predict(X_valid)
    ##calculation of uncertainty by the model
    interval = (y_pred - np.sqrt(1 + 1 / len(X_train)) * np.percentile(residuals, (1 - alpha / 2) * 100),
                y_pred + np.sqrt(1 + 1 / len(X_train)) * np.percentile(residuals, (1 - alpha / 2) * 100))
    point_estimate = np.mean(interval)  
    intervals.append(interval)
    point_estimates.append(point_estimate)
    
print(f'The RMSE of validation set is {np.sqrt(mean_squared_error(y_valid, point_estimates+average_residual))}')
#avg_set_size = np.mean([len(interval) for interval in intervals])
#avg_set_size  ##2.0


# In[16]:


pred_df = pd.DataFrame({'actual':y_valid,'predicted':point_estimates,
                       'interval':intervals})
pred_df['lower_interval'] = pred_df['interval'].apply(lambda x: x[0][0])
pred_df['upper_interval'] = pred_df['interval'].apply(lambda x: x[1][0])
pred_df = pred_df.drop('interval', axis=1)
pred_df


# In[17]:


df1 = pred_df.sample(n=20, replace=True, random_state=15)
plt.scatter(x=df1.index, y= df1['actual'],color='b')
plt.scatter(x=df1.index,y=df1['lower_interval'], color = 'r')
plt.scatter(x=df1.index,y=df1['upper_interval'], color = 'r')
plt.show()


# ## Note:
# 
# <font size="4">In the case of conformalized quantile regression, the formula used to calculate the prediction intervals is as follows:</font>
# 
# * <font size="4">**Lower bound of the interval** = Predicted value - Scaling factor * Percentile of residuals</font>
# * <font size="4">**Upper bound of the interval** = Predicted value + Scaling factor * Percentile of residuals</font>
# 
# 
# <font size="4">The intuition behind the formula is that the prediction interval should be centered around the predicted value, and the width of the interval should be proportional to the uncertainty in the prediction. The scaling factor controls the width of the interval, and the percentile of residuals is used to calculate the scaling factor in a way that ensures that the prediction interval covers the true quantile with a certain probability (typically 95%).</font>

# ## Prediction on test set

# In[18]:


test_features = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv')
test = fe.engineer_features(test_features)


# In[19]:


# Calculate the prediction intervals using conformalized quantile regression
test_df = apply_logarithm(test.drop('id',axis=1))

test_intervals = []
#lgbm.set_params(weights=optimized_weights)
test_predictions = model.predict(test_df)
for i in range(len(test_df)):
    x = test_df.iloc[i]
    y_pred = test_predictions[i]
    residuals = y_valid - model.predict(X_valid)  ## using residuals from validation set as there is no target column in test set
    interval = (y_pred - np.sqrt(1 + 1 / len(X_valid)) * np.percentile(residuals, (1 - alpha / 2) * 100),
                y_pred + np.sqrt(1 + 1 / len(X_valid)) * np.percentile(residuals, (1 - alpha / 2) * 100))
    test_intervals.append(interval)


# In[20]:


test_pred_df = pd.DataFrame({'lower': [interval[0] for interval in test_intervals],
                        'upper': [interval[1] for interval in test_intervals],
                        'predicted':test_predictions})
test_pred_df


# In[21]:


submit = pd.DataFrame()
submit['id'] = test['id']
submit['score'] = test_predictions + average_residual
submit.to_csv('submission.csv',index=False)
submit

