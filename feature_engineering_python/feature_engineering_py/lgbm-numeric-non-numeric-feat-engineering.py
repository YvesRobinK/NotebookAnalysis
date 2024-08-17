#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import re

from tqdm import tqdm
tqdm.pandas()

import copy

from lightgbm import LGBMRegressor
from sklearn import model_selection, metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import kurtosis, skew
from scipy.stats import linregress
from scipy.spatial.distance import jensenshannon

import string
from nltk.corpus import stopwords
from sklearn.model_selection import KFold

import optuna
from lightgbm.callback import log_evaluation, early_stopping
from collections import Counter


# # Introduction

# In this notebook, we will use LGBMRegressor to examine the score of writing based on typeing activities
# - in this version all typing features are used for engineering
# - numeric features are calculated for mean, std, sem, skew, kurtosis, 90 percentile, 70 percentile, median
# - non-numeric features will be processed accordingly
# - new features: followed by @Abdullah Meda (https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs#Feature-Engineering)'s work, adding the basic keystroke features of the {ENTER}ing the TimeSeries {SPACE} paper
# - new features 2: adding the time-sentitive features of the {ENTER}ing the TimeSeries {SPACE} paper
# 
# references:
# - https://www.kaggle.com/code/yunsuxiaozi/write-processes-write-quality-xgboost
# - https://www.kaggle.com/code/hengzheng/link-writing-simple-lgbm-baseline
# - https://www.kaggle.com/code/mcpenguin/writing-processes-to-quality-baseline
# - https://www.kaggle.com/code/ravi20076/writingquality-baseline-models
# - https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs#Feature-Engineering
# - 

# Change log (continuously adding features):
# - 0. base version: version 16, LB 0.619 (stat features for all numeric feats, non-numeric feat counts except text_change, puncture_word_counts)
# - 1. adding counts of text_change, only using the texts in self.selected_text_chg: LB 0.616
# - 2. adding kFOLD: LB 0.613
# - 3. adding gap feature for cursor_position and word_counts: LB 0.612
# - 4. adding gap feature between down_time and shifted_up_time: LB :0.609
# - 5. adding event_id as numeric feature: LB 0.61
# - 6. adding first 3 ratios: LB 0.61
# - 7. add action time gap: LB 0.61
# - 8. add action time related features, use optuna to update the parameters for LGBM: LB 0.609
# - 9. change gap amount to [1, 2, 3, 5, 10, 20]: LB 0.608 (ver.35)
# -10. utilize a selected set of statistic features for each column rather than apply all the statistical calculations, rather applying all statistical features to all column,s plus optuna hyperparameter search: LB: 0.604
# -11. add additional statistical features to ['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count'], add in {ENTER}ing the TimeSeries {SPACE} features, fine tune the model: LB 0.602
# -12. reduce the n_estimator parameter to 75: LB 0.601
# - 13. change gap amount to [1, 2, 3, 5, 10, 20, 50, 100] while preserving n_estimator = 75: LB: 0.602
# 

# # Read data

# In[2]:


train = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv')
test = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv')
train_score = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv')


# In[3]:


train


# In[4]:


test


# In[5]:


train_score


# # EDA: Check nans

# In[6]:


print(train.isna().sum())
print(test.isna().sum())


# # EDA: score distribution

# In[7]:


plt.hist(train_score['score'])


# # Preprocessing: feature engineering

# In[8]:


class Preprocessor:
    def __init__(self):
        self.activity_type=['Input','Move','Nonproduction', 'Paste', 'Remove/Cut', 'Replace']
        self.event_type=['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-','.', '/', '0', '1', '2', '5', ':', ';', '<', '=', '>', '?', '@','A', 'Alt', 'AltGraph', 'ArrowDown', 'ArrowLeft', 'ArrowRight','ArrowUp', 'AudioVolumeDown', 'AudioVolumeMute', 'AudioVolumeUp','Backspace', 'C', 'Cancel', 'CapsLock', 'Clear', 'ContextMenu','Control', 'Dead', 'Delete', 'End', 'Enter', 'Escape', 'F', 'F1','F10', 'F11', 'F12', 'F15', 'F2', 'F3', 'F6', 'Home', 'I','Insert', 'Leftclick', 'M', 'MediaPlayPause', 'MediaTrackNext','MediaTrackPrevious', 'Meta', 'Middleclick', 'ModeChange','NumLock', 'OS', 'PageDown', 'PageUp', 'Pause', 'Process','Rightclick', 'S', 'ScrollLock', 'Shift', 'Space', 'T', 'Tab','Unidentified', 'Unknownclick', 'V', '[', '\\', ']', '^', '_', '`','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm','n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','{', '|', '}', '~', '\x80', '\x96', '\x97', '\x9b', '¡', '¿', 'Â´','Ä±', 'Å\x9f', 'Ë\x86', 'â\x80\x93', 'ä']
        self.punctuations = ['"', '.', ',', "'", '-', ';', ':', '?', '!', '<', '>', '/','@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+']
        self.selected_text_chg=['q', ' ', 'NoChange', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']
        self.gaps=[1, 2, 3, 5, 10, 20]
              
            
    def calculate_localExtremes(self,nums):
        # localExtremes is defined as "Number of time windows for which the direction of the evolution of keystroke events changes"
        localExtremes=0
        mem=0
        for i in range(1,len(nums)):
            memCurr=nums[i]-nums[i-1]
            if (mem>=0 and memCurr<0) or (mem<=0 and memCurr>0):
                localExtremes=localExtremes+1
            mem=memCurr # record current evolution direction
        return localExtremes
    
    
    def time_gap_feats(self,df):
        for gap in tqdm(self.gaps):
            df[f'up_time_shift{gap}'] = df.groupby('id')['up_time'].shift(gap)
            df[f'action_time_gap{gap}'] = df['down_time'] - df[f'up_time_shift{gap}']
        df.drop(columns=[f'up_time_shift{gap}' for gap in self.gaps], inplace=True)
        return df
    
    
    def gap_feats(self,df,colname):
        for gap in tqdm(self.gaps):
            df[f'{colname}_shift{gap}'] = df.groupby('id')[colname].shift(gap);
            df[f'{colname}_change{gap}'] = df[colname] - df[f'{colname}_shift{gap}'];
        df.drop([f'{colname}_shift{gap}' for gap in self.gaps], axis=1)
        return df
        
        
    def non_numeric_counts(self,df,colname,type_list):
        # 1. change move log to 'Move' if doing activity
        if colname == 'activity':
            activity_list=list(df[colname])
            for i in range(0,len(activity_list)):
                if 'Move' in activity_list[i]:
                    activity_list[i]='Move'
            df[colname]=activity_list
        
        # 2. count the freq of each type of entry
        tmp_df = df.groupby('id').agg({colname: list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df[colname].values):
            items = list(Counter(li).items())
            di = dict()
            for k in type_list:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'{colname}_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols
        ret['id']=tmp_df['id']
        ret=ret.fillna(0)
        return ret
    
    
    def match_punctuations(self,df):
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
        ret['id']=tmp_df['id']
        ret=ret.fillna(0)
        return ret
    
    
    def text_change_feat(self,df):
        df.loc[train['text_change'].str.contains('=>'),'text_change']='q' # single character change
        tmp_df = df[(df['text_change'] != 'NoChange')].reset_index(drop=True)
        tmp_df = tmp_df.groupby('id').agg({'text_change': list}).reset_index()
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: ''.join(x))
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: re.findall(r'q+', x))
        tmp_df['input_word_count'] = tmp_df['text_change'].apply(len)
        tmp_df['input_word_length_mean'] = tmp_df['text_change'].apply(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_max'] = tmp_df['text_change'].apply(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_std'] = tmp_df['text_change'].apply(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_median'] = tmp_df['text_change'].apply(lambda x: np.median([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_skew'] = tmp_df['text_change'].apply(lambda x: skew([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_kurtosis'] = tmp_df['text_change'].apply(lambda x: kurtosis([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df.drop(['text_change'], axis=1, inplace=True)
        return tmp_df
    
    
    def additional_feat_1(self,df):
        # from https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs#Feature-Engineering
        # Allen et al. {ENTER}ing the Time Series {SPACE}: Uncovering the Writing Process through Keystroke Analyses
        # these are the "basic keystroke Indices", Table 1 
        group = df.groupby('id')['action_time_gap1']
        largest_lantency = group.max()
        smallest_lantency = group.min()
        median_lantency = group.median()
        initial_pause = df.groupby('id')['down_time'].first() / 1000
        pauses_half_sec = group.apply(lambda x: ((x > 0.5) & (x < 1)).sum())
        pauses_1_sec = group.apply(lambda x: ((x > 1) & (x < 1.5)).sum())
        pauses_1_half_sec = group.apply(lambda x: ((x > 1.5) & (x < 2)).sum())
        pauses_2_sec = group.apply(lambda x: ((x > 2) & (x < 3)).sum())
        pauses_3_sec = group.apply(lambda x: (x > 3).sum())
        df_out=pd.DataFrame({
            'id': largest_lantency.index,
            'largest_lantency': largest_lantency,
            'smallest_lantency': smallest_lantency,
            'median_lantency': median_lantency,
            'initial_pause': initial_pause,
            'pauses_half_sec': pauses_half_sec,
            'pauses_1_sec': pauses_1_sec,
            'pauses_1_half_sec': pauses_1_half_sec,
            'pauses_2_sec': pauses_2_sec,
            'pauses_3_sec': pauses_3_sec,
            }).reset_index(drop=True)
        return df_out
    
    
    def additional_feat_2(self,df):
        # from Allen et al. {ENTER}ing the Time Series {SPACE}: Uncovering the Writing Process through Keystroke Analyses, 
        # these are the "Time-Sensitive Keystroke Indices", Table 2
        # for each activity, the action_time represents the input time, and the action_time_gap1 represents the hesitation between two input events.
        # the sum of action_time and action_time_gap1 is the whole input time
        # use the sum to create 30sec time windows 
        df_g=df.groupby('id')
        time1=df_g['action_time'].apply(list)
        time2=df_g['action_time_gap1'].apply(list)
        activities=df_g['activity'].apply(list)
        
        stDevEvents=[]
        slopeDegree=[]
        entropy=[]
        localExtremes=[]
        bursts=[]
        # degreeOfUniformity, AverageRecurrence, StdDevRecurrence are not calculated as I'm haven't fully figure out how to do them
        # bursts is my own feature that not appeared in the text, which represent 30sec windows with fast input (>=30 keys)
        
        for i in tqdm(range(0,len(time1))):
            t1=time1[i]
            t2=time2[i]
            a=activities[i]
            
            # calculate time for each event plus its hesitation afterwards
            t2[0]=0
            time_diff=[]
            for j in range(0,len(t1)):
                time_diff.append(t2[j]+t1[j]) # now each time_diff represent the time when finishing current input and hesitation
                
            # find 30sec boundaries
            boundaries=[0]
            counter=0
            for i in range(0,len(time_diff)):
                counter=counter+time_diff[i]
                if counter>30000:
                    boundaries.append(i)
                    counter=0
                    
            # for each 30sec window, calculate # of keystrokes
            num_keystrokes_per_win=[]
            for i in range(len(boundaries)-1):
                startRow=boundaries[i]
                endRow=boundaries[i+1]
                # using activity to calculate # of keystroke, nonproduction counts as 0 keystrokes, else as 1 keystrokes
                activities_currWin=list(a[startRow:endRow+1])
                num_keystrokes=0
                for j in range(0,len(activities_currWin)):
                    if activities_currWin[j]!='Nonproduction':
                        num_keystrokes=num_keystrokes+1
                num_keystrokes_per_win.append(num_keystrokes)
                
            # calculate time-sentitive keystroke indices
            if len(num_keystrokes_per_win)>0 and np.sum(num_keystrokes_per_win)==0:
                stDevEvents.append(np.std(num_keystrokes_per_win))
                slopeDegree.append(linregress(np.linspace(0,len(num_keystrokes_per_win),num=len(num_keystrokes_per_win)),num_keystrokes_per_win).slope)
                entropy.append(-np.sum(np.multiply(num_keystrokes_per_win/np.sum(num_keystrokes_per_win),np.log2(num_keystrokes_per_win/np.sum(num_keystrokes_per_win)))))
                localExtremes.append(self.calculate_localExtremes(num_keystrokes_per_win))
                bursts.append(np.sum(num_keystrokes_per_win>=30))
            else:
                stDevEvents.append(np.nan)
                slopeDegree.append(np.nan)
                entropy.append(np.nan)
                localExtremes.append(np.nan)
                bursts.append(np.nan)
                
        df_out=pd.DataFrame({
            'id': activities.index,
            'stDevEvents': stDevEvents,
            'slopeDegree': slopeDegree,
            'entropy': entropy,
            'localExtremes': localExtremes,
            'bursts': bursts,
            }).reset_index(drop=True)
        
        return df_out
    
    
    def preprocessing(self,df,min_max_scaler_in):
        # new dataframe
        df_res=pd.DataFrame({'id': df['id'].unique().tolist()})
        
        # gap feats
        df=self.time_gap_feats(df)
        df=self.gap_feats(df,'action_time')
        df=self.gap_feats(df,'cursor_position')
        df=self.gap_feats(df,'word_count')
        
        # stats feats, each feat use different set of features
        feats_stat = [
            ('event_id', ['max']),
            ('up_time', ['max']),
            ('action_time', ['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum']),
            ('activity', ['nunique']),
            ('down_event', ['nunique']),
            ('up_event', ['nunique']),
            ('text_change', ['nunique']),
            ('cursor_position', ['nunique', 'mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum']),
            ('word_count', ['nunique', 'max', 'mean', 'std', 'min', 'last', 'first', 'sem', 'median', 'sum']),
            ('down_time', ['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])]
            
        for gap in self.gaps:
            feats_stat.extend([
                (f'action_time_gap{gap}', ['max', 'min', 'mean', 'std', 'sum', skew, kurtosis]),
                (f'cursor_position_change{gap}', ['max', 'mean', 'std', 'sum', skew, kurtosis]),
                (f'word_count_change{gap}', ['max', 'mean', 'std', 'sum', skew, kurtosis])
            ])
            
        for colname_method in tqdm(feats_stat):
            for idx,method in enumerate(colname_method[1]):    
                tmp_df = df.groupby(['id']).agg({colname_method[0]: method}).reset_index().rename(columns={colname_method[0]: f'{colname_method[0]}_{method}'})
                df_res = df_res.merge(tmp_df, on='id', how='left')
        
        # non numeric feats
        df_res=df_res.merge(self.non_numeric_counts(df,'activity',self.activity_type), on='id',how='left')
        df_res=df_res.merge(self.non_numeric_counts(df,'up_event',self.event_type), on='id',how='left')
        df_res=df_res.merge(self.non_numeric_counts(df,'down_event',self.event_type), on='id',how='left')
        df_res=df_res.merge(self.non_numeric_counts(df,'text_change',self.selected_text_chg), on='id',how='left')
        df_res=df_res.merge(self.match_punctuations(df), on='id',how='left')
        df_res=df_res.merge(self.text_change_feat(df), on='id',how='left')
        
        # additional feats from {ENTER}ing the TimeSeries {SPACE} Sec 3.1
        df_res=df_res.merge(self.additional_feat_1(df), on='id',how='left')
        
        # additional feats from {ENTER}ing the TimeSeries {SPACE} Sec 3.2
        df_res=df_res.merge(self.additional_feat_2(df), on='id',how='left')
        
        # standardScaling
        if type(min_max_scaler_in) is list:
            min_max_scaler_curr = MinMaxScaler()
        else:
            min_max_scaler_curr = min_max_scaler_in
        df_res.iloc[:,1:]=min_max_scaler_curr.fit_transform(df_res.iloc[:,1:])    
        
        return df_res, min_max_scaler_curr


# In[9]:


preprocessor=Preprocessor()


# In[10]:


#all_train_features=train.drop(['event_id','activity','down_event','up_event','text_change'],axis=1)
all_train_features, min_max_scaler_curr=preprocessor.preprocessing(train,[])

all_train_features=all_train_features.merge(train_score,on='id',how='left')
train_score=all_train_features['score']
all_train_features=all_train_features.drop(['id','score'],axis=1)


# ## LGBM model + KFold

# In[11]:


# LGBM model, use sklearn's MultiOutputRegressor to zip 6 prediction models

def KFold_model_training(train_feats,train_score,params):

    Y_pred=[]
    Y_ori=[]
    df_importance_list=[]
    kfold = KFold(n_splits=17, shuffle=True, random_state=42)

    final_models=[]

    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train_feats)):
        X_train = train_feats.iloc[trn_idx,:]
        Y_train = train_score.iloc[trn_idx]

        X_val = train_feats.iloc[val_idx,:]
        Y_val = train_score.iloc[val_idx]

        print('\nFold_{} Training ================================\n'.format(fold_id+1))
        model= LGBMRegressor(**params)
        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              eval_metric='rmse')

        pred_val = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)


        df_importance = pd.DataFrame({
            'column': list(train_feats.columns),
            'importance': lgb_model.feature_importances_,
        })
        df_importance_list.append(df_importance)

        final_models.append(lgb_model)
        Y_pred.extend(pred_val)
        Y_ori.extend(Y_val)

    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby(['column'])['importance'].agg('mean').sort_values(ascending=False).reset_index()

    return Y_pred, Y_ori, df_importance, final_models


# In[12]:


#best_params={'n_estimators': 601, 'reg_alpha': 0.017681413705908925, 'reg_lambda': 2.2871589125734175, 'colsample_bytree': 0.1, 'subsample': 0.6, 'learning_rate': 0.01, 'max_depth': 150, 'min_child_samples': 48}
#best_params={'n_estimators': 340, 'reg_alpha': 0.0982476456129482, 'reg_lambda': 6.527477769798132, 'colsample_bytree': 0.1, 'subsample': 0.3, 'learning_rate': 0.02, 'max_depth': 200, 'num_leaves': 836, 'min_child_samples': 56, 'min_data_per_groups': 96}
#best_params={'n_estimators': 636, 'reg_alpha': 0.012184436950722766, 'reg_lambda': 0.701608495535993, 'colsample_bytree': 0.1, 'subsample': 0.1, 'learning_rate': 0.01, 'max_depth': 200, 'num_leaves': 927, 'min_child_samples': 66, 'min_data_per_groups': 11}
#best_params={'n_estimators': 882, 'reg_alpha': 2.0706434500436512, 'reg_lambda': 9.803069587005806, 'colsample_bytree': 0.1, 'subsample': 0.3, 'learning_rate': 0.02, 'max_depth': 162, 'num_leaves': 783, 'min_child_samples': 70, 'min_data_per_groups': 60}
#best_params={'n_estimators': 75, 'max_depth': 5, 'num_leaves': 254, 'reg_alpha': 0.01842590643986959, 'reg_lambda': 0.11098765814049767, 'colsample_bytree': 0.9537035390212603, 'subsample': 0.7181082058966907, 'reg_sqrt': 'false'}
best_params={'boosting_type': 'gbdt', 'feature_fraction': 0.9996855150761665, 'lambda_l1': 2.3145014520527085, 'lambda_l2': 1.2555776670675756, 'min_data_in_leaf': 59, 'min_child_samples': 25, 'n_estimators': 76, 'subsample': 0.43829872228428457, 'learning_rate': 0.08689782191963774, 'max_depth': 10, 'max_bin': 40, 'num_leaves': 22, 'reg_sqrt': 'false','metric': 'rmse','early_stoppping':100}


# # training

# In[13]:


print(f'Fitting Model')
Y_pred, Y_ori, df_importance, final_models = KFold_model_training(all_train_features,train_score,best_params)


# Check feature importance

# In[14]:


plt.figure(figsize=(10, 120))
ax = sns.barplot(data=df_importance, x='importance', y='column', dodge=False)
ax.set_title(f"Mean feature importances")


# # generate output

# In[15]:


def KFold_model_predict(final_models,test_feats):
    Y_pred=[]

    for model in final_models:
        pred_val = model.predict(test_feats)
        Y_pred.append(pred_val)

    return np.mean(np.array(Y_pred),axis=0)


# In[16]:


test_id=test['id']
test,_=preprocessor.preprocessing(test, min_max_scaler_curr)
test=test.drop('id',axis=1)

predictions = KFold_model_predict(final_models,test)
test['predictions']=predictions
test['id']=test_id


# In[17]:


submission = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/sample_submission.csv')
submission['score']=test['predictions']


# In[18]:


submission


# In[19]:


submission.to_csv("submission.csv", index=False)


# In[20]:


print('done')


# In[ ]:




