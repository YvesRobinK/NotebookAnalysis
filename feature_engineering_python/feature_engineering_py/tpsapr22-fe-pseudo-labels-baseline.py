#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Hey, thanks for viewing my Kernel!
# 
# If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! :)
# 
# EDA was done in this [notebook](https://www.kaggle.com/code/hasanbasriakcay/tpsapr22-eda-fe-baseline)

# In[1]:


import pandas as pd
import numpy as np
import warnings 

warnings.simplefilter("ignore")
train_ = pd.read_csv("../input/tabular-playground-series-apr-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-apr-2022/test.csv")
train_labels = pd.read_csv("../input/tabular-playground-series-apr-2022/train_labels.csv")
sub = pd.read_csv("../input/tabular-playground-series-apr-2022/sample_submission.csv")

display(train_.head())
display(test.head())
display(train_labels.head())
display(sub.head())


# In[2]:


train = train_.merge(train_labels, on='sequence', how='left')
train.shape


# # Feature Engineering

# In[3]:


def create_new_features(df, aggregation_cols=['sequence'], prefix=''):
    df['sensor_02_num'] = df['sensor_02'] > -15
    df['sensor_02_num'] = df['sensor_02_num'].astype(int)
    df['sensor_sum1'] = (df['sensor_00'] + df['sensor_09'] + df['sensor_06'] + df['sensor_01'])
    df['sensor_sum2'] = (df['sensor_01'] + df['sensor_11'] + df['sensor_09'] + df['sensor_06'] + df['sensor_00'])
    df['sensor_sum3'] = (df['sensor_03'] + df['sensor_11'] + df['sensor_07'])
    df['sensor_sum4'] = (df['sensor_04'] + df['sensor_10'])
    
    agg_strategy = {
                    'sensor_00': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_01': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_02': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_03': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_04': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_05': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_06': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_07': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_08': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_09': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_10': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_11': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_12': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_02_num': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_sum1': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_sum2': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_sum3': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_sum4': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                   }
    
    group = df.groupby(aggregation_cols).aggregate(agg_strategy)
    group.columns = ['_'.join(col).strip() for col in group.columns]
    group.columns = [str(prefix) + str(col) for col in group.columns]
    group.reset_index(inplace = True)
    
    temp = (df.groupby(aggregation_cols).size().reset_index(name = str(prefix) + 'size'))
    group = pd.merge(temp, group, how = 'left', on = aggregation_cols,)
    return group


# In[4]:


train_fe = create_new_features(train, aggregation_cols=['sequence', 'subject'])
test_fe = create_new_features(test, aggregation_cols=['sequence', 'subject'])


# In[5]:


train_fe_subjects = create_new_features(train, aggregation_cols = ['subject'], prefix = 'subject_')
test_fe_subjects = create_new_features(test, aggregation_cols = ['subject'], prefix = 'subject_')


# In[6]:


train_fe = train_fe.merge(train_fe_subjects, on='subject', how='left')
train_fe = train_fe.merge(train_labels, on='sequence', how='left')
test_fe = test_fe.merge(test_fe_subjects, on='subject', how='left')


# In[7]:


print(train_fe.shape, test_fe.shape)


# # Adding Pseudo Labels

# In[8]:


def pseudo_labeling(df_train, df_test, target, features, fold=10):
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    
    X_train = df_train[features]
    X_test = df_test[features]
    y_train = df_train[[target]]
    
    oof = np.zeros(len(X_train))
    preds = np.zeros(len(df_test))
    
    idx1 = X_train.index; idx2 = X_test.index
    
    skf = StratifiedKFold(n_splits=fold, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(X_train, y_train):
        clf = LGBMClassifier(verbose=0, force_col_wise=True)
        clf.fit(X_train.loc[train_index,:], y_train.loc[train_index, target], 
                eval_set = [(X_train.loc[test_index,:], y_train.loc[test_index, target])], verbose=0)
        oof[idx1[test_index]] = clf.predict_proba(X_train.loc[test_index,:])[:,1]
        preds[idx2] += clf.predict_proba(X_test)[:,1] / skf.n_splits
    
    pseudo_labeled_test = df_test.copy()
    pseudo_labeled_test[target + "_proba"] = preds
    
    auc = roc_auc_score(df_train[target], oof)
    print('LGBM scores CV =',round(auc,5))
    
    return pseudo_labeled_test


# In[9]:


features = list(test_fe.columns)
features.remove("sequence")
features.remove("subject")


# In[10]:


pseudo_labeled_test = pseudo_labeling(train_fe, test_fe, "state", features)
pseudo_labeled_test.head()


# In[11]:


def print_pseudo_label_th(df, th_list=[]):
    for th in th_list:
        temp_df = df.loc[((df['state_proba']>=th) | (df['state_proba']<=(1 - th))), :]
        print(th, '-', temp_df.shape[0])


# In[12]:


print_pseudo_label_th(pseudo_labeled_test, th_list=[0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9])


# In[13]:


pseudo_labeled_test.to_csv('pseudo_labeled_test.csv', index=False)


# # Modeling

# In[14]:


def select_pseudo_labeled_test(df_train, df, th=0.99):
    temp_df = df.loc[((df['state_proba']>=th) | (df['state_proba']<=(1 - th))), :]
    temp_df['state_proba'] = temp_df['state_proba'].round()
    temp_df = temp_df.rename(columns={'state_proba':'state'})
    new_df = pd.concat([df_train, temp_df])
    return new_df


# In[15]:


def submission_with_pseudo_labels(df_train, df_test, df_pseudo, th_list=[]):
    from lightgbm import LGBMClassifier
    
    for th in th_list:
        new_df = select_pseudo_labeled_test(df_train, df_pseudo, th=th)
        X_test = df_test.drop(['sequence', 'subject'], 1)
        X_train = new_df[X_test.columns]
        y_train = new_df[['state']]

        model = LGBMClassifier()
        model.fit(X_train, y_train)
        sub['state'] = model.predict_proba(X_test)[:, 1]
        sub.to_csv(f'submission_{th}.csv', index=False)


# In[16]:


submission_with_pseudo_labels(train_fe, test_fe, pseudo_labeled_test, 
                              th_list=[0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9])

