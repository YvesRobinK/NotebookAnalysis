#!/usr/bin/env python
# coding: utf-8

# In this notebook we will build simple XGBoost-based solution. But as a metric we will use balanced accuracy. LB=0.953 using one fold mode and 20000000 samples only
# 
# Sources:
# 
# https://www.kaggle.com/rumasinha/featureselectionanddiffmodelexperiments https://www.kaggle.com/tunguz/tps-02-21-feature-importance-with-xgboost-and-shap https://www.kaggle.com/hamzaghanmi/make-it-simple
# 
# Contents:
# 
# - Simple basic EDA
# 
# - Feature preprocessing
# 
# - Classic ML baselines
# 
# - Modern ML baselines
# 
# - Best Baseline model
# 
# - Add new features using XGBoost and SHAP
# 
# - Baseline model with added features
# 
# - Hyperparameters tuning (Optuna)
# 
# - Cross-validation with optimized params
# 
# - Submission prepare

# In[ ]:


get_ipython().system('pip install xgboost==1.5.1')
get_ipython().system('pip install shap')
get_ipython().system('pip install optuna')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install pandas_profiling==3.1.0')
get_ipython().system('pip install scikit-learn-intelex')


# In[ ]:


import numpy as np 
import pandas as pd 

from pandas_profiling import ProfileReport

from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, \
                                SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, classification_report, f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler, PowerTransformer
from sklearn.utils import shuffle
from sklearn import metrics

import seaborn as sns
from matplotlib import pyplot as plt

import optuna
from optuna.samplers import TPESampler

from tqdm.notebook import tqdm
import time
import gc
import shap
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')

#plt.rcParams['figure.dpi'] = 100
#plt.rcParams.update({'font.size': 16})

# load JS visualization code to notebook
shap.initjs()


# In[ ]:


path_with_data = '/kaggle/input/tabular-playground-series-dec-2021/'
path_to_data = '/kaggle/working/'


# In[ ]:


DEBUG = True
TRAIN_MODEL = True
INFER_TEST = True
ONE_FOLD_ONLY = True
COMPUTE_IMPORTANCE = True
OOF = True


# In[ ]:


train, test, sub = pd.read_csv(path_with_data + "train.csv"), \
    pd.read_csv(path_with_data + "test.csv"), \
    pd.read_csv(path_with_data + "sample_submission.csv")

if DEBUG:
    train = train.sample(n=2000000, random_state=0)
    #test = test.sample(n=100000, random_state=0)

print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')

target = 'Cover_Type'


# In[ ]:


train.head(5)


# ### Memory reducing

# In[ ]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


# ### Simple EDA

# #### Pandas profiler

# In[ ]:


profile = ProfileReport(train, title="Pandas Profiling Report", explorative=False, minimal=True, dark_mode=True)
profile


# #### Seaborn plots

# In[ ]:


sns.relplot(data=train, x=train['Elevation'], y=train['Wilderness_Area1'], kind='scatter', hue=target)


# In[ ]:


sns.displot(data=train, x='Elevation', kind='hist', hue=target)


# In[ ]:


sns.displot(data=train, x='Elevation', kind='kde', hue=target, fill=True)


# In[ ]:


sns.jointplot(data=train, x=train['Elevation'], y=train['Wilderness_Area1'], kind='scatter', hue=target)


# #### Correlation heatmap

# In[ ]:


predictors_amount = 20 + 1  # should div by 4  + 1

colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Spearman Correlation of Features', y=1.05, size=15)
corrmat = train.corr(method='spearman').abs()
cols = corrmat.nlargest(predictors_amount, target)[target].index
cm = abs(np.corrcoef(train[cols].values.T))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
most_correlated = list(set(cols) - set([target]))


# In[ ]:


# plot the first most correlated features 

i = 1
cols_amount = 4
rows_amount = int(len(most_correlated) / cols_amount) 
plt.figure()
fig, ax = plt.subplots(rows_amount, cols_amount, figsize=(20, 22))
for feature in most_correlated:
    plt.subplot(rows_amount, cols_amount, i)
    sns.histplot(train[feature],color="blue", kde=True, bins=100, label='train_'+feature)
    sns.histplot(test[feature],color="olive", kde=True, bins=100, label='test_'+feature)
    plt.xlabel(feature, fontsize=9); plt.legend()
    i += 1
plt.show()


# In[ ]:


sns.boxplot(data=train[most_correlated])


# ### Feature preprocessing

# In[ ]:


columns = train.columns
preproc = dict()
preproc['target'] = target


# In[ ]:


to_drop = [target]


# In[ ]:


train.drop(train[train[target]==5].index,inplace=True)


# In[ ]:


categoricals_features = []


# #### Select features

# In[ ]:


features = [col for col in train.columns if col not in to_drop ]
preproc['features'] = features


# #### Select column names with <80% missing values

# In[ ]:


cols_large_miss_val = train[features].columns[(train[features].isnull().mean() > 0.8)]
print(cols_large_miss_val)
features = [col for col in features if col not in cols_large_miss_val]
preproc['features'] = features


# #### Collinear (highly correlated) features

# In[ ]:


# Threshold for removing correlated variables
threshold = 0.90
# Absolute value correlation matrix
corr_matrix = train[features].corr(method='spearman').abs()
# Upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Select columns with correlations above threshold
highly_correlated = [column for column in upper.columns if any(upper[column] > threshold)]
print(highly_correlated)
features = [col for col in features if col not in highly_correlated]
preproc['features'] = features


# #### Zero standard deviation

# In[ ]:


threshold = 0
zero_std = train[features].std().index[train[features].std() <= threshold]
print(zero_std)
features = [col for col in features if col not in zero_std]    
preproc['features'] = features


# #### Zero coefficient of variantion

# In[ ]:


threshold = 1  # in %
zero_cv = (100 * train[features].std() / train[features].mean()).index[(100 * train[features].std() / train[features].mean()) <= threshold]
print(zero_cv)
features = [col for col in features if col not in zero_cv]
preproc['features'] = features


# #### Scaler transform

# In[ ]:


scaler = RobustScaler()
scaler.fit(train[features])
train[features] = scaler.transform(train[features])
test[features] = scaler.transform(test[features])
preproc['scaler'] = scaler


# #### Power transform

# In[ ]:


if 0:
  pt = PowerTransformer()
  pt.fit(train[features])
  train[features] = pt.transform(train[features])
  test[features] = pt.transform(test[features])
  preproc['power_transformer'] = pt


# #### Distribution Plots with changes

# In[ ]:


# plot the first most correlated features 

i = 1
cols_amount = 4
rows_amount = int(len(most_correlated) / cols_amount) 
plt.figure()
fig, ax = plt.subplots(rows_amount, cols_amount, figsize=(20, 22))
for feature in most_correlated:
    plt.subplot(rows_amount, cols_amount, i)
    sns.histplot(train[feature],color="blue", kde=True, bins=100, label='train_'+feature)
    sns.histplot(test[feature],color="olive", kde=True, bins=100, label='test_'+feature)
    plt.xlabel(feature, fontsize=9); plt.legend()
    i += 1
plt.show()


# ## Classic ML baselines

# In[ ]:


features = preproc['features']
print(features)


# In[ ]:


# Mapping for XGBoost
forward_map = {6:0, 7:5, 1:1, 2:2, 3:3, 4:4}
backward_map = {0:6, 5:7, 1:1, 2:2, 3:3, 4:4}


# In[ ]:


train[target] = train[target].map(forward_map)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train[features], 
                                                    train[target],
                                                    stratify=train[target], 
                                                    test_size=0.25, 
                                                    random_state=42)


# In[ ]:


clfs = {
        #'Logistic Regression': LogisticRegression(random_state=0), 
        'Naive Bayes': GaussianNB(),
        #'SVM': SVC(gamma='auto'),
        #'Random Forest': RandomForestClassifier(random_state=0),
        'SGD Classifier': SGDClassifier(random_state=0),
        'Ridge': RidgeClassifier(random_state=0),
        'Passive Aggressive Classifier': PassiveAggressiveClassifier(random_state=0),
        #'KNN': KNeighborsClassifier(),
        #'MLP': MLPClassifier(),
        'Decision Tree': DecisionTreeClassifier()
       }


# In[ ]:


for clf_name in clfs:   
    clf = clfs[clf_name].fit(X_train, y_train)
    y_pred = clf.predict(X_test)       
    print(f'{clf_name}: , Accuracy = {accuracy_score(y_test, y_pred)}, Balanced Accuracy = {balanced_accuracy_score(y_test, y_pred)}')


# ## Modern ML baselines

# In[ ]:


clfs = {
        'LGB': LGBMClassifier(),
        'XGBoost': XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', 
                                 use_label_encoder=False, eval_metric='mlogloss'),
       }


# In[ ]:


for clf_name in clfs:   
    clf = clfs[clf_name].fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'{clf_name}: , Accuracy = {accuracy_score(y_test, y_pred)}, Balanced Accuracy = {balanced_accuracy_score(y_test, y_pred)}')


# ## Baseline model

# In[ ]:


baseline_model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', 
                               use_label_encoder=False, eval_metric='mlogloss')
baseline_model.fit(X_train, y_train)


# In[ ]:


y_pred = baseline_model.predict(X_test) 
print(f' XGBoost: , Accuracy = {accuracy_score(y_test, y_pred)}, Balanced Accuracy = {balanced_accuracy_score(y_test, y_pred)}')


# In[ ]:


baseline_model.get_booster().feature_names = features
preds = baseline_model.predict(test[features])


# In[ ]:


sub[target] = preds
sub[target] = sub[target].map(backward_map)
sub.to_csv(path_to_data + 'submission_bl.csv', index=False)


# #### Intermediate conclusion:
# Tree-based methods are most accurate for this data

# ## Add new features using XGBoost and SHAP

# In[ ]:


train_oof = np.zeros((train.shape[0],))
test_preds = 0
train_oof.shape


# In[ ]:


xgb_params= {
        #"objective": "multi:softprob",        
        #"num_class": len(np.unique(train[target])),
        #"seed": 2001,
        'tree_method': "gpu_hist",
        'predictor': 'gpu_predictor',
        #'use_label_encoder': False, 
        'eval_metric': 'mlogloss'
    }


# In[ ]:


test_xgb = xgb.DMatrix(test[features])


# In[ ]:


NUM_FOLDS = 3
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)

for f, (train_ind, val_ind) in tqdm(enumerate(kf.split(train[features], train[target]))):
        #print(f'Fold {f}')
        train_df, val_df = train[features].iloc[train_ind], train[features].iloc[val_ind]
        train_target, val_target = train[target].iloc[train_ind], train[target].iloc[val_ind]
                      
        train_df = xgb.DMatrix(train_df, label=train_target)
        val_df = xgb.DMatrix(val_df, label=val_target)
        
        model =  xgb.train(xgb_params, train_df, 100)
        temp_oof = model.predict(val_df)
        temp_test = model.predict(test_xgb)

        train_oof[val_ind] = temp_oof
        test_preds += temp_test/NUM_FOLDS
        
        print(accuracy_score(np.round(temp_oof), val_target))        


# In[ ]:


get_ipython().run_cell_magic('time', '', 'shap_preds = model.predict(test_xgb, pred_contribs=True)\n')


# In[ ]:


# summarize the effects of all the features
shap.summary_plot(shap_preds[:,:-1], test[features])


# In[ ]:


shap.summary_plot(shap_preds[:,:-1], test[features], plot_type="bar")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'shap_interactions = model.predict(xgb.DMatrix(test[features][:50000]), pred_interactions=True)\n')


# In[ ]:


def plot_top_k_interactions(feature_names, shap_interactions, k):
    # Get the mean absolute contribution for each feature interaction
    aggregate_interactions = np.mean(np.abs(shap_interactions[:, :-1, :-1]), axis=0)
    interactions = []
    for i in range(aggregate_interactions.shape[0]):
        for j in range(aggregate_interactions.shape[1]):
            if j < i:
                interactions.append(
                    (feature_names[i] + "*" + feature_names[j], aggregate_interactions[i][j] * 2))
    # sort by magnitude
    interactions.sort(key=lambda x: x[1], reverse=True)
    interaction_features, interaction_values = map(tuple, zip(*interactions))
    plt.bar(interaction_features[:k], interaction_values[:k])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    return interaction_features

interactions_to_add = 16
interaction_features = plot_top_k_interactions(features, shap_interactions, interactions_to_add)


# In[ ]:


if 0:
    interaction_features = ('Wilderness_Area3*Elevation',
      'Horizontal_Distance_To_Roadways*Elevation',
      'Horizontal_Distance_To_Fire_Points*Elevation',
      'Soil_Type38*Elevation',
      'Soil_Type39*Elevation',
      'Horizontal_Distance_To_Fire_Points*Horizontal_Distance_To_Roadways',
      'Vertical_Distance_To_Hydrology*Elevation',
      'Wilderness_Area1*Elevation')
    interactions_to_add = 8


# In[ ]:


def add_new_features(df, interaction_features, amount_of_features):
    features_list = interaction_features[:amount_of_features]
    for feat in features_list:
        first_name, second_name = feat.split('*')
        df[feat] = df[first_name]*df[second_name]
    return df, features_list


# In[ ]:


train, features_added = add_new_features(train, interaction_features, interactions_to_add)
test, _ = add_new_features(test, interaction_features, interactions_to_add)
features += list(features_added)

try:
    del test_xgb
    del shap_interactions
except:
    pass
gc.collect()


# In[ ]:


features_added


# #### Scaler transform

# In[ ]:


scaler = RobustScaler()
scaler.fit(train[features])
train[features] = scaler.transform(train[features])
test[features] = scaler.transform(test[features])
preproc['scaler'] = scaler


# ## Baseline model with added features

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train[features], 
                                                    train[target],
                                                    stratify=train[target], 
                                                    test_size=0.25, 
                                                    random_state=42)


# In[ ]:


baseline_model_af = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', 
                               use_label_encoder=False, eval_metric='mlogloss')
baseline_model_af.fit(X_train, y_train)
y_pred = baseline_model_af.predict(X_test) 
print(f' XGBoost: , Accuracy = {accuracy_score(y_test, y_pred)}, Balanced Accuracy = {balanced_accuracy_score(y_test, y_pred)}')


# In[ ]:


baseline_model_af.get_booster().feature_names = features
preds = baseline_model_af.predict(test[features])


# In[ ]:


sub[target] = preds
sub[target] = sub[target].map(backward_map)
sub.to_csv(path_to_data + 'submission_blaf.csv', index=False)


# ## Hyperparameters tuning (Optuna)

# In[ ]:


# HPO using opuna

def xgb_objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        #'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1.0),       
        'random_state': 42,
        'tree_method' : 'gpu_hist',
        'predictor' : 'gpu_predictor',
        'eval_metric': 'mlogloss',
        'use_label_encoder': False
    }
    
    X_train, X_val, y_train, y_val = train_test_split(train[features], train[target], test_size = 0.25, random_state = 42)
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    pred_val = model.predict(X_val)
    
    return balanced_accuracy_score(y_val, pred_val)


# In[ ]:


sampler = TPESampler(seed = 42)
study = optuna.create_study(study_name = 'XGBoost optimization',
                            direction = 'maximize',
                            sampler = sampler)
study.optimize(xgb_objective, n_trials = 20)

print("Best logloss:", study.best_value)
print("Best params:", study.best_params)


# In[ ]:


if 1:
    params = study.best_params    
    params['random_state'] = 42
    params['tree_method'] = 'gpu_hist'
    params['predictor'] = 'gpu_predictor'
    params['eval_metric'] = 'mlogloss'
    params['use_label_encoder'] = False    
else:    
    params = {
        'max_depth': 15, 
        'n_estimators': 159,
        'learning_rate': 0.7579479953348001,
        'random_state': 42,
        'tree_method' : 'gpu_hist',
        'predictor' : 'gpu_predictor',
        'eval_metric': 'mlogloss',
        'use_label_encoder': False
    }
print(params)


# ## Cross-validation with optimized params

# In[ ]:


#EPOCH = 250
#BATCH_SIZE = 512
NUM_FOLDS = 7
COLS = features.copy()

kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
test_preds = []
oof_preds = []
for fold, (train_idx, test_idx) in enumerate(kf.split(train[features], train[target])):
        
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = train[features].iloc[train_idx], train[features].iloc[test_idx]
        y_train, y_valid = train[target].iloc[train_idx], train[target].iloc[test_idx]
        
        filename = f"folds{fold}.pkl"
        
        if TRAIN_MODEL:            
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            pickle.dump(model, open(path_to_data + filename, 'wb'))                                    
            
        else:                  
            model = pickle.load(open(path_to_data + filename, 'rb'))                  
    
        if OOF:
            print(' Predicting OOF data...')                
            oof = model.predict(X_valid)
            baseline_accuracy = accuracy_score(y_valid, oof)            
            oof_preds.append(baseline_accuracy)
            print('OOF Accuracy = {0}'.format(baseline_accuracy))
            print(' Done!')
                       
        if INFER_TEST:
            print(' Predicting test data...')
            model.get_booster().feature_names = features
            preds = model.predict(test[features])
            test_preds.append(np.array(preds))
            print(' Done!')
                    
        if COMPUTE_IMPORTANCE:
            # from  https://www.kaggle.com/cdeotte/lstm-feature-importance
            results = []
            print(' Computing feature importance...')
            
            # COMPUTE BASELINE (NO SHUFFLE)
            oof = model.predict(X_valid)
            baseline_accuracy = accuracy_score(y_valid, oof)
            results.append({'feature':'BASELINE','accuracy':baseline_accuracy})
                                    
            for k in tqdm(range(len(COLS))):
                
                # SHUFFLE FEATURE K
                save_col = X_valid.copy()
                np.random.shuffle(X_valid[COLS[k]].values)
                                
                # COMPUTE OOF Accuracy WITH FEATURE K SHUFFLED
                oof = model.predict(X_valid)
                acc = accuracy_score(y_valid, oof)
                results.append({'feature':COLS[k],'accuracy':acc})                               
                
                X_valid = save_col.copy()
         
            # DISPLAY FEATURE IMPORTANCE
            print()
            df = pd.DataFrame(results)
            df = df.sort_values('accuracy')
            plt.figure(figsize=(10,20))
            plt.barh(np.arange(len(COLS)+1),df.accuracy)
            plt.yticks(np.arange(len(COLS)+1),df.feature.values)
            plt.title('Feature Importance',size=16)
            plt.ylim((-1,len(COLS)+1))
            plt.plot([baseline_accuracy,baseline_accuracy],[-1,len(COLS)+1], '--', color='orange',
                     label=f'Baseline OOF\naccuracy={baseline_accuracy:.3f}')
            plt.xlabel(f'Fold {fold+1} OOF accuracy with feature permuted',size=14)
            plt.ylabel('Feature',size=14)
            plt.legend()
            plt.show()
                               
            # SAVE LSTM FEATURE IMPORTANCE
            df = df.sort_values('accuracy',ascending=False)
            df.to_csv(f'feature_importance_fold_{fold+1}.csv',index=False)
                               
        # ONLY DO ONE FOLD
        if ONE_FOLD_ONLY: break


# In[ ]:


print('Mean of OOF: {0}, StD of OOF: {1}'.format(np.mean(oof_preds), np.std(oof_preds)))


# ### Submission prepare

# In[ ]:


if ONE_FOLD_ONLY:
    sub[target] = np.array(test_preds[0])
    sub[target] = sub[target].map(backward_map)
    sub.to_csv(path_to_data + 'submission_final.csv', index=False)
else:
    sub[target] = np.array(sum(test_preds)[0] / NUM_FOLDS)
    sub[target] = sub[target].map(backward_map)
    sub.to_csv(path_to_data + 'submission_final.csv', index=False)


# In[ ]:




