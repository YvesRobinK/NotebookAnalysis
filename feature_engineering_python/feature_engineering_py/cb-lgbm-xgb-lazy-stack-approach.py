#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import optuna
from sklearn.metrics import log_loss
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
pal = sns.color_palette("viridis", 10)
sns.set_palette(pal)


# In[2]:


train = pd.read_csv('../input/tabular-playground-series-may-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2021/test.csv')


# In[3]:


train.info()


# In[4]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['target'] = le.fit_transform(train['target'])


# In[5]:


train.isnull().sum()


# # Plotting + Report with Dataprep

# In[6]:


get_ipython().system('pip install dataprep')


# In[7]:


from dataprep.eda import plot, plot_correlation, create_report, plot_missing


# In[8]:


plot(train.drop(['id','target'],axis=1))


# In[9]:


# create_report(train)


# ## Insights from EDA
# > #### 1. There is no corelation between the features even with the target variable.
# > #### 2. Most of the features are skewed with 0 values even >90%, that means feature selection will be necessary.
# > #### 3. Baseline model can overfit because of skewness in data.
# > #### 4. Outlier Detection and removal will also be handy to improve score.
# > #### 5. No corelation means that there are some unnecessary features.
# > #### 6. Also we can gain some info by feature engineering by trying feature interaction or ratio and increase corelation.

# In[10]:


X = train.drop(['id','target'],axis=1)
y = train['target']


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X,y,train_size=0.8,random_state=42)


# # Baseline CATBoost Classifier

# In[12]:


from catboost import CatBoostClassifier, Pool
train_pool = Pool(data=X_train, label=y_train)
test_pool = Pool(data=X_test, label=y_test.values) 


# In[13]:


model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='MultiClass',
    verbose=False
)
model.fit(train_pool,plot=True,eval_set=test_pool)


# In[14]:


y_pred = model.predict_proba(X_test)
log_loss(y_test,y_pred)


# # Feature Selection with Permutation Importance

# In[15]:


import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(model, random_state=13, scoring = 'neg_log_loss')
perm.fit(X_test,y_test)


# In[16]:


feat_importance = pd.DataFrame({'Feature':X_train.columns, 'Importance':perm.feature_importances_}).sort_values(by='Importance',ascending=False)
plt.figure(figsize= (8,15))
sns.barplot(data = feat_importance, y = 'Feature', x= 'Importance',orient='h')


# In[17]:


a = perm.feature_importances_
l = []
for i in range(50):
    if a[i]<0:
        l.append('feature_'+str(i))
        
print('Dropped Features')
print(l)


# In[18]:


train_new = train.drop(l,axis=1)
test_new =test.drop(l,axis=1)
X_new = train_new.drop(['id','target'],axis=1)


# # Optimizing Catboost Classifier with OPTUNA

# In[19]:


def fun(trial,data=X_new,target=y):
    
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2,random_state=42)
    param = {
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'learning_rate' : trial.suggest_uniform('learning_rate',1e-3,0.1),
        
        'reg_lambda': trial.suggest_uniform('reg_lambda',1e-5,30),
        'subsample': trial.suggest_uniform('subsample',0,1),
        'random_strength': trial.suggest_uniform('random_strength',0,1),
        'depth': trial.suggest_int('depth',5,12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,100),
        'num_leaves' : trial.suggest_int('num_leaves',16,64),
        'leaf_estimation_method' : 'Newton',
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,10),
        'verbose':False,
        'bootstrap_type': 'Bernoulli',
        'random_state' : trial.suggest_categorical('random_state',[13]),
        'task_type' : 'GPU',
        'grow_policy' : 'Lossguide'
        
    }
    model = CatBoostClassifier(**param)  
    
    model.fit(train_x,train_y,eval_set=[(test_x,test_y)],early_stopping_rounds=200,verbose=False)
    
    preds = model.predict_proba(test_x)
    
    ll = log_loss(test_y, preds)
    
    return ll


# In[20]:


study = optuna.create_study(direction='minimize')
study.optimize(fun, n_trials=100)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


# # Making Predictions with tuned Model

# In[21]:


best_params_cb = study.best_params
best_params_cb['loss_function'] = 'MultiClass'
best_params_cb['eval_metric'] = 'MultiClass'
best_params_cb['verbose'] = False
best_params_cb['n_estimators'] = 10000
best_params_cb['bootstrap_type']= 'Bernoulli'
best_params_cb['leaf_estimation_method'] = 'Newton'
best_params_cb['task_type'] = 'GPU'
best_params_cb['grow_policy'] = 'Lossguide'


# # Predictions on Kfold

# In[22]:


stacked_df = pd.DataFrame(columns = ['Class1m1', 'Class2m1','Class3m1','Class4m1','Class1m2', 'Class2m2','Class3m2','Class4m2','Class1m3', 'Class2m3','Class3m3','Class4m3','target'])


# In[23]:


columns = train_new.drop(['id','target'],axis=1).columns
cb_df = pd.DataFrame(columns = ['Class1m1', 'Class2m1','Class3m1','Class4m1','target'])
preds = np.zeros((test.shape[0],4))
kf = StratifiedKFold(n_splits = 10 , random_state = 13 , shuffle = True)
ll =[]
n=0

for tr_idx, test_idx in kf.split(train_new[columns], train_new['target']):
    
    X_tr, X_val = train_new[columns].iloc[tr_idx], train_new[columns].iloc[test_idx]
    y_tr, y_val = train_new['target'].iloc[tr_idx], train_new['target'].iloc[test_idx]
    
    model = CatBoostClassifier(**best_params_cb)
    
    model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],early_stopping_rounds=500,verbose=False)
    y_pred  = model.predict_proba(X_val)
    df = pd.DataFrame(y_pred,columns=['Class1m1', 'Class2m1','Class3m1','Class4m1'])
    df['target'] = list(y_val)
    
    cb_df = pd.concat([cb_df,df])
    preds+=model.predict_proba(test_new.drop(['id'],axis=1))/kf.n_splits
    ll.append(log_loss(y_val, y_pred))
    print(n+1,ll[n])
    n+=1


# In[24]:


cb_df


# In[25]:


np.mean(ll)


# In[26]:


df_kfold = pd.DataFrame(preds,columns=['Class_1','Class_2','Class_3','Class_4'])
df_kfold['id']  = test['id']
df_kfold = df_kfold[['id','Class_1','Class_2','Class_3','Class_4']]


# In[27]:


df_kfold


# In[28]:


output_3 = df_kfold.to_csv('submit_3.csv',index=False)


# # LGBM

# In[29]:


from lightgbm import LGBMClassifier


# In[30]:


model = LGBMClassifier(random_state= 13, objective= 'multiclass', metric = 'multi_logloss').fit(X_train, y_train)


# In[31]:


perm = PermutationImportance(model, random_state=13, scoring = 'neg_log_loss')
perm.fit(X_test,y_test)


# In[32]:


feat_importance = pd.DataFrame({'Feature':X_train.columns, 'Importance':perm.feature_importances_}).sort_values(by='Importance',ascending=False)
plt.figure(figsize= (8,15))
sns.barplot(data = feat_importance, y = 'Feature', x= 'Importance',orient='h')


# In[33]:


a = perm.feature_importances_
l = []
for i in range(50):
    if a[i]<0:
        l.append('feature_'+str(i))
        
print('Dropped Features')
print(l)


# In[34]:


train_new = train.drop(l,axis=1)
test_new =test.drop(l,axis=1)
X_new = train_new.drop(['id','target'],axis=1)


# # Tuning with OPTUNA

# In[35]:


def fun2(trial, data = X_new, target=y):
    train_x, test_x, train_y, test_y = train_test_split(data,target,train_size=0.8,random_state=42)
    param = {
         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 30.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 30.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),

        'subsample': trial.suggest_uniform('subsample', 0,1),
        'learning_rate': trial.suggest_uniform('learning_rate', 0, 0.1 ),
        'max_depth': trial.suggest_int('max_depth', 1,100),
        'num_leaves' : trial.suggest_int('num_leaves', 2, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'min_child_weight' : trial.suggest_loguniform('min_child_weight' , 1e-5 , 1),
        'cat_smooth' : trial.suggest_int('cat_smooth', 1, 100),
        'cat_l2': trial.suggest_int('cat_l2',1,20),
        'metric': 'multi_logloss', 
        'random_state' : trial.suggest_categorical('random_state',[13]),
        'n_estimators': 10000,
        'objective': 'multiclass',
        'device_type':'gpu'
        
    }
    model = LGBMClassifier(**param)  
    
    model.fit(train_x,train_y,eval_set=[(test_x,test_y)],early_stopping_rounds=200,verbose=False)
    
    pred = model.predict_proba(test_x)
    
    ll = log_loss(test_y, pred)
    
    return ll


# In[36]:


study_2 = optuna.create_study(direction='minimize')
study_2.optimize(fun2, n_trials=100)
print('Number of finished trials:', len(study_2.trials))
print('Best trial:', study_2.best_trial.params)


# In[37]:


best_params_lgbm = study_2.best_params
best_params_lgbm['objective'] = 'multiclass'
best_params_lgbm['metric'] = 'multi_logloss'
best_params_lgbm['n_estimators'] = 10000
best_params_lgbm['device_type'] : 'gpu'


# # LGBM Kfold Predictions

# In[38]:


columns = train_new.drop(['id','target'],axis=1).columns
preds_2 = np.zeros((test.shape[0],4))
lgbm_df = pd.DataFrame(columns = ['Class1m2', 'Class2m2','Class3m2','Class4m2','target'])
kf = StratifiedKFold(n_splits = 10 , random_state = 13 , shuffle = True)
ll =[]
n=0

for tr_idx, test_idx in kf.split(train_new[columns], train_new['target']):
    
    X_tr, X_val = train_new[columns].iloc[tr_idx], train_new[columns].iloc[test_idx]
    y_tr, y_val = train_new['target'].iloc[tr_idx], train_new['target'].iloc[test_idx]
    
    model = LGBMClassifier(**best_params_lgbm)
    
    model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],early_stopping_rounds=500,verbose=False)
    y_pred  = model.predict_proba(X_val)
    df = pd.DataFrame(y_pred,columns=['Class1m2', 'Class2m2','Class3m2','Class4m2'])
    df['target'] = list(y_val)
    
    lgbm_df = pd.concat([lgbm_df,df])
    preds_2+=model.predict_proba(test_new.drop(['id'],axis=1))/kf.n_splits
    ll.append(log_loss(y_val, y_pred))
    print(n+1,ll[n])
    n+=1


# In[39]:


lgbm_df


# In[40]:


np.mean(ll)


# In[41]:


df_kfold_lgbm = pd.DataFrame(preds_2,columns=['Class_1','Class_2','Class_3','Class_4'])
df_kfold_lgbm['id']  = test['id']
df_kfold_lgbm = df_kfold_lgbm[['id','Class_1','Class_2','Class_3','Class_4']]


# In[42]:


df_kfold_lgbm


# In[43]:


output_5 = df_kfold_lgbm.to_csv('submit_5.csv',index=False)


# # XGBoost

# In[44]:


from xgboost import XGBClassifier


# # Feature Selection with Permutation Importance

# In[45]:


model = XGBClassifier(random_State=13).fit(X_train, y_train)
perm = PermutationImportance(model, random_state=13, scoring = 'neg_log_loss')
perm.fit(X_test,y_test)


# In[46]:


feat_importance = pd.DataFrame({'Feature':X_train.columns, 'Importance':perm.feature_importances_}).sort_values(by='Importance',ascending=False)
plt.figure(figsize= (8,15))
sns.barplot(data = feat_importance, y = 'Feature', x= 'Importance',orient='h')


# In[47]:


a = perm.feature_importances_
l = []
for i in range(50):
    if a[i]<0:
        l.append('feature_'+str(i))
        
print('Dropped Features')
print(l)


# In[48]:


train_new = train.drop(l,axis=1)
test_new =test.drop(l,axis=1)
X_new = train_new.drop(['id','target'],axis=1)


# # Tuning with OPTUNA

# In[49]:


def fun3(trial, data = X_new, target = y):
    train_x, test_x, train_y, test_y = train_test_split(data,target,train_size=0.8,random_state=42)

    param = {
       'learning_rate' : trial.suggest_uniform('learning_rate',0,1),
        'gamma' : trial.suggest_uniform('gamma',0,100),
        'max_depth': trial.suggest_int('max_depth', 1,100),
        'min_child_weight' : trial.suggest_uniform('min_child_weight', 0,100),
        'max_delta_step' : trial.suggest_uniform('max_delta_step',1,10),
        'subsample' : trial.suggest_uniform('subsample',0,1),
        'colsample_bytree' : trial.suggest_uniform('colsample_bytree',0,1),
        'lambda' : trial.suggest_uniform('lambda',1e-5,30),
        'alpha' : trial.suggest_uniform('alpha',1e-5,30),
        'tree_method' :'gpu_hist',
        'grow_policy':'lossguide',
        'max_leaves': trial.suggest_int('max_leaves',16,64),
        'random_state' : trial.suggest_categorical('random_state',[13]),
        'objective':'multi:softprob',
        'eval_metric':'mlogloss',
        'predictor':'gpu_predictor'

        
    }
    model = XGBClassifier(**param)  
    
    model.fit(train_x,train_y,eval_set=[(test_x,test_y)],early_stopping_rounds=200,verbose=False)
    pred_y = model.predict_proba(test_x)
    
    ll = log_loss(test_y, pred_y)
    
    return ll
    


# In[50]:


study_3 = optuna.create_study(direction='minimize')
study_3.optimize(fun3, n_trials=100)
print('Number of finished trials:', len(study_3.trials))
print('Best trial:', study_3.best_trial.params)


# In[51]:


best_params_xgb = study_3.best_params
best_params_xgb['objective'] = 'multi:softprob'
best_params_xgb['eval_metric'] = 'mlogloss'
best_params_xgb['grow_policy'] = 'lossguide'
best_params_xgb['n_estimators'] = 10000
best_params_xgb['tree_method'] ='gpu_hist'
best_params_xgb['predictor'] ='gpu_predictor'


# # XGBoost KFOLD Predictions 

# In[52]:


columns = train_new.drop(['id','target'],axis=1).columns
preds_3 = np.zeros((test.shape[0],4))
kf = StratifiedKFold(n_splits = 10 , random_state = 13 , shuffle = True)
xgb_df = pd.DataFrame(columns = ['Class1m3', 'Class2m3','Class3m3','Class4m3','target'])
ll =[]
n=0

for tr_idx, test_idx in kf.split(train_new[columns], train_new['target']):
    
    X_tr, X_val = train_new[columns].iloc[tr_idx], train_new[columns].iloc[test_idx]
    y_tr, y_val = train_new['target'].iloc[tr_idx], train_new['target'].iloc[test_idx]
    
    model = XGBClassifier(**best_params_xgb)
    
    model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],early_stopping_rounds=500,verbose = False)
    y_pred  = model.predict_proba(X_val)
    df = pd.DataFrame(y_pred,columns=['Class1m3', 'Class2m3','Class3m3','Class4m3'])
    df['target'] = list(y_val)
    xgb_df = pd.concat([xgb_df,df])
    
    preds_3+=model.predict_proba(test_new.drop(['id'],axis=1))/kf.n_splits
    ll.append(log_loss(y_val, model.predict_proba(X_val)))
    print(n+1,ll[n])
    n+=1


# In[53]:


xgb_df


# In[54]:


np.mean(ll)


# In[55]:


df_kfold_xgb = pd.DataFrame(preds_3,columns=['Class_1','Class_2','Class_3','Class_4'])
df_kfold_xgb['id']  = test['id']
df_kfold_xgb = df_kfold_xgb[['id','Class_1','Class_2','Class_3','Class_4']]


# In[56]:


df_kfold_xgb


# In[57]:


output_6 = df_kfold_xgb.to_csv('submit_6.csv',index=False)


# # Voting Classifier (Catboost+LGBM+XGBoost)

# In[58]:


preds_combined = (preds+preds_2+preds_3)/3
preds_combined = np.clip(preds_combined,0.05, 0.95)
df_combined = pd.DataFrame(preds_combined,columns=['Class_1','Class_2','Class_3','Class_4'])
df_combined['id'] = test['id']
df_combined = df_combined[['id','Class_1','Class_2','Class_3','Class_4']]


# In[59]:


df_combined


# In[60]:


final_output = df_combined.to_csv('final_submit.csv',index=False)


# # Stacked Model

# In[61]:


stacked_df['Class1m1'] = cb_df['Class1m1']
stacked_df['Class2m1'] = cb_df['Class2m1']
stacked_df['Class3m1'] = cb_df['Class3m1']
stacked_df['Class4m1'] = cb_df['Class4m1']

stacked_df['Class1m2'] = lgbm_df['Class1m2']
stacked_df['Class2m2'] = lgbm_df['Class2m2']
stacked_df['Class3m2'] = lgbm_df['Class3m2']
stacked_df['Class4m2'] = lgbm_df['Class4m2']

stacked_df['Class1m3'] = xgb_df['Class1m3']
stacked_df['Class2m3'] = xgb_df['Class2m3']
stacked_df['Class3m3'] = xgb_df['Class3m3']
stacked_df['Class4m3'] = xgb_df['Class4m3']

stacked_df['target'] = cb_df['target']


test_stacked_df = pd.DataFrame(columns = ['Class1m1', 'Class2m1','Class3m1','Class4m1','Class1m2', 'Class2m2','Class3m2','Class4m2','Class1m3', 'Class2m3','Class3m3','Class4m3'])
test_stacked_df['Class1m1'] = df_kfold['Class_1']
test_stacked_df['Class2m1'] = df_kfold['Class_2']
test_stacked_df['Class3m1'] = df_kfold['Class_3']
test_stacked_df['Class4m1'] = df_kfold['Class_4']

test_stacked_df['Class1m2'] = df_kfold_lgbm['Class_1']
test_stacked_df['Class2m2'] = df_kfold_lgbm['Class_2']
test_stacked_df['Class3m2'] = df_kfold_lgbm['Class_3']
test_stacked_df['Class4m2'] = df_kfold_lgbm['Class_4']

test_stacked_df['Class1m3'] = df_kfold_xgb['Class_1']
test_stacked_df['Class2m3'] = df_kfold_xgb['Class_2']
test_stacked_df['Class3m3'] = df_kfold_xgb['Class_3']
test_stacked_df['Class4m3'] = df_kfold_xgb['Class_4']


# In[62]:


stacked_df


# In[63]:


l=[]
for i in stacked_df['target']:
    l.append(int(i))
    
stacked_df['target'] = l


# In[64]:


preds_stacked = np.zeros((test.shape[0],4))
columns = ['Class1m1', 'Class2m1','Class3m1','Class4m1','Class1m2', 'Class2m2','Class3m2','Class4m2','Class1m3', 'Class2m3','Class3m3','Class4m3']
kf = StratifiedKFold(n_splits = 10 , random_state = 13 , shuffle = True)
ll =[]
n=0

for tr_idx, test_idx in kf.split(stacked_df[columns], stacked_df['target']):
    
    X_tr, X_val = stacked_df[columns].iloc[tr_idx], stacked_df[columns].iloc[test_idx]
    y_tr, y_val = stacked_df['target'].iloc[tr_idx], stacked_df['target'].iloc[test_idx]
    
    model = LGBMClassifier(random_state= 13, objective= 'multiclass', metric = 'multi_logloss')
    
    model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],early_stopping_rounds=500,verbose=False)
    y_pred  = model.predict_proba(X_val)
    
    preds_stacked+=model.predict_proba(test_stacked_df)/kf.n_splits
    ll.append(log_loss(y_val, y_pred))
    print(n+1,ll[n])
    n+=1


# In[65]:


np.mean(ll)


# In[66]:


df_kfold_st = pd.DataFrame(preds_stacked,columns=['Class_1','Class_2','Class_3','Class_4'])
df_kfold_st['id']  = test['id']
df_kfold_st = df_kfold_st[['id','Class_1','Class_2','Class_3','Class_4']]


# In[67]:


df_kfold_st


# In[68]:


stacked_submit = df_kfold_st.to_csv('stacked_submit.csv',index=False)


# ## Thanks, and don't forget to upvote, I'm a beginner it will motivate me!!

# In[ ]:




