#!/usr/bin/env python
# coding: utf-8

# <div style="display:fill;
#            background-color:#DDDDDD;
#            letter-spacing:0.5px;border-bottom: 2px solid black;">
# <img src="https://raw.githubusercontent.com/IqmanS/Machine-Learning-Notebooks/main/smoker_status_prediction/smoking-banner.jpg">
#     
# <H1 style="padding: 20px; color:black; font-weight:600;font-family: 'Garamond', 'Lucida Sans', sans-serif; text-align: center; font-size: 36px;">PREDICTION OF SMOKER STATUS</H1>
# </div>
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import os
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style("dark") # Theme for plots as Dark
sns.set_palette("viridis")
# sns.color_palette("flare")
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import optuna
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from catboost import Pool, CatBoostClassifier, cv


# <div style="background-color: #DDDDDD; padding: 20px; border-radius: 20px; border: 2px solid black;">
#     <h1 style="font-family:  'Garamond', 'Lucida Sans', sans-serif; text-align: center; color: black; font-weight: bold; font-size: 42px;">
#     Table of Contents
#     </h1>
# </div>
# 
# <a href="#1" style="font-family: 'Lucida Sans', 'Lucida Sans', sans-serif; text-align: left; color: #323232;font-size: 22px;"> 1. Dataset Overview </a><br>
# <a href="#2" style="font-family: 'Lucida Sans', 'Lucida Sans', sans-serif; text-align: left; color: #323232;font-size: 22px;"> 2. Basic Feature Engineering </a> <br>
# <a href="#3" style="font-family: 'Lucida Sans', 'Lucida Sans', sans-serif; text-align: left; color: #323232;font-size: 22px;"> 3. Exploratory Data Analysis & Visualization </a> <br>
# <a href="#4" style="font-family: 'Lucida Sans', 'Lucida Sans', sans-serif; text-align: left; color: #323232;font-size: 22px;"> 4. Training Models </a><br>
# <a href="#4.1" style="font-family: 'Lucida Sans', 'Lucida Sans', sans-serif; text-align: left; color: #323232;font-size: 16px;padding-left: 25px;"> 4.1 Baseline LGBM Model </a><br>
# <a href="#4.2" style="font-family: 'Lucida Sans', 'Lucida Sans', sans-serif; text-align: left; color: #323232;font-size: 16px;padding-left: 25px;"> 4.2 Baseline XGB Model </a><br>
# <a href="#4.3" style="font-family: 'Lucida Sans', 'Lucida Sans', sans-serif; text-align: left; color: #323232;font-size: 16px;padding-left: 25px;"> 4.3 Optuna Tuning LGBM Model </a><br>
# <a href="#4.4" style="font-family: 'Lucida Sans', 'Lucida Sans', sans-serif; text-align: left; color: #323232;font-size: 16px;padding-left: 25px;"> 4.4 Optuna Tuning XGB Model </a><br>
# <a href="#4.5" style="font-family: 'Lucida Sans', 'Lucida Sans', sans-serif; text-align: left; color: #323232;font-size: 16px;padding-left: 25px;"> 4.5 Generating More Train Data from Orig Test Data </a><br>
# <a href="#4.6" style="font-family: 'Lucida Sans', 'Lucida Sans', sans-serif; text-align: left; color: #323232;font-size: 16px;padding-left: 25px;"> 4.6 OOF LGBM+CatBoost+XGB Test Preds </a><br>
# <a href="#6" style="font-family: 'Lucida Sans', 'Lucida Sans', sans-serif; text-align: left; color: #323232;font-size: 22px;"> 5. Plotting Feature Importance </a><br>
# <a href="#7" style="font-family: 'Lucida Sans', 'Lucida Sans', sans-serif; text-align: left; color: #323232;font-size: 22px;"> 6. Creating 'submission.csv' </a><br>
# 

# <div id="1" style="background-color: #DDDDDD; padding: 20px; border-radius: 20px; border: 2px solid black;">
#     <h1 style="font-family:  'Garamond', 'Lucida Sans', sans-serif; text-align: center; color: #000000; font-weight: bold; font-size: 42px;">
#     Dataset Overview
#     </h1>
# </div>

# In[ ]:


train_data = pd.read_csv("/kaggle/input/playground-series-s3e24/train.csv",index_col="id")
orig_data = pd.read_csv("/kaggle/input/smoker-status-prediction-using-biosignals/train_dataset.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s3e24/test.csv",index_col="id")
orig_test = pd.read_csv("/kaggle/input/smoker-status-prediction-using-biosignals/test_dataset.csv")
train_data = pd.concat([train_data,orig_data])


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


train_data.describe()


# <div id="2" style="background-color: #DDDDDD; padding: 20px; border-radius: 20px; border: 2px solid black;">
#     <h1 style="font-family:  'Garamond', 'Lucida Sans', sans-serif; text-align: center; color: #000000; font-weight: bold; font-size: 42px;">
#     Basic Feature Engineering
#     </h1>
# </div>

# In[ ]:


train_data['Gtp'].clip(lower = 0, upper = 300)
train_data['HDL'].clip(lower = 0, upper = 150)
train_data['LDL'].clip(lower = 0, upper = 200)
train_data['ALT'].clip(lower = 0, upper = 150)
train_data['AST'].clip(lower = 0, upper = 100)
train_data['serum creatinine'].clip(lower = 0, upper = 3)  

train_data["BMI"] = train_data["weight(kg)"] / (train_data["height(cm)"]/100)**2
train_data["HDL-LDL Ratio"] = train_data["HDL"] / train_data["LDL"]
# train_data["HDL-triglyceride Ratio"] = train_data["HDL"] / train_data["triglyceride"]
# train_data["LDL-triglyceride Ratio"] = train_data["LDL"] / train_data["triglyceride"]
# train_data["HDL-Cholesterol Ratio"] = train_data["HDL"] / train_data["Cholesterol"]
# train_data["LDL-Cholesterol Ratio"] = train_data["LDL"] / train_data["Cholesterol"]
train_data["Liver Enzyme Ratio"] = train_data["AST"] / train_data["ALT"]


# In[ ]:


test_data['Gtp'].clip(lower = 0, upper = 300)
test_data['HDL'].clip(lower = 0, upper = 150)
test_data['LDL'].clip(lower = 0, upper = 200)
test_data['ALT'].clip(lower = 0, upper = 150)
test_data['AST'].clip(lower = 0, upper = 100)
test_data['serum creatinine'].clip(lower = 0, upper = 3)  

test_data["BMI"] = test_data["weight(kg)"] / (test_data["height(cm)"]/100)**2
test_data["HDL-LDL Ratio"] = test_data["HDL"] / test_data["LDL"]
# test_data["HDL-triglyceride Ratio"] = test_data["HDL"] / test_data["triglyceride"]
# test_data["LDL-triglyceride Ratio"] = test_data["LDL"] / test_data["triglyceride"]
# test_data["HDL-Cholesterol Ratio"] = test_data["HDL"] / test_data["Cholesterol"]
# test_data["LDL-Cholesterol Ratio"] = test_data["LDL"] / test_data["Cholesterol"]
test_data["Liver Enzyme Ratio"] = test_data["AST"] / test_data["ALT"]


# In[ ]:


orig_test["BMI"] = orig_test["weight(kg)"] / (orig_test["height(cm)"]/100)**2
orig_test["HDL-LDL Ratio"] = orig_test["HDL"] / orig_test["LDL"]
# orig_test["HDL-triglyceride Ratio"] = orig_test["HDL"] / orig_test["triglyceride"]
# orig_test["LDL-triglyceride Ratio"] = orig_test["LDL"] / orig_test["triglyceride"]
# orig_test["HDL-Cholesterol Ratio"] = orig_test["HDL"] / orig_test["Cholesterol"]
# orig_test["LDL-Cholesterol Ratio"] = orig_test["LDL"] / orig_test["Cholesterol"]
orig_test["Liver Enzyme Ratio"] = orig_test["AST"] / orig_test["ALT"]


# In[ ]:


train_data.head()


# <div id="3" style="background-color: #DDDDDD; padding: 20px; border-radius: 20px; border: 2px solid black;">
#     <h1 style="font-family:  'Garamond', 'Lucida Sans', sans-serif; text-align: center; color: #000000; font-weight: bold; font-size: 42px;">
#     Exploratory Data Analysis & Visualization
#     </h1>
# </div>

# In[ ]:


mask = np.triu(np.ones_like(train_data.corr()))
plt.figure(figsize=(20,12))
sns.heatmap(train_data.corr(), cmap="BuGn", annot=True, mask=mask,vmin=-1,vmax=1);


# In[ ]:


fig,axes = plt.subplots(23,2,figsize=(15, 60),dpi=300)

for ind,col in enumerate(orig_data.columns):
    if train_data[col].nunique()!=2:
        plt.subplot(23,2,2*ind+1)
        sns.histplot(orig_data[col],bins=15,kde=True)
        plt.gca().set_title(col)
    elif col!="smoking":
        plt.subplot(23,2,2*ind+1)
        sns.countplot(data = orig_data,x=col,hue="smoking")
        plt.gca().set_title(col)
    else:
        plt.subplot(23,2,2*ind+1)
        sns.countplot(data = orig_data,x=col)
        plt.gca().set_title(col)
    
    if train_data[col].nunique()!=2:
        plt.subplot(23,2,2*ind+2)
        sns.boxplot(orig_data[col],orient="h",palette="BuGn")
        plt.gca().set_title(col)
    else:
        plt.subplot(23,2,2*ind+2)
        sns.histplot(binwidth=0.5, x=col, hue="smoking", data=orig_data, stat="count", multiple="stack",palette="BuGn")
        plt.gca().set_title(col)

fig.tight_layout()
plt.show()


# <div id="4" style="background-color: #DDDDDD; padding: 20px; border-radius: 20px; border: 2px solid black;">
#     <h1 style="font-family:  'Garamond', 'Lucida Sans', sans-serif; text-align: center; color: #000000; font-weight: bold; font-size: 42px;">
#    Training Models
#     </h1>
# </div>

# In[ ]:


seed = np.random.seed(6)

X = train_data.drop(["smoking"],axis=1)
y = train_data["smoking"]


# <div id="4.1" >
#     <h1 style="font-family:  'Garamond', 'Lucida Sans', sans-serif; text-align: center; color: #263A29; font-weight: bold; font-size: 36px;">
#    4.1 Baseline LGBM Model
#     </h1>
# </div>
# <hr>

# In[ ]:


lgbmmodel = LGBMClassifier(random_state=seed, device="gpu")
print("CV score of LGBM is ",cross_val_score(lgbmmodel,X,y,cv=4, scoring = 'roc_auc').mean())


# <div id="4.2" >
#     <h1 style="font-family:  'Garamond', 'Lucida Sans', sans-serif; text-align: center; color: #263A29; font-weight: bold; font-size: 36px;">
#    4.2 Baseline XGB Model
#     </h1>
# </div>
# <hr>

# In[ ]:


xgbmodel = XGBClassifier(random_state=seed, tree_method= 'gpu_hist')
print("CV score of XGB is ",cross_val_score(xgbmodel,X,y,cv=4, scoring = 'roc_auc').mean())


# <div id="4.3" >
#     <h1 style="font-family:  'Garamond', 'Lucida Sans', sans-serif; text-align: center; color: #263A29; font-weight: bold; font-size: 36px;">
#    4.3 Optuna Tuning LGBM
#     </h1>
# </div>
# <hr>

# In[ ]:


# def objective(trial):
#     params = {
#         'n_estimators' : trial.suggest_int('n_estimators',500,1000),
#         "max_depth":trial.suggest_int('max_depth',3,50),
#         "learning_rate" : trial.suggest_float('learning_rate',1e-4, 0.25, log=True),
#         "min_child_weight" : trial.suggest_float('min_child_weight', 0.5,4),
#         "min_child_samples" : trial.suggest_int('min_child_samples',1,250),
#         "subsample" : trial.suggest_float('subsample', 0.2, 1),
#         "subsample_freq" : trial.suggest_int('subsample_freq',0,5),
#         "colsample_bytree" : trial.suggest_float('colsample_bytree',0.2,1),
#         'num_leaves' : trial.suggest_int('num_leaves', 2, 128),
#     }
#     lgbmmodel_optuna = LGBMClassifier(**params,random_state=seed,device="gpu")
#     cv = cross_val_score(lgbmmodel_optuna, X, y, cv = 4,scoring='roc_auc').mean()
#     return cv

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100,timeout=2000)


# In[ ]:


# 0.8686512553111709
lgbm_params = {'max_depth': 46, 'learning_rate': 0.06953273561619135,
               'min_child_weight': 2.4187716216112944, 'min_child_samples': 230, 'subsample': 0.9515130309407626,
               'subsample_freq': 4, 'colsample_bytree': 0.402284262124352, 'num_leaves': 71}

lgbm_opt = LGBMClassifier(**lgbm_params,n_estimators=624,random_state=seed,device="gpu")
print("CV score of LGBM Optuna is ",cross_val_score(lgbm_opt,X,y,cv=4, scoring = 'roc_auc').mean())


# <div id="4.4" >
#     <h1 style="font-family:  'Garamond', 'Lucida Sans', sans-serif; text-align: center; color: #263A29; font-weight: bold; font-size: 36px;">
#    4.4 Optuna Tuning XGB
#     </h1>
# </div>
# <hr>

# In[ ]:


# def objective(trial):
#     params = {
#         'n_estimators' : trial.suggest_int('n_estimators',500,750),
#         'max_depth':  trial.suggest_int('max_depth',3,50),
#         'min_child_weight': trial.suggest_float('min_child_weight', 2,50),
#         "learning_rate" : trial.suggest_float('learning_rate',1e-4, 0.2,log=True),
#         'subsample': trial.suggest_float('subsample', 0.2, 1),
#         'gamma': trial.suggest_float("gamma", 1e-4, 1.0),
#         "colsample_bytree" : trial.suggest_float('colsample_bytree',0.2,1),
#         "colsample_bylevel" : trial.suggest_float('colsample_bylevel',0.2,1),
#         "colsample_bynode" : trial.suggest_float('colsample_bynode',0.2,1),
#     }
#     xgbmodel_optuna = XGBClassifier(**params,random_state=seed,tree_method = "gpu_hist",eval_metric= "auc")
#     cv = cross_val_score(xgbmodel_optuna, X, y, cv = 4,scoring='roc_auc').mean()
#     return cv

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100,timeout=5000)


# In[ ]:


# CV: 0.869769000693272
xgb_params =   {'max_depth': 44, 'min_child_weight': 42.394074475465935,
                'learning_rate': 0.018945904767046495, 'subsample': 0.9976305222111156,
                'gamma': 0.23054785929528437, 'colsample_bytree': 0.4156956766282452,
                'colsample_bylevel': 0.9225226228188033, 'colsample_bynode': 0.686558727709571}

xgb_opt = XGBClassifier(**xgb_params,n_estimators= 727,random_state=seed,tree_method = "gpu_hist",eval_metric= "auc")
print("CV score of XGB Optuna is ",cross_val_score(xgb_opt,X,y,cv=4, scoring = 'roc_auc').mean())


# <div id="4.5" >
#     <h1 style="font-family:  'Garamond', 'Lucida Sans', sans-serif; text-align: center; color: #263A29; font-weight: bold; font-size: 36px;">
#    4.5 Generating More Train Data from Test Data
#     </h1>
# </div>
# <hr>

# In[ ]:


lgbm_opt.fit(X,y)
xgb_opt.fit(X,y)


# In[ ]:


orig_test_preds = (xgb_opt.predict_proba(orig_test)[:,1]+lgbm_opt.predict_proba(orig_test)[:,1])/2
orig_test["smoking"] = orig_test_preds
orig_test["smoking"][orig_test["smoking"]>0.85] = 1
orig_test["smoking"][orig_test["smoking"]<0.15] = 0
orig_test = orig_test.query('smoking == 1 | smoking == 0')
orig_test["smoking"] = orig_test["smoking"].astype("int")


# In[ ]:


test_pseudo = test_data.copy()
test_pseudo_preds = (xgb_opt.predict_proba(test_pseudo)[:,1]+lgbm_opt.predict_proba(test_pseudo)[:,1])/2
test_pseudo["smoking"] = test_pseudo_preds
test_pseudo["smoking"][test_pseudo["smoking"]>0.85] = 1
test_pseudo["smoking"][test_pseudo["smoking"]<0.15] = 0
test_pseudo = test_pseudo.query('smoking == 1 | smoking == 0')
test_pseudo["smoking"] = test_pseudo["smoking"].astype("int")


# In[ ]:


train_data = pd.concat([train_data,orig_test,test_pseudo])

X = train_data.drop(["smoking"],axis=1)
y = train_data["smoking"]


# <div id="4.6" >
#     <h1 style="font-family:  'Garamond', 'Lucida Sans', sans-serif; text-align: center; color: #263A29; font-weight: bold; font-size: 36px;">
#    4.6 Out-of-Fold Predictions LGBM + CatBoost + XGB
#     </h1>
# </div>
# <hr>

# In[ ]:


submission = pd.DataFrame()
submission["id"] = test_data.index
submission["smoking"] = 0


# In[ ]:


SPLITS = 4
REPEATS = 1
lgbm_auc_score = []
cat_auc_score = []
xgb_auc_score = []
denominator = 0

for i,(tr,val) in enumerate(RepeatedStratifiedKFold(n_splits=SPLITS, n_repeats=REPEATS,random_state=seed).split(X,y)):
    
    print("-"*30,f"FOLD {i+1}/{SPLITS*REPEATS}","-"*30)
    X_train, X_test, y_train, y_test = X.iloc[tr,:],X.iloc[val,:],y.iloc[tr],y.iloc[val]
    
    print("\n->","LGBM:")
    lgbmmodel = LGBMClassifier(**lgbm_params,n_estimators= 7500,random_state=seed,device="gpu")
    lgbmmodel.fit(X_train,y_train, eval_set=[(X_test,y_test)], eval_names=["valid"],eval_metric=['auc'], early_stopping_rounds=500,verbose = 500)
    auc = roc_auc_score(y_test, lgbmmodel.predict_proba(X_test)[:,1])
    lgbm_auc_score.append(auc)
    print(f"\nFold {i+1} ROC_AUC of LGBM =", auc,"\n")
    denominator+=1
    submission["smoking"] += lgbmmodel.predict_proba(test_data)[:,1]
    
    print("\n->","CAT:")
    train_dataset = Pool(data=X.iloc[tr,:],label=y.iloc[tr])
    eval_dataset = Pool(data=X.iloc[val,:],label=y.iloc[val])
    
    catmodel = CatBoostClassifier(iterations=7500,verbose=500, od_type="Iter",eval_metric="AUC", random_seed=seed,early_stopping_rounds=500)
    catmodel.fit(train_dataset, use_best_model=True, eval_set=eval_dataset)
    auc = roc_auc_score(y.iloc[val], catmodel.predict_proba(X.iloc[val,:])[:,1])
    cat_auc_score.append(auc)
    denominator+=1
    submission["smoking"] += catmodel.predict_proba(test_data)[:,1]
    
    print("\n->","XGB:")
    xgbmodel = XGBClassifier(**xgb_params,n_estimators= 7500,random_state=seed, tree_method= 'gpu_hist',eval_metric="auc",early_stopping_rounds = 500)
    xgbmodel.fit(X_train,y_train, eval_set=[(X_test,y_test)],verbose = 500,callbacks=[EarlyStopping(rounds = 500,save_best=True)])
    
    auc = roc_auc_score(y_test, xgbmodel.predict_proba(X_test)[:,1])
    xgb_auc_score.append(auc)
    print(f"\nFold {i+1} ROC_AUC of XGB =", auc,"\n")
    denominator+=1
    submission["smoking"] += xgbmodel.predict_proba(test_data)[:,1]
    
print("\n\n","-"*50,sep="")
print("CV score of LGBM is ",np.array(lgbm_auc_score).mean())
print("CV score of CAT is ",np.array(cat_auc_score).mean())
print("CV score of XGB is ",np.array(xgb_auc_score).mean())


# <div id="6" style="background-color: #DDDDDD; padding: 20px; border-radius: 20px; border: 2px solid black;">
#     <h1 style="font-family:  'Garamond', 'Lucida Sans', sans-serif; text-align: center; color: #000000; font-weight: bold; font-size: 42px;">
#    Plotting Feature Importance
#     </h1>
# </div>

# In[ ]:


def plotImportance(modelNames,models):
    plt.subplots(len(modelNames),1,figsize=(14,5*len(modelNames)),dpi=300)
    for ind,modelName in enumerate(modelNames):
        history = pd.DataFrame()
        history["cols"] = X_test.columns
        if modelNames[ind] == "CatBoost":
            history["imp"] = models[ind].get_feature_importance()
        else:  
            history["imp"] = models[ind].feature_importances_
        history.sort_values("imp",inplace=True,ascending=False)
        history.reset_index(drop=True)
        plt.subplot(len(modelNames),1,ind+1)
        sns.barplot(x=history["imp"],y=history["cols"],palette="rocket");
        plt.title("Feature Imporance of "+modelName)


# In[ ]:


plotImportance(["LGBM","XGB","CatBoost"],[lgbmmodel,xgbmodel,catmodel])


# <div id="7" style="background-color: #DDDDDD; padding: 20px; border-radius: 20px; border: 2px solid black;">
#     <h1 style="font-family:  'Garamond', 'Lucida Sans', sans-serif; text-align: center; color: #000000; font-weight: bold; font-size: 42px;">
#    Creating 'submission.csv'
#     </h1>
# </div>

# In[ ]:


submission.head()


# In[ ]:


submission["smoking"] = submission["smoking"]/denominator
submission.head(10)


# In[ ]:


submission[["id","smoking"]].to_csv("submission.csv",header=True,index=False)

