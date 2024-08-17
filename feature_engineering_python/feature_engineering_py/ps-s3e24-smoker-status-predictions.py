#!/usr/bin/env python
# coding: utf-8

# # <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#243139; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #000000"> Import</p>

# In[1]:


# Misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from copy import deepcopy
from functools import partial
import gc
import warnings
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Import sklearn classes for model selection, cross validation, and performance evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from category_encoders import OneHotEncoder, OrdinalEncoder, CountEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing

# Import libraries for Hypertuning
import optuna

# Import libraries for gradient boosting
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from catboost import Pool


# In[2]:


# Seaborn
rc = {
    #FAEEE9
    "axes.facecolor": "#243139",
    "figure.facecolor": "#243139",
    "axes.edgecolor": "#000000",
    "grid.color": "#000000",
    "font.family": "arial",
    "axes.labelcolor": "#FFFFFF",
    "xtick.color": "#FFFFFF",
    "ytick.color": "#FFFFFF",
    "grid.alpha": 0.4
}
sns.set(rc=rc)

# Useful line of code to set the display option so we could see all the columns in pd dataframe
pd.set_option('display.max_columns', None)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Functions
def print_sl():
    print("=" * 50)
    print()


# # <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#243139; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #000000"> Load Data</p>

# In[3]:


train = pd.read_csv('/kaggle/input/playground-series-s3e24/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e24/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s3e24/sample_submission.csv')

train_orig = pd.read_csv('/kaggle/input/smoker-status-prediction-using-biosignals/train_dataset.csv')

train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)

print('Data Loaded Succesfully!')
print_sl()

print(f'train shape: {train.shape}')
print(f'are there any null values in train: {train.isnull().any().any()}\n')

print(f'test shape: {test.shape}')
print(f'are there any null values in test: {test.isnull().any().any()}\n')

print(f'train_orig shape: {train_orig.shape}')
print(f'are there any null values in test: {train_orig.isnull().any().any()}\n')

num_cols = train.columns.delete(-1)

target = 'smoking'

train.head()


# # <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#243139; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #000000">EDA</p>

# In[4]:


# https://www.kaggle.com/code/kimtaehun/eda-and-baseline-with-multiple-models
def plot_count(df: pd.core.frame.DataFrame, col: str, title_name: str='Train') -> None:
    # Set background color
    
    f, ax = plt.subplots(1, 2, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.2)

    s1 = df[col].value_counts()
    N = len(s1)

    outer_sizes = s1
    inner_sizes = s1/N

    outer_colors = ['#59b3a3', '#433C64']
    inner_colors = ['#59b3a3', '#433C64']
    #['#ff6905', '#ff8838', '#ffa66b']

    ax[0].pie(
        outer_sizes,colors=outer_colors, 
        labels=s1.index.tolist(), 
        startangle=90, frame=True, radius=1.3, 
        explode=([0.05]*(N-1) + [.3]),
        wedgeprops={'linewidth' : 1, 'edgecolor' : 'white'}, 
        textprops={'fontsize': 12, 'weight': 'bold', 'color': 'white'}
    )

    textprops = {
        'size': 13, 
        'weight': 'bold', 
        'color': 'white'
    }

    ax[0].pie(
        inner_sizes, colors=inner_colors,
        radius=1, startangle=90,
        autopct='%1.f%%', explode=([.1]*(N-1) + [.3]),
        pctdistance=0.8, textprops=textprops
    )

    center_circle = plt.Circle((0,0), .68, color='black', fc='#243139', linewidth=0)
    ax[0].add_artist(center_circle)

    x = s1
    y = s1.index.tolist()
    sns.barplot(
        x=x, y=y, ax=ax[1],
        palette='mako', orient='horizontal'
    )

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(
        axis='x',         
        which='both',      
        bottom=False,       
        labelbottom=False
    )

    for i, v in enumerate(s1):
        ax[1].text(v, i+0.1, str(v), color='white', fontweight='bold', fontsize=12)

    plt.setp(ax[1].get_yticklabels(), fontweight="bold")
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    ax[1].set_xlabel(col, fontweight="bold", color='white')
    ax[1].set_ylabel('count', fontweight="bold", color='white')

    f.suptitle(f'{title_name}', fontsize=14, fontweight='bold', color='white')
    plt.tight_layout() 
    plt.show()


# In[5]:


plot_count(train, 'smoking', 'Target Variable Distribution of Synthesized Data')


# In[6]:


plot_count(train_orig, 'smoking', 'Target Variable Distribution of Original Data')


# In[7]:


# Create a copy of the dataframe
df = train.copy()

def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str = 'Train correlation') -> None:
    excluded_columns = ['id']
    columns_without_excluded = [col for col in df.columns if col not in excluded_columns]
    corr = df[columns_without_excluded].corr()
    
    fig, axes = plt.subplots(figsize=(14, 10))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap='mako', annot=True, annot_kws={"size": 6})
    plt.title(title_name, color='white')
    plt.show()

# Plot correlation heatmap for encoded dataframe
plot_correlation_heatmap(df, 'Dataset Correlation')


# # <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#243139; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #000000">Feature Engineering</p>

# In[8]:


total = pd.concat([train, train_orig], ignore_index=True)
total.drop_duplicates(inplace=True)


# In[9]:


# https://www.kaggle.com/code/arunklenin/ps3e24-eda-feature-engineering-ensemble#4.-FEATURE-ENGINEERING
def create_extra_features(df):
    best = np.where(df['hearing(left)'] < df['hearing(right)'], 
                    df['hearing(left)'],  df['hearing(right)'])
    worst = np.where(df['hearing(left)'] < df['hearing(right)'], 
                     df['hearing(right)'],  df['hearing(left)'])
    df['hearing(left)'] = best - 1
    df['hearing(right)'] = worst - 1
    
    df['eyesight(left)'] = np.where(df['eyesight(left)'] > 9, 0, df['eyesight(left)'])
    df['eyesight(right)'] = np.where(df['eyesight(right)'] > 9, 0, df['eyesight(right)'])
    best = np.where(df['eyesight(left)'] < df['eyesight(right)'], 
                    df['eyesight(left)'],  df['eyesight(right)'])
    worst = np.where(df['eyesight(left)'] < df['eyesight(right)'], 
                     df['eyesight(right)'],  df['eyesight(left)'])
    df['eyesight(left)'] = best
    df['eyesight(right)'] = worst
    ##
    df['Gtp'] = np.clip(df['Gtp'], 0, 300)
    df['HDL'] = np.clip(df['HDL'], 0, 110)
    df['LDL'] = np.clip(df['LDL'], 0, 200)
    df['ALT'] = np.clip(df['ALT'], 0, 150)
    df['AST'] = np.clip(df['AST'], 0, 100)
    df['serum creatinine'] = np.clip(df['serum creatinine'], 0, 3)  
    
    return df
total=create_extra_features(total)
test=create_extra_features(test)


# In[10]:


y_train = total.smoking
X_train = total.drop(columns=['smoking'])
X_test = test

X_train.reset_index(drop='index', inplace=True)
y_train.reset_index(drop='index', inplace=True)


# # <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#243139; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #000000">Model Building</p>

# In[11]:


class Splitter:
    def __init__(self, n_splits=5, cat_df=pd.DataFrame(), test_size=0.5):
        self.n_splits = n_splits
        self.cat_df = cat_df
        self.test_size = test_size

    def split_data(self, X, y, random_state_list):
        for random_state in random_state_list:
            kf = KFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
            for train_index, val_index in kf.split(X, y):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                yield X_train, X_val, y_train, y_val, val_index


# In[12]:


class Classifier:
    def __init__(self, n_estimators=100, device="cpu", random_state=42):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self.get_models()
        self.models_name = list(self.get_models().keys())
        self.len_models = len(self.models)
        
    def get_models(self):
        
        xgb_optuna1 = {
            'n_estimators': 1500,
            'learning_rate': 0.08901459197907591,
            'booster': 'gbtree',
            'lambda': 8.550251116462702,
            'alpha': 6.92130114930949,
            'eta': 0.7719873740829137,
            'grow_policy': 'lossguide',
            'n_jobs': -1,
            'objective': 'binary:logistic',
            'verbosity': 0,
            'random_state': self.random_state
        }
        
        xgb_optuna2 = {
            'n_estimators': 550,
            'learning_rate': 0.014551680348136895,
            'booster': 'gbtree',
            'lambda': 0.028738149876528587,
            'alpha': 0.014056635017117198,
            'subsample': 0.538653498449084,
            'colsample_bytree': 0.518050828371974, 
            'max_depth': 4, 'min_child_weight': 4,
            'eta': 0.6953619445477833,
            'gamma': 0.9036568111424781,
            'grow_policy': 'lossguide',
            'n_jobs': -1,
            'objective': 'binary:logistic',
            'verbosity': 0,
            'random_state': self.random_state
        }
        
        xgb0_params = {
            'n_estimators': 2048,
            'max_depth': 9,
            'learning_rate': 0.045,
            'booster': 'gbtree',
            'subsample': 0.75,
            'colsample_bytree': 0.30,
            'reg_lambda': 1.00,
            'reg_alpha': 0.80,
            'gamma': 0.80,
            'random_state': self.random_state,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
        }
        
        models = {
            "xgbo1": xgb.XGBClassifier(**xgb_optuna1),
            #"xgbo2": xgb.XGBClassifier(**xgb_optuna2),
            "xgb0": xgb.XGBClassifier(**xgb0_params),
            'rf': RandomForestClassifier(n_estimators=500, n_jobs=-1, class_weight="balanced", random_state=self.random_state),
            #'lr': LogisticRegressionCV(max_iter=2000, random_state=self.random_state)
        }
        return models


# In[13]:


class OptunaWeights:
    def __init__(self, random_state, n_trials=100):
        self.study = None
        self.weights = None
        self.random_state = random_state
        self.n_trials = n_trials

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 1e-15, 1) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)

        # Calculate the score for the weighted prediction
        score = roc_auc_score(y_true, weighted_pred)
        return score

    def fit(self, y_true, y_preds):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptunaWeights", direction='maximize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds):
        self.fit(y_true, y_preds)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights


# In[14]:


# Config
n_splits = 20
random_state = 42
random_state_list =[42]
n_estimators = 333
device = 'cpu'
early_stopping_rounds = 444
verbose = False

# Split Data
splitter = Splitter(n_splits=n_splits)
splits = splitter.split_data(X_train, y_train, random_state_list=random_state_list)

# Initialize an array for storing test predictions
classifier = Classifier(n_estimators=n_estimators, device=device, random_state=random_state)
test_predss = np.zeros((X_test.shape[0]))
oof_predss = np.zeros((X_train.shape[0]))
ensemble_score = []
weights = []
models_name = [_ for _ in classifier.models_name if ('xgb' in _) or ('lgb' in _) or ('cat' in _)]
trained_models = dict(zip(models_name, [[] for _ in range(classifier.len_models)]))
score_dict = dict(zip(classifier.models_name, [[] for _ in range(len(classifier.models_name))]))

for i, (X_train_, X_val, y_train_, y_val, val_index) in enumerate(splits):
    
    n = i % n_splits
    m = i // n_splits
    

    # Classifier models
    classifier = Classifier(n_estimators, device, random_state)
    models = classifier.models

    # Store oof and test predictions for each base model
    oof_preds = []
    test_preds = []

    # Loop over each base model and fit it
    for name, model in models.items():
        model.fit(X_train_, y_train_)
            
        if name in trained_models.keys():
            trained_models[f'{name}'].append(deepcopy(model))

        test_pred = model.predict_proba(X_test)[:, 1]
        y_val_pred = model.predict_proba(X_val)[:, 1]

        score = roc_auc_score(y_val, y_val_pred)
        score_dict[name].append(score)
        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] ROC-AUC score: {score:.5f}')

        oof_preds.append(y_val_pred)
        test_preds.append(test_pred)

    # Use OptunaWeights
    optweights = OptunaWeights(random_state)
    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)

    score = roc_auc_score(y_val, y_val_pred)
    print(f'Ensemble [FOLD-{n} SEED-{random_state_list[m]}] ROC-AUC score {score:.5f} \n')
    ensemble_score.append(score)
    weights.append(optweights.weights)

    # Predict to X_test by the best ensemble weights
    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))
    oof_predss[X_val.index] = optweights.predict(oof_preds)

    gc.collect()


# In[15]:


# Calculate the mean score of the ensemble
mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Mean Optuna Ensemble {mean_score:.5f} ± {std_score:.5f} \n')

print('--- Optuna Weights---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name}: {mean_weight:.5f} ± {std_weight:.5f}')


# # <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#243139; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #000000">Submission</p>

# In[16]:


sample_sub['smoking'] = test_predss
sample_sub.to_csv(f'submission.csv', index=False)
sample_sub

