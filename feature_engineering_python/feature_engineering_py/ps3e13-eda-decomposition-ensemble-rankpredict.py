#!/usr/bin/env python
# coding: utf-8

# ## Libraries

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Set the default color palette to "pastel"
sns.set_palette("muted")
import random
import os
from copy import deepcopy
from functools import partial
from itertools import combinations
import random
import gc

# Import sklearn classes for model selection, cross validation, and performance evaluation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from category_encoders import OneHotEncoder, OrdinalEncoder, CountEncoder, CatBoostEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, NMF
from umap import UMAP
from sklearn.manifold import TSNE

# Import libraries for Hypertuning
import optuna

# Import libraries for gradient boosting
import xgboost as xgb
import lightgbm as lgb
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.svm import NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from catboost import Pool

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ## Data

# In[2]:


filepath = '/kaggle/input/playground-series-s3e13'

df_train = pd.read_csv(os.path.join(filepath, 'train.csv'), index_col=[0])
df_test = pd.read_csv(os.path.join(filepath, 'test.csv'), index_col=[0])
original = pd.read_csv('/kaggle/input/vector-borne-disease-prediction/trainn.csv')

df_train['is_generated'] = 1
df_test['is_generated'] = 1
original['is_generated'] = 0

original = original.reset_index()
original['id'] = original['index'] + df_test.index[-1] + 1
original = original.drop(columns = ['index']).set_index('id')
original.prognosis = original.prognosis.str.replace(' ', '_')

df_concat = pd.concat([df_train, original], axis=0).reset_index(drop=True)

target_col = 'prognosis'


# In[3]:


df_train.head()


# In[4]:


df_test.head()


# In[5]:


original.head()


# ## Target Featrue

# In[6]:


def plot_target_feature(df_train, target_col, figsize=(16,5), palette='colorblind', name='Train'):

    fig, ax = plt.subplots(1, 2, figsize = figsize)
    ax = ax.flatten()

    # Pie chart
    ax[0].pie(
        df_train[target_col].value_counts(), 
        shadow=True, 
        explode=[0.05] * len(df_train[target_col].unique()),
        autopct='%1.f%%',
        textprops={'size': 15, 'color': 'white'},
        colors=sns.color_palette(palette, len(df_train[target_col].unique()))
    )

    # Bar plot
    sns.countplot(
        data=df_train, 
        y=target_col, 
        ax=ax[1], 
        palette=palette
    )
    ax[1].yaxis.label.set_size(18)
    plt.yticks(fontsize=12)
    ax[1].set_xlabel('Count', fontsize=20)
    plt.xticks(fontsize=12)

    fig.suptitle(f'Target Feature in {name} Dataset', fontsize=25, fontweight='bold')
    plt.tight_layout()

    # Show the plot
    plt.show()


# In[7]:


plot_target_feature(df_train, target_col, figsize=(16,5), palette='colorblind', name='Train')


# In[8]:


plot_target_feature(original, target_col, figsize=(16,5), palette='colorblind', name='Original')


# In[9]:


plot_target_feature(df_concat, target_col, figsize=(16,5), palette='colorblind', name='Train + Original')


# ### Check Duplicates

# In[10]:


df_concat[df_concat.duplicated()]


# ## Featrue Bar Chart

# In[11]:


def plot_countplots(df, num_cols):
    num_rows = (len(df_test.columns) - 1) // num_cols
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 4*num_rows))
    sns.set(font_scale=1.2, style='whitegrid')

    for i, col_name in enumerate(df_test.columns):
        #if (col_name != 'is_generated') or (col_name != target_col):
        ax = axes[(i-1) // num_cols, (i-1) % num_cols]
        sns.countplot(data=df, x=col_name, ax=ax)
        ax.set_title(f'{col_name.title()}', fontsize=18)
        ax.set_xlabel(col_name.title(), fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()


# In[12]:


plot_countplots(df_concat, num_cols=4)


# In[13]:


def plot_correlation_heatmap(dataframe, name='Train'):
    # Calculate the correlation matrix of a Pandas DataFrame
    corr = dataframe.corr()

    # Create a mask for the upper triangle of the correlation matrix
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # Create a heatmap of the correlation matrix with the masked upper triangle
    plt.figure(figsize=(21, 21))
    sns.heatmap(corr, mask=mask, cmap="BuPu")

    # Add a title and axis labels to the heatmap
    plt.title(f"{name} Correlation Matrix Heatmap", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Show the heatmap
    plt.show()


# In[14]:


plot_correlation_heatmap(df_concat, name='Train')


# In[15]:


plot_correlation_heatmap(df_test, name='Test')


# ## Decomposition Techniques

# In[16]:


class Decomp:
    def __init__(self, n_components, method="pca", scaler_method='standard'):
        self.n_components = n_components
        self.method = method
        self.scaler_method = scaler_method
        
    def dimension_reduction(self, df):
            
        X_reduced = self.dimension_method(df)
        df_comp = pd.DataFrame(X_reduced, columns=[f'{self.method.upper()}_{_}' for _ in range(self.n_components)], index=df.index)
        
        return df_comp
    
    def dimension_method(self, df):
        
        X = self.scaler(df)
        if self.method == "pca":
            comp = PCA(n_components=self.n_components, random_state=0)
            X_reduced = comp.fit_transform(X)
        elif self.method == "nmf":
            comp = NMF(n_components=self.n_components, random_state=0)
            X_reduced = comp.fit_transform(X)
        elif self.method == "umap":
            comp = UMAP(n_components=self.n_components, random_state=0)
            X_reduced = comp.fit_transform(X)
        elif self.method == "tsne":
            comp = TSNE(n_components=self.n_components, random_state=0) # Recommend n_components=2
            X_reduced = comp.fit_transform(X)
        else:
            raise ValueError(f"Invalid method name: {method}")
        
        self.comp = comp
        return X_reduced
    
    def scaler(self, df):
        
        _df = df.copy()
            
        if self.scaler_method == "standard":
            return StandardScaler().fit_transform(_df)
        elif self.scaler_method == "minmax":
            return MinMaxScaler().fit_transform(_df)
        elif self.scaler_method == None:
            return _df.values
        else:
            raise ValueError(f"Invalid scaler_method name")
        
    def get_columns(self):
        return [f'{self.method.upper()}_{_}' for _ in range(self.n_components)]
    
    def transform(self, df):
        X = self.scaler(df)
        X_reduced = self.comp.transform(X)
        df_comp = pd.DataFrame(X_reduced, columns=[f'{self.method.upper()}_{_}' for _ in range(self.n_components)], index=df.index)
        
        return df_comp
    
    @property
    def get_explained_variance_ratio(self):
        
        return np.sum(self.comp.explained_variance_ratio_)


# In[17]:


def decomp_plot(tmp, label, ax):
    sns.scatterplot(x=f"{label}_0", y=f"{label}_1", data=tmp, hue=target_col, alpha=0.7, s=100, palette='muted', ax=ax)

    ax.set_title(f'{label} on prognosis', fontsize=25)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel(f"{label} Component 1", fontsize=20)
    ax.set_ylabel(f"{label} Component 2", fontsize=20)
    ax.legend(prop={'size': 12})

fig, axs = plt.subplots(2, 2, figsize=(20, 20))

for i, method in enumerate(['pca', 'nmf', 'umap', 'tsne']):
    decomp = Decomp(n_components=2, method=method, scaler_method=None)
    tmp = decomp.dimension_reduction(df_concat.drop(target_col, axis=1))
    tmp = pd.concat([df_train, tmp], axis=1)
    decomp_plot(tmp, method.upper(), axs[i//2, i%2])

plt.tight_layout()
plt.suptitle(f"Two-dimensional Decomposition", fontsize=30, fontweight='bold', y=1.02)
plt.show()


# In[18]:


for i, method in enumerate(['pca', 'nmf', 'umap']):
    decomp = Decomp(n_components=5, method=method, scaler_method=None)
    tmp = decomp.dimension_reduction(df_concat.drop(target_col, axis=1))
    sns.set(style="ticks", color_codes=True)
    g = sns.pairplot(
        pd.concat([tmp, df_concat[target_col]], axis=1), 
        diag_kind='kde', 
        kind='scatter', 
        hue=target_col, 
        palette='muted', 
        plot_kws={'s': 20}
    )
    g.fig.suptitle(f"{method.upper()} Five-dimensional Decomposition Pair Plot with KDE diagonal", fontsize=16, fontweight='bold', y=1.02)
    plt.show()


# In[19]:


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))

for i, method in enumerate(['pca', 'nmf', 'umap']):
    decomp = Decomp(n_components=10, method=method, scaler_method=None)
    tmp = decomp.dimension_reduction(df_concat.drop(target_col, axis=1))
    corr = tmp.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, cmap="BuPu", ax=axs[i])
    axs[i].set_title(f"{method.upper()} Correlation Heatmap", fontsize=20)
    
plt.suptitle(f"10-dimensional Decomposition Heatmap", fontsize=30, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# In[20]:


# Create a dictionary mapping each disease name to a corresponding integer value
target_map = {
    'Lyme_disease': 0,
    'Tungiasis': 1,
    'Zika': 2,
    'Rift_Valley_fever': 3,
    'West_Nile_fever': 4,
    'Malaria': 5,
    'Chikungunya': 6,
    'Plague': 7,
    'Dengue': 8,
    'Yellow_Fever': 9,
    'Japanese_encephalitis': 10
}
df_train = df_concat.copy()
swapped_map = {v: k for k, v in target_map.items()}
df_concat[target_col] = df_concat[target_col].replace(target_map).astype(int)

# Concatenate train and original dataframes, and prepare train and test sets
drop_columns = [] # 'is_generated'
X_train = df_concat.drop([target_col]+drop_columns, axis=1).reset_index(drop=True).astype(int)
y_train = df_concat[target_col].reset_index(drop=True)
X_test = df_test.drop(drop_columns, axis=1).reset_index(drop=True).astype(int)

# Feature engineering
X_train_ori, X_test_ori = X_train.copy(), X_test.copy()

# Add dimension_reduction Featrues
n_components = 5
decomp = Decomp(n_components=n_components, method='umap', scaler_method=None)
umap_train = decomp.dimension_reduction(X_train).reset_index(drop=True)
umap_test = decomp.transform(X_test).reset_index(drop=True)
print(f'  --> UMAP(n_components={n_components})')

# Concat Data
X_train = pd.concat([X_train_ori, umap_train], axis=1).reset_index(drop=True)
X_test = pd.concat([X_test_ori, umap_test], axis=1).reset_index(drop=True)
# X_train, X_test = X_train_ori.copy(), X_test_ori.copy()

print("")
print(f"X_train shape :{X_train.shape}")
print(f"y_train shape :{y_train.shape}")
print(f"X_test shape :{X_test.shape}")

# Delete the train and test dataframes to free up memory
del df_train, df_test, df_concat, original, umap_train, umap_test

X_train.head(5)


# ## Define Model

# In[21]:


class Splitter:
    def __init__(self, kfold=True, n_splits=5):
        self.n_splits = n_splits
        self.kfold = kfold

    def split_data(self, X, y, random_state_list):
        if self.kfold:
            for random_state in random_state_list:
                kf = StratifiedKFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, y):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val
        else:
            raise ValueError(f"Invalid kfold: Must be True")

class Classifier:
    def __init__(self, n_estimators=100, device="cpu", random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.len_models = len(self.models)
        
    def _define_model(self):
        
        xgb_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.1,
            'n_jobs': -1,
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob',
            'tree_method': 'hist',
            'verbosity': 0,
            'random_state': self.random_state,
        }
        if self.device == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['predictor'] = 'gpu_predictor'
        
        lgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.20,
            'colsample_bytree': 0.56,
            'reg_alpha': 0.25,
            'reg_lambda': 5e-08,
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state
        }
                
        cb_params = {
            'iterations': self.n_estimators,
            'depth': 7,
            'learning_rate': 0.1,
            'l2_leaf_reg': 0.7,
            'random_strength': 0.2,
            'max_bin': 200,
            'od_wait': 65,
            'one_hot_max_size': 70,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'MultiClass',
            'loss_function': 'MultiClass',
            'task_type': self.device.upper(),
            'random_state': self.random_state
        }
                
        models = {
            'svc': SVC(gamma="auto", probability=True, random_state=self.random_state),
            'xgb': xgb.XGBClassifier(**xgb_params),
            'lgb': lgb.LGBMClassifier(**lgb_params),
            'cat': CatBoostClassifier(**cb_params),
            'brf': BalancedRandomForestClassifier(n_estimators=4000, n_jobs=-1, random_state=self.random_state),
            'rf': RandomForestClassifier(n_estimators=1000, random_state=self.random_state),
        }
        
        return models


# ## Optimizer (--> Optimize Logloss)

# In[22]:


class OptunaWeights:
    def __init__(self, random_state, n_trials=3000):
        self.study = None
        self.weights = None
        self.random_state = random_state
        self.n_trials = n_trials

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 1e-12, 1) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds), axis=0, weights=weights)

        # Calculate the Logloss score for the weighted prediction
        score = log_loss(y_true, weighted_pred)
        return score

    def fit(self, y_true, y_preds):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptunaWeights", direction='minimize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds), axis=0, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds):
        self.fit(y_true, y_preds)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights


# ## Optimizer (--> Optimize MAP@3)
# I'm using this Optimize MAP@3 this time. [Original code I created](https://www.kaggle.com/code/tetsutani/ps3e13-ensemble-by-map-3-baseline)

# In[23]:


class OptunaWeights:
    def __init__(self, random_state, n_trials=2000):
        self.study = None
        self.weights = None
        self.random_state = random_state
        self.n_trials = n_trials

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 1e-12, 1) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds), axis=0, weights=weights)

        # Calculate the MAP@3 score for the weighted prediction
        top_preds = np.argsort(-weighted_pred, axis=1)[:, :3]
        score = mapk(y_true.reshape(-1, 1), top_preds, 3)
        
        return score

    def fit(self, y_true, y_preds):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        self.study = optuna.create_study(sampler=sampler, study_name="OptunaWeights", direction='maximize') # minimize
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds), axis=0, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds):
        self.fit(y_true, y_preds)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights
    
def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


# ## Train Model by K-fold

# In[24]:


kfold = True
n_splits = 1 if not kfold else 5 # 15
random_state = 8741
random_state_list = [70669, 26564, 12642] # used by split_data [70669, 26564]
n_estimators = 9999 # 9999
early_stopping_rounds = 200
verbose = False
device = 'cpu'
atk = 3
splitter = Splitter(kfold=kfold, n_splits=n_splits)

# Initialize an array for storing test predictions
test_predss = np.zeros((X_test.shape[0], len(y_train.unique())))
ensemble_score = []
ensemble_mapk_score = []
weights = []
trained_models = {'xgb':[], 'lgb':[], 'cat':[], 'rf':[]}
rank_df = pd.DataFrame(columns=swapped_map.keys(), index=X_test.index).fillna(0)

    
for i, (X_train_, X_val, y_train_, y_val) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):
    n = i % n_splits
    m = i // n_splits
            
    # Get a set of Regressor models
    classifier = Classifier(n_estimators, device, random_state)
    models = classifier.models
    
    # Initialize lists to store oof and test predictions for each base model
    oof_preds = []
    test_preds = []
    
    # Loop over each base model and fit it to the training data, evaluate on validation data, and store predictions
    for name, model in models.items():
        if name in ['xgb', 'lgb', 'cat', 'lgb2']:
            model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        else:
            model.fit(X_train_, y_train_)
            
        if name in trained_models.keys():
            trained_models[f'{name}'].append(deepcopy(model))
        
        test_pred = model.predict_proba(X_test)
        y_val_pred = model.predict_proba(X_val)

        top_preds = np.argsort(-y_val_pred, axis=1)[:, :atk]
        mapk_score = mapk(y_val.values.reshape(-1, 1), top_preds, atk)
        
        score = log_loss(y_val, y_val_pred)
        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] MAP@{atk}: {mapk_score:.5f}, Logloss: {score:.5f}')
        
        oof_preds.append(y_val_pred)
        test_preds.append(test_pred)
    
    # Use Optuna to find the best ensemble weights
    optweights = OptunaWeights(random_state=random_state)
    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)
    
    score = log_loss(y_val, y_val_pred)
    print(f'Ensemble [FOLD-{n} SEED-{random_state_list[m]}] MAP@{atk}: {mapk_score:.5f}, Logloss: {score:.5f}')
    ensemble_score.append(score)
    ensemble_mapk_score.append(mapk_score)
    weights.append(optweights.weights)
    
    # Predict to X_test by the best ensemble weights
    _test_preds = optweights.predict(test_preds)
    test_predss += _test_preds / (n_splits * len(random_state_list))
    
    # Rank Prediction
    for i in range(_test_preds.shape[0]):
        arr = _test_preds[i]
        sorted_indices = np.argsort(arr)
        for k, p in zip(range(1, 10), [9, 8, 7, 6, 5, 4, 3, 2, 1]):
            second_largest_index = sorted_indices[-k]
            rank_df.loc[i, second_largest_index] += p
    
    gc.collect()


# In[25]:


# Calculate the mean LogLoss score of the ensemble
mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Ensemble Logloss score {mean_score:.5f} ± {std_score:.5f}')

# Print the mean and standard deviation of the ensemble weights for each model
print('--- Model Weights ---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name}: {mean_weight:.5f} ± {std_weight:.5f}')


# ## Visualize Feature importance (XGBoost, LightGBM, Catboost)

# In[26]:


def visualize_importance(models, feature_cols, title, head=10):
    importances = []
    feature_importance = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["importance"] = model.feature_importances_
        _df["feature"] = pd.Series(feature_cols)
        _df["fold"] = i
        _df = _df.sort_values('importance', ascending=False)
        _df = _df.head(head)
        feature_importance = pd.concat([feature_importance, _df], axis=0, ignore_index=True)
        
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    # display(feature_importance.groupby(["feature"]).mean().reset_index().drop('fold', axis=1))
    plt.figure(figsize=(18, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance, color='skyblue', errorbar='sd')
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(f'{title} Feature Importance', fontsize=18)
    plt.grid(True, axis='x')
    plt.show()
    
for name, models in trained_models.items():
    visualize_importance(models, list(X_train.columns), name)


# ## Make Submission

# In[27]:


n_cols = 4
n_rows = np.ceil(test_predss.shape[1] / n_cols).astype(int)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10))
axs = axs.ravel()

for i in range(test_predss.shape[1]):
    sns.histplot(data=test_predss[:, i], ax=axs[i])
    axs[i].set_title(f"{swapped_map[i]}")
    
fig.suptitle(f'Ensemble softmax output', fontweight='bold')
fig.tight_layout(pad=1.5)
plt.show()


# ### Rank Prediction

# In[28]:


sub = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
for i in range(test_predss.shape[0]):
    rank_list = list(rank_df.iloc[i, :].sort_values(0, ascending=False).iloc[:3].index)
    sub.loc[i, 'prognosis'] = ' '.join([swapped_map[_] for _ in rank_list])
    
# Save submission file
sub.to_csv('submission_rank.csv', index=False)
sub


# ### SoftMax Prediction

# In[29]:


sub = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
df = pd.DataFrame(test_predss, columns=swapped_map.values())
sub[f'{target_col}'] = df.apply(lambda x: ' '.join(x.nlargest(3).index.tolist()), axis=1)
sub.to_csv('submission.csv', index=False)
sub

