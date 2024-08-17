#!/usr/bin/env python
# coding: utf-8

# In this notebook, I demonstrate how to use the multiclass focal loss that should help you score better with such imbalanced classes. The focal loss function is from https://github.com/artemmavrin/focal-loss/blob/master/docs/source/index.rst
# 
# The focal loss is a loss that has been devised for object detection problems where the background is more prominent than the objects to be detected. 
# 
# ![](https://github.com/Atomwh/FocalLoss_Keras/raw/master/images/fig1-focal%20loss%20results.png)
# 
# As you increase the gamma value, you put more emphasis on hard to classify examples. There is clearly a trade-off for this (high gamma values can be detrimental), but overall if you set the right value it should perform much better than using other tricks for imbalanced data.
# 
# In order to implement the multiclass focal loss, I referred to this article: 
# 
# 
# 
# This notebook owes quite a lot of ideas from "TPSDEC21-01-Keras Quickstart" (https://www.kaggle.com/ambrosm/tpsdec21-01-keras-quickstart) by @ambrosm please consider upvoting also his work.
# 
# It also implements the feature engineering suggested by @aguschin (see my post https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/291839 for all the references).

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from warnings import filterwarnings
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
import lightgbm as lgbm

filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# In[2]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[3]:


from scipy import optimize
from scipy import special

class FocalLoss:
    """
    source: https://maxhalford.github.io/blog/lightgbm-focal-loss/
    """

    def __init__(self, gamma, alpha=None):
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(self, y_true):
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(),
            bounds=(0, 1),
            method='bounded'
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def lgb_obj(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        return self.grad(y, p), self.hess(y, p)

    def lgb_eval(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        is_higher_better = False
        return 'focal_loss', self(y, p).mean(), is_higher_better


# In[4]:


from joblib import Parallel, delayed
from sklearn.multiclass import _ConstantPredictor
from sklearn.preprocessing import LabelBinarizer
from scipy import special


class OneVsRestLightGBMWithCustomizedLoss:
    """
    source: https://towardsdatascience.com/multi-class-classification-using-focal-loss-and-lightgbm-a6a6dec28872
    """

    def __init__(self, loss, n_jobs=3):
        self.loss = loss
        self.n_jobs = n_jobs

    def fit(self, X, y, **fit_params):

        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        if 'eval_set' in fit_params:
            # use eval_set for early stopping
            X_val, y_val = fit_params['eval_set'][0]
            Y_val = self.label_binarizer_.transform(y_val)
            Y_val = Y_val.tocsc()
            columns_val = (col.toarray().ravel() for col in Y_val.T)
            self.results_ = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_binary)
                                                         (X, column, X_val, column_val, **fit_params) for
                                                         i, (column, column_val) in
                                                         enumerate(zip(columns, columns_val)))
        else:
            # eval set not available
            self.results_ = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_binary)
                                                         (X, column, None, None, **fit_params) for i, column
                                                         in enumerate(columns))

        return self

    def _fit_binary(self, X, y, X_val, y_val, **fit_params):
        unique_y = np.unique(y)
        init_score_value = self.loss.init_score(y)
        if len(unique_y) == 1:
            estimator = _ConstantPredictor().fit(X, unique_y)
        else:
            fit = lgbm.Dataset(X, y, init_score=np.full_like(y, init_score_value, dtype=float))
            filtering = ['eval_set', 'early_stopping_rounds', 'verbose_eval', 'num_boost_round']
            local_fit_params = {item:value for item, value in fit_params.items() if item!='eval_set'}
            
            if 'num_boost_round' in fit_params:
                num_boost_round = fit_params['num_boost_round']
            else:
                num_boost_round = 100
                
            if 'early_stopping_rounds' in fit_params:
                early_stopping_rounds = fit_params['early_stopping_rounds']
            else:
                early_stopping_rounds = 10
                
            if 'verbose_eval'  in fit_params:
                verbose_eval = fit_params['verbose_eval']
            else:
                verbose_eval = 10
                    
            if 'eval_set' in fit_params:
                val = lgbm.Dataset(X_val, y_val, init_score=np.full_like(y_val, init_score_value, dtype=float),
                                  reference=fit)
        
                estimator = lgbm.train(params=local_fit_params,
                                       train_set=fit,
                                       valid_sets=(fit, val),
                                       valid_names=('fit', 'val'),
                                       fobj=self.loss.lgb_obj,
                                       feval=self.loss.lgb_eval,
                                       num_boost_round=num_boost_round,
                                       early_stopping_rounds=early_stopping_rounds,
                                       verbose_eval=verbose_eval)
            else:
                                   
                estimator = lgbm.train(params=local_fit_params,
                                       train_set=fit,
                                       fobj=self.loss.lgb_obj,
                                       feval=self.loss.lgb_eval,
                                       num_boost_round=num_boost_round,
                                       early_stopping_rounds=early_stopping_rounds,
                                       verbose_eval=verbose_eval)

        return estimator, init_score_value

    def predict(self, X):

        n_samples = X.shape[0]
        maxima = np.empty(n_samples, dtype=float)
        maxima.fill(-np.inf)
        argmaxima = np.zeros(n_samples, dtype=int)

        for i, (e, init_score) in enumerate(self.results_):
            margins = e.predict(X, raw_score=True)
            prob = special.expit(margins + init_score)
            np.maximum(maxima, prob, out=maxima)
            argmaxima[maxima == prob] = i

        return argmaxima

    def predict_proba(self, X):
        y = np.zeros((X.shape[0], len(self.results_)))
        for i, (e, init_score) in enumerate(self.results_):
            margins = e.predict(X, raw_score=True)
            y[:, i] = special.expit(margins + init_score)
        y /= np.sum(y, axis=1)[:, np.newaxis]
        return y


# In[5]:


train = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
test = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")
submission = pd.read_csv("../input/tabular-playground-series-dec-2021/sample_submission.csv")


# In[6]:


print("The target class distribution:")
print((train.groupby('Cover_Type').Id.nunique() / len(train)).apply(lambda p: f"{p:.3%}"))


# In[7]:


# Droping Cover_Type 5 label, since there is only one instance of it
train = train[train.Cover_Type != 5]


# In[8]:


# remove unuseful features
train = train.drop([ 'Soil_Type7', 'Soil_Type15'], axis=1)
test = test.drop(['Soil_Type7', 'Soil_Type15'], axis=1)

# extra feature engineering
def r(x):
    if x+180>360:
        return x-180
    else:
        return x+180

def fe(df):
    df['EHiElv'] = df['Horizontal_Distance_To_Roadways'] * df['Elevation']
    df['EViElv'] = df['Vertical_Distance_To_Hydrology'] * df['Elevation']
    df['Aspect2'] = df.Aspect.map(r)
    ### source: https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/293373
    df["Aspect"][df["Aspect"] < 0] += 360
    df["Aspect"][df["Aspect"] > 359] -= 360
    df.loc[df["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
    df.loc[df["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
    df.loc[df["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
    df.loc[df["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
    df.loc[df["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
    df.loc[df["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255
    ########
    df['Highwater'] = (df.Vertical_Distance_To_Hydrology < 0).astype(int)
    df['EVDtH'] = df.Elevation - df.Vertical_Distance_To_Hydrology
    df['EHDtH'] = df.Elevation - df.Horizontal_Distance_To_Hydrology * 0.2
    df['Euclidean_Distance_to_Hydrolody'] = (df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)**0.5
    df['Manhattan_Distance_to_Hydrolody'] = df['Horizontal_Distance_To_Hydrology'] + df['Vertical_Distance_To_Hydrology']
    df['Hydro_Fire_1'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']
    df['Hydro_Fire_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])
    df['Hydro_Road_1'] = abs(df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways'])
    df['Hydro_Road_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])
    df['Fire_Road_1'] = abs(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Roadways'])
    df['Fire_Road_2'] = abs(df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways'])
    df['Hillshade_3pm_is_zero'] = (df.Hillshade_3pm == 0).astype(int)
    return df

train = fe(train)
test = fe(test)

# Summed features pointed out by @craigmthomas (https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/292823)
soil_features = [x for x in train.columns if x.startswith("Soil_Type")]
wilderness_features = [x for x in train.columns if x.startswith("Wilderness_Area")]

train["soil_type_count"] = train[soil_features].sum(axis=1)
test["soil_type_count"] = test[soil_features].sum(axis=1)

train["wilderness_area_count"] = train[wilderness_features].sum(axis=1)
test["wilderness_area_count"] = test[wilderness_features].sum(axis=1)


# In[9]:


y = train.Cover_Type.values - 1
X = reduce_mem_usage(train.drop("Cover_Type", axis=1)).set_index("Id")
Xt = reduce_mem_usage(test).set_index("Id")


# In[10]:


import gc
del([train, test])
_ = [gc.collect() for i in range(5)]


# In[11]:


le = LabelEncoder()
target = le.fit_transform(y)

_, classes_num = np.unique(target, return_counts=True)


# In[12]:


N_FOLDS = 5

### cross-validation 
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1)

predictions = np.zeros((len(Xt), len(le.classes_)))
oof = np.zeros((len(X), len(le.classes_)))
scores = list()

for fold, (idx_train, idx_valid) in enumerate(cv.split(X, y)):
    X_train, y_train = X.iloc[idx_train, :], target[idx_train]
    X_valid, y_valid = X.iloc[idx_valid, :], target[idx_valid]
    
    fit_params = {'eval_set': [(X_valid, y_valid)],
                  'num_boost_round': 1500,
                  'early_stopping_rounds': 30,
                  'verbose_eval': 100
                 }
    
    loss = FocalLoss(alpha=0.75, gamma=2.0)
    model = OneVsRestLightGBMWithCustomizedLoss(loss=loss)

    print('**'*20)
    print(f"Fold {fold+1} || Training")
    print('**'*20)

    model.fit(X_train, y_train, **fit_params)

    predictions += model.predict_proba(Xt) / N_FOLDS
    oof[idx_valid] = model.predict_proba(X_valid)
        
    scores.append(accuracy_score(y_true=y_valid, y_pred=np.argmax(oof[idx_valid], axis=1)))
    print(f"cv accuracy fold {fold+1}: {scores[-1]:0.5f}")


# In[13]:


print(f"Average cv accuracy: {np.mean(scores):0.5f} (std={np.std(scores):0.5f})")


# In[14]:


submission.Cover_Type = le.inverse_transform(np.argmax(predictions, axis=1)) + 1
submission.to_csv("submission.csv", index=False)


# In[15]:


oof = pd.DataFrame(oof, columns=[f"prob_{i}" for i in le.classes_])
oof.insert(loc=0, column='Id', value=range(len(X)))
oof.to_csv("oof.csv", index=False)

