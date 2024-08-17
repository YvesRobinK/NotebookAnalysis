#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
from sklearn.metrics import mean_squared_error

import gc
from joblib import dump, Parallel, delayed
from tqdm import tqdm
import multiprocessing
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# In[2]:


def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=-1)(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)


# In[3]:


from decimal import ROUND_HALF_UP, Decimal
def adjust_price(price):
    # transform Date column into datetime
    price.loc[:, "Date"] = pd.to_datetime(price.loc[:, "Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # generate CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        # generate AdjustedClose
        df.loc[:, "Close"] = (
            df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        # reverse order
        df = df.sort_values("Date")
        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df["Close"] == 0, "Close"] = np.nan
        # forward fill AdjustedClose
        df.loc[:, "Close"] = df.loc[:, "Close"].ffill()
        return df

    # generate AdjustedClose
    price = price.sort_values(["SecuritiesCode", "Date"])
    
    price = applyParallel(price.groupby('SecuritiesCode'), generate_adjusted_close)
    price = price.reset_index(drop=True)

    price.set_index("Date", inplace=True)
    return price


# In[4]:


import numpy as np
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _BaseKFold

class GroupTimeSeriesSplit(_BaseKFold):
    def __init__(self, n_splits=5, *, max_train_size=None):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        
        n_splits = self.n_splits
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_folds = n_splits + 1
        indices = np.arange(n_samples)
        group_counts = np.unique(groups, return_counts=True)[1]
        groups = np.split(indices, np.cumsum(group_counts)[:-1])
        n_groups = _num_samples(groups)
        if n_folds > n_groups:
            raise ValueError(("Cannot have number of folds ={0} greater than the number of groups: {1}.").format(n_folds, n_groups))
            
        test_size = (n_groups // n_folds)
        test_starts = range(test_size + n_groups % n_folds, n_groups, test_size)
        for test_start in test_starts:
            if self.max_train_size:
                train_start = np.searchsorted(
                    np.cumsum(
                        group_counts[:test_start][::-1])[::-1] < self.max_train_size + 1, 
                        True)
                yield (np.concatenate(groups[train_start:test_start]),
                       np.concatenate(groups[test_start:test_start + test_size]))
            else:
                yield (np.concatenate(groups[:test_start]),
                       np.concatenate(groups[test_start:test_start + test_size]))


# In[5]:


def set_rank(df):
    """
    Args:
        df (pd.DataFrame): including predict column
    Returns:
        df (pd.DataFrame): df with Rank
    """
    # sort records to set Rank
    df = df.sort_values("predict", ascending=False)
    # set Rank starting from 0
    df.loc[:, "Rank"] = np.arange(len(df["predict"]))
    return df

# https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition

def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio


# # Config

# In[6]:


class CONFIG:
    use_lb = False
    kaggle = True
    splits = 5
    kaggle_path = "../input/jpx-tokyo-stock-exchange-prediction/"
    local_path = ""
    random_seed = 69420


# # Preprocessing

# In[7]:


get_ipython().run_cell_magic('time', '', 'if CONFIG.use_lb:\n    if CONFIG.kaggle:\n        prices = pd.concat([\n            pd.read_csv(CONFIG.kaggle_path+"train_files/stock_prices.csv"),\n            pd.read_csv(CONFIG.kaggle_path+"supplemental_files/stock_prices.csv")\n        ])\n    else:\n        prices = pd.concat([\n            pd.read_csv(CONFIG.local_path+"train_files/stock_prices.csv", engine="pyarrow"),\n            pd.read_csv(CONFIG.local_path+"supplemental_files/stock_prices.csv", engine="pyarrow")\n        ])\nelse:\n    if CONFIG.kaggle:\n        prices = pd.concat([\n            pd.read_csv(CONFIG.kaggle_path+"train_files/stock_prices.csv"),\n        ])\n    else:\n        prices = pd.concat([\n            pd.read_csv(CONFIG.local_path+"train_files/stock_prices.csv", engine="pyarrow"),\n        ])\n')


# In[8]:


prices


# In[9]:


get_ipython().run_cell_magic('time', '', 'prices = adjust_price(prices)\n')


# In[10]:


prices.head()


# In[11]:


prices = prices.sort_index()


# In[12]:


prices = prices.drop(
    [
        "ExpectedDividend", "RowId", "AdjustmentFactor", "SupervisionFlag", "CumulativeAdjustmentFactor",
    ],
    axis=1
)


# In[13]:


prices.head()


# In[14]:


nullvaluecheck = pd.DataFrame(prices.isna().sum().sort_values(ascending=False)*100/prices.shape[0],columns=['missing %']).head(10)
nullvaluecheck.style.background_gradient(cmap='PuBu')


# In[15]:


prices = prices.fillna(method='pad')


# In[16]:


nullvaluecheck = pd.DataFrame(prices.isna().sum().sort_values(ascending=False)*100/prices.shape[0],columns=['missing %']).head(10)
nullvaluecheck.style.background_gradient(cmap='PuBu')


# # Feature Engineering

# In[17]:


def feature_engineer(df):
    df['feature-avg_price'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    df['feature-median_price'] = df[['Open', 'High', 'Low', 'Close']].median(axis=1)
    df['feature-price_std'] = df[['feature-median_price', 'feature-avg_price']].std(axis=1)
    df['feature-ohlc_std'] = df[['Open', 'High', 'Low', 'Close']].std(axis=1)
      
    df['feature-v_avg'] = np.log(df['Volume']*df['feature-avg_price']+1)
    
    df['feature-median/avg'] = df['feature-median_price'] / df['feature-avg_price']
    df['feature-median-avg'] = df['feature-median_price'] - df['feature-avg_price']
    
    df['feature-BOP'] = (df['Open']-df['Close'])/(df['High']-df['Low'])
    df['feature-OC'] = df['Open'] * df['Close']
    df['feature-HL'] = df['High'] * df['Low']
    df['feature-logC'] = np.log(df['Close']+1)
    df['feature-OHLCskew'] = df[['Open','Close','High','Low']].skew(axis=1)
    df['feature-OHLCkur'] = df[['Open','Close','High','Low']].kurtosis(axis=1)
    df['feature-Cpos'] = (df['Close']-df['Low'])/(df['High']-df['Low']) -0.5
    df['feature-bsforce'] = df['feature-Cpos'] * df['Volume']
    df['feature-Opos'] = (df['Open']-df['Low'])/(df['High']-df['Low']) -0.5
    
    feat_cols = df.columns[df.columns.str.contains('feature')]
    for col in feat_cols:
        df[f'reciprocal-{col}'] = 1/df[col]
        
    df = df.replace([np.inf, -np.inf], 0)

    return df


# In[18]:


get_ipython().run_cell_magic('time', '', "prices = applyParallel(prices.groupby('SecuritiesCode'), feature_engineer)\n")


# In[19]:


_ = gc.collect


# In[20]:


print(prices.shape)
prices.head()


# In[21]:


from joblib.externals.loky import get_reusable_executor
get_reusable_executor().shutdown(wait=True)


# In[22]:


_ = gc.collect()


# In[23]:


prices.shape


# In[24]:


nullvaluecheck = pd.DataFrame(prices.isna().sum().sort_values(ascending=False)*100/prices.shape[0],columns=['missing %']).head(5)
nullvaluecheck.style.background_gradient(cmap='PuBu')


# In[25]:


prices = prices.dropna()


# In[26]:


nullvaluecheck = pd.DataFrame(prices.isna().sum().sort_values(ascending=False)*100/prices.shape[0],columns=['missing %']).head(5)
nullvaluecheck.style.background_gradient(cmap='PuBu')


# In[27]:


prices = prices.sort_index()


# In[28]:


prices.head()


# # Features

# In[29]:


features = prices.columns.drop('Target').to_list()
features


# In[30]:


dump(features, "features.joblib")


# # Cross Validation Split

# In[31]:


import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]

                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size

            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]


            if self.verbose > 0:
                    pass

            yield [int(i) for i in train_array], [int(i) for i in test_array]


# # Data check

# In[32]:


prices[np.isinf(prices.values)]


# In[33]:


nullvaluecheck = pd.DataFrame(prices.isna().sum().sort_values(ascending=False)*100/prices.shape[0],columns=['missing %']).head(5)
nullvaluecheck.style.background_gradient(cmap='PuBu')


# # Train Models

# # LGBM

# In[34]:


from lightgbm import LGBMRegressor, early_stopping, log_evaluation
def train_lgbm(prices, folds):
    models = []
    scores = []
    feature_importance = []
    
    groups, _ = pd.factorize(prices.index.day.astype('str') + '_' + prices.index.month.astype('str') + '_' + prices.index.year.astype('str'))
    
    FOLDS                = 5
    GROUP_GAP            = 20

    kf = PurgedGroupTimeSeriesSplit(
        n_splits = FOLDS,
        group_gap = GROUP_GAP,
    )
    
    for f, (t_, v_) in enumerate(kf.split(X=prices, groups=groups)):
        print(f"{'='*25} Fold {f+1} {'='*25}")
        
        X_train = prices[features].iloc[t_]
        y_train = prices["Target"].iloc[t_]
        X_valid = prices[features].iloc[v_]
        y_valid = prices["Target"].iloc[v_]

        params = {
            'objective': 'rmse',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators':50000,
            'device':'gpu',
            'n_jobs':-1,
            'random_state':CONFIG.random_seed,
            'extra_trees': True,
        }

        
        model = LGBMRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[early_stopping(100), log_evaluation(0)]
        )

        feature_importance.append(model.feature_importances_)
        
        oof_preds = model.predict(X_valid)
        oof_score = np.sqrt(mean_squared_error(y_valid, oof_preds))
        
        print(f"RMSE: {round(oof_score, 4)}")
        models.append(model)
        dump(model, f"model_{f}.joblib", compress=3)
        
        result = prices.iloc[v_]
        result.loc[:, "predict"] = oof_preds
        result.loc[:, "Target"] = y_valid
        result = result.sort_values(["Date", "predict"], ascending=[True, False])
        result = result.groupby("Date").apply(set_rank)
        
        sharpe_scores = calc_spread_return_sharpe(result, portfolio_size=200)
        scores.append(sharpe_scores)
        print('Validation sharpe = {:.4f}'.format(sharpe_scores))
        
        del X_train, y_train, X_valid, y_valid, result, model
        _ = gc.collect()

    return models, scores, feature_importance


# In[35]:


get_ipython().run_cell_magic('time', '', 'models, scores, feature_importance = train_lgbm(prices, CONFIG.splits)\n')


# In[36]:


median_score = np.median(scores)
std = np.std(scores)
score = round(median_score-(median_score*std), 4)

plt.figure(figsize=[6, 5])
plt.boxplot(scores)
plt.title("Scores Box Plot")
plt.axhline(np.median(scores), c='r', ls=':', label='Median Score')
plt.axhline(np.mean(scores), c='g', ls=':', label='Mean Score')
plt.ylabel("Validation Sharpe")
plt.legend()
plt.tight_layout()

print(f"Median Sharpe: {np.median(scores):.6}, Std: {np.std(scores):.2}")
print(f"Fold-Adjusted Score: {score}")


# # Make Predictions & Submit

# In[37]:


del prices
_ = gc.collect()


# In[38]:


import jpx_tokyo_market_prediction
from tqdm.auto import tqdm
env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

for (prices, options, financials, trades, secondary_prices, sample_prediction) in tqdm(iter_test):
    print("Adjusting Price...")
    prices = adjust_price(prices)
    print("Adding Features...")
    prices = applyParallel(prices.groupby('SecuritiesCode'), feature_engineer)
    prices.fillna(method='pad')
    
    prices = prices[features]
    
    print("Predicting Model...")
    lgbm_preds = []
    for model in models:
        lgbm_preds.append(model.predict(prices))
        
    lgbm_preds = np.mean(lgbm_preds, axis=0)
    sample_prediction["Prediction"] = lgbm_preds
    
    print("Ranking...")
    sample_prediction = sample_prediction.sort_values(by = "Prediction", ascending=False)
    sample_prediction.Rank = np.arange(0,2000)
    sample_prediction = sample_prediction.sort_values(by = "SecuritiesCode", ascending=True)
    sample_prediction.drop(["Prediction"],axis=1)
    submission = sample_prediction[["Date","SecuritiesCode","Rank"]]
    
    display(submission)
    
    env.predict(submission)


# In[ ]:





# In[ ]:




