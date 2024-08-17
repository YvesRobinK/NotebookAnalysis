#!/usr/bin/env python
# coding: utf-8

# # 📆 Statement dates - Is there any valuable information for feature engineering?
# ______________________________
# *updated 2022-07-17 [@roma-upgini](https://www.kaggle.com/romaupgini)*  🗣 Share this notebook: [Shareable Link](https://www.kaggle.com/romaupgini/statement-dates-to-use-or-not-to-use)
# 
# 
# ## Four hypothesis on statement dates for feature engineering:
# 
# 1️⃣ There is last statement date seasonality for default rate and we can increase prediction accuracy by adding time dependent features for the last statement date (sin, cos, number of day, week etc). We have 31 days of statements (from 03/01 till 03/31). So, we can only check weekly seasonality.  
# 
# 2️⃣ There is an influence on default rate from holidays before/after the last statement, which might change income or spend structure for the households. We have to specify country for the holidays, but we can guess that for AMEX with a relatively high accuracy.  
# 
# 3️⃣ There is an influence on default rate from local economic situation, like emloyment, price inflation rates, central bank rates etc. Same - we'll need a country for that. However, 31 days of March 2018 most probably won't be enough to catch correlation with macroeconomical situation on training phase. But we have test set for April 2019 (public LB part) and October 2019 (private LB part) with 13-19 month shift from train period - enough for macroeconomics influence. It's worth to try.  
# 
# 4️⃣ Changes in statement dates for 13 month observation period has additional information correlated with default. From what I know about credit cards - consumer might change statement date, for example after salary date change. Which, in turn, might be the signal for employment change.
# 
# **Let's check them one by one.**
# ______________________________
# 
# ## Packages and functions
# 
# 📚 In this notebook we'll use:
# * [Upgini](https://github.com/upgini/upgini#readme) - Free automated data and feature enrichment library for machine learning applications <a href="https://github.com/upgini/upgini">
#     <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"  align='center'>
# </a>
# * [CuDF from RAPIDS.ai](https://github.com/rapidsai/cudf) - DataFrame on GPU
# 
# **Switch on Internet and GPU for this kernel!**

# In[1]:


get_ipython().run_line_magic('pip', 'install -Uq upgini')
import pandas as pd, numpy as np
import sklearn
import matplotlib.pyplot as plt, gc, os
import seaborn as sns
import cupy, cudf

# RANDOM SEED
SEED = 42
# FILL NAN VALUE
NAN_VALUE = -127

def read_file2cudf(path = '', usecols = None):
    # LOAD DATAFRAME
    if usecols is not None: df = cudf.read_parquet(path, columns=usecols)
    else: df = cudf.read_parquet(path)
    # REDUCE DTYPE FOR CUSTOMER AND DATE
    df['customer_ID'] = df['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    df.S_2 = cudf.to_datetime( df.S_2 )
    print('shape of data:', df.shape)
    return df

# CALCULATE SIZE OF EACH SEPARATE TEST PART
def get_rows(customers, test, NUM_PARTS = 4, verbose = ''):
    chunk = len(customers)//NUM_PARTS
    if verbose != '':
        print(f'We will process {verbose} data as {NUM_PARTS} separate parts.')
        print(f'There will be {chunk} customers in each part (except the last part).')
        print('Below are number of rows in each part:')
    rows = []

    for k in range(NUM_PARTS):
        if k==NUM_PARTS-1: cc = customers[k*chunk:]
        else: cc = customers[k*chunk:(k+1)*chunk]
        s = test.loc[test.customer_ID.isin(cc)].shape[0]
        rows.append(s)
    if verbose != '': print( rows )
    return rows,chunk

def xgb_amex(y_pred, y_true):
    return 'amex', amex_metric_np(y_pred,y_true.get_label())
def lgb_amex_metric(y_pred, y_true):
    return 'amex', amex_metric_np(y_pred,y_true.get_label()), True

# code by @https://www.kaggle.com/yunchonggan
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/328020
def amex_metric_np(preds: np.ndarray, target: np.ndarray) -> float:
    n_pos = np.sum(target)
    n_neg = target.shape[0] - n_pos

    indices = np.argsort(preds)[::-1]
    preds, target = preds[indices], target[indices]

    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight * (1 / weight.sum())).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = np.sum(target[four_pct_mask]) / n_pos

    lorentz = (target * (1 / n_pos)).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    g = gini / gini_max
    return 0.5 * (g + d)

# official metric
import pandas as pd
def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


# ## Quick data exploration - Last statement date

# In[2]:


train = pd.read_parquet('../input/amex-data-integer-dtypes-parquet-format/train.parquet', columns=['S_2','customer_ID'])
train['S_2'] = pd.to_datetime(train['S_2'])
train = train.groupby('customer_ID')['S_2'].agg('max').reset_index() #last statement only

df_train_labels = pd.read_csv('../input/amex-default-prediction/train_labels.csv')
train = train.merge(df_train_labels, on='customer_ID')
del df_train_labels
_ = gc.collect()
print("Shape of data: ",train.shape)


# First, let's check number of statements and default rates by day. Where **default rate** is a ratio of customers with default on specific date.

# In[3]:


plot_df=train.groupby("S_2").target.count()
new_train = pd.DataFrame(train.groupby("S_2").target.mean())
print (f"Number of days in train set: {new_train['target'].count()}")
print (f"Standard deviation of Default ratio: {new_train['target'].std()}")

fig, ax = plt.subplots(2,1,figsize = (20,8))
plot_df.plot(title = "# Statements", ax = ax[0])
new_train.plot(title = "% of Defaults", ax = ax[1])
plt.show()
del plot_df, ax, fig


# Interesting, most likely there are
# * either a strong card sales seasonality (most probably statement date derived from card activation date) - ie very few card sales on Sundays, and a big spike on Saturdays
# * or strong preferences on statement dates by customers themselfs
# 
# >Disclaimer - I'm not an AMEX customer, but for most of the banks you can choose statement date, so I'm extrapolating here
# 
# Next, default ratio by day.  
# There is a deviation between days, but it's hard to guess weither it is significant or not.
# **And we have two options here:**
# 1. Let's assume it's a time series, where **y** is a Default Ratio. Then, if there is an influence like in Hypothesis #3 (macroeconomic influence), it must be some trend component in this TS. So we can check Stationarity of TS using [Augmented Dickey-Fuller test](https://en.wikipedia.org/wiki/Augmented_Dickey–Fuller_test) with a significance level of less than 5%.  
# The intuition behind this test is that it determines how strongly a time series is defined by a trend. However, there is a case, when it's doesn't help us - Hypothesis #1. As Augmented Dickey-Fuller test  won't detect seasonal component (it's a stationary TS). **So we have to use something different.**
# 
# 2. Let's auto generate **A LOT** of features from Holiday calendars, Workweek calendars, Political calendars, Sport calendars, sin, cos for month/week, add economic indicators and financial market data by using any data enrichment library. Then do feature selection with feature permutation. In this case - only features which has a statistically significant influence on model accuracy will be picked up. **Here we'll be able to test ALL Hypothesis #1, #2, #3 AT ONCE.**
# 
# So let's do Option 2, as it's quicker, using [Upgini](https://github.com/upgini/upgini#readme) - Free automated data and feature enrichment library for machine learning applications.   
# It will add automatically a lot of external information about dates, holidays, events, financial markets, consumer sentiments, weather etc. all for the specific country / location. Than automatically checks for relevance (ie influence on prediction accuracy improvement) and select only features which will improve it.  
# Full [list of data scources and features, such as weather features, calendar features, financial features, etc ](https://github.com/upgini/upgini#-connected-data-sources-and-coverage)

# ## Hypothesis 1️⃣, 2️⃣, 3️⃣ test with Upgini automated data and feature enrichment library
# 
# To initiate search with Upgini library, you need to define so called [*search keys*](https://github.com/upgini/upgini#-search-key-types-we-support-more-is-coming) - a set of columns to join external data sources and features. In this competition we can use the following keys:
# 
# 1. Column **date** should be used as **SearchKey.DATE**.;  
# 2. **Country** as "US" and "UK" (ISO-3166 country code), as most of AMEX customers are from US, next major market is UK.
#     
# With this set of search keys, our X dataset will be matched with [different date-specific features](https://github.com/upgini/upgini#-connected-data-sources-and-coverage), taking into account the country. Than relevant selection and ranking will be done.  
#   
# To start the search, we need to initiate *scikit-learn* compartible `FeaturesEnricher` transformer with appropriate **search** parameters.    
# After that, we can call the **fit** or **fit_transform**  method of `features_enricher`.

# In[4]:


from upgini import FeaturesEnricher, SearchKey
from upgini.dataset import Dataset

enricher = FeaturesEnricher(
    date_format="%Y-%m-%d",
    search_keys={"S_2": SearchKey.DATE},
    country_code = "US", # change that to UK for another run
)
Dataset.MIN_ROWS_COUNT = 20 #small X dataset, removed internal checks


# For `FeaturesEnricher.fit()` method, just like in all scikit-learn transformers, we should pass **X_train** as the first argument and **y_train** as the second argument.   
# **y_train** is needed to select **only relevant features & datasets, which will improve accuracy**. And rank new external features according to their prediction contribution, calculated as a SHAP values.  

# In[5]:


enricher.fit(
    new_train.drop(columns="target").reset_index(),
    new_train["target"]
)
del enricher, train, new_train
_ = gc.collect()


# ### 🏁 Conclusion for Hypothesis #1, #2 and #3
# 
# **No relevant external features on dates found - both for US, and for UK.**  
# So we have **to reject** Hypothesis #1,#2 and #3 for this training dataset.  
# 
# IMHO, there must be some influence, but to catch that, we need different training data structured in a following way:
# 
# 1. 3 months of statements for default customers, minimum
# 2. gap window between months in 3 month set - at least 6 months  
# 
# Than we'll have 15 months observation window in a following schema: **1 + 6 months gap + 1 + 6 months gap + 1**

# ## Hypothesis 4️⃣ test, using optimized public notebook from [@cdeotte](https://www.kaggle.com/code/cdeotte/xgboost-starter-0-793) (🙏)
# 
# Baseline Public score for this notebook was **0.793**, local CV **0.794**    
# I made following impovements
# * CV with Stratification and 4 Folds
# * Changed CV metric from OOF Score to Average Score
# * Removed features with changes in cals methodology between train and LBs: B_29 and S_9
# * Added "after-pay" features
# * Removed NaN replacement
# * Permutation feature selection
# * More feature eng. with additional statistics on last observations (1500+ features)
# * Changed hyperparams for XGB
# 
# Public score after these changes **0.796** with local CV **0.79639** which is more consistent **"Local CV to LB match"** than before
# 
# Now, let's calculate features from statement dates distances and compare results on the Public LB after enrichment with this new features.   
# For **quick** estimation we'll do 2 steps:
# 1. Calculate Distance between statement dates. Impute first observation / only one statement case with mean() value.
# 2. Calculate Stat. features for Statement dates distance column
# 
# ### Quick feature engineering on distance between statement dates:

# In[6]:


def feature_engineer(df):
    cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
    
    # Initial feature selection to speed up fitting, based on @ambros
    # https://www.kaggle.com/code/ambrosm/amex-lightgbm-quickstart/notebook
    features_avg = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_28', 'B_30', 'B_32', 'B_33', 'B_37', 'B_38', 'B_39', 'B_40', 'B_41', 'B_42', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_50', 'D_51', 'D_53', 'D_54', 'D_55', 'D_58', 'D_59', 'D_60', 'D_61', 'D_62', 'D_65', 'D_66', 'D_69', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_75', 'D_76', 'D_77', 'D_78', 'D_80', 'D_82', 'D_84', 'D_86', 'D_91', 'D_92', 'D_94', 'D_96', 'D_103', 'D_104', 'D_108', 'D_112', 'D_113', 'D_114', 'D_115', 'D_117', 'D_118', 'D_119', 'D_120', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_126', 'D_128', 'D_129', 'D_131', 'D_132', 'D_133', 'D_134', 'D_135', 'D_136', 'D_140', 'D_141', 'D_142', 'D_144', 'D_145', 'P_2', 'P_3', 'P_4', 'R_1', 'R_2', 'R_3', 'R_7', 'R_8', 'R_9', 'R_10', 'R_11', 'R_14', 'R_15', 'R_16', 'R_17', 'R_20', 'R_21', 'R_22', 'R_24', 'R_26', 'R_27', 'S_3', 'S_5', 'S_6', 'S_7', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_18', 'S_22', 'S_23', 'S_25', 'S_26']
    features_min = ['B_2', 'B_4', 'B_5', 'B_9', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_19', 'B_20', 'B_28', 'B_33', 'B_36', 'B_42', 'D_39', 'D_41', 'D_42', 'D_45', 'D_46', 'D_48', 'D_50', 'D_51', 'D_53', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_62', 'D_70', 'D_71', 'D_74', 'D_75', 'D_78', 'D_83', 'D_102', 'D_112', 'D_113', 'D_115', 'D_118', 'D_119', 'D_121', 'D_122', 'D_128', 'D_132', 'D_140', 'D_141', 'D_144', 'D_145', 'P_2', 'P_3', 'R_1', 'R_27', 'S_3', 'S_5', 'S_7', 'S_11', 'S_12', 'S_23', 'S_25']
    features_max = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_21', 'B_23', 'B_24', 'B_25', 'B_30', 'B_33', 'B_37', 'B_38', 'B_39', 'B_40', 'B_42', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_49', 'D_50', 'D_52', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_61', 'D_63', 'D_64', 'D_65', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_76', 'D_77', 'D_78', 'D_80', 'D_82', 'D_84', 'D_91', 'D_102', 'D_105', 'D_107', 'D_110', 'D_111', 'D_112', 'D_115', 'D_116', 'D_117', 'D_118', 'D_119', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_126', 'D_128', 'D_131', 'D_132', 'D_133', 'D_134', 'D_135', 'D_136', 'D_138', 'D_140', 'D_141', 'D_142', 'D_144', 'D_145', 'P_2', 'P_3', 'P_4', 'R_1', 'R_3', 'R_5', 'R_6', 'R_7', 'R_8', 'R_10', 'R_11', 'R_14', 'R_17', 'R_20', 'R_26', 'R_27', 'S_3', 'S_5', 'S_7', 'S_8', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27']
    features_last = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_26', 'B_28', 'B_30', 'B_32', 'B_33', 'B_36', 'B_37', 'B_38', 'B_39', 'B_40', 'B_41', 'B_42', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_49', 'D_50', 'D_51', 'D_52', 'D_53', 'D_54', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_61', 'D_62', 'D_63', 'D_64', 'D_65', 'D_69', 'D_70', 'D_71', 'D_72', 'D_73', 'D_75', 'D_76', 'D_77', 'D_78', 'D_79', 'D_80', 'D_81', 'D_82', 'D_83', 'D_86', 'D_91', 'D_96', 'D_105', 'D_106', 'D_112', 'D_114', 'D_119', 'D_120', 'D_121', 'D_122', 'D_124', 'D_125', 'D_126', 'D_127', 'D_130', 'D_131', 'D_132', 'D_133', 'D_134', 'D_138', 'D_140', 'D_141', 'D_142', 'D_145', 'P_2', 'P_3', 'P_4', 'R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6', 'R_7', 'R_8', 'R_9', 'R_10', 'R_11', 'R_12', 'R_13', 'R_14', 'R_15', 'R_19', 'R_20', 'R_26', 'R_27', 'S_3', 'S_5', 'S_6', 'S_7', 'S_8', 'S_11', 'S_12', 'S_13', 'S_16', 'S_19', 'S_20', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27']
    features_last = list(set(features_last)-set(cat_features))
    features_max = list(set(features_max)-set(cat_features))
    features_min = list(set(features_min)-set(cat_features))
    features_avg = list(set(features_avg)-set(cat_features))
    
    # Drop non stable features for train-test, based on % of NaNs
    #https://www.kaggle.com/code/onodera1/amex-eda-comparison-of-training-and-test-data    
    df.drop(["B_29","S_9"], axis=1, inplace = True)
    
    # Hypothesis #4 - retrieve info from statement dates as distance between the dates
    # Than calculate 'mean', 'std', 'max', 'last' statistics for distances
    # cudf doesn't support diff() as GroupBy function, slow pandas DF used
    temp = df[["customer_ID","S_2"]].to_pandas()
    temp["SDist"]=temp.groupby("customer_ID")["S_2"].diff() / np.timedelta64(1, 'D')
    # Impute with average distance 30.53 days
    temp['SDist'].fillna(30.53, inplace=True)
    df = cudf.concat([df,cudf.from_pandas(temp["SDist"])], axis=1)
    del temp
    _ = gc.collect()
    features_last.append('SDist')
    features_avg.append('SDist')
    features_max.append('SDist')
    features_min.append('SDist')
    
    #https://www.kaggle.com/competitions/amex-default-prediction/discussion/328514
    df.loc[(df.R_13==0) & (df.R_17==0) & (df.R_20==0) & (df.R_8==0), 'R_6'] = 0
    df.loc[df.B_39==-1, 'B_36'] = 0
    
    # Compute "after pay" features
    # https://www.kaggle.com/code/jiweiliu/rapids-cudf-feature-engineering-xgb
    for bcol in [f'B_{i}' for i in [11,14,17]]+['D_39','D_131']+[f'S_{i}' for i in [16,23]]:
        for pcol in ['P_2','P_3']:
            if bcol in df.columns:
                df[[f'{bcol}-{pcol}']] = df[bcol] - df[pcol]
                features_last.append(f'{bcol}-{pcol}')
                features_avg.append(f'{bcol}-{pcol}')
                features_max.append(f'{bcol}-{pcol}')
                features_min.append(f'{bcol}-{pcol}')
                
    # BASIC FEATURE ENGINEERING
    # https://www.kaggle.com/code/huseyincot/amex-agg-data-how-it-created
    # https://www.kaggle.com/code/jiweiliu/rapids-cudf-feature-engineering-xgb
    
    test_num_last = df.groupby("customer_ID")[features_last].agg(['last','first'])
    test_num_last.columns = ['_'.join(x) for x in test_num_last.columns]
    test_num_min = df.groupby("customer_ID")[features_min].agg(['min'])
    test_num_min.columns = ['_'.join(x) for x in test_num_min.columns]
    test_num_max = df.groupby("customer_ID")[features_max].agg(['max'])
    test_num_max.columns = ['_'.join(x) for x in test_num_max.columns]
    test_num_avg = df.groupby("customer_ID")[features_avg].agg(['mean'])
    test_num_avg.columns = ['_'.join(x) for x in test_num_avg.columns]
    test_num_std = df.groupby("customer_ID")[list(set().union(features_avg,features_last,features_min,features_max))].agg(['std','quantile'])
    test_num_std.columns = ['_'.join(x) for x in test_num_std.columns]

    test_cat_agg = df.groupby("customer_ID")[cat_features].agg(['last','first'])
    test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]
   
    #add last statement date, statements count and "new customer" category (LT=0.5)
    test_date_agg = df.groupby("customer_ID")[["S_2","B_3","D_104"]].agg(['last','count'])
    test_date_agg.columns = ['_'.join(x) for x in test_date_agg.columns]
    test_date_agg.rename(columns = {'S_2_count':'LT','S_2_last':'S_2'}, inplace = True)
    test_date_agg.loc[(test_date_agg.B_3_last.isnull()) & (test_date_agg.LT==1),'LT'] = 0.5
    test_date_agg.loc[(test_date_agg.D_104_last.isnull()) & (test_date_agg.LT==1),'LT'] = 0.5
    test_date_agg.drop(["B_3_last","D_104_last","B_3_count","D_104_count"], axis=1, inplace = True)
    
    df = cudf.concat([test_date_agg, test_num_last, test_num_min, test_num_max, test_num_avg, test_num_std, test_cat_agg], axis=1)
    del test_date_agg, test_num_last, test_num_min, test_num_max, test_num_avg, test_num_std, test_cat_agg
    
    # Ratios/diffs on last values as features, based on @ragnar123
    # https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977
    for col in list(set().union(features_last,features_avg)):
        try:
            df[f'{col}_last_first_div'] = df[f'{col}_last'] / df[f'{col}_first']
            df[f'{col}_last_mean_sub'] = df[f'{col}_last'] - df[f'{col}_mean']
            df[f'{col}_last_mean_div'] = df[f'{col}_last'] / df[f'{col}_mean']
            df[f'{col}_last_max_div'] = df[f'{col}_last'] / df[f'{col}_max']
            df[f'{col}_last_min_div'] = df[f'{col}_last'] / df[f'{col}_min']
        except:
            pass
        
    print('shape after engineering', df.shape )
    return df


# In[7]:


train = []
_ = gc.collect()
# raddar Kaggle dataset
# https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format
PATH="../input/amex-data-integer-dtypes-parquet-format/train.parquet"
train = read_file2cudf(path = PATH)
train = feature_engineer(train)

# ADD TARGETS
targets = cudf.read_csv('../input/amex-default-prediction/train_labels.csv')
#targets = cudf.read_csv('/content/drive/MyDrive/colab/train_labels.csv')
targets['customer_ID'] = targets['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
targets = targets.set_index('customer_ID')
train = train.merge(targets, left_index=True, right_index=True, how='left')
train.target = train.target.astype('int8')
del targets

# cudf merge above randomly shuffles rows
train = train.sort_index().reset_index()

# FEATURES
# remove S_2 from FEATURES list
FEATURES = train.columns[2:-1]
print(f'There are {len(FEATURES)} features!')


# ### XGB Training on GPU
# Training with 4 folds, it will take 1h 30min on Kaggle's P100 GPU

# In[8]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import StratifiedKFold\nimport xgboost as xgb\n\n# FOLDS PER MODEL, as number of weeks in a month\nFOLDS = 4\n\n# XGB MODEL PARAMETERS\nxgb_parms = {\n            \'objective\': \'binary:logitraw\', \n            \'tree_method\': \'gpu_hist\',\n            \'predictor\':\'gpu_predictor\',\n            \'max_depth\': 7,\n            \'subsample\':0.88,\n            \'colsample_bytree\': 0.1,\n            \'gamma\':1.5,\n            \'min_child_weight\':8,\n            \'lambda\': 50,\n            \'eta\':0.03,\n            \'learning_rate\':0.02,\n            \'random_state\':SEED\n    }\n\nimportances = []\noof = []\nTRAIN_SUBSAMPLE = 1.0\n_ = gc.collect()\n\nfor j, cat_index in enumerate ([train[train.LT>0].index]):\n    score = 0\n    skf = StratifiedKFold(n_splits=FOLDS, shuffle = True, random_state=SEED)\n    for fold,(train_idx, valid_idx) in enumerate(skf.split(\n                train.loc[cat_index,:][["customer_ID"]],\n                train.loc[cat_index,:].target.values.get())):\n\n        # TRAIN WITH SUBSAMPLE OF TRAIN FOLD DATA\n        if TRAIN_SUBSAMPLE<1.0:\n            np.random.seed(SEED)\n            train_idx = np.random.choice(train_idx, \n                           int(len(train_idx)*TRAIN_SUBSAMPLE), replace=False)\n            np.random.seed(None)\n\n        print(\'#\'*25)\n        print(\'### Fold\',fold+1)\n        print(\'### Train size\',len(train_idx),\'Valid size\',len(valid_idx))\n        print(f\'### Training with {int(TRAIN_SUBSAMPLE*100)}% fold data...\')\n        print(\'#\'*25)\n\n        # TRAIN, VALID, TEST FOR FOLD K\n        y_valid = train.loc[valid_idx, \'target\']\n        dtrain = xgb.DMatrix(data=train.loc[train_idx, FEATURES],\n                             label=train.loc[train_idx, \'target\'],\n                             )\n        dvalid = xgb.DMatrix(data=train.loc[valid_idx, FEATURES],\n                             label=y_valid,\n                             )\n        \n        # TRAIN MODEL FOLD K\n        model = xgb.train(xgb_parms, \n                    dtrain=dtrain,\n                    evals=[(dtrain,\'train\'),(dvalid,\'valid\')],\n                    num_boost_round=8000,\n                    early_stopping_rounds=1800,\n                    custom_metric=xgb_amex,\n                    maximize=True,\n                    verbose_eval=200) \n        model.save_model(f\'XGB_fold{fold}_LT{j}.json\')\n        del dtrain\n        _ = gc.collect()\n\n        # GET FEATURE IMPORTANCE FOR FOLD K\n        dd = model.get_score(importance_type=\'weight\')\n        df = pd.DataFrame({\'feature\':dd.keys(),f\'importance_{fold}\':dd.values()})\n        importances.append(df)\n\n        # INFER OOF FOLD K\n        oof_preds = model.predict(dvalid, iteration_range=(0,model.best_ntree_limit))\n        acc = amex_metric(pd.DataFrame({\'target\':y_valid.values.get()}), \n                                        pd.DataFrame({\'prediction\':oof_preds}))\n        print(\'Kaggle Metric =\',acc,\'\\n\')\n        score += acc\n\n        # SAVE OOF\n        df = train.loc[valid_idx, [\'customer_ID\',\'target\'] ].to_pandas()\n        df[\'oof_pred\'] = oof_preds\n        oof.append( df )\n\n        del dvalid, y_valid, model, dd, df\n        _ = gc.collect()\n\n    score /= FOLDS\n    print(\'Average CV Kaggle Metric for group =\',score)\n\nprint(\'#\'*25)\noof = pd.concat(oof,axis=0,ignore_index=True).set_index(\'customer_ID\')\nscore = amex_metric(pd.DataFrame({\'target\':oof.target.values}), \n                                pd.DataFrame({\'prediction\':oof.oof_pred.values}))\nprint(\'OOF CV Kaggle Metric =\',score)\n# CLEAN RAM\ndel oof, skf, cat_index\ndel train\n_ = gc.collect()\n')


# Local CV with the new features has a score **0.7964**  
# Baseline solution without Statement distance features had **0.79639** on local CV, which is slightly less.   
# Keep going.
# 
# ### Feature importance of Statement Dates features
# Let's check feature importance for TOP 20 vars and for "SDist" vars (derived from Statement dates distance)

# In[9]:


import matplotlib.pyplot as plt

df = importances[0].copy()
for k in range(1,FOLDS*(j+1)): df = df.merge(importances[k], on='feature', how='left')
df['importance'] = df.iloc[:,1:].mean(axis=1)
df = df.sort_values('importance',ascending=False)

NUM_FEATURES = 40
plt.figure(figsize=(10,5*NUM_FEATURES//10))
plt.barh(np.arange(NUM_FEATURES,0,-1), df.importance.values[:NUM_FEATURES])
plt.yticks(np.arange(NUM_FEATURES,0,-1), df.feature.values[:NUM_FEATURES])
plt.title(f'Feature Importance - Top {NUM_FEATURES}')
plt.show()

df = df[df.feature.str.find("SDis") != -1]
plt.figure(figsize=(10,5*df.shape[0]//10))
plt.barh(np.arange(df.shape[0],0,-1), df.importance.values[:df.shape[0]])
plt.yticks(np.arange(df.shape[0],0,-1), df.feature.values[:df.shape[0]])
plt.title(f'XGB Feature Importance - Statement date features')
plt.show()

del df, plt, importances
_ = gc.collect()


# **11 features** from Statement distance was selected, but none of them in TOP 40.   
# So we might notice small improvement from them on Public LB.  
# Let's check that.  
# 
# ### Process Test Data, Predict and Submit
# We will load @raddar dataset from [here][1] with discussion [here][2].
# 
# [1]: https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format
# [2]: https://www.kaggle.com/competitions/amex-default-prediction/discussion/328514

# In[10]:


# COMPUTE SIZE OF 4 PARTS FOR TEST DATA
NUM_PARTS = 4
TEST_PATH = '../input/amex-data-integer-dtypes-parquet-format/test.parquet'

print(f'Reading test data...')
test = read_file2cudf(path = TEST_PATH, usecols = ['customer_ID','S_2'])
customers = test[['customer_ID']].drop_duplicates().sort_index().values.flatten()
rows,num_cust = get_rows(customers, test[['customer_ID']], NUM_PARTS = NUM_PARTS, verbose = 'test')

# INFER TEST DATA IN PARTS
skip_rows = 0
skip_cust = 0
test_preds = []

for k in range(NUM_PARTS):
    
    # READ PART OF TEST DATA
    print(f'\nReading test data...')
    test = read_file2cudf(path = TEST_PATH)
    test = test.iloc[skip_rows:skip_rows+rows[k]]
    skip_rows += rows[k]
    print(f'=> Test part {k+1} has shape', test.shape)
    
    # PROCESS AND FEATURE ENGINEER PART OF TEST DATA
    test = feature_engineer(test)
    if k==NUM_PARTS-1: test = test.loc[customers[skip_cust:]]
    else: test = test.loc[customers[skip_cust:skip_cust+num_cust]]
    skip_cust += num_cust
    
    for j, cat_index in enumerate ([test[test.LT>0].index]):
        # XGB
        dtest = xgb.DMatrix(data=test.loc[cat_index,:][FEATURES])
        model = xgb.Booster()
        model.load_model(f'XGB_fold0_LT{j}.json')
        preds = model.predict(dtest, iteration_range=(0,model.best_ntree_limit))
        for f in range(1,FOLDS):
            model.load_model(f'XGB_fold{f}_LT{j}.json')
            preds += model.predict(dtest, iteration_range=(0,model.best_ntree_limit))
        del dtest, model
        _ = gc.collect()
        preds /= FOLDS
        # SAVE
        df =  test.loc[cat_index].reset_index()[["customer_ID"]].to_pandas()
        df['prediction'] = preds
        test_preds.append(df)
        del df, preds
        _ = gc.collect()

    # CLEAN MEMORY
    del test
    _ = gc.collect()


# In[11]:


# WRITE SUBMISSION FILE
test = cudf.DataFrame.from_pandas(pd.concat(test_preds,axis=0,ignore_index=True).set_index("customer_ID"))
sub = cudf.read_csv('../input/amex-default-prediction/sample_submission.csv')[['customer_ID']]
sub['customer_ID_hash'] = sub['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
sub = sub.set_index('customer_ID_hash')
sub = sub.merge(test[['prediction']], left_index=True, right_index=True, how='left')
sub = sub.reset_index(drop=True)

# DISPLAY PREDICTIONS
sub.to_csv(f'submission.csv',index=False)
print('Submission file shape is', sub.shape )


# ### 🏁 Conclusion for Hypothesis #4
# 
# Submission with the new features has a score **0.796** on Public LB, and that's more than **0.796** for baseline solution, based on 4th digit after point. Which is not shown ;-).   
# Hint - you can check that ranking despite LB rounding to 3 digits in Edit mode -> Competitions. It actually shows what notebook version had maximum Public LB score WITHOUT rounding to 3 digits under the hood (BEST SCORE vs. LATEST SCORE).
# 
# We've got a small improvement both on Local CV and Public LB, as result - for this train-test datasets **we can accept** Hypothesis #4: *Changes in statement dates for 13 month observation period has additional information correlated with default*   

# ### 🚀 Useful links with data and feature enrichment guides   
# 
# #### [Guide #1 How to improve accuracy of Kaggle TOP1 leaderboard notebook in 10 minutes](https://www.kaggle.com/code/romaupgini/how-to-find-external-data-for-1-private-lb-4-50)
# #### [Guide #2 Zero feature engineering with low-code libraries: Upgini + PyCaret](https://www.kaggle.com/code/romaupgini/zero-feature-engineering-with-upgini-pycaret)
# #### [Guide #3 How to improve accuracy of Multivariate Time Series kernel from external features & data](https://www.kaggle.com/code/romaupgini/guide-external-data-features-for-multivariatets)  
# 
# 
# #### Happy kaggling! 
# <sup>😔 Found error in the library or a bug in notebook code? Our bad! <a href="https://github.com/upgini/upgini/issues/new?assignees=&title=readme%2Fbug">
# Please report it here.</a></sup>
