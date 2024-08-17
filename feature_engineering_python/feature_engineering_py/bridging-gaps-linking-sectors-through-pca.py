#!/usr/bin/env python
# coding: utf-8

# # 🎉 Building on Foundations: Incorporating PCA into Stock Analysis 📈
# 
# Greetings, fellow Kagglers! 🙌
# 
# As we embark on this analytical odyssey, we aim to dig deeper into the enthralling world of stocks. In our prior exploration, ["Sectors & Industries Fiesta! 🎉: Unraveling with PCA Magic 🪄"](https://www.kaggle.com/code/verracodeguacas/sectors-industries-fiesta-pca-magic?scriptVersionId=147363787), we unlocked the potent synergies of sectors, industries, and the mesmerizing dance of PCA.
# 
# Why is this exploration paramount, you might ask? Consider the electrifying atmosphere of the close auction at NASDAQ. The quest to determine those pivotal prices often rests on discerning the subtle movements and correlations among different stocks. The capability to access and evaluate 🕵️‍♂️ data from correlated stocks during these crucial moments becomes a game-changer. Detecting imbalances and swiftly connecting the dots can make all the difference in your predictions.
# 
# One formidable approach to capture this interwoven tapestry of stock relationships is through synthetically conceived factors. Today, our spotlight will shine brightly on **4 principal factors**. Emanating from our diligent PCA analysis, these factors shall be our north stars ⭐, guiding us through the labyrinthine corridors of stock predictions for the close auction.
# 
# Strap in, as we set forth on this exhilarating analytical journey! 🚀
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
from itertools import combinations
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from warnings import simplefilter
import joblib
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

N_Folds = 4


# In[2]:


import os
os.system('mkdir lgb-modelos-con-sectores-y-pca')


# # 📊 Stock Order Book Analysis: Diving Deeper into Bid & Ask Sizes 📉📈
# 
# In this segment, we're diving deeper into the order book, focusing specifically on the bid and ask sizes.
# 
# **What we'll analyze:**
# 
# - **Median Sizes**: A central value giving us a balanced figure of the bid and ask sizes across stocks.
#   
# - **Standard Deviation**: The spread or dispersion of the bid and ask sizes, indicating volatility.
#   
# - **Max & Min Sizes**: The peaks and valleys! These figures show us the highest and lowest bid and ask sizes observed.
#   
# - **Mean Sizes**: The average, a collective representation of bid and ask sizes.
#   
# - **First & Last Sizes**: Representing the starting and ending bid and ask sizes for each stock. It's always interesting to see how things kick off and conclude!
# 
# I'm leaving the chinese comments in there because I think they look cool!
# 
# Lastly, we ensure our dataset's integrity by dropping any rows with missing `target` values, ensuring our analysis is built on a solid foundation!
# 

# In[3]:


train = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')

#整体特征
median_sizes = train.groupby('stock_id')['bid_size'].median() + train.groupby('stock_id')['ask_size'].median()
std_sizes = train.groupby('stock_id')['bid_size'].std() + train.groupby('stock_id')['ask_size'].std()
max_sizes = train.groupby('stock_id')['bid_size'].max() + train.groupby('stock_id')['ask_size'].max()
min_sizes = train.groupby('stock_id')['bid_size'].min() + train.groupby('stock_id')['ask_size'].min()
mean_sizes = train.groupby('stock_id')['bid_size'].mean() + train.groupby('stock_id')['ask_size'].mean()
first_sizes = train.groupby('stock_id')['bid_size'].first() + train.groupby('stock_id')['ask_size'].first()
last_sizes = train.groupby('stock_id')['bid_size'].last() + train.groupby('stock_id')['ask_size'].last()
#可以再做日期的（好像没看到drop掉日期列）

train = train.dropna(subset=['target'])


# In[4]:


# Load PCA Loadings
pca_loadings = pd.read_csv('/kaggle/input/sectors-industries-fiesta-pca-magic/principal_components.csv')


# # PCA-weighted Average Prices Calculation
# 
# The following steps outline the procedure to compute PCA-weighted average prices for our stocks using principal component analysis (PCA) loadings:
# 
# 1. **Data Preparation**:
#     - Pivot the `train` dataframe to create a matrix of `wap` values.
#     - Align the order of `stock_id` in the pivoted table with the PCA loadings dataframe.
#     - Handle missing values by substituting NaNs with zeros.
# 
# 2. **PCA-weighted WAP Calculation**:
#     - Initialize a new dataframe to capture the PCA-weighted average prices.
#     - Compute these values utilizing the first four principal components.
# 
# 3. **Integration**:
#     - Reintegrate the calculated PCA-weighted average prices back into the primary `train` dataframe.
# 

# In[5]:


import pandas as pd
import numpy as np

# Assuming train and pca_loadings_df have been loaded

# Create a pivot table for wap
price_pivot = train.pivot_table(index=['date_id', 'seconds_in_bucket'], columns='stock_id', values='wap')

# Ensure the ordering of stock_id in price_pivot and pca_loadings is consistent
ordered_columns = price_pivot.columns
pca_loadings = pca_loadings.set_index('stock_id').loc[ordered_columns].reset_index()

# Handle NaN values (replace with 0 for this example)
price_pivot.fillna(0, inplace=True)
pca_loadings.fillna(0, inplace=True)

# Initialize a dataframe to hold the PCA_WAP values
pca_wap_df = pd.DataFrame(index=price_pivot.index)

# Compute 4 WAPs using the PCA loadings
for i in range(1, 5): # For first 4 PCs
    # Use .values to get numpy array multiplication (which is element-wise) 
    pca_wap_df[f'PCA_WAP_{i}'] = (price_pivot.values * pca_loadings.set_index('stock_id')[f'PC{i}'].values).sum(axis=1)

# Resetting index for merging purposes
pca_wap_df = pca_wap_df.reset_index()

# Merging the PCA_WAP columns with the train dataset
train = train.merge(pca_wap_df, on=['date_id', 'seconds_in_bucket'], how='left')


# In[6]:


train.tail()


# # Feature Engineering on Stock Data
# 
# The `feature_eng` function takes a dataframe `df` and returns the dataframe augmented with various engineered features. Below are the steps and the reasoning behind each:
# 
# 1. **Initial Data Preparation**:
#     - Filter out unnecessary columns ('row_id', 'date_id', 'time_id').
#     
# 2. **Ratio Calculations**:
#     - `imbalance_ratio`: The ratio of the imbalance size to the matched size.
#     - `bid_ask_volume_diff`: The difference between ask and bid sizes.
#     - `bid_plus_ask_sizes`: The sum of ask and bid sizes.
#     - `mid_price`: The average of ask and bid prices.
# 
# 3. **Stock Statistics Retrieval**:
#     - The function incorporates median, standard deviation, max, min, mean, first, and last sizes for each stock ID. These statistics give insights into the trading volume's distribution characteristics for each stock.
#     - `high_volume`: A binary feature indicating if the total bid and ask sizes exceed the median size for that stock.
# 
# 4. **Price Interactions**:
#     - For each pair of prices, two new features are created: 
#         - The difference between the two prices.
#         - Their imbalance, calculated as the difference over their sum.
#     - For every combination of three prices, a new feature is created capturing the imbalance between the highest and middle price over the difference of middle and the lowest price.
#     
# 5. **Imbalance Buy-Sell Flag Encoding**:
#     - One-hot encode the `imbalance_buy_sell_flag` column, which gets transformed into three separate columns representing sell-side imbalance, no imbalance, and buy-side imbalance.
# 
# 6. **Memory Management**:
#     - To ensure that the function does not consume excessive memory, especially when working with large dataframes, a call to `gc.collect()` is made. This triggers garbage collection, which frees up unused memory.
#     
# The function returns the dataframe with all the newly engineered features.
# 

# In[7]:


def feature_eng(df):
    cols = [c for c in df.columns if c not in ['row_id', 'date_id','time_id']]
    df = df[cols]
    
    #匹配失败数量和匹配成功数量的比率
    df['imbalance_ratio'] = df['imbalance_size'] / df['matched_size']
    #供需市场的差额
    df['bid_ask_volume_diff'] = df['ask_size'] - df['bid_size']
    #供需市场总和
    df['bid_plus_ask_sizes'] = df['bid_size'] + df['ask_size']
    
    #供需价格的均值
    df['mid_price'] = (df['ask_price'] + df['bid_price']) / 2
    
    #整体数据情况
    df['median_size'] = df['stock_id'].map(median_sizes.to_dict())
    df['std_size'] = df['stock_id'].map(std_sizes.to_dict())
    df['max_size'] = df['stock_id'].map(max_sizes.to_dict())
    df['min_size'] = df['stock_id'].map(min_sizes.to_dict())
    df['mean_size'] = df['stock_id'].map(mean_sizes.to_dict())
    df['first_size'] = df['stock_id'].map(first_sizes.to_dict())    
    df['last_size'] = df['stock_id'].map(last_sizes.to_dict())       
    
    #整体市场规模和当前的市场规模比较
    df['high_volume'] = np.where(df['bid_plus_ask_sizes'] > df['median_size'], 1, 0)
    
    prices = ['reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
    
    #价格之间做差，做差/求和
    for c in combinations(prices, 2):
        df[f'{c[0]}_minus_{c[1]}'] = (df[f'{c[0]}'] - df[f'{c[1]}']).astype(np.float32)
        df[f'{c[0]}_{c[1]}_imb'] = df.eval(f'({c[0]} - {c[1]})/({c[0]} + {c[1]})')
        
    for c in combinations(prices, 3):
        max_ = df[list(c)].max(axis=1)
        min_ = df[list(c)].min(axis=1)
        mid_ = df[list(c)].sum(axis=1) - min_ - max_
        
        df[f'{c[0]}_{c[1]}_{c[2]}_imb2'] = (max_-mid_)/(mid_-min_)
    
    #另外的特征
    #df['less_5min'] = df['seconds_in_bucket'].apply(lambda x: 1 if x < 300 else 0)
    #df['5min_8min'] = df['seconds_in_bucket'].apply(lambda x: 1 if 300 <= x else 0)
    #df['more_8min'] = df['seconds_in_bucket'].apply(lambda x: 1 if 480 <= x else 0)
        
    df_encoded = pd.get_dummies(df['imbalance_buy_sell_flag'])
    df_encoded = df_encoded.rename(columns={-1: 'sell-side imbalance', 0: 'no imbalance', 1: 'buy-side imbalance'})

    df = pd.concat([df, df_encoded], axis=1)
    
    gc.collect()
    
    return df


# # Stock Clustering and Sector Data Incorporation
# 
# The following steps detail the process of incorporating sector data (based on clustering) into our main training dataset:
# 
# 1. **Data Loading**:
#     - Read the `stock_clusters.csv` file which contains the clusters (or sectors) assigned to each stock. This file was previously generated using PCA and clustering in the "Fiesta" notebook.
#     - Remove the column named 'Unnamed: 0' which seems to be an unnecessary index column.
# 
# 2. **Cluster Distribution**:
#     - Using the `Counter` function from the `collections` module, we get a distribution of the number of stocks in each cluster (or sector). This provides an understanding of how many stocks belong to each sector.
# 
# The resulting `cluster_df` dataframe can be merged with the main `train` dataframe on the appropriate keys (likely `stock_id`) to incorporate sector information into our dataset.
# 

# In[8]:


from collections import Counter
cluster_df = pd.read_csv('/kaggle/input/sectors-industries-fiesta-pca-magic/stock_clusters.csv')
cluster_df = cluster_df.drop(columns=['Unnamed: 0'])
Counter(cluster_df.cluster)


# In[9]:


train.tail()


# In[10]:


# Assuming clusters_df is the dataframe containing stock_id and its associated cluster
train = train.merge(cluster_df, on='stock_id', how='left')

# Now, ensure that you have the stock_clusters column available and that it matches in length and order with your training set
assert 'cluster' in train.columns, "Cluster column is missing after merging!"


# In[11]:


industry_sectors = train['cluster'].values
train.drop(columns = "cluster", inplace = True)
train.tail()


# # Model Training using Stratified K-Fold Strategy with Sector Groupings
# 
# This cell encapsulates the end-to-end process of data preprocessing and model training. The focus here is on training a LightGBM regressor using a stratified K-Fold strategy based on the sectors.
# 
# 1. **Feature Engineering**:
#     - Using the `feature_eng` function, we process our main `train` dataframe to generate a set of engineered features stored in `X`. The target variable (`target`) is stored separately in `y`.
# 
# 2. **Parameters Initialization**:
#     - Setting up hyperparameters for the LightGBM model. These hyperparameters include learning rate, maximum depth, number of estimators, and regularization parameters.
# 
# 3. **Stratified K-Fold Strategy**:
#     - The core idea here is to ensure that each fold has a good representation of each industry sector, which is particularly crucial if different sectors have different data distributions.
#     - This strategy helps in capturing the underlying patterns in different sectors, offering a more generalized model. While the sector division was statistically built (meaning they might not directly correlate to real-world sectors), having this stratification ensures that our training process respects the structure found in the data.
#     - A stratified K-Fold is initialized, splitting the dataset into folds based on the `industry_sectors` variable.
# 
# 4. **Model Training and Validation**:
#     - For each fold, a LightGBM model is trained using the training set and is validated using the validation set.
#     - The trained model is saved to a file for potential future use.
#     - Predictions are made for the validation set, and the Mean Absolute Error (MAE) is computed between the predicted and actual values.
# 
# 5. **Performance Evaluation**:
#     - After training on all folds, the average MAE over all folds is calculated and printed. This gives an overall idea of the model's performance across different subsets (folds) of the dataset.
# 
# This K-Fold strategy, combined with the feature engineering steps, aims to provide a robust model that is well-tuned to the patterns present in the data while ensuring minimal overfitting to any specific sector or subset.
# 

# In[12]:


from sklearn.model_selection import GroupKFold

y = train['target'].values
X = feature_eng(train.drop(columns='target'))

print(X.columns)
# X.columns = X.columns.str.replace(' ', '_').str.replace('-', '_')
# X.columns = X.columns.str.replace('[^a-zA-Z0-9]', '_')
# print(X.columns)

y_min = np.min(y)
y_max = np.max(y)

params = {
    'learning_rate': 0.018,
    'max_depth': 9,
    'n_estimators': 600,
    'num_leaves': 440,
    'objective': 'mae',
    'random_state': 42,
    'reg_alpha': 0.01,
    'reg_lambda': 0.01,
    #'device': 'gpu'
}

# Using StratifiedKFold based on these bins
skf = StratifiedKFold(n_splits = N_Folds, shuffle = True, random_state = 42)
# kf = GroupKFold(n_splits = N_Folds)
# kf = KFold(n_splits = N_Folds, shuffle = True, random_state = 42)

mae_scores = []

#for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
#for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y, groups = groups)):
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, industry_sectors)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    m = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=50, early_stopping_rounds=50)
    print(f"Fold {fold+1} Trainning finished.")

    model_filename = f"/kaggle/working/lgb-modelos-con-sectores-y-pca/model_fold_{fold+1}.pkl"
    joblib.dump(m, model_filename)
    y_pred_valid = m.predict(X_valid)

    y_pred_valid = np.nan_to_num(y_pred_valid)
    y_valid = np.nan_to_num(y_valid)
    mae = mean_absolute_error(y_valid, y_pred_valid)
    mae_scores.append(mae)

# 计算4折平均的MAE
average_mae = np.mean(mae_scores)
print(f"4 fold MAE: {average_mae}")


# In[ ]:




