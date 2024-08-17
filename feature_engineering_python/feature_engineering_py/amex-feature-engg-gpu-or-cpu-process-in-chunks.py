#!/usr/bin/env python
# coding: utf-8

# # AMEX Feature Engineering with GPU *or* CPU! No memory overflow!
# 
# 
# Key improvements:
# 1. Process *both* train and test in chunks.
# 2. Handle GPU or CPU, without running out of memory or hitting cudf errors when switching!
# 
# Disclaimer: I ran into a lot of gotcha's while getting this to work on both CPU and GPU with same code. I can't 100% guarantee it works *correctly* even now, much less with your own customizations. Use at your own risk. Depending on your feature engineering, it is possible to run out of memory even with increasing the chunk size, however, I do expect it is much more likely to work with your own custom feature engineering code than most other popular starter notebooks.
# 
# 
# CUDF (GPU) code template and feature engineering acknowledgements:
# 1. https://www.kaggle.com/code/cdeotte/xgboost-starter-0-793
#     a. https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format
#     b. https://www.kaggle.com/competitions/amex-default-prediction/discussion/328514
#     c. https://www.kaggle.com/code/huseyincot/amex-catboost-0-793
#     d. https://www.kaggle.com/code/huseyincot/amex-agg-data-how-it-created
# 2. https://www.kaggle.com/code/jiweiliu/rapids-cudf-feature-engineering-xgb 
# 
# 
# Feature Engineering, besides in above notebooks:
# 1. Convert date-time to simple 0-12 value based on year and month. Ignore day. Add S_2_min and S_2_count to the feature set. Normalize S_2_min.
# 2. Don't fill NaN until *after* creating aggregation features. NaN will be ignored when calculating std, mean, etc.
# 3. Add delta columns, and match columns, comparing last row vs prior row.
# 4. Removed 6 engineered columns that, in batches, often only has single value for all rows.
# 5. Fill any nans in D_50 and S_23-based columns with -32783, to ensure lowest possible number.
# 6. Get total count of data per customer, and total count in last row.
# 7. Drop B_29. (I might end up submitting one submission with B_29 dropped, one with it included. https://www.kaggle.com/competitions/amex-default-prediction/discussion/328756 )
# 8. Calculate a simplified hull moving average over the 13 monthly statements for each customer and column.
# 
# 
# Todo:
# Plenty of things, some notables are:
# 1. Some form(s) of last-mean, last/mean, last-first, last/first, last/next to last (see the top two public notebooks as of July 9th). (Partially done in V2)
# 2. Round floats to 2 decimals, also from the top notebook (https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977)
# 3. Max-min (Done in V2)
# 4. Also try max/min, and try eliminating max and min as standalone features.
# 5. (Last-min) / (max-min): I have a really good feeling about this one. Might need to test compared with last-mean, though, they are similar, and might be better to just take one of them?
# 6. Experiment with some form of feature selection to reduce keep total columns to a a manageable or slightly larger size, but allow picking the 'best' from similar metrics, like last-mean vs last/mean, which one is better might vary by column. However, overall my philosophy is to err on the side of using more columns, relatively "unimportant" features still might just slightly help with ranking very similar customers. Want a programmatic way to try everything and only use the better results, though. 
# 
# 
# V2: 
# 1. Drop delta, min, max. Replace with:
# 2. CurrentLevel = (last-min) / (max-min)
# 3. Magnitude = max-min. Hopefully is 98% as much information as min and max separately, plus a slightly useful combined feature, so more total information for half the space. But minimal testing done, especially compared to inherent variance of CV score.
# 4. Last-mean.
# 
# V3: Fix CPU dtype for customer_ID

# # Load Libraries

# In[1]:


# LOAD LIBRARIES
import pandas as pd, numpy as np # CPU libraries
import gc, os

GPU = True
try:
    import cupy, cudf
except ImportError:
    GPU = False

if GPU:
    print('RAPIDS version',cudf.__version__)
else:
    print("Disabling cudf, using pandas instead")
    cudf = pd


# # Config

# In[2]:


PROCESS_TEST_DATA = True

# VERSION NAME FOR SAVED PARQUET FILES
VER = 111

# FILL NAN VALUE
NAN_VALUE = -127 # will fit in int8

if GPU:
    TRAIN_NUM_PARTS = 2
    TEST_SECTIONS = 2
    TEST_NUM_PARTS = 2
else:
    TRAIN_NUM_PARTS = 6
    TEST_SECTIONS = 2
    TEST_NUM_PARTS = 6

print("VER:", VER)
if not PROCESS_TEST_DATA:
    print("NOT processing test data!")


# # Pre-Processing and reading in chunks

# In[3]:


## Basic reading and formatting from initial raddar parquet file

def process_customer_columns(df):
    # REDUCE DTYPE FOR CUSTOMER AND DATE
    if GPU:
        df['customer_ID'] = df['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    else:
        df['customer_ID'] = df['customer_ID'].str[-16:].apply(int, base=16).astype('int64')
    year = cudf.to_numeric(df['S_2'].str[:4])
    month = cudf.to_numeric(df['S_2'].str[5:7])
    df['S_2'] = year.mul(12).add(month).sub(24207).astype('int8')
    return df

def read_file(path='', usecols=None):
    # LOAD DATAFRAME
    if usecols is not None:
        df = cudf.read_parquet(path, columns=usecols)
        df = process_customer_columns(df)
    else:
        df = cudf.read_parquet(path)

    print('Shape of data:', df.shape)
    
    return df


# In[4]:


## CALCULATE SIZE OF EACH SEPARATE PART
def get_rows(customers, df, NUM_PARTS=4, verbose=''):
    chunk = len(customers)//NUM_PARTS
    if verbose != '':
        print(f'We will process {verbose} data as {NUM_PARTS} separate parts.')
        print(f'There will be {chunk} customers in each part (except the last part).')
        print('Below are number of rows in each part:')
    rows = []

    for k in range(NUM_PARTS):
        if k==NUM_PARTS-1: cc = customers[k*chunk:]
        else: cc = customers[k*chunk:(k+1)*chunk]
        s = df.loc[df.customer_ID.isin(cc)].shape[0]
        rows.append(s)
    if verbose != '': print( rows )
    return rows,chunk

def getAndProcessDataInChunks(filename, is_train=False, NUM_PARTS=4, NUM_SECTIONS=1, split_k=0, verbose=''):
    gc.collect()

    print(f'Reading customer_IDs from {verbose} data...')
    df = read_file(path = filename, usecols = ['customer_ID','S_2'])
    customers = df[['customer_ID']].drop_duplicates().sort_index().values.flatten()
    rows,num_cust = get_rows(customers, df[['customer_ID']], NUM_PARTS=NUM_PARTS*NUM_SECTIONS, verbose=verbose)

    # INFER DATA IN PARTS
    skip_rows = 0
    skip_cust = 0
    allData = []

    del df
    gc.collect()

    print(f'\nReading {verbose} data...')
    df_file = read_file(path = filename)

    if is_train:
        assert(NUM_SECTIONS == 1) ## Splitting not implemented for target labels
        targets = cudf.read_csv('../input/amex-default-prediction/train_labels.csv')
        if GPU:
            targets['customer_ID'] = targets['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
        else:
            targets['customer_ID'] = targets['customer_ID'].str[-16:].apply(int, base=16).astype('int64')
        targets = targets.set_index('customer_ID')
        targets.target = targets.target.astype('int8')

    if NUM_SECTIONS > 1:
        startRow = 0
        for i in range(NUM_SECTIONS):
            if i == split_k:
                startRow = skip_rows
            for k in range(NUM_PARTS):
                skip_rows += rows[i*NUM_PARTS + k]
            if i == split_k:
                df_file = df_file.iloc[startRow:skip_rows].reset_index(drop=True)
                rows = rows[i*NUM_PARTS:(i+1)*NUM_PARTS]
                gc.collect()
                skip_rows = 0
                break
    for k in range(NUM_PARTS):
        # READ PART OF DATA
        df = df_file.iloc[skip_rows:skip_rows+rows[k]].reset_index(drop=True)
        skip_rows += rows[k]
        print(f'=> {verbose} part {k+1} has shape', df.shape )

        # PROCESS AND FEATURE ENGINEER PART OF DATA
        df = process_and_feature_engineer(df)

        if is_train:
            ## Relies on assumption that initial train data has customer IDs in same sorted order as train_labels.csv
            if k==NUM_PARTS-1: targetSlice = targets.iloc[skip_cust:]
            else: targetSlice = targets.iloc[skip_cust:skip_cust+num_cust]
            skip_cust += num_cust

            print("|...")
            df = cudf.concat([df, targetSlice], axis=1)
            print(" ...|")

        if GPU:
            print("|...")
            df = df.to_pandas()
            print(" ...|")

        allData.append(df)
        gc.collect()

    print(".", end='')
    del df_file
    gc.collect()
    allData = pd.concat(allData, axis=0)
    del df
    gc.collect()
    if is_train:
        print(".", end='')
        allData = allData.sort_index()
        gc.collect()
        print(".", end='')
        allData = allData.reset_index()
    print("|")
    return allData


# # Feature Engineering

# In[5]:


##
## Easy to replace or modify this function with your own custom feature engineering,
## and keep the other boilerplate untouched to allow switching from GPU to CPU based on GPU quota.
##
def process_and_feature_engineer(df):
    print(".", end = '')

    ## Save space on customer ID, and encode S_2 based on month and year as 0-12 for train set, 13-25 or 19-31 for test set.
    df = process_customer_columns(df)

    print(".", end = '')

    ## Consider dropping B_29: https://www.kaggle.com/competitions/amex-default-prediction/discussion/328756
    df.drop("B_29",inplace=True,axis=1)

    # compute "after pay" features
    # https://www.kaggle.com/code/jiweiliu/rapids-cudf-feature-engineering-xgb 
    for bcol in [f'B_{i}' for i in [11,14,17]]+['D_39','D_131']+[f'S_{i}' for i in [16,23]]:
        for pcol in ['P_2','P_3']:
            if bcol in df.columns:
                result = df[bcol] - df[pcol]
                df[f'{bcol}-{pcol}'] = result.fillna(0)

    # FEATURE ENGINEERING heavily modified, started from: 
    # https://www.kaggle.com/code/huseyincot/amex-agg-data-how-it-created
    all_cols = [c for c in list(df.columns) if c not in ['customer_ID']]
    cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
    num_features = [col for col in all_cols if col not in (cat_features + ["S_2"])]

    print(".", end = '')

    ## For each customer, count all NaN in any row, and count all NaN in the last row. Later we will add it as two columns
    df_nan = (df.mul(0) + 1).fillna(0)
    df_nan['customer_ID'] = df['customer_ID']
    nan_sum = df_nan.groupby("customer_ID").sum().sum(axis=1)
    nan_last = df_nan.groupby("customer_ID").last().sum(axis=1)
    del df_nan
    print(".", end = '')

    groups = df.groupby("customer_ID")
    test_num_agg = groups[num_features].agg(['mean', 'std', 'last'])
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]

    print("+", end = '')

    ## TODO: One-hot encode or convert to non-numeric? I believe raddar's clean dataset (and original data?) stores as numeric,
    ## and XGBoost doesn't(?) have a way to explicitly call out categorical columns
    test_cat_agg = groups[cat_features].last()
    test_cat_agg.columns = [x + "_last" for x in test_cat_agg.columns]

    print(".", end = '')

    ## S_2 test data has different range of values for S_2, normalize 'min' by subtracting max-12. Aka min = min + 12 - max.
    ## S_2 max will always be the same as last, and (after normalization), the same for every customer.
    ## Min tells us if they are a shorter term customer or not. Count under 13 tells us they're EITHER a shorter term customer OR they have some gap months.
    ## TODO: Min is more relevant (much higher default rate for short term customers than for gap customers), but might be worth encoding both into single column.
    ##   Basically, count + 143 minus 12*min (normal customer gets 156, long term gap customer gets 145-155, short term customer gets 0-143)
    ##   If not combining into one, could arguably use count+min instead of count, to directly highlight gap customers.
    test_s2_agg = groups[["S_2"]].agg(['min', 'count', 'max'])
    test_s2_agg.columns = ['_'.join(x) for x in test_s2_agg.columns]
    test_s2_agg['S_2_min'] = test_s2_agg['S_2_min'] + 12 - test_s2_agg['S_2_max']
    test_s2_agg.drop(['S_2_max'],inplace=True,axis=1)

    ## Quick sanity sorting check: confirm last value of each group is from the last (max) statement by checking S_2:
    assert_s2_check = groups["S_2"].last() - groups["S_2"].max()
    assert((assert_s2_check == 0).all())

    ## Drop delta for now. I haven't had success getting 1500+ features without running out of memory.
    ##   Out of memory: Not only while feature engineering (probably solvable), but again while running the model from pre-processed feature dataset (harder to solve)
    ## TODO: try swapping out other feature to add this one in?
    ##   Feature selection on subsets, then combine best features?
    ##   Reduce float64, int64, etc, to lower precision and fit more that way?
#     ### Add delta
#     test_num_agg2 = groups[num_features].nth(-1) - groups[num_features].nth(-2)
#     test_num_agg2 = test_num_agg2.fillna(0)
#     test_num_agg2.columns = [x + '_delta' for x in test_num_agg2.columns]

    ## Note for optimization: several of these are probably super inefficient to re-calculate it from the group data. Should be able to re-use test_num_agg in some way.

    ### Add current level: range from 0.0-1.0. For example: 0.0 means last = min; 0.5 means last = (max+min)/2; 1.0 means last = max.
    test_num_agg2 = (groups[num_features].last() - groups[num_features].min()) / (groups[num_features].max() - groups[num_features].min())
    test_num_agg2 = test_num_agg2.fillna(0)
    test_num_agg2.columns = [x + '_curLevel' for x in test_num_agg2.columns]

    ### Add magnitude: max - min
    test_num_agg3 = groups[num_features].max() - groups[num_features].min()
    test_num_agg3 = test_num_agg3.fillna(0)
    test_num_agg3.columns = [x + '_magnitude' for x in test_num_agg3.columns]

    ### Add last-mean
    test_num_agg4 = groups[num_features].last() - groups[num_features].mean()
    test_num_agg4 = test_num_agg4.fillna(0)
    test_num_agg4.columns = [x + '_last-mean' for x in test_num_agg4.columns]

    ### Add match for categorical: 1 if last and next to last are the same, 0 if not. If both are nan, or next to last is nan, treat as the same via fillna(1)
    ## TODO: in case it matters, move below forwards/backwards fill, just below.
    test_cat_agg2 = 1 + groups[cat_features].nth(-1) - groups[cat_features].nth(-2)
    test_cat_agg2 = test_cat_agg2.fillna(1)
    test_cat_agg2[test_cat_agg2 != 1] = 0
    test_cat_agg2.columns = [x + '_match' for x in test_cat_agg2.columns]

    print(".", end = '')

    ## Forward fill, and then backward fill to remove all nans before using "nth" to calc hma
    groups = groups.ffill()
    groups["customer_ID"] = df["customer_ID"]
    groups = groups.groupby("customer_ID").bfill()
    groups["customer_ID"] = df["customer_ID"]
    groups = groups.groupby("customer_ID")

    print(".", end = '')

    ### Add HMA. Hull moving average is a smoothed moving average sometimes used on time series data. (e.g. stock price).
    ## I happen to like it. As usual, I can't actually say whether it helps a lot, a little, or not at all.
    ##   Especially with the inherent CV variance, I haven't done nearly enough (any) experiments to see whether it helps a lot, a little, or not at all.
    ##   It didn't obviously help compared with other XGB public notebooks until I lowered column subsampling and learning rate.
    ##   With those hyper parameter changes helping a ton, I haven't checked which personal feature engg touches are actually contributing (if any).
    ## For calculation simplicity, this version of HMA is a bit simpler, while keeping the key idea that mean(range(10)) = 5, but hma(range(10)) = 10.
    ## TODO: Maybe consider calculating reverse HMA, for example last - reverse HMA, or HMA-reverseHMA to more directly look at slope of the data.
    ##
    ## Calculate the biggest down to the smallest, then take the biggest that isn't nan, to handle varying group sizes
    h13 = hma13(groups, num_features)
    h11 = hma11(groups, num_features)
    h9 = hma9(groups, num_features)
    h7 = hma7(groups, num_features)
    h5 = hma5(groups, num_features)
    h3 = hma3(groups, num_features)
    h1 = hma1(groups, num_features)
    hma_df = cudf.concat([h1, h3, h5, h7, h9, h11, h13], axis=0)
    del h1, h3, h5, h7, h9, h11, h13, assert_s2_check
    gc.collect()
    print(".", end = '')
    hma_df = hma_df.sort_index()
    hma_df = hma_df.reset_index()
    hma_df = hma_df.groupby("customer_ID").last()
    hma_df.columns = [x + "_hma" for x in hma_df.columns]

    print(".", end = '')

    df = cudf.concat([test_s2_agg, test_num_agg, test_cat_agg, test_num_agg2, test_cat_agg2, hma_df, test_num_agg3, test_num_agg4], axis=1)

    print(".")

    ## Finally add NaN counts from earlier
    df["total_data_count"] = nan_sum
    df["total_data_last"] = nan_last

    ## Per discussion I forgot to save the link, maybe on XGBoost Starter notebook, there's two columns with actual numbers often going below '-127', the default fillna.
    ## TODO is to handle some categories of nan differently anyways (per raddar's work), and agg often ignores them (which I think is a fine way) anyways.
    ##   The remaining nans: it's probably better in XGBoost to just leave them there, and let XGBoost decide!
    nan_col = ['D_50_mean', 'D_50_std', 'D_50_last', 'S_23_mean', 'S_23_std', 'S_23_last']
    df[nan_col] = df[nan_col].fillna(-32783)
    df = df.fillna(NAN_VALUE)

    print('shape after engineering', df.shape )
    for col in df.columns:
        if len(df[col].unique()) == 1:
            print("Consider dropping column!?", col)

    del test_s2_agg, test_num_agg, test_cat_agg
    del test_num_agg2, test_cat_agg2, test_num_agg3, test_num_agg4

    return df


def hma13(groups, columns):
    return (5/14)*groups[columns].nth(-1) + (27/91)*groups[columns].nth(-2) + (43/182)*groups[columns].nth(-3) + (16/91)*groups[columns].nth(-4) + (3/26)*groups[columns].nth(-5) + (5/91)*groups[columns].nth(-6) - (1/182)*groups[columns].nth(-7) - (6/91)*groups[columns].nth(-8) - (5/91)*groups[columns].nth(-9) - (4/91)*groups[columns].nth(-10) - (3/91)*groups[columns].nth(-11) - (2/91)*groups[columns].nth(-12) - (1/91)*groups[columns].nth(-13)
def hma11(groups, columns):
    return (17/42)*groups[columns].nth(-1) + (25/77)*groups[columns].nth(-2) + (113/462)*groups[columns].nth(-3) + (38/231)*groups[columns].nth(-4) + (13/154)*groups[columns].nth(-5) + (1/231)*groups[columns].nth(-6) - (5/66)*groups[columns].nth(-7) - (2/33)*groups[columns].nth(-8) - (1/22)*groups[columns].nth(-9) - (1/33)*groups[columns].nth(-10) - (1/66)*groups[columns].nth(-11)
def hma9(groups, columns):
    return (7/15)*groups[columns].nth(-1) + (16/45)*groups[columns].nth(-2) + (11/45)*groups[columns].nth(-3) + (2/15)*groups[columns].nth(-4) + (1/45)*groups[columns].nth(-5) - (4/45)*groups[columns].nth(-6) - (1/15)*groups[columns].nth(-7) - (2/45)*groups[columns].nth(-8) - (1/45)*groups[columns].nth(-9)
def hma7(groups, columns):
    return (11/20)*groups[columns].nth(-1) + (27/70)*groups[columns].nth(-2) + (31/140)*groups[columns].nth(-3) + (2/35)*groups[columns].nth(-4) - (3/28)*groups[columns].nth(-5) - (1/14)*groups[columns].nth(-6) - (1/28)*groups[columns].nth(-7)
def hma5(groups, columns):
    return (2/3)*groups[columns].nth(-1) + (2/5)*groups[columns].nth(-2) + (2/15)*groups[columns].nth(-3) - (2/15)*groups[columns].nth(-4) - (1/15)*groups[columns].nth(-5)
def hma3(groups, columns):
    return (5/6)*groups[columns].nth(-1) + (1/3)*groups[columns].nth(-2) - (1/6)*groups[columns].nth(-3)
def hma1(groups, columns):
    return groups[columns].nth(-1)


# # PROCESS TRAIN DATA

# In[6]:


TRAIN_PATH = '../input/amex-data-integer-dtypes-parquet-format/train.parquet'
train = getAndProcessDataInChunks(TRAIN_PATH, is_train=True, NUM_PARTS=TRAIN_NUM_PARTS, verbose='train')

print(train.shape)
print(train.head())
train.to_parquet(f'train_fe_v{VER}.parquet')
print("done")

## ~2 minutes GPU execution time with 1 part, versus
## ~2.5-3 minutes GPU execution time with 4 parts

## ~9 minutes CPU execution time with 4 parts


# # PROCESS TEST DATA

# In[7]:


if PROCESS_TEST_DATA:
    del train
    gc.collect()

    TEST_PATH = '../input/amex-data-integer-dtypes-parquet-format/test.parquet'
    for k in range(TEST_SECTIONS):
        test = getAndProcessDataInChunks(TEST_PATH, NUM_PARTS=TEST_NUM_PARTS, NUM_SECTIONS=TEST_SECTIONS, split_k=k, verbose='test')

        print(test.shape)
        print(test.head())
        test.to_parquet(f'test{k}_fe_v{VER}.parquet')
        print("done")

        del test
        gc.collect()

## ~4 minutes GPU: 2*1 parts
## ~5-6 minutes GPU: 2*4 parts

## ~20 minutes CPU: 2*4

