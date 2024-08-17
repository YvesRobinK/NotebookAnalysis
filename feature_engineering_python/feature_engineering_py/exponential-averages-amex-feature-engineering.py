#!/usr/bin/env python
# coding: utf-8

# # Exponential Averages: AMEX Feature Engineering
# 
# A split off of my other feature engineering notebook: https://www.kaggle.com/code/roberthatch/amex-feature-engg-gpu-or-cpu-process-in-chunks
# The key purpose of this notebook is to demonstate a fast (enough) method to create exponential moving averages (well, not actually moving) of every numeric column. This is a natural fit for this competition, since logically more recent data is more relevant, but capturing the signal from less recent data is both important and non-trivial.
# 
# Using these features here: https://www.kaggle.com/roberthatch/train-with-3000-features-xgb-pyramid
# 
# Todo:
# Since 'last' can and should be captured separately, consider creating the ema of all observations EXCEPT last? I created the weights (e.g. ema2_ignore_last) but didn't test it. Customers with single row would just get a bunch of nan, as well. 
# 

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
    test_num_agg = groups[num_features].agg(['mean', 'last', 'max'])
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

    ### Add match for categorical: 1 if last and next to last are the same, 0 if not. If both are nan, or next to last is nan, treat as the same via fillna(1)
    ## TODO: in case it matters, move below forwards/backwards fill, just below.
    test_cat_agg2 = 1 + groups[cat_features].nth(-1) - groups[cat_features].nth(-2)
    test_cat_agg2 = test_cat_agg2.fillna(1)
    test_cat_agg2[test_cat_agg2 != 1] = 0
    test_cat_agg2.columns = [x + '_match' for x in test_cat_agg2.columns]

    print(".", end = '')

    ## Add exponential moving average
    groups = df.groupby("customer_ID")

    print(".")
    e2 = ema(groups, num_features, ema2)
    print("+")
    e2.columns = [x + "_e2" for x in e2.columns]
    e3 = ema(groups, num_features, ema3)
    print("+")
    e3.columns = [x + "_e3" for x in e3.columns]
    e5 = ema(groups, num_features, ema5)
    print("+")
    e5.columns = [x + "_e5" for x in e5.columns]
    e7 = ema(groups, num_features, ema7)
    print("+")
    e7.columns = [x + "_e7" for x in e7.columns]
    e11 = ema(groups, num_features, ema11)
    print("+")
    e11.columns = [x + "_e11" for x in e11.columns]


    df = cudf.concat([test_s2_agg, test_num_agg, test_cat_agg, test_cat_agg2, e2, e3, e5, e7, e11], axis=1)

    print("*")

    ## Finally add NaN counts from earlier
    df["total_data_count"] = nan_sum
    df["total_data_last"] = nan_last

    ## Per discussion in cdeotte's XGBoost Starter notebook, there's two columns with actual numbers often going below '-127', the default fillna.
    ## TODO is to handle some categories of nan differently anyways (per raddar's work), and agg often ignores them (which I think is a fine way) anyways.
    ##   The remaining nans: it's might be better in XGBoost to just leave them there, and let XGBoost decide!
    nan_col = ['D_50_mean', 'D_50_last', 'S_23_mean', 'S_23_last']
    df[nan_col] = df[nan_col].fillna(-32783)
    df = df.fillna(NAN_VALUE)

    print('shape after engineering', df.shape )
    for col in df.columns:
        if len(df[col].unique()) == 1:
            print("Consider dropping column!?", col)

    del test_s2_agg, test_num_agg, test_cat_agg
    del test_cat_agg2
    del e2, e3, e5, e7, e11

    return df


# In[6]:


p2 = 1/3
p3 = 1/2
p5 = 2/3
p7 = 3/4
p11 = 5/6
ema2 = [1.0, p2**1, p2**2, p2**3, p2**4, p2**5, p2**6, p2**7, p2**8, p2**9, p2**10, p2**11, p2**12]
ema3 = [1.0, p3**1, p3**2, p3**3, p3**4, p3**5, p3**6, p3**7, p3**8, p3**9, p3**10, p3**11, p3**12]
ema5 = [1.0, p5**1, p5**2, p5**3, p5**4, p5**5, p5**6, p5**7, p5**8, p5**9, p5**10, p5**11, p5**12]
ema7 = [1.0, p7**1, p7**2, p7**3, p7**4, p7**5, p7**6, p7**7, p7**8, p7**9, p7**10, p7**11, p7**12]
ema11 = [1.0, p11**1, p11**2, p11**3, p11**4, p11**5, p11**6, p11**7, p11**8, p11**9, p11**10, p11**11, p11**12]
ema2_ignore_last = [0.0, p2**1, p2**2, p2**3, p2**4, p2**5, p2**6, p2**7, p2**8, p2**9, p2**10, p2**11, p2**12]
ema3_ignore_last = [0.0, p3**1, p3**2, p3**3, p3**4, p3**5, p3**6, p3**7, p3**8, p3**9, p3**10, p3**11, p3**12]
ema5_ignore_last = [0.0, p5**1, p5**2, p5**3, p5**4, p5**5, p5**6, p5**7, p5**8, p5**9, p5**10, p5**11, p5**12]
ema7_ignore_last = [0.0, p7**1, p7**2, p7**3, p7**4, p7**5, p7**6, p7**7, p7**8, p7**9, p7**10, p7**11, p7**12]
ema11_ignore_last = [0.0, p11**1, p11**2, p11**3, p11**4, p11**5, p11**6, p11**7, p11**8, p11**9, p11**10, p11**11, p11**12]


def ema(groups, columns, weights):
    x1 = groups[columns].nth(-1)
    x2 = groups[columns].nth(-2)
    x3 = groups[columns].nth(-3)
    x4 = groups[columns].nth(-4)
    x5 = groups[columns].nth(-5)
    x6 = groups[columns].nth(-6)
    x7 = groups[columns].nth(-7)
    x8 = groups[columns].nth(-8)
    x9 = groups[columns].nth(-9)
    x10 = groups[columns].nth(-10)
    x11 = groups[columns].nth(-11)
    x12 = groups[columns].nth(-12)
    x13 = groups[columns].nth(-13)
    w1 = x1.notna().astype('int8') * weights[0]
    w2 = x2.notna().astype('int8') * weights[1]
    w3 = x3.notna().astype('int8') * weights[2]
    w4 = x4.notna().astype('int8') * weights[3]
    w5 = x5.notna().astype('int8') * weights[4]
    w6 = x6.notna().astype('int8') * weights[5]
    w7 = x7.notna().astype('int8') * weights[6]
    w8 = x8.notna().astype('int8') * weights[7]
    w9 = x9.notna().astype('int8') * weights[8]
    w10 = x10.notna().astype('int8') * weights[9]
    w11 = x11.notna().astype('int8') * weights[10]
    w12 = x12.notna().astype('int8') * weights[11]
    w13 = x13.notna().astype('int8') * weights[12]
    x1 = x1 * w1
    x2 = x2 * w2
    x3 = x3 * w3
    x4 = x4 * w4
    x5 = x5 * w5
    x6 = x6 * w6
    x7 = x7 * w7
    x8 = x8 * w8
    x9 = x9 * w9
    x10 = x10 * w10
    x11 = x11 * w11
    x12 = x12 * w12
    x13 = x13 * w13

    x = x1.add(x2, fill_value=0).add(x3, fill_value=0).add(x4, fill_value=0).add(x5, fill_value=0).add(x6, fill_value=0).add(x7, fill_value=0).add(x8, fill_value=0).add(x9, fill_value=0).add(x10, fill_value=0).add(x11, fill_value=0).add(x12, fill_value=0).add(x13, fill_value=0)
    w = w1.add(w2, fill_value=0).add(w3, fill_value=0).add(w4, fill_value=0).add(w5, fill_value=0).add(w6, fill_value=0).add(w7, fill_value=0).add(w8, fill_value=0).add(w9, fill_value=0).add(w10, fill_value=0).add(w11, fill_value=0).add(w12, fill_value=0).add(w13, fill_value=0)
    x = x / w

    return(x)


# # PROCESS TRAIN DATA

# In[7]:


TRAIN_PATH = '../input/amex-data-integer-dtypes-parquet-format/train.parquet'
train = getAndProcessDataInChunks(TRAIN_PATH, is_train=True, NUM_PARTS=TRAIN_NUM_PARTS, verbose='train')

print(train.shape)
print(train.head())
train.to_parquet(f'train_fe_v{VER}.parquet')
print("done")


# # PROCESS TEST DATA

# In[8]:


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

