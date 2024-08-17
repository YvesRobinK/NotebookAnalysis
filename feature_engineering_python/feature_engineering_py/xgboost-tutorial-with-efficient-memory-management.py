#!/usr/bin/env python
# coding: utf-8

# > This kernel is a XGB Tutorial with managing the memory that added code-comments and detailed descriptions based on [XGBoost Starter - LB 0.793](https://www.kaggle.com/code/cdeotte/xgboost-starter-0-793).

# > TOC
# ```
# 1. Load Libraries
# 2. Load Dataset and Manage the GPU Memory
# 3. Feature Engineering
# 4. Train XGB
# 5. Save OOF Preds
# 6. Feature Importance
# 7. Data Processing and Feature Engineering for Test Data
# 8. Infer Test
# 9. Create Submission CSV
# ```

# # 1. Load Libraries

# When we train machine learning and deep learning, it is common to use the CPU to perform data preprocessing,etc. and to train the model with the GPU. And in this case, the process of copying(moving) the data loaded to the CPU to the GPU is required. For example, Working with dataframes with pandas, processing data on GPU memory with torch, and so on.
# 
# Instead of doing that now, RAPIDS came out with the concept of 'Yo, Let's do the whole process on the GPU!'. RAPIDS is a CUDA process-based data science platform built and operated by NVIDIA.
# 
# All packages are named cuxx to emphasize that they are based on CUDA. By replacing pandas with cudf, numpy with cupy, and sklearn with cuml, it has been developed to use almost the same functions as before.
# 

# In[1]:


import pandas as pd
import numpy as np

import cupy
import cudf

import matplotlib.pyplot as plt, gc, os

print('cudf version',cudf.__version__)


# # 2. Load Dataset and Manage the GPU Memory
# 
# When using a GPU, we need to manage our GPU resources well with monitoring currently available GPU resources through the nvidia-smi shell command, etc.

# In[2]:


get_ipython().system('nvidia-smi')


# One Tesla P100 is available and there are currently no processes running with that GPU.
# 
# Now we will load the data onto GPU memory with cudf. The basic functions and usage are the same as in pandas.

# In[3]:


# read_parquet() function reads a file in parquet format.
df = cudf.read_parquet('../input/amex-data-integer-dtypes-parquet-format/train.parquet')


# In[4]:


df


# In[5]:


get_ipython().system('nvidia-smi')


# Putting data on the GPU like this will take up 3669 MiB of memory in Memory-Usage. Given that the total available memory is 16280 MiB, it should be borne in mind that memory may be overwitten if the variable is copied several times or another dataset(eg, evaluation data) is loaded.

# So, if you want to cleanly convert `customer_ID` to a number, you can use the hex_to_int() and astype() functions.
# 
# Let's apply that only for the last 16 characters. That's the unique characters.

# In[6]:


df['customer_ID'] = df['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
df


# In[7]:


get_ipython().system('nvidia-smi')


# In order to get used to it, we will continue to track the memory as much as possible. Just truncating some of the data like this can reduce the memory being used.
# 
# About 300 MiB has been reduced.

# Column `S_2`contained time information. Like pandas, the data type is changed with the to_datetime() function.

# In[8]:


df['S_2'] = cudf.to_datetime(df['S_2'])
df


# In[9]:


get_ipython().system('nvidia-smi')


# It is also important to change it to an appropriate data type. In Python's memory operation process, string types use a lot of memory by default. So, if you can encode a categorical variable or change it to an integer type, it is advantageous for memory management.

# In[10]:


df.isna().sum()


# There are a lot of empty cells. We will use the fillna() function to fill empty cells with -127. The lowest number that can be expressed in 1 byte(8-bits) is -129. A signed integer can be represented from -128 to 127, and it seems that the original author wanted to replace null values with minimal memory usage.
# 
# However, since it is a processing that does not consider the distribution of each variable, there may be some disadvantages in model training.
# 
# And in the original text, when executing the fillna() function with cudf, it was written as `df = df.fillna(NAN_VALUE)`, but we will write it as `df.fillna(NAN_VALUE, inplace=True)`. When written as in the original code, the GPU memory usage nearly doubles (3321MiB -> 6079MiB) as the null-padded df is copied and allocated to a new memory address. Since we are not working on a sufficient memory environment, we must set the `inplace=Tru` option to overwrite the currently allocated memory address.

# In[11]:


df.info()


# In[12]:


NAN_VALUE = -127 
df.fillna(NAN_VALUE, inplace=True)
df.isna().sum()


# In[13]:


get_ipython().system('nvidia-smi')


# The read_file_GPU() function below performs this process at once.
# 
# In the original code, it was written only in such a way that GPU is used, but it is difficult to proceed with Kaggle servel. Therefore, we use the CPU in the process of loading the entire data, and in the process of learning or evaluating the data, So we will make another read_file_CPU() function.
# 
# And we do not have enough CPU on Kaggle server. So if data is uploaded at once in the process of loading the test dataset later, the CPU memory will be exceeded and the Kaggle kernel is reset. Our CPU based function can load the data to CPU in batches and then, load the data to GPU in iteration form. 
# 
# If you want to load data to the CPU, you can use pandas, and if you load data to the GPU,  you can use cudf.

# In[14]:


from pyarrow.parquet import ParquetFile


# In[15]:


# original code
def read_file_GPU(path = '', usecols = None):
    # read_parquet() function can read the parquet-type file.
    # if you want to specify columns:
    if usecols is not None: 
        df = cudf.read_parquet(path, columns=usecols)
    # if you want to read all columns:
    else: df = cudf.read_parquet(path)
    
    df['customer_ID'] = df['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    df.S_2 = cudf.to_datetime( df.S_2 )
    df = df.fillna(NAN_VALUE) 
    print('shape of data:', df.shape)
    
    return df

# modified code(CPU, batch load)
def read_file_CPU(path = '', iter_batch = None, usecols = None):
    if usecols is not None:
        # when retrieving only some columns(1~3), there is no problem even if all rows are retrieved.
        df = pd.read_parquet(path, columns=usecols)
    else:
        # When importing all columns, data is imported in batch format.
        df = iter_batch
    
    # it performs the same processing as it did with cudf.
    df['customer_ID'] = df['customer_ID'].apply(lambda x : int(x[-16:],16)).astype('int64') 
    df.S_2 = pd.to_datetime( df.S_2 )
    df.fillna(NAN_VALUE, inplace=True)
    print('shape of data:', df.shape)
    
    return df

# print('Reading train data...')
# TRAIN_PATH = '../input/amex-data-integer-dtypes-parquet-format/train.parquet'
# train = read_file(path = TRAIN_PATH)


# The original kernel used variable name as train, not df. So we'll make it the same.
# 
# However, the copy() function is not used here fore memory management. If you use the copy() function, it will be copied to another memory address.

# In[16]:


train = df
train.head()


# In[17]:


get_ipython().system('nvidia-smi')


# You can see that additional memory is not used because only the variable name is changed(refer to memory) without copying.

# # 3. Feature Engineering

# We do not put data as it is to the model, but transform it into statistical values before training and then train it.
# 
# In the process of data aggregation, the column takes on a multiindex format, but in order to include it in the model, a one-dimensional column must be maintained. So, Let's take a look at this task first and then run the whole process through a function.

# In[18]:


multi_index_col_sample = train.groupby('customer_ID')[['B_30','B_38','D_114']].agg(['count','last','nunique']).columns
multi_index_col_sample


# In[19]:


['_'.join(x) for x in multi_index_col_sample]


# We can make it easier to see by attaching the double index as a one-dimensional index. Then, Let's check the entire function.

# In[20]:


def process_and_feature_engineer(df):
    # Put the remaining column names int the all_cols variable except for the customer_ID column and S_2 column in a list comprehension method.
    # Columns in all_cols are the variables for training.
    all_cols = [c for c in list(df.columns) if c not in ['customer_ID','S_2']]
    
    # Let's split the all_cols categorical variables and numerical variables.
    # In the original code, categorical variables are splited like below but i find the others remained.
    # But to maintain the countinuity and avoid confusion, we will use it as is.
    cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
    num_features = [col for col in all_cols if col not in cat_features]

    # And aggregate numerical variables into statistics for each customer_ID.
    test_num_agg = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    # Aggregated columns are the type of MultiIndex. So connect them with '_' charactor like that we checked above.
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]

    # Similary, Aggregate categorical variables for each customer_ID.
    # 'count' is count the duplicate customer_ID's.
    # 'last' is get the last value of each categorical variables.
    # nunique() count the unique values of each categorical variables.
    test_cat_agg = df.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
    test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]

    # Merge the numerical variables and categorical variables.
    df = cudf.concat([test_num_agg, test_cat_agg], axis=1)
    del test_num_agg, test_cat_agg
    print('shape after engineering', df.shape)
    
    return df


train = process_and_feature_engineer(train)


# In[21]:


train


# In[22]:


get_ipython().system('nvidia-smi')


# As the number of columns increased, the data capacity also increased.

# We can load the target(label) data also. We will merge it into the train dataset created above.
# 
# At this time, customer_ID must be processed as an integer in the same way as before. 
# 
# This task is to predict whether the customer will or will not pay the card expenses. Each customer's real repayment status is contained in train_labels.csv.

# In[23]:


targets = cudf.read_csv('../input/amex-default-prediction/train_labels.csv')
targets['customer_ID'] = targets['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
targets = targets.set_index('customer_ID')

# The index is the same as customer_ID, Merge based on the corresponding index.
train = train.merge(targets, left_index=True, right_index=True, how='left')
# Store the target data as an 8-bit integer.
train.target = train.target.astype('int8')


# In[24]:


train = train.reset_index()
train


# In[25]:


get_ipython().system('nvidia-smi')


# The targets are now merged into the train dataset, so deallocate them in memory.
# 
# Although the capacity is small, it is better to always manage the memory after using the variable.

# In[26]:


del targets


# In[27]:


get_ipython().system('nvidia-smi')


# Count the number of features to train the model, The first column is the id value and the last column is the label. Excluding 2 columns, the remaining number of columns is 198.

# In[28]:


FEATURES = train.columns[1:-1]
print(f'There are {len(FEATURES)} features!')


# # 4. Train XGB
# 
# When training the model, we will utilize KFold cross-validation to avoid overfitting by using all data at lease once for traininig.
# 
# There is a problem in that the parts corresponding to 30 and 20 cannot be used for learning when the data is divided into learning/verification, such as 0:30 or 80:20 which are generally used for convenience. KFold splits training data and validation data(expressed as creating K-Folds) so that all datasets can be used for trainig.

# In[29]:


# LOAD XGB LIBRARY
from sklearn.model_selection import KFold
import xgboost as xgb
print('XGB Version',xgb.__version__)

# XGB MODEL PARAMETERS
xgb_parms = { 
    'max_depth':4, 
    'learning_rate':0.05, 
    'subsample':0.8,
    'colsample_bytree':0.6, 
    'eval_metric':'logloss',
    'objective':'binary:logistic',
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'random_state':42
}


# When traininig, we use DeviceQuantileDMatrix. All of usable-GPU memory is used at once for each calculation. And this DeviceQuantileDMatrix allows you to use GPU memory by dividing it into smaller units.
# 
# In order to use DeviceQuantileDMatrix, it is necessary to define a class that can pass data in batch form by repeatedly calling the iteration method, that is, the next() function.

# In[30]:


class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=256*1024):
        self.features = features
        self.target = target
        self.df = df
        # It will start from 0 and increase by 1 until all of data is passed.
        self.it = 0 
        self.batch_size = batch_size
        # np.ceil()은 a function that rounds up the float.
        # Calculate the number of batches by dividing the data by the batch size
        self.batches = int( np.ceil( len(df) / self.batch_size ) )
        super().__init__()

    def reset(self):
        '''Reset the iterator'''
        # If you need to perform iteration again from the beginning, 
        # you can initialize it with the reset() function.
        self.it = 0

    def next(self, input_data):
        '''Yield next batch of data.'''
        # self.batches defined at class instance creation. It contains the total number of batches that can be passed.
        # End the iteration when self.it has reached the number of deliverable batches.
        if self.it == self.batches:
            # 
            return 0 
        
        # Get the start-end point for indexing.
        a = self.it * self.batch_size
        b = min( (self.it + 1) * self.batch_size, len(self.df) )
        # Contain the data that will be passed by batch to the variable dt.
        dt = cudf.DataFrame(self.df.iloc[a:b])
        # Pass the feature and target from dt to input.
        input_data(data=dt[self.features], label=dt[self.target]) 
        self.it += 1
        return 1


# The function below is the logic to evaluate the model in the American Express - Default Prediction contest. It will be used as a criterion for optimization during model training.
# 
# In this tutorial, I expanded the original kernel that uses only CPUs for this calculation to select CPU and GPU. If you want to train the model using the train data on the GPU as it is, you must use the amex_metric_mod_GPU() function, which calculate optimization on the GPU. Here, we use amex_metric_mod_CPU().

# In[31]:


def amex_metric_mod_CPU(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

def amex_metric_mod_GPU(y_true, y_pred):

    labels     = cupy.transpose(cupy.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = cupy.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[cupy.cumsum(weights) <= int(0.04 * cupy.sum(weights))]
    top_four   = cupy.sum(cut_vals[:,0]) / cupy.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = cupy.transpose(cupy.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = cupy.where(labels[:,0]==0, 20, 1)
        weight_random  = cupy.cumsum(weight / cupy.sum(weight))
        total_pos      = cupy.sum(labels[:, 0] *  weight)
        cum_pos_found  = cupy.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = cupy.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)


# Finally, we train the model, To make efficient use of the limited GPU, we will load the dataset down to the CPU first. Then, in the IterLoadForDMatrix() function, indexing in small units through cudf and loading memory to the GPU goes through.
# 
# Since we have already loaded the training data into cudf and allocated it to the GPU memory, we will load it down to the CPU through the to_pandas() function. The to_pandas() function allows you to safely copy to CPU resources while transform cudf object into a pandas dataframe object.
# 
# In fact, if you use the RAPIDS platform, it is effective to perform all processes with a GPU, but there are many practical difficulties in applying it completely with a small amount of memory. Since we use Kaggle servers, it is important to make the best use of CPU and GPU memory like this.

# To 'copy' means to remain in GPU memory. Let's check the resource status before and after copying together.

# In[32]:


get_ipython().system('nvidia-smi')


# In[33]:


train_cpu = train.to_pandas()


# In[34]:


get_ipython().system('nvidia-smi')


# From now on, as multiple functions refer to multiple variables, and the allocated memory is called around, instantaneous memory usage will fluctuate. In Python, it is not necessary to manually manage memory because the garbage collector(which increases the available memory by deallocating the memory when the value allocated to a specific memory is not being referenced more than once) works internally.
# 
# However, since optimization is not performed in real time for every execution, it is very helpful for optimization by manually operating the garbage collector through the gc.collect() function for every batch during model training and evaluation. So, let's run the function and move on.

# In[35]:


import gc


# In[36]:


gc.collect()


# We will set The K of KFold to 5. Then, for each of the 5 folds, we will have 4 learning folds and 1 validatio fold, and repeat learning a total of 5 times by moving the position of the valiation folds. As a result, the optimization is carried out through the average value of the 5 verification results.
# 
# SEED can be specified with any number.

# In[37]:


FOLDS = 5
SEED = 42
skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)


# In[38]:


importances = []
oof = []
TRAIN_SUBSAMPLE = 1.0
VER = 1 # Version information to record when saving the model.

# To perform cross-validation on the KFold object skf, iterative validation is performed after separating into training and validation folds.
for fold,(train_idx, valid_idx) in enumerate(skf.split(train_cpu, train_cpu.target)):
    
    # If you want to train model by only using some sample of the train data, set TRAIN_SUBSAMLE to less than 1.
    # Then, you can configure the train set again by randomly extracting the ratio through the if statement below.
    if TRAIN_SUBSAMPLE<1.0:
        np.random.seed(SEED)
        train_idx = np.random.choice(train_idx, int(len(train_idx)*TRAIN_SUBSAMPLE), replace=False)
        np.random.seed(None)
    
    print('#'*25)
    print('### Fold',fold+1)
    print('### Train size',len(train_idx),'Valid size',len(valid_idx))
    print(f'### Training with {int(TRAIN_SUBSAMPLE*100)}% fold data...')
    print('#'*25)
    
    # Create the IterLoadForDMatrix instance. It will throw data in batches for training model. 
    Xy_train = IterLoadForDMatrix(train_cpu.loc[train_idx], FEATURES, 'target')
    
    # Define validation data to verify performance while training.
    X_valid = train_cpu.loc[valid_idx, FEATURES]
    y_valid = train_cpu.loc[valid_idx, 'target']
    
    dtrain = xgb.DeviceQuantileDMatrix(Xy_train, max_bin=256)
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid)
    
    # train the model
    model = xgb.train(xgb_parms, 
                dtrain=dtrain,
                evals=[(dtrain,'train'),(dvalid,'valid')],
                num_boost_round=9999,
                early_stopping_rounds=100,
                verbose_eval=100) 
    # During cross-validation, the model is saved at each verification.
    model.save_model(f'XGB_v{VER}_fold{fold}.xgb')
    
    # After training the model, we will check the feature importance. 
    # For this, importance is calculated for each training and stored in a variable dd.
    dd = model.get_score(importance_type='weight')
    df = pd.DataFrame({'feature':dd.keys(),f'importance_{fold}':dd.values()})
    importances.append(df)
    
    # Validate the model. 
    oof_preds = model.predict(dvalid)
    
    # For calculating Accuracy, we uses the competition evaluation metric.
    # In the original code, y_valied.values is put as it is,
    # If not train_cpu but train that loaded in GPU memory was used as an argument,
    # y_valid.values would be cupy._core.core.ndarray rather than np.ndarray.
    # In this case, use cupy to let it compute directly on the GPU.
    # -> acc = amex_metric_mod_GPU(y_valid.values, oof_preds)

    # Since we use train_cpu loaded to the cpu as in the original code,
    # The variable is np.ndarray(). So, we can just put it in amex_metric_mod_CPU().
    acc = amex_metric_mod_CPU(y_valid.values, oof_preds)
    print('Kaggle Metric =',acc,'\n')
    
    # Also save the verification score(oof_pred) separately.
    df = train_cpu.loc[valid_idx, ['customer_ID','target']].copy()
    df['oof_pred'] = oof_preds
    oof.append( df )
    
    # Let's free All variables used for training from memory.
    del dtrain, Xy_train, dd, df
    del X_valid, y_valid, dvalid, model
    # And excecute garbage collection to completely remove the remaining non-referenced memory values even after the variable has been removed.
    _ = gc.collect()
    
print('#'*25)
# When training is finished, the entire verification result is saved as a DataFrame.
# Then, calculate the actual value and the verification value as an evaluation metric.
# Here, if you use the train variable loaded in GPU memory as an argument like y_valid,
# It would be not a pandas.core.frame.DataFrame, but a cudf.core.dataframe.DataFrame.
# So, In this case, either merge them with cudf or replace all elements in oof with pandas's DataFrame objects.
# Since we used train_cpu, we use pandas as is.
oof = pd.concat(oof,axis=0,ignore_index=True).set_index('customer_ID')
acc = amex_metric_mod_GPU(oof.target.values, oof.oof_pred.values)
print('OVERALL CV Kaggle Metric =',acc)


# In[39]:


# Now that training is over, the train_cpu dataset is no longer needed.
# We will also clean up the memory referring to the train_cpu data. 
del train, train_cpu
_ = gc.collect()


# In[40]:


get_ipython().system('nvidia-smi')


# # 5. Save OOF Preds

# We have to save the prediction result with customer_ID. Just get the unique id information from the data file, change the hexadecimal number to an integer type as we did before, and merge the predicted values obtained during the training process.

# In[41]:


TRAIN_PATH = '../input/amex-data-integer-dtypes-parquet-format/train.parquet'
oof_xgb = pd.read_parquet(TRAIN_PATH, columns=['customer_ID']).drop_duplicates()
oof_xgb['customer_ID_hash'] = oof_xgb['customer_ID'].apply(lambda x: int(x[-16:],16) ).astype('int64')
oof_xgb = oof_xgb.set_index('customer_ID_hash')
oof_xgb = oof_xgb.merge(oof, left_index=True, right_index=True)
oof_xgb = oof_xgb.sort_index().reset_index(drop=True)
oof_xgb.to_csv(f'oof_xgb_v{VER}.csv',index=False)
oof_xgb.head()


# Visualize the prediction results. The prediction result is a probability between 0 and 1. Since only 5% default(1) and the rest should be predicted as 0, the visualization is biased towards 0 and 1 in both directions, but there should be more values distributed at 0.

# In[42]:


plt.hist(oof_xgb.oof_pred.values, bins=100)
plt.title('OOF Predictions')
plt.show()


# Now that we have the forecasts saved as a csv file, we remove the variables. The reason for repeating this process of saving to a file and freeing memory is that the RAM supported by Kaggle is not that large. In general, even in the local environment, RAM is not enough, so it may shut down while referencing or copying the memory. 
# 
# So, to prevent this, it is recommended to save files to the hard-disk and keep the RAM lightly.

# In[43]:


del oof_xgb, oof
_ = gc.collect()


# In[44]:


get_ipython().system('nvidia-smi')


# # 6. Feature Importance

# Feature Importance is information about which variable is highly utilized in the model's task(prediction. here). 
# 
# After traininig the model, looking at this, if there are variables that are not important to the prediction, they will be removed, and the variables with excessive importance will go through a feedback process such as checking causality or correlation with the prediction target.

# In[45]:


import matplotlib.pyplot as plt

# While performing cross-validation, importances must have been accumulated as mush as the number of FOLDs(5).
# So, Calculate the average importance per feature by merging into the df variable.
df = importances[0].copy()
for k in range(1,FOLDS):
    df = df.merge(importances[k], on='feature', how='left')
df['importance'] = df.iloc[:,1:].mean(axis=1)
df = df.sort_values('importance',ascending=False)
df.to_csv(f'xgb_feature_importance_v{VER}.csv',index=False)


# In[46]:


df


# Visualize only the top 20 features by importance with a bar chart.

# In[47]:


NUM_FEATURES = 20
plt.figure(figsize=(10,5*NUM_FEATURES//10))
plt.barh(np.arange(NUM_FEATURES,0,-1), df.importance.values[:NUM_FEATURES])
plt.yticks(np.arange(NUM_FEATURES,0,-1), df.feature.values[:NUM_FEATURES])
plt.title(f'XGB Feature Importance - Top {NUM_FEATURES}')
plt.show()


# # 7. Data Processing and Feature Engineering for Test Data

# We wonder if a particular customer will repay the expense. So, let's get the unique customer_ID first.

# In[48]:


TEST_PATH = '../input/amex-data-integer-dtypes-parquet-format/test.parquet'
test = read_file_CPU(path = TEST_PATH, usecols = ['customer_ID','S_2'])
test


# In[49]:


customers = test[['customer_ID']].drop_duplicates().sort_index().values.flatten()
customers


# In this way, ID information is secured, and the default is predicted for each ID. Now we will load the test dataset. For now we needs only 2 columns, but like the train data, full dataset is very large. Therefore, we will load the dataset to the CPU in batch form as many as PART's, then pass each PART to the GPU so that the model can make predictions.
# 
# To do this, a criterion for dividing the PART is required. And the criterion requires two things: the row size in the test dataset before the duplicate customer_ID is removed, and the constant chunk size that after removing the duplicate.
# 
# Do you remember the process_and_feature_engineer() function we created earlier? If the function passes, the duplicate of customer_ID is removed through group_by aggregation and it is replaced with statistical values. This is a method of performing prediction by passing the transformed dataset in chunk size.
# 
# So, let's create a function for this process and execute that.

# In[50]:


def get_rows(customers, test, NUM_PARTS = 10, verbose = ''):
    # One chunk(size of PART) can be obtained by dividing the size of the entire test-dataset by the number of PARTs.
    # It is similar to finding the batch size when training the model.
    chunk = len(customers)//NUM_PARTS
    if verbose != '':
        print(f'We will process {verbose} data as {NUM_PARTS} separate parts.')
        print(f'There will be {chunk} customers in each part (except the last part).')
        print('Below are number of rows in each part:')
    rows = []

    for k in range(NUM_PARTS):
        # Keep the remain to cc if this PART is the last one.
        if k==NUM_PARTS-1: 
            cc = customers[k*chunk:]
        # If not the last PART, cut it in chunks from the front and put it in cc.
        else: 
            cc = customers[k*chunk:(k+1)*chunk]
        # Calculate the PART size by finding the number of customer_IDs included in the current PART.
        s = test.loc[test.customer_ID.isin(cc)].shape[0]
        # rows contain the size of 10 PARTs.
        rows.append(s)
    if verbose != '': print( rows )
    return rows, chunk



# Now, with the function created above, we will divide customer_ID into a total of 10 groups(PARTs)

# In[51]:


# In the original code, the test dataset was divided into four.
# But in case of using a kaggle server, you will get a GPU memory overflow error when you specify 4r, so we will divide it into 10.
NUM_PARTS = 10
rows,num_cust = get_rows(customers, test[['customer_ID']], NUM_PARTS = NUM_PARTS, verbose = 'test')


# On the last line, the size of each PART is printed. The size referred to here is the size including duplicates from the raw test dataset, not the chunk size.
# 
# Therefore, it should be equal to the size of the test dataset when added together.

# In[52]:


sum(rows) == len(test)


# and chunk size is,

# In[53]:


num_cust


# Remove the loaded test data and deallocate memory.

# In[54]:


del test
_ = gc.collect()


# In[55]:


get_ipython().system('nvidia-smi')


# # 8. Infer Test

# Through the code below, we divide the data into 10 PARTs through iteration, and then process the data for modeling and predict default using our model. The specific process is as follows.
# 
# 1. Devide the original test dataset into batches so that each PART can be included and load it to the CPU memory.
# 2. The PART is loaded to the GPU for calculation again,
# 3. Using the process_and_feature_engineering() function, convert raw data to statistical dataset for prediction.
# 4. Then, the shape of dataset will be different. At this time, indexing can be performed with the chunk size obtained above.
# 5. Index by chunk size and then predict default with the model.
# 6. When prediction for all chunks is finished, merge and return the prediction result for the entire customer_ID.

# In[56]:


skip_rows = 0
skip_cust = 0
test_preds = []

# Create an iteration object. By calling the object, you can raise as many rows as PART units to the CPU memory.
TEST_PATH = '../input/amex-data-integer-dtypes-parquet-format/test.parquet'
batch = ParquetFile(TEST_PATH)
# Since batch_size is fixed at one time call, it is not possible to get a different PARt size each time.
# Because of this, the customer_ID that should be predicted in the current batch is possible to already be loaded in the previous batch.
# So, set the batch size to the largest PART size, and merge it with the previous PART so that all customer_IDs can be indexed.
prev_batch = pd.DataFrame()
for pres_batch, k in zip(batch.iter_batches(batch_size=max(rows)), range(NUM_PARTS)): 
    print(f'\nReading test data...')
    # Merge the dataset loaded from the previous batch and the current batch.
    iter_batch = pd.concat([prev_batch, pres_batch.to_pandas()])
    test_cpu = read_file_CPU(iter_batch = iter_batch, path = TEST_PATH)
    # Then, Overwitten the previous batch with the current(present) batch.
    # And clear the memory.
    prev_batch = pres_batch.to_pandas()
    del pres_batch, iter_batch
    _ = gc.collect()
    
    # Load the PART into the GPU memory for pre-processing.
    test_gpu = cudf.DataFrame(test_cpu)
    skip_rows += rows[k]
    print(f'=> Test part {k+1} has shape', test_gpu.shape)
    
    # With preprocessing, statistics are obtained for each customer_ID, and duplicate customer_IDs are removed.
    # process_and_feature_engineer() uses cudf internally, not pandas.
    # So, when preprocessing, the data must be placed on the GPU.
    test_gpu = process_and_feature_engineer(test_gpu)
    
    # num_cust is the chunk size for de-duplicated customer_ID.
    # We will do the credit default prediction for all customer_IDs by increasing the chunk size.
    if k==NUM_PARTS-1: 
        test_gpu = test_gpu.loc[customers[skip_cust:]]
    else: 
        test_gpu = test_gpu.loc[customers[skip_cust:skip_cust+num_cust]]
    skip_cust += num_cust
    print('shape after indexing(by chunk size)', test_gpu.shape)
    
    # Pass the features excluding the label from the test data to X_test.
    X_test = test_gpu[FEATURES]
    dtest = xgb.DMatrix(data=X_test)
    del X_test
    gc.collect()

    # With our trained model, we can predict default.
    # Our model was trained by cross-validation.
    # Prediction is perfromed in the same way, and the prediction result can be obtained as the average value of all FOLD results.
    model = xgb.Booster()
    model.load_model(f'XGB_v{VER}_fold0.xgb')
    preds = model.predict(dtest)
    for f in range(1,FOLDS):
        model.load_model(f'XGB_v{VER}_fold{f}.xgb')
        preds += model.predict(dtest)
    preds /= FOLDS
    test_preds.append(preds)

    # free the memory
    del dtest, model
    _ = gc.collect()


# # 9. Create Submission CSV
# 
# Finally, create a file for submission and visualize the test results.

# In[57]:


len(test_preds)


# In[58]:


# File for submission can be saved according to the format of the competition guide.
test_preds = np.concatenate(test_preds)
test = cudf.DataFrame(index=customers,data={'prediction':test_preds})
sub = cudf.read_csv('../input/amex-default-prediction/sample_submission.csv')[['customer_ID']]
sub['customer_ID_hash'] = sub['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
sub = sub.set_index('customer_ID_hash')
sub = sub.merge(test[['prediction']], left_index=True, right_index=True, how='left')
sub = sub.reset_index(drop=True)

sub.to_csv(f'submission_xgb_v{VER}.csv',index=False)
print('Submission file shape is', sub.shape )
sub.head()


# In[59]:


# visualize with hist plot.
plt.hist(sub.to_pandas().prediction, bins=100)
plt.title('Test Predictions')
plt.show()

