#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering + XGB
# 
# The main idea of this notebook is to build a simple pipeline using custom features on the summary text, and XGB regressor for each target column.

# In[1]:


import re
import string
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold


# # Load in relevant files

# First, let's see which files we need to read

# In[2]:


get_ipython().system('ls -l /kaggle/input/commonlit-evaluate-student-summaries')


# In[3]:


prompts_test = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/prompts_test.csv")
prompts_train = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/prompts_train.csv")
summaries_train = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/summaries_train.csv")
summaries_test = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/summaries_test.csv")
sample_submission = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/sample_submission.csv")


# We are going to merge prompt & summaries dataframes for convenience

# In[4]:


df_test = prompts_test.merge(summaries_test, on='prompt_id')
df_train = prompts_train.merge(summaries_train, on='prompt_id')


# In[5]:


df_train.head()


# In[6]:


df_test.head()


# In[7]:


sample_submission.head()


# # Feature Engineering
# 
# Create new features, using https://www.kaggle.com/code/sercanyesiloz/commonlit-tf-idf-xgb-baseline#4.-Feature-Engineering
# - ```Text length```
# - ```Word count```
# - ```Stopword count```
# - ```Punctuation count```
# - ```Number counts```

# In[8]:


def count_total_words(text: str) -> int:
    words = text.split()
    total_words = len(words)
    return total_words

def count_stopwords(text: str) -> int:
    stopword_list = set(stopwords.words('english'))
    words = text.split()
    stopwords_count = sum(1 for word in words if word.lower() in stopword_list)
    return stopwords_count

def count_punctuation(text: str) -> int:
    punctuation_set = set(string.punctuation)
    punctuation_count = sum(1 for char in text if char in punctuation_set)
    return punctuation_count

def count_numbers(text: str) -> int:
    numbers = re.findall(r'\d+', text)
    numbers_count = len(numbers)
    return numbers_count

def feature_engineer(dataframe: pd.DataFrame, feature: str = 'text') -> pd.DataFrame:
    dataframe[f'{feature}_length'] = dataframe[feature].apply(lambda x: len(x))
    dataframe[f'{feature}_word_cnt'] = dataframe[feature].apply(lambda x: count_total_words(x))
    dataframe[f'{feature}_stopword_cnt'] = dataframe[feature].apply(lambda x: count_stopwords(x))
    dataframe[f'{feature}_punct_cnt'] = dataframe[feature].apply(lambda x: count_punctuation(x))
    dataframe[f'{feature}_number_cnt'] = dataframe[feature].apply(lambda x: count_numbers(x))
    return dataframe


# In[9]:


df_train = feature_engineer(df_train)
df_test = feature_engineer(df_test)


# In[10]:


df_train.head()


# In[11]:


df_test.head()


# # Training
# 

# Define function to calculate ```RMSE``` metric.

# In[12]:


def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Define feature & target columns.

# In[13]:


FEATURES = ['text_length', 'text_word_cnt', 'text_stopword_cnt', 'text_punct_cnt', 'text_number_cnt']
TARGETS = ['content', 'wording']


# Split the training data into folds, using ```GroupKFold```. The groups are determined using the ```prompt_id``` column.

# In[14]:


FOLDS = 4

gkf = GroupKFold(n_splits=FOLDS)
groups = df_train["prompt_id"]


# Train ```XGBRegressor``` for every target column.

# In[15]:


models = {target: [] for target in TARGETS}
val_true = {target: [] for target in TARGETS}
val_pred = {target: [] for target in TARGETS}

for fold, (train_ids, val_ids) in enumerate(gkf.split(df_train, groups=groups)):
    print("=" * 50)
    print(f"FOLD {fold}")
    
    train_results = []
    val_results = []
    
    for target in TARGETS:
        X_train = df_train.loc[train_ids, FEATURES]
        y_train = df_train.loc[train_ids, target]
        X_val = df_train.loc[val_ids, FEATURES]
        y_val = df_train.loc[val_ids, target]

        model = XGBRegressor(max_depth=1).fit(X_train, y_train)
        
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        
        rmse_train = compute_rmse(y_train, pred_train)
        rmse_val = compute_rmse(y_val, pred_val)
        
        print(f"  {target}: Training RMSE {rmse_train:.3f}, Validation RMSE {rmse_val:.3f}")
        
        models[target].append(model)
        train_results.append(rmse_train)
        val_results.append(rmse_val)
        val_true[target].extend(y_val)
        val_pred[target].extend(pred_val)
        
    mcrmse_train = np.mean(train_results)
    mcrmse_val = np.mean(val_results)
    print(f"  Training MCRMSE: {mcrmse_train:.3f}, Validation MCRMSE: {mcrmse_val:.3f}")

print("=" * 50)
print("CV Results")
rmses = []
for target in TARGETS:
    rmse = compute_rmse(val_true[target], val_pred[target])
    print(f"{target} RMSE: {rmse:.3f}")
    rmses.append(rmse)
print(f"MCRMSE: {np.mean(rmses):.3f}")
print("=" * 50)


# # Calculate predictions

# Predict test targets using trained models.

# In[16]:


for target in TARGETS:
    for fold, model in enumerate(models[target]):
        X_test = df_test[FEATURES]
        df_test[f'{target}_pred_{fold}'] = model.predict(X_test)


# # Submission

# Ensemble the model predictions.

# In[17]:


submission = df_test[['student_id']].sort_values('student_id').reset_index(drop=True)

for target in TARGETS:
    submission[target] = np.mean(df_test[[f'{target}_pred_{fold}' for fold in range(FOLDS)]], axis=1)
    
submission


# In[18]:


submission.to_csv('submission.csv', index=False)

