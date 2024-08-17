#!/usr/bin/env python
# coding: utf-8

# ### This kernel is forked from https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm . I have done some changes to get better score 

# In[ ]:


import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn2_circles
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import scipy
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold



sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

NFOLDS = 5
SEED = 42


# # Data Loading
# 

# In[ ]:


train = pd.read_csv('../input/avito-demand-prediction/train.csv')
test = pd.read_csv('../input/avito-demand-prediction/test.csv')
gp = pd.read_csv('../input/aggregated-features-lightgbm/aggregated_features.csv') 
train = train.merge(gp, on='user_id', how='left')
test = test.merge(gp, on='user_id', how='left')

agg_cols = list(gp.columns)[1:]

del gp
gc.collect()

train.head()


# # Feature Engineering
# ### Text cleaning does not help So, I am commenting them

# In[ ]:


# def cleanup(s):                      
#     """
#     function to clean text data
    
#     """
#     s = str(s)
#     s = s.lower()
# #     s = re.sub('\s\W',' ',s)
# #     s = re.sub("https\S+\w+","",s)
# #     s=[word if word not in ss else "" for word in TweetTokenizer().tokenize(s)]
# #     s = " ".join(s)
# #     s = re.sub('rt*.@\w+',' ',s)
# #     s = re.sub('@\w+',' ',s)
# #     s = re.sub('\W,\s',' ',s)
# #     s = re.sub(r'[^\w,]', ' ', s)
#     s = re.sub("\d+", "", s)
#     s = re.sub('\s+',' ',s)
#     s = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', s)
# #     s = s.replace(".co","")
# #     s = s.replace(",","")
# #     s = s.replace("[\w*"," ")
#     s = ''.join(''.join(a)[:2] for _, a in itertools.groupby(s))
#     return s


# In[ ]:


for df in [train, test]:
    df['description'].fillna('unknowndesc', inplace=True)
    df['title'].fillna('unknowntitle', inplace=True)

    df['weekday'] = pd.to_datetime(df['activation_date']).dt.day
    
    for col in ['description', 'title']:
        df['num_words_' + col] = df[col].apply(lambda comment: len(comment.split()))
        df['num_unique_words_' + col] = df[col].apply(lambda comment: len(set(w for w in comment.split())))

    df['words_vs_unique_title'] = df['num_unique_words_title'] / df['num_words_title'] * 100
    df['words_vs_unique_description'] = df['num_unique_words_description'] / df['num_words_description'] * 100
    
    df['city'] = df['region'] + '_' + df['city']
    df['num_desc_punct'] = df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    
    for col in agg_cols:
        df[col].fillna(-1, inplace=True)


# In[ ]:


count_vectorizer_title = CountVectorizer(stop_words=stopwords.words('russian'), lowercase=True, min_df=2)

title_counts = count_vectorizer_title.fit_transform(train['title'].append(test['title']))

train_title_counts = title_counts[:len(train)]
test_title_counts = title_counts[len(train):]


count_vectorizer_desc = TfidfVectorizer(stop_words=stopwords.words('russian'), 
                                        lowercase=True, ngram_range=(1, 2),
                                        max_features=17000)

desc_counts = count_vectorizer_desc.fit_transform(train['description'].append(test['description']))

train_desc_counts = desc_counts[:len(train)]
test_desc_counts = desc_counts[len(train):]

train_title_counts.shape, train_desc_counts.shape


# In[ ]:


target = 'deal_probability'
predictors = [
    'num_desc_punct', 
    'words_vs_unique_description', 'num_unique_words_description', 'num_unique_words_title', 'num_words_description', 'num_words_title',
    'avg_times_up_user', 'avg_days_up_user', 'n_user_items', 
    'price', 'item_seq_number'
]
categorical = [
    'image_top_1', 'param_1', 'param_2', 'param_3', 
    'city', 'region', 'category_name', 'parent_category_name', 'user_type'
]

predictors = predictors + categorical


# In[ ]:


for feature in categorical:
    print(f'Transforming {feature}...')
    encoder = LabelEncoder()
    train[feature].fillna('unknown',inplace=True)
    test[feature].fillna('unknown',inplace=True)
    encoder.fit(train[feature].append(test[feature]).astype(str))
    
    train[feature] = encoder.transform(train[feature].astype(str))
    test[feature] = encoder.transform(test[feature].astype(str))


# # Hyper Parameter Tuning
# 
# ### I did it on cloud so I m just commenting it out to save time

# In[ ]:


# def objective(space):
#     mod = lgb.LGBMRegressor(n_estimators = 5000, 
#             num_leaves = int(space['num_leaves']),
#             subsample = space['subsample'],min_child_weight = space['min_child_weight'],
#             colsample_bytree=space['colsample_bytree'],
#             learning_rate =space['learning_rate'],n_jobs=-1,
#                 )
# #     temp_train=copy.copy(newtrain)
#     folds=KFold(5,random_state=100)
#     fold_score=[]
#     i=1
#     st=time.time()
#     print('=================*=================')
#     print(space)
#     for train_index,test_index in folds.split(X=X):
#         mod.fit(X=X[train_index],y=y.values[train_index],eval_set=[ (X[test_index],y.values[test_index])],early_stopping_rounds=20,verbose=30,eval_metric='rmse')    
#         score=mod.best_score_.get('valid_0').get('rmse')
#         print('cv',i,': ', score)
#         i=i+1
#         fold_score.append(score)                
#     print("SCORE:") 
#     print(np.mean(fold_score))
#     print('time',time.time()-st)
#     return 1-np.mean(fold_score) 

# space ={
#     #'max_depth':hp.quniform('max_depth',2,10,1),
#     'num_leaves': hp.quniform('num_leaves', 200, 300, 4),
#     'min_child_weight': hp.quniform ('min_child_weight', 1, 2, 1),
#     'subsample': hp.quniform ('subsample', 0.8, .95,0.05),
#     'learning_rate': hp.quniform('learning_rate', 0.01,0.2,.03),
#    # A problem with max_depth casted to float instead of int with
#    # the hp.quniform method.
# #     'gamma': hp.quniform('gamma', 0, 0.6, 0.1),
#     'colsample_bytree': hp.quniform('colsample_bytree', 0.7, .95, 0.05),
#    }  
# trials = Trials()
# best = fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=80)
# print(best)


# In[ ]:


train["price"] = np.log(train["price"]+0.001)
train["price"].fillna(-999,inplace=True)
train["image_top_1"].fillna(-999,inplace=True)

test["price"] = np.log(test["price"]+0.001)
test["price"].fillna(-999,inplace=True)
test["image_top_1"].fillna(-999,inplace=True)


# # LightGBM 
# 

# In[ ]:


rounds = 20000
early_stop_rounds = 50
lgbm_params = {
    'objective' : 'regression',
    'metric' : 'rmse',
    'num_leaves' : 300,
#     'max_depth': 15,
    'learning_rate' : 0.021,
    'feature_fraction' : 0.6,
    'bagging_fraction' : .8,
    'verbosity' : -1
}

feature_names = np.hstack([
    count_vectorizer_desc.get_feature_names(),
    count_vectorizer_title.get_feature_names(),
    predictors
])
print('Number of features:', len(feature_names))


# In[ ]:


VALID = False


# In[ ]:


x_test = scipy.sparse.hstack([
    test_desc_counts,
    test_title_counts,
    test.loc[:, predictors]
], format='csr')

if VALID == True:
    train_index, valid_index = train_test_split(np.arange(len(train)), test_size=0.1, random_state=42)

    x_train = scipy.sparse.hstack([
            train_desc_counts[train_index],
            train_title_counts[train_index],
            train.loc[train_index, predictors]
    ], format='csr')
    y_train = train.loc[train_index, target]

    x_valid = scipy.sparse.hstack([
        train_desc_counts[valid_index],
        train_title_counts[valid_index],
        train.loc[valid_index, predictors]
    ], format='csr')
    y_valid = train.loc[valid_index, target]

     
    # LGBM Dataset Formatting 
    lgtrain = lgb.Dataset(x_train, y_train,
                    feature_name=list(feature_names),
                    categorical_feature = categorical)
    lgvalid = lgb.Dataset(x_valid, y_valid,
                    feature_name=list(feature_names),
                    categorical_feature = categorical)
     
    # Go Go Go
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=20000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=50,
        verbose_eval=100)
    print("Model Evaluation Stage")
    del x_valid ; x_train; gc.collect()

else:
    # LGBM Dataset Formatting 
    X = scipy.sparse.hstack([
        train_desc_counts,
        train_title_counts,
        train.loc[: , predictors]
    ], format='csr')
    y = train.deal_probability
    
    lgtrain = lgb.Dataset(X, y.values,
                    feature_name=list(feature_names),
                    categorical_feature = categorical)
     # Go Go Go
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=2200)
    
    del X; gc.collect()

del train,test
gc.collect()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 14))
lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")


# In[ ]:


subm = pd.read_csv('../input/avito-demand-prediction/sample_submission.csv')
subm['deal_probability'] = np.clip(lgb_clf.predict(x_test), 0, 1)
subm.to_csv('submission.csv', index=False)


# In[ ]:




