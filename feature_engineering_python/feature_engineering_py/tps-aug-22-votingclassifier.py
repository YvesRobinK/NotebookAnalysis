#!/usr/bin/env python
# coding: utf-8

# ## TPS Aug 22 - VotingClassifier

# # References
# 
# [desalegngeb notebook](https://www.kaggle.com/code/desalegngeb/tps08-logisticregression-and-some-fe/notebook?scriptVersionId=102691691)

# In[1]:


get_ipython().run_cell_magic('capture', '', "!pip install feature_engine\n!pip install missingno\n# all may not be needed\n\nimport os\nimport sys\nimport numpy as np \nimport pandas as pd \nimport seaborn as sns\nimport matplotlib.pylab as plt\n\n\nfrom scipy.stats import uniform\nfrom sklearn.metrics import accuracy_score\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.naive_bayes import GaussianNB\nfrom sklearn.ensemble import VotingClassifier\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.naive_bayes import GaussianNB\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis\nfrom sklearn.ensemble import StackingClassifier,VotingClassifier,StackingClassifier\n\nfrom catboost import CatBoostClassifier\n\nfrom sklearn.model_selection import cross_validate, StratifiedKFold, RepeatedStratifiedKFold, RandomizedSearchCV, GridSearchCV\nfrom sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score, accuracy_score\nfrom sklearn import metrics\n\nfrom sklearn.preprocessing import LabelEncoder\nfrom sklearn.experimental import enable_iterative_imputer\nfrom sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer\nfrom sklearn import preprocessing\nfrom sklearn.preprocessing import OneHotEncoder, RobustScaler, PowerTransformer, LabelEncoder, StandardScaler, MinMaxScaler\n\nfor dirname, _, filenames in os.walk('/kaggle/input'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n        \nimport warnings\nwarnings.filterwarnings('ignore')\n")


# ## Feature Engineering

# In[2]:


train = pd.read_csv('../input/tabular-playground-series-aug-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-aug-2022/test.csv')
submission = pd.read_csv('../input/tabular-playground-series-aug-2022/sample_submission.csv')


# #### Null values
# - Around 3% of the data (cells) is missing in both train and test datset.
# - We will need to impute.

# In[3]:


train_na_cols = [col for col in train.columns if train[col].isnull().sum()!=0]
print('Train data cols with missing values ares: \n', train_na_cols)

print('\n')

test_na_cols = [col for col in test.columns if test[col].isnull().sum()!=0]
print('Train data cols with missing values ares: \n', test_na_cols)


# In[4]:


target = train.pop('failure')


# In[5]:


float_cols = [col for col in train.columns if train[col].dtypes == 'float64']
object_cols = [col for col in train.columns if train[col].dtypes == 'object']
int_object_cols = [col for col in train.columns[1:-1] if (train[col].dtypes == 'object' or train[col].dtypes == 'int64')]
nullValue_cols = [col for col in train.columns if train[col].isnull().sum()!=0]


# In[6]:


# Ambros' idea of adding missing values as extra columns
# https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/discussion/342319
train['m_3_missing'] = train.measurement_3.isna()
train['m_5_missing'] = train.measurement_5.isna()

test['m_3_missing'] = test.measurement_3.isna()
test['m_5_missing'] = test.measurement_5.isna()


# ### 3. Missing Value Imputation
# - Impute based on product code. We first group the data based on similar product code and impute the missing values using the same group data. This way of imputing missing values was discussed here in [Ambros' notebook](https://www.kaggle.com/code/ambrosm/tpsaug22-eda-which-makes-sense). 
# - One option we can use to impute the missing values is LGBMImputer. 

# In[7]:


# !rm -r kuma_utils
get_ipython().system('git clone https://github.com/analokmaus/kuma_utils.git')


# In[8]:


sys.path.append("kuma_utils/")
from kuma_utils.preprocessing.imputer import LGBMImputer


# In[9]:


display(train['product_code'].unique())
df_A = train[train['product_code']=='A']
df_B = train[train['product_code']=='B']
df_C = train[train['product_code']=='C']
df_D = train[train['product_code']=='D']
df_E = train[train['product_code']=='E']


# In[10]:


display(test['product_code'].unique())
df_F_t = test[test['product_code']=='F']
df_G_t = test[test['product_code']=='G']
df_H_t = test[test['product_code']=='H']
df_I_t = test[test['product_code']=='I']


# In[11]:


lgbm_imtr = LGBMImputer(cat_features=object_cols, n_iter=50)

# train dataset
train_iterimp_A = lgbm_imtr.fit_transform(df_A[nullValue_cols])
train_iterimp_B = lgbm_imtr.fit_transform(df_B[nullValue_cols])
train_iterimp_C = lgbm_imtr.fit_transform(df_C[nullValue_cols])
train_iterimp_D = lgbm_imtr.fit_transform(df_D[nullValue_cols])
train_iterimp_E = lgbm_imtr.fit_transform(df_E[nullValue_cols])

# test dataset
test_iterimp_F = lgbm_imtr.fit_transform(df_F_t[nullValue_cols])
test_iterimp_G = lgbm_imtr.fit_transform(df_G_t[nullValue_cols])
test_iterimp_H = lgbm_imtr.fit_transform(df_H_t[nullValue_cols])
test_iterimp_I = lgbm_imtr.fit_transform(df_I_t[nullValue_cols])


# In[12]:


none_na_cols = [col for col in train.columns if col not in nullValue_cols]
df_train = train[none_na_cols]
df_test = test[none_na_cols]

train_ = pd.concat([train_iterimp_A, train_iterimp_B,train_iterimp_C,train_iterimp_D,train_iterimp_E], axis=0)
train = pd.concat([df_train, train_], axis=1)

test_ = pd.concat([test_iterimp_F, test_iterimp_G,test_iterimp_H,test_iterimp_I], axis=0)
test = pd.concat([df_test, test_], axis=1)


# In[13]:


print("Missing values in train dataset after pre-peocessing is: ", format(train.isna().sum().sum()))


# ### 4. Feature Engineering Ideas
# - This dataset, being not too obscure what the columns maight be, gives a good opportunity to engineer new features.
# - From initial observations, some of the features seem to be `dimension` measurements and we may be able to combine and derive, area or volume features for example. `attribute_2` and `attribute_3` for example seem to be `width` and `length` dimensions?.
# - Looking at the values of `measurement_3` to `measurement_16`, they all look different variants of the same type of measurement. So we may aggregate into one or two features (mean, std) for example.

# In[14]:


display(train['attribute_2'].unique())
display(train['attribute_3'].unique())
print()
display(train['measurement_0'].unique())
display(train['measurement_1'].unique())
display(train['measurement_2'].unique())


# #### 4.1 Combine features and create new 
# - Multiply `attribute_2` and `attribute_2` and create a new feature
# - The idea here is that these two seem to me that they are dimensions of the material used for cleaning i.e, width and length for example. We could multiply and get area as new feature instead. Then drop the them.

# In[15]:


train['attribute_2*3'] = train['attribute_2'] * train['attribute_3']
test['attribute_2*3'] = test['attribute_2'] * test['attribute_3']


# <!-- train['meas17_loading_ratio'] = train['measurement_17']*train['attribute_2*3']
# test['meas17_loading_ratio'] = test['measurement_17']*test['attribute_2*3']
# 
# train['measurement_012_avg'] = np.mean(train['measurement_0']+train['measurement_1']+train['measurement_2'])
# test['measurement_012_avg'] = np.mean(test['measurement_0']+test['measurement_1']+test['measurement_2']) -->

# #### 4.2 Use aggregation
# - We see from the data that features `measuremsnt_3` to `measurement_16` belong to the same family. More like different variants of the same measurement type. So we could aggregate them into  their averages and may be standard deviations of them.

# In[16]:


meas_gr1_cols = [f"measurement_{i:d}" for i in list(range(3, 5)) + list(range(9, 17))]
train['meas_gr1_avg'] = np.mean(train[meas_gr1_cols], axis=1)
train['meas_gr1_std'] = np.std(train[meas_gr1_cols], axis=1)

test['meas_gr1_avg'] = np.mean(test[meas_gr1_cols], axis=1)
test['meas_gr1_std'] = np.std(test[meas_gr1_cols], axis=1) 

meas_gr2_cols = [f"measurement_{i:d}" for i in list(range(5, 9))]
train['meas_gr2_avg'] = np.mean(train[meas_gr2_cols], axis=1)
test['meas_gr2_avg'] = np.mean(test[meas_gr2_cols], axis=1)


# In[17]:


train['meas17/meas_gr2_avg'] = train['measurement_17'] / train['meas_gr2_avg']
test['meas17/meas_gr2_avg'] = test['measurement_17'] / test['meas_gr2_avg']


# ### 5.Using Weight of Evidence
# That is my only new idea for the notebook. It helps a bit. I would suggest some more feature engineering using both feature engine (https://feature-engine.readthedocs.io/en/latest/index.html) and feature tools (https://www.featuretools.com/)

# In[18]:


train.set_index('id',inplace=True)
test.set_index('id',inplace=True)


# In[19]:


combined_data = pd.concat([train,test],axis = 0).reset_index(drop=True)
def LableEncoder(df_org, df_comb, cats):
    #https://www.kaggle.com/code/pourchot/update-for-keras-optuna-for-lr/notebook
    for col in cats :
        le = LabelEncoder()
        df_comb[col] = le.fit_transform(df_comb[col])
    return train, test


# In[20]:


cats = ['attribute_0', 'attribute_1', 'm_3_missing', 'm_5_missing']
train, test = LableEncoder(train,combined_data, cats)


# <!-- # from sklearn.feature_selection import mutual_info_regression
# 
# # features = train.dtypes != object
# 
# # def make_mi_scores(train, y, discrete_features):
# #     mi_scores = mutual_info_regression(train, y, discrete_features=features)
# #     mi_scores = pd.Series(mi_scores, name="MI Scores", index=train.columns)
# #     mi_scores = mi_scores.sort_values(ascending=False)
# #     return mi_scores
# 
# # mi_scores = make_mi_scores(train, y, features)
# # mi_scores -->

# In[21]:


from feature_engine.encoding import WoEEncoder, RareLabelEncoder


# In[22]:


# here we could try some more features and some more targeting encoding, but I am focused on other competitions right now. I would suggest mean encoding and treatement of nulls in the test set
woe_encoder = WoEEncoder(variables=['attribute_0'])
                                


# In[23]:


X, y = train, target 

seed = 0
fold = 5


# In[24]:


X.drop(['product_code'],axis=1,inplace=True)
test.drop(['product_code'],axis=1,inplace=True)


# In[25]:


woe_encoder.fit(X, y)


# In[26]:


train_t = woe_encoder.transform(X)
test_t = woe_encoder.transform(test)


# In[27]:


cols_to_use = ['attribute_0','measurement_0', 'measurement_1', 'measurement_2', 'attribute_0','m_3_missing', 'm_5_missing',
               'meas_gr1_avg', 'meas_gr1_std', 'attribute_2*3', 'loading', 'measurement_17', 'meas17/meas_gr2_avg']
train =train_t[cols_to_use]
test =  test_t[cols_to_use]


# In[28]:


def score(X, y, model, cv):
    scoring = ["roc_auc"]
    scores = cross_validate(
        model, X, y, scoring=scoring, cv=cv, return_train_score=True,
    )
    scores = pd.DataFrame(scores).T
    return scores.assign(
        mean = lambda x: x.mean(axis=1),
        std = lambda x: x.std(axis=1),
    )

skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)


# <!-- model = LogisticRegression(tol = 1e-4, max_iter=500,random_state=seed)
# 
# search_space = dict(C=[0.0001, 0.01, 0.1, 1],
#                      penalty=['l2', 'l1'],
#                      solver= ['saga', 'liblinear', 'newton-cg'])
# 
# search = RandomizedSearchCV(model,
#                             search_space, 
#                             random_state=seed,
#                             cv = 5, 
#                             scoring='roc_auc')
# 
# rand_search = search.fit(X, y)
# 
# print('Best Hyperparameters: %s' % rand_search.best_params_)
# print("Best Estimator: \n{}\n".format(rand_search.best_estimator_))
# print("Best Score: \n{}\n".format(rand_search.best_score_)) -->

# ### 4. Models
# ### 4.1 Ensemble

# In[29]:


def get_models():
    models = list()
    models.append(('lr', LogisticRegression(max_iter=500, C=0.0001, penalty='l2', solver='newton-cg')))
    models.append(('bayes', GaussianNB(var_smoothing=0.5, priors=[len(y[y == 0]) / len(y), len(y[y == 1])/len(y)])))
    return models


# In[30]:


def evaluate_models(models, X_train, X_val, y_train, y_val):
    scores = list()
    for name, model in models:
        model.fit(X_train, y_train)
        yhat = model.predict(X_val)
        acc = accuracy_score(y_val, yhat)
        scores.append(acc)
    return scores


# In[31]:


X_train_full, X_test, y_train_full, y_test = train_test_split(train, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=1)


# In[32]:


models = get_models()
scores = evaluate_models(models, X_train, X_val, y_train, y_val)
print(scores)


# In[33]:


ensemble = VotingClassifier(estimators=models, voting='soft', weights=scores)
ensemble.fit(X_train_full, y_train_full)

yhat = ensemble.predict(X_test)


# In[34]:


score = accuracy_score(y_test, yhat)
print('Weighted Avg Accuracy: %.3f' % (score*100))


# ### 5. Submissions
# 

# In[35]:


sub = pd.DataFrame({'id': submission.id, 'failure': ensemble.predict_proba(test)[:,1]})
sub.to_csv("submission.csv", index=False)

