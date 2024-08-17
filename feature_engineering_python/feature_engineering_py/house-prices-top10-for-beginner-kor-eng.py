#!/usr/bin/env python
# coding: utf-8

# **This is written in Korean with reference to this link. Reference was made to MUHAMMAD RAAFAT's NoteBook.**
# - **Link** : https://www.kaggle.com/adamml/how-to-be-in-top-10-for-beginner
# 
# **전에 Titanic을 작성하고나서 많은 격려주셔서 감사합니다.Data Science를 입문하고자 하는 분과 한국어로 케글에 입문하는 사람을 위해 케글을 작성해보도록 하겠습니다.앞으로 더 많은 글과 제가 실력이 발전할수록 더 난이도있는 내용을 작성해보도록하겠습니다. 전에 분류 관련하여 작성하여 요번에는 회귀관련하여 beginner들이 입문하기에 괜찮은 난이도를 찾아서 번역해보려고 합니다. 오타 및 수정해야할 부분 있으면 comment 남겨주시면 정말 감사하겠습니다.추가적으로 upvote도 해주시면 감사하겠습니다.**
# 
# **Kaggle Korea에 많은 정보가 자주 올라오니 많은 관심 부탁드립니다.**
# 
# - **Kaggle Korea** : https://www.facebook.com/groups/KaggleKoreaOpenGroup/
# - **GitHub** : https://github.com/kalelpark/Kaggle_for_Korean

# # 목차
# - <strong>1. 데이터 수집하기 (Gathering Data)</strong>
# - <strong>2. 데이터에 대한 분석 및 중요한 Feature 이해하기 (Analysis the target and understand what is the important features)</strong>
# - <strong>3. 누락된 데이터 관찰 (Looking for missing values)</strong>
# - <strong>4. 데이터 전처리 (Feature Engineering)</strong>
# - <strong>5. 범주형 데이터 수치형 데이터로 전환하기 (Converting categorical to numerical)</strong>
# - <strong>7. 모델 생성 (Modeling)</strong>

# In[ ]:


# 데이터 수집 및 전처리
import numpy as np
import pandas as pd
from scipy.stats import norm, skew
import sklearn.metrics as metrics
from sklearn import preprocessing
# 모델
from xgboost import XGBRegressor
# 데이터 시각화
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # <strong>1. 데이터 수집하기(Gathering Data)</strong>

# In[ ]:


train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_df.shape, test_df.shape


# In[ ]:


train_df.head()


# - info : 데이터에 대한 전반적인 정보를 나타냅니다.
# 
# **<strong>info를 통하여 feature에 누락된 데이터가 있음을 확인할 수 있습니다. 이러한 feature들을 추후에 다루겠습니다.</strong>**

# In[ ]:


train_df.info()


# # <strong>2. Target 데이터에 관하여 더 분석 및 이해하기(Analysis the target and understand what is the important features)</strong>
# - Target인 SalePrice를 확인해보겠습니다.

# In[ ]:


print(train_df['SalePrice'].describe())


# In[ ]:


sns.distplot(train_df['SalePrice'])


# <strong><h4>보다시피, Target Data의 분포가,왼쪽으로 치우쳐저 있음을 알 수 있습니다.</h4></strong>
# <br>
# 
# - <strong>선형회귀 분석을 사용할 때에는 3가지 조건이 존재합니다.</strong>
#     - <strong>독립변수(X)값에 해당하는 종속변수(y)값들은 정규분포를 이뤄야 하고 모든 정규분포의 분산은 동일해야 합니다. (이를 위해서는 왜도(Skewness), 첨도(Kurtosis)를 파악해야 합니다.)</strong>
#     
#     - <strong>종속변수값들은 통계적으로 서로 독립적이여 합니다.</strong>
#     
#     - <strong>다중회귀분석의 경우, 독립변수끼리 다중공선성(multicollinearity)이 존재하지 않아야 합니다.</strong>
#     
#         - <strong>다중공선성에 관하여 간단하게 에시를 들어보겠습니다.</strong>
#         
#             - <strong>나이와 학년은 비슷한 변수인데, 거의 같은 개념(교집합)이라고 볼수 있습니다. 나이와 학년이 같이 변수로 존재하는 경우 다중공선성이라고 합니다.</strong>

# In[ ]:


print("왜도(Skewness) : %f " % train_df['SalePrice'].skew())
print("첨도(Kurtosis) : %f " % train_df['SalePrice'].kurt())


# <strong>Target Data를 정규화하겠습니다.</strong>

# In[ ]:


train_df['SalePrice'] = np.log1p(train_df['SalePrice'])
sns.distplot(train_df['SalePrice'], fit = norm)


# ### <strong>Target Data인 집 매매 값에 영향을 미치는 중요한 Feature에 대하여 알아보겠습니다.</strong>
# - HeatMap을 사용하여, 다양한 정보를 열 분포 형태의 그래픅으로 출력합니다. Heatmap을 사용하면 두 개의 카테고리 값에 대한 값 변화를 한눈에 알아보기 쉬워집니다.

# In[ ]:


cormat = train_df.corr()
plt.figure(figsize = (20 , 20))
sns.heatmap(cormat, vmax = .8, square = True)


# #### <strong>Feature들의 상관관계를 파악해보았습니다.</strong>
# -  1에 가까울 수록 경우 Feature간에 양의 상관관계, -1에 가까울수록 Feature간에 음의 상관관계 입니다.
# 
# <strong>저희에게 필요한 것은 SalePrice와 가장 상관관계가 높은 Feature가 필요합니다.</strong>

# In[ ]:


corr = train_df.corr()
# SalePrice Feauture 와의 상관관계가 0.5이상인 index만 추출하는 의미입니다.
highest_corr_features = corr.index[abs(corr['SalePrice']) > 0.5]
plt.figure(figsize = (10, 10))
# annot 을 사용하면, Heatmap 위에 숫자를 표기합니다.
g = sns.heatmap(train_df[highest_corr_features].corr(), annot = True, cmap = "RdYlGn")


# # <strong>위의 내용을 한번 정리하고 넘어가겠습니다.</strong>
# 
# - <strong>OverQual Feature이 Saleprice 와 상관관계가 가장 높다는 것을 알 수 있습니다.(0.82)</strong>
# 
# - <strong>GarageCars & GarageArea 는 각각 다른 Feature임을 알 수 있습니다. (0.88)</strong>
# 
# - <strong>TotalBsmtSF & 1stFlrSF 또한 각각 다른 Feature임을 알 수 있습니다. (0.82)</strong>
# 
#     - <strong>TotalBsmtSF & 1stFlrSF 를 유지하거나 TotalBsmtSF에 1stFlrSF를 추가할 수 있습니다.</strong>
#         - <strong>1stFlrSF에 추가하는 것이 아니라 TotalBsmtSF에 추가하는 이유는 SalePrice와의 상관관계가 약간 더 높기 때문입니다.</strong>
#              
#              
# - <strong>TotRmsAbvGrd & GrLivArea 또한 매우 상관관계가 높음을 알 수 있습니다.(0.83)</strong>
#     - <strong>GrLivArea 가 TotRmsAbvGrd 보다 SalePrice와 상관관계가 유지시키겠습니다.</strong>
#     
# 
# 

# In[ ]:


corr["SalePrice"].sort_values(ascending=False)


# ## <strong>SalePrice와 상관관계가 높은 순서대로 Pair Plot을 이용하여 시각화 해보겠습니다.</strong>

# In[ ]:


cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_df[cols])


# # <strong>3. 누락된 데이터 관찰하기(Looking for missing values)</strong>
# - <strong>누락된 데이터를 파악하기 전에 해야할 사항</strong>
#     - <strong>train data 와 test data를 결합하여, 데이터를 전처리를 한 후에 다시 나누도록 하겠습니다. 왜냐하면 결합하여 데이터를 전처리하는 것이 훨씬 편리하기 때문입니다.</strong>

# In[ ]:


Y_train = train_df['SalePrice']
test_id = test_df['Id']
all_data = pd.concat([train_df, test_df], axis = 0, sort = False)
all_data = all_data.drop(['Id', 'SalePrice'], axis = 1)


# In[ ]:


Total = all_data.isnull().sum().sort_values(ascending = False)
percent = (all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([Total, percent], axis = 1, keys = ['Total', 'Percent'])
missing_data.head(25)


# <strong>누락(결측값)이 많은 Feature라는 것을 보면 중요한 기능이 아니며, 어느것도 Heatmap을 참고했을 때, 어느것도 상관관계가 0.5이상을 갖지 않으므로, 삭제해도 data를 놓치지 않을 것입니다.</strong>

# In[ ]:


all_data.drop((missing_data[missing_data['Total'] > 5]).index, axis = 1, inplace = True)
print(all_data.isnull().sum().max())


# In[ ]:


total = all_data.isnull().sum().sort_values(ascending = False)
total.head(19)


# #### <strong>누락(결측값)이 있는 data를 채우도록 하겠습니다.</strong>
#     - 위의 info를 통하여 파악한 Data Type별로 채우도록 하겠습니다.
#         - 수치형 Feature(numeric Feature)중 몇 개는 0으로 채우겠습니다.
#             - 나머지는 fillna
#         - 문자형 Feature(Categorical Feature)는 최빈값으로 채우도록하겠습니다.

# In[ ]:


numeric_missed = ['BsmtFinSF1',
                  'BsmtFinSF2',
                  'BsmtUnfSF',
                  'TotalBsmtSF',
                  'BsmtFullBath',
                  'BsmtHalfBath',
                  'GarageArea',
                  'GarageCars']

for feature in numeric_missed:
    all_data[feature] = all_data[feature].fillna(0)

categorical_missed = ['Exterior1st',
                      'Exterior2nd',
                      'SaleType',
                      'MSZoning',
                      'Electrical',
                      'KitchenQual']    
for feature in categorical_missed:
    all_data[feature] = all_data[feature].fillna(all_data[feature].mode()[0])


# In[ ]:


# test_df.info()
# test data에 Functional의 데이터가 존재하지 않으므로 fillna를 이용하여 누락된 값을 추가하겠습니다.
all_data['Functional'] = all_data['Functional'].fillna('Typ')

# test data셋에만 존재하므로, drop하겠습니다.
all_data.drop(['Utilities'], axis=1, inplace=True)

all_data.isnull().sum().max()


# # <strong>4. 데이터 전처리(Feature Engineering)</strong>
# - <strong>Feature 중 왜곡이 심한 데이터를 수정하겠습니다.</strong>

# In[ ]:


numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[numeric_feats].apply(lambda x : skew(x)).sort_values(ascending = False)
high_skew = skewed_feats[abs(skewed_feats) > 0.5]
high_skew


# In[ ]:


for feature in high_skew.index:
    all_data[feature] = np.log1p(all_data[feature])    


# In[ ]:


# 독립변수가 많으므로, 데이터를 결합해보겠습니다.
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# # <strong>5. 범주형 데이터(Categorical Feature)를 수치형 데이터(numerical Feature)로 변환</strong>
# - <strong>(Converting categorical to numerical)</strong>

# In[ ]:


all_data = pd.get_dummies(all_data)
all_data.head()


# ### <strong>데이터 전처리를 완료하였으므로, 이젠 원래의 train data 와 test data로 분리하겠습니다.</strong>

# In[ ]:


X_train = all_data[:len(Y_train)]
X_test = all_data[len(Y_train):]

X_train.shape , X_test.shape


# # <strong>6. 모델 학습(Apply ML Model)</strong>

# In[ ]:


from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

scorer = make_scorer(mean_squared_error, greather_is_better = False)

def rmse_CV_train(model):
    kf = KFold(5, shuffle = True, random_state = 42).get_n_splits(X_train.values)
    rmse = np.sqrt(-cross_val_score(model, X_train, Y_train, scoring = 'neg_mean_squared_error', cv = kf))
    
    return rmse

def rmse_CV_test(model):
    kf = KFold(5, shuffle = True, random_state = 42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, X_test, Y_test, scoring = 'neg_mean_squared_error', cv = kf))
    
    return rmse


# In[ ]:


import xgboost as XGB
## GridSearch 통하여, 모델링 할 수 있습니다.
## 참고 : https://www.kaggle.com/lifesailor/xgboost
the_model = XGB.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, random_state =7, nthread = -1)

the_model.fit(X_train, Y_train)


# In[ ]:


y_predict = np.floor(np.expm1(the_model.predict(X_test)))
y_predict


# In[ ]:


submission_park = pd.DataFrame({
    'Id' : test_id,
    'SalePrice' : y_predict
})

submission_park.to_csv('submission.csv',index=False)

