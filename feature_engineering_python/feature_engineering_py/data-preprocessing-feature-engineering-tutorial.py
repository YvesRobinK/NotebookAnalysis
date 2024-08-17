#!/usr/bin/env python
# coding: utf-8

# ### 데이터 전처리, 피처엔지니어링 학습한 내용 정리
# - 본 노트북은 집값 예측 뿐아니라 일반적인 데이터 전처리(피처엔지니어링)을 다루고 있습니다.
# - 영상을 함께 만들었으니 함께 보시면서 학습하셔도 됩니다!

# # index
# 
# ## 결측치 
# - [결측치 튜토리얼 영상보기](https://youtu.be/krvH9gdcXw0)
# - dataset-load : house-prices-advanced-regression-techniques
# - 결측치 확인
# - 결측치 삭제
# - zero, mean, median, min, max, freq
# - groupby 활용: 특정 컬럼을 기준으로 평균값, 중앙값
# - 시계열 데이터: 앞에 값, 뒤에 값 채우기
# - 시계열 데이터: 보간법
# - sklearn.impute 활용
# -----
# ## 아웃라이어 
# - [아웃라이어 튜토리얼 영상보기](https://youtu.be/5fr_DhUohyE)
# - 간단한 아웃라이어 제거법
# -----
# ## Categorical Features 
# - [범주형 변수 튜토리얼 영상보기](https://youtu.be/owUHKCcpda0)
# - Label encoding
# - Onehot encoding
# - Count encoding
# - LabelCount encoding (랭킹)
# - Hash encoding
# - Sum Encoding
# - Polynomial Encoder
# - Target(Mean) encoding
# -----
# ## Numerical Features 
# - [수치형 변수 튜토리얼 영상보기](https://youtu.be/V5l0z3Uznlw)
# - Scaling
#     - Standard Scaling
#     - MinMax Scaling
#     - Nomalization
#     - Log Scaling
# - Binning
#     - Quantile
#     - 이진화
# -----
# ## Date Feature 
# - [날짜형 변수 튜토리얼 영상보기](https://youtu.be/VJFZ4kj6oWw)
# - 날짜 데이터 1개로 13개 피처 생성하기
# 
# 
# -----
# ## Weather Feature (join, merge)
# - [날씨 변수 튜토리얼 영상보기](https://youtu.be/M_KLFxRTiTY)
# - dataset-load : austin-weather
# - 외부데이터(날씨)를 불러 온 뒤 날짜 기준으로 데이터 합치기
# 
# 
# -----
# ## Group by (Feature Generation)
# - [Group by 튜토리얼 영상보기](https://youtu.be/p-b2OoFmstM)
# - Groupby 후 count, max, min, mean 피처생성 방법 2가지 (transform, merge)
# 
# -----
# ## TEXT 관련 간단한 피처생성 (피처엔지니어링)
# 
# - [TEXT를 통해 간단히 피처생성 방법 튜토리얼 영상보기](https://youtu.be/DTYREoDCyqY)
# - dataset-load : Real or Not? NLP with Disaster Tweets
# - TEXT 길이로 피처 생성하기
# - 포함된 단어를 찾아 피처 생성하기
# 
# -----
# ## Moving Average: 이동평균 (피처엔지니어링)
# 
# - [이동평균 튜토리얼 영상보기](https://youtu.be/PNDe8pZXxVU)
# - dataset-load : S&P 500 stock data
# - 이동평균 rolling(windows=x): 이전 x개의 평균
# - 이동평균 피처 생성 후, 그래프 그려보기
# - Shift, 밀어서 변동비율 피처 생성
# -----

# In[1]:


import numpy as np
import pandas as pd

pd.set_option('max_columns', 500)
pd.set_option('max_rows', 500)

import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='NanumBarunGothic') 

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm_notebook

from sklearn import preprocessing
import category_encoders as ce


# # Data load & View (house-prices)

# In[ ]:


path_house = "../input/house-prices-advanced-regression-techniques/train.csv"
df = pd.read_csv(path_house)
print(df.shape)
df.head()


# In[ ]:


df.describe()


# In[ ]:


df.describe(include=['O'])


# # Imputation 결측치 처리
# - 결측치 확인
# - 결측치 삭제
# - zero, mean, median, min, max, freq
# - groupby 활용: 특정 컬럼을 기준으로 평균값, 중앙값
# - 시계열 데이터: 앞에 값, 뒤에 값 채우기
# - 시계열 데이터: 보간법
# - sklearn.impute 활용

# ### 결측치 확인

# In[ ]:


df.isnull().sum()[:10]


# ### 결측치 삭제

# In[ ]:


df = pd.read_csv(path_house)

cols=['Alley', 'PoolQC']
df = df.drop(cols, axis=1)


# ### zero, mean, median, min, max, freq

# In[ ]:


df = pd.read_csv(path_house)
col = ["LotFrontage"]

#zero
df[col] = df[col].fillna(0)

# mean
df[col] = df[col].fillna(df[col].mean())

# median
df[col] = df[col].fillna(df[col].median())

# min
df[col] = df[col].fillna(df[col].min())

# max
df[col] = df[col].fillna(df[col].max())

#freq(최빈값)
# df[col] = df[col].fillna(df[col].mode()[0])


# ### 특정 컬럼을 기준으로 평균값, 중앙값

# In[ ]:


df = pd.read_csv(path_house)
col = ["LotFrontage"]

# 평균값
df[col] = df[col].fillna(df.groupby('MSZoning')[col].transform('mean'))

# 중앙
df[col] = df[col].fillna(df.groupby('MSZoning')[col].transform('median'))


# ### 시계열 데이터: 앞에 값, 뒤에 값 채우기

# In[ ]:


df = pd.read_csv(path_house)
col = ["LotFrontage"]

# 앞 값으로 채우기
df[col] = df[col].fillna(method='ffill')

# 뒷 값으로 채우기
df[col] = df[col].fillna(method='bfill')


# ### 시계열 데이터: 보간법

# In[ ]:


# 시계열데이터에서 선형으로 비례하는 방식으로 결측값 보간

df = pd.read_csv(path_house)

df = df.interpolate() # method='values
df = df.interpolate(method='time') # 날자기준으로 보간
df = df.interpolate(method='values', limit=1) #사이에 결측치가 여러개 있더라도 하나만 채우기
df =df.interpolate(method='values', limit=1, limit_direction='backward') #보간 방향 설정 뒤에서 앞으로


# ### sklearn.impute 활용

# In[ ]:


from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])

X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
print(imp_mean.transform(X))


# # Outlier(아웃라이어)

# In[ ]:


df = pd.read_csv(path_house)
plt.scatter(x=df['GrLivArea'], y=df['SalePrice'])
plt.xlabel('GrLivArea', fontsize=12)
plt.ylabel('SalePrice', fontsize=12)


# In[ ]:


outlier = df[(df['GrLivArea']>4000)&(df['SalePrice']<500000)].index
df=df.drop(outlier, axis=0)


# # Categorical Features(범주형)
# - Label encoding
# - Onehot encoding
# - Count encoding
# - LabelCount encoding (랭킹)
# - Hash encoding
# - Sum Encoding
# - Polynomial Encoder
# - Target(Mean) encoding

# In[ ]:


df = pd.read_csv(path_house)
df.info()


# In[ ]:


df = pd.read_csv(path_house)
col = ['MSZoning']
cols = ['MSZoning', 'Neighborhood']


# Object -> Categorical

# 1개 변환
df[col] = df[col].astype('category')

# 여러개 변환
for c in cols : 
    df[c] = df[c].astype('category') 


# In[ ]:


df.info()


# ### Label encoding

# In[ ]:


# 라벨 인코딩 
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(path_house)
cols = ['SaleType', 'SaleCondition']

display(df[cols].head(1))

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    df[col]=le.fit_transform(df[col])

display(df[cols].head(1))


# ### Onehot encoding

# In[ ]:


# 원핫 인코딩 
df = pd.read_csv(path_house)
cols = ['SaleType', 'SaleCondition']

display(df.head(1))

df_oh = pd.get_dummies(df[cols])
df = pd.concat([df, df_oh], axis=1)
df = df.drop(cols, axis=1)

display(df.head(1))


# ### Count encoding

# In[ ]:


# !pip install category_encoders


# In[ ]:


# 카운터 인코딩


df = pd.read_csv(path_house)
col =['MSZoning']

display(df[col].head(1))

for col in tqdm_notebook(cols):
    count_enc = ce.CountEncoder()
    df[col]=count_enc.fit_transform(df[col])

display(df[col].head(1))


# ### LabelCount encoding (랭킹)

# In[ ]:


def labelcount_encode(X, categorical_features, ascending=False):
    print('LabelCount encoding: {}'.format(categorical_features))
    X_ = pd.DataFrame()
    for cat_feature in categorical_features:
        cat_feature_value_counts = X[cat_feature].value_counts()
        value_counts_list = cat_feature_value_counts.index.tolist()
        if ascending:
            # for ascending ordering
            value_counts_range = list(
                reversed(range(len(cat_feature_value_counts))))
        else:
            # for descending ordering
            value_counts_range = list(range(len(cat_feature_value_counts)))
        labelcount_dict = dict(zip(value_counts_list, value_counts_range))
        X_[cat_feature] = X[cat_feature].map(
            labelcount_dict)
    X_ = X_.add_suffix('_labelcount_encoded')
    if ascending:
        X_ = X_.add_suffix('_ascending')
    else:
        X_ = X_.add_suffix('_descending')
    X_ = X_.astype(np.uint32)
    return X_


# In[ ]:


df = pd.read_csv(path_house)
df['LotArea'] = labelcount_encode(df, ['LotArea'])
df.head(3)


# ### Hash encoding

# In[ ]:


df = pd.read_csv(path_house)
y = df['LotArea']
X = df['MSZoning']
Hashing_encoder = ce.HashingEncoder(cols = ['MSZoning'])
Hashing_encoder.fit_transform(X, y)


# ### Sum Encoding

# In[ ]:


df = pd.read_csv(path_house)
y = df['LotArea']
X = df['MSZoning']
Sum_encoder = ce.SumEncoder(cols = ['MSZoning'])
Sum_encoder.fit_transform(X, y)


# ### Target(Mean) encoding

# In[ ]:


df = pd.read_csv(path_house)
y = df['LotArea']
X = df['SaleCondition']
ce_target = ce.TargetEncoder(cols = ['SaleCondition'])
ce_target.fit(X, y)
ce_target.transform(X, y)


# # Numerical Features
# 
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
# 
# - Scaling
#     - Standard Scaling
#     - MinMax Scaling
#     - Nomalization
#     - RobustScaler
#     - Log Scaling
# - Binning
#     - Quantile
#     - 이진화
# 

# ### Standard Scaling

# In[ ]:


# Standard Scaling (평균을 0, 분산을 1로 변경)
from sklearn.preprocessing import StandardScaler
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
print(scaler.fit(data))
print(scaler.mean_)
print(scaler.transform(data))
print(scaler.transform([[2, 2]]))


# ### MinMax Scaling

# In[ ]:


# MinMax Scaling 0과 1사이
from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
print(scaler.fit(data))
print(scaler.data_max_)
print(scaler.transform(data))
print(scaler.transform([[2, 2]]))


# ### Nomalization

# In[ ]:


# Nomalization 정규화
from sklearn.preprocessing import Normalizer
X = [[4, 1, 2, 2],
     [1, 3, 9, 3],
     [5, 7, 5, 1]]
transformer = Normalizer().fit(X)  # fit does nothing.
transformer
transformer.transform(X)


# ### RobustScaler

# In[ ]:


# Standard와 유사 하나 평균과 분산 대신, median과 quartile을 사용
from sklearn.preprocessing import RobustScaler
X = [[ 1., -2.,  2.],
     [ -2.,  1.,  3.],
     [ 4.,  1., -2.]]
transformer = RobustScaler().fit(X)
transformer
transformer.transform(X)


# ### Log Scaling

# In[ ]:


# Log Scaling
df = pd.read_csv(path_house)
col =['SalePrice']
display(df[col].head(3))
df[col].plot(kind='kde')
df[col] = np.log1p(df[col]) # 원본 값
df[col].plot(kind='kde')

display(df[col].head(3)) # 로그 스케일

display(np.expm1(df[col]).head(3)) # expm으로 환원


# In[ ]:





# ### Quantile binning

# In[ ]:


# Quantile binning
df = pd.read_csv(path_house)
col =['LotArea']

q = df[col].quantile([.1,.5,1])


# In[ ]:


df[col].describe()


# In[ ]:


q


# ### 이진화 (0 또는 1)

# In[ ]:


#이진화 0 또는 1

df = pd.read_csv(path_house)
col =['LotArea']

binarizer = preprocessing.Binarizer(threshold=10000)
b = binarizer.transform(df[col])
b = pd.DataFrame(b)
display(df[col])
display(b)


# # Date Feature

# In[ ]:


df = pd.DataFrame({'일시':['2020.7.1 19:00',
                   '2020.8.1 20:10',
                   '2021.9.1 21:20',
                   '2022.10.1 22:30',
                   '2022.11.1 23:30',
                   '2022.12.1 23:40',
                   '2023.1.1 08:30']})
df


# In[ ]:


df.info()


# In[ ]:


# 문자열을 datetime 타입으로 변경
df['일시'] = df.일시.apply(pd.to_datetime)


# In[ ]:


df.info()


# In[ ]:


# s1
df = df.assign(
               year=df.일시.dt.year,
               month=df.일시.dt.month,
               day=df.일시.dt.day,
               hour=df.일시.dt.hour,
               minute=df.일시.dt.minute,
    
               quarter=df.일시.dt.quarter,
               weekday=df.일시.dt.weekday,
               weekofyear=df.일시.dt.weekofyear,
    
               month_start=df.일시.dt.is_month_start,
               month_end=df.일시.dt.is_month_end,
               quarter_start=df.일시.dt.is_quarter_start,
               quarter_end=df.일시.dt.is_quarter_end,
    
               daysinmonth=df.일시.dt.daysinmonth
               )


# In[ ]:


df.head(7)


# In[ ]:


# datetime 타입에서 년, 월, 일, 시간 추출
#2
df['year'] = df.일시.apply(lambda x : x.year)
df['month'] = df.일시.apply(lambda x : x.month)
df['day'] = df.일시.apply(lambda x : x.day)
df['hour'] = df.일시.apply(lambda x: x.hour)
df['minute'] = df.일시.apply(lambda x: x.minute)

#3
df['weekday'] = df['일시'].dt.weekday
df['weekofyear'] = df["일시"].dt.weekofyear
df['quarter'] = df["일시"].dt.quarter


# # Weather Feature (join)
#  - merge를 통해 외부데이터 합치기

# In[ ]:


# kaggle 데이터 셋에서 날씨로 된 데이터를 불러옵니다! 
path_house = "../input/austin-weather/austin_weather.csv"
w = pd.read_csv(path_house)
w.head()


# In[ ]:


w.info()


# In[ ]:


w['Date'] = w['Date'].apply(pd.to_datetime)


# In[ ]:


w.info()


# In[ ]:


w.head(8)


# In[ ]:


df = pd.DataFrame({'date':['2013.12.22 19:00',
                   '2013.12.23 20:10',
                   '2013.12.24 21:20',
                   '2013.12.25 22:30',
                   '2013.12.26 23:30',
                   '2013.12.27 23:40',
                   '2013.12.28 08:30'], 
                   'name':['A',
                   'B',
                   'C',
                   'D',
                   'E',
                   'F',
                   'G']})
df['date'] = df.date.apply(pd.to_datetime)
df


# In[ ]:


df = df.assign(
               year=df.date.dt.year,
               month=df.date.dt.month,
               day=df.date.dt.day
               )
df.head()


# In[ ]:


w = w.assign(
               year=w.Date.dt.year,
               month=w.Date.dt.month,
               day=w.Date.dt.day
               )
w.head()


# In[ ]:


df = pd.merge(df, w, how='left', on=['year','month','day'])
df.head()


# # Group by (피처엔지니어링)
#  - Group by를 통해 피처생성하기

# In[ ]:


# house-prices-advanced-regression Data-set
df = pd.read_csv(path_house)
df.head()


# In[ ]:


df.head()


# In[ ]:


# groupby 작성법1
df.groupby('MSZoning')['LotArea'].max()
df.groupby('MSZoning')['LotArea'].min()
df.groupby('MSZoning')['LotArea'].mean()
df.groupby(['MSZoning','LotShape'])['LotArea'].count()


# In[ ]:


# groupby 작성법2
df_group = df.groupby('MSZoning')
df_group['LotArea'].max() 


# In[ ]:


# 피처생성방법1
df['new_max'] = df.groupby('MSZoning')['LotArea'].transform(lambda x: x.max())
df.head()


# In[ ]:


df_agg = pd.DataFrame()
df_agg =  df.groupby('MSZoning')['LotArea'].agg(['max', 'min', 'mean'])


# In[ ]:


# 피처생성방법2
df_all = pd.merge(df, df_agg, how='left', on=['MSZoning'])
df_all.head()


# In[ ]:





# # TEXT 관련 간단한 피처생성 (피처엔지니어링)
# - TEXT 길이로 피처 생성하기
# - 포함된 단어를 찾아 피처 생성하기

# In[ ]:


# Real or Not? NLP with Disaster Tweets Data-set
df = pd.read_csv('../input/nlp-getting-started/train.csv')


# In[ ]:


df.head(10)


# ### 문자열 길이

# In[ ]:


df["length"] = df["text"].str.len()
df.head()


# ### 키워드(단어)

# In[ ]:


df[df['text'].str.contains('911')].head()


# In[ ]:


df['k'] = 0
df.loc[df[df['text'].str.contains('emergency')].index,'k'] = 1
df.loc[df[df['text'].str.contains('help')].index,'k'] = 2
df.loc[df[df['text'].str.contains('accident')].index,'k'] = 3


# In[ ]:





# # Moving Average: 이동평균
# - 이동평균 rolling(windows=x): 이전 x개의 평균
# - 이동평균 피처 생성 후, 그래프 그려보기

# ### 이동평균

# In[2]:


df = pd.read_csv('../input/sandp500/all_stocks_5yr.csv')
print(df.shape)
df.head()


# In[7]:


df = df[:300]


# In[8]:


df['ma5'] = df['close'].rolling(window=5).mean()
df['ma30'] = df['close'].rolling(window=30).mean()
df['ma60'] = df['close'].rolling(window=60).mean()
df.tail(6)


# ### 시각화

# In[10]:


plt.plot(df.index, df['ma5'], label = "ma5")
plt.plot(df.index, df['ma30'], label = "ma30")
plt.plot(df.index, df['ma60'], label = "ma60")
plt.plot(df.index, df['close'], label='close')
plt.legend()
plt.grid()


# # Shift

# In[12]:


# shift(-1) 밀어서 피처 생성
df['nextClose']= df['close'].shift(-1)

# 주가변동
df['fluctuation'] = df['nextClose'] - df['close']
df['f_rate'] = df['fluctuation'] / df['nextClose']

df.head()


# In[13]:


plt.figure(figsize=(10,6))
plt.plot(df.index, df['f_rate'])
plt.axhline(y=0, color='gray', ls = '-')


# In[15]:


# 분포 
df['f_rate'].plot.hist()
# df['f_rate'].plot.kde()


# In[16]:


df['f_rate'].plot.kde()


# In[ ]:




