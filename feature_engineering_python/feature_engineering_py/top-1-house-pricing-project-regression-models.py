#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.oguzerdogan.com/">
#     <img src="https://www.oguzerdogan.com/wp-content/uploads/2020/10/logo_oz.png" width="200" align="right">
# </a>

# <a href="">
#     <img src="https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png" align="center">
# </a>

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Exploratory-Data-Analysis" data-toc-modified-id="Exploratory-Data-Analysis-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Exploratory Data Analysis</a></span><ul class="toc-item"><li><span><a href="#Hedef-Değişken-SalePrice" data-toc-modified-id="Hedef-Değişken-SalePrice-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Hedef Değişken SalePrice</a></span></li></ul></li><li><span><a href="#En-Önemli-Sayısal-Değişkenler" data-toc-modified-id="En-Önemli-Sayısal-Değişkenler-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>En Önemli Sayısal Değişkenler</a></span></li><li><span><a href="#Sayısal-Değişken-Analizi" data-toc-modified-id="Sayısal-Değişken-Analizi-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Sayısal Değişken Analizi</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#OverallQual" data-toc-modified-id="OverallQual-3.0.1"><span class="toc-item-num">3.0.1&nbsp;&nbsp;</span>OverallQual</a></span></li><li><span><a href="#GarageCars" data-toc-modified-id="GarageCars-3.0.2"><span class="toc-item-num">3.0.2&nbsp;&nbsp;</span>GarageCars</a></span></li><li><span><a href="#GrLivArea" data-toc-modified-id="GrLivArea-3.0.3"><span class="toc-item-num">3.0.3&nbsp;&nbsp;</span>GrLivArea</a></span></li><li><span><a href="#YearBuilt" data-toc-modified-id="YearBuilt-3.0.4"><span class="toc-item-num">3.0.4&nbsp;&nbsp;</span>YearBuilt</a></span></li><li><span><a href="#FullBath" data-toc-modified-id="FullBath-3.0.5"><span class="toc-item-num">3.0.5&nbsp;&nbsp;</span>FullBath</a></span></li><li><span><a href="#1stFlrSF" data-toc-modified-id="1stFlrSF-3.0.6"><span class="toc-item-num">3.0.6&nbsp;&nbsp;</span>1stFlrSF</a></span></li><li><span><a href="#TotalBsmtSF" data-toc-modified-id="TotalBsmtSF-3.0.7"><span class="toc-item-num">3.0.7&nbsp;&nbsp;</span>TotalBsmtSF</a></span></li><li><span><a href="#GarageArea" data-toc-modified-id="GarageArea-3.0.8"><span class="toc-item-num">3.0.8&nbsp;&nbsp;</span>GarageArea</a></span></li></ul></li></ul></li><li><span><a href="#En-İyi-Kategorik-Değişkenler" data-toc-modified-id="En-İyi-Kategorik-Değişkenler-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>En İyi Kategorik Değişkenler</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Neighborhood" data-toc-modified-id="Neighborhood-4.0.1"><span class="toc-item-num">4.0.1&nbsp;&nbsp;</span>Neighborhood</a></span></li><li><span><a href="#ExterQual" data-toc-modified-id="ExterQual-4.0.2"><span class="toc-item-num">4.0.2&nbsp;&nbsp;</span>ExterQual</a></span></li><li><span><a href="#BsmtQual" data-toc-modified-id="BsmtQual-4.0.3"><span class="toc-item-num">4.0.3&nbsp;&nbsp;</span>BsmtQual</a></span></li><li><span><a href="#KitchenQual" data-toc-modified-id="KitchenQual-4.0.4"><span class="toc-item-num">4.0.4&nbsp;&nbsp;</span>KitchenQual</a></span></li></ul></li></ul></li><li><span><a href="#Data-Preprocessing" data-toc-modified-id="Data-Preprocessing-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Data Preprocessing</a></span><ul class="toc-item"><li><span><a href="#Eksik-Gözlemler" data-toc-modified-id="Eksik-Gözlemler-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Eksik Gözlemler</a></span></li><li><span><a href="#Eksik-Gözlemleri-Doldurma" data-toc-modified-id="Eksik-Gözlemleri-Doldurma-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Eksik Gözlemleri Doldurma</a></span></li></ul></li><li><span><a href="#Feature-Engineering" data-toc-modified-id="Feature-Engineering-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Feature Engineering</a></span><ul class="toc-item"><li><span><a href="#Rare-Analyze" data-toc-modified-id="Rare-Analyze-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Rare Analyze</a></span></li></ul></li><li><span><a href="#Outliers" data-toc-modified-id="Outliers-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Outliers</a></span><ul class="toc-item"><li><span><a href="#Yeni-Değişkenler" data-toc-modified-id="Yeni-Değişkenler-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Yeni Değişkenler</a></span></li><li><span><a href="#Box-Cox-dönüşümü" data-toc-modified-id="Box-Cox-dönüşümü-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Box-Cox dönüşümü</a></span></li></ul></li><li><span><a href="#Son-İşlemler" data-toc-modified-id="Son-İşlemler-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Son İşlemler</a></span></li><li><span><a href="#MODELLEME" data-toc-modified-id="MODELLEME-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>MODELLEME</a></span></li><li><span><a href="#Cross-Validation" data-toc-modified-id="Cross-Validation-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Cross Validation</a></span></li><li><span><a href="#Model-Results" data-toc-modified-id="Model-Results-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Model Results</a></span><ul class="toc-item"><li><span><a href="#Stacking-&amp;-Blending" data-toc-modified-id="Stacking-&amp;-Blending-11.1"><span class="toc-item-num">11.1&nbsp;&nbsp;</span>Stacking &amp; Blending</a></span></li><li><span><a href="#Submission" data-toc-modified-id="Submission-11.2"><span class="toc-item-num">11.2&nbsp;&nbsp;</span>Submission</a></span></li></ul></li></ul></div>

# Important note: I'll update this notebook for English soon!

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('fivethirtyeight')

from datetime import datetime
from category_encoders import TargetEncoder

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from scipy import stats
from scipy.stats import skew, boxcox_normmax, norm
from scipy.special import boxcox1p
from lightgbm import LGBMRegressor

import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

import warnings
warnings.simplefilter('ignore')
print('Setup complete')


# In[2]:


# Read data
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print("Train set size:", train.shape)
print("Test set size:", test.shape)


# In[3]:


train.head()


# In[4]:


# Helper Functions
# https://www.kaggle.com/mviola/house-prices-eda-lasso-lightgbm-0-11635 special thanks to this notebook its helped me a lot, for eda part.
def plot_numerical(col, discrete=False):
    if discrete:
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        sns.stripplot(x=col, y='SalePrice', data=train, ax=ax[0])
        sns.countplot(train[col], ax=ax[1])
        fig.suptitle(str(col) + ' analysis')
    else:
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        sns.scatterplot(x=col, y='SalePrice', data=train, ax=ax[0])
        sns.distplot(train[col], kde=False, ax=ax[1])
        fig.suptitle(str(col) + ' analysis')

def plot_categorical(col):
    fig, ax = plt.subplots(1,2,figsize=(12,6), sharey=True)
    sns.stripplot(x=col, y='SalePrice', data=train, ax=ax[0])
    sns.boxplot(x=col, y='SalePrice', data=train, ax=ax[1])
    fig.suptitle(str(col) + ' analysis')
    
print('Plot functions are ready to use')


# # Exploratory Data Analysis

# ## Hedef Değişken SalePrice

# In[5]:


plt.figure(figsize=(8,5))
a = sns.distplot(train.SalePrice, kde=False)
plt.title('SalePrice distribution')
a = plt.axvline(train.SalePrice.describe()['25%'], color='b')
a = plt.axvline(train.SalePrice.describe()['75%'], color='b')
print('SalePrice description:')
print(train.SalePrice.describe().to_string())


# # En Önemli Sayısal Değişkenler
# 
# Büyük ve çok fazla değişkenin olduğu bir veri setinde ilk olarak önemli değişkenleri gözlemlemek bize zaman ve önfikir sağlayacaktır.  
# Bağımlı Değişkeni en iyi açıklayan sayısal değişkenlere bakmak için ExtraTreesRegressor kullanacağım.

# In[6]:


# Select numerical features only
num_features = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]
# Remove Id & SalePrice 
num_features.remove('Id')
num_features.remove('SalePrice')
# Create a numerical columns only dataframe
num_analysis = train[num_features].copy()
# Impute missing values with the median just for the moment
for col in num_features:
    if num_analysis[col].isnull().sum() > 0:
        num_analysis[col] = SimpleImputer(strategy='median').fit_transform(num_analysis[col].values.reshape(-1,1))
# Train a model   
clf = ExtraTreesRegressor(random_state=42)
h = clf.fit(num_analysis, train.SalePrice)
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,num_features)), columns=['Value','Feature'])
plt.figure(figsize=(16,10))
sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False))
plt.title('Most important numerical features with ExtraTreesRegressor')
del clf, h;


# İlk 8 değişkene bakmak veriyi anlamak için yeterli diye düşünüyorum.  
# Bu 8 değişkenin bağımlı değişken ve kendi aralarındaki korelasyonuna da bakmak iyi olacaktır.

# In[7]:


plt.figure(figsize=(8,8))
plt.title('Correlation matrix with SalePrice')
selected_columns = ['OverallQual', 'GarageCars', 'GrLivArea', 'YearBuilt', 'FullBath', '1stFlrSF', 'TotalBsmtSF', 'GarageArea']
a = sns.heatmap(train[selected_columns + ['SalePrice']].corr(), annot=True, square=True)


# # Sayısal Değişken Analizi 

# `GarageArea` - `GarageCars` ve `1stFlSF` - `TotalBsmtSF` arasında yüksek korelasyon göze çarpıyor.
# ###  OverallQual

# In[8]:


plot_numerical('OverallQual', True)


# Derecelendirmeye uygun güzel bir trend görünüyor. 
# Fakat `OverallQual` 10 derecesinde 2 tane çok ucuza ev bulunuyor, bunlar büyük ihtimalle outlier.
# ### GarageCars
# Garaj sayısı ile beraber garaj kapasitesini ölçen bir değişken.

# In[9]:


plot_numerical('GarageCars', True)


#  `GarageCars` 4 araç kapasiteli garajların çok az olduğunu ve 3 kapasiteli araçlardan daha ucuz olduğunu görebiliyoruz. Buraya ilerleyen kısımlarda el atacağım.
# Geri kalan trendler güzel  görünüyor. `GarageCars` 3 araç kapasiteli garajlarda çok pahalı 2 tane lüks evin olduğu gözlemleniyor.
# ### GrLivArea
# Bu değer, evin fit kare cinsinden (zemin) yaşam alanını temsil eder ve insanların satın alırken dikkate aldıkları en yaygın özelliklerden biridir.  
# İnsanların önem verdiği bir özelliğin satış fiyatıyla bu kadar korelasyonunun olması şaşırtıcı değil. Bağımlı değişken ile arasında % 71 korelasyon var.  
# Sağ alt tarafta iki tane outlier gözlemleniyor.

# In[10]:


plot_numerical('GrLivArea')


# ### YearBuilt
# Evin orjinal inşa tarihi.

# In[11]:


plot_numerical('YearBuilt')


# Daha yeni evlerin genellikle daha yüksek bir fiyata sahip olduğunu görebiliyoruz.  
# Orta-yüksek fiyatlı bazı tarihi evler olsa da, pahalı evlerden (400k) bahsettiğimizde, biri hariç hepsi son 20 yılda inşa edildiğini görüyoruz.  
# Ayrıca bir evin asgari fiyatının, savaş sonrası ikinci dönemden sonra oldukça güçlü bir eğilimle arttığını görüyoruz.
# ### FullBath
# Tam banyoların toplam sayısını gösterir. Önem bakımından 5. sırada görünüyor.

# In[12]:


plot_numerical('FullBath', True)


# Trend güzel görünüyor fakat 0 ve 3 teki değerler biraz garip.
# ### 1stFlrSF
# Bu sütun bize fit kare cinsinden birinci katın boyutunu anlatıyor ve ağaç modelimiz için önemli olması tamamen mantıklı.  
# `SalePrice` ile 0,61'lik bir korelasyonu var, hiç fena değil.  
# Aslında `GrLivArea` alan bakımından zaten aynı görevi görmekte yine de modelde tutacağım.   
# İki aykırı değerin burada da mevcut olduğunu ve öncekiyle aynı olduğunu not ediyoruz.

# In[13]:


plot_numerical('1stFlrSF')


# ### TotalBsmtSF
# Bu sütun, bodrum alanının toplam fit karesini içeriyor.   
# Bağımlı değişken ile güzel bir korelasyonu var % 61.   
# Gördüğümüz üzere bazı 0 değerleri var, her evin bodrum katı olmadığı için bu doğal bir durum.   
# Burada da outlierlar var gibi görünüyor.

# In[14]:


plot_numerical('TotalBsmtSF')


# ### GarageArea
# 

# In[15]:


plot_numerical('GarageArea')


# # En İyi Kategorik Değişkenler
# 

# In[16]:


# Select categorical features only
cat_features = [col for col in train.columns if train[col].dtype =='object']
# Create a categorical columns only dataframe
cat_analysis = train[cat_features].copy()
# Impute missing values with NA just for the moment
for col in cat_analysis:
    if cat_analysis[col].isnull().sum() > 0:
        cat_analysis[col] = SimpleImputer(strategy='constant').fit_transform(cat_analysis[col].values.reshape(-1,1))
# Target encoding
target_enc = TargetEncoder(cols=cat_features)
cat_analysis = target_enc.fit_transform(cat_analysis, train.SalePrice) 
# Train a model 
clf = ExtraTreesRegressor(random_state=42)
h = clf.fit(cat_analysis, train.SalePrice)
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,cat_features)), columns=['Value','Feature'])
plt.figure(figsize=(16,10))
sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False))
plt.title('Most important categorical features with ExtraTreesRegressor')
del clf, h;


# ### Neighborhood
# Bu değişken Ames şehir sınırları içindeki fiziksel konumları temsil eder ve modelimize göre en kullanışlı kategorik değişken görünüyor.  
# Sanırım bunun nedeni bölgelerin yoksulluğu ve zenginliği ile alakalı olabilir.

# In[17]:


fig, ax = plt.subplots(1,2,figsize=(16,6), sharey=True)
sns.stripplot(x='Neighborhood', y='SalePrice', data=train, ax=ax[0])
sns.boxplot(x='Neighborhood', y='SalePrice', data=train, ax=ax[1])
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
fig.suptitle('Neighborhood analysis')
plt.show()


# ### ExterQual
# 
# ```
# Ex	Excellent
# Gd	Good
# TA	Average/Typical
# Fa	Fair
# Po	Poor
# ```
# 

# In[18]:


plot_categorical('ExterQual')


# ### BsmtQual
# 
# ```
# Ex	Excellent (100+ inches)	
# Gd	Good (90-99 inches)
# TA	Typical (80-89 inches)
# Fa	Fair (70-79 inches)
# Po	Poor (<70 inches)
# ```
# 

# In[19]:


plot_categorical('BsmtQual')


# ### KitchenQual
# 
# 
# ```
# Ex	Excellent
# Gd	Good
# TA	Typical/Average
# Fa	Fair
# Po	Poor
# ```  

# In[20]:


plot_categorical('KitchenQual')


# # Data Preprocessing

# 
# ## Eksik Gözlemler
# 

# In[21]:


fig, ax = plt.subplots(1,2,figsize=(16,6), sharey=True)
train_missing = round(train.isnull().mean()*100, 2)
train_missing = train_missing[train_missing > 0]
train_missing.sort_values(inplace=True)
sns.barplot(train_missing.index, train_missing, ax=ax[0], color='orange')
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
ax[0].set_ylabel('Percentage of missing values')
ax[0].set_title('Train set')
test_missing = round(test.isnull().mean()*100, 2)
test_missing = test_missing[test_missing > 0]
test_missing.sort_values(inplace=True)
sns.barplot(test_missing.index, test_missing, ax=ax[1], color='orange')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
ax[1].set_title('Test set')
plt.show()


# Bazı eksik değerler kasıtlı olarak boş bırakılmıştır.Bu o özelliğin olmadığı anlamına geliyor. Yani bodrumkatı olmayan bir evin bodrum katı ile ilgili kısımları NA

# In[22]:


plot_numerical('LotFrontage')


# In[23]:


print('LotFrontage minimum:', train.LotFrontage.min())


# In[24]:


plot_categorical('FireplaceQu')


# In[25]:


fig, ax = plt.subplots(2,2,figsize=(12,10), sharey=True)
sns.stripplot(x='Fence', y='SalePrice', data=train, ax=ax[0][0])
sns.stripplot(x='Alley', y='SalePrice', data=train, ax=ax[0][1])
sns.stripplot(x='MiscFeature', y='SalePrice', data=train, ax=ax[1][0])
sns.stripplot(x='PoolQC', y='SalePrice', data=train, ax=ax[1][1])
fig.suptitle('Analysis of columns with more than 80% of missing values')
plt.show()


# ## Eksik Gözlemleri Doldurma

# In[26]:


# Gereksiz olan ID sütununu çıkarıyorum.

train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

train_features = train
test_features = test


# In[27]:


# Ön işleme işlemleri için train ve test veri setini birleştirme.

features = pd.concat([train, test]).reset_index(drop=True)
print(features.shape)


# In[28]:


def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    dtypes = dataframe.dtypes
    dtypesna = dtypes.loc[(np.sum(features.isnull()) != 0)]
    missing_df = pd.concat([n_miss, np.round(ratio, 2), dtypesna], axis=1, keys=['n_miss', 'ratio', 'type'])
    if len(missing_df)>0:
        print(missing_df)
        missing = train.isnull().sum()
        missing = missing[missing > 0]
        missing.sort_values(inplace=True)
        missing.plot.bar()
        print("\nThere are {} columns with missing values\n".format(len(missing_df)))
    else:
        print("\nThere is no missing value")
missing_values_table(features)


# In[29]:


# Buradaki eksik gözlemler o özelliğin olmadığı anlamına gelmekte. Bu yüzden None atayacağım.

none_cols = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType',
             'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']

# Sayısal değişkenlerdeki eksik gözlemler de aynı şekilde, bunlara 0 atıyorum.

zero_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath','BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea']

# Bu değişkenlerdeki eksik gözlemlere mod atayacağım.

freq_cols = ['Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual','SaleType', 'Utilities']


for col in zero_cols:
    features[col].replace(np.nan, 0, inplace=True)

for col in none_cols:
    features[col].replace(np.nan, 'None', inplace=True)

for col in freq_cols:
    features[col].replace(np.nan, features[col].mode()[0], inplace=True)


# In[30]:


# MsZoning değişkenindeki boş değerleri MSSubClassa göre doldurma.

features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].apply(
    lambda x: x.fillna(x.mode()[0]))

# LotFrontage mülkiyetin cadde ile bağlantısını gösteren bir değişken, her mahallenin cadde bağlantısının birbirine benzeyebileceğinden bunu Neighborhood'a a göre doldurdum.

features['LotFrontage'] = features.groupby(
    ['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))

# Sayısal değişken olup aslında kategorik değişken olması gerekenleri düzeltme

features['MSSubClass'] = features['MSSubClass'].astype(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)


# In[31]:


missing_values_table(features)


# # Feature Engineering

# In[32]:


def stalk(dataframe, var, target="SalePrice"):
    print("{}  | type: {}\n".format(var, dataframe[var].dtype))
    print(pd.DataFrame({"n": dataframe[var].value_counts(),
                                "Ratio": 100 * dataframe[var].value_counts() / len(dataframe),
                                "TARGET_MEDIAN": dataframe.groupby(var)[target].median(),
                                "Target_MEAN": dataframe.groupby(var)[target].mean()}), end="\n\n\n")
    
    plt.figure(figsize=(15,5))
    chart = sns.countplot(
    data=features,
    x=features[var])
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show();


# In[33]:


stalk(features, "Neighborhood")


# In[34]:


# Neighboor içerisindeki benzer değerde olanları birbiri ile grupladım.

neigh_map = {'MeadowV': 1,'IDOTRR': 1,'BrDale': 1,'BrkSide': 2,'OldTown': 2,'Edwards': 2,
             'Sawyer': 3,'Blueste': 3,'SWISU': 3,'NPkVill': 3,'NAmes': 3,'Mitchel': 4,
             'SawyerW': 5,'NWAmes': 5,'Gilbert': 5,'Blmngtn': 5,'CollgCr': 5,
             'ClearCr': 6,'Crawfor': 6,'Veenker': 7,'Somerst': 7,'Timber': 8,
             'StoneBr': 9,'NridgHt': 10,'NoRidge': 10}

features['Neighborhood'] = features['Neighborhood'].map(neigh_map).astype('int')


# In[35]:


# Derecelendirme içeren değişkenleri ordinal yapıya getirdim.

ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['ExterQual'] = features['ExterQual'].map(ext_map).astype('int')

ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['ExterCond'] = features['ExterCond'].map(ext_map).astype('int')

bsm_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['BsmtQual'] = features['BsmtQual'].map(bsm_map).astype('int')
features['BsmtCond'] = features['BsmtCond'].map(bsm_map).astype('int')

bsmf_map = {'None': 0,'Unf': 1,'LwQ': 2,'Rec': 3,'BLQ': 4,'ALQ': 5,'GLQ': 6}
features['BsmtFinType1'] = features['BsmtFinType1'].map(bsmf_map).astype('int')
features['BsmtFinType2'] = features['BsmtFinType2'].map(bsmf_map).astype('int')

heat_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['HeatingQC'] = features['HeatingQC'].map(heat_map).astype('int')
features['KitchenQual'] = features['KitchenQual'].map(heat_map).astype('int')
features['FireplaceQu'] = features['FireplaceQu'].map(bsm_map).astype('int')
features['GarageCond'] = features['GarageCond'].map(bsm_map).astype('int')
features['GarageQual'] = features['GarageQual'].map(bsm_map).astype('int')


# ## Rare Analyze

# In[36]:


# RARE ANALYZER
def rare_analyser(dataframe, target, rare_perc):
    rare_columns = [col for col in dataframe.columns if len(dataframe[col].value_counts()) <= 20
                    and (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]
    for var in rare_columns:
        print(var, ":", len(dataframe[var].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[var].value_counts(),
                            "RATIO": dataframe[var].value_counts() / len(dataframe),
                            "TARGET_MEDIAN": dataframe.groupby(var)[target].median()}), end="\n\n\n")


rare_analyser(features, "SalePrice", 0.01)


# In[37]:


def stalk(dataframe, var, target="SalePrice"):
    print("{}  | type: {}\n".format(var, dataframe[var].dtype))
    print(pd.DataFrame({"n": dataframe[var].value_counts(),
                                "Ratio": 100 * dataframe[var].value_counts() / len(dataframe),
                                "TARGET_MEDIAN": dataframe.groupby(var)[target].median(),
                                "Target_MEAN": dataframe.groupby(var)[target].mean()}), end="\n\n\n")
    
    plt.figure(figsize=(10,5))
    chart = sns.countplot(
    data=features,
    x=features[var])
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show();


# In[38]:


stalk(features,"LotShape")


# In[39]:


features.loc[(features["LotShape"] == "Reg"), "LotShape"] = 1
features.loc[(features["LotShape"] == "IR1"), "LotShape"] = 2
features.loc[(features["LotShape"] == "IR2"), "LotShape"] = 3 
features.loc[(features["LotShape"] == "IR3"), "LotShape"] = 3 

features["LotShape"] = features["LotShape"].astype("int")


# In[40]:


stalk(features,"GarageCars")


# In[41]:


features.loc[(features["GarageCars"] == "4"), "GarageCars"] = 3


# In[42]:


stalk(features,"LotConfig")


# In[43]:


features.loc[(features["LotConfig"]=="Inside"),"LotConfig"] = 1
features.loc[(features["LotConfig"]=="FR2"),"LotConfig"] = 1
features.loc[(features["LotConfig"]=="Corner"),"LotConfig"] = 1

features.loc[(features["LotConfig"]=="FR3"),"LotConfig"] = 2
features.loc[(features["LotConfig"]=="CulDSac"),"LotConfig"] = 2


# In[44]:


stalk(features, "LandSlope")


# In[45]:


features.loc[features["LandSlope"] == "Gtl", "LandSlope"] = 1

features.loc[features["LandSlope"] == "Sev", "LandSlope"] = 2
features.loc[features["LandSlope"] == "Mod", "LandSlope"] = 2
features["LandSlope"]= features["LandSlope"].astype("int")


# In[46]:


stalk(features,"OverallQual")


# In[47]:


features.loc[features["OverallQual"] == 1, "OverallQual"] = 1
features.loc[features["OverallQual"] == 2, "OverallQual"] = 1
features.loc[features["OverallQual"] == 3, "OverallQual"] = 1
features.loc[features["OverallQual"] == 4, "OverallQual"] = 2
features.loc[features["OverallQual"] == 5, "OverallQual"] = 3
features.loc[features["OverallQual"] == 6, "OverallQual"] = 4
features.loc[features["OverallQual"] == 7, "OverallQual"] = 5
features.loc[features["OverallQual"] == 8, "OverallQual"] = 6
features.loc[features["OverallQual"] == 9, "OverallQual"] = 7
features.loc[features["OverallQual"] == 10, "OverallQual"] = 8


# In[48]:


stalk(features,"Exterior1st")


# In[49]:


stalk(features,"MasVnrType")


# In[50]:


features.loc[features["MasVnrType"] == "BrkCmn" , "MasVnrType"] = "None" 


# In[51]:


stalk(features,"Foundation")


# In[52]:


features.loc[features["Foundation"] == "Stone", "Foundation"] = "BrkTil"
features.loc[features["Foundation"] == "Wood", "Foundation"] = "CBlock"


# In[53]:


stalk(features,"Fence")


# In[54]:


features.loc[features["Fence"] == "MnWw", "Fence"] = "MnPrv"
features.loc[features["Fence"] == "GdWo", "Fence"] = "MnPrv"


# In[55]:


# RARE ANALYZER
def rare_analyser(dataframe, target, rare_perc):
    rare_columns = [col for col in dataframe.columns if len(dataframe[col].value_counts()) <= 20
                    and (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]
    for var in rare_columns:
        print(var, ":", len(dataframe[var].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[var].value_counts(),
                            "RATIO": dataframe[var].value_counts() / len(dataframe),
                            "TARGET_MEDIAN": dataframe.groupby(var)[target].median()}), end="\n\n\n")


rare_analyser(features, "SalePrice", 0.01)


# In[56]:


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    return temp_df

features = rare_encoder(features, 0.01)


# # Outliers
# 
# In this part special thanks to Dear Ertuğrul. He and his notebook helped me a lot. https://www.kaggle.com/datafan07/top-1-approach-eda-new-models-and-stacking
# 

# In[57]:


# Plotting numerical features with polynomial order to detect outliers by eye.

def srt_reg(y, df):
    fig, axes = plt.subplots(12, 3, figsize=(25, 80))
    axes = axes.flatten()

    for i, j in zip(df.select_dtypes(include=['number']).columns, axes):

        sns.regplot(x=i,
                    y=y,
                    data=df,
                    ax=j,
                    order=3,
                    ci=None,
                    color='#25B89B',
                    line_kws={'color': 'grey'},
                    scatter_kws={'alpha':0.4})
        j.tick_params(labelrotation=45)
        j.yaxis.set_major_locator(MaxNLocator(nbins=8))

        plt.tight_layout()


# In[58]:


srt_reg('SalePrice', train)


# In[59]:


features.shape


# In[60]:


# Dropping outliers after detecting them by eye.
features.loc[2590, 'GarageYrBlt'] = 2007 # missing value it was 2207

features = features.drop(features[(features['OverallQual'] < 5)
                                  & (features['SalePrice'] > 200000)].index)
features = features.drop(features[(features['GrLivArea'] > 4000)
                                  & (features['SalePrice'] < 200000)].index)
features = features.drop(features[(features['GarageArea'] > 1200)
                                  & (features['SalePrice'] < 200000)].index)
features = features.drop(features[(features['TotalBsmtSF'] > 3000)
                                  & (features['SalePrice'] > 320000)].index)
features = features.drop(features[(features['1stFlrSF'] < 3000)
                                  & (features['SalePrice'] > 600000)].index)
features = features.drop(features[(features['1stFlrSF'] > 3000)
                                  & (features['SalePrice'] < 200000)].index)


# In[61]:


# Dropping target value
y = features['SalePrice']
y.dropna(inplace=True)
features.drop(columns='SalePrice', inplace=True)


# In[62]:


features.shape


# ## Yeni Değişkenler
# 
# Bu kısımda olan değişkenlerden yeni değişkenler türetiliyor. İlk çalışmam olduğu için kendi düşündüklerimi ve çoğunlukla toplulukta kullanılan değişkenleri derledim.  

# In[63]:


# Creating new features  based on previous observations. There might be some highly correlated features now. You cab drop them if you want to...

features['TotalSF'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                       features['1stFlrSF'] + features['2ndFlrSF'])
features['TotalBathrooms'] = (features['FullBath'] +
                              (0.5 * features['HalfBath']) +
                              features['BsmtFullBath'] +
                              (0.5 * features['BsmtHalfBath']))

features['TotalPorchSF'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                            features['EnclosedPorch'] +
                            features['ScreenPorch'] + features['WoodDeckSF'])

features['YearBlRm'] = (features['YearBuilt'] + features['YearRemodAdd'])

# Merging quality and conditions.

features['TotalExtQual'] = (features['ExterQual'] + features['ExterCond'])
features['TotalBsmQual'] = (features['BsmtQual'] + features['BsmtCond'] +
                            features['BsmtFinType1'] +
                            features['BsmtFinType2'])
features['TotalGrgQual'] = (features['GarageQual'] + features['GarageCond'])
features['TotalQual'] = features['OverallQual'] + features[
    'TotalExtQual'] + features['TotalBsmQual'] + features[
        'TotalGrgQual'] + features['KitchenQual'] + features['HeatingQC']

# Creating new features by using new quality indicators.

features['QualGr'] = features['TotalQual'] * features['GrLivArea']
features['QualBsm'] = features['TotalBsmQual'] * (features['BsmtFinSF1'] +
                                                  features['BsmtFinSF2'])
features['QualPorch'] = features['TotalExtQual'] * features['TotalPorchSF']
features['QualExt'] = features['TotalExtQual'] * features['MasVnrArea']
features['QualGrg'] = features['TotalGrgQual'] * features['GarageArea']
features['QlLivArea'] = (features['GrLivArea'] -
                         features['LowQualFinSF']) * (features['TotalQual'])
features['QualSFNg'] = features['QualGr'] * features['Neighborhood']

features["new_home"] = features["YearBuilt"]
features.loc[features["new_home"] == features["YearRemodAdd"], "new_home"] = 0
features.loc[features["new_home"] != features["YearRemodAdd"], "new_home"] = 1

# Creating some simple features.

features['HasPool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['Has2ndFloor'] = features['2ndFlrSF'].apply(lambda x: 1
                                                     if x > 0 else 0)
features['HasGarage'] = features['QualGrg'].apply(lambda x: 1 if x > 0 else 0)
features['HasBsmt'] = features['QualBsm'].apply(lambda x: 1 if x > 0 else 0)
features['HasFireplace'] = features['Fireplaces'].apply(lambda x: 1
                                                        if x > 0 else 0)
features['HasPorch'] = features['QualPorch'].apply(lambda x: 1 if x > 0 else 0)


# In[64]:


# Observing the effects of newly created features on sale price.

def srt_reg(feature):
    merged = features.join(y)
    fig, axes = plt.subplots(5, 3, figsize=(25, 40))
    axes = axes.flatten()

    new_features = [
        'TotalSF', 'TotalBathrooms', 'TotalPorchSF', 'YearBlRm',
        'TotalExtQual', 'TotalBsmQual', 'TotalGrgQual', 'TotalQual', 'QualGr',
        'QualBsm', 'QualPorch', 'QualExt', 'QualGrg', 'QlLivArea', 'QualSFNg'
    ]

    for i, j in zip(new_features, axes):

        sns.regplot(x=i,
                    y=feature,
                    data=merged,
                    ax=j,
                    order=3,
                    ci=None,
                    color='#25B89B',
                    line_kws={'color': 'grey'},
                    scatter_kws={'alpha':0.4})
        j.tick_params(labelrotation=45)
        j.yaxis.set_major_locator(MaxNLocator(nbins=10))

        plt.tight_layout()




# In[65]:


srt_reg('SalePrice')


# ## Box-Cox dönüşümü
# Bazı sayısal değişkenler çok fazla sağa çarpık, bunları biraz düzeltmek için Box-Cox Transformation uyguladım

# In[66]:


skewed = [
    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'LowQualFinSF', 'MiscVal'
]


# In[67]:


# Skewnesslık derecesini bulma

skew_features = np.abs(features[skewed].apply(lambda x: skew(x)).sort_values(
    ascending=False))

# Skewnesslık derecesine göre filtreleme

high_skew = skew_features[skew_features > 0.3]

# Yüksek skewnesslığa sahip olanların indexini alma

skew_index = high_skew.index

# Yüksek skewnessa sahip değişkenlere Box-Cox Transformation uygulanması

for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))


# **Rare Analysis sonrası gözüme çarpan bazı gereksiz değişkenleri atıyorum.**

# In[68]:


# Atılacaklar listesi

to_drop = ['Utilities','PoolQC','YrSold','MoSold','ExterQual','BsmtQual','GarageQual','KitchenQual','HeatingQC',]

features.drop(columns=to_drop, inplace=True)


# In[69]:


# Kategorik değişkenlere O.H.E uyguluyorum.
# Normalde regresyon modellerinde First-Drop = True yapılması gerekiyor fakat, bazı ağaç modelleri de kullanacağım için ilk dummy'leri atmıyorum. 
# Hatta bunun biraz puanımı arttırdığımı da söyleyebilirim.

features = pd.get_dummies(data=features)


# # Son İşlemler

# In[70]:


print(f'Toplam eksik gözlem sayısı: {features.isna().sum().sum()}')


# In[71]:


features.shape


# In[72]:


# train ve test ayırma
submission_model = features.copy()
train = features.iloc[:len(y), :]
test = features.iloc[len(train):, :]


# In[73]:


correlations = train.join(y).corrwith(train.join(y)['SalePrice']).iloc[:-1].to_frame()
correlations['Abs Corr'] = correlations[0].abs()
sorted_correlations = correlations.sort_values('Abs Corr', ascending=False)['Abs Corr']
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(sorted_correlations.to_frame()[sorted_correlations>=.5], cmap='coolwarm', annot=True, vmin=-1, vmax=1, ax=ax);


# In[74]:


def plot_dist3(df, feature, title):
    
    # Creating a customized chart. and giving in figsize and everything.
    
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    
    # creating a grid of 3 cols and 3 rows.
    
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    # Customizing the histogram grid.
    
    ax1 = fig.add_subplot(grid[0, :2])
    
    # Set the title.
    
    ax1.set_title('Histogram')
    
    # plot the histogram.
    
    sns.distplot(df.loc[:, feature],
                 hist=True,
                 kde=True,
                 fit=norm,
                 ax=ax1,
                 color='#e74c3c')
    ax1.legend(labels=['Normal', 'Actual'])

    # customizing the QQ_plot.
    
    ax2 = fig.add_subplot(grid[1, :2])
    
    # Set the title.
    
    ax2.set_title('Probability Plot')
    
    # Plotting the QQ_Plot.
    stats.probplot(df.loc[:, feature].fillna(np.mean(df.loc[:, feature])),
                   plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor('#e74c3c')
    ax2.get_lines()[0].set_markersize(12.0)

    # Customizing the Box Plot:
    
    ax3 = fig.add_subplot(grid[:, 2])
    # Set title.
    
    ax3.set_title('Box Plot')
    
    # Plotting the box plot.
    
    sns.boxplot(df.loc[:, feature], orient='v', ax=ax3, color='#e74c3c')
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=24))

    plt.suptitle(f'{title}', fontsize=24)


# In[75]:


# Checking target variable.

plot_dist3(train.join(y), 'SalePrice', 'Log Dönüşüm Öncesi Sale Price')


# In[76]:


# Setting model data.

X_my = train
X_test_my = test
y_ = y
y_log =y
y_log = np.log1p(y_log)


# In[77]:


plot_dist3(train.join(y_log), 'SalePrice', 'Log Dönüşüm Sonrası Sale Price')


# # MODELLEME

# In[78]:


X_my = RobustScaler().fit_transform(X_my)
X_test_my = RobustScaler().fit_transform(X_test_my)


# In[79]:


X_train, X_test, y_train, y_test = train_test_split(X_my, y_log, test_size=0.20, random_state=46)


# In[80]:


### NON LINEAR MODELS
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor()),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

# Base modellerin test hataları
print("Base Test RMSE")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmsLe = np.sqrt(mean_squared_error(y_test, y_pred))
    msg = "%s: %f" % (name, rmsLe)
    print(msg)


# In[81]:


# Base modellerin test hataları
print("Base Train RMSE")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    rmsLe = np.sqrt(mean_squared_error(y_train, y_pred))
    msg = "%s: %f" % (name, rmsLe)
    print(msg)


# In[82]:


X_train, X_test, y_train, y_test = train_test_split(X_my, y_, test_size=0.20, random_state=46)


# In[83]:


models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor()),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

# Base modellerin test hataları
print("Test RMSE")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmsLe = np.sqrt(mean_squared_error(y_test, y_pred))
    msg = "%s: %f" % (name, rmsLe)
    print(msg)


# In[84]:


# Base modellerin train hataları
print("Train RMSE")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    rmsLe = np.sqrt(mean_squared_error(y_train, y_pred))
    msg = "%s: %f" % (name, rmsLe)
    print(msg)


# **Buraya kadar olan modelleme projem içindi, submisson için blending models uygulayacağım.**

# In[85]:


X = train
X_test = test
y = np.log1p(y)


# In[86]:


# Loading neccesary packages for modelling.

from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, TweedieRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor # This is for stacking part, works well with sklearn and others...


# In[87]:


# Setting kfold for future use.

kf = KFold(10, random_state=42)


# In[88]:


# Some parameters for ridge, lasso and elasticnet.

alphas_alt = [15.5, 15.6, 15.7, 15.8, 15.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [
    5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008
]
e_alphas = [
    0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007
]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

# ridge_cv

ridge = make_pipeline(RobustScaler(), RidgeCV(
    alphas=alphas_alt,
    cv=kf,
))

# lasso_cv:

lasso = make_pipeline(
    RobustScaler(),
    LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kf))

# elasticnet_cv:

elasticnet = make_pipeline(
    RobustScaler(),
    ElasticNetCV(max_iter=1e7,
                 alphas=e_alphas,
                 cv=kf,
                 random_state=42,
                 l1_ratio=e_l1ratio))

# svr:

svr = make_pipeline(RobustScaler(),
                    SVR(C=21, epsilon=0.0099, gamma=0.00017, tol=0.000121))

# gradientboosting:

gbr = GradientBoostingRegressor(n_estimators=2900,
                                learning_rate=0.0161,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=17,
                                loss='huber',
                                random_state=42)

# lightgbm:

lightgbm = LGBMRegressor(objective='regression',
                         n_estimators=3500,
                         num_leaves=5,
                         learning_rate=0.00721,
                         max_bin=163,
                         bagging_fraction=0.35711,
                         n_jobs=-1,
                         bagging_seed=42,
                         feature_fraction_seed=42,
                         bagging_freq=7,
                         feature_fraction=0.1294,
                         min_data_in_leaf=8)

# xgboost:

xgboost = XGBRegressor(
    learning_rate=0.0139,
    n_estimators=4500,
    max_depth=4,
    min_child_weight=0,
    subsample=0.7968,
    colsample_bytree=0.4064,
    nthread=-1,
    scale_pos_weight=2,
    seed=42,
)


# hist gradient boosting regressor:

hgrd= HistGradientBoostingRegressor(    loss= 'least_squares',
    max_depth= 2,
    min_samples_leaf= 40,
    max_leaf_nodes= 29,
    learning_rate= 0.15,
    max_iter= 225,
                                    random_state=42)

# tweedie regressor:
 
tweed = make_pipeline(RobustScaler(),TweedieRegressor(alpha=0.005))


# stacking regressor:

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr,
                                            xgboost, lightgbm,hgrd, tweed),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)


# # Cross Validation

# In[89]:


def model_check(X, y, estimators, cv):
    
    ''' A function for testing multiple estimators.'''
    
    model_table = pd.DataFrame()

    row_index = 0
    for est, label in zip(estimators, labels):

        MLA_name = label
        model_table.loc[row_index, 'Model Name'] = MLA_name

        cv_results = cross_validate(est,
                                    X,
                                    y,
                                    cv=cv,
                                    scoring='neg_root_mean_squared_error',
                                    return_train_score=True,
                                    n_jobs=-1)

        model_table.loc[row_index, 'Train RMSE'] = -cv_results[
            'train_score'].mean()
        model_table.loc[row_index, 'Test RMSE'] = -cv_results[
            'test_score'].mean()
        model_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()
        model_table.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

        row_index += 1

    model_table.sort_values(by=['Test RMSE'],
                            ascending=True,
                            inplace=True)

    return model_table


# In[90]:


# Setting list of estimators and labels for them:

estimators = [ridge, lasso, elasticnet, gbr, xgboost, lightgbm, svr, hgrd, tweed]
labels = [
    'Ridge', 'Lasso', 'Elasticnet', 'GradientBoostingRegressor',
    'XGBRegressor', 'LGBMRegressor', 'SVR', 'HistGradientBoostingRegressor','TweedieRegressor'
]


# # Model Sonuçları
# 

# In[91]:


# Executing cross validation.

raw_models = model_check(X, y, estimators, kf)
display(raw_models.style.background_gradient(cmap='summer'))


# ## Stacking & Blending
# 

# In[92]:


# Fitting the models on train data.

print('=' * 20, 'START Fitting', '=' * 20)
print('=' * 55)

print(datetime.now(), 'StackingCVRegressor')
stack_gen_model = stack_gen.fit(X.values, y.values)
print(datetime.now(), 'Elasticnet')
elastic_model_full_data = elasticnet.fit(X, y)
print(datetime.now(), 'Lasso')
lasso_model_full_data = lasso.fit(X, y)
print(datetime.now(), 'Ridge')
ridge_model_full_data = ridge.fit(X, y)
print(datetime.now(), 'SVR')
svr_model_full_data = svr.fit(X, y)
print(datetime.now(), 'GradientBoosting')
gbr_model_full_data = gbr.fit(X, y)
print(datetime.now(), 'XGboost')
xgb_model_full_data = xgboost.fit(X, y)
print(datetime.now(), 'Lightgbm')
lgb_model_full_data = lightgbm.fit(X, y)
print(datetime.now(), 'Hist')
hist_full_data = hgrd.fit(X, y)
print(datetime.now(), 'Tweed')
tweed_full_data = tweed.fit(X, y)
print('=' * 20, 'FINISHED Fitting', '=' * 20)
print('=' * 58)


# In[93]:


# Blending models by assigning weights:

def blend_models_predict(X):
    return ((0.1 * elastic_model_full_data.predict(X)) +
            (0.1 * lasso_model_full_data.predict(X)) +
            (0.1 * ridge_model_full_data.predict(X)) +
            (0.1 * svr_model_full_data.predict(X)) +
            (0.05 * gbr_model_full_data.predict(X)) +
            (0.1 * xgb_model_full_data.predict(X)) +
            (0.05 * lgb_model_full_data.predict(X)) +
            (0.05 * hist_full_data.predict(X)) +
            (0.1 * tweed_full_data.predict(X)) +
            (0.25 * stack_gen_model.predict(X.values)))


# ## Submission
# 

# In[94]:


submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
# Inversing and flooring log scaled sale price predictions
submission['SalePrice'] = np.floor(np.expm1(blend_models_predict(X_test)))
# Defining outlier quartile ranges
q1 = submission['SalePrice'].quantile(0.0050)
q2 = submission['SalePrice'].quantile(0.99)

# Applying weights to outlier ranges to smooth them
submission['SalePrice'] = submission['SalePrice'].apply(
    lambda x: x if x > q1 else x * 0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x
                                                        if x < q2 else x * 1.1)
submission = submission[['Id', 'SalePrice']]


# In[95]:


# Saving submission file

submission.to_csv('mysubmission.csv', index=False)
print(
    'Save submission',
    datetime.now(),
)
submission.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




