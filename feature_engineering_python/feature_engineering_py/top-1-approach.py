#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Hello there,
# 
# First of all, if you are completely new to data science field I highly recommend checking out [kaggle courses](https://www.kaggle.com/learn/overview) to get started. Furthermore, I'd like to recommend a few amazing kernels about this particular competition:
# 1. https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# 2. https://www.kaggle.com/cheesu/house-prices-1st-approach-to-data-science-process
# 3. https://www.kaggle.com/angqx95/data-science-workflow-top-2-with-tuning
# 4. https://www.kaggle.com/datafan07/top-1-approach-eda-new-models-and-stacking
# 
# These notebooks are amazing, and I learnt a ton from them so hope you will too :)
# 
# In this kernel you will find my approach to this regression problem.
# 
# Here's a table of contents:
# 
# 1. Meeting our data
# 
# 2. Visualization and data analysis
#     
#     2.1 Target variable and numerical data
#     
#     2.2 Categorical data
#     
# 3. Data cleaning
# 
#     3.1 Dealing with null values
#     
#     3.2 Label encoding
#     
#     3.3 Dealing with outliers
#     
# 4. Feature engineering
# 
# 5. Data normalization and one-hot encoding
# 
# 6. Creating and evaluating a model
# 
#     6.1 Parameter tuning
#     
#     6.2 Models evaluations
#     
#     6.3 Model stacking

# # 1. Meeting our data:

# In[1]:


import numpy as np
import pandas as pd

train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col = 'Id')
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col = 'Id')

train.tail(10)


# In[2]:


test.head(10)


# In[3]:


train.dtypes.unique()


# In[4]:


train.select_dtypes(exclude = 'object').describe()


# In[5]:


train.select_dtypes(include = ['object']).describe()


# In[6]:


target = train.SalePrice.copy()
target.describe()


# In[7]:


print('In train data there are: {} categorical features;\n\t\t\t {} numerical features'.format(train.select_dtypes(include = ['object']).columns.size,
                                                                                   train.drop('SalePrice', axis = 1).select_dtypes(exclude = ['object']).columns.size))

print('In test data there are: {} categorical features;\n\t\t\t {} numerical features.'.format(test.select_dtypes(include = ['object']).columns.size,
                                                                                   test.select_dtypes(exclude = ['object']).columns.size))


# In[8]:


(train.drop('SalePrice', axis = 1).columns).equals(test.columns)


# # 2. Visualization and data analysis

# In[9]:


# Setting up Seaborn library
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from scipy import stats


# # 2.1 Target variable and numerical data

# In[10]:


sns.set_style('whitegrid')
# plt.figure(figsize = (16,6))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 6), gridspec_kw={'width_ratios': [3, 2]})

sns.histplot(target, kde = True, color = 'red', stat = 'count', ax = ax1)
ax1.set_title('Histogram of SalePrice', fontsize = 16)
stats.probplot(target, plot = sns.lineplot(ax = ax2))
ax2.set_title('Probability Plot of SalePrice', fontsize = 16)
ax2.get_lines()[0].set_color('red')
ax2.get_lines()[1].set_color('black')


# In[11]:


def plot_grid(data, fig_size, grid_size, plot_type, target = ''):
    """
    Custom function for plotting grid of plots.
    It takes: DataFrame of data, size of a grid, type of plots, string name of target variable;
    And it outputs: grid of plots.
    """
    fig = plt.figure(figsize = fig_size)
    if plot_type == 'histplot':
        for i, column_name in enumerate(data.select_dtypes(exclude = 'object').columns):
            fig.add_subplot(grid_size[0], grid_size[1], i + 1)
            plot = sns.histplot(data[column_name], kde = True, color = 'red', stat = 'count')
            plot.set_xlabel(column_name, fontsize = 16)
    if plot_type == 'boxplot':
        for i, column_name in enumerate(data.select_dtypes(exclude = 'object').columns):
            fig.add_subplot(grid_size[0], grid_size[1], i + 1)
            plot = sns.boxplot(x = data[column_name], color = 'red')
            plot.set_xlabel(column_name, fontsize = 16)
    if plot_type == 'scatterplot':
        for i, column_name in enumerate(data.drop(target, axis = 1).select_dtypes(exclude = 'object').columns):
            fig.add_subplot(grid_size[0], grid_size[1], i + 1)
            plot = sns.scatterplot(x = data[column_name], y = data[target], color = 'red')
            plot.set_xlabel(column_name, fontsize = 16)
    if plot_type == 'boxplot_cat':
        for i, column_name in enumerate(data.select_dtypes(include = 'object').columns):
            fig.add_subplot(grid_size[0], grid_size[1], i + 1)
            sort = data.groupby([column_name])[target].median().sort_values(ascending = False) # This is here to make sure boxes are sorted by median
            plot = sns.boxplot(x = data[column_name], y = data[target], order = sort.index, palette = 'Reds')
            plot.set_xlabel(column_name, fontsize = 16)
    plt.tight_layout()


# In[12]:


# numerical_data = train.drop('SalePrice', axis = 1).select_dtypes(exclude = 'object')
    
plot_grid(train.drop('SalePrice', axis = 1), fig_size = (20, 40), grid_size = (12, 3), plot_type = 'histplot')


# In[13]:


correlation = train.corr()
plt.figure(figsize = (20,10))
sns.heatmap(correlation.loc[::-1,::-1], 
            square = True, 
            vmax = 0.8,)


# In[14]:


# Heatmap of numerical features correlation with target sorted by value of correlation coefficient in descending order
plt.figure(figsize = (40,20))
sns.heatmap(correlation.sort_values(by = 'SalePrice', axis = 0, ascending = False).iloc[:,-1:], 
            square = True, 
            annot = True, 
            fmt = '.2f', 
            cbar = False,)


# In[15]:


# Heatmap for first n numerical features that correlate with target the most 
n = 20
plt.figure(figsize = (32,10))
sns.heatmap(train[correlation.nlargest(n, 'SalePrice').index].corr(), 
            annot = True, 
            fmt = '.2f', 
            square = True, 
            cbar = False,)


# In[16]:


plot_grid(train, fig_size = (20, 40), grid_size = (12, 3), plot_type = 'scatterplot', target = 'SalePrice')


# In[17]:


plot_grid(train.drop('SalePrice', axis = 1), fig_size = (20, 40), grid_size = (12, 3), plot_type = 'boxplot')


# # 2.2 Categorical data

# In[18]:


train.select_dtypes(include = 'object').nunique().sort_values(ascending = False)


# Plotting categorical features sorted by cardinality in descending order.

# In[19]:


plot_grid(pd.concat([train[list(train.select_dtypes(include = 'object').nunique().sort_values(ascending = False).index)], 
                     target], axis = 1), 
          fig_size = (20, 40), grid_size = (15, 3), plot_type = 'boxplot_cat', target = 'SalePrice')


# In[20]:


plot_grid(train[['Neighborhood', 'Exterior2nd', 'Exterior1st', 'SalePrice']], 
          fig_size = (20, 40), grid_size = (3, 1), 
          plot_type = 'boxplot_cat', target = 'SalePrice')


# # 3. Data cleaning

# # 3.1 Dealing with null values

# In[21]:


train_cleaning = train.drop('SalePrice', axis = 1).copy()
test_cleaning = test.copy()
train_test = pd.concat([train_cleaning, test_cleaning])
missing_values = pd.concat([train_test.isnull().sum().sort_values(ascending = False),
                            train_test.isnull().sum().sort_values(ascending = False).apply(lambda x: (x / train_test.shape[0]) * 100)],
                            axis = 1, keys = ['Values missing', 'Percent of missing'])
missing_values[missing_values['Values missing'] > 0].style.background_gradient('Reds')


# In[22]:


replace_zero = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1', 'GarageCars', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea', 'BsmtFinSF2']
replace_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageType', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType', 'Exterior2nd', 'Exterior1st']
replace_mode = ['Functional', 'Utilities', 'KitchenQual', 'SaleType', 'Electrical']

# Replace null values in MSZoning according to MSSubClass
# train_cleaning.MSZoning = train_cleaning.groupby('MSSubClass')['MSZoning'].apply(lambda x: x.fillna(x.mode()[0]))

train_cleaning[replace_zero] = train_cleaning[replace_zero].fillna(0)

train_cleaning[replace_none] = train_cleaning[replace_none].fillna('None')

for col_name in replace_mode:
    train_cleaning[col_name].replace(np.nan, train_cleaning[col_name].mode()[0], inplace = True)

# Replace null values in test data separately from train data  
test_cleaning.MSZoning = test_cleaning.groupby('MSSubClass')['MSZoning'].apply(lambda x: x.fillna(x.mode()[0]))

test_cleaning[replace_zero] = test_cleaning[replace_zero].fillna(0)

test_cleaning[replace_none] = test_cleaning[replace_none].fillna('None')

for col_name in replace_mode:
    test_cleaning[col_name].replace(np.nan, train_cleaning[col_name].mode()[0], inplace = True)


# In[23]:


train_cleaning.isnull().sum().max()


# In[24]:


test_cleaning.isnull().sum().max()


# # 3.2 Label encoding

# Label encoding three categorical features with high cardinality.

# In[25]:


# Converting some of the categorical values to numeric ones. Choosing similar values for closer groups to balance linear relations
neigh_map = {'MeadowV': 1, 
             'IDOTRR': 1, 
             'BrDale': 1,
             'BrkSide': 2,
             'OldTown': 2,
             'Edwards': 2,
             'Sawyer': 3,
             'Blueste': 3,
             'SWISU': 3,
             'NPkVill': 3,
             'NAmes': 3,
             'Mitchel': 4,
             'SawyerW': 5,
             'NWAmes': 5,
             'Gilbert': 5,
             'Blmngtn': 5,
             'CollgCr': 5,
             'ClearCr': 6,
             'Crawfor': 6,
             'Veenker': 7,
             'Somerst': 7,
             'Timber': 8,
             'StoneBr': 9,
             'NridgHt': 10,
             'NoRidge': 10}
train_cleaning['Neighborhood'] = train_cleaning['Neighborhood'].map(neigh_map).astype('int')
test_cleaning['Neighborhood'] = test_cleaning['Neighborhood'].map(neigh_map).astype('int')

# Replacing misspelled values
test_cleaning['Exterior2nd'] = test_cleaning['Exterior2nd'].apply(lambda x: 'BrkComm' if (x == 'Brk Cmn') else 'CemntBd' if (x == 'CmentBd') else x)
train_cleaning['Exterior2nd'] = train_cleaning['Exterior2nd'].apply(lambda x: 'BrkComm' if (x == 'Brk Cmn') else 'CemntBd' if (x == 'CmentBd') else x)
# Creating new simple feature
train_cleaning['ExteriorSame'] = (train_cleaning['Exterior1st'] == train_cleaning['Exterior2nd']).apply(lambda x: 1 if x == True else 0)
test_cleaning['ExteriorSame'] = (test_cleaning['Exterior1st'] == test_cleaning['Exterior2nd']).apply(lambda x: 1 if x == True else 0)

ext1_map = {'None': 0, 
            'BrkComm': 1, 
            'AsphShn': 2,
            'CBlock': 2,
            'AsbShng': 3,
            'WdShing': 4,
            'Wd Sdng': 5,
            'MetalSd': 5,
            'Stucco': 6,
            'HdBoard': 7,
            'BrkFace': 8,
            'Plywood': 8,
            'VinylSd': 9,
            'CemntBd': 10,
            'Stone': 11,
            'ImStucc': 12}
train_cleaning['Exterior1st'] = train_cleaning['Exterior1st'].map(ext1_map).astype('int')
test_cleaning['Exterior1st'] = test_cleaning['Exterior1st'].map(ext1_map).astype('int')

ext2_map = {'None': 0, 
            'BrkComm': 4, 
            'AsphShn': 3,
            'CBlock': 1,
            'AsbShng': 2,
            'WdShing': 4,
            'Wd Sdng': 3,
            'Wd Shng': 3,
            'MetalSd': 3,
            'Stucco': 4,
            'HdBoard': 5,
            'BrkFace': 6,
            'Plywood': 6,
            'VinylSd': 9,
            'CemntBd': 10,
            'Stone': 7,
            'ImStucc': 8,
            'Other': 11}
train_cleaning['Exterior2nd'] = train_cleaning['Exterior2nd'].map(ext2_map).astype('int')
test_cleaning['Exterior2nd'] = test_cleaning['Exterior2nd'].map(ext2_map).astype('int')


# Label encoding other features where it's appropriate. (you can check it by looking into dataset documentation)

# In[26]:


qual_map = {'None': 0, 
            'Po': 1, 
            'Fa': 2, 
            'TA': 3, 
            'Gd': 4, 
            'Ex': 5}
train_cleaning['ExterQual'] = train_cleaning['ExterQual'].map(qual_map).astype('int')
test_cleaning['ExterQual'] = test_cleaning['ExterQual'].map(qual_map).astype('int')

train_cleaning['ExterCond'] = train_cleaning['ExterCond'].map(qual_map).astype('int')
test_cleaning['ExterCond'] = test_cleaning['ExterCond'].map(qual_map).astype('int')

train_cleaning['BsmtQual'] = train_cleaning['BsmtQual'].map(qual_map).astype('int')
test_cleaning['BsmtQual'] = test_cleaning['BsmtQual'].map(qual_map).astype('int')

train_cleaning['BsmtCond'] = train_cleaning['BsmtCond'].map(qual_map).astype('int')
test_cleaning['BsmtCond'] = test_cleaning['BsmtCond'].map(qual_map).astype('int')

train_cleaning['HeatingQC'] = train_cleaning['HeatingQC'].map(qual_map).astype('int')
test_cleaning['HeatingQC'] = test_cleaning['HeatingQC'].map(qual_map).astype('int')

train_cleaning['KitchenQual'] = train_cleaning['KitchenQual'].map(qual_map).astype('int')
test_cleaning['KitchenQual'] = test_cleaning['KitchenQual'].map(qual_map).astype('int')

train_cleaning['FireplaceQu'] = train_cleaning['FireplaceQu'].map(qual_map).astype('int')
test_cleaning['FireplaceQu'] = test_cleaning['FireplaceQu'].map(qual_map).astype('int')

train_cleaning['GarageQual'] = train_cleaning['GarageQual'].map(qual_map).astype('int')
test_cleaning['GarageQual'] = test_cleaning['GarageQual'].map(qual_map).astype('int')

train_cleaning['GarageCond'] = train_cleaning['GarageCond'].map(qual_map).astype('int')
test_cleaning['GarageCond'] = test_cleaning['GarageCond'].map(qual_map).astype('int')

bsmtexposure_map = {'None': 0, 
                    'No': 1, 
                    'Mn': 2, 
                    'Av': 3, 
                    'Gd': 4}
train_cleaning['BsmtExposure'] = train_cleaning['BsmtExposure'].map(bsmtexposure_map).astype('int')
test_cleaning['BsmtExposure'] = test_cleaning['BsmtExposure'].map(bsmtexposure_map).astype('int')

fence_map = {'None': 0, 
             'MnWw': 1, 
             'GdWo': 2, 
             'MnPrv': 3, 
             'GdPrv': 4}
train_cleaning['Fence'] = train_cleaning['Fence'].map(fence_map).astype('int')
test_cleaning['Fence'] = test_cleaning['Fence'].map(fence_map).astype('int')

bsmf_map = {'None': 0,
            'Unf': 1,
            'LwQ': 2,
            'Rec': 3,
            'BLQ': 4,
            'ALQ': 5,
            'GLQ': 6}
train_cleaning['BsmtFinType1'] = train_cleaning['BsmtFinType1'].map(bsmf_map).astype('int')
test_cleaning['BsmtFinType1'] = test_cleaning['BsmtFinType1'].map(bsmf_map).astype('int')
train_cleaning['BsmtFinType2'] = train_cleaning['BsmtFinType2'].map(bsmf_map).astype('int')
test_cleaning['BsmtFinType2'] = test_cleaning['BsmtFinType2'].map(bsmf_map).astype('int')

garagef_map = {'None': 0,
               'Unf': 1,
               'RFn': 2,
               'Fin': 3}
train_cleaning['GarageFinish'] = train_cleaning['GarageFinish'].map(garagef_map).astype('int')
test_cleaning['GarageFinish'] = test_cleaning['GarageFinish'].map(garagef_map).astype('int')

poolqc_map = {'None': 0, 
              'Fa': 2, 
              'TA': 3, 
              'Gd': 4, 
              'Ex': 5}
train_cleaning['PoolQC'] = train_cleaning['PoolQC'].map(poolqc_map).astype('int')
test_cleaning['PoolQC'] = test_cleaning['PoolQC'].map(poolqc_map).astype('int')

str_all_map = {'None': 0, 
               'Grvl': 1, 
               'Pave': 2}
train_cleaning['Street'] = train_cleaning['Street'].map(str_all_map).astype('int')
test_cleaning['Street'] = test_cleaning['Street'].map(str_all_map).astype('int')
train_cleaning['Alley'] = train_cleaning['Alley'].map(str_all_map).astype('int')
test_cleaning['Alley'] = test_cleaning['Alley'].map(str_all_map).astype('int')

cent_air_map = {'N': 0, 
                'Y': 1}
train_cleaning['CentralAir'] = train_cleaning['CentralAir'].map(cent_air_map).astype('int')
test_cleaning['CentralAir'] = test_cleaning['CentralAir'].map(cent_air_map).astype('int')

pave_drive_map = {'N': 0, 
                  'P': 1,
                  'Y': 2}
train_cleaning['PavedDrive'] = train_cleaning['PavedDrive'].map(pave_drive_map).astype('int')
test_cleaning['PavedDrive'] = test_cleaning['PavedDrive'].map(pave_drive_map).astype('int')


# In[27]:


train_cleaning.select_dtypes(include = 'object').nunique().sort_values(ascending = False)


# # 3.3 Dealing with outliers

# In[28]:


def get_outliers(X_y, cols):
    """
    Custom function for dealing with outliers.
    It takes: DataFrame of data, list of columns;
    And it returns: list of unique indexes of outliers.(Also it outputs all outliers with indexes for each column)
    (value is considered an outlier if absolute value of its z-score is > 3)
    """
    outliers_index = []
    for col in cols:
        right_outliers = X_y[col][(X_y[col] - X_y[col].mean()) / X_y[col].std() > 3]
        left_outliers = X_y[col][(X_y[col] - X_y[col].mean()) / X_y[col].std() < -3]
        all_outliers = right_outliers.append(left_outliers)
        outliers_index += (list(all_outliers.index))
        print('{} right outliers:\n{} \n {} left outliers:\n{} \n {} has TOTAL {} rows of outliers\n'.format(col, right_outliers, col, left_outliers, col, all_outliers.count()))
    outliers_index = list(set(outliers_index)) # Removing duplicates
    print('There are {} unique rows with outliers in dataset'.format(len(outliers_index)))
    return outliers_index


# In[29]:


cols = ['GrLivArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd']
X_y = pd.concat([train_cleaning, target], axis = 1)
outliers_index = get_outliers(X_y, cols)
X_y = X_y.drop(outliers_index, axis = 0)

train_cleaning = X_y.drop('SalePrice', axis = 1).copy()
target_cleaned = X_y.SalePrice


# In[30]:


train_cleaning


# # 4. Feature engineering

# In[31]:


train_test = pd.concat([train_cleaning, test_cleaning], keys = ['train', 'test'], axis = 0)

train_test['TotalPorchSF'] = (train_test['OpenPorchSF'] + train_test['3SsnPorch'] + 
                              train_test['EnclosedPorch'] + train_test['ScreenPorch'] + train_test['WoodDeckSF'])

train_test['TotalSF'] = (train_test['BsmtFinSF1'] + train_test['BsmtFinSF2'] + 
                         train_test['1stFlrSF'] + train_test['2ndFlrSF'] + 
                         train_test['TotalPorchSF'])

train_test['TotalBathrooms'] = (train_test['FullBath'] + (0.5 * train_test['HalfBath']) + 
                                train_test['BsmtFullBath'] + (0.5 * train_test['BsmtHalfBath']))

train_test['TotalRms'] = (train_test['TotRmsAbvGrd'] + train_test['TotalBathrooms'])

train_test['YearSold'] = ((train_test['MoSold'] / 12) + train_test['YrSold']).astype('int')

train_test['YearsAfterB'] = (train_test['YearSold'] - train_test['YearBuilt'])

train_test['YearsAfterR'] = (train_test['YearSold'] - train_test['YearRemodAdd'])
    
# Merging quality and conditions

train_test['TotalExtQual'] = (train_test['ExterQual'] + train_test['ExterCond'])

train_test['TotalBsmtQual'] = (train_test['BsmtQual'] + train_test['BsmtCond'] + 
                               train_test['BsmtFinType1'] + train_test['BsmtFinType2'] + train_test['BsmtExposure'])

train_test['TotalGrgQual'] = (train_test['GarageQual'] + train_test['GarageCond'] + train_test['GarageFinish'])

# train_test['TotalPaved'] = (train_test['Street'] + train_test['Alley'] + train_test['PavedDrive'])

train_test['TotalQual'] = (train_test['OverallQual'] + train_test['OverallCond'] + 
                           train_test['TotalExtQual'] + train_test['TotalBsmtQual'] + 
                           train_test['TotalGrgQual'] + train_test['KitchenQual'] + train_test['HeatingQC'] + 
                           train_test['FireplaceQu'] + train_test['PoolQC'] + train_test['Fence'] + 
                           train_test['CentralAir'])

# Creating new features by using new quality indicators

train_test['QualGr'] = train_test['TotalQual'] * train_test['GrLivArea']

train_test['QualBsm'] = train_test['TotalBsmtQual'] * (train_test['BsmtFinSF1'] + train_test['BsmtFinSF2'])

train_test['QualPorch'] = train_test['TotalExtQual'] * train_test['TotalPorchSF']

train_test['QualExt'] = (train_test['TotalExtQual'] * 
                        (train_test['Exterior1st'] + train_test['Exterior2nd']) * train_test['MasVnrArea'])

train_test['QualGrg'] = train_test['TotalGrgQual'] * train_test['GarageArea']

train_test['QualFirepl'] = train_test['FireplaceQu'] * train_test['Fireplaces']

train_test['QlLivArea'] = (train_test['GrLivArea'] - train_test['LowQualFinSF']) * (train_test['TotalQual'])

train_test['QualSFNg'] = train_test['QualGr'] * train_test['Neighborhood']

train_test['QualSF'] = train_test['TotalQual'] * train_test['TotalSF']
train_test['QlSF'] = (train_test['TotalSF'] - train_test['LowQualFinSF']) * (train_test['TotalQual'])
train_test['QualSFNg2'] = (train_test['QualGr'] + train_test['QualSF']) * train_test['Neighborhood']
train_test['QualGrgNg'] = train_test['QualGrg'] * train_test['Neighborhood']


# In[32]:


# Creating new simple features

train_test['HasPool'] = train_test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
train_test['Has2ndFloor'] = train_test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
train_test['HasGarage'] = train_test['QualGrg'].apply(lambda x: 1 if x > 0 else 0)
train_test['HasBsmt'] = train_test['QualBsm'].apply(lambda x: 1 if x > 0 else 0)
train_test['HasFireplace'] = train_test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
train_test['HasPorch'] = train_test['QualPorch'].apply(lambda x: 1 if x > 0 else 0)
train_test['HasLotFr'] = train_test['LotFrontage'].apply(lambda x: 1 if x > 0 else 0)
train_test['HasFence'] = train_test['Fence'].apply(lambda x: 1 if x > 0 else 0)
train_test['WasRemod'] = (train_test['YearRemodAdd'] != train_test['YearBuilt']).apply(lambda x: 1 if x == True else 0)


# Dropping all of the features, I found out to be useless during exploratory data analysis.

# In[33]:


to_drop = [
    'Utilities',
    'PoolQC',
    'YrSold',
    'MoSold',
    'ExterQual',
    'BsmtFinType2',
    'BsmtQual',
    'GarageQual',
    'GarageFinish',
    'KitchenQual',
    'HeatingQC',
    'FireplaceQu',
    'YearSold',
    'MiscVal',
    'MiscFeature',
    'Alley',
    'PoolArea',
    'LowQualFinSF',
]
train_test.drop(columns = to_drop, inplace=True)


# In[34]:


# Visualizing new features
train_cleaned = train_test.xs('train').copy()

plot_grid(pd.concat([train_cleaned[['TotalPorchSF',
                                    'TotalSF',
                                    'TotalBathrooms',
                                    'TotalRms',
                                    'YearsAfterB',
                                    'YearsAfterR',
                                    'TotalExtQual',
                                    'TotalBsmtQual',
                                    'TotalGrgQual',
                                    'TotalQual',
                                    'QualGr',
                                    'QualBsm',
                                    'QualPorch',
                                    'QualExt',
                                    'QualGrg',
                                    'QualFirepl',
                                    'QlLivArea',
                                    'QualSFNg',
                                    'QualSF',
                                    'QlSF',
                                    'QualSFNg2',
                                    'QualGrgNg']], target_cleaned], axis = 1), 
          fig_size = (20, 40), grid_size = (8, 3), plot_type = 'scatterplot', target = 'SalePrice')


# In[35]:


fig = plt.figure(figsize = (32,16))
for i, col_name in enumerate(['HasPool',
                              'Has2ndFloor',
                              'HasGarage',
                              'HasBsmt',
                              'HasFireplace',
                              'HasPorch',
                              'HasLotFr',
                              'HasFence',
                              'WasRemod']):
    fig.add_subplot(5, 2, i + 1)
    plot = sns.boxplot(x = train_cleaned[col_name], y = target_cleaned, palette = 'Reds')
    plot.set_xlabel(col_name, fontsize = 16)
plt.tight_layout()


# In[36]:


# Visualizing all numeric
plot_grid(train_cleaned, fig_size = (20, 60), grid_size = (25, 3), plot_type = 'histplot')


# # 5. Data normalization and one-hot encoding

# In[37]:


from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p

skewed = [
    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'Fence', 'TotalSF', 'TotalRms', 'TotalQual', 'TotalPorchSF',
    'TotalBsmtQual', 'TotalGrgQual', 'QualPorch', 'QualFirepl', 'QualGr', 
    'QualGrg', 'QlLivArea', 'QualSFNg', 'QualExt',
    'QualSF', 'QlSF', 'QualSFNg2', 'QualGrgNg', 'ExterCond', 
    'BsmtFinType1', 'BsmtCond', 'BsmtExposure', 'GarageCond',
]

# Finding skewness of the numerical features.
skew_train = np.abs(train_cleaned[skewed].apply(lambda x: skew(x))).sort_values(ascending = False)

# Filtering skewed features.
high_skew_train = skew_train[skew_train > 0.3]

# Taking column names of high skew.
skew_columns_train = high_skew_train.index

test_cleaned = train_test.xs('test').copy()

# Applying boxcox transformation to fix skewness.
for i in skew_columns_train:
    lamb = boxcox_normmax(train_cleaned[i] + 1)
    train_cleaned[i] = boxcox1p(train_cleaned[i], lamb)
    test_cleaned[i] = boxcox1p(test_cleaned[i], lamb)
    
high_skew_train


# In[38]:


skew_train = np.abs(train_cleaned[skewed].apply(lambda x: skew(x))).sort_values(ascending = False)
high_skew_train = skew_train[skew_train > 0.3]
high_skew_train


# In[39]:


plot_grid(train_cleaned[skewed], fig_size = (20, 40), grid_size = (14, 3), plot_type = 'histplot')


# In[40]:


# categorical_features = [col_name for col_name in train_test.columns 
#                         if ((train_test[col_name].dtype == 'object' and train_test[col_name].nunique() < 10) 
#                             or (train_test[col_name].dtype in ['int64', 'float64']))]
train_test_cleaned = pd.concat([train_cleaned, test_cleaned], keys = ['train', 'test'], axis = 0)
train_test = pd.get_dummies(train_test_cleaned)

# for col_name in train_test.columns:
#     train_test[col_name] = (train_test[col_name] - train_test[col_name].mean()) / train_test[col_name].std()


# In[41]:


# from mlxtend.preprocessing import minmax_scaling

# train_test = minmax_scaling(train_test, columns = train_test.columns)

X_train_full, X_test = train_test.xs('train'), train_test.xs('test')


# In[42]:


X_train_full


# In[43]:


y_train_full = np.log1p(target_cleaned)
y_train_full


# Comparing distributions of a log transformed target variable and just target variable.

# In[44]:


fig, axs = plt.subplots(2, 2, figsize = (24, 12), gridspec_kw={'width_ratios': [3, 2]})

sns.histplot(y_train_full, kde = True, color = 'red', stat = 'count', ax = axs[0, 0])
axs[0, 0].set_title('Histogram of log transfomed SalePrice', fontsize = 16)
stats.probplot(y_train_full, plot = sns.lineplot(ax = axs[0, 1]))
axs[0, 1].set_title('Probability Plot of log transfomed SalePrice', fontsize = 16)
axs[0, 1].get_lines()[0].set_color('red')
axs[0, 1].get_lines()[1].set_color('black')

sns.histplot(target, kde = True, color = 'red', stat = 'count', ax = axs[1, 0])
axs[1, 0].set_title('Histogram of SalePrice', fontsize = 16)
stats.probplot(target, plot = sns.lineplot(ax = axs[1, 1]))
axs[1, 1].set_title('Probability Plot of SalePrice', fontsize = 16)
axs[1, 1].get_lines()[0].set_color('red')
axs[1, 1].get_lines()[1].set_color('black')

fig.tight_layout()


# # 6. Creating and evaluating a model

# In[45]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, ElasticNetCV

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# def scorer(y, y_pred):
#     return mean_absolute_error(np.expm1(y_pred), np.expm1(y))

model = LinearRegression()
scores = cross_val_score(model, X_train_full, y_train_full, scoring = 'neg_root_mean_squared_error', cv = 10)

print('Mean of RMSE values from 10-fold cross validation of LinearRegression model: {}'.format(-scores.mean()))

model = Lasso()
scores = cross_val_score(model, X_train_full, y_train_full, scoring = 'neg_root_mean_squared_error', cv = 10)

print('Mean of RMSE values from 10-fold cross validation of Lasso model: {}'.format(-scores.mean()))

model = Ridge()
scores = cross_val_score(model, X_train_full, y_train_full, scoring = 'neg_root_mean_squared_error', cv = 10)

print('Mean of RMSE values from 10-fold cross validation of Ridge model: {}'.format(-scores.mean()))

model = LGBMRegressor()
scores = cross_val_score(model, X_train_full, y_train_full, scoring = 'neg_root_mean_squared_error', cv = 10)

print('Mean of RMSE values from 10-fold cross validation of LGBM model: {}'.format(-scores.mean()))

# from sklearn.metrics import SCORERS # This two lines were written to see a list of possible strings for scoring
# SCORERS.keys()


# # 6.1 Parameter tuning

# In[46]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# def scorer(y, y_pred):
#     return mean_absolute_error(np.expm1(y_pred), np.expm1(y))
def get_best_parameters(model, parameters, cv, search):
    if (search == 'grid'):
        grid = GridSearchCV(model, 
                            parameters,
                            cv = cv, 
                            scoring = 'neg_root_mean_squared_error',
                            n_jobs = -1)
    
    elif (search == 'randomized'):
        grid = RandomizedSearchCV(model,
                                  param_distributions = parameters,
                                  n_iter = 100,
                                  cv = cv, 
                                  scoring = 'neg_root_mean_squared_error',
                                  n_jobs = -1)
    
    grid.fit(X_train_full, y_train_full)
    return str(grid.best_params_)


# In[47]:


ridge_params = {
    'alpha': [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 
              6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5]
}

lasso_params = {
    'alpha': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 
              0.0006, 0.0007, 0.0008, 0.0009]
}

elasticnet_params = {
    'alpha' : [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007],
    'l1_ratio' : [0, 0.5, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 
                  0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1]
}

xgboost_params = {
    'learning_rate' : [0.01, 0.1, 0.15, 0.3, 0.5],
    'n_estimators' : [100, 500, 1000, 2000, 3000],
    'max_depth' : [3, 6, 9],
    'min_child_weight' : [1, 5, 10, 20],
    'reg_alpha' : [0.001, 0.01, 0.1],
    'reg_lambda' : [0.001, 0.01, 0.1]
}

lightgbm_params = {
    'max_depth' : [2, 5, 8, 10],
    'learning_rate' : [0.001, 0.01, 0.1, 0.2],
    'n_estimators' : [100, 300, 500, 1000, 1500],
    'lambda_l1' : [0.0001, 0.001, 0.01],
    'lambda_l2' : [0, 0.0001, 0.001, 0.01],
    'feature_fraction' : [0.4, 0.6, 0.8],
    'min_child_samples' : [5, 10, 20, 25]
}

gbr_params = {
    'learning_rate' : [0.01, 0.1, 0.15, 0.3, 0.5],
    'n_estimators' : [500, 1000, 1500, 2000, 2500, 3000, 3500],
    'max_depth' : [3, 6, 9]
}

cbr_params = {
    'n_estimators' : [100, 300, 500, 1000, 1300, 1600],
    'learning_rate' : [0.0001, 0.001, 0.01, 0.1],
    'l2_leaf_reg' : [0.001, 0.01, 0.1],
    'random_strength' : [0.25, 0.5 ,1],
    'max_depth' : [3, 6, 9],
    'min_child_samples' : [2, 5, 10, 15, 20],
    'rsm' : [0.5, 0.7, 0.9],
}

svr_params = {
    'svr__C' : [10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16,],
    'svr__gamma' : [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007],
}

ridge = Ridge() 
lasso = Lasso() 
elasticnet = ElasticNet()
xgboost = XGBRegressor(booster = 'gbtree', objective = 'reg:squarederror')
lightgbm = LGBMRegressor(boosting_type = 'gbdt',objective = 'regression')
gbr = GradientBoostingRegressor()
cbr = CatBoostRegressor(loss_function = 'RMSE', allow_writing_files = False, logging_level='Silent')
svr = make_pipeline(StandardScaler(), SVR())

estimators = [ridge,
              lasso,
              elasticnet,]
#               svr, 
#               lightgbm, 
#               gbr, 
#               cbr, 
#               xgboost]
labels = ['Ridge',
          'Lasso',
          'Elasticnet',]
#           'SVR',
#           'LightGBM', 
#           'GBR', 
#           'CBR']
#           'XGBoost']
estimators_params = [ridge_params,
                     lasso_params,
                     elasticnet_params,]
#                      svr_params
#                      lightgbm_params, 
#                      gbr_params,
#                      cbr_params]
#                      xgboost_params]


# Finding the best parameters for all of these models using GridSearchCV and RandomizedSearchCV is time consuming, especially gradient boosting models. (in previous version of this kernel during commit it took almost 7 hours to compute parameters for all of these models excluding SVR and XGBoost)
# 
# If you want to try it yourself, then don't forget that Kaggle kernel stops working automatically after 9 hours of a session and that they are running on cloud machines so computation abilities are limited. It's better to download jupyter notebook and try it on your own computer. (also don't forget to set n_jobs parameter to -1 in both GridSearchCV and RandomizedSearchCV)
# 
# So here, for the sake of saving some time, I'm gonna find parameters only for 'Ridge', 'Lasso' and 'Elasticnet' models.
# For other models I will use parameters from this great kernel:
# https://www.kaggle.com/datafan07/top-1-approach-eda-new-models-and-stacking

# In[48]:


best_parameters = pd.DataFrame(columns = ['Model name', 'Best parameters'])

for i in range(len(estimators)):
    best_parameters.loc[i, 'Model name'] = labels[i]
    if (labels[i] in ['XGBoost', 'LightGBM', 'GBR', 'CBR']):
        best_parameters.loc[i, 'Best parameters'] = get_best_parameters(estimators[i], 
                                                                    estimators_params[i], 
                                                                    cv = 10, 
                                                                    search = 'randomized')
    else:
        best_parameters.loc[i, 'Best parameters'] = get_best_parameters(estimators[i], 
                                                                    estimators_params[i], 
                                                                    cv = 10, 
                                                                    search = 'grid')


# # 6.2 Models evaluations

# In[49]:


from sklearn.model_selection import cross_validate

def test_estimators(X, y, estimators, labels, cv):
    ''' 
    A function for testing multiple estimators.
    It takes: full train data and target, list of estimators, 
              list of labels or names of estimators,
              cross validation splitting strategy;
    And it returns: a DataFrame of table with results of tests
    '''
    result_table = pd.DataFrame()

    row_index = 0
    for est, label in zip(estimators, labels):

        est_name = label
        result_table.loc[row_index, 'Model Name'] = est_name

        cv_results = cross_validate(est,
                                    X,
                                    y,
                                    cv = cv,
                                    scoring = 'neg_root_mean_squared_error',
#                                     return_train_score = True,
                                    n_jobs = -1)

#         result_table.loc[row_index, 'Train RMSE'] = -cv_results['train_score'].mean()
        result_table.loc[row_index, 'Test RMSE'] = -cv_results['test_score'].mean()
        result_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()
        result_table.loc[row_index, 'Fit Time'] = cv_results['fit_time'].mean()

        row_index += 1

    result_table.sort_values(by=['Test RMSE'], ascending = True, inplace = True)

    return result_table


# In[50]:


from ast import literal_eval # To convert string to dictionary

linear = LinearRegression()
ridge = Ridge(**literal_eval(best_parameters.loc[0, 'Best parameters']))
lasso = Lasso(**literal_eval(best_parameters.loc[1, 'Best parameters']) )
elasticnet = ElasticNet(**literal_eval(best_parameters.loc[2, 'Best parameters']))

svr = make_pipeline(StandardScaler(), SVR(C = 21,
                                          epsilon = 0.0099, 
                                          gamma = 0.00017, 
                                          tol = 0.000121))

lightgbm = LGBMRegressor(objective = 'regression',
                         n_estimators = 3500,
                         num_leaves = 5,
                         learning_rate = 0.00721,
                         max_bin = 163,
                         bagging_fraction = 0.35711,
                         n_jobs = -1,
                         bagging_seed = 42,
                         feature_fraction_seed = 42,
                         bagging_freq = 7,
                         feature_fraction = 0.1294,
                         min_data_in_leaf = 8)

gbr = GradientBoostingRegressor(n_estimators = 2900,
                                learning_rate = 0.0161,
                                max_depth = 4,
                                max_features = 'sqrt',
                                min_samples_leaf = 17,
                                loss = 'huber',
                                random_state = 42)

cbr = CatBoostRegressor(loss_function = 'RMSE', 
                        allow_writing_files = False, 
                        logging_level='Silent')

xgboost = XGBRegressor(learning_rate = 0.0139,
                       n_estimators = 4500,
                       max_depth = 4,
                       min_child_weight = 0,
                       subsample = 0.7968,
                       colsample_bytree = 0.4064,
                       nthread = -1,
                       scale_pos_weight = 2,
                       seed = 42,)

estimators = [linear,
              ridge, 
              lasso, 
              elasticnet, 
              svr,
              lightgbm, 
              gbr, 
              cbr,] 
#               xgboost]

labels = ['Linear',
          'Ridge', 
          'Lasso', 
          'Elasticnet',
          'SVR', 
          'LightGBM', 
          'GBR', 
          'CBR',]
#           'XGBoost']

results = test_estimators(X_train_full, y_train_full, estimators, labels, cv = 10)
results.style.background_gradient(cmap = 'Reds')


# # 6.3 Model stacking

# In[51]:


from sklearn.ensemble import StackingRegressor

estimators = [
    ('1', ridge),
    ('2', lasso),
    ('3', elasticnet),
    ('4', lightgbm),
    ('5', gbr),
    ('6', cbr),
    ('7', xgboost),
    ('8', svr)
]

stacked = StackingRegressor(estimators = estimators, final_estimator = elasticnet, 
                            n_jobs = -1, verbose = 4, cv = 10)
stacked.fit(X_train_full, y_train_full)

predictions = np.floor(np.expm1(stacked.predict(X_test)))


# In[52]:


submission = pd.DataFrame({'Id': X_test.index, 'SalePrice': predictions})
submission.to_csv('submission.csv', index = False)

