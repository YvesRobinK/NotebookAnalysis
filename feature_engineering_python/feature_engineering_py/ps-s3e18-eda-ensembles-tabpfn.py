#!/usr/bin/env python
# coding: utf-8

# # <h2 style = "font-family:Georgia;font-weight: bold; font-size:30px; background-color: white; color : #1192AA; border-radius: 100px 100px; text-align:left">Table of Contents</h2>
# 
# * &nbsp; **[Introduction](#Introduction)**
# 
#     * &nbsp; **[Getting Started](#Getting-Started)** 
#     
#     * &nbsp; **[Metadata](#Metadata)** 
#     
#     * &nbsp; **[Objective](#Objective)** 
#     
# * &nbsp; **[Import](#Import)**
# 
# * &nbsp; **[Check Dataset](#Check-Dataset)**
#    
# * &nbsp; **[Exploratory Data Analysis](#EDA)**
#     
# * &nbsp; **[Feature Engineering](#Feature-Engineering)**
#     
# * &nbsp; **[Model Building](#Model-Building)**
# 
#     * &nbsp; **[First Method: Optuna Ensembels](#First-Method:-Optuna-Ensembels)** 
#     
#     * &nbsp; **[Second Method: TabPFN](#Second-Method:-TabPFN)** 
# 
# * &nbsp; **[Submission](#Submission)**

# <h1 style = "font-family: Georgia;font-weight: bold; font-size: 30px; color: #1192AA; text-align:left">Introduction</h1>

# <img src = 'https://www.thoughtco.com/thmb/ZAyr-1lDbfchbzgVaaOJicjzHQ0=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/what-is-enzyme-structure-and-function-375555_v4-6f22f82931824e76b1c31401230deac8.png'>

# ## Getting Started

# * Welcome to my Kaggle Notebook for the latest Kaggle Playground Series, ["Explore Multi-Label Classification with an Enzyme Substrate Dataset."](https://www.kaggle.com/competitions/playground-series-s3e18) In this competition, our goal is to predict the probability of two targets, EC1 and EC2, for the test dataset.
# 
# * The [dataset](https://www.kaggle.com/competitions/playground-series-s3e18/data) provided for this challenge contains a collection of molecular data, specifically related to enzyme classes. Each entry represents a unique molecule, and the dataset includes various features that capture important molecular properties and characteristics. Some of these features include the Bertz complexity index, molecular connectivity indices, electrotopological states, molecular weights, and other molecular descriptors.
# 
# * [Enzymes](https://en.wikipedia.org/wiki/Enzyme) are known to act on molecules with structural similarities with their substrates. This behaviour is called promiscuity. Scientists working in drug discovery use this behaviour to target/design drugs to either block or promote biological actions. But, correct prediction of EC class(s) of substrates associated with enzymes has been a challenge in biology. Since there is no shortage of data, ML techniques can be employed to solve the aforementioned problem.

# ## Metadata

# (from https://www.kaggle.com/code/tumpanjawat/s3e18-eda-cluster-ensemble-ada-cat-gb#INTRODUCTION)
# 
# 1. Id: This feature represents the identifier or unique identification number of a molecule. It serves as a reference but doesn't directly contribute to the predictive model.
# 
# 2. BertzCT: This feature corresponds to the Bertz complexity index, which measures the structural complexity of a molecule. It can provide insights into the intricacy of molecular structures.
# 
# 3. Chi1: The Chi1 feature denotes the 1st order molecular connectivity index, which describes the topological connectivity of atoms in a molecule. It characterizes the atomic bonding pattern within the molecule.
# 
# 4. Chi1n: This feature is the normalized version of the Chi1 index. It allows for standardized comparisons of the 1st order molecular connectivity across different molecules.
# 
# 5. Chi1v: The Chi1v feature represents the 1st order molecular variance connectivity index. It captures the variance or diversity in the connectivity of atoms within a molecule.
# 
# 6. Chi2n: The Chi2n feature indicates the 2nd order molecular connectivity index, which provides information about the extended connectivity of atoms in a molecule. It considers the neighboring atoms of each atom in the molecule.
# 
# 7. Chi2v: Similar to Chi2n, the Chi2v feature measures the variance or diversity in the extended connectivity of atoms within a molecule at the 2nd order level.
# 
# 8. Chi3v: The Chi3v feature represents the 3rd order molecular variance connectivity index. It captures the variance in the 3rd order connectivity patterns among atoms in a molecule.
# 
# 9. Chi4n: This feature corresponds to the 4th order molecular connectivity index, which provides information about the extended connectivity of atoms in a molecule. The Chi4n index is normalized to allow for consistent comparisons across molecules.
# 
# 10. EState_VSA1: EState_VSA1 is a feature that relates to the electrotopological state of a molecule. Specifically, it represents the Van der Waals surface area contribution for a specific atom type, contributing to the overall electrotopological state.
# 
# 11. EState_VSA2: Similar to EState_VSA1, EState_VSA2 also represents the electrotopological state but for a different specific atom type.
# 
# 12. ExactMolWt: This feature denotes the exact molecular weight of a molecule. It provides an accurate measurement of the mass of the molecule.
# 
# 13. FpDensityMorgan1: FpDensityMorgan1 represents the Morgan fingerprint density for a specific radius of 1. Morgan fingerprints are a method for generating molecular fingerprints, and this feature captures the density of those fingerprints.
# 
# 14. FpDensityMorgan2: Similar to FpDensityMorgan1, this feature represents the Morgan fingerprint density for a specific radius of 2.
# 
# 15. FpDensityMorgan3: FpDensityMorgan3 corresponds to the Morgan fingerprint density for a specific radius of 3.
# 
# 16. HallkierAlpha: The HallkierAlpha feature denotes the Hall-Kier alpha value for a molecule. It is a measure of molecular shape and can provide insights into the overall structure of the molecule.
# 
# 17. HeavyAtomMolWt: This feature represents the molecular weight of heavy atoms only, excluding hydrogen atoms. It focuses on the mass of non-hydrogen atoms within the molecule.
# 
# 18. Kappa3: The Kappa3 feature corresponds to the Hall-Kier Kappa3 value, which is a molecular shape descriptor. It provides information about the shape and spatial arrangement of atoms within the molecule.
# 
# 19. MaxAbsEStateIndex: This feature represents the maximum absolute value of the E-state index. The E-state index relates to the electronic properties of a molecule, and its maximum absolute value can indicate the presence of specific electronic characteristics.
# 
# 20. MinEStateIndex: MinEStateIndex denotes the minimum value of the E-state index. It provides information about the lowest observed electronic property value within the molecule.
# 
# 21. NumHeteroatoms: This feature indicates the number of heteroatoms present in a molecule. Heteroatoms are atoms other than carbon and hydrogen, such as oxygen, nitrogen, sulfur, etc. This feature provides insights into the diversity and composition of atoms within the molecule.
# 
# 22. PEOE_VSA10: PEOE_VSA10 represents the partial equalization of orbital electronegativity Van der Waals surface area contribution for a specific atom type. It captures the surface area contribution of a particular atom type to the overall electrostatic properties.
# 
# 23. PEOE_VSA14: Similar to PEOE_VSA10, PEOE_VSA14 also represents the partial equalization of orbital electronegativity Van der Waals surface area contribution for a specific atom type.
# 
# 24. PEOE_VSA6: This feature corresponds to the partial equalization of orbital electronegativity Van der Waals surface area contribution for a specific atom type at a different level.
# 
# 25. PEOE_VSA7: Similar to PEOE_VSA6, PEOE_VSA7 represents the partial equalization of orbital electronegativity Van der Waals surface area contribution for a specific atom type.
# 
# 26. PEOE_VSA8: PEOE_VSA8 denotes the partial equalization of orbital electronegativity Van der Waals surface area contribution for a specific atom type.
# 
# 27. SMR_VSA10: SMR_VSA10 represents the solvent-accessible surface area Van der Waals surface area contribution for a specific atom type. It captures the contribution of a specific atom type to the solvent-accessible surface area.
# 
# 28. SMR_VSA5: Similar to SMR_VSA10, this feature denotes the solvent-accessible surface area Van der Waals surface area contribution for a specific atom type at a different level.
# 
# 29. SlogP_VSA3: The SlogP_VSA3 feature represents the LogP-based surface area contribution. It captures the contribution of a specific atom type to the surface area based on its logarithmic partition coefficient.
# 
# 30. VSA_EState9: This feature denotes the E-state fragment contribution for the Van der Waals surface area calculation. It captures the fragment-specific contribution to the electrostatic properties of the molecule.
# 
# 31. fr_COO: The fr_COO feature represents the number of carboxyl (COO) functional groups present in the molecule. It ranges from 0 to 8, providing insights into the presence and abundance of carboxyl groups.
# 
# 32. fr_COO2: Similar to fr_COO, fr_COO2 represents the number of carboxyl (COO) functional groups, ranging from 0 to 8.
# 
# 33. EC1: EC1 is a binary feature representing a predicted label related to Oxidoreductases. It serves as one of the target variables for prediction.
# 
# 32. EC2: EC2 is another binary feature representing a predicted label related to Transferases. It serves as another target variable for prediction.

# ## Objective

# * In this notebook, I will present two different approaches to tackle this multi-label classification problem. The first approach involves using two ensembles of models, such as XGBoost, LGBM, and CatBoost, to predict each target feature (EC1 and EC2) separately. This ensemble approach will allow us to leverage the strengths of each model and improve the overall predictive performance.
# 
# * The second approach is an experimental one, where I will explore the use of [TabPFN](https://github.com/automl/TabPFN) (Tabular Pretrained Fusion Networks). TabPFN is a cutting-edge technique that combines pretraining on large-scale tabular data with fine-tuning on the target task. By leveraging the power of transfer learning, TabPFN has shown promising results in various tabular data classification problems. We will experiment with TabPFN to see if it can effectively predict the enzyme classes in this dataset.

# # <h1 style = "font-family: Georgia;font-weight: bold; font-size: 30px; color: #1192AA; text-align:left">Import</h1>

# In[1]:


# Misc
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from copy import deepcopy
from functools import partial
from itertools import combinations
import random
import gc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Import sklearn classes for model selection, cross validation, and performance evaluation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from category_encoders import OrdinalEncoder, CountEncoder, CatBoostEncoder, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer, LabelEncoder # OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.decomposition import PCA, NMF
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

# Import libraries for Hypertuning
import optuna

# Import libraries for gradient boosting
import lightgbm as lgb
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.svm import NuSVC, SVC
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from catboost import Pool

# Useful line of code to set the display option so we could see all the columns in pd dataframe
pd.set_option('display.max_columns', None)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# # <h1 style = "font-family: Georgia;font-weight: bold; font-size: 30px; color: #1192AA; text-align:left">Check Dataset</h1>

# In[2]:


df_train = pd.read_csv("/kaggle/input/playground-series-s3e18/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s3e18/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s3e18/sample_submission.csv")
original_desc = pd.read_csv("/kaggle/input/ec-mixed-class/mixed_desc.csv")

target_col = ['EC1', 'EC2']

columns_to_keep = ['CIDs', 'BertzCT', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3v',
                   'Chi4n', 'EState_VSA1', 'EState_VSA2', 'ExactMolWt', 'FpDensityMorgan1',
                   'FpDensityMorgan2', 'FpDensityMorgan3', 'HallKierAlpha',
                   'HeavyAtomMolWt', 'Kappa3', 'MaxAbsEStateIndex', 'MinEStateIndex',
                   'NumHeteroatoms', 'PEOE_VSA10', 'PEOE_VSA14', 'PEOE_VSA6', 'PEOE_VSA7',
                   'PEOE_VSA8', 'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA3', 'VSA_EState9',
                   'fr_COO', 'fr_COO2', 'EC1_EC2_EC3_EC4_EC5_EC6']

original = original_desc.loc[:, columns_to_keep]

# There is probably a better way to do this, but that is was first came to my mind
feature1 = []
feature2 = []
for x in original['EC1_EC2_EC3_EC4_EC5_EC6']:
    feature1.append(int(x.split('_')[0]))
    feature2.append(int(x.split('_')[1]))

original['EC1'] = feature1
original['EC2'] = feature2

original.drop(columns = ['EC1_EC2_EC3_EC4_EC5_EC6', 'CIDs'], inplace=True)
original['id'] = original.reset_index().index

df_train.drop(columns = ['EC3', 'EC4', 'EC5', 'EC6'], inplace=True)


numerical_columns = ['BertzCT', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3v',
                   'Chi4n', 'EState_VSA1', 'EState_VSA2', 'ExactMolWt', 'FpDensityMorgan1',
                   'FpDensityMorgan2', 'FpDensityMorgan3', 'HallKierAlpha',
                   'HeavyAtomMolWt', 'Kappa3', 'MaxAbsEStateIndex', 'MinEStateIndex',
                   'NumHeteroatoms', 'PEOE_VSA10', 'PEOE_VSA14', 'PEOE_VSA6', 'PEOE_VSA7',
                   'PEOE_VSA8', 'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA3', 'VSA_EState9']

cat_cols = ['fr_COO', 'fr_COO2']

print(f'Data Successfully Loaded \n')

print(f'[INFO] Shapes:'
      f'\n original: {original.shape}'
      f'\n train: {df_train.shape}'
      f'\n test: {df_test.shape}\n')

print(f'[INFO] Any missing values:'
      f'\n original: {original.isna().any().any()}'
      f'\n train: {df_train.isna().any().any()}'
      f'\n test: {df_test.isna().any().any()}')


# In[3]:


df_train.head()


# In[4]:


original.head()


# In[5]:


df_train.drop(index=3422, inplace=True) # df_train[df_train['FpDensityMorgan1'] == -666]


# # <h1 style = "font-family: Georgia;font-weight: bold; font-size: 30px; color: #1192AA; text-align:left">EDA</h1>

# In[6]:


# Create palette

my_palette = sns.cubehelix_palette(n_colors = 7, start=.46, rot=-.45, dark = .2, hue=0.95)
sns.palplot(my_palette)
plt.gcf().set_size_inches(13,2)

for idx,values in enumerate(my_palette.as_hex()):
    plt.text(idx-0.375,0, my_palette.as_hex()[idx],{'font': "Courier New", 'size':16, 'weight':'bold','color':'black'}, alpha =0.7)
plt.gcf().set_facecolor('white')

plt.show()


# In[7]:


# create figure and set style with white background
plt.figure(figsize = (14, 8))
sns.set_style('white')

# set colors
colors = my_palette

# plot
plt.barh(df_train['EC1'].value_counts().index,
        df_train['EC1'].value_counts(),
        color = colors[1:3])

# set title
plt.title('EC1 Distribution in df_train', fontsize = 14, fontweight = 'bold')

# remove spines from plot
sns.despine()

# display all open figures
plt.show()


# In[8]:


# create figure and set style with white background
plt.figure(figsize = (14, 8))
sns.set_style('white')

# set colors
colors = my_palette

# plot
plt.barh(df_train['EC2'].value_counts().index,
        df_train['EC2'].value_counts(),
        color = colors[1:3])

# set title
plt.title('EC2 Distribution in df_train', fontsize = 14, fontweight = 'bold')

# remove spines from plot
sns.despine()

# display all open figures
plt.show()


# In[9]:


# create figure and set style with white background
plt.figure(figsize = (14, 8))
sns.set_style('white')

# set colors
colors = my_palette

# plot
plt.barh(original["EC1"].value_counts().index,
        original["EC1"].value_counts(),
        color = colors[3:5])

# set title
plt.title('EC1 Distribution in original', fontsize = 14, fontweight = 'bold')

# remove spines from plot
sns.despine()

# display all open figures
plt.show()


# In[10]:


# create figure and set style with white background
plt.figure(figsize = (14, 8))
sns.set_style('white')

# set colors
colors = my_palette

# plot
plt.barh(original["EC2"].value_counts().index,
        original["EC2"].value_counts(),
        color = colors[3:5])

# set title
plt.title('EC2 Distribution in original', fontsize = 14, fontweight = 'bold')

# remove spines from plot
sns.despine()

# display all open figures
plt.show()


# In[11]:


# Create subplots
fig, axes = plt.subplots(len(numerical_columns), 2, figsize=(14, 120))

# Plot the histograms and box plots
for i, column in enumerate(numerical_columns):
    # Histogram
    sns.histplot(df_train[column], bins=30, kde=True, ax=axes[i, 0], color = my_palette[2])
    axes[i, 0].set_title(f'Distribution of {column} in df_train')
    axes[i, 0].set_xlabel('Value')
    axes[i, 0].set_ylabel('Frequency')

    # Box plot
    sns.boxplot(df_train[column], ax=axes[i, 1], color = my_palette[1])
    axes[i, 1].set_title(f'Box plot of {column} in df_train')
    axes[i, 1].set_xlabel(column)
    axes[i, 1].set_ylabel('Value')

plt.tight_layout()
plt.show()


# In[12]:


# Create subplots
fig, axes = plt.subplots(len(numerical_columns), 2, figsize=(14, 120))

# Plot the histograms and box plots
for i, column in enumerate(numerical_columns):
    # Histogram
    sns.histplot(df_train[column], bins=30, kde=True, ax=axes[i, 0], color = my_palette[4])
    axes[i, 0].set_title(f'Distribution of {column} in original')
    axes[i, 0].set_xlabel('Value')
    axes[i, 0].set_ylabel('Frequency')

    # Box plot
    sns.boxplot(df_train[column], ax=axes[i, 1], color = my_palette[3])
    axes[i, 1].set_title(f'Box plot of {column} in original')
    axes[i, 1].set_xlabel(column)
    axes[i, 1].set_ylabel('Value')

plt.tight_layout()
plt.show()


# # <h1 style = "font-family: Georgia;font-weight: bold; font-size: 30px; color: #1192AA; text-align:left">Feature Engineering </h1>

# In[13]:


def cat_encoder(X_train, X_test, cat_cols, encode='label'):
    
    if encode == 'label':
        ## Label Encoder
        encoder = OrdinalEncoder(cols=cat_cols)
        train_encoder = encoder.fit_transform(X_train[cat_cols]).astype(int)
        test_encoder = encoder.transform(X_test[cat_cols]).astype(int)
        X_train[cat_cols] = train_encoder[cat_cols]
        X_test[cat_cols] = test_encoder[cat_cols]
        encoder_cols = cat_cols
    
    else:
        ## OneHot Encoder
        encoder = OneHotEncoder(handle_unknown='ignore')
        train_encoder = encoder.fit_transform(X_train[cat_cols]).astype(int)
        test_encoder = encoder.transform(X_test[cat_cols]).astype(int)
        X_train = pd.concat([X_train, train_encoder], axis=1)
        X_test = pd.concat([X_test, test_encoder], axis=1)
        X_train.drop(cat_cols, axis=1, inplace=True)
        X_test.drop(cat_cols, axis=1, inplace=True)
        encoder_cols = list(train_encoder.columns)
        
    return X_train, X_test, encoder_cols


# In[14]:


def create_features(df):
    
    new_features = {
        'BertzCT_MaxAbsEStateIndex_Ratio': df['BertzCT'] / (df['MaxAbsEStateIndex'] + 1e-12),
        'BertzCT_ExactMolWt_Product': df['BertzCT'] * df['ExactMolWt'],
        'NumHeteroatoms_FpDensityMorgan1_Ratio': df['NumHeteroatoms'] / (df['FpDensityMorgan1'] + 1e-12),
        'VSA_EState9_EState_VSA1_Ratio': df['VSA_EState9'] / (df['EState_VSA1'] + 1e-12),
        'PEOE_VSA10_SMR_VSA5_Ratio': df['PEOE_VSA10'] / (df['SMR_VSA5'] + 1e-12),
        'Chi1v_ExactMolWt_Product': df['Chi1v'] * df['ExactMolWt'],
        'Chi2v_ExactMolWt_Product': df['Chi2v'] * df['ExactMolWt'],
        'Chi3v_ExactMolWt_Product': df['Chi3v'] * df['ExactMolWt'],
        'EState_VSA1_NumHeteroatoms_Product': df['EState_VSA1'] * df['NumHeteroatoms'],
        'PEOE_VSA10_Chi1_Ratio': df['PEOE_VSA10'] / (df['Chi1'] + 1e-12),
        'MaxAbsEStateIndex_NumHeteroatoms_Ratio': df['MaxAbsEStateIndex'] / (df['NumHeteroatoms'] + 1e-12),
        'BertzCT_Chi1_Ratio': df['BertzCT'] / (df['Chi1'] + 1e-12),
    }
    
    df = df.assign(**new_features)
    new_cols = list(new_features.keys())
    
    return df, new_cols


# In[15]:


class AggFeatureExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, group_col, agg_col, agg_func):
        self.group_col = group_col
        self.group_col_name = ''
        for col in group_col:
            self.group_col_name += col
        self.agg_col = agg_col
        self.agg_func = agg_func
        self.agg_df = None
        self.medians = None
        
    def fit(self, X, y=None):
        group_col = self.group_col
        agg_col = self.agg_col
        agg_func = self.agg_func
        
        self.agg_df = X.groupby(group_col)[agg_col].agg(agg_func)
        self.agg_df.columns = [f'{self.group_col_name}_{agg}_{_agg_col}' for _agg_col in agg_col for agg in agg_func]
        self.medians = X[agg_col].median()
        
        return self
    
    def transform(self, X):
        group_col = self.group_col
        agg_col = self.agg_col
        agg_func = self.agg_func
        agg_df = self.agg_df
        medians = self.medians
        
        X_merged = pd.merge(X, agg_df, left_on=group_col, right_index=True, how='left')
        X_merged.fillna(medians, inplace=True)
        X_agg = X_merged.loc[:, [f'{self.group_col_name}_{agg}_{_agg_col}' for _agg_col in agg_col for agg in agg_func]]
        
        return X_agg
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        X_agg = self.transform(X)
        return X_agg


# In[16]:


# Concatenate train and original dataframes, and prepare train and test sets
train = pd.concat([df_train, original])
test = df_test.copy()

X_train = train.drop(columns = target_col).reset_index(drop=True)
y_train = train.loc[:, target_col].reset_index(drop=True)
X_test = test.reset_index(drop=True)

# Create combination Features
X_train, _ = create_features(X_train)
X_test, _ = create_features(X_test)

# Aggregate Features
group_cols = [
        ['EState_VSA2'], ['HallKierAlpha'], ['NumHeteroatoms'], 
        ['PEOE_VSA10'], ['PEOE_VSA14'], ['PEOE_VSA6'], ['PEOE_VSA7'], ['PEOE_VSA8'],
        ['SMR_VSA10'], ['SMR_VSA5'], ['SlogP_VSA3'], ['fr_COO'], #['fr_COO2'],
]
agg_col = [
    'BertzCT',
    'Chi1', 
    'Chi1n', 
    'Chi1v', 
    'Chi2n', 
    'Chi2v', 
    'Chi3v',
    'EState_VSA1', 
    'ExactMolWt', 
    'FpDensityMorgan1', 
    'FpDensityMorgan2', 
    'FpDensityMorgan3',
    'HeavyAtomMolWt',  
    'MaxAbsEStateIndex', 
    'MinEStateIndex', 
    'VSA_EState9'
]
agg_train, agg_test = [], []
for group_col in group_cols:
    agg_extractor = AggFeatureExtractor(group_col=group_col, agg_col=agg_col, agg_func=['mean', 'std'])
    agg_extractor.fit(pd.concat([X_train, X_test], axis=0))
    agg_train.append(agg_extractor.transform(X_train))
    agg_test.append(agg_extractor.transform(X_test))
X_train = pd.concat([X_train] + agg_train, axis=1).fillna(0)
X_test = pd.concat([X_test] + agg_test, axis=1).fillna(0)

# Category Encoders
X_train, X_test, _ = cat_encoder(X_train, X_test, cat_cols, encode='label')


# StandardScaler
sc = StandardScaler() # MinMaxScaler or StandardScaler
X_train[numerical_columns] = sc.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = sc.transform(X_test[numerical_columns])

# Drop_col
drop_cols = ['id']
X_train.drop(drop_cols, axis=1, inplace=True)
X_test.drop(drop_cols, axis=1, inplace=True)

del train, test, df_train, df_test

print(f"X_train shape :{X_train.shape} , y_train shape :{y_train.shape}")
print(f"X_test shape :{X_test.shape}")

X_train.head()


# # <h1 style = "font-family: Georgia;font-weight: bold; font-size: 30px; color: #1192AA; text-align:left">Model Building</h1>

# ## First Method: Optuna Ensembels

# In[17]:


y_train_copy = y_train.copy()


# In[18]:


class Splitter:
    def __init__(self, n_splits=5, cat_df=pd.DataFrame(), test_size=0.5):
        self.n_splits = n_splits
        self.cat_df = cat_df
        self.test_size = test_size

    def split_data(self, X, y, random_state_list):
        for random_state in random_state_list:
            kf = KFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
            for train_index, val_index in kf.split(X, y):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                yield X_train, X_val, y_train, y_val, val_index


# In[19]:


class OptunaWeights:
    def __init__(self, random_state, n_trials=100):
        self.study = None
        self.weights = None
        self.random_state = random_state
        self.n_trials = n_trials

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 1e-15, 1) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)

        # Calculate the score for the weighted prediction
        score = roc_auc_score(y_true, weighted_pred)
        return score

    def fit(self, y_true, y_preds):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptunaWeights", direction='maximize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds):
        self.fit(y_true, y_preds)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights


# ### Predict EC1

# In[20]:


# Config
y_train = y_train_copy.loc[:, 'EC1']

n_splits = 10
random_state = 42
random_state_list =[42]
n_estimators = 100
device = 'cpu'
early_stopping_rounds = 100
verbose = False


# In[21]:


scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
scale_pos_weight


# In[22]:


class_weight_0 = 1.0
class_weight_1 = 1.0 / scale_pos_weight

class_weights = [class_weight_0, class_weight_1]


# In[23]:


class Classifier:
    def __init__(self, n_estimators=100, device="cpu", random_state=42):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self.get_models()
        self.models_name = list(self.get_models().keys())
        self.len_models = len(self.models)
        
    def get_models(self):
             
        xgb_optuna0 = {
            'n_estimators': 500,
            'learning_rate': 0.010014042393742822,
            'booster': 'gbtree',
            'lambda': 0.03131321744836397,
            'alpha': 0.03690926667179868,
            'subsample': 0.5415480288839364,
            'colsample_bytree': 0.534352840297025,
            'max_depth': 5,
            'min_child_weight': 1,
            'eta': 0.263110198744306,
            'gamma': 0.2833843987379326,
            'verbosity': 0,
            'random_state': self.random_state,
            'scale_pos_weight': scale_pos_weight,
        }
        
        xgb_optuna1 = {
            'n_estimators': 1500,
            'learning_rate': 0.08901459197907591,
            'booster': 'gbtree',
            'lambda': 8.550251116462702,
            'alpha': 6.92130114930949,
            'eta': 0.7719873740829137,
            'grow_policy': 'lossguide',
            'n_jobs': -1,
            'objective': 'binary:logistic',
            'verbosity': 0,
            'random_state': self.random_state,
            'scale_pos_weight': scale_pos_weight,
        }
        
        xgb_optuna2 = {
            'n_estimators': 550,
            'learning_rate': 0.014551680348136895,
            'booster': 'gbtree',
            'lambda': 0.028738149876528587,
            'alpha': 0.014056635017117198,
            'subsample': 0.538653498449084,
            'colsample_bytree': 0.518050828371974, 
            'max_depth': 4, 'min_child_weight': 4,
            'eta': 0.6953619445477833,
            'gamma': 0.9036568111424781,
            'scale_pos_weight': 60,
            'grow_policy': 'lossguide',
            'n_jobs': -1,
            'objective': 'binary:logistic',
            'verbosity': 0,
            'random_state': self.random_state,
            'scale_pos_weight': scale_pos_weight,
        }
    
        xgb1_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.0503196477566407,
            'booster': 'gbtree',
            'lambda': 0.00379319640405843,
            'alpha': 0.106754104302093,
            'subsample': 0.938028434508189,
            'colsample_bytree': 0.212545425027345,
            'max_depth': 9,
            'min_child_weight': 2,
            'eta': 1.03662446190642E-07,
            'gamma': 0.000063826049787043,
            'grow_policy': 'lossguide',
            'n_jobs': -1,
            'objective': 'binary:logistic',
            #'eval_metric': 'auc',
            'verbosity': 0,
            'random_state': self.random_state,
            'scale_pos_weight': scale_pos_weight,
        }
        xgb2_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.00282353606391198,
            'booster': 'gbtree',
            'lambda': 0.399776698351379,
            'alpha': 1.01836149061356E-07,
            'subsample': 0.957123754766769,
            'colsample_bytree': 0.229857555596548,
            'max_depth': 9,
            'min_child_weight': 4,
            'eta': 2.10637756839133E-07,
            'gamma': 0.00314857715085414,
            'grow_policy': 'depthwise',
            'n_jobs': -1,
            'objective': 'binary:logistic',
            #'eval_metric': 'auc',
            'verbosity': 0,
            'random_state': self.random_state,
            'scale_pos_weight': scale_pos_weight,
        }
        xgb3_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.00349356650247156,
            'booster': 'gbtree',
            'lambda': 0.0002963239871324443,
            'alpha': 0.0000162103492458353,
            'subsample': 0.822994064549709,
            'colsample_bytree': 0.244618079894501,
            'max_depth': 10,
            'min_child_weight': 2,
            'eta': 8.03406601824666E-06,
            'gamma': 3.91180893163099E-07,
            'grow_policy': 'depthwise',
            'n_jobs': -1,
            'objective': 'binary:logistic',
            #'eval_metric': 'auc',
            'verbosity': 0,
            'random_state': self.random_state,
            'scale_pos_weight': scale_pos_weight,
        }
        
        lgb_optuna0 = {
            'num_iterations': 200,
            'learning_rate': 0.0177811006863138,
            'max_depth': 4,
            'lambda': 3.217623354163234,
            'alpha': 8.493354544976775,
            'subsample': 0.5718058301387113,
            'colsample_bytree': 0.5131315279744134,
            'min_child_weight': 5,
            'device': self.device,
            'random_state': self.random_state
        }
        
        lgb_optuna1 = {
            'num_iterations': 200,
            'learning_rate': 0.024714536811915398,
            'max_depth': 9,
            'lambda': 9.498413255934212,
            'alpha': 7.627590925937886,
            'subsample': 0.9680186598781285,
            'colsample_bytree': 0.5645599877042381,
            'min_child_weight': 1,
            'device': self.device,
            'random_state': self.random_state
        }
        
        lgb_optuna2 = {
            'num_iterations': 950,
            'learning_rate': 0.012019976156417951,
            'max_depth': 4,
            'lambda': 6.958643473661789,
            'alpha': 0.0012598800466591953, 
            'subsample': 0.9344619448867001,
            'colsample_bytree': 0.9864399750557648, 
            'min_child_weight': 1,
            'device': self.device,
            'random_state': self.random_state
        }
        
        cat_optuna0 = {
            'iterations': 650,
            'learning_rate': 0.01484756439623765,
            'depth': 6,
            'l2_leaf_reg': 5.6061100632887,
            'bagging_temperature': 1.4247109406038643,
            'random_strength': 0.3339464318780084,
            'random_state': self.random_state,
            'verbose': False,
            'class_weights': class_weights
        }
        
        hist_params = {
                'l2_regularization': 0.654926989031482,
                'learning_rate': 0.0366207257406611,
                'max_iter': self.n_estimators,
                'max_depth': 30,
                'max_bins': 255,
                'min_samples_leaf': 52,
                'max_leaf_nodes':12,
                'early_stopping': True,
                'n_iter_no_change': 50,
                #'class_weight':'balanced',
                'random_state': self.random_state
            }
        
        mlp_params = {
            'max_iter': 800,
            'early_stopping': True,
            'n_iter_no_change': 20,
            'random_state': self.random_state,
        }
        
        models = {
            "xgbo0": xgb.XGBClassifier(**xgb_optuna0),
            "xgbo1": xgb.XGBClassifier(**xgb_optuna1),
            "xgbo2": xgb.XGBClassifier(**xgb_optuna2),
            #"xgb1": xgb.XGBClassifier(**xgb1_params),
            "xgb2": xgb.XGBClassifier(**xgb2_params),
            "xgb3": xgb.XGBClassifier(**xgb3_params),
            "lgbo0": lgb.LGBMClassifier(**lgb_optuna0),
            "lgbo1": lgb.LGBMClassifier(**lgb_optuna1),
            "lgbo2": lgb.LGBMClassifier(**lgb_optuna2),
            "cato0": CatBoostClassifier(**cat_optuna0),
            'hgb': HistGradientBoostingClassifier(**hist_params),
            #'mlp': MLPClassifier(**mlp_params, hidden_layer_sizes=(100,)),
            'rf': RandomForestClassifier(n_estimators=500, n_jobs=-1, class_weight="balanced", random_state=self.random_state),
            #'lr': LogisticRegressionCV(max_iter=2000, random_state=self.random_state)
        }
        return models


# In[24]:


# Split Data
splitter = Splitter(n_splits=n_splits, cat_df= y_train)
splits = splitter.split_data(X_train, y_train, random_state_list=random_state_list)

# Initialize an array for storing test predictions
classifier = Classifier(n_estimators=n_estimators, device=device, random_state=random_state)
test_predss = np.zeros((X_test.shape[0]))
oof_predss = np.zeros((X_train.shape[0]))
ensemble_score = []
weights = []
models_name = [_ for _ in classifier.models_name if ('xgb' in _) or ('lgb' in _) or ('cat' in _)]
trained_models = dict(zip(models_name, [[] for _ in range(classifier.len_models)]))
score_dict = dict(zip(classifier.models_name, [[] for _ in range(len(classifier.models_name))]))

for i, (X_train_, X_val, y_train_, y_val, val_index) in enumerate(splits):
    
    n = i % n_splits
    m = i // n_splits
    

    # Classifier models
    classifier = Classifier(n_estimators, device, random_state)
    models = classifier.models

    # Store oof and test predictions for each base model
    oof_preds = []
    test_preds = []

    # Loop over each base model and fit it
    for name, model in models.items():
        if ('xgb' in name) or ('lgb' in name):
            model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
            
        elif 'cat' in name :
                model.fit(
                    Pool(X_train_, y_train_, cat_features=cat_cols), eval_set=Pool(X_val, y_val, cat_features=cat_cols),
                    early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        else:
            model.fit(X_train_, y_train_)
            
        if name in trained_models.keys():
            trained_models[f'{name}'].append(deepcopy(model))

        test_pred = model.predict_proba(X_test)[:, 1]
        y_val_pred = model.predict_proba(X_val)[:, 1]

        score = roc_auc_score(y_val, y_val_pred)
        score_dict[name].append(score)
        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] ROC-AUC score: {score:.5f}')

        oof_preds.append(y_val_pred)
        test_preds.append(test_pred)

    # Use OptunaWeights
    optweights = OptunaWeights(random_state)
    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)

    score = roc_auc_score(y_val, y_val_pred)
    print(f'Ensemble [FOLD-{n} SEED-{random_state_list[m]}] ROC-AUC score {score:.5f} \n')
    ensemble_score.append(score)
    weights.append(optweights.weights)

    # Predict to X_test by the best ensemble weights
    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))
    oof_predss[X_val.index] = optweights.predict(oof_preds)

    gc.collect()


# In[25]:


# Calculate the mean score of the ensemble
mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Mean Optuna Ensemble {mean_score:.5f} ± {std_score:.5f} \n')

print('--- Optuna Weights---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name}: {mean_weight:.5f} ± {std_weight:.5f}')


# In[26]:


ec1_test_predss = test_predss


# In[27]:


# https://www.kaggle.com/code/tetsutani/ps3e18-eda-ensemble-ml-pipeline-binarypredictict#Make-Submission

my_palette = sns.cubehelix_palette(n_colors = 7, start=.46, rot=-.45, dark = .2, hue=0.95, as_cmap=True)

def show_confusion_roc(oof, title='Model Evaluation Results'):
    f, ax = plt.subplots(1, 2, figsize=(16, 6))
    df = pd.DataFrame({'preds': oof[0], 'target': oof[1]})
    cm = confusion_matrix(df.target, df.preds.ge(0.5).astype(int))
    cm_display = ConfusionMatrixDisplay(cm).plot(cmap=my_palette, ax=ax[0])
    ax[0].grid(False)
    RocCurveDisplay.from_predictions(df.target, df.preds, ax=ax[1])
    ax[1].grid(True)
    plt.suptitle(f'{title}', fontsize=12, fontweight='bold')
    plt.tight_layout()

show_confusion_roc(oof=[oof_predss, y_train], title='EC1 OOF Evaluation Results')


# In[28]:


oof_predss


# ### Predict EC2

# I decided to add EC1 oof predicts as a feature to train dataset for EC2 predict

# In[29]:


# Config
y_train = y_train_copy.loc[:, 'EC2']

n_splits = 5
random_state = 2042
random_state_list =[2042]
n_estimators = 500
device = 'cpu'
early_stopping_rounds = 300
verbose = False


# In[30]:


# X_train['EC1_pred'] = oof_predss
# X_test['EC1_pred'] = test_predss


# In[31]:


scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
scale_pos_weight


# In[32]:


class_weight_0 = 1.0
class_weight_1 = 1.0 / scale_pos_weight

class_weights = [class_weight_0, class_weight_1]


# In[33]:


class Classifier:
    def __init__(self, n_estimators=100, device="cpu", random_state=42):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self.get_models()
        self.models_name = list(self.get_models().keys())
        self.len_models = len(self.models)
        
    def get_models(self):
        
        lgb_optuna0 = {
            'num_iterations': 700,
            'learning_rate': 0.02360156976543837,
            'max_depth': 3,
            'lambda': 9.224535268287106,
            'alpha': 5.103018845370762,
            'subsample': 0.8672389489074398,
            'colsample_bytree': 0.5454358875778684,
            'min_child_weight': 1,
            
            'verbose': 0,
            'random_state': self.random_state
        }
        
        lgb1 = {
            'objective': 'binary',
            'metric': 'auc',
            'feature_pre_filter': False,
            'lambda_l1': 0.0,
            'lambda_l2': 0.0,
            'num_leaves': 11,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7499409223062861,
            'bagging_freq': 4,
            'min_child_samples': 20,
            'num_iterations': 100,
            'random_state': self.random_state
        }
        
        xgb_optuna0 = {
            'n_estimators': 750,
            'learning_rate': 0.010553675405166283,
            'booster': 'gbtree',
            'lambda': 0.08612513132877414,
            'alpha': 0.09126508798663525,
            'subsample': 0.6643168242053055,
            'colsample_bytree': 0.7098374281060121,
            'max_depth': 3,
            'min_child_weight': 1,
            'eta': 0.20458543042851302,
            'gamma': 0.15002104219322876,
            'verbosity': 0,
            'random_state': self.random_state,
            'scale_pos_weight': scale_pos_weight
        }
        
        xgb_optuna2 = {
            'n_estimators': 550,
            'learning_rate': 0.014551680348136895,
            'booster': 'gbtree',
            'lambda': 0.028738149876528587,
            'alpha': 0.014056635017117198,
            'subsample': 0.538653498449084,
            'colsample_bytree': 0.518050828371974, 
            'max_depth': 4, 'min_child_weight': 4,
            'eta': 0.6953619445477833,
            'gamma': 0.9036568111424781,
            'scale_pos_weight': 60,
            'grow_policy': 'lossguide',
            'n_jobs': -1,
            'objective': 'binary:logistic',
            'verbosity': 0,
            'random_state': self.random_state,
            'scale_pos_weight': scale_pos_weight
        }
    
        xgb1_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.0503196477566407,
            'booster': 'gbtree',
            'lambda': 0.00379319640405843,
            'alpha': 0.106754104302093,
            'subsample': 0.938028434508189,
            'colsample_bytree': 0.212545425027345,
            'max_depth': 9,
            'min_child_weight': 2,
            'eta': 1.03662446190642E-07,
            'gamma': 0.000063826049787043,
            'grow_policy': 'lossguide',
            'n_jobs': -1,
            'objective': 'binary:logistic',
            #'eval_metric': 'auc',
            'verbosity': 0,
            'random_state': self.random_state,
            'scale_pos_weight': scale_pos_weight
        }
        
        xgb2_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.00282353606391198,
            'booster': 'gbtree',
            'lambda': 0.399776698351379,
            'alpha': 1.01836149061356E-07,
            'subsample': 0.957123754766769,
            'colsample_bytree': 0.229857555596548,
            'max_depth': 9,
            'min_child_weight': 4,
            'eta': 2.10637756839133E-07,
            'gamma': 0.00314857715085414,
            'grow_policy': 'depthwise',
            'n_jobs': -1,
            'objective': 'binary:logistic',
            #'eval_metric': 'auc',
            'verbosity': 0,
            'random_state': self.random_state,
            'scale_pos_weight': scale_pos_weight
        }
        
        xgb3_params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 0.00349356650247156,
            'booster': 'gbtree',
            'lambda': 0.0002963239871324443,
            'alpha': 0.0000162103492458353,
            'subsample': 0.822994064549709,
            'colsample_bytree': 0.244618079894501,
            'max_depth': 10,
            'min_child_weight': 2,
            'eta': 8.03406601824666E-06,
            'gamma': 3.91180893163099E-07,
            'grow_policy': 'depthwise',
            'n_jobs': -1,
            'objective': 'binary:logistic',
            #'eval_metric': 'auc',
            'verbosity': 0,
            'random_state': self.random_state,
            'scale_pos_weight': scale_pos_weight
        }
        
        xgb4_params = {
                'n_estimators': self.n_estimators,
                'learning_rate': 0.0258060514910791,
                'booster': 'gbtree',
                'lambda': 7.46721185757775E-06,
                'alpha': 2.76013165565544E-08,
                'subsample': 0.20132629296478,
                'colsample_bytree': 0.45781987213833,
                'max_depth': 5,
                'min_child_weight': 5,
                'eta': 3.9844926835765E-07,
                'gamma': 0.0000620888806796158,
                'grow_policy': 'depthwise',
                'n_jobs': -1,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'verbosity': 0,
                'random_state': self.random_state,
            }
        xgb5_params = {
                'n_estimators': self.n_estimators,
                'learning_rate': 0.03045801481188,
                'booster': 'gbtree',
                'lambda': 0.141226751984267,
                'alpha': 0.0000169212384166775,
                'subsample': 0.354547691277393,
                'colsample_bytree': 0.741230587323123,
                'max_depth': 3,
                'min_child_weight': 8,
                'eta': 0.000200365560443557,
                'gamma': 0.000793115073634548,
                'grow_policy': 'depthwise',
                'n_jobs': -1,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'verbosity': 0,
                'random_state': self.random_state,
            }
        
        cat_optuna0 = {
            'iterations': 700,
            'learning_rate': 0.02057458403498283,
            'depth': 5,
            'l2_leaf_reg': 5.157899753023626,
            'bagging_temperature': 0.8389428403768708,
            'random_strength': 3.025644507100712,
            'random_state': self.random_state,
            'verbose': False,
        }
        
        cat_optuna1 = {
            'iterations': 600,
            'learning_rate': 0.019499308200732167,
            'depth': 8,
            'l2_leaf_reg': 9.024309909697191,
            'bagging_temperature': 7.9669359481998825,
            'random_strength': 5.293875378529096,
            'border_count': 235,
            'auto_class_weights': 'Balanced',
            'task_type': self.device.upper(),
            'verbose': False,
            'allow_writing_files': False,
            'random_state': self.random_state
        }
        
        cat_optuna2 = {
            'iterations': 1000,
            'learning_rate': 0.013171032440433215,
            'depth': 5, 
            'l2_leaf_reg': 2.805405544410651,
            'bagging_temperature': 5.869195302151575,
            'random_strength': 9.103415468292203,
            'task_type': self.device.upper(),
            'verbose': False,
            'allow_writing_files': False,
            'random_state': self.random_state,
            'class_weights': class_weights
        }
        
        cat1_params = {
            'iterations': self.n_estimators,
            'depth': 3,
            'learning_rate': 0.020258010893459,
            'l2_leaf_reg': 0.583685138705941,
            'random_strength': 0.177768021213223,
            'od_type': "Iter", 
            'od_wait': 116,
            'bootstrap_type': "Bayesian",
            'grow_policy': 'Depthwise',
            'bagging_temperature': 0.478048798393903,
            'eval_metric': 'Logloss', # AUC
            'loss_function': 'Logloss',
            'task_type': self.device.upper(),
            'verbose': False,
            'allow_writing_files': False,
            'random_state': self.random_state
        }
        
        cat2_params = {
            'iterations': self.n_estimators,
            'depth': 5,
            'learning_rate': 0.00666304601039438,
            'l2_leaf_reg': 0.0567881687170355,
            'random_strength': 0.00564702921370138,
            'od_type': "Iter", 
            'od_wait': 93,
            'bootstrap_type': "Bayesian",
            'grow_policy': 'Depthwise',
            'bagging_temperature': 2.48298505165348,
            'eval_metric': 'Logloss', # AUC
            'loss_function': 'Logloss',
            'task_type': self.device.upper(),
            'verbose': False,
            'allow_writing_files': False,
            'random_state': self.random_state
        }
        
        cat3_params = {
            'iterations': self.n_estimators,
            'depth': 5,
            'learning_rate': 0.0135730417743519,
            'l2_leaf_reg': 0.0597353604503262,
            'random_strength': 0.0675876600077264,
            'od_type': "Iter", 
            'od_wait': 122,
            'bootstrap_type': "Bayesian",
            'grow_policy': 'Depthwise',
            'bagging_temperature': 1.85898154006468,
            'eval_metric': 'Logloss', # AUC
            'loss_function': 'Logloss',
            'task_type': self.device.upper(),
            'verbose': False,
            'allow_writing_files': False,
            'random_state': self.random_state
        }

        cat4_params = {
            'iterations': self.n_estimators,
            'depth': 4,
            'learning_rate': 0.0533074594005429,
            'l2_leaf_reg': 4.33121673696473,
            'random_strength': 0.00420305570017096,
            'od_type': "IncToDec", 
            'od_wait': 41,
            'bootstrap_type': "Bayesian",
            'grow_policy': 'Lossguide',
            'bagging_temperature': 9.20357081888618,
            'eval_metric': 'AUC',
            'loss_function': 'Logloss',
            'task_type': self.device.upper(),
            'verbose': False,
            'allow_writing_files': False,
            'random_state': self.random_state
        }
        
        models = {
            #"lgbo0": lgb.LGBMClassifier(**lgb_optuna0),
            #"xgbo1": xgb.XGBClassifier(**xgb_optuna1),
            "xgbo0": xgb.XGBClassifier(**xgb_optuna0),
            "xgbo2": xgb.XGBClassifier(**xgb_optuna2),
            "xgb1": xgb.XGBClassifier(**xgb1_params),
            "xgb2": xgb.XGBClassifier(**xgb2_params),
            "xgb3": xgb.XGBClassifier(**xgb3_params),
            "xgb5": xgb.XGBClassifier(**xgb3_params),
            "cato0": CatBoostClassifier(**cat_optuna0),
            #"cato1": CatBoostClassifier(**cat_optuna1),
            "cato2": CatBoostClassifier(**cat_optuna2),
            #"cat1": CatBoostClassifier(**cat1_params),
            #"cat2": CatBoostClassifier(**cat2_params),
            #"cat3": CatBoostClassifier(**cat3_params),
            "cat4": CatBoostClassifier(**cat4_params),
            'rf': RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=self.random_state),
            #'lr': LogisticRegressionCV(max_iter=2000, random_state=self.random_state)
        }
        return models


# In[34]:


# Split Data
splitter = Splitter(n_splits=n_splits, cat_df= y_train)
splits = splitter.split_data(X_train, y_train, random_state_list=random_state_list)

# Initialize an array for storing test predictions
classifier = Classifier(n_estimators=n_estimators, device=device, random_state=random_state)
test_predss = np.zeros((X_test.shape[0]))
oof_predss = np.zeros((X_train.shape[0]))
ensemble_score = []
weights = []
models_name = [_ for _ in classifier.models_name if ('xgb' in _) or ('lgb' in _) or ('cat' in _)]
trained_models = dict(zip(models_name, [[] for _ in range(classifier.len_models)]))
score_dict = dict(zip(classifier.models_name, [[] for _ in range(len(classifier.models_name))]))

for i, (X_train_, X_val, y_train_, y_val, val_index) in enumerate(splits):
    
    n = i % n_splits
    m = i // n_splits
    

    # Classifier models
    classifier = Classifier(n_estimators, device, random_state)
    models = classifier.models

    # Store oof and test predictions for each base model
    oof_preds = []
    test_preds = []

    # Loop over each base model and fit it
    for name, model in models.items():
        if ('xgb' in name) or ('lgb' in name):
            model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
            
        elif 'cat' in name :
                model.fit(
                    Pool(X_train_, y_train_, cat_features=cat_cols), eval_set=Pool(X_val, y_val, cat_features=cat_cols),
                    early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        else:
            model.fit(X_train_, y_train_)
            
        if name in trained_models.keys():
            trained_models[f'{name}'].append(deepcopy(model))

        test_pred = model.predict_proba(X_test)[:, 1]
        y_val_pred = model.predict_proba(X_val)[:, 1]

        score = roc_auc_score(y_val, y_val_pred)
        score_dict[name].append(score)
        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] ROC-AUC score: {score:.5f}')

        oof_preds.append(y_val_pred)
        test_preds.append(test_pred)

    # Use OptunaWeights
    optweights = OptunaWeights(random_state)
    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)

    score = roc_auc_score(y_val, y_val_pred)
    print(f'Ensemble [FOLD-{n} SEED-{random_state_list[m]}] ROC-AUC score {score:.5f} \n')
    ensemble_score.append(score)
    weights.append(optweights.weights)

    # Predict to X_test by the best ensemble weights
    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))
    oof_predss[X_val.index] = optweights.predict(oof_preds)

    gc.collect()


# In[35]:


# Calculate the mean score of the ensemble
mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Mean Optuna Ensemble {mean_score:.5f} ± {std_score:.5f} \n')

print('--- Optuna Weights---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name}: {mean_weight:.5f} ± {std_weight:.5f}')


# In[36]:


ec2_test_predss = test_predss


# In[37]:


def show_confusion_roc(oof, title='Model Evaluation Results'):
    f, ax = plt.subplots(1, 2, figsize=(16, 6))
    df = pd.DataFrame({'preds': oof[0], 'target': oof[1]})
    cm = confusion_matrix(df.target, df.preds.ge(0.5).astype(int))
    cm_display = ConfusionMatrixDisplay(cm).plot(cmap=my_palette, ax=ax[0])
    ax[0].grid(False)
    RocCurveDisplay.from_predictions(df.target, df.preds, ax=ax[1])
    ax[1].grid(True)
    plt.suptitle(f'{title}', fontsize=12, fontweight='bold')
    plt.tight_layout()

show_confusion_roc(oof=[oof_predss, y_train], title='EC2 OOF Evaluation Results')


# ### RF

# In[38]:


# Initialize singe RF model 
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Perform cross-val for ROC AUC scores
roc_auc_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')

# Perform cross-val for F1 scores
f1_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')

# Perform cross-validation and calculate precision scores
precision_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='precision')

# Perform cross-validation and calculate recall scores
recall_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='recall')

# Print the cross-validation scores
print("ROC AUC scores:", roc_auc_scores)
print("F1 scores:", f1_scores)
print("Precision scores:", precision_scores)
print("Recall scores:", recall_scores)

# Calculate the mean of the scores
mean_roc_auc = roc_auc_scores.mean()
mean_f1 = f1_scores.mean()
mean_precision = precision_scores.mean()
mean_recall = recall_scores.mean()

# Print the mean scores
print("Mean ROC AUC score:", mean_roc_auc)
print("Mean F1 score:", mean_f1)
print("Mean Precision score:", mean_precision)
print("Mean Recall score:", mean_recall)


# In[39]:


rf_predss = rf.predict_proba(X_test)
rf_predss


# In[40]:


ec2_test_predss_rf = rf_predss[:, 1]*0.5 + ec2_test_predss*0.5


# ## Second Method: TabPFN

# Here I decided to train my model only on the original data. This is the only way to use TabPFN as it can only work with datasets < 1000 rows. Since the original and synthesized data are different, and most likely LB testing data similar to synthesized data, the final result may not be very good. However, I decided to try this method anyway, at least to test how it works.

# In[41]:


# install TabPFN

get_ipython().system('pip install /kaggle/input/tabpfn-019-whl/tabpfn-0.1.9-py3-none-any.whl')
get_ipython().system('mkdir /opt/conda/lib/python3.10/site-packages/tabpfn/models_diff')
get_ipython().system('cp /kaggle/input/tabpfn-019-whl/prior_diff_real_checkpoint_n_0_epoch_100.cpkt /opt/conda/lib/python3.10/site-packages/tabpfn/models_diff/')


# In[42]:


from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from catboost import Pool, CatBoostClassifier
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from sklearn.model_selection import cross_val_score


# In[43]:


df_train = pd.read_csv("/kaggle/input/playground-series-s3e18/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s3e18/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s3e18/sample_submission.csv")
original_desc = pd.read_csv("/kaggle/input/ec-mixed-class/mixed_desc.csv")

target_col = ['EC1', 'EC2']

columns_to_keep = ['CIDs', 'BertzCT', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3v',
                   'Chi4n', 'EState_VSA1', 'EState_VSA2', 'ExactMolWt', 'FpDensityMorgan1',
                   'FpDensityMorgan2', 'FpDensityMorgan3', 'HallKierAlpha',
                   'HeavyAtomMolWt', 'Kappa3', 'MaxAbsEStateIndex', 'MinEStateIndex',
                   'NumHeteroatoms', 'PEOE_VSA10', 'PEOE_VSA14', 'PEOE_VSA6', 'PEOE_VSA7',
                   'PEOE_VSA8', 'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA3', 'VSA_EState9',
                   'fr_COO', 'fr_COO2', 'EC1_EC2_EC3_EC4_EC5_EC6']

original = original_desc.loc[:, columns_to_keep]

# There is probably a better way to do this, but that is was first came to my mind
feature1 = []
feature2 = []
for x in original['EC1_EC2_EC3_EC4_EC5_EC6']:
    feature1.append(int(x.split('_')[0]))
    feature2.append(int(x.split('_')[1]))

original['EC1'] = feature1
original['EC2'] = feature2

original.drop(columns = ['EC1_EC2_EC3_EC4_EC5_EC6', 'CIDs'], inplace=True)
original['id'] = original.reset_index().index

df_train.drop(columns = ['EC3', 'EC4', 'EC5', 'EC6'], inplace=True)


numerical_columns = ['BertzCT', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3v',
                   'Chi4n', 'EState_VSA1', 'EState_VSA2', 'ExactMolWt', 'FpDensityMorgan1',
                   'FpDensityMorgan2', 'FpDensityMorgan3', 'HallKierAlpha',
                   'HeavyAtomMolWt', 'Kappa3', 'MaxAbsEStateIndex', 'MinEStateIndex',
                   'NumHeteroatoms', 'PEOE_VSA10', 'PEOE_VSA14', 'PEOE_VSA6', 'PEOE_VSA7',
                   'PEOE_VSA8', 'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA3', 'VSA_EState9']

cat_cols = ['fr_COO', 'fr_COO2']


# In[44]:


# Concatenate train and original dataframes, and prepare train and test sets
train = pd.concat([original]).sample(frac=0.985, random_state=42)
test = df_test.copy()

X_train = train.drop(columns = target_col).reset_index(drop=True)
y_train = train.loc[:, target_col].reset_index(drop=True)
X_test = test.reset_index(drop=True)


# Category Encoders
X_train, X_test, _ = cat_encoder(X_train, X_test, cat_cols, encode='label')

# Drop_col
drop_cols = ['id']
X_train.drop(drop_cols, axis=1, inplace=True)
X_test.drop(drop_cols, axis=1, inplace=True)

del train, test, df_train, df_test

print(f"X_train shape :{X_train.shape} , y_train shape :{y_train.shape}")
print(f"X_test shape :{X_test.shape}")

X_train.head()


# In[45]:


# Weighted Ensemble uses XGBoost and TabPFN to create predictions
class WeightedEnsemble(BaseEstimator):
    def __init__(self):
        self.classifiers = [xgb.XGBClassifier(), TabPFNClassifier()]
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    
    def fit(self, X, y):
        unique_classes, y = np.unique(y, return_inverse=True)
        self.classes_ = unique_classes
        X = self.imputer.fit_transform(X)
        for classifier in self.classifiers:
            classifier.fit(X, y)
    
    def predict_proba(self, X):
        X = self.imputer.transform(X)
        probabilities = np.stack([classifier.predict_proba(X) for classifier in self.classifiers])
        averaged_probabilities = np.mean(probabilities, axis=0)
        class_0_est_instances = averaged_probabilities[:, 0].sum()
        others_est_instances = averaged_probabilities[:, 1:].sum()
        # Weighted probabilities based on class imbalance
        new_probabilities = averaged_probabilities * np.array([[1/(class_0_est_instances if i==0 else others_est_instances) for i in range(averaged_probabilities.shape[1])]])
        return new_probabilities / np.sum(new_probabilities, axis=1, keepdims=1)
    
# Define Model
model = WeightedEnsemble()


# In[46]:


# Perform cross-validation and compute ROC AUC scores for EC1
cv_scores = cross_val_score(model, X_train, y_train['EC1'], cv=5, scoring='roc_auc')

# Print the mean and standard deviation of the cross-validation scores
print("Cross-Validation ROC AUC Scores:")
print(cv_scores)
print(f"Mean ROC AUC: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")


# In[47]:


# Perform cross-validation and compute ROC AUC scores for EC2
cv_scores = cross_val_score(model, X_train, y_train['EC2'], cv=5, scoring='roc_auc')

# Print the mean and standard deviation of the cross-validation scores
print("Cross-Validation ROC AUC Scores:")
print(cv_scores)
print(f"Mean ROC AUC: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")


# ### Predict EC1

# In[48]:


model = WeightedEnsemble()
model.fit(np.array(X_train), np.array(y_train['EC1']))

ec1_proba = model.predict_proba(X_test)


# ### Predict EC2

# In[49]:


model = WeightedEnsemble()
model.fit(np.array(X_train), np.array(y_train['EC2']))

ec2_proba = model.predict_proba(X_test)


# # <h1 style = "font-family: Georgia;font-weight: bold; font-size: 30px; color: #1192AA; text-align:left">Submission</h1>

# ### Submission for TabPFN

# In[50]:


submission = sample_submission

submission['EC1'] = ec1_proba[:, 1]
submission['EC2'] = ec2_proba[:, 1]

submission.to_csv(f'submission_tab.csv', index=False)
submission


# In[51]:


def plot_distribution(filepath, sub, target_col):

    plt.figure(figsize=(16, 6))
    sns.set_theme(style="whitegrid")

    sns.kdeplot(data=sub, x=target_col, fill=True, alpha=0.5, common_norm=False, label=f"{target_col} Predict")

    plt.title('Predictive vs Training Distribution')
    plt.legend()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
for target_col in ['EC1', 'EC2']:
    plot_distribution('/kaggle/input/playground-series-s3e18', submission, target_col)


# ### Submission for Ensembles

# In[52]:


submission = sample_submission

submission['EC1'] = ec1_test_predss
submission['EC2'] = ec2_test_predss

submission.to_csv(f'submission_ens.csv', index=False)
submission


# In[53]:


def plot_distribution(filepath, sub, target_col):

    plt.figure(figsize=(16, 6))
    sns.set_theme(style="whitegrid")

    sns.kdeplot(data=sub, x=target_col, fill=True, alpha=0.5, common_norm=False, label=f"{target_col} Predict")

    plt.title('Predictive vs Training Distribution')
    plt.legend()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
for target_col in ['EC1', 'EC2']:
    plot_distribution('/kaggle/input/playground-series-s3e18', submission, target_col)


# In[54]:


ec2_test_predss_1 = ec2_test_predss.copy()
ec2_test_predss_1[ec2_test_predss_1 > 0.75] = 1


submission = sample_submission

submission['EC1'] = ec1_test_predss
submission['EC2'] = ec2_test_predss_1

submission.to_csv(f'submission_ens_1.csv', index=False)
submission


# In[55]:


def plot_distribution(filepath, sub, target_col):

    plt.figure(figsize=(16, 6))
    sns.set_theme(style="whitegrid")

    sns.kdeplot(data=sub, x=target_col, fill=True, alpha=0.5, common_norm=False, label=f"{target_col} Predict")

    plt.title('Predictive vs Training Distribution')
    plt.legend()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
for target_col in ['EC1', 'EC2']:
    plot_distribution('/kaggle/input/playground-series-s3e18', submission, target_col)


# In[56]:


ec2_test_predss_2 = ec2_test_predss.copy()
ec2_test_predss_2[ec2_test_predss_2 > 0.75] = 1
ec2_test_predss_2[ec2_test_predss_2 < 0.4] = 0
ec1_test_predss_2 = ec1_test_predss.copy()
ec1_test_predss_2[ec1_test_predss_2 > 0.8] = 1
ec1_test_predss_2[ec1_test_predss_2 < 0.25] = 0

submission = sample_submission

submission['EC1'] = ec1_test_predss
submission['EC2'] = ec2_test_predss_2

submission.to_csv(f'submission_ens_2.csv', index=False)
submission


# In[57]:


def plot_distribution(filepath, sub, target_col):

    plt.figure(figsize=(16, 6))
    sns.set_theme(style="whitegrid")

    sns.kdeplot(data=sub, x=target_col, fill=True, alpha=0.5, common_norm=False, label=f"{target_col} Predict")

    plt.title('Predictive vs Training Distribution')
    plt.legend()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
for target_col in ['EC1', 'EC2']:
    plot_distribution('/kaggle/input/playground-series-s3e18', submission, target_col)


# In[58]:


ec2_test_predss_3 = ec2_test_predss.copy()
ec2_test_predss_3[ec2_test_predss_3 > 0.7] = 1
ec1_test_predss_3 = ec1_test_predss.copy()
ec1_test_predss_3[ec1_test_predss_3 < 0.3] = 1

submission = sample_submission

submission['EC1'] = ec1_test_predss_3
submission['EC2'] = ec2_test_predss_3

submission.to_csv(f'submission_ens_3.csv', index=False)
submission


# ### Submission for Ensembles + TabPFN

# In[59]:


submission = sample_submission

submission['EC1'] = ec1_test_predss*0.9 + ec1_proba[:, 1]*0.1
submission['EC2'] = ec2_test_predss*0.7 + ec2_proba[:, 1]*0.3

submission.to_csv(f'submission.csv', index=False)
submission


# In[60]:


def plot_distribution(filepath, sub, target_col):

    plt.figure(figsize=(16, 6))
    sns.set_theme(style="whitegrid")

    sns.kdeplot(data=sub, x=target_col, fill=True, alpha=0.5, common_norm=False, label=f"{target_col} Predict")

    plt.title('Predictive vs Training Distribution')
    plt.legend()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
for target_col in ['EC1', 'EC2']:
    plot_distribution('/kaggle/input/playground-series-s3e18', submission, target_col)


# In[61]:


submission = sample_submission

submission['EC1'] = ec1_test_predss
submission['EC2'] = rf_predss[:, 1]

submission.to_csv(f'submission_rf_1.csv', index=False)
submission


# In[62]:


def plot_distribution(filepath, sub, target_col):

    plt.figure(figsize=(16, 6))
    sns.set_theme(style="whitegrid")

    sns.kdeplot(data=sub, x=target_col, fill=True, alpha=0.5, common_norm=False, label=f"{target_col} Predict")

    plt.title('Predictive vs Training Distribution')
    plt.legend()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
for target_col in ['EC2']:
    plot_distribution('/kaggle/input/playground-series-s3e18', submission, target_col)


# In[63]:


submission = sample_submission

submission['EC1'] = ec1_test_predss
submission['EC2'] = ec2_test_predss_rf

submission.to_csv(f'submission_rf_2.csv', index=False)
submission


# In[64]:


def plot_distribution(filepath, sub, target_col):

    plt.figure(figsize=(16, 6))
    sns.set_theme(style="whitegrid")

    sns.kdeplot(data=sub, x=target_col, fill=True, alpha=0.5, common_norm=False, label=f"{target_col} Predict")

    plt.title('Predictive vs Training Distribution')
    plt.legend()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
for target_col in ['EC2']:
    plot_distribution('/kaggle/input/playground-series-s3e18', submission, target_col)

