#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool, metrics, cv
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import optuna
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import spearmanr


# # Load the data

# In[2]:


# load the training and test data
train = pd.read_csv("../input/novozymes-enzyme-stability-prediction/train.csv")
train_corrected = pd.read_csv("../input/novozymes-enzyme-stability-prediction/train_updates_20220929.csv")
test = pd.read_csv("../input/novozymes-enzyme-stability-prediction/test.csv")


# In[3]:


# check null 
train.isnull().sum()


# In[4]:


# correct data shape
train_corrected.shape


# In[5]:


train_corrected.isnull().sum()


# # Easily replace wrong train data with corrected train data

# In[6]:


# rows to drop
drop_seq_ids = train_corrected[train_corrected.isnull().sum(axis=1)==4]['seq_id']


# In[7]:


len(drop_seq_ids)


# In[8]:


train.shape


# In[9]:


# drop rows from train data which are not correct
train = train[~train.seq_id.isin(drop_seq_ids)]


# In[10]:


train.isnull().sum()


# In[11]:


# train data shape
train.shape


# In[12]:


# rows to replace values
display(train_corrected[train_corrected.isnull().sum(axis=1)!=4].head())
replace_seq_ids = train_corrected.loc[train_corrected.isnull().sum(axis=1)!=4,"seq_id"]


# In[13]:


train.shape


# In[14]:


# drop error rows where values are swapped
train = train[~train.seq_id.isin(replace_seq_ids)]


# In[15]:


train.isnull().sum()


# In[16]:


# get the correct rows
correct_df = train_corrected[train_corrected.isnull().sum(axis=1)!=4]


# In[17]:


train.head()


# In[18]:


# merge back the correct data
train = pd.concat([train, correct_df], ignore_index=True)


# In[19]:


train.isnull().sum()


# In[20]:


submission = pd.read_csv("../input/novozymes-enzyme-stability-prediction/sample_submission.csv")


# # EDA

# In[21]:


# print first few rows of data
train.head()


# In[22]:


test.head()


# In[23]:


# Check data shape
print(train.shape)
print(test.shape)


# In[24]:


# check for missing values
train.isnull().sum()


# In[25]:


train.isnull().sum()/len(train)*100


# Train data hs 3.38% missing values in data_source and 0.9868% missing values in pH columns.

# In[26]:


test.isnull().sum()


# In[27]:


# check datatypes
train.dtypes


# In[28]:


test.dtypes


# In[29]:


train.describe()


# In[30]:


test.describe()


# In[31]:


train.head()


# In[32]:


# Distribution of pH in train data
sns.histplot(data = train,x ="pH")
plt.title("Distribution of train pH")


# From the train data pH histogram, we can see that enzyme pH values don't follow a normal distribution.

# In[33]:


# Distribution of pH in train data
sns.histplot(data = train[train.pH!=7],x ="pH")
plt.title("Distribution of train pH without pH value 7")


# In[34]:


# boxplot
sns.boxplot(data = train,x ="pH")
plt.title("Boxplot of train pH")


# In[35]:


# percentage of neutral enzymes
np.round((train.pH == 7).mean()*100, 2)


# From the training data pH boxplot, we can see that the pH column has a large number of outliers. Median pH values in 7. Most of the enzymes have ph 7 i.e. 88.02%.

# In[36]:


test.describe()


# In[37]:


# no of unique values of pH in test data
test.pH.nunique()


# In[38]:


# test data pH value
test.pH.unique()


# In[39]:


(test.pH==8).mean()*100


# All the test enzymes have only a pH value of 8.

# In[40]:


test['pH'].hist()


# In[41]:


# tm
sns.histplot(data = train, x = "tm")
plt.title("Distribution of tm in Train data.")


# From the histogram, we can see that tm in train data is positively skewed(right-skewed).

# In[42]:


sns.boxplot(data = train,x ="tm")
plt.title("Boxplot of train tm")


# From the boxplot of tm in train data, we can see that tm in train data has a large number of outliers. It has a very high number of very large outlier values compared to a low number of very small outlier values.

# In[43]:


# sns.scatterplot(data = train,x="pH",y="tm",color="blue",alpha = 0.5)
sns.regplot(data = train,x="pH",y="tm",marker="+")
plt.title("Relationship between tm and pH");


# # Correlation Analysis

# In[44]:


train[['pH','tm']].corr()


# In[45]:


# Correlation test
sns.heatmap(train[['pH','tm']].corr())
plt.title("Correlation Heatmap between pH and tm")


# From the scatterplot and correlation value 0.028009, we can see that there is a very negligible positive correlation between pH and tm.

# # Quantitative analysis of Enzymes(Protein) sequences

# In[46]:


train.head()


# # Feature Engineering using biopython
# we can use biological features of proteins such as molecular weight, and amino acids percent to predict them.

# In[47]:


# first we create a ProteinAnalysis object
analysed_seq = ProteinAnalysis(train.protein_sequence[0])


# In[48]:


# using methods on the object we can get enzyme characteristics.
# for a complete list you can see this link: https://biopython.org/docs/1.76/api/Bio.SeqUtils.ProtParam.html
analysed_seq.molecular_weight()


# In[49]:


train.isnull().sum()


# In[50]:


# create sequence objects for train and test data protein_sequence and store them in the sequence column
train['sequence'] = train.protein_sequence.apply(ProteinAnalysis)
test['sequence'] = test.protein_sequence.apply(ProteinAnalysis)


# In[51]:


# make a count and each count value as a column
# Count standard amino acid
train.sequence[0].count_amino_acids()


# In[52]:


# make a dataframe of amino acid count
acid_df_train = pd.DataFrame(train.sequence.apply(lambda x: x.count_amino_acids()).tolist())
acid_df_test = pd.DataFrame(test.sequence.apply(lambda x: x.count_amino_acids()).tolist())


# In[53]:


# ad sequence id for merge
acid_df_train['seq_id'] = train['seq_id']
acid_df_test['seq_id'] = test['seq_id']


# In[54]:


train.shape


# In[55]:


acid_df_train.shape


# In[56]:


# merge amino acid count
train = train.merge(acid_df_train)
test = test.merge(acid_df_test)


# In[57]:


train.head()


# ## Calculate the aromaticity according to Lobry, 1994.
# aromaticity is related to the stability of enzymes.
# Definition:
# Aromaticity is defined as a property of the conjugated cycloalkenes which enhances the stability of a molecule due to the delocalization of electrons present in the π-π orbitals. Aromatic molecules are said to be very stable, and they do not break so easily and also react with other types of substances.
# Source: https://byjus.com/chemistry/aromaticity/

# In[58]:


train['aromaticity'] = train.sequence.apply(lambda x: x.aromaticity())
test['aromaticity'] = test.sequence.apply(lambda x: x.aromaticity())


# In[59]:


train['aromaticity'].hist()


# ## Calculate the instability index according to Guruprasad et al 1990.
# Related paper: https://pubmed.ncbi.nlm.nih.gov/2075190/

# In[60]:


train['instability_index'] = train.sequence.apply(lambda x: x.instability_index())
test['instability_index'] = test.sequence.apply(lambda x: x.instability_index())


# ## Calculate MW from Protein sequence.

# In[61]:


train['molecular_weight'] = train.sequence.apply(lambda x: x.molecular_weight())
test['molecular_weight'] = test.sequence.apply(lambda x: x.molecular_weight())


# ### Grand average of hydropathicity index
# Grand average of hydropathicity index (GRAVY) is used to represent the hydrophobicity value of a peptide, which calculates the sum of the hydropathy values of all the amino acids divided by the sequence length.
# Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3734225/

# In[62]:


train['gravy'] = train.sequence.apply(lambda x: x.gravy())
test['gravy'] = test.sequence.apply(lambda x: x.gravy())


# In[63]:


### Calculate the flexibility according to Vihinen, 1994.


# In[64]:


# train['flexibility'] = train.sequence.apply(lambda x: x.flexibility())
# test['flexibility'] = test.sequence.apply(lambda x: x.flexibility())


# ### Calculate the isoelectric point.
# Uses the module IsoelectricPoint to calculate the pI of a protein.

# In[65]:


train['isoelectric_point'] = train.sequence.apply(lambda x: x.isoelectric_point())
test['isoelectric_point'] = test.sequence.apply(lambda x: x.isoelectric_point())


# ### charge_at_pH
# The charge of a protein at given pH

# In[66]:


train['charge_at_pH'] = train[['pH','sequence']].apply(lambda row: row['sequence'].charge_at_pH(row['pH']), axis = 1)
train.head()


# In[67]:


test['charge_at_pH'] = test[['pH','sequence']].apply(lambda row: row['sequence'].charge_at_pH(row['pH']), axis = 1)


# ### Calculate the molar extinction coefficient.
# 
# Calculates the molar extinction coefficient assuming cysteines (reduced) and cystines residues (Cys-Cys-bond)

# In[68]:


train['molar_extinction_coefficient'] = train.sequence.apply(lambda x: x.molar_extinction_coefficient())
test['molar_extinction_coefficient'] = test.sequence.apply(lambda x: x.molar_extinction_coefficient())


# In[69]:


train['molar_extinction_coefficient_a'], train['molar_extinction_coefficient_a'] = zip(*train['molar_extinction_coefficient'])
test['molar_extinction_coefficient_a'], test['molar_extinction_coefficient_a'] = zip(*test['molar_extinction_coefficient'])


# ### Calculate fraction of helix, turn and sheet.
# The fraction of amino acids which tend to be in Helix, Turn or Sheet.

# In[70]:


train['secondary_structure_fraction'] = train.sequence.apply(lambda x: x.secondary_structure_fraction())
test['secondary_structure_fraction'] = test.sequence.apply(lambda x: x.secondary_structure_fraction())


# In[71]:


train['secondary_structure_fraction_helix'], train['secondary_structure_fraction_turn'],train['secondary_structure_fraction_turn_sheet']  =\
zip(*train['secondary_structure_fraction'])
test['secondary_structure_fraction_helix'], test['secondary_structure_fraction_turn'],test['secondary_structure_fraction_turn_sheet']  =\
zip(*test['secondary_structure_fraction'])


# In[72]:


train.describe()


# In[73]:


train['molecular_weight'].describe()


# In[74]:


train['gravy'].describe()


# In[75]:


train.head()


# In[76]:


sns.heatmap(train.corr())


# In[77]:


train.corr()


# https://biopython.org/wiki/ProtParam
# https://biopython.org/docs/1.76/api/Bio.SeqUtils.ProtParam.html

# # Fit a baseline model

# In[78]:


train[['secondary_structure_fraction']].head()


# In[79]:


train.columns


# In[80]:


train.head()


# ## Drop unnecessary columns

# In[81]:


# tm =  thermal stability
X = train.drop(['seq_id','data_source','sequence','molar_extinction_coefficient','secondary_structure_fraction','protein_sequence','tm'], axis=1)
y = train.tm


# In[82]:


X_test = test.drop(['seq_id','data_source','sequence','molar_extinction_coefficient','secondary_structure_fraction','protein_sequence'], axis=1)


# In[83]:


X.columns 


# In[84]:


X_test.columns


# In[85]:


# check column equality
sum(X.columns != X_test.columns)


# ## Split the data into train and valid set

# In[86]:


X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)


# In[87]:


predict_pool = Pool(X_test)


# In[88]:


train_pool = Pool(X_train, 
                  y_train)
test_pool = Pool(X_validation) 

# specify the training parameters 
model = CatBoostRegressor(loss_function='RMSE',verbose=0,random_seed=42)
#train the model
model.fit(train_pool)
# make the prediction using the resulting model
y_preds = model.predict(test_pool)
print(y_preds)
# performance metrics
rho, pval = spearmanr(y_validation,y_preds)
rho


# For evaluaton we've to use  Spearman's correlation coefficient. For the baseline model the Spearman's correlation coefficient is

# In[89]:


submission.head()


# # Hypearparamter Tuning using optuna

# In[90]:


# soruce: https://github.com/optuna/optuna-examples/blob/main/catboost/catboost_pruning.py
def objective(trial: optuna.Trial) -> float:


    param = {
        "depth": trial.suggest_int("depth", 1, 16),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    gbm = CatBoostRegressor(**param, task_type="GPU",random_seed=42)
    gbm.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=0,
        early_stopping_rounds=100
    )

    y_preds = gbm.predict(X_validation)
    # performance metrics
    rho, pval = spearmanr(y_validation,y_preds)
    return rho


# In[91]:


TUNE = True


# In[92]:


if TUNE:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


# In[93]:


# baseline 0.5816071043776863


# In[94]:


if TUNE:
    params = trial.params
else:
    params = {
         "depth": 12,
        "boosting_type": "Ordered",
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8837007062949978
    }


# In[95]:


model = CatBoostRegressor(**params,task_type='GPU',verbose=0)
#train the model
model.fit(train_pool)
# make the prediction using the resulting model
y_preds = model.predict(test_pool)
print(y_preds)


# ## Do prediction

# In[96]:


# performance metrics
rho, p = spearmanr(y_validation,y_preds)
print(rho)


# ## Make submission file

# In[97]:


test['tm'] =  model.predict(predict_pool)
test[['seq_id','tm']].to_csv("submission.csv",index=False)

