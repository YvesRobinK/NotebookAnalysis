#!/usr/bin/env python
# coding: utf-8

# In[1]:


# General imports:-
import numpy as np;
import pandas as pd;
from termcolor import colored;
from warnings import filterwarnings; filterwarnings('ignore');
from tqdm.notebook import tqdm;
from IPython.display import clear_output;
from gc import collect;

import matplotlib.pyplot as plt;
import seaborn as sns;
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Fiverr spammer prediction challenge
# 
# **This is a binary classifier challenge that predicts a potential spammer from his/ her activities on the Fiverr platform post registration.
# This is a tabular classification challenge**
# 
# Further details are available on the competition link-
# https://www.kaggle.com/competitions/predict-potential-spammers-on-fiverr

# In[2]:


# Initializing relevant global variables:-
target = 'label';
grid_specs = {'visible': 'True','which': 'both','color':'lightgrey','linestyle': '--','linewidth': 0.75};
nb_corr_ftre = 10;


# # Step1- Data import and basic checks:-
# 
# In this section, we import the data and pre-process it with basic checks including length, target balance, column information and null checks

# In[3]:


# Importing data:-
xytrain = pd.read_csv('../input/predict-potential-spammers-on-fiverr/train.csv', encoding= 'utf8', index_col= 'user_id');
xtest = pd.read_csv('../input/predict-potential-spammers-on-fiverr/test.csv', encoding= 'utf8',index_col= 'user_id');
sub_fl = pd.read_csv('../input/predict-potential-spammers-on-fiverr/sample_submission.csv', encoding= 'utf8');

print(colored(f"\nTrain-data\n", color = 'blue', attrs= ['bold', 'dark']));
display(xytrain.head(5).style.format(precision= 3));

print(colored(f"\nTest-data\n", color = 'blue', attrs= ['bold', 'dark']));
display(xtest.head(5).style.format(precision= 3));

print(colored(f"\nSubmission file\n", color = 'blue', attrs= ['bold', 'dark']));
display(sub_fl.head(5).style.format(precision= 3));

Ftre_Lst = xtest.columns;
print(colored(f"\nFeatures-list\n", color = 'blue', attrs= ['bold', 'dark']));
display(Ftre_Lst);

# Assessing data leakage:-
print(colored(f"\nAssessing data leakage\n", color = 'blue', attrs= ['bold', 'dark']));
_ = pd.concat([pd.Series(xytrain.index), pd.Series(xtest.index)], axis=0).value_counts();
display(_.loc[_ > 1]);

del _;
for i in range(3): collect(i);


# In[4]:


# Assessing target balance:-
fig,ax = plt.subplots(1,2,figsize= (12,6));
xytrain[target].value_counts().plot.bar(ax= ax[0]);
ax[0].set_title(f"Target balance check\n", color= 'tab:blue');
ax[0].grid(**grid_specs);
ax[0].set_yticks(np.arange(0,500001,50000))

xytrain[target].value_counts(normalize= True).plot.bar(ax= ax[1]);
ax[1].set_title(f"Target balance check-normalized\n", color= 'tab:blue');
ax[1].grid(**grid_specs);
ax[1].set_yticks(np.arange(0.0, 1.05,0.05))

plt.tight_layout();
plt.show();


# In[5]:


# Assessing column information:-
print(colored(f"\nFeature information for train data\n", color = 'blue', attrs= ['bold', 'dark']));
display(xytrain[Ftre_Lst].info());

print(colored(f"\nFeature information for test data\n", color = 'blue', attrs= ['bold', 'dark']));
display(xtest[Ftre_Lst].info());

# Collating null value information:-
_ = xytrain.isna().sum(axis=0);
print(colored(f"\nNull information for train-data\n", color = 'blue', attrs= ['bold', 'dark']));
display(_.loc[_ > 0]);

_ = xtest.isna().sum(axis=0);
print(colored(f"\nNull information for test-data\n", color = 'blue', attrs= ['bold', 'dark']));
display(_.loc[_ > 0]);


# In[6]:


# Collating numerical column distribution characteristics:-
print(colored(f"\nFeature descriptions for train data\n", color = 'blue', attrs= ['bold', 'dark']));
_ = xytrain[Ftre_Lst].nunique();
display(pd.concat((xytrain[Ftre_Lst].describe().transpose().drop(['count'], axis=1),_), axis=1).\
rename({0:'nb_unique'}, axis=1).style.format('{:,.2f}'));

print(colored(f"\nFeature descriptions for test data\n", color = 'blue', attrs= ['bold', 'dark']));
display(pd.concat((xtest[Ftre_Lst].describe().transpose().drop(['count'], axis=1),xtest[Ftre_Lst].nunique()), axis=1).\
rename({0:'nb_unique'}, axis=1).style.format('{:,.2f}'));

# Collating all constant features from the training data:-
Ftre_Constant = _.loc[_ == 1].index;
print(colored(f"\nTrain-set features with constant values- {Ftre_Constant}", color = 'blue', attrs= ['bold', 'dark']));


# # Step2- Null treatment
# 
# **In this step, we impute null values in the training data as below-**
# 1. We extract the 10 top correlated columns with the column X13 that elicits nulls
# 2. We extract other rows in the training data that have the same values for the strongly correlated columns to X13
# 3. We then extract the mode of this subset for X13 (for those indices) and use this as a simple imputer

# In[7]:


# Obtaining strongly correlated columns for X13:-
_ = xytrain.corr()[['X13']].dropna().drop(['X13'], axis=0);
_['Abs_Corr'] = abs(_);
_ = _.sort_values(['Abs_Corr'], ascending= False);

# Obtaining the mode of X13 values for the indices (rows) with the same combination as in null cases:-
keys = list(_.iloc[0:nb_corr_ftre+1].index)
_ = xytrain.loc[xytrain.X13.isna(), keys].merge(xytrain[['X13']+keys], how='left', left_on=keys, right_on=keys)['X13'].value_counts();

# Performing missing value imputation using the mode of X13 for matched rows:-
xytrain['X13'] = xytrain['X13'].fillna(_.index[0]).astype(np.int8);


# # Step3- Memory reduction
# 
# **In this step, we aim to reduce the dataset memory usage by at least 60-70%**

# In[8]:


def ReduceMemory(df:pd.DataFrame, df_label:str, Ftre_Constant: np.array = Ftre_Constant):
    """
    This function downsizes the column values using the min and max information and helps in reducing table memory.
    We also drop features with constant values as elicited in a yester step to clean up the data even further.
    
    Inputs- 
    1. df (analysis dataframe)
    2. df_label (label for train-test)
    3. Ftre_Constant (features with constant values to be excluded)
    
    Returns- df (memory reduced dataframe)
    """;   
    
    #  Dropping features with constant values:-
    df = df.drop(Ftre_Constant, axis=1);
    
    for col in df.select_dtypes(int):
        col_min, col_max = np.amin(df[col]), np.amax(df[col]);
        if (np.iinfo(np.int8).min <= col_min) & (np.iinfo(np.int8).max >= col_max): df[col] = df[col].astype(np.int8);
        elif (np.iinfo(np.int16).min <= col_min) & (np.iinfo(np.int16).max >= col_max): df[col] = df[col].astype(np.int16);
        elif (np.iinfo(np.int32).min <= col_min) & (np.iinfo(np.int32).max >= col_max): df[col] = df[col].astype(np.int32); 
        del col_min, col_max;
        collect();   
        
    print(colored(f"\n{df_label} data after memory reduction\n", color= 'blue', attrs= ['bold', 'dark']));
    display(df.info());
    collect();
    return df;


# In[9]:


# Implementing the memory reducer on the train-test data:-
xytrain = ReduceMemory(df=xytrain, df_label='Training');
xtest = ReduceMemory(df=xtest, df_label='Test');

# Updating the feature list:-
Ftre_Lst = xtest.columns;


# ***We have now reduced the training data memory usage from 180Mb to 27Mb, a whooping 82% reduction!! This table retains almost all the information in the yester table and is much smaller, causing faster training and easy processing.***

# # Step4- Data processing and feature engineering
# 
# **In this data processing layer, we aim to do the below-**
# 1. Generate visualizations for the data based on target values
# 2. Understand feature correlations
# 3. Create model development specific inferences
# 4. Perform feature reduction and selection

# In[10]:


# Plotting the features with hist-plots for better insights:- 
Ftre = xytrain[Ftre_Lst].nunique().loc[xytrain[Ftre_Lst].nunique() > 50].index;
print(colored(f"\nHistogram plots\n", color = 'blue', attrs = ['bold', 'dark']));

for i, col in tqdm(enumerate(Ftre)):
    fig, ax = plt.subplots(1,2, figsize= (25,6.5));
        
    sns.histplot(x = xytrain[[col, target]].loc[xytrain[target]==0, col], bins= 50, color = 'tab:blue', ax= ax[0]);
    ax[0].set_title(f"\n{col}- target = 0\n", color = 'tab:blue');
    ax[0].set(xlabel = '', ylabel='');
    ax[0].grid(**grid_specs);
    
    sns.histplot(x = xytrain[[col, target]].loc[xytrain[target]==1, col], bins= 50, color = 'tab:blue', ax = ax[1]);
    ax[1].set_title(f"\n{col}- target = 1\n", color = 'tab:blue');
    ax[1].set(xlabel = '', ylabel='');
    ax[1].grid(**grid_specs);
    
    plt.tight_layout();
    plt.show();
    collect();

del Ftre;
collect();


# In[11]:


# Plotting bar-plots for n-levels:-
Ftre = xytrain[Ftre_Lst].nunique().loc[xytrain[Ftre_Lst].nunique() <= 50].index;
print(colored(f"\nBar plots\n", color = 'blue', attrs = ['bold', 'dark']));

for i, col in tqdm(enumerate(Ftre)):
    fig, ax = plt.subplots(1,2, figsize= (25,6));
        
    xytrain[[col, target]].loc[xytrain[target] ==0, col].value_counts(normalize= True).sort_index().plot.bar(ax = ax[0]);
    ax[0].set_title(f"\n{col}- target = 0\n", color = 'tab:blue');
    ax[0].set(xlabel = '', ylabel='');
    ax[0].grid(**grid_specs);
    
    xytrain[[col, target]].loc[xytrain[target] ==1, col].value_counts(normalize= True).sort_index().plot.bar(ax = ax[1]);
    ax[1].set_title(f"\n{col}- target = 1\n", color = 'tab:blue');
    ax[1].set(xlabel = '', ylabel='');
    ax[1].grid(**grid_specs);
    
    plt.tight_layout();
    plt.show();
    collect();

del Ftre;
collect();


# In[12]:


# Creating an output structure for the correlations:-
Unv_Snp = pd.DataFrame(data=None, index= Ftre_Lst);

for method in tqdm(['pearson', 'spearman']):
    print();
    _ = pd.DataFrame(xytrain.corr(method=method)[target].drop([target], axis=0)).rename({target: 'Correlation'}, axis=1);
    _['Abs_Correlation'] = abs(_['Correlation']);
    
    fig, ax = plt.subplots(1,1, figsize = (20,8));
    _.sort_values('Abs_Correlation', ascending= False)['Correlation'].plot.bar(ax=ax, color= 'tab:blue');
    ax.set_title(f"Correlation analysis- {method.capitalize()}\n", color= 'tab:blue');
    ax.grid(**grid_specs);
    ax.set_yticks(np.arange(-0.3, 0.45, 0.03));
    ax.set(xlabel = '\ncolumns', ylabel = 'correlations\n')
    plt.tight_layout();
    plt.show();
    
    Unv_Snp = pd.concat((Unv_Snp, _[['Correlation']]), axis=1).rename({'Correlation': method}, axis=1);
    del _; 
    collect();
collect();


# In[13]:


# Plotting feature correlation heatmap:-
def Make_CorrHeatMap(df:pd.DataFrame):
    """
    This function develops the correlation heatmaps using Pearson and Spearman correlation methods for the analysis dataframe
    """;
    
    for i,method in tqdm(enumerate(['pearson', 'spearman'])):
        _ = df.corr(method = method);
        fig, ax = plt.subplots(1,1, figsize= (25,20));
        sns.heatmap(data=_,cbar=False, fmt='.1f', annot= True,linewidths=1.2, linecolor= 'white', mask= np.triu(np.ones_like(_)), 
                    cmap = 'Blues',ax=ax);
        ax.set_title(f"\nFeature correlation heatmap- {method.capitalize()}\n", fontweight= 'bold', fontsize= 12, color = 'tab:blue');
        plt.yticks(rotation = 0);
        plt.tight_layout();
        plt.show();
        del _;
        collect();
        print();


# In[14]:


Make_CorrHeatMap(df=xytrain.drop([target], axis=1));


# In[15]:


# Displaying the correlations between the target and other features:-
print(colored(f"\nCorrelation values- Pearson and Spearman\n", color=  'blue', attrs= ['dark', 'bold']));
display(Unv_Snp.style.format('{:.2%}'));


# **This ends the data visualization component for this analysis. We now proceed to the ML model design.**
# 
# **Key inferences:-**
# 1. The target column is highly imbalanced. Only 12000 spammers exist in the training data (around 3%)
# 2. All constant features are eliminated from the next steps
# 3. Features are mostly integers and many of them are categories. Handful of them are continuous.
# 4. Feature behaviour for target = 0 and target = 1 are interesting to observe
# 5. Collinearity issues need to be resolved in model development

# In[16]:


# Saving the reduced data for further usage- 
xytrain.to_csv('xytrain.csv');
xtest.to_csv('xtest.csv');
collect();

