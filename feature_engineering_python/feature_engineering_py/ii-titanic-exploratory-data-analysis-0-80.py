#!/usr/bin/env python
# coding: utf-8

# # Titanic Exploratory Data Analysis
# ### Prediction of survival rate using 9 variables

# For this analysis we use the <b>Titanic Kaggle Competition</b> dataset found on the Kaggle Data
# Repository at the following location:
# 
# <a href=https://www.kaggle.com/c/titanic/data>Titanic Kaggle Competition Dataset</a>
# 
# The objective of the analysis is to classify passengers on the Titanic during the disaster
# of 1912 according to survival
# 
# We aim to achieve this by following the ML pipeline approach of deploying a variety of ML
# techniques to build a final predictive model with the highest possible accuracy. This
# particular analysis comprises 4 notebooks as follows:
# 
#  1. <i>titanic_baseline</i> - Baseline predictive models (quick and dirty) to
#  compare later results against
#  2. <i>titanic_eda</i> - <b>This notebook</b>, Exploratory Descriptive Analysis (EDA)
#  3. <i>titanic_features</i> - Perform feature engineering
#  4. <i>titanic_final_model</i> - Final model
# 
# We hope to gain valuable insights by following this process. The various steps in the
# process can be elaborated on as follows (the various notebooks will focus on different parts
#  of the process as indicated):
# 
# - Load data (<i>all notebooks</i>)
# - Prepare data
#     - Clean data (<i>notebook 2</i>)
#         - Missing values
#         - Outliers
#         - Erroneous values
#     - Explore data (<i>notebook 2</i>)
#         - Exploratory descriptive analysis (EDA)
#         - Correlation analysis
#         - Variable cluster analysis
#     - Transform Data (<i>notebook 3</i>)
#         - Engineer features
#         - Encode data
#         - Scale & normalise data
#         - Impute data (if not done in previous steps)
#         - Feature selection/ importance analysis
# - Build model (<i>notebooks 1 & 4</i>)
#     - Model selection
#     - Data sampling (validation strategy, imbalanced classification)
#     - Hyperparameter optimisation
# - Validate model (<i>notebooks 1 & 4</i>)
#     - Accuracy testing
# - Analysis of results (<i>notebook 1 & 4</i>)
#     - Response curves
#     - Accuracy analysis
#     - Commentary
# 
# The data dictionary for this dataset is as follows:
# 
# | Variable | Definition | Key |
# |----------|------------|-----|
# | survival | Survival	| 0 = No, 1 = Yes |
# | pclass   | Ticket class |	1 = 1st, 2 = 2nd, 3 = 3rd |
# |sex | Sex | male, female |
# |Age | Age in years | Continuous |
# |sibsp | # of siblings / spouses aboard the Titanic | 0, 1, 2, ..|
# |parch | # of parents / children aboard the Titanic | 0, 1, 2 ..|
# |ticket | Ticket number | PC 17599, STON/O2. 3101282, 330877 |
# |fare | Passenger fare | Continuous |
# |cabin | Cabin number | C123, C85, E46 |
# |embarked | Port of Embarkation	| C = Cherbourg, Q = Queenstown, S = Southampton |
# 
# Let us start the analysis for <b>notebook 2</b>!
# 
# This notebook follows on the previous notebook. We previously concluded that there is a very
#  strong signal in the data, even without any pre-processing. We also found that
#  several variables have potential for improving the model if pre-processed. The analysis
#  also suggests that improvement could be gained by feature engineering. In this notebook we
#  will gain an understanding of the data and consider ways of improving our existing score by
#   re-doing our missing data imputation and performing feature engineering.
# 
# We will also learn more about the domain i.e. the events surrounding the accident.
# 
# We will perform an analysis on the entire training dataset i.e. 891 records. We will sense
# check our findings against the test set to ensure the two are aligned. We will then make
# some high level improvements to our models based on our findings and see if our accuracy
# improves.
# 
# We start by discussing the data with domain experts, to build up an intuition about the data
# . In my case I consulted the most knowledgeable person in our household regarding the matter
#  i.e. my daughter whom has watched the movie several times. On asking her the question as to
#   who the most likely people to survive the Titanic accident she came up with the following
#   (without any hesitation or thought):
# 
# - The wealthier passengers, as they would have cabins on the higher decks, closer to the
# lifeboats.
# - Woman, children and older passengers, as at the time chivalry still existed, and it is
# likely that able-bodied younger men would have assisted these passengers with evacuation
# (which is not the case anymore today she however added :( ).
# - She also stated that individuals working on the Titanic staying on the lower decks
# were less likely to survive (which prompted me to think that these individuals probably also
#  did not pay a fare, or at least a very low fare...).
# 
# This is a nice place to start! Let us see what the data says.
# 
# Also, to start off with, thank you to the following authors for the inspiration and ideas (and
# some code:)) which I have obtained from your excellent work:
# 
# Gunes Evitan: https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial
# 
# Ata Saygın Odabaşı: https://www.kaggle.com/atasaygin/titanic-randomforestclassifier-and-visualization (had to use your map!)

# In[1]:


# Import libraries
from subprocess import call

import patsy
import folium as folium
from folium import plugins
from folium.plugins import HeatMap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.core.display import Image
from matplotlib import pyplot
from numpy import isnan
from patsy.highlevel import dmatrices, dmatrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.impute import KNNImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.linear_model import LogisticRegression
from visualize_titanic import plot_confusion_matrix, plot_roc_curve, \
    plotVar, plotAge, plot_feature_importance, plot_feature_importance_dec
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from pandasql import sqldf
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


# <div class="alert alert-block alert-info">
# <b>Load data</b>
# </div>

# In[2]:


# Import data
df_train = pd.read_csv('../input/titanic/train.csv', header = None,
                       names = ['passenger_id', 'survived', 'p_class', 'name', 'sex', 'age',
                                'sib_sp', 'parch', 'ticket', 'fare', 'cabin', 'embarked'],
                       index_col=False, usecols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                       skiprows=1, sep=',', skipinitialspace=True)


# #### Train Data

# In[3]:


df_train.head(20)
print(df_train.shape)


# In[4]:


# Import data
df_test = pd.read_csv('../input/titanic/test.csv', header = None,
                      names = ['passenger_id', 'p_class', 'name', 'sex', 'age', 'sib_sp',
                               'parch', 'ticket', 'fare', 'cabin', 'embarked'],
                      index_col=False, usecols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      skiprows=1, sep=',', skipinitialspace=True)


# #### Test Data

# In[5]:


df_test.head(20)
print(df_test.shape)
df_orig = df_test.copy()


# We will peform our analysis as follows:
# 
# - Missing data checks.
# - Model accuracy check.
# - EDA (without missing data replacement).
# - Variable adjustments & missing value imputation.
# - Model accuracy check
# 
# First thing we will attend to is to bring the data we ignored in the first round, back into our
# dataset i.e. name, ticket and cabin.

# <div class="alert alert-block alert-info">
# <b>Missing values</b>
# </div>

# In[6]:


# We use will use all of the variables from here onwards.
df_train = df_train.loc[:, ['survived', 'p_class', 'name', 'sex', 'age', 'sib_sp', 'parch',
                            'ticket', 'fare', 'cabin', 'embarked']]


# In[7]:


df_test = df_test.loc[:, ['p_class', 'name', 'sex', 'age', 'sib_sp', 'parch', 'ticket',
                          'fare', 'cabin', 'embarked']]


# We previously noted that there are missing values in the following fields: age, fare and
# embarked.
# 
# We quantify the exact number of missing values in the training set:

# In[8]:


# Check for null values
missing_values_train = df_train.isnull().sum()
missing_values_train = missing_values_train.to_frame(name='num_missing')
missing_values_train['perc_missing'] = (missing_values_train['num_missing']/df_train.shape[0])*100
for index, row in missing_values_train.iterrows():
    if (row['num_missing'] > 0):
        print ("For \"%s\" the number of missing values are: %d (%.0f%%)" %  (index,
                                                                     row['num_missing'],
                                                                    row['perc_missing']))


# Consider a sample of missing values from the training set:
# 

# In[9]:


df_train[df_train.isnull().any(axis=1)]


# Quantify the exact number of missing values in the test set:

# In[10]:


# Check for null values
missing_values_test = df_test.isnull().sum()
missing_values_test = missing_values_test.to_frame(name='num_missing')
missing_values_test['perc_missing'] = (missing_values_test['num_missing']/df_test.shape[0])*100
for index, row in missing_values_test.iterrows():
    if (row['num_missing'] > 0):
        print ("For \"%s\" the number of missing values are: %d (%.0f%%)" %  (index,
                                                                     row['num_missing'],
                                                                    row['perc_missing']))


# Consider a sample of missing values from the test set:

# In[11]:


# Actual null values
df_test[df_test.isnull().any(axis=1)]


# We observed 177 (20%) null values for <i>age</i>, 687 (77%) for <i>cabin</i> and 2 (0%) for
# <i>embarked</i> for <i>training data</i> and 86 (21%) null values for <i>age</i>, 327 (78%)
#  for <i>cabin</i> and 1 (0%) for <i>fare</i> for <i>testing data</i>.
# 
# The visualised missing values look as follows for the training set:

# In[12]:


#%matplotlib inline
_ = plt.figure(figsize=(20, 10))

# cubehelix palette is a part of seaborn that produces a colormap
cmap = sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
_ = sns.heatmap(df_train.isnull(), cmap=cmap)


# And as follows for the testing set:

# In[13]:


#%matplotlib inline
_ = plt.figure(figsize=(20, 10))

# cubehelix palette is a part of seaborn that produces a colormap
cmap = sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
_ = sns.heatmap(df_test.isnull(), cmap=cmap)



# From the evidence we have thus far we can conclude the following:
# 
# - The age (continuous) and cabin (categorical) variables have a significant number of missing
# values.
# - The cabin variable has in excess of 70% missing entries, which is substantive.
# - The missing values for these variables seem to be randomly scattered throughout the data.
# Without more information it is difficult to tell whether these variables are missing at
# random or for some systemic reason. We will do some more analysis to try and ascertain this.
# 
# It is important for us to decide on a missing value replacement strategy for age and cabin.
# The other variables have an insignficant number of missing values and hence simple
# imputation will be performed.
# 
# We will start by analysing the distributions for age first to get a feel for the data.

# In[14]:


# Continuous density plot
fig_missing, axes = plt.subplots(1, 1, figsize=(15, 12))

# Plot frequency plot/ histogram
_ = sns.histplot(x="age", kde=True, data=df_train, ax=axes, bins=40);
_ = axes.set(xlabel="Age", ylabel='Density');
axes.xaxis.label.set_size(24)
axes.yaxis.label.set_size(24)
axes.tick_params('y', labelsize = 20);
axes.tick_params('x', labelsize = 20);

## Continuous density plot
#fig_missing, axes = plt.subplots(1, 1, figsize=(15, 12))
#
## Plot frequency plot/ histogram
#_ = sns.histplot(x="age", kde=True, data=df_test, ax=axes, bins=40);
#_ = axes.set(xlabel="Age", ylabel='Density');
#axes.xaxis.label.set_size(24)
#axes.yaxis.label.set_size(24)
#axes.tick_params('y', labelsize = 20);
#axes.tick_params('x', labelsize = 20);
#


# In[15]:


print("Age summary statistics (training set):\n")
print(df_train['age'].describe())
print("\nAge summary statistics (testing set):\n")
print(df_test['age'].describe())


# The age variable is similarly distributed for both the training and test set i.e. there is a
#  skew to the right with a slight bump on the left for lower ages. Maximum age was 80 and
#  minimum age was around 2 months.
# 
# High level analysis does not seem to show any specific problematic values within either set
# of data.
# 
# Next we consider whether there is correlation between missing age values and the outcome
# variable i.e. survival. There are a significant number of missing age values, and if the
# missing values are correlated with the outcome we should consider a more complex mechanism
# for missing value substitution such as multiple imputation.

# In[16]:


fig_age, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), squeeze=False)
legend_labels = ['Died', 'Survived']
colors = ["lightslategrey", "#1F77B4"]

df_train['age_missing'] = df_train.apply(lambda row: 1 if np.isnan(row['age']) else 0, axis=1)

age_missing_stacked = df_train.loc[:, ["survived", "age_missing"]]
age_missing_stacked.index.name = "passenger_num"
ctable_survival_missing = pd.crosstab(age_missing_stacked.survived, age_missing_stacked
                         .age_missing, colnames=["Missing Age"], rownames=["Survived"])
ctable_survival_missing_perc = pd.crosstab(age_missing_stacked.survived, age_missing_stacked
                         .age_missing, colnames=["Missing Age"], rownames=["Survived"],
                                           normalize="index")

#ctable_survival_missing.columns = ["Not Missing", "Missing"]
#ctable_survival_missing["Survived"] = ["Died", "Survived"]
#ctable_survival_missing = ctable_survival_missing.set_index("Survived")
#print(ctable_survival_missing)

ctable_survival_missing_perc.columns = ["Not Missing", "Missing"]
ctable_survival_missing_perc["Survived"] = ["Died", "Survived"]
ctable_survival_missing_perc = ctable_survival_missing_perc.set_index("Survived")

axs = plt.gca()
axs.set_frame_on(False)
#age_miss_stack = ctable_survival_missing.plot.bar(stacked=True, ax=axs, color=colors)
age_miss_stack = ctable_survival_missing_perc.plot.bar(stacked=True, ax=axs, color=colors)
_ = plt.xlabel('Survival class', fontsize=20)
#_ = plt.ylabel('Percentage Missing', fontsize=20)
_ = plt.xticks([0, 1], legend_labels, fontsize=15, alpha=0.8)
_ = plt.xticks(fontsize=20)
#_ = plt.yticks(fontsize=20)

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False,
                labelbottom=True)

# Bit of a hack - TODO, clean up here.
#totals = ctable_survival_missing.iloc[:,0] + ctable_survival_missing_.iloc[:,1]
totals = ctable_survival_missing_perc.iloc[:,0] + ctable_survival_missing_perc.iloc[:,1]
#tot_arr = totals.to_numpy()

for i, rec in enumerate(age_miss_stack.patches):
    height = rec.get_height()
    j = i//2
    _ = age_miss_stack.text(rec.get_x() + rec.get_width() / 2,
              rec.get_y() + height / 2,
              "{:.0f}%".format(height/totals[j]*100),
              ha='center',
              va='bottom',
              color="w",
              fontsize = 15)

plt.tight_layout()


# We observe that there is a larger proportion of missing values in the group of individuals
# that died compared to the survival group, which suggests correlation between the
# missing values and outcome variable (survival).
# 
# We will now perform a Chi-Squared test for independence to determine whether there is
# correlation between missing age values and survival.

# In[17]:


survived = df_train.loc[df_train["survived"] == 1,:]
died = df_train.loc[df_train["survived"] == 0,:]

survived_cont = survived["age_missing"].value_counts(normalize=False)
died_cont = died["age_missing"].value_counts(normalize=False)

_ = survived_cont.rename("survived", inplace=True)
_ = died_cont.rename("died", inplace=True)

#cont_table = pd.concat([survived_cont, died_cont], axis=1)

print(ctable_survival_missing)

#stat, p, dof, expected = chi2_contingency(cont_table)
stat, p, dof, expected = chi2_contingency(ctable_survival_missing)
print('dof=%d' % dof)
print(expected)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')



# We can see from the results of the Chi-Squared test that the missing values in the age data
# is strongly correlated with the outcome of survival (p=0.008).
# 
# This suggests strongly that we will have to take care when imputing values for age i.e.
# inaccurate imputation might lead to an adverse effect on model accuracy. It is therefore
# important to ensure that missing values are imputed as accurately as possible, especially
# taking other variables into account i.e. missing values in particular groupings of values
# might have strong correlation with the response. An advance technique such as multiple
# imputation (MICE) or another predictive model such as K-Nearest Neighbour (KNN) is required
# to ensure age is imputed correctly for these sub-groupings of values.
# 
# This also indicates that we could use the missing values later during feature engineering
# and assess possibility of improved model performance based on additional variables built
# using the missing value information.
# 
# Next we will look at the distribution of missing age values in relation to the other
# variables to gain more insight into the missing values. We start with the continuous variables
# and therefore we consider the fare variable first. We start with some simple distribution
# plots (histograms).

# In[18]:


fig_age, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 12), squeeze=False)

ax1 = sns.histplot(data = df_train, x = "fare", hue = "age_missing", ax=axes[0][0],
                   legend=True)

_ = plt.xlabel('Fare', fontsize=20)
_ = plt.ylabel('Count', fontsize=20)
_ = plt.xticks(fontsize=20)
_ = plt.yticks(fontsize=20)
#ax1.legend.set_title("Missing Age")

plt.tight_layout()


# In[19]:


fig_age, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 12), squeeze=False)

df_test['age_missing'] = df_test.apply(lambda row: 1 if np.isnan(row['age']) else 0, axis=1)
_ = sns.histplot(data = df_test, x = "fare", hue = "age_missing", ax=axes[0][0])

_ = plt.xlabel('Fare', fontsize=20)
_ = plt.ylabel('Count', fontsize=20)
_ = plt.xticks(fontsize=20)
_ = plt.yticks(fontsize=20)
plt.tight_layout()


# It is evident by inspection that the number of missing age values is higher for lower fares
# in both the testing and training datasets. We now look at the same data plotted on
# histograms to further explore the relationship between missing values and fare. We now
# however add another dimension in that we also consider survival rate as an additional factor.

# In[20]:


sns.set_style("white")
ax = sns.catplot(x="survived", y="fare", hue="age_missing", kind="box", data=df_train,
                 height = 5, aspect = 1.5, legend=False)
_ = ax.set(ylim=(0, 300))
_ = plt.xlabel('Survived', fontsize=20)
_ = plt.ylabel('Fare', fontsize=20)
_ = plt.legend(title='Missing Age')
plt.show()


# It must be noted that we adapted the y-axis to cut off outlier values above 300. There were
# a few values clustered at approximately 500.
# 
# The boxplots shows the same pattern we observed when considering the histograms i.e. the
# distribution of fare values is different for the missing and non-missing (age) groups. For the
# missing age group values are concentrated around smaller fare values.
# 
# Additionally and interestingly this plot also shows that when taking survival into account
# the effect is enlarged i.e. for those who survived and paid higher fares age is typically
# not missing.
# 
# At this point we could do a correlation test between missing values and fare by means of
# using a logistic regression. We however leave as a later exercise as we have enough evidence
# to include fare in any imputation method used to impute missing values.
# 
# Next we look at the categorical variables. We start with passenger class travelled first.

# In[21]:


sns.set_style("white")
fig_age, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), squeeze=False)
legend_labels = ['Third', 'First', 'Second']
colors = ["lightslategrey", "#1F77B4"]

age_pclass_stacked = df_train.loc[:, ["p_class", "age_missing"]]
age_pclass_stacked.index.name = "passenger_num"
ctable_pclass_missing = pd.crosstab(age_pclass_stacked.p_class, age_pclass_stacked
                         .age_missing, colnames=["Missing Age"], rownames=["Passenger Class"])
ctable_pclass_missing_perc = pd.crosstab(age_pclass_stacked.p_class, age_pclass_stacked
                         .age_missing, colnames=["Missing Age"], rownames=["Passenger Class"],
                                           normalize="index")

#ctable_pclass_missing.columns = ["Not Missing", "Missing"]
#ctable_pclass_missing["Passenger Class"] = ["First", "Second", "Third"]
#ctable_pclass_missing = ctable_pclass_missing.set_index("Passenger Class")
#ctable_pclass_missing = ctable_pclass_missing.sort_values(by= 'Missing', axis=0,
#                                                          ascending=False)
print(ctable_pclass_missing)

ctable_pclass_missing_perc.columns = ["Not Missing", "Missing"]
ctable_pclass_missing_perc["Passenger Class"] = ["First", "Second", "Third"]
ctable_pclass_missing_perc = ctable_pclass_missing_perc.set_index("Passenger Class")
ctable_pclass_missing_perc = ctable_pclass_missing_perc.sort_values(by= 'Missing', axis=0,
                                                          ascending=False)

axs = plt.gca()
axs.set_frame_on(False)

age_pclass_stack = ctable_pclass_missing_perc.plot.bar(stacked=True, ax=axs, color=colors)
_ = plt.xlabel('Passenger Class', fontsize=20)
#_ = plt.ylabel('Percentage Missing', fontsize=20)
_ = plt.xticks([0, 1, 2], legend_labels, fontsize=15, alpha=0.8)
_ = plt.xticks(fontsize=20)
#_ = plt.yticks(fontsize=20)

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False,
                labelbottom=True)

# Bit of a hack - TODO, clean up here.
totals = [ctable_pclass_missing_perc.iloc[0,:].sum(), ctable_pclass_missing_perc.iloc[1,:].sum(),
          pd.Series(ctable_pclass_missing_perc.iloc[2,:]).sum()]

for i, rec in enumerate(age_pclass_stack.patches):
    height = rec.get_height()
    j = i%3
    bar_height = height/totals[j]*100
    #print ("Debug\n j: {}\nheight: {}\ntotals: {}\n".format(j, height, totals[j]))
    _ = age_pclass_stack.text(rec.get_x() + rec.get_width()/2,
              rec.get_y() + height/2,
              "{:.0f}%".format(bar_height),
              ha='center',
              va='bottom',
              color="w",
              fontsize = 15)

plt.tight_layout()


# It is evident by inspection that the number of missing age values is higher in the third
# passenger class than the other two classes.
# 
# Next we perform a Chi-Squared test of correlation.

# In[22]:


#stat, p, dof, expected = chi2_contingency(cont_table)
stat, p, dof, expected = chi2_contingency(ctable_pclass_missing)
print('dof=%d' % dof)
print(expected)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')



# We can see from the results of the Chi-Squared test that passenger class is strongly
# correlated with a value being missing (p=0.000).
# 
# At this point we can conclude that missing values are correlated with several key variables,
# which leads us to conclude that replacing missing values by using predictive model is
# necessary. We will use KNN imputation as it is frequently used for this type of substitution.
# We won't use MICE as we will impute one variable only i.e. age. MICE is overkill in this
# instance as its utility derives from the ability to impute several values concurrently. The
# other missing variables in our dataset we will deal with manually.
# 
# We will impute missing values later in this analysis at the time when we split the datasets
# into training and testing sets in order to avoid data leakage between training and testing
# sets. For now, we will continue with our EDA.
# 
# 

# The number of null values after missing values have been replaced in the test set is:

# In[23]:


# Replace missing values for test set
df_test = df_test.copy()
median = df_test['age'].median()
df_test['age'].fillna(median, inplace=True)
print("Number of null values in age column: {}".format(df_test['age'].isnull().sum()))

median = df_test['fare'].median()
df_test['fare'].fillna(median, inplace=True)
print("Number of null values in fare column: {}".format(df_test['fare'].isnull().sum()))
print("Dataframe dimension: {}".format(df_test.shape))
df_test = df_test.copy()


# We observe that the null values have been removed. We now have a dataset ready for further analysis -
# albeit a bit of a black box hack :) We will now do some very limited EDA just to get a feel for the data
# as previously discussed.

# <div class="alert alert-block alert-info">
# <b>Exploration of data</b>
# </div>

# We start by looking at the number of unique records per variable.

# In[24]:


print(df_train.nunique())


# There are no columns with only one value. We therefore retain all columns for ML purposes as there is
# enough variability to warrant using the data. There are many variables with fewer than 10 levels which
# could be considered as categorical. Based on our initial assessment of the data we will work with
# levels of measurement for the data as follows:
# 
# - p_class (ordinal) - we will revisit type of encoding later
# - sex (binary) - recode (female - yes or no)
# - age (continuous)
# - sib_sp (ordinal) - check correlation - revisit encoding
# - parch (ordinal) - check correlation - revisit encoding
# - fare (continuous)
# - embarked (nominal) - recode (one hot encode) - probably categorical
# 
# We start by separating continuous and categorical variables for further high level analysis.

# In[25]:


# We will use datasets for this analysis as follows:
# df_train, df_test:                Original data, supplemented with extra fields e.g.
#                                   missing values.
# df_train_con*, df_train_cat*:     Datasets used for plotting continuous and categorical
#                                   variables.
# df_train_trans*, df_test_trans*:  Transformed data, as result of enrichment/ wrangling/
#                                   feature engineering.
# X, y, X_train, X_test:            Training and testing sets.

# Separate continuous and categorical variables
names_con = ('fare', 'age')
names_con_plot = ('survived', 'fare', 'age')
names_cat = ('survived', 'p_class', 'sex', 'sib_sp', 'parch', 'embarked')
names_cat_test = ('p_class', 'sex', 'sib_sp', 'parch', 'embarked')
names_all_orig = ('survived', 'p_class', 'sex', 'sib_sp', 'parch', 'embarked', 'fare', 'age')

df_train_con = df_train.loc[:, names_con]
df_train_con_plot = df_train.loc[:, names_con_plot]
df_train_cat = df_train.loc[:, names_cat]

df_test_con = df_test.loc[:, names_con]
df_test_cat = df_test.loc[:, names_cat_test]

# Plotting label dictionary
plot_con = [('fare', 'Fare'),
            ('age', 'Age')]
plot_con_plot = [('survived', 'Survived'),
            ('fare', 'Fare'),
            ('age', 'Age')]
plot_cat = {'survived': ['Died', 'Survived'],
            'p_class': ['3rd', '1st', '2nd'],
            'sex': ['Male', 'Female'],
            'sib_sp': ['0', '1', '2', '4', '3', '8', '5'],
            'parch': ['0', '1', '2', '3', '5', '4', '6'],
            'embarked': ['Southampton', 'Cherbourg', 'Queenstown']}
plot_cat_plot = {'survived': 'Survival Rate',
            'p_class': 'Passenger Class Travelled',
            'sex': 'Gender',
            'sib_sp': '# Siblings or spouses',
            'parch': '# Parents or children',
            'embarked': 'Port of Embarkation'}


# ### High level overview
# 

# We observe that we have two candidates for continuous variables here (age and fare). With all the
# categorical variables present, it is likely that a tree model would be better suited to this problem
# unless significant feature engineering on categorical features is performed to ensure features are
# optimally encoded, transformed and scaled for a linear model or neural network.
# 
# Let's continue with the high level analysis.
# 
# The overall survival rate was as follows (based on the training dataset):

# In[26]:


_ = plt.figure()

# Plot outcome counts.a
outcome_counts = df_train_cat['survived'].value_counts(normalize = True)
legend_labels = ['Died', 'Survived']

# change the background bar colors to be light grey
bars = plt.bar(outcome_counts.index, outcome_counts.values, align='center', linewidth=0,
               color='lightslategrey')
# make one bar, the survived bar, a contrasting color
bars[1].set_color('#1F77B4')

# soften all labels by turning grey
_ = plt.xticks(outcome_counts.index, legend_labels, fontsize=15, alpha=0.8)
_ = plt.title('Survival Rate', fontsize=15, pad=30, alpha=0.8)

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False,
                labelbottom=True)

# Remove the frame - my method
ax = plt.gca()
ax.set_frame_on(False)

# Remove the frame of the chart - instructor's method
#for spine in plt.gca().spines.values():
#    spine.set_visible(False)

# direct label each bar with Y axis values
for bar in bars:
    _ = plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05,
                       str(round((bar.get_height()*100))) + '%', ha='center', color='w',
                       fontsize=15)

plt.show()


# The survival statistics are as follows:
# 

# In[27]:


print(df_train_cat['survived'].value_counts())
print("\n")

#print(df_train_cat['survived'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%')
#print("\n")
#


# We observe that 38% of passengers survived and 62% died. These statistics correspond with the narrative on survival
# rate quoted in the background information on Kaggle. There it is quoted that around 32% survived and 68% died. The
# sample we are working with is thus representative of the overall population, which is important to note.
# 
# We observe that the target variable contains unbalanced classes. We need to consider revisiting the unbalanced
# classes at a later stage - depending on the accuracy of our models. For now, we will forge ahead.
# 
# Next we will consider class level counts for categorical variables.

# #### Categorical variable overview
# 
# Class percentages:

# In[28]:


# Bar chart plot of categorical variables.
fig, ax = plt.subplots(2, 3, figsize=(20, 15));
base_color = '#1F77B4'
for variable, subplot in zip(names_cat, ax.flatten()):
    subplot.xaxis.label.set_size(24)
    subplot.yaxis.label.set_size(24)
    subplot.tick_params('y', labelsize = 20);
    subplot.tick_params('x', labelsize = 20);
    subplot.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False,
                        labelbottom=True)
    subplot.set_frame_on(False)

    outcome_counts = df_train_cat[variable].value_counts(normalize=True)
    bars = subplot.bar(outcome_counts.index.sort_values(), outcome_counts.values, align='center',
                       linewidth=0,
               color='lightslategrey')
    # make one bar, the highest value bar, a contrasting color
    bars[0].set_color('#1F77B4')

    plt.sca(subplot)
    _ = plt.xticks(outcome_counts.index.sort_values(), plot_cat[variable], fontsize=15, alpha=0.8)
    _ = plt.title(plot_cat_plot[variable], fontsize=15, pad=30, alpha=0.8)

    # direct label each bar with Y axis values
    for bar in bars:
        _ = plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05,
                           str(round((bar.get_height()*100))) + '%', ha='center', color='w',
                           fontsize=15)

    plt.tight_layout()


# Class counts:

# In[29]:


# Class level counts for categorical variables.
for variable in names_cat:
    #print(df_train_cat[variable].value_counts(normalize = True).mul(100).round(1).astype(str) +
    # '%')
    print(df_train_cat[variable].value_counts())
    print("\n")


# From the categorical variables we observe that there were approximately twice as many passengers in class 3 than either
# class 1 or 2. We also observe that there were nearly twice as many males as females on the Titanic. We also observe
# that more than two thirds of passengers did not have any siblings on board. Likewise we observe that more than two
# thirds did not have a father or child on board.
# 
# It is therefore fair to say that the majority of passengers were either couples or single travellers without children
# . In the case where families did travel, the majority of families had one or two children. Very few families with
# more children were on board the Titanic.
# 
# Many of these variables could contribute to correlation with survival at face value e.g. it
# stands to reason that preference would have been given in lifeboats to women and children,
# and that more affluent travellers would have had access to better lifeboats. We will however
#  test these hypotheses in this analysis.
# 
# We also see that more than two thirds of passengers departed from Southampton. The relative
# distribution between the different ports can be observed from the following heatmap.

# In[30]:


count_towns = df_train_cat.groupby(
    pd.Grouper(key='embarked')).size().reset_index(name='count')

latitude_embark = ['50.897', '49.6423', ' 51.84914']
longitude_embark = ['-1.404', '-1.62551', '-8.2975265']

count_towns['latitude_embark'] = latitude_embark
count_towns['longitude_embark'] = longitude_embark

m = folium.Map([49.922935, -6.068136], zoom_start=6, width='%100', height='%100')

heat_data = count_towns.groupby(["latitude_embark", "longitude_embark"])['count'].mean().reset_index().values.tolist()
_ = folium.plugins.HeatMap(heat_data).add_to(m)
m


# #### Continuous variable overview
# 
# We start by considering the age distribution of the passengers. At this point we do some limited
# EDA in that we will consider the age profiles of those who survived vs. those who died. As with
# many natural phenomena we expect age to have some influence on mortality.

# In[31]:


fig_age, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), squeeze=False)

_ = sns.kdeplot(data=df_train_con_plot.loc[(df_train_con_plot['survived'] == 0), 'age'], shade =
True, label = 'Died')
_ = sns.kdeplot(data=df_train_con_plot.loc[(df_train_con_plot['survived'] == 1), 'age'], shade =
True, label = 'Survived')
_ = plt.xlabel('Age', fontsize=20)
_ = plt.ylabel('Density', fontsize=20)
_ = plt.xticks(fontsize=20)
_ = plt.yticks(fontsize=20)
_ = plt.legend(fontsize=15)
plt.show()


# From the age distribution plot we can see that more children under the age of 15 survived than
# died in the incident. We can see that more individuals between the ages of 20 and 40 died than
# survived. We can also see that more individuals above the age of 80 survived compared to dying.
# 
# We can also see that the majority of individuals on the cruise were between the ages of 20 to 40.
# There were fewer teenagers compared to children under 10. There were comparatively fewer elderly
# people on board i.e. above 60.

# In[32]:


# 5 number summary.
df_train_con.describe()


# We now consider the fare distribution too. As with age, our assumption would be that fare has a
# correlation with mortality too.
# 
# The fare distribution is severely skewed to the right. The kurtosis of the plot is very high with
#  most values clustered closely around the median value of 14. There was a non-significant but
#  relatively smaller number of fares spread between teh values of 30 and 500.
# 
# The age distribution was as previously discussed, with a minimum of 6 months and maximum of 80
# years old. The distribution is fairly symmetrical with a slight skew to the right. There is a
# young child bump to the left of the distribution.
# 
# We now do a more in depth visual analysis of the correlation between survival and fare and age
# respectively. We do this by creating a set of Violin plots.

# In[33]:


# Continuous density plot
fig_continuous, axes = plt.subplots(nrows=len(names_con_plot)-1, ncols=2, figsize=(15, 12))

# Plot frequency plot/ histogram
_ = sns.histplot(x=plot_con[0][0], kde=True, data=df_train_con_plot, ax=axes[0][0], bins=40);
_ = axes[0][0].set(xlabel=plot_con[0][1], ylabel='Density');
axes[0][0].xaxis.label.set_size(24)
axes[0][0].yaxis.label.set_size(24)
axes[0][0].tick_params('y', labelsize = 20);
axes[0][0].tick_params('x', labelsize = 20);

# Plot violin plot
_ = sns.violinplot(x='survived', y=plot_con[0][0], data=df_train_con_plot, ax=axes[0][1]);
_ = axes[0][1].set(xlabel='', ylabel=plot_con[0][1]);
axes[0][1].xaxis.label.set_size(24)
axes[0][1].yaxis.label.set_size(24)
axes[0][1].tick_params('y', labelsize = 20);
axes[0][1].tick_params('x', labelsize = 20);
_ = axes[0][1].set_xticklabels(['Died', 'Survived'])

# Plot frequency plot/ histogram
_ = sns.histplot(x=plot_con[1][0], kde=True, data=df_train_con_plot, ax=axes[1][0], bins=40);
_ = axes[1][0].set(xlabel=plot_con[1][1], ylabel='Density');
axes[1][0].xaxis.label.set_size(24)
axes[1][0].yaxis.label.set_size(24)
axes[1][0].tick_params('y', labelsize = 20);
axes[1][0].tick_params('x', labelsize = 20);

# Plot violin plot
_ = sns.violinplot(x='survived', y=plot_con[1][0], data=df_train_con_plot, ax=axes[1][1]);
_ = axes[1][1].set(ylabel=plot_con[1][1], xlabel='');
axes[1][1].xaxis.label.set_size(24)
axes[1][1].yaxis.label.set_size(24)
axes[1][1].tick_params('y', labelsize = 20);
axes[1][1].tick_params('x', labelsize = 20);
_ = axes[1][1].set_xticklabels(['Died', 'Survived'])

plt.tight_layout()



# The violin plot for <i>fare</i> indicates that there is correlation between fare and survival as
# more people paying a low fare died and chances of survival increased for higher fares, as well as
#  lower fares close to zero (possibly for children travelling at very low cost).
# 
# The plot for <i>age</i> indicates a similar pattern with higher survival for children below 10
# and higher mortality between ages of 20 and 40. The relative likelihood of survival increases
# again around 40 years of age as you go into the older ages.
# 
# We will investigate these observations in more detail in our next Notebook.
# 
# Lastly we will look at the Box plots for both age and fare. We do this to get a better feel for
# the spread of the data and the possibility of outliers. This will be important for us to consider
#  whether further analysis and possible processing of variables are required in later stages.

# In[34]:


# Boxplot of continuous variables
medianprops = {'color': 'magenta', 'linewidth': 2}
boxprops = {'color': 'black', 'linestyle': '-', 'linewidth': 2}
whiskerprops = {'color': 'black', 'linestyle': '-', 'linewidth': 2}
capprops = {'color': 'black', 'linestyle': '-', 'linewidth': 2}
flierprops = {'color': 'black', 'marker': 'x', 'markersize': 20}

_ = df_train_con.plot(kind='box', subplots=True, figsize=(20, 8), layout=(1,2), fontsize = 20,
                      medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops,
                      capprops=capprops, flierprops=flierprops);
_ = plt.tight_layout();
_ = plt.show();


# The distributions of the <i>fare</i> and <i>age</i> variables show that fare is skewed heavily to
#  the right, with the median skewed to the left of the distribution as expected. The values in the
#   final quintile are spread over wide area with quite a few outliers. This distribution is heavy
#   tailed, as can be expected of many financial distributions. We observe that there are zero
#   values, and there is a large peak in the bin containing zero. We will need to investigate this
#   group of travellers as they could be different from the general population. The heavy tail and
#   many outliers are also good candidates for further processing.
# 
# The <i>age</i> distribution is fairly symmetrical, with a few outliers to the right, but nothing
# out of the ordinary. Most of the values are bundled symmetrically around the median of 28, which
# is quite a young age for the average traveller. The large spike in ages at this interval is also
# concerning and seems out of place. This will need to be further investigated.

# ### Correlation Analysis

# At this point we have learnt the following about the data:
# 
# - Ascertained best missing value strategy based on data characteristics:
#     - Use KNN imputation for age variable due to strong correlation between missing age
#     variables and outcome variable.
#     - Use median and mode substitution for remaining missing values, as there are few
#     missing values and more complicated methods are hence not warranted.
#     - Missing values in cabin have to be assessed given a transformation to the variable
#     which we must still decide on.
# - Analysis to date (Decision Tree feature strength) and the fact that a gender only model has
#  an accuracy of 77% has shown that gender is very strongly correlated with survival rate.
# - There are many other data elements correlated with survival rate as well as with each other.
# - The majority of variables are categorical.
# - The best performing models are those that have been adjusted for overfitting e.g. MLP
# models with regularisation parameters. Given the small dataset, many correlated variables
# and overpowering effect of a few strongly correlated variables the models created to date
# are very prone to overfitting.
# 
# The next step in our analysis is to better understand correlation with the response as well
# as between feature variables. This analysis will enable to decide how to best engineer
# better features that capture the correlation without overfitting. The problem we face is
# that the strong correlation between gender and survival is in effect eclipsing all other
# correlations and hence reducing their effect. We need to find a way to combine variables to
# create a more balanced representation of the correlations.
# 
# As we have a binary response variable, Pearson's Correlation will be of no use here as it
# assumes a normal distribution and Homoscedasticity which is clearly not the case (I have
# noticed that many of the Titanic analyses here on Kaggle use Pearson's correlation heatmap
# as implemented by the Pandas <i>corr()</i> function, which has no value or relevance in this
# setting).
# 
# We will perform correlation analysis as follows:
# 
# - Univariate Logistic Regression to measure correlation between categorical response and
# continuous feature variables (age, fare)
# - Chi-Squared test for association for correlation between categorical response and
# categorical feature variables (ticket class, ticket number, # siblings, # parents, fare, cabin
# number, port of embarkation).
# 
# We start by first performing some transformations on our dataset to get the data into a
# format enabling further analysis e.g. encode categorical variables for numeric analysis.
# 
# We start by considering the distribution of values for the "deck" variable we create from
# the original "cabin" variable by taking the first character from the "cabin" field.

# In[35]:


# We will now transform some variables by grouping categories together based on our EDA
# analysis. We also encode categorical variables to numeric values in order to do the ML
# analysis.
# These transformations would not result in data leakage, and can hence be done before we
# split the data into training and testing sets.

# Make a copy of original dataset before imputation - we need the original for further
# analysis.
df_train_trans = df_train.copy()
df_test_trans = df_test.copy()

# Creating Deck field from the first letter of the cabin field (we create a new category for
# missing, which is called M). As this is a categorical variable we will leave the missing
# value field as is.
df_train_trans['deck'] = df_train_trans['cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
df_test_trans['deck'] = df_test_trans['cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')


# In[36]:


df_train_trans['deck'].value_counts()


# We observe that most of the values are missing. Seeing as this is a categorical variable
# creating a "missing category might be insightful". Next we consider the proportion of
# survivors within each category.

# In[37]:


_ = plt.title('Survival rate by Deck')
ax = sns.barplot(x='deck', y='survived', data=df_train_trans).set_ylabel ('Survival Rate')
_ = plt.xlabel('Deck')


# There are many values in the smaller deck categories, which we can collapse into other decks
#  based on the proportion of survivors in each category.
# The missing values do have a significantly lower survival rate and will hence be very useful
#  information for the model to use.
# There is only one passenger on deck T, and the test set has no values for deck T. The closest
#  category is deck 'A' (based on Googling the deck placement on the ship), so we change the
#  single occurrence of T to category A.

# In[38]:


# There is only one passenger on deck T and the test set has no values for deck T.
# The closest category is deck 'A' (checking on deck arrangements image found via Google), so
# we change all occurrences of T to A.
idx = df_train_trans[df_train_trans['deck'] == 'T'].index
df_train_trans.loc[idx, 'deck'] = 'A'

## Some of the classes have very few values, we group adjacent classes together.
df_train_trans['deck'] = df_train_trans['deck'].replace(['A', 'B', 'C'], 'ABC')
df_train_trans['deck'] = df_train_trans['deck'].replace(['D', 'E'], 'DE')
df_train_trans['deck'] = df_train_trans['deck'].replace(['F', 'G'], 'FG')

df_test_trans['deck'] = df_test_trans['deck'].replace(['A', 'B', 'C'], 'ABC')
df_test_trans['deck'] = df_test_trans['deck'].replace(['D', 'E'], 'DE')
df_test_trans['deck'] = df_test_trans['deck'].replace(['F', 'G'], 'FG')


# In[39]:


df_train_trans['deck'].value_counts()
df_train_trans.groupby('deck').survived.mean()


# In[40]:


#df_test_trans['deck'].value_counts()

##%% md

#The values for the training and test sets look aligned. We don't have survival data for the
#test set (obviously!) so cannot check on this distribution. We can however check form the
#training set.


# In[41]:


_ = plt.title('Survival rate by Deck')
ax = sns.barplot(x='deck', y='survived', data=df_train_trans).set_ylabel ('Survival Rate')
_ = plt.xlabel('Deck')


##%% md

#Not a great split as most of the values are missing, but not bad considering the information
# we are working with.
#
#Next we consider titles extracted from the "name" field.


# In[42]:


# Next we extract the title variable from the name field
df_train_trans['title'] = df_train_trans['name'].str.split(', ', expand=True)[1].str.split('.',
                                                                            expand=True)[0]
# Next we extract the title variable from the name field
df_test_trans['title'] = df_test_trans['name'].str.split(', ', expand=True)[1].str.split('.',
                                                                            expand=True)[0]


# In[43]:


df_train_trans['title'].value_counts()


# We observe many values with few occurrences. We will group these into the larger four
# categories.

# In[44]:


df_train_trans['is_married'] = 0
df_train_trans['is_married'].loc[df_train_trans['title'] == 'Mrs'] = 1

df_test_trans['is_married'] = 0
df_test_trans['is_married'].loc[df_test_trans['title'] == 'Mrs'] = 1


# In[45]:


df_train_trans['title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)
df_test_trans['title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)

df_train_trans['title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer', 'Rev',
                                 'Dr'], 'Mr', inplace=True)
df_test_trans['title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer', 'Rev',
                                'Dr'], 'Mr', inplace=True)


# In[46]:


df_train_trans['title'].value_counts()
df_train_trans.groupby('title').survived.mean()


# In[47]:


_ = plt.title('Survival rate by Title')
ax = sns.barplot(x='title', y='survived', data=df_train_trans).set_ylabel ('Survival Rate')
_ = plt.xlabel('Title')


# Here we get a very nice split with a large proportion of values at a considerably lower
# survival rate.
# 
# We now encode the gender/ sex value into numeric values.
# 
# 

# In[48]:


df_train_trans['sex'].value_counts()
# Transform sex variable - don't need one hot encoding as variable is binary
df_train_trans['sex'] = df_train_trans['sex'].apply(lambda x: 1 if x == 'female' else 0)
# Same transformation for test set - don't need one hot encoding as variable is binary
df_test_trans['sex'] = df_test_trans['sex'].apply(lambda x: 1 if x == 'female' else 0)


# In[49]:


df_train_trans['sex'].value_counts()
df_train_trans.groupby('sex').survived.mean()


# In[50]:


legend_labels = ['Male', 'Female']

_ = plt.title('Survival rate by Gender')
ax = sns.barplot(x='sex', y='survived', data=df_train_trans).set_ylabel ('Survival Rate')
_ = plt.xlabel('Gender')
_ = plt.xticks([0, 1], legend_labels)


# We already know that sex is a very strong indicator of survival.
# 
# We now impute the missing values in the embarked field and create a family size feature.

# In[51]:


# Replace embarked with mode training set - no values missing in test set, so not required
# to further impute. Some leakage takes place here, but only one value so not important -
# TODO: fix this, just as a matter of principle.
train_emb_mode = df_train_trans['embarked'].mode()
df_train_trans['embarked'].fillna(train_emb_mode.iloc[0], inplace=True)


# In[52]:


# Create family size feature
df_train_trans['fam_num'] = df_train_trans['sib_sp'] + df_train_trans['parch'] + 1
df_test_trans['fam_num'] = df_test_trans['sib_sp'] + df_test_trans['parch'] + 1

# Create family size groupings
df_train_trans['fam_size'] = pd.cut(df_train_trans.fam_num, [0,1,4,7,11], labels=['single',
                           'small', 'large', 'very_large'])
df_test_trans['fam_size'] = pd.cut(df_test_trans.fam_num, [0,1,4,7,11], labels=['single',
                           'small', 'large', 'very_large'])

# Now we One Hot Encode Categorical variables. We leave the dimension variables for now, as
# we might generate some cross terms later. We don't One Hot Encode variables with missing
# values e.g. age, as we will impute these during training, and will One Hot Encode at that
# stage.

# Transform embarked and deck variables for training set
categorical_cols = ['embarked', 'deck', 'title', 'p_class', 'fam_size']
df_train_trans['dim_embarked'] = df_train_trans['embarked']
df_train_trans['dim_deck'] = df_train_trans['deck']
df_train_trans['dim_title'] = df_train_trans['title']
df_train_trans['dim_p_class'] = df_train_trans['p_class']
df_train_trans['dim_fam_size'] = df_train_trans['fam_size']
df_train_trans = pd.get_dummies(df_train_trans, columns = categorical_cols, drop_first=True)

# Transform embarked and deck variables for test set
df_test_trans['dim_embarked'] = df_test_trans['embarked']
df_test_trans['dim_deck'] = df_test_trans['deck']
df_test_trans['dim_title'] = df_test_trans['title']
df_test_trans['dim_p_class'] = df_test_trans['p_class']
df_test_trans['dim_fam_size'] = df_test_trans['fam_size']
df_test_trans = pd.get_dummies(df_test_trans, columns = categorical_cols, drop_first=True)


# In[53]:


df_train_trans['dim_fam_size'].value_counts()
df_train_trans.groupby('dim_fam_size').survived.mean()


# In[54]:


legend_labels = ['Single', 'Small', 'Large']
_ = plt.title('Survival rate by Family size')
ax = sns.barplot(x='dim_fam_size', y='survived', data=df_train_trans).set_ylabel ('Survival '
                                                                                'Rate')
_ = plt.xlabel('Family Size')
_ = plt.xticks([0, 1, 2], legend_labels)


# Fairly good split on family size too. Clearly mid-sized families had a higher survival rate.
# 
# Next we consider what information we can extract from the "ticket" field. We group tickets
# by name and see if the counts yield any information.

# In[55]:


df_train_trans['ticket'].value_counts()
df_train_trans.groupby('ticket').survived.mean()


# We observe that some ticket numbers have duplicates. We also observe that ticket numbers
# typically either survived or some members perished, suggesting that tickets were sold in
# batches to families/ groups.
# We therefore group ticket numbers by frequency and see if there is any value in this grouping.

# In[56]:


# Now we create a variable
df_train_trans['ticket_freq'] = df_train_trans.groupby('ticket')['ticket'].transform('count')
df_test_trans['ticket_freq'] = df_test_trans.groupby('ticket')['ticket'].transform('count')


# Let us see what the survival distributions for ticket groups of different sizes are.

# In[57]:


fig, axs = plt.subplots(figsize=(12, 9))
_ = sns.countplot(x='ticket_freq', hue='survived', data=df_train_trans)

_ = plt.xlabel('Ticket Frequency', size=15, labelpad=20)
_ = plt.ylabel('Passenger Count', size=15, labelpad=20)
_ = plt.tick_params(axis='x', labelsize=15)
_ = plt.tick_params(axis='y', labelsize=15)

_ = plt.legend(['Died', 'Survived'], loc='upper right', prop={'size': 15})
_ = plt.title('Count of Grouped Tickets', size=15, y=1.05)

plt.show()
#df_train_trans.head()

##%% md

#We clearly observe that tickets bought in isolation had a much higher death rate. We
#therefore group the variable as a binary indicator variable.
#
#There is not much variation in the lower frequencies, so we group them all together in one
#category.


# In[58]:


df_train_trans['ticket_freq'] = df_train_trans['ticket_freq'].apply(lambda x: 1 if x == 1
else 0)
# Same transformation for test set - don't need one hot encoding as variable is binary
df_test_trans['ticket_freq'] = df_test_trans['ticket_freq'].apply(lambda x: 1 if x == 1
else 0)

df_train_trans['ticket_freq'].value_counts()


# In[59]:


fig, axs = plt.subplots(figsize=(12, 9))
_ = sns.countplot(x='ticket_freq', hue='survived', data=df_train_trans)

_ = plt.xlabel('Ticket Frequency', size=15, labelpad=20)
_ = plt.ylabel('Passenger Count', size=15, labelpad=20)
_ = plt.tick_params(axis='x', labelsize=15)
_ = plt.tick_params(axis='y', labelsize=15)

_ = plt.legend(['Died', 'Survived'], loc='upper right', prop={'size': 15})
_ = plt.title('Count of Grouped Tickets', size=15, y=1.05)

plt.show()


# This clearly looks better.
# 

# In[60]:


names_all = list(df_train_trans.columns)
print(names_all)


# In[61]:


# Update dataframe fieldname values
drop_cols = ['name', 'sib_sp', 'parch', 'ticket',
       'cabin', 'age_missing', 'fam_num', 'dim_embarked', 'dim_deck',
       'dim_title', 'dim_p_class', 'dim_fam_size']

# These stay static
names_con = ('fare', 'age')
names_con_plot = ('survived', 'fare', 'age')

# These change depending on prior analyses
names_cat = names_all.copy()
for x in drop_cols:
    names_cat.remove(x)
for x in ['survived', 'age', 'fare']:
    names_cat.remove(x)

print("names_cat: {}".format(names_cat))

names_cat_plot = names_all.copy()
for x in drop_cols:
    names_cat_plot.remove(x)
for x in ['age', 'fare']:
    names_cat_plot.remove(x)

for x in drop_cols:
    names_all.remove(x)
for x in ['survived']:
    names_all.remove(x)


# Next we run a uni-variate Logistic regression for each categorical variable on its own and
# record the accuracy scores obtained.

# In[62]:


X_train = df_train_trans.loc[:, names_cat]
y_train = df_train_trans.loc[:, "survived"]

logval = LogisticRegression(fit_intercept = False)

efs1 = EFS(logval,
           min_features=1,
           max_features=1,
           scoring='accuracy',
           print_progress=True,
           cv=5)

efs1 = efs1.fit(X_train, y_train, custom_feature_names=names_cat)

print('Best accuracy score: %.2f' % efs1.best_score_)
print('Best subset (indices):', efs1.best_idx_)
print('Best subset (corresponding names):', efs1.best_feature_names_)

#efs1 = efs1.fit(X, y, custom_feature_names=feature_names)

df_efs = pd.DataFrame.from_dict(efs1.get_metric_dict()).T
df_efs.sort_values('avg_score', inplace=True, ascending=False)

metric_dict = efs1.get_metric_dict()

fig = plt.figure()
k_feat = sorted(metric_dict.keys())
avg = [metric_dict[k]['avg_score'] for k in k_feat]

upper, lower = [], []
for k in k_feat:
    upper.append(metric_dict[k]['avg_score'] +
                 metric_dict[k]['std_dev'])
    lower.append(metric_dict[k]['avg_score'] -
                 metric_dict[k]['std_dev'])

plt.fill_between(k_feat,
                 upper,
                 lower,
                 alpha=0.2,
                 color='blue',
                 lw=1)

_ = plt.plot(k_feat, avg, color='blue', marker='o');
_ = plt.ylabel('Accuracy +/- Standard Deviation', size = 15)
_ = plt.xlabel('Feature', size = 15)
feature_min = len(metric_dict[k_feat[0]]['feature_idx'])
feature_max = len(metric_dict[k_feat[-1]]['feature_idx'])
_ = plt.xticks(k_feat,
           [str(metric_dict[k]['feature_names']) for k in k_feat],
           rotation=90, size = 15)
_ = plt.yticks(size = 15)
plt.show();

df_efs

#eda_all = pd.concat([df_train, df_test], sort=True).reset_index(drop=True)

#sns.pairplot(df_train, vars=names_all, hue='survived', plot_kws = {'alpha': 0.6, 's': 40,
# 'edgecolor': 'k'}, palette=sns.color_palette("hls", 2), size=8)



# We already knew that gender was going to be the winner here. Again it is useful to reflect
# on just how strongly this variable is correlated with the outcome. Gender by itself
# obtaining an accuracy of 79% is noteworthy.
# 
# Our newly generated title variable does not seem to be adding much to the party here, it
# comes in second but is likely to contain the same information as gender. We can probably
# drop this variable, or alternatively attempt to improve gender with information contained in
#  the title variable.
# 
# All the other variables show strong correlation with the outcome variable. We will have
#  to carefully consider variable interactions to find combinations of some variables
#  that avoid multi-collinearity and hence maximises the signal in the data.
# 
# Next we will consider correlation between the continuous variables and the response. We have
# two continuous variables, <i>age</i> and <i>fare</i>.

# In[63]:


X_train = df_train_trans.loc[:, names_con]
y_train = df_train_trans.loc[:, "survived"]

# Replace missing values for continuous variables - else we cannot compute correlation
# statistics.

# Replace fare in test set with median from train set - to prevent data leakage.
median_fare = X_train['fare'].median()
X_train['fare'].fillna(median_fare, inplace=True)

# Replace missing values for training set
#print("Number of null values in age column: {}".format(X_train['age'].isnull().sum()))

# Define imputer
imputer = KNNImputer()
# fit on the dataset
_ = imputer.fit(X_train)
# transform the dataset
X_train_array = imputer.transform(X_train)
X_train = pd.DataFrame(X_train_array, columns=names_con)
# summarize total missing
#print('Missing: %d' % sum(isnan(X_train).flatten()))

# Feature extraction set to retain all - we want to see scores for all variables.
test = SelectKBest(score_func=f_classif, k='all')
fit_kbest = test.fit(X_train, y_train)
features_kbest = np.array(names_con)
plot_feature_importance(fit = fit_kbest, features = features_kbest)


# It is clear from the output of the Chi-Squared test that fare has a higher correlation with
# survival rate than age.
# 
# In order to see how feature importances amongst all the variables stack up (categorical and
# continuous) we will now analyse correlation and feature importance by considering the output
#  of a Random Forest model which deals with categorical and continuous variables in the same
#  manner.
# 

# In[64]:


X_train = df_train_trans.loc[:, names_all]
y_train = df_train_trans.loc[:, "survived"]

# Replace fare in test set with median from train set - to prevent data leakage.
median_fare = X_train['fare'].median()
X_train['fare'].fillna(median_fare, inplace=True)

# Replace missing values for training set
#print("Number of null values in age column: {}".format(X_train['age'].isnull().sum()))

# Define imputer
imputer = KNNImputer()
# fit on the dataset
_ = imputer.fit(X_train)
# transform the dataset
X_train_array = imputer.transform(X_train)
X_train = pd.DataFrame(X_train_array, columns=names_all)
# summarize total missing
#print('Missing: %d' % sum(isnan(X_train).flatten()))

rand_forest = RandomForestClassifier(max_features=0.25, n_estimators=1000, criterion= 'gini',
                                     random_state=0)
rand_forest.fit(X_train, y_train)


# In[65]:


importances = np.array(rand_forest.feature_importances_)
feature_list = np.array(X_train.columns)
plot_feature_importance_dec(fit = importances, features = feature_list)


# This is really interesting. Even though Chi-square ranks fare higher than age in terms of
# importance, the Random Forest ranks age as the most important feature in this analysis. From
#  this we can deduce that age has significant interactions with other variables that result
#  survival being classified more accurately.
# 
# We can use this information to engineer more meaningful feature variables.
# 
# Next we will analyse the results of the actual Decision Tree.

# In[66]:


estimator = rand_forest.estimators_[1]

# Show the first few levels of the tree
_ = export_graphviz(estimator, out_file='tree.dot',
                feature_names = X_train.columns,
                max_depth = 6,
                class_names = ['Died', 'Survived'],
                rounded = True, proportion = True,
                label='root',
                precision = 2, filled = True);


# In[67]:


_ = call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
#_ = call(['dot', '-Tsvg', 'tree.dot', '-o tree.svg', 'tree.svg', '-Gdpi=600'])
Image(filename = 'tree.png')
#plt.show()


# From the Decision Tree output we can observe many interesting variable interactions
# (combinations of branches that result in clear distinctions of survival or death) such as the
# following:
# 
# - gender and age (first 3 levels of splits are mainly on gender and age)
# - gender and deck
# - fare and p_class
# - fare and age
# - fare and embarked
# 
# Further investigation is warranted as there are many meaningful interactions. Let us look at a
# few more graphs illustrating these interactions.

# In[68]:


chivalry = sns.FacetGrid(df_train, col = 'embarked')
_ = chivalry.map(sns.pointplot, 'p_class', 'survived', 'sex', ci=95.0, palette = 'deep',
                 order = [1,2,3], hue_order = ['male','female'])
_ = chivalry.add_legend()


# <div class="alert alert-block alert-info">
# <b>Transform variables</b>
# </div>

# It is clear from all our analyses that there are many categorical variables strongly
# correlated with survival. It is also clear that there are many strong variable interactions
# in the data.
# 
# It therefore makes sense to experiment with binning of the two continuous variables i.e. age
#  and fare and to manually perform some interactions modelling to see if we can obtain more
#  consistent results with our predictive models.
# 
# We have already looked at the age variable in detail before, let's have another look at the
# fare variable:

# In[69]:


fig_fare, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), squeeze=False)

_ = sns.kdeplot(data=df_train_con_plot.loc[(df_train_con_plot['survived'] == 0), 'fare'],
                shade = True, label = 'Died')
_ = sns.kdeplot(data=df_train_con_plot.loc[(df_train_con_plot['survived'] == 1), 'fare'],
                shade = True, label = 'Survived')
_ = plt.xlabel('Fare', fontsize=20)
_ = plt.ylabel('Density', fontsize=20)
_ = plt.xticks(fontsize=20)
_ = plt.yticks(fontsize=20)
_ = plt.legend(fontsize=15)
plt.show()


# We can bin this variable quite easily.
# 
# As age and fare have missing values, we need to impute this at time of model building to
# avoid data leakage. We will therefore go ahead with the model building process and bin these
#  variables after missing values have been replaced.
# 
# We will now build the following models as previously discussed:
# 
#  1. Logistic regression
#  2. Multi-layer Perceptron (MLP)
#  3. Decision Tree
#  4. Random Forest
# 
# Our strategy is to build our own <i>validation strategy</i> based on the training set for which
# we have labels. We will do this by splitting this set into training and testing sets according to
# a 75%/ 25% split. Any hyper-parameter optimisation will be done by using <i>cross validation</i>
# on the 75% test set. The 25% testing set will be used for our final test before we apply the
# results to the provided test set for submission.
# 
# We therefore now start by splitting the response and features for the training set as previously
# discussed.
# 
# We will be using this dataset for all our models from here onwards. We also perform minor
# transformations such as encoding the <i>sex</i> variable for test and training sets. We also One
# Hot Encode the <i>embarked</i> variable. We drop one of the categories for embarked to avoid
# multi-collinearity (dummy variable trap).
# 
# 

# In[70]:


# Group response values to form binary response
y = df_train_trans.loc[:, 'survived']

# Split data into features (X) and response (y)
X = df_train_trans.loc[:, names_all]

# Consider using another dataframe for applying testing
df_test_trans = df_test_trans.loc[:, names_all]

# Put the response y into an array
y = np.ravel(y)


# #### Split, impute missing values and transform the data

# We now split the data into training and test sets according to a 75/ 25% split. We next
# impute missing values without data leakage on the training and test sets.
# 
# We then transform the final continuous variables into categorical variables. The fact that
# we have transformed all the variables to categorical means that we don't have to scale the
# data.

# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#print('Percentage holdout data: {}%'.format(round(100*(len(X_test)/len(X)),0)))
names_train = X.columns

# Replace fare in test set with median from train set - to prevent data leakage.
median_fare = X_train['fare'].median()
df_test_trans['fare'].fillna(median_fare, inplace=True)

# Replace missing values for training set
print("Number of null values in age column: {}".format(X_train['age'].isnull().sum()))

# Define imputer
imputer = KNNImputer()
# fit on the dataset
imputer.fit(X_train)
# transform the dataset
X_train_array = imputer.transform(X_train)
# summarize total missing
#print('Missing: %d' % sum(isnan(X_train).flatten()))
X_train = pd.DataFrame(X_train_array, columns=names_train)

X_test_array = imputer.transform(X_test)
# summarize total missing
#print('Missing: %d' % sum(isnan(X_test).flatten()))
X_test = pd.DataFrame(X_test_array, columns=names_train)


# Now we fit and transform for the final model.
# Fit and apply to the final dataset: TODO: Test whether rebuilding model on complete set
#  performs better
#imputer.fit(X)

X_array = imputer.transform(X)
X = pd.DataFrame(X_array, columns=names_train)

# summarize total missing
#print('Missing: %d' % sum(isnan(X).flatten()))
df_test_trans_array = imputer.transform(df_test_trans)
# summarize total missing
#print('Missing: %d' % sum(isnan(df_test_trans).flatten()))
df_test_trans = pd.DataFrame(df_test_trans_array, columns=names_train)


# In[72]:


# Binning fare: TODO: fix problem with ranges not understood and undercounting
fare_bins= [0, 8, 15, 30, 100, 300, 520]
labels = ['very_low', 'low', 'average', 'above_average', 'high', 'very_high']
df_train['fare_bin'] = pd.cut(df_train['fare'], bins=fare_bins, labels=labels, right=False)
df_train_trans['fare_bin'] = pd.cut(df_train_trans['fare'], bins=fare_bins, labels=labels,
                                    right=False)
X_train['dim_fare'] = X_train['fare']
X_train['dim_age'] = X_train['age']
X_train['dim_ticket_freq'] = X_train['ticket_freq']
X_test['dim_fare'] = X_test['fare']
X_test['dim_age'] = X_test['age']
X_test['dim_ticket_freq'] = X_test['ticket_freq']
df_test_trans['dim_fare'] = df_test_trans['fare']
df_test_trans['dim_age'] = df_test_trans['age']
df_test_trans['dim_ticket_freq'] = df_test_trans['ticket_freq']

X_train['fare_bin'] = pd.cut(X_train['fare'], bins=fare_bins, labels=labels, right=False)
X_test['fare_bin'] = pd.cut(X_test['fare'], bins=fare_bins, labels=labels,
                                    right=False)
df_train_trans['fare_bin'] = pd.cut(df_train_trans['fare'], bins=fare_bins, labels=labels,
                                   right=False)
df_test_trans['fare_bin'] = pd.cut(df_test_trans['fare'], bins=fare_bins, labels=labels,
                                   right=False)

#Binning age: TODO: fix problem with ranges not understood and undercounting
bins= [0, 4, 13, 20, 40, 60, 110]
labels = ['infant','child','teen','adult', 'middle_aged', 'elderly']
#df_train['age_bin'] = pd.cut(df_train['age'], bins=bins, labels=labels, right=False)
#df_train_trans['age_bin'] = pd.cut(df_train_trans['age'], bins=bins, labels=labels, right=False)
X_train['age_bin'] = pd.cut(X_train['age'], bins=bins, labels=labels, right=False)
X_test['age_bin'] = pd.cut(X_test['age'], bins=bins, labels=labels, right=False)
df_train_trans['age_bin'] = pd.cut(df_train_trans['age'], bins=bins, labels=labels,
                                   right=False)
df_test_trans['age_bin'] = pd.cut(df_test_trans['age'], bins=bins, labels=labels, right=False)

#
X_train['dim_age_bin'] = X_train['age_bin']
X_train['dim_fare_bin'] = X_train['fare_bin']
X_test['dim_age_bin'] = X_test['age_bin']
X_test['dim_fare_bin'] = X_test['fare_bin']

# Transform embarked and deck variables for training set - try not binning fare
#binning_cols = ['fare_bin', 'age_bin', 'ticket_freq']
binning_cols = ['age_bin', 'fare_bin']
X_train = pd.get_dummies(X_train, columns = binning_cols, drop_first=True)

# Transform embarked and deck variables for testing set
#binning_cols = ['fare_bin', 'age_bin', 'ticket_freq']
binning_cols = ['age_bin', 'fare_bin']
X_test = pd.get_dummies(X_test, columns = binning_cols, drop_first=True)

# Transform embarked and deck variables for test set
df_train_trans['dim_age_bin'] = df_train_trans['age_bin']
df_train_trans['dim_fare_bin'] = df_train_trans['fare_bin']

df_test_trans['dim_age_bin'] = df_test_trans['age_bin']
df_test_trans['dim_fare_bin'] = df_test_trans['fare_bin']

df_train_trans = pd.get_dummies(df_train_trans, columns = binning_cols, drop_first=True)
df_test_trans = pd.get_dummies(df_test_trans, columns = binning_cols, drop_first=True)

# Hack to fix ticket_freq distribution: TODO: Fix this.
#df_test_trans['ticket_freq_6.0'] = 0
#df_test_trans['ticket_freq_7.0'] = 0

#X_train.drop('age', axis=1, inplace=True)
#X_test.drop('age', axis=1, inplace=True)
#df_test_trans.drop('age', axis=1, inplace=True)
#df_train_trans.drop('age', axis=1, inplace=True)
#X_train.drop('fare', axis=1, inplace=True)
#X_test.drop('fare', axis=1, inplace=True)
#df_test_trans.drop('fare', axis=1, inplace=True)
#df_train_trans.drop('fare', axis=1, inplace=True)


# Create interaction terms.

# In[73]:


X_train = pd.merge(X_train, df_train_trans[['dim_deck']],left_index=True, right_index=True)

# create dummy variables, and their interactions: TODO: Check if deck is better cross-term
X_train_interactions = \
    dmatrix('C(dim_deck) * C(sex)', X_train,
              return_type="dataframe")
X_train_interactions.drop('Intercept', inplace=True, axis=1)
X_train = pd.concat([X_train, X_train_interactions], axis=1)

X_train.drop('sex', axis=1, inplace=True)
X_train.drop('dim_fare', axis=1, inplace=True)
X_train.drop('dim_age', axis=1, inplace=True)
X_train.drop('dim_fare_bin', axis=1, inplace=True)
X_train.drop('dim_age_bin', axis=1, inplace=True)
X_train.drop('dim_deck', axis=1, inplace=True)
X_train.drop('dim_ticket_freq', axis=1, inplace=True)

X_test = pd.merge(X_test, df_train_trans[['dim_deck']],left_index=True, right_index=True)
X_test_interactions = \
    dmatrix('C(dim_deck) * C(sex)', X_test,
              return_type="dataframe")
X_test_interactions.drop('Intercept', inplace=True, axis=1)
X_test = pd.concat([X_test, X_test_interactions], axis=1)

X_test.drop('sex', axis=1, inplace=True)
X_test.drop('dim_fare', axis=1, inplace=True)
X_test.drop('dim_age', axis=1, inplace=True)
X_test.drop('dim_fare_bin', axis=1, inplace=True)
X_test.drop('dim_age_bin', axis=1, inplace=True)
X_test.drop('dim_deck', axis=1, inplace=True)
X_test.drop('dim_ticket_freq', axis=1, inplace=True)

df_test_trans = pd.merge(df_test_trans, df_train_trans[['dim_deck']],left_index=True,
                         right_index=True)
df_test_trans_interactions = \
    dmatrix('C(dim_deck) * C(sex)', df_test_trans,
              return_type="dataframe")
df_test_trans_interactions.drop('Intercept', inplace=True, axis=1)
df_test_trans = pd.concat([df_test_trans, df_test_trans_interactions], axis=1)

df_test_trans.drop('sex', axis=1, inplace=True)
df_test_trans.drop('dim_fare', axis=1, inplace=True)
df_test_trans.drop('dim_age', axis=1, inplace=True)
df_test_trans.drop('dim_fare_bin', axis=1, inplace=True)
df_test_trans.drop('dim_age_bin', axis=1, inplace=True)
df_test_trans.drop('dim_deck', axis=1, inplace=True)
df_test_trans.drop('dim_ticket_freq', axis=1, inplace=True)


# In[74]:


from sklearn.feature_selection import chi2

# Separate continuous variables for this step, to be added back afterwards.
X_train_con = X_train.loc[:, ['age', 'fare', 'ticket_freq']]
X_train.drop(['age', 'fare', 'ticket_freq'], axis=1, inplace=True)
# Separate continuous variables for this step, to be added back afterwards.
X_test_con = X_test.loc[:, ['age', 'fare', 'ticket_freq']]
X_test.drop(['age', 'fare', 'ticket_freq'], axis=1, inplace=True)
# Separate continuous variables for this step, to be added back afterwards.
df_test_trans_con = df_test_trans.loc[:, ['age', 'fare', 'ticket_freq']]
df_test_trans.drop(['age', 'fare', 'ticket_freq'], axis=1, inplace=True)

# Finally we scale our data - separately from categorical variables.
#scaler = StandardScaler()
#
## Fit on training data set
#names_training = list(X_train_con.columns.values)
#_ = scaler.fit(X_train_con)
#X_train_new = scaler.transform(X_train_con)
#X_train_con = pd.DataFrame(X_train_new, columns=names_training)

# Apply to test data (training)
#X_test_new = scaler.transform(X_test_con)
#X_test_con = pd.DataFrame(X_test_new, columns=names_training)

# Scale age and fare on final dataset to final test data
#df_test_trans_new = scaler.transform(df_test_trans_con)
#df_test_trans_con = pd.DataFrame(df_test_trans_new, columns=names_training)

# Perform categorical feature selection
X_train = X_train.astype(float)
y_train = y_train.astype(float)
X_test = X_test.astype(float)
y_test = y_test.astype(float)
df_test_trans =  df_test_trans.astype(float)

best_feat = SelectKBest(chi2, k="all").fit(X_train, y_train)
mask = X_train.columns.values[best_feat.get_support()]
X_train_new = best_feat.transform(X_train)
X_train = pd.DataFrame(X_train_new, columns=mask)

X_test_new = best_feat.transform(X_test)
X_test = pd.DataFrame(X_test_new, columns=mask)

df_test_trans_new = best_feat.transform(df_test_trans)
df_test_trans = pd.DataFrame(df_test_trans_new, columns=mask)

# What are scores for the features
for i in range(len(mask)):
	print('%s: \t\t\t\t%f' % (mask[i], best_feat.scores_[i]))

# plot the scores
_ = pyplot.bar([i for i in range(len(best_feat.scores_))], best_feat.scores_)
pyplot.show()

# Add continuous variables back again.
X_train = pd.merge(X_train, X_train_con['ticket_freq'], left_index=True, right_index=True)
X_test = pd.merge(X_test, X_test_con['ticket_freq'], left_index=True, right_index=True)
df_test_trans = pd.merge(df_test_trans, df_test_trans_con['ticket_freq'], left_index=True,
                         right_index=True)

# Drop interactions for testing purposes
interactions_list = list(X_train_interactions.columns.values)
X_train.drop(interactions_list, axis=1, inplace=True)
X_test.drop(interactions_list, axis=1, inplace=True)
df_test_trans.drop(interactions_list, axis=1, inplace=True)

# Drop is_married for testing purposes
X_train.drop(['is_married'], axis=1, inplace=True)
X_test.drop(['is_married'], axis=1, inplace=True)
df_test_trans.drop(['is_married'], axis=1, inplace=True)


# In[75]:


#%%
# Final conversion to float.
convert_dict = {'embarked_Q': float,
                'embarked_S': float,
                'deck_DE': float,
                'deck_FG': float,
                'deck_M': float,
                'title_Miss': float,
                'title_Mr': float,
                'title_Mrs': float,
                'p_class_2': float,
                'p_class_3': float,
                'fam_size_small': float,
                'fam_size_large': float,
                'fam_size_very_large': float,
                'age_bin_child': int,
                'age_bin_teen': int,
                'age_bin_adult': int,
                'age_bin_middle_aged': int,
                'age_bin_elderly': int,
                'fare_bin_low': int,
                'fare_bin_average': int,
                'fare_bin_above_average': int,
                'fare_bin_high': int,
                'fare_bin_very_high': int,
                'ticket_freq': float}

X_train = X_train.astype(convert_dict)
X_test = X_test.astype(convert_dict)
df_test_trans =  df_test_trans.astype(convert_dict)

# Rearrange columns
col_names = ['embarked_Q', 'embarked_S', 'deck_DE', 'deck_FG', 'deck_M', 'title_Miss',
           'title_Mr', 'title_Mrs', 'p_class_2', 'p_class_3', 'fam_size_small',
             'fam_size_large', 'fam_size_very_large', 'ticket_freq', 'fare_bin_low',
             'fare_bin_average', 'fare_bin_above_average', 'fare_bin_high',
             'fare_bin_very_high', 'age_bin_child', 'age_bin_teen', 'age_bin_adult',
             'age_bin_middle_aged', 'age_bin_elderly']

X_train = X_train[col_names]
X_test = X_test[col_names]
df_test_trans = df_test_trans[col_names]


# Our final dataset for model building looks as follows:

# In[76]:


X_train.head()


# <div class="alert alert-block alert-info">
# <b>Build models</b>
# </div>

# We can now start building our first model, yay! We build and test a naive logistic regression 
# model - without any transformations or optimisations.
# 
# The objective is to ascertain the strength of association between features and responses on 
# unprocessed data. 

# #### Naive Logistic Regression

# In[77]:


# Initial model
log_reg = LogisticRegression(max_iter=2000000, fit_intercept = False)

# Probability scores for test set
y_score_init = log_reg.fit(X_train, y_train).decision_function(X_test)

# False positive Rate and true positive rate
fpr_roc, tpr_roc, thresholds = roc_curve(y_test, y_score_init)

plot_roc_curve(fpr = fpr_roc, tpr = tpr_roc)


# In[78]:


y_pred = log_reg.predict(X_test)
# Accuracy before model parameter optimisation
cnf_matrix = confusion_matrix(y_pred, y_test)
plot_confusion_matrix(cnf_matrix, classes=[0,1], normalize=True)


# As can be seen from the accuracy measurements the baseline model performs marginally better
# than our baseline models. A C-statistic (Area Under the Curve - AUC) of 88% compared to 86%
# for the baseline model.
# 
# The model has a precision of 83% which is the same as the previously obtained 83%.
# 
# The sensitivity of 77% is up from 68% which means more survivors are being picked up with
# this model. Specificity us down to 83% from 86% which means the increase in accuracy is at
# the expense of false negatives.

# #### Naive MLP

# In[79]:


# Fit and check MSE before regularisation
mlp_reg = MLPClassifier(max_iter=50000, solver="adam", activation="tanh", hidden_layer_sizes=(5, 5),
                    random_state=1)
mlp_reg = mlp_reg.fit(X_train, y_train)

# Predict
y_pred = mlp_reg.predict(X_test)

# Accuracy before model parameter optimisation
print ("Accuracy Score: %0.5f" % accuracy_score(y_pred,y_test))

# False positive Rate and true positive rate
fpr_roc, tpr_roc, thresholds = roc_curve(y_test, y_pred)

plot_roc_curve(fpr = fpr_roc, tpr = tpr_roc)

# Accuracy before model parameter optimisation
cnf_matrix = confusion_matrix(y_pred, y_test)
plot_confusion_matrix(cnf_matrix, classes=[0,1], normalize=True)


# The naive MLP has an AUC score of 85% which is higher than the previous score of
# Regression of 79%.

# In[80]:


# Optimise numbers of nodes on both layers
validation_scores = {}
print("Nodes |Validation")
print("      | score")

for hidden_layer_size in [(i,j) for i in range(2,6) for j in range(2,6)]:

    reg = MLPClassifier(max_iter=1000000, hidden_layer_sizes=hidden_layer_size, random_state=1)

    score = cross_val_score(estimator=reg, X=X_train, y=y_train, cv=2)
    validation_scores[hidden_layer_size] = score.mean()
    print(hidden_layer_size, ": %0.5f" % validation_scores[hidden_layer_size])


# In[81]:


# Check scores
print("The highest validation score is: %0.4f" % max(validation_scores.values()))
optimal_hidden_layer_size = [name for name, score in validation_scores.items()
                              if score==max(validation_scores.values())][0]
print("This corresponds to nodes", optimal_hidden_layer_size )


# Now we optimise the neural network regularisation parameter.

# In[82]:


# Select range over which to find regularisation parameter - exponential used for even
# distribution of values
reg_par = [np.e**n for n in np.arange(-2,4,0.5)]

validation_scores = {}
print(" alpha  |  Accuracy")
for param in reg_par:
    reg = MLPClassifier(max_iter=1000000, solver="adam", activation="tanh",
                        hidden_layer_sizes=optimal_hidden_layer_size, alpha=param, random_state=1)
    score = cross_val_score(estimator=reg, X=X_train, y=y_train, cv=2, scoring="accuracy")
    validation_scores[param] = score.mean()
    print("%0.5f |  %s" % (param, score.mean()))

# Plot the accuracy function against regularisation parameter
plt.plot([np.log(i) for i in validation_scores.keys()], list(validation_scores.values()));
plt.xlabel("Ln of alpha");
plt.ylabel("Accuracy");


# The highest cross-validation accuracy score and hence the value to use for the `alpha` parameter
# is as follows.

# In[83]:


max_score = ([np.log(name) for name, score in validation_scores.items() if score==max
(validation_scores.values())][0])
# Find lowest value.
print("The highest accuracy score is: %s" % (max(validation_scores.values())))
print("This corresponds to regularisation parameter e**%s" % max_score)


# #### MSE after regularisation

# In[84]:


# Fit data with the best parameter
mlp_reg_optim = MLPClassifier(max_iter=1000000, solver="adam", activation="tanh",
                    hidden_layer_sizes=optimal_hidden_layer_size, alpha=np.e**(max_score),
                              random_state=1)

mlp_reg_optim.fit(X_train, y_train)

# Predict
y_pred = mlp_reg_optim.predict(X_test)

# Accuracy after model parameter optimisation
accuracy_score(y_pred,y_test)


# #### Accuracy analysis

# In[85]:


# False positive Rate and true positive rate
fpr_roc, tpr_roc, thresholds = roc_curve(y_test, y_pred)
plot_roc_curve(fpr = fpr_roc, tpr = tpr_roc)

cnf_matrix = confusion_matrix(y_pred, y_test)
plot_confusion_matrix(cnf_matrix, classes=[0,1], normalize=True)


# The optimised MLP has an AUC score of 78% which is higher than the previous score
# of 77%. This is encouraging as this was also the model that fared best on the public data.

# #### Naive Decision Tree
# 
# We now build a naive Decision Tree to see it performs against the previous models. We also use
# the Decision Tree to calculate feature importance. This will provide us with a better feeling for
#  strength of association between features and the response.

# In[86]:


# Fit a Decision Tree to data
samples = [sample for sample in range(1,30)]
validation_scores = []
for sample in samples:
    classifier1 = DecisionTreeClassifier(random_state=1, min_samples_leaf=sample)
    score = cross_val_score(estimator=classifier1, X=X_train, y=y_train, cv=5)
    validation_scores.append(score.mean())

# Obtain the minimum leaf samples with the highest validation score
samples_optimum = samples[validation_scores.index(max(validation_scores))]

classifier2 = DecisionTreeClassifier(random_state=0, min_samples_leaf=samples_optimum)
classifier2.fit(X_train, y_train)


# Feature importances for the Decision Tree is as follows:

# In[87]:


cols_model = X_train.columns.to_list()

importances = np.array(classifier2.feature_importances_)
feature_list = np.array(cols_model)

# summarize feature importance
for i,v in enumerate(importances):
	print('Feature: %10s\tScore:\t%.5f' % (feature_list[i],v))
# plot feature importance
sorted_ID=np.array(np.argsort(importances)[::-1])
plt.figure(figsize=[10,10])
plt.xticks(rotation='vertical')
_ = plt.bar(feature_list[sorted_ID], importances[sorted_ID]);
plt.show();


# Here we have an interesting turn of events. The previously most important features was sex,
# but has now been replaced by title_Mr. Passenger class and family size are in 2nd and 3rd
# places.

# #### Accuracy analysis

# In[88]:


# Probability scores for test set
y_pred = classifier2.predict(X_test)

accuracy_score(y_pred,y_test)

# False positive Rate and true positive rate
fpr_roc, tpr_roc, thresholds = roc_curve(y_test, y_pred)
plot_roc_curve(fpr = fpr_roc, tpr = tpr_roc)

cnf_matrix = confusion_matrix(y_pred, y_test)
plot_confusion_matrix(cnf_matrix, classes=[0,1], normalize=True)


# The Decision Tree obtained an AUC score of 75% which is lower than the previous score of
# 77%.
# 

# #### Naive Random Forest
# Now we build a Random Forest and see what happens!

# In[89]:


rand_forest = RandomForestClassifier(criterion= 'gini', random_state=0)
rand_forest.fit(X_train, y_train)

# Probability scores for test set
y_pred = rand_forest.predict(X_test)

accuracy_score(y_pred,y_test)


# #### Accuracy analysis

# In[90]:


# False positive Rate and true positive rate
fpr_roc, tpr_roc, thresholds = roc_curve(y_test, y_pred)
plot_roc_curve(fpr = fpr_roc, tpr = tpr_roc)

cnf_matrix = confusion_matrix(y_pred, y_test)
plot_confusion_matrix(cnf_matrix, classes=[0,1], normalize=True)


# The Random Forest obtained an AUC score of 83% which is higher than the previous score
# obtained.

# #### Optimised Random Forest

# In[91]:


rand_forest = RandomForestClassifier(max_features='auto')
param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10],
               "min_samples_split" : [2, 4, 10, 12], "n_estimators": [50, 100, 400, 700]}
gs = GridSearchCV(estimator=rand_forest, param_grid=param_grid, scoring='accuracy', cv=3,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)


# In[92]:


gs.best_params_


# In[93]:


# Final prediction - MLP
#rand_forest = RandomForestClassifier(criterion= 'gini', min_samples_leaf=5,
#                                     min_samples_split=2, n_estimators=100, random_state=0)
rand_forest = RandomForestClassifier(criterion= 'entropy', min_samples_leaf=5,
                                     min_samples_split=4, n_estimators=100, random_state=0)
rand_forest.fit(X_train, y_train)
#rand_forest.fit(x_train_prev, y_train)

# Probability scores for test set
y_pred = rand_forest.predict(X_test)
#y_pred = rand_forest.predict(x_test_prev)
accuracy_score(y_pred,y_test)


# In[94]:


importances = np.array(rand_forest.feature_importances_)
feature_list = np.array(X_train.columns)
importances = np.array(importances)
sorted_ID=np.array(np.argsort(importances))
reverse_features = feature_list[sorted_ID][::-1]
reverse_importances = importances[sorted_ID][::-1]

for i,v in enumerate(reverse_importances):
    print('Feature: %20s\tScore:\t%.5f' % (reverse_features[i],v))

# Plot feature importance
#sorted_ID=np.array(np.argsort(scores)[::-1])
#sns.set(font_scale=1);
_ = plt.figure(figsize=[10,10]);
_ = plt.xticks(rotation='horizontal', fontsize=20)
_ = plt.barh(feature_list[sorted_ID], importances[sorted_ID], align='center');
_ = plt.yticks(fontsize=20)
_ = plt.show();


# <div class="alert alert-block alert-info">
# <b>Conclusion</b>
# </div>

# Previously we chose the optimised (regularised) MLP for submission based on overall
# <i>public accuracy</i> score of 0.7799. This public score was in the top 24% and corresponds
# more or less with our validation scores, so looks sensible! Not bad for a model with
# absolutely no fine-tuning.
# 
# For this notebook we performed data cleaning and EDA. We spent a lot of time imputing
# missing values and calculating correlation. We also spent some time doing some basic feature
#  engineering. For this notebook the Random Forest model overtook the MLP in terms of
#  performance, and scored a respectable public accuracy score of 0.79665! At this point we
#  have used only hand-coded scikit-learn models, with no automated hyperparameter
#  optimisation or fancy boosting models. Our approach is also different to other approaches
#  on Kaggle in that we opted for the use of dummy variables only, given the fact that most
#  variables in this problem are categorical. Seeing as we encoded all variables there was no
#  need for scaling of values.
# 
# Not bad, but not a huge improvement taking into consideration all the time spent on data
# cleaning and EDA...
# 
# We have however found some useful information from our Decision Trees regarding variable
# interactions, and are confident that we can build some interesting cross-terms from these in
#  our next notebook!
# 
# On to the next notebook on feature engineering and use of ML pipelines!
# 
# Hopefully we can crack the 80% level target by building some awesome cross-terms.

# In[95]:


# Gender submission - Score: 0.77

# Final prediction - Random Forest - Score:  0.79665
y_pred = rand_forest.predict(df_test_trans)
y_pred = y_pred.astype(int)

#Prepare submission code
my_submission = pd.DataFrame({'PassengerId': df_orig.passenger_id, 'Survived': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_rand_forest.csv', index=False)

# Final prediction - MLP (optimised) - Score: ?
#mlp_reg_optim.fit(X, y)
y_pred = mlp_reg_optim.predict(df_test_trans)
y_pred = y_pred.astype(int)

#Prepare submission code
my_submission = pd.DataFrame({'PassengerId': df_orig.passenger_id, 'Survived': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_mlp_optim.csv', index=False)

# TODO: Final prediction - MLP (non-optimised) - Score: ?
y_pred = mlp_reg.predict(df_test_trans)
y_pred = y_pred.astype(int)

#Prepare submission code
my_submission = pd.DataFrame({'PassengerId': df_orig.passenger_id, 'Survived': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_mlp.csv', index=False)

# Final prediction - Logistic Regression - Score: ?
y_pred = log_reg.predict(df_test_trans)
y_pred = y_pred.astype(int)

#Prepare submission code
my_submission = pd.DataFrame({'PassengerId': df_orig.passenger_id, 'Survived': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_logreg.csv', index=False)

# Final prediction - Decision Tree - Score: ?
y_pred = classifier2.predict(df_test_trans)
y_pred = y_pred.astype(int)

#Prepare submission code
my_submission = pd.DataFrame({'PassengerId': df_orig.passenger_id, 'Survived': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_dec_tree.csv', index=False)

