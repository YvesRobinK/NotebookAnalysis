#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Model

# **Dataset Story** 
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# Titanic dataset contains information about the people involved in the Titanic shipwreck.  
# 
# **Goal**
# 
# Predict if a passenger survived the sinking of the Titanic or not. 
# 
# **Variables Description**
# * PassengerID : ID of the Passenger.
# * Survived: Survival (0 = No; 1 = Yes)
# * Pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# * Name : Name of the Passenger
# * Sex: Sex of the Passenger (Female / Male)
# * Age: Age of the Passenger.
# * Sibsp: Number of siblings/spouses aboard
# * Parch: Number of parents/children aboard
# * Ticket : Ticket number.
# * Fare: Passenger fare (British pound)
# * Cabin: Cabin number
# * Embarked: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
#  
#  
# **Steps**
# 
# * Exploratory Data Analysis
# * Data Preprocessing & Feature Engineering
# * Encoding ( Label Encoding / One Hot Encoding / Rare Encoding
# * Model Buildind & Performance Metrics
# * Model Validation
# * Summary
# 
# 
#  **References**
#  
#  http://rstudio-pubs-static.s3.amazonaws.com/278621_8ab6e10f7b6941dba0dc8968955e73fe.html
#  
#  https://towardsdatascience.com/importance-of-feature-engineering-methods-73e4c41ae5a3#:~:text=To%20improve%20the%20performance%20of,existing%20features%20into%20new%20features.
#  
#  

# **Import Libraries & Setting Configurations**

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

import math as mt
import missingno as msno
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

from sklearn.linear_model import LogisticRegression


import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# **Import Data**

# In[2]:


titanic_ = pd.read_csv('../input/titanic-dataset/titanic.csv')
titanic_df = titanic_.copy()
titanic_df.head()


# **Exploratory Data Analysis**

# In[3]:


def upper_col_name(dataframe):
    upper_cols = [col.upper() for col in dataframe.columns]
    dataframe.columns = upper_cols
    return dataframe.head()


# In[4]:


upper_col_name(titanic_df)


# In[5]:


titanic_df.info()


# In[6]:


titanic_df.describe().T


# In[7]:


#Selection of Categorical and Numerical Variables:

def grab_col_names(dataframe, cat_th=5, car_th=20):
    """
    This function to perform the selection of numeric and categorical variables in the data set in a parametric way.
    Note: Variables with numeric data type but with categorical properties are included in categorical variables.

    Parameters
    ----------
    dataframe: dataframe
        The data set in which Variable types need to be parsed
    cat_th: int, optional
        The threshold value for number of distinct observations in numerical variables with categorical properties.
        cat_th is used to specify that if number of distinct observations in numerical variable is less than
        cat_th, this variables can be categorized as a categorical variable.

    car_th: int, optional
        The threshold value for categorical variables with  a wide range of cardinality.
        If the number of distinct observations in a categorical variables is greater than car_th, this
        variable can be categorized as a categorical variable.

    Returns
    -------
        cat_cols: list
            List of categorical variables.
        num_cols: list
            List of numerical variables.
        cat_but_car: list
            List of categorical variables with  a wide range of cardinality.

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        Sum of elements in lists the cat_cols,num_cols  and  cat_but_car give the total number of variables in dataframe.
    """

    # cat cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and
                   dataframe[col].nunique() < cat_th]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and
                   dataframe[col].nunique() > car_th]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and "ID" not in col.upper()]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols,num_cols,cat_but_car


# In[8]:


grab_col_names(titanic_df)


# In[9]:


cat_cols, num_cols, cat_but_car = grab_col_names(titanic_df)


# In[10]:


# General Exploration for Categorical Variables:

def cat_summary(dataframe, plot=False):
   for col_name in cat_cols:
       print("############## Unique Observations of Categorical Data ###############")
       print("The unique number of "+ col_name+": "+ str(dataframe[col_name].nunique()))

       print("############## Frequency of Categorical Data ########################")
       print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                           "Ratio": dataframe[col_name].value_counts()/len(dataframe)}))
       if plot == True:
           rgb_values = sns.color_palette("Set2", 6)
           sns.set_theme(style="darkgrid")
           ax = sns.countplot(x=dataframe[col_name], data=dataframe, palette=rgb_values)
           for p in ax.patches:
               ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=10)
           plt.show()


# In[11]:


cat_summary(titanic_df, plot=True)


# In[12]:


# General Exploration for Numerical Variables:

def num_summary(dataframe,  plot=False):
    quantiles = [0.25, 0.50, 0.75, 1]
    for col_name in num_cols:
        print("########## Summary Statistics of " +  col_name + " ############")
        print(dataframe[col_name].describe(quantiles).T)

        if plot:
            sns.histplot(data=dataframe, x=col_name  )
            plt.xlabel(col_name)
            plt.title("The distribution of "+ col_name)
            plt.grid(True)
            plt.show(block=True)


# In[13]:


num_summary(titanic_df, plot=True)


# **DATA PREPROCESSING & FEATURE ENGINEERING**

# 
# * ***Feature Extraction & Interactions***
# 
# Feature Engineering is beneficial step is a data preparation process, that increase the performance of models. 
# 
# This step will be performed first in order to deal with the missing values and outliers together with the newly derived features. 

# In[14]:


# Only passengers have cabin numbers, so "Deck" feature can be extracted by using Cabin feature:  
titanic_df["NEW_DECK"] = titanic_df["CABIN"].notnull().astype('int')

# Name word count
titanic_df["NEW_NAME_WORD_COUNT"] = titanic_df["NAME"].apply(lambda x: len(str(x).split(" ")))

# Name that includes "Dr"
titanic_df["NEW_NAME_DR"] = titanic_df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr. ")]))

# Family size:
titanic_df["NEW_FAMILY_SIZE"] = titanic_df["SIBSP"] + titanic_df["PARCH"] + 1

# Fare per passenger:
titanic_df['NEW_FARE_PER_PERSON'] = titanic_df['FARE'] / (titanic_df['NEW_FAMILY_SIZE'])

# Title:
titanic_df['NEW_TITLE'] = titanic_df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)

# Ticket:
titanic_df['NEW_TICKET'] = titanic_df['TICKET'].str.isalnum().astype('int')

# Age & Pclass
titanic_df["NEW_AGE_PCLASS"] = titanic_df["AGE"] * titanic_df["PCLASS"]

# Is Alone?
titanic_df["NEW_IS_ALONE"] = np.where(titanic_df['SIBSP'] + titanic_df['PARCH'] > 0, "NO", "YES") 
    
# Age Level 
titanic_df.loc[(titanic_df['AGE'] < 18), 'NEW_AGE_CAT'] = 'Young'
titanic_df.loc[(titanic_df['AGE'] >= 18) & (titanic_df['AGE'] < 56), 'NEW_AGE_CAT'] = 'Mature'
titanic_df.loc[(titanic_df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'Senior'

 # Age & Sex
titanic_df.loc[(titanic_df['SEX'] == 'male') & (titanic_df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'Young_Male'
titanic_df.loc[(titanic_df['SEX'] == 'male') & ((titanic_df['AGE'] > 21) & (titanic_df['AGE']) < 50), 'NEW_SEX_CAT'] = 'Mature_Male'
titanic_df.loc[(titanic_df['SEX'] == 'male') & (titanic_df['AGE'] > 50), 'NEW_SEX_CAT'] = 'Senior_Male'
titanic_df.loc[(titanic_df['SEX'] == 'female') & (titanic_df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'Young_Female'
titanic_df.loc[(titanic_df['SEX'] == 'female') & ((titanic_df['AGE'] > 21) & (titanic_df['AGE']) < 50), 'NEW_SEX_CAT'] = 'Mature_Female'
titanic_df.loc[(titanic_df['SEX'] == 'female') & (titanic_df['AGE'] > 50), 'NEW_SEX_CAT'] = 'Senior_Female'


# In[15]:


titanic_df.columns


# In[16]:


titanic_df.drop(columns=["PASSENGERID","NAME","TICKET","CABIN"], axis=1, inplace=True)


# In[17]:


titanic_df.head(3)


# * ***Outlier Detection:***

# In[18]:


def outlier_thresholds(dataframe, col_name, q1 = 0.25, q3 = 0.75):
    Q1 = dataframe[col_name].quantile(q1)
    Q3 = dataframe[col_name].quantile(q3)
    IQR = Q3 - Q1
    low_limit = Q1 - 1.5 * IQR
    up_limit = Q3 + 1.5 * IQR
    
    return low_limit, up_limit


# In[19]:


cat_cols, num_cols, cat_but_car = grab_col_names(titanic_df)


# In[20]:


for col in num_cols:
    print(col,":",outlier_thresholds(titanic_df,col))


# ***Showing Outliers with Boxplot:***

# In[21]:


def check_outlier(dataframe, q1=0.25, q3=0.75):
    for col_name in num_cols:
        low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
        if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
            sns.boxplot(x=dataframe[col_name])
            plt.show()
        else:
            return False


# In[22]:


check_outlier(titanic_df)


# * The maximum age value in the dataset appears to be 80 whereas it is not an impossible situation. So this value may not be considered an outlier for the relevant dataset.
# 
# * If we examine the upper and lower limit values for other variables and consider them from this point of view, we can recheck outliers by replacing q1 value as 0.05 and q3  value as 0.95.
# 
# * In the following steps, we will examine whether the variables together form an outlier by using Local Outlier Factor (LOF).

# In[23]:


check_outlier(titanic_df, q1=0.05, q3=0.95)


# * ***Missing Values:***
# 
# If we know that the missing values are random, NaN values can be removed or filled. Bu if there is no randomness, that is, if there is nullity correlation between the variables, applying the fill/delete operations will break the structure of the data set. 

# In[24]:


msno.matrix(titanic_df, figsize=(10,10), fontsize=10, labels=8)
plt.show()


# In[25]:


msno.heatmap(titanic_df, figsize=(8,8), fontsize=12)
plt.show()


# As it can be seen above from two graphs, there is no "Nullity Correlation" in the variables other than the variables derived from each other (etc Age).

# In[26]:


# Check the features containing NaN values:

def missing_values_df(dataframe, na_col_name=False):
    na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    na_cols_number = dataframe[na_cols].isnull().sum()
    na_cols_ratio = dataframe[na_cols].isnull().sum() / dataframe.shape[0]
    missing_values_table = pd.DataFrame({"Missing_Values (#)": na_cols_number, \
                                         "Ratio (%)": na_cols_ratio * 100,
                                         "Type" : dataframe[na_cols].dtypes})
    print(missing_values_table)
    print("************* Number of Missing Values *************")
    print(dataframe.isnull().sum().sum())
    if na_col_name:
        print("************* Nullable variables *************")
        return na_cols


# In[27]:


missing_values_df(titanic_df)


# In[28]:


def missing_cat_cols_fill(dataframe):
    na_cols = [col for col in titanic_df.columns if titanic_df[col].isnull().sum() > 0 and titanic_df[col].dtype == "O"]
    for col in na_cols:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])
        return dataframe.head()


# In[29]:


missing_cat_cols_fill(titanic_df)


# In[30]:


missing_values_df(titanic_df)


# In[31]:


def observe_missing_values(dataframe, na_col, related_col, target, target_method="mean", na_col_method="median"):
    print(dataframe.groupby(related_col).agg({target : target_method, 
                                               na_col : na_col_method}))


# In[32]:


cat_cols = [col for col in cat_cols if col not in "SURVIVED"]
for col in cat_cols:
    observe_missing_values(titanic_df, "AGE",col,"SURVIVED")


# In[33]:


titanic_df.drop(columns="NEW_NAME_DR",axis=1, inplace=True)


# In[34]:


titanic_df["AGE"] = titanic_df["AGE"].fillna(titanic_df.groupby("NEW_TITLE")["AGE"].transform("median"))


# In[35]:


# We need to update features which have been derived with AGE:
    
# Age & Pclass
titanic_df["NEW_AGE_PCLASS"] = titanic_df["AGE"] * titanic_df["PCLASS"]

# Is Alone?
titanic_df["NEW_IS_ALONE"] = np.where(titanic_df['SIBSP'] + titanic_df['PARCH'] > 0, "NO", "YES") 
    
# Age Level 
titanic_df.loc[(titanic_df['AGE'] < 18), 'NEW_AGE_CAT'] = 'Young'
titanic_df.loc[(titanic_df['AGE'] >= 18) & (titanic_df['AGE'] < 56), 'NEW_AGE_CAT'] = 'Mature'
titanic_df.loc[(titanic_df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'Senior'

 # Age & Sex
titanic_df.loc[(titanic_df['SEX'] == 'male') & (titanic_df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'Young_Male'
titanic_df.loc[(titanic_df['SEX'] == 'male') & ((titanic_df['AGE'] > 21) & (titanic_df['AGE']) < 50), 'NEW_SEX_CAT'] = 'Mature_Male'
titanic_df.loc[(titanic_df['SEX'] == 'male') & (titanic_df['AGE'] > 50), 'NEW_SEX_CAT'] = 'Senior_Male'
titanic_df.loc[(titanic_df['SEX'] == 'female') & (titanic_df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'Young_Female'
titanic_df.loc[(titanic_df['SEX'] == 'female') & ((titanic_df['AGE'] > 21) & (titanic_df['AGE']) < 50), 'NEW_SEX_CAT'] = 'Mature_Female'
titanic_df.loc[(titanic_df['SEX'] == 'female') & (titanic_df['AGE'] > 50), 'NEW_SEX_CAT'] = 'Senior_Female'
    


# In[36]:


missing_values_df(titanic_df)


# In[37]:


# Let's take a head at the dataset again:

titanic_df.info()


# **LocalOutlierFactor**

# In[38]:


cat_cols, num_cols, cat_but_car = grab_col_names(titanic_df)
df = titanic_df[num_cols]


# In[39]:


df.head()


# In[40]:


clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:5]


# In[41]:


# Visualization: 

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()


# In[42]:


# Let's determine the threshold by using Elbow Method

th = np.sort(df_scores)[8]

df[df_scores < th]


# In[43]:


titanic_df.drop(df[df_scores < th].index, inplace=True)


# In[44]:


titanic_df.shape


# **ENCODING**

# * ***Label Encoding:***

# In[45]:


# Defining binary cols:

def binary_cols(dataframe):
    binary_col_names = [col for col in dataframe.columns if ((dataframe[col].dtype == "O") and (dataframe[col].nunique() == 2))]
    return binary_col_names


# In[46]:


binary_col_names = binary_cols(titanic_df)


# In[47]:


binary_col_names


# In[48]:


# Label Encoding:

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


# In[49]:


for col in binary_col_names:
    label_encoder(titanic_df, col)


# In[50]:


titanic_df.head()


# * ***Rare Encoding***
# 
# Let's examine the class frequencies of categorical variables, if  any class distribution of these variables below 1%, we can combine  then as the "Rare" category.

# In[51]:


def rare_analyser(dataframe, target):
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    cat_cols = [col for col in cat_cols if  col != target in cat_cols]

    for col in cat_cols:
        print(col, ":", dataframe[col].nunique())
        print("dtype:", dataframe[col].dtype)
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(), \
                            "RATIO (%)": dataframe[col].value_counts() / dataframe.shape[0], \
                            "TARGET_MEAN (%) ": dataframe.groupby(col)[target].mean() * 100}))



# In[52]:


rare_analyser(titanic_df, "SURVIVED")


# In[53]:


# Rare Encoder: 

def rare_encoder(dataframe, rare_perc=0.0100):
    rare_df = dataframe.copy()

    rare_columns = [col for col in rare_df.columns if rare_df[col].dtypes == 'O'
                    and (rare_df[col].value_counts() / rare_df.shape[0] <= rare_perc).any(axis=None)]

    for col in rare_columns:
        tmp = rare_df[col].value_counts() / rare_df.shape[0]
        rare_labels = tmp[tmp <= rare_perc].index
        rare_df[col] = np.where(rare_df[col].isin(rare_labels), 'Rare', rare_df[col])

    return rare_df


# In[54]:


new_titanic_df = rare_encoder(titanic_df)


# In[55]:


rare_analyser(new_titanic_df, "SURVIVED")


# Since 2-class variables with frequency less than 1% do not carry any information, we can delete these variables.

# In[56]:


def useless_cols(dataframe, rare_perc=0.01):
    useless_cols = [col for col in dataframe.columns if dataframe[col].nunique() == 2
                    and (dataframe[col].value_counts() / len(dataframe) <= rare_perc).any(axis=None)]
    new_df = dataframe.drop(useless_cols, axis=1)
    return useless_cols 


# In[57]:


# It has been observed that there is no variable which can be considered as useless variable. 

useless_cols(new_titanic_df)


# * ***One-Hot-Encoding***

# In[58]:


def ohe_cols(dataframe):
    ohe_cols = [col for col in dataframe.columns if (dataframe[col].dtype == "O" and 10 >= dataframe[col].nunique() > 2)]
    return ohe_cols


# In[59]:


ohe_col_names = ohe_cols(new_titanic_df)


# In[60]:


def one_hot_encoder(dataframe, ohe_col_names, drop_first=True):
    dms = pd.get_dummies(dataframe[ohe_col_names], drop_first=drop_first)    
    df_ = dataframe.drop(columns=ohe_col_names, axis=1)              
    dataframe = pd.concat([df_, dms],axis=1)                     
    return dataframe


# In[61]:


new_titanic_df = one_hot_encoder(new_titanic_df, ohe_col_names)


# In[62]:


new_titanic_df.head()


# In[63]:


upper_col_name(new_titanic_df)


# In[64]:


new_titanic_df.head(3)


# **STANDARDIZATION**

# In[65]:


cat_cols, num_cols, cat_but_car = grab_col_names(new_titanic_df)


# In[66]:


num_cols


# In[67]:


scaler = StandardScaler()
new_titanic_df[num_cols] = scaler.fit_transform(new_titanic_df[num_cols])


# In[68]:


new_titanic_df.head()


# ***Correlation***

# In[69]:


def high_correlated_cols(dataframe, plot=False, corr_th=0.75):
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    cor_matrix = dataframe[num_cols].corr().abs()
    #corr = dataframe.corr()
    #cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (12, 12)})
        sns.set(font_scale=1) 
        sns.heatmap(cor_matrix, cmap="RdBu",annot=True)
        plt.show()
    return drop_list


# In[70]:


high_correlated_cols(new_titanic_df, plot=True)


# In[71]:


drop_list = ["FARE","SIBSP","PARCH"]

# drop_list = high_correlated_cols(new_titanic_df)


# In[72]:


new_titanic_df = new_titanic_df.drop(drop_list, axis=1)


# In[73]:


new_titanic_df.head()


# In[74]:


# Since different variables related to the "AGE" variable are derived, the "AGE" variable will be excluded and the modeling stage will be started.

new_titanic_df = new_titanic_df.drop(columns="AGE",axis=1)


# # **MODELING**

# # Logistic Regression

# In[75]:


X = new_titanic_df.drop(columns="SURVIVED",axis=1)
y = new_titanic_df[["SURVIVED"]]

# Train- test split:

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=112)

# Model Training

log_model = LogisticRegression().fit(X_train,y_train)

# Prediction

y_pred = log_model.predict(X_test)


# ***Model Performance Metrics***

# In[76]:


# Accuracy Score:
print("Accuracy Score:",accuracy_score(y_test,y_pred))

# Precision:
print("Precision Score:", precision_score(y_test,y_pred))

# Recall:
print("Recall Score:" ,recall_score(y_test,y_pred))

# F1 Score:
print("F1 Score:", f1_score(y_test,y_pred))


# In[77]:


#ROC CURVE 

AUC = logit_roc_auc =roc_auc_score(y_test,y_pred)

plt.figure(figsize=(6,6))
fpr ,tpr,thresholds= roc_curve(y_test,log_model.predict_proba(X_test)[:,1])
plt.plot(fpr,tpr,label ="AUC (area=%0.2f)" % logit_roc_auc)
plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend()
plt.savefig("Log_ROC")
plt.show();


# ***Model Validation***

# In[78]:


cross_val_score(log_model, X_test,y_test,cv=10,scoring= "neg_mean_squared_error")
np.mean(cross_val_score(log_model, X_test,y_test,cv=10))


# ***Feature Importane***

# In[79]:


feature_importance = pd.DataFrame(X_train.columns, columns = ["feature"])
feature_importance["importance"] = pow(mt.e, log_model.coef_[0])
feature_importance = feature_importance.sort_values(by = ["importance"], ascending=False)
 
# Visualization 
ax = feature_importance.plot.barh(x='feature', y='importance', figsize=(12,12), fontsize=10)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.show()


# In[80]:


feature_importance[0:10]


# In[81]:


new_features = feature_importance[0:10]
cols = [col for col in new_features["feature"]]


# In[82]:


X_ = new_titanic_df[cols]
y_ = new_titanic_df[["SURVIVED"]]


X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, 
                                                    test_size=0.20, 
                                                    random_state=112)


log_model_ = LogisticRegression().fit(X_train_,y_train_)


y_pred_ = log_model_.predict(X_test_)

# Accuracy Score:

print("Accuracy Score:",accuracy_score(y_test_,y_pred_))

# Precision:
print("Precision Score:", precision_score(y_test_,y_pred_))

# Recall:
print("Recall Score:" ,recall_score(y_test_,y_pred_))

# F1 Score:
print("F1 Score:", f1_score(y_test_,y_pred_))


# In[83]:


#ROC CURVE 

AUC = logit_roc_auc =roc_auc_score(y_test_,y_pred_)

fpr ,tpr,thresholds= roc_curve(y_test_,log_model_.predict_proba(X_test_)[:,1])
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,label ="AUC (area=%0.2f)" % logit_roc_auc)
plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend()
plt.savefig("Log_ROC")
plt.show();


# In[84]:


# Model Validation 

cross_val_score(log_model, X_test,y_test,cv=10,scoring= "neg_mean_squared_error")
#print(cross_val_score(log_model, X_test,y_test,cv=10))    
np.mean(cross_val_score(log_model, X_test,y_test,cv=10))


# # SUMMARY
# 
# **1. Dataset was read.**
# 
# **2. Exploratory Data Analysis :** 
# 
#     * Data exploration stage has been completed by examining descriptive statistics and seperating categorical and numeric columns.
# 
# **3. Data Preprocessing:**
# 
#     * New features have been extracted from existing features.
#     
#     * The noisy variables (Name, Ticket, Cabin) using for newly derived features, have been removed.
#     
#     * Outliers have been checked by using Boxplot.
#     
#     * Variables with missing values have been handled. By creating Nullity Correlation Matrix, it has been checked whether it was a correlation between missing values.
#     
#     * Missing values in the numeric variables have been filled with the median value by grouping the basis of categorical variables. 
#     
#     * Missing values of categorical variables have been filled by  the mode of the data.
#     
#     * Rare analysis have been applied for categorical variables, if any class distribution of these variables below 1%.
#     
#     * Useless features have been removed.
#     
#     * Outliers have been detected by LOF were dropped.
#     
#     * Dummy variables have been created.
#     
#     * Numerical variables have been standardized.
#     
#     * By examining the correlation between the variables, one of the highly correlated variables have been deleted. 
# 
# 
# **4. Model Building:**
# 
#     * Survival of Titanic passengers have been predicted by using 19 dependent variables and Logistic Regression Model. 
#     
#     * Accuracy, Precision, Recall and F1 scores demonstrating the explanatory and performance of the model have been calculated. 
#     
#     * AUC has been calculated by drawing the ROC curve.
#     
#     * Feature Importance has been calculated for selection of features that contributes the most in predicting the target variable.
#     
# **5. Model Evaluation:**
#  
#      * Cross-validation has been used to estimate the performance of a model for both models (Before and After Feature selection)
#      
#      * Logistic regression model has been created both with all dependent variables and after Feature selection, and model performance metrics have calculated for both cases.   
#      
#      * The best Recall, Precision and F1 scores have been obtained by creating model with all dependent variables. 
#      
# 
# 
# Thank you for your comments and votes:)

# In[ ]:




