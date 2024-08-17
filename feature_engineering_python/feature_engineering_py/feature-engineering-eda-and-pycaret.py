#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pycaret')

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing


# # Introduction
# 
# While working on another experimental [notebook](https://www.kaggle.com/taranmarley/automl-from-scratch-1), I realised I have never taken a real attempt at the original Titanic dataset. This surprises me because it is a fun and approachable dataset. Let's see what we get. 

# # Feature Engineering
# 
# The aim here is to examine the features and then see if we can change everything to numerical so that it can be easily understood by machine learning algorithms.

# **Load Data**

# In[2]:


import numpy as np 
import pandas as pd 

df = pd.read_csv("../input/titanic/train.csv")
test_df = pd.read_csv("../input/titanic/test.csv")


# **Look at Datatypes**
# 
# I've highlighted the objects in red to indicate they are the ones we need to work on.

# In[3]:


from termcolor import colored
for idx, item in df.dtypes.iteritems():
    if item == "object":
        print("{:<15}".format(idx), colored(item, "red"))
    else:
        print("{:<15}".format(idx), colored(item, "green"))


# **Detect NaNs in Dataset**
# 
# NaN or null values can't be comprehended by most machine learning methods, so it is important to detect them.

# In[4]:


def detect_NaNs(df_temp): 
    print('NaNs in data: ', df_temp.isnull().sum().sum())
    print('******')
    count_nulls = df_temp.isnull().sum().sum()
    if count_nulls > 0:
        for col in df_temp.columns:
            print('NaNs in', col + ": ", df_temp[col].isnull().sum().sum())
    print('******')
    print('')
detect_NaNs(df)
detect_NaNs(test_df)


# **Plot NaNs**

# In[5]:


ax = sns.barplot(x=df[["Age","Cabin","Embarked"]].isnull().sum().index, y=df[["Age","Cabin","Embarked"]].isnull().sum().values)
ax.set_title("Training Data")
plt.show()
ax2 = sns.barplot(x=df[["Age","Fare","Cabin"]].isnull().sum().index, y=df[["Age","Fare","Cabin"]].isnull().sum().values)
ax2.set_title("Test Data")
plt.show()


# **Fill in NaNs**
# 
# I will fill the NaN values and creates a new column that records they were NaNsfor col in columns:
#         df_temp[col + "_was_null"] = df_temp[col].isnull().astype(int)
#         df_temp[col] = df_temp[col].fillna(value)

# In[6]:


for col in df.columns:
    if df[col].isnull().values.any(): 
        df[col + "_was_null"] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
        
for col in test_df.columns:
    if test_df[col].isnull().values.any(): 
        test_df[col + "_was_null"] = test_df[col].isnull().astype(int)
        test_df[col] = df[col].fillna(0)
df.head()


# **Let's look at Cabin**

# In[7]:


df["Cabin"].unique()


# The first thing that jumps to mind is that the first letter can be extracted out of this. 

# In[8]:


df["Cabin_First_Letter"] = df["Cabin"].str[:1]
df["Cabin_First_Letter"] = df["Cabin_First_Letter"].fillna(0)
test_df["Cabin_First_Letter"] = test_df["Cabin"].str[:1]
test_df["Cabin_First_Letter"] = test_df["Cabin_First_Letter"].fillna(0)


# In[9]:


df.head(2)


# **Let's look at the ticket**

# In[10]:


df["Ticket"].unique()[:20]


# Pretty clear we would want to break out the first word

# In[11]:


df["Ticket_First"] = df.Ticket.str.split().str.get(0)
test_df["Ticket_First"] = test_df.Ticket.str.split().str.get(0)


# This could also be helpful in consideration of the name where the last name might imply family relations

# In[12]:


df["Name_First"] = df.Name.str.split().str.get(0)
test_df["Name_First"] = test_df.Name.str.split().str.get(0)


# **Create Interactions**
# 
# We can create some interactions between various columns to generate new data that may help a machine learning method

# In[13]:


df.head()


# In[14]:


df["PclassXSibSp"] = df["Pclass"] * df["SibSp"]
test_df["PclassXSibSp"] = test_df["Pclass"] * test_df["SibSp"]
df["AgeXSibSp"] = df["Age"] * df["SibSp"]
test_df["AgeXSibSp"] = test_df["Age"] * test_df["SibSp"]
df["AgeXFare"] = df["Age"] * df["Fare"]
test_df["AgeXFare"] = test_df["Age"] * test_df["Fare"]
df["PclassXFare"] = df["Pclass"] * df["Fare"]
test_df["PclassXFare"] = test_df["Pclass"] * test_df["Fare"]


# **Check for id columns**
# 
# If a column has a unique value for every row we should delete it because that is of little predictive value for use and would get used by a machine learning algorithm as a cheat.

# In[15]:


for col in df.columns:
    if len(df[col]) == len(df[col].unique()):
        df.drop(columns=col, inplace=True)
        test_df.drop(columns=col, inplace=True)


# **Encode Columns**

# In[16]:


def encode_columns(df, columns, test_df = None):
    for col in columns:
        le = preprocessing.LabelEncoder()
        classes_to_encode = df[col].astype(str).unique().tolist()
        classes_to_encode.sort()
        classes_to_encode.append('None')
        le.fit(classes_to_encode)
        if len(le.classes_) < 12:
            df = pd.get_dummies(df, columns = [col])
            if test_df is not None:
                test_df = pd.get_dummies(test_df, columns = [col])
        else:
            check_col = df.copy()[col]
            df[col] = le.transform(df[col].astype(str))
            if test_df is not None:
                #Clean out unseen labels
                inputs = []
                for idx, row in test_df.iterrows():
                    if row[col] in pd.unique(check_col):
                        inputs.append(row[col])
                    else:
                        inputs.append('None')
                test_df[col] = inputs
                test_df[col] = le.transform(test_df[col].astype(str))
    return df, test_df
#encode_columns(df, ["HomePlanet", "CryoSleep", "Destination", "VIP", "Name", "letters", "final_letters"], test_df)
df, test_df = encode_columns(df, ["Sex", "Ticket","Embarked","Cabin", "Cabin_First_Letter", "Ticket_First", "Name_First"], test_df)


# Let's check this has gotten rid of any objects:

# In[17]:


from termcolor import colored
for idx, item in df.dtypes.iteritems():
    if item == "object":
        print("{:<15}".format(idx), colored(item, "red"))
    else:
        print("{:<15}".format(idx), colored(item, "green"))


# All green is great! Now we need to check that the columns are matched between test and training data by removing those that aren't.

# In[18]:


for col in df.columns:
    if col not in test_df.columns:
        if col != "Survived":
            df.drop(columns=col, axis=1, inplace=True)
for col in test_df.columns:
    if col not in df.columns:
        if col  != "Survived":
            test_df.drop(columns=col, axis=1, inplace=True)


# **Search for Anomolies**
# 
# We can look for anomalies and also add this information to the dataset. 

# In[19]:


from sklearn.ensemble import IsolationForest

X = df.copy().drop(columns="Survived")
test_X = test_df.copy()
iforest = IsolationForest(random_state=0).fit(X)
df["anomaly"] = iforest.predict(X)
df["anomaly_score"] = iforest.score_samples(X)
test_df["anomaly"] = iforest.predict(test_X)
test_df["anomaly_score"] = iforest.score_samples(test_X)


# In[20]:


df["anomaly"] = df["anomaly"].replace({-1:0})
test_df["anomaly"] = df["anomaly"].replace({-1:0})


# In[21]:


from yellowbrick.features import PCA as yellowPCA

y = df["anomaly"]
X = df.drop(columns=["anomaly"]).replace({-1:0})

visualizer = yellowPCA(scale=True, projection=2, alpha=0.4)
visualizer.fit_transform(X, y)
visualizer.show()
plt.show()


# **Add PCA Features**
# 
# The above graph shows that PCA with anomaly generates some clusters and that could be used to classify. Below I will combine test and training dataframes to do a pca on both. 

# In[22]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

df_temp = pd.concat([df.copy(), test_df.copy()], ignore_index=True)
y = df_temp["Survived"]
X = df_temp.drop(columns="Survived", axis=1)
X_scaled = MinMaxScaler().fit_transform(X)
pca = PCA(n_components=3)
X_p = pca.fit(X_scaled).transform(X_scaled)
df["PCA_0"] = X_p[:891,0]
df["PCA_1"] = X_p[:891,1]
df["PCA_2"] = X_p[:891,2]
test_df["PCA_0"] = X_p[891:,0]
test_df["PCA_1"] = X_p[891:,1]
test_df["PCA_2"] = X_p[891:,2]


# # Exploratory Data Analysis
# 
# **Let's look at the Target First**

# In[23]:


sns.countplot(x=df["Survived"])
plt.show()


# **Plot Data**

# In[24]:


pltdf = df.copy()
pltdf = pltdf.sample(frac=1, random_state=42).reset_index(drop=True)
pltdf.iloc[:50, :16].plot(subplots=True, layout=(7,4), figsize=(15,10))

plt.show()


# **Pivot Table**

# In[25]:


from sklearn.preprocessing import QuantileTransformer

def quantile_column_wise(df_temp, target_col = ""):
    df_temp = df_temp.copy()
    for col in df_temp.columns:
        if col != target_col:
            df_temp[col] = QuantileTransformer(n_quantiles=500).fit_transform(df_temp[col].values.reshape(-1, 1))
    return df_temp

def pivot_table(df_temp, target_col):
    df_temp = df_temp.copy()
    y = df_temp[target_col]
    X = df_temp.drop(columns=target_col, axis=1)
    df_temp = pd.DataFrame(X)  
    df_temp.columns = X.columns
    df_temp[target_col] = y
    table = pd.pivot_table(data=df_temp,index=[target_col]).T
    sns.heatmap(table, annot=True, cmap="Blues")
    return table


# In[26]:


plt.figure(figsize=(10,15))
table = pivot_table(quantile_column_wise(df, "Survived"), "Survived")
plt.show()


# Gender is very visible in this as a difference.
# 
# **See Important Correlations**
# 
# I will generate a heatmap below with the more important data correlations

# In[27]:


from sklearn import preprocessing

def calculate_correlations(df_temp, target_col, ratio, verbose=1):
    df_temp = df_temp.copy()
    cols = []
    cols_done = []
    if df_temp[target_col].dtype == object:
        le = preprocessing.LabelEncoder()
        df_temp[target_col] = le.fit_transform(df_temp[target_col])
    df_temp[target_col] = MinMaxScaler().fit_transform(df_temp[target_col].values.reshape(-1, 1))
    if verbose == 1:
        print("Correlations with",target_col + ":")
    for col_one in df_temp.iloc[:,:].columns:
        correlation_value =  abs(df_temp[col_one].corr(df_temp[target_col]))
        if verbose == 1:
            print(col_one, ":", df_temp[col_one].corr(df_temp[target_col]))
        if correlation_value > ratio:
            cols.append(col_one)
        cols_done.append(col_one)
    corrdf = df_temp.copy()
    corrdf = corrdf[cols].corr()
    sns.heatmap(abs(corrdf), cmap="Blues")
    return cols


# In[28]:


correlation_cols = calculate_correlations(df.drop(columns="Sex_female"), "Survived", 0.2, 0)


# # Plot Bar Plots

# In[29]:


def bar_plots(df, columns_to_plot, target_col):
    df_temp = df.copy()
    fig, axs = plt.subplots(len(columns_to_plot), 1, figsize=(10, 15))
    i = 0 
    for col in columns_to_plot:
        sns.barplot(ax=axs[i], x=target_col, y=col, data=df)# .set_title(col + " X " + target_col)
        axs[i].set_title = "test"
        i = i + 1
    # fig.subplots_adjust(hspace=0.4)
bar_plots(df, ["Sex_male", "Pclass", "Cabin", "AgeXFare"], "Survived")


# # Pair Grid

# In[30]:


def pair_grid_plot(df, cols):
    g = sns.PairGrid(df[cols].iloc[:500,:], diag_sharey=False)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)
    
pair_grid_plot(df, ["Sex_male", "Pclass", "Cabin", "AgeXFare", "Survived"])


# # Dimensionality Reduction

# In[31]:


from sklearn.decomposition import PCA

def pca_dimension_reduction_info(df_temp, target_col):
    df_temp = df_temp.copy()
    y = df_temp[target_col]
    X = df_temp.drop(columns=target_col, axis=1)
    X_scaled = MinMaxScaler().fit_transform(X)
    print(str(len(X_scaled[0])) + " initial feature components")
    pca = PCA(n_components=0.95)
    X_p = pca.fit(X_scaled).transform(X_scaled)
    print("95% variance explained by " + str(len(X_p[0])) + " components by principle component analysis")
    pca = PCA(n_components=3)
    X_p = pca.fit(X_scaled).transform(X_scaled)
    print(str(round(pca.explained_variance_ratio_.sum() * 100)) + "% variance explained by 3 components by principle component analysis")
    pca = PCA(n_components=2)
    X_p = pca.fit(X_scaled).transform(X_scaled)
    print(str(round(pca.explained_variance_ratio_.sum() * 100)) + "% variance explained by 2 components by principle component analysis")

pca_dimension_reduction_info(df.drop(columns=["PCA_0","PCA_1","PCA_2"]), "Survived")


# In[32]:


from yellowbrick.features import PCA as yellowPCA

y = df["Survived"]
X = df.drop(columns=["Survived","PCA_0","PCA_1","PCA_2"])

visualizer = yellowPCA(scale=True, projection=2, alpha=0.4)
visualizer.fit_transform(X, y)
visualizer.show()
visualizer = yellowPCA(scale=True, projection=3, alpha=0.4, size=(700,700))
visualizer.fit_transform(X, y)
visualizer.show()
plt.show()


# # Decision Tree
# 
# Simple decision trees can be easily interpreted for knowledge about the data

# In[33]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

def decision_tree(df_temp, depth, target_col):
    tree_set = df_temp.copy()
    target = tree_set[target_col]
    tree_set.drop([target_col], axis=1, inplace=True)
    tree_clf = DecisionTreeClassifier(max_depth=depth, random_state=1)
    tree_clf.fit(tree_set, target)
    text_representation = tree.export_text(tree_clf, feature_names=tree_set.columns.tolist())
    print("accuracy: " + str(tree_clf.score(tree_set, target)))    
    plt.figure(figsize=(18,18))
    # tree.plot_tree(tree_clf, feature_names=tree_set.columns, filled=True)
    class_column_values = df_temp[target_col].values.ravel()
    class_unique_values = pd.unique(class_column_values)
    class_unique_values = np.sort(class_unique_values)
    class_unique_values = class_unique_values.astype('str')
    le = preprocessing.LabelEncoder()
    target = le.fit_transform(target)
    dot_data = tree.export_graphviz(tree_clf, out_file=None, 
                                    feature_names=tree_set.columns,  
                                    class_names=class_unique_values,
                                    filled=True)
    display(graphviz.Source(dot_data, format="png")) 


# In[34]:


decision_tree(df, 3, "Survived")
plt.show()


# Wow 82% accuracy from 3 splits.

# # Pycaret
# 
# I will go through here and determine the best model to classify on this dataset. I may exclude some models to keep the time to run this notebook on kaggle under control.

# I will remove some columns here that in testing just weren't helpful:

# In[35]:


df = df.drop(columns="Name_First")
test_df = test_df.drop(columns="Name_First")


# In[36]:


from pycaret.classification import *
from sklearn import preprocessing

setup(data = df.copy(), 
             target = "Survived",
             silent = True, normalize = True, session_id=1, data_split_stratify=True)
display()


# In[37]:


top3 = compare_models(n_select=3, exclude=["xgboost","catboost","gbc"])


# I will blend the top 3 for a combined model

# In[38]:


blend = blend_models(top3)


# In[39]:


final_blend = finalize_model(blend)


# In[40]:


plot_model(final_blend, "confusion_matrix")


# In[41]:


plot_model(final_blend, "error")


# In[42]:


plot_model(final_blend, "boundary")


# # Create Submission

# In[43]:


predictions = predict_model(final_blend, data=test_df)


# In[44]:


predictions.head()


# In[45]:


submission = pd.read_csv("../input/titanic/gender_submission.csv")
submission["Survived"] = predictions["Label"]
submission.to_csv("submission.csv", index=False)

