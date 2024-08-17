#!/usr/bin/env python
# coding: utf-8

# # TPS MAY 22

# > For this challenge, you are given (simulated) manufacturing control data and are tasked to predict whether the machine is in state 0 or state 1. The data has various feature interactions that may be important in determining the machine state.

# This notebook includes:
# 
# - A brief EDA to get familar with the dataset
# - Feature engineering:
#     - Dealing with the f_27 column by
#     - Introducing the unique_characters feature
#     - Adding 3 interaction features as implemented in [AmbrosM](https://www.kaggle.com/code/ambrosm/tpsmay22-advanced-keras) - inspired by [wti200](https://www.kaggle.com/code/wti200/analysing-interactions-with-shap)
# - Implementing a LightGBM model using LGBMs sklearn API 
# - k-fold validation to estimate performance
# - Basic feature importance estimation
# - Inference on test data
# - A note on ROC AUC score

# My full EDA can be found at: https://www.kaggle.com/code/cabaxiom/tps-may-22-in-depth-eda-feature-engineering
# 
# My notebook visualising feature interactions using target value: https://www.kaggle.com/code/cabaxiom/tps-may-22-visualising-feature-interaction
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')


# In[2]:


train_df = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test_df = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")


# In[3]:


display(train_df.head())
display(test_df.head())


# In[4]:


print(train_df.shape)
print(test_df.shape)
train_df.info()


# ## Target

# In[5]:


def val_count_df(df, column_name, sort=True):
    value_count = df[column_name].value_counts(sort=sort).reset_index().rename(columns={column_name:"Value Count","index":column_name}).set_index(column_name)
    value_count["Percentage"] = df[column_name].value_counts(sort=sort,normalize=True)*100
    value_count = value_count.reset_index()
    return value_count


# In[6]:


target_count = val_count_df(train_df, "target")
display(target_count)
target_count.set_index("target").plot.pie(y="Value Count", figsize=(10,7), legend=False, ylabel="");


# ## Features

# In[7]:


feature_cols = [col for col in train_df.columns if "f_" in col]
dtype_cols = [train_df[i].dtype for i in feature_cols]
dtypes = pd.DataFrame({"features":feature_cols, "dtype":dtype_cols})
float_cols = dtypes.loc[dtypes["dtype"] == "float64", "features"].values.tolist()
int_cols = dtypes.loc[dtypes["dtype"] == "int64", "features"].values.tolist()


# In[8]:


plt.subplots(figsize=(25,20))
sns.heatmap(train_df.corr(),annot=True, cmap="RdYlGn", fmt = '0.2f', vmin=-1, vmax=1, cbar=False);


# In[9]:


plt.subplots(figsize=(25,35))
for i, column in enumerate(float_cols):
    plt.subplot(6,3,i+1)
    sns.histplot(data=train_df, x=column, hue="target")
    plt.title(column)


# In[10]:


plt.subplots(figsize=(25,30))
for i, column in enumerate(int_cols):
    val_count = train_df[column].value_counts()
    ax = plt.subplot(5,3,i+1)
    #sns.barplot(x=val_count.index,y=val_count.values)
    ax.bar(val_count.index, val_count.values)
    ax.set_xticks(val_count.index)
    plt.title(column)


# ## f_27

# In[11]:


import string
alphabet_upper = list(string.ascii_uppercase)

char_counts = []
for character in alphabet_upper:
    char_counts.append(train_df["f_27"].str.count(character).sum())
char_counts_df = pd.DataFrame({"Character": alphabet_upper, "Character Count": char_counts})
char_counts_df = char_counts_df.loc[char_counts_df["Character Count"] > 0]
print(np.sum(char_counts)) #No other hidden characters

plt.subplots(figsize=(20,7))
sns.barplot(data = char_counts_df, x="Character", y="Character Count", color="blue");
plt.title("Total number of characters in f_27 - train");


# In[12]:


char_counts_df = char_counts_df.set_index("Character", drop=False)
for i in range(10):
    char_counts_df["character"+str(i+1)] = train_df["f_27"].str[i].value_counts()
char_counts_df = char_counts_df.fillna(0)


f,ax = plt.subplots(figsize=(20,30))
character_cols = [i for i in char_counts_df.columns if "character" in i]
for i, column in enumerate(character_cols):
    ax = plt.subplot(5,2,i+1)
    ax = sns.barplot(data = char_counts_df, x="Character", y=column, color="blue");
    plt.title("Character value counts in position: " +str(i+1));
    ax.set_ylabel("Character Count")


# # Feature Engineering

# **In this section I describe, explain and implement newly created features.**
# 
# 
# We can use the string from f_27 to create new features.
# 
# Firstly we can create a seperate feature for all 10 character positions in f_27. Instead of using the character values it may be more useful to encode the characters ordinally (A=0, B=1, C=2, ...) 
# 
# For example the character in the first posititions could be

# In[13]:


display(train_df["f_27"].head(5)) #f_27 column
display(train_df["f_27"].str[0].head(5)) # character is position 1
display(train_df["f_27"].str[0].apply(lambda x: ord(x) - ord("A")).head(5)) # Encode the characters ordinally


# Another important feature I found is the number of unique characters for the f_27 string. Forr example "AABAABAABA" has 2 unique characters ("A" and "B").

# In[14]:


display(train_df["f_27"].head(5)) #f_27 column
train_df["f_27"].apply(lambda x: len(set(x))).head(5) #number of unique characters


# There are also some important interaction features as first shown by AmbrosM ([View](https://www.kaggle.com/code/ambrosm/tpsmay22-advanced-keras/notebook)) whose work was inspired by wti200's [Notebook](https://www.kaggle.com/code/wti200/analysing-interactions-with-shap).
# 
# We can view these important interactions in a simple scatter plot:

# In[15]:


f,ax = plt.subplots(figsize=(20,20))

plt.subplot(2,2,1)
sns.scatterplot(data = train_df, x="f_00", y="f_26", hue="target", s=2);
plt.subplot(2,2,2)
sns.scatterplot(data = train_df, x="f_01", y="f_26", hue="target", s=2);
plt.subplot(2,2,3)
sns.scatterplot(data = train_df, x="f_02", y="f_21", hue="target", s=2);
plt.subplot(2,2,4)
sns.scatterplot(data = train_df, x="f_05", y="f_22", hue="target", s=2);


# We notice the graphs of f_00 vs f_26 and f_01 vs f_26 are very similar. Lets try adding f_00 and f_01 and then plotting against f_26:

# In[16]:


train_df["f_00 + f_01"] =  train_df["f_00"] + train_df["f_01"]
f,ax = plt.subplots(figsize=(10,10))
sns.scatterplot(data = train_df, x="f_00 + f_01", y="f_26", hue="target", s=2);


# We notice a very unusual interaction creating three very distinct regions for each of these interactions. Lets try adding all involved features together and plotting them against a random number drawn from a normal distribution: 

# In[17]:


train_df["f_00 + f_01 + f_26"] = train_df["f_00"] + train_df["f_01"] + train_df["f_26"]
train_df["f_02 + f_21"] = train_df["f_02"] + train_df["f_21"]
train_df["f_05 + f_22"] = train_df["f_05"] + train_df["f_22"]
train_df["random"] = np.random.randn(len(train_df))


# In[18]:


f,ax = plt.subplots(figsize=(20,20))

plt.subplot(2,2,1)
sns.scatterplot(data = train_df, y="f_00 + f_01 + f_26", x="random", hue="target", s=2);
plt.subplot(2,2,2)
sns.scatterplot(data = train_df, y="f_02 + f_21", x="random", hue="target", s=2);
plt.subplot(2,2,3)
sns.scatterplot(data = train_df, y="f_05 + f_22", x="random", hue="target", s=2);


# This makes the interaction a lot easier to visualise. We can easily create a feature which returns a value of 1 if the point falls in the top region, a value of 0 if the point falls in the middle region and  and a value of -1 if the point falls in the bottom region. 
# 
# For example if f_05 + f_22 > 5.1 then we want a feature that will return 1, and if  f_05 + f_22 < -5.4 return -1 - else return 0. We can do this with the following line from the next cell:
# 
# `new_df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)`
# 
# We could just use the `f_05 + f_22` as a feature and the model will likely find the boundary but it also helps to explicitly state the boundaries by hand.
# 
# If you want to search for more interaction features I have plotted all possible features against each other in the following notebook:
# 
# https://www.kaggle.com/code/cabaxiom/tps-may-22-visualising-feature-interaction

# In[19]:


def feature_engineer(df):
    new_df = df.copy()
    
    # Interaction features from AmbrosM https://www.kaggle.com/code/ambrosm/tpsmay22-advanced-keras/notebook
    # Inspired by wti200 https://www.kaggle.com/code/wti200/analysing-interactions-with-shap
    new_df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
    new_df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
    
    i_00_01_26 = df.f_00 + df.f_01 + df.f_26
    new_df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)
    
    #Good features
    for i in range(10):
        new_df["f_27_"+str(i)] = new_df["f_27"].str[i].apply(lambda x: ord(x) - ord("A"))
    
    #good feature:
    new_df["unique_characters"] = new_df["f_27"].apply(lambda x: len(set(x)))
    
    new_df = new_df.drop(columns=["f_27", "id"])
    return new_df


# In[20]:


get_ipython().run_cell_magic('time', '', 'train_df.drop(columns = ["f_00 + f_01", "f_00 + f_01 + f_26", "f_02 + f_21", "f_05 + f_22", "random"], inplace=True) # drop the features we made earlier for demonstration\ntrain_df = feature_engineer(train_df)\ntest_df = feature_engineer(test_df)\n')


# In[21]:


train_df["unique_characters"].value_counts()


# # Model

# In[22]:


from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold


# In[23]:


y = train_df["target"]
X = train_df.drop(columns=["target"])
X_test = test_df
X.head(2)


# In[24]:


model = LGBMClassifier(n_estimators = 10000, learning_rate = 0.1, random_state=0, min_child_samples=90, num_leaves=150, max_bins=511, n_jobs=-1)


# The variation in roc_auc score across folds is very small - so we save time and use 5-fold cross validation but only evaluate 2 of the 5 folds.

# In[25]:


def k_fold_cv(model,X,y):
    kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state = 0)

    feature_imp, y_pred_list, y_true_list, acc_list, roc_list  = [],[],[],[],[]
    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
        if fold < 2: # only evaluate 2/5 folds to save time
            print("==fold==", fold)
            X_train = X.loc[train_index]
            X_val = X.loc[val_index]

            y_train = y.loc[train_index]
            y_val = y.loc[val_index]

            model.fit(X_train,y_train)

            y_pred = model.predict_proba(X_val)[:,1]

            y_pred_list = np.append(y_pred_list, y_pred)
            y_true_list = np.append(y_true_list, y_val)

            roc_list.append(roc_auc_score(y_val,y_pred))
            acc_list.append(accuracy_score(y_pred.round(), y_val))
            print("roc auc", roc_auc_score(y_val,y_pred))
            print('Acc', accuracy_score(y_pred.round(), y_val))

            try:
                feature_imp.append(model.feature_importances_)
            except AttributeError: # if model does not have .feature_importances_ attribute
                pass # returns empty list
    return feature_imp, y_pred_list, y_true_list, acc_list, roc_list, X_val, y_val


# In[26]:


get_ipython().run_cell_magic('time', '', 'feature_imp, y_pred_list, y_true_list, acc_list, roc_list, X_val, y_val = k_fold_cv(model=model,X=X,y=y)\n')


# In[27]:


print("Mean accuracy Score:", np.mean(acc_list))
print("Mean ROC AUC Score:", np.mean(roc_list))


# In[28]:


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
def plot_cm(preds,true,ax=None):
    cm = confusion_matrix(preds.round(), true)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)#display_labels 
    disp.plot(ax=ax, colorbar=False, values_format = '.6g')
    plt.grid(False)
    return disp


# In[29]:


plot_cm(y_pred_list, y_true_list);


# In[30]:


val_preds = pd.DataFrame({"pred_prob=1":y_pred_list, "y_val":y_true_list})
f,ax = plt.subplots(figsize=(20,20))
plt.subplot(2,1,1)
ax = sns.histplot(data=val_preds, x="pred_prob=1", hue="y_val", multiple="stack", bins = 100)
#Same plot "zoomed in"
plt.subplot(2,1,2)
ax = sns.histplot(data=val_preds, x="pred_prob=1", hue="y_val", multiple="stack", bins = 100)
ax.set_ylim([0,1000]);


# # Feature Importance

# In[31]:


def fold_feature_importances(model_importances, column_names, model_name, n_folds = 5, ax=None, boxplot=False):
    importances_df = pd.DataFrame({"feature_cols": column_names, "importances_fold_0": model_importances[0]})
    for i in range(1,n_folds):
        importances_df["importances_fold_"+str(i)] = model_importances[i]
    importances_df["importances_fold_median"] = importances_df.drop(columns=["feature_cols"]).median(axis=1)
    importances_df = importances_df.sort_values(by="importances_fold_median", ascending=False)
    if ax == None:
        f, ax = plt.subplots(figsize=(15, 25))
    if boxplot == False:
        ax = sns.barplot(data = importances_df, x = "importances_fold_median", y="feature_cols", color="blue")
        ax.set_xlabel("Median Feature importance across all folds");
    elif boxplot == True:
        importances_df = importances_df.drop(columns="importances_fold_median")
        importances_df = importances_df.set_index("feature_cols").stack().reset_index().rename(columns={0:"feature_importance"})
        ax = sns.boxplot(data = importances_df, y = "feature_cols", x="feature_importance", color="blue", orient="h")
        ax.set_xlabel("Feature importance across all folds");
    plt.title(model_name)
    ax.set_ylabel("Feature Columns")
    return ax


# In[32]:


f, ax = plt.subplots(figsize=(15, 20))
fold_feature_importances(model_importances = feature_imp, column_names = X_val.columns, model_name = "LGBM", n_folds = 2, ax=ax, boxplot=False);


# # Submission

# In[33]:


def pred_test():
    pred_list = []
    for seed in range(5):
        model = LGBMClassifier(n_estimators = 10000, learning_rate = 0.1, min_child_samples=90, num_leaves=150, max_bins=511, random_state=seed, n_jobs=-1)
        model.fit(X,y)

        preds = model.predict_proba(X_test)[:,1]
        pred_list.append(preds)
    return pred_list


# In[34]:


pred_list = pred_test()
pred_df = pd.DataFrame(pred_list).T
pred_df = pred_df.rank()
pred_df["mean"] = pred_df.mean(axis=1)
pred_df


# In[35]:


sample_sub = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
sample_sub["target"] = pred_df["mean"]
sample_sub


# **Question:**
# 
# If we are predicting probabilities, why do these target scores not fall between 0 and 1?
# 
# **Answer:**
# 
# The evaluation metric is ROC AUC.
# 
# One way of interpreting AUC is: **the probability that the model ranks a random positive example more highly than a random negative example.**
# 
# Our model can be used to output the predicted probability. The absolute values of the predictions do not matter - it does not matter how much higher the random positive example is than the random negative example, we are only interested in the rankings between them.
# 
# In other words the ROC AUC score is scale invariant. **AUC measures how well the predictions are ranked**.
# 
# Therefore we can use the predicted probability ranks rather than the predicted probabilities when calculating the ROC AUC score.
# 
# 
# Example:

# In[36]:


pred_df = pd.DataFrame(y_pred_list, columns=["pred_prob"])
pred_df["rank"] = pred_df.rank()
display(pred_df.head(10))

print("roc auc using prediction probabilities:", roc_auc_score(y_true_list, pred_df["pred_prob"]))
print("roc auc using predicted probabilities ranks:", roc_auc_score(y_true_list, pred_df["rank"]))


# It may be better to use ranks rather than probabilities as it allows us to combine multiple sets of predictions together without bias towards one set of predictions.

# In[37]:


sample_sub.to_csv('submission.csv', index = False)


# # Next steps

# - Further feature engineering, such as searching for more interaction features - you may find this notebook useful for this: https://www.kaggle.com/code/cabaxiom/tps-may-22-visualising-feature-interaction.
# - Hyperparameter tuning.
# - Considering using a different Gradient Boosted Decision Tree algorithm (XGBoost, CatBoost, [SKLearn's HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html),  [SKLearn's GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier)).
# - Consider not using the LightGBMs SKLearn API - LGBMs SKlearns API works well but I find it can be a little less flexible / throws warnings.
# - Consider implementing both a GBDT and a NN model e.g. https://www.kaggle.com/code/ambrosm/tpsmay22-advanced-keras.
