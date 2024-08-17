#!/usr/bin/env python
# coding: utf-8

# # 1 Introduction
# 
# The purpose of this EDA to is more closely examine null values within the dataset. Specifically, our goal is to understand how null values are distributed, and come up with a way of possibly interpolating values for nulls that can provide useful information for our machine learning models to work with.

# # 2 Null Value Distribution
# 
# Let's take a look at how null values are spread out throughout the dataset. Understanding how our null data is spread out is very useful when it comes to engineering features, as well as choosing machine learning models. For example, depending on how our null values are distributed, we may be able to get away with filling nulls with zero values or with averages. We need to carefully examine this however, to see what impact this may have on our machine learning models.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[2]:


train = pd.read_csv("../input/tabular-playground-series-sep-2021/train.csv")
test = pd.read_csv("../input/tabular-playground-series-sep-2021/test.csv")


# ### 2.1 Null Value Impact Per Feature
# 
# Let's take a look at how each feature has been impacted. Specifically, let's look at how prevalent null values are across each feature in the dataset. To do that, we'll one-hot encode each feature based on whether it is null. For example, given feature `f1`, we'll generate a new feature called `f1_is_null` and encode it with the following values:
# 
# * `0` - indicates the value is present
# * `1` - indicates the value is null
# 
# We'll then use the `_is_null` features to determine what percentage of each feature contains null values.

# In[3]:


# Define the features we want to examine, names for one-hot encoding them, and the total number of records
features = [x for x in train.columns if x.startswith("f") and not x.endswith("_is_null")]
null_features = ["{}_is_null".format(x) for x in features]
total_rows = float(train.shape[0])

# One-hot encode whether a feature is null per row
for feature, null_feature in zip(features, null_features):
    train[null_feature] = train[feature].isnull().astype(int)

# Generate counts of number of null values we see per feature
null_counts = pd.DataFrame.from_dict({k : [round((train[(train[k]) == 1][k].count() / total_rows) * 100, 3)] for k in null_features})

# Plot percentage of rows impacted by feature
sns.set_style("whitegrid")
bar, ax = plt.subplots(figsize=(10, 35))
ax = sns.barplot(data=null_counts, ci=None, palette="muted", orient="h")
ax.set_title("Percentage of Null Values Per Feature (Train Data)", fontsize=15)
ax.set_xlabel("Percentage")
ax.set_ylabel("Feature")
for rect in ax.patches:
    ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2, "%.3f%%" % rect.get_width())


# Here we can see that every feature - `f1` to `f118` - is impacted by a null value. In particular, roughly 1.6% of the data for each feature is null. We should check to see if we have similar results for the testing dataset as well.

# In[4]:


# Define the features we want to examine, names for one-hot encoding them, and the total number of records
features = [x for x in test.columns if x.startswith("f") and not x.endswith("_is_null")]
null_features = ["{}_is_null".format(x) for x in features]
total_rows = float(test.shape[0])

# One-hot encode whether a feature is null per row
for feature, null_feature in zip(features, null_features):
    test[null_feature] = test[feature].isnull().astype(int)

# Generate counts of number of null values we see per feature
null_counts = pd.DataFrame.from_dict({k : [round((test[(test[k]) == 1][k].count() / total_rows) * 100, 3)] for k in null_features})

# Plot percentage of rows impacted by feature
sns.set_style("whitegrid")
bar, ax = plt.subplots(figsize=(10, 35))
ax = sns.barplot(data=null_counts, ci=None, palette="muted", orient="h")
ax.set_title("Percentage of Null Values Per Feature (Test Data)", fontsize=15)
ax.set_xlabel("Percentage")
ax.set_ylabel("Feature")
for rect in ax.patches:
    ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2, "%.3f%%" % rect.get_width())


# It appears our null value percentages per feature is roughly the same between our training data and our testing data. The question now becomes: how overlapped is the null data? In other words, is every row impacted by a null value, or are there rows that have complete values for each feature?

# ### 2.2 Null Value Impact Per Row
# 
# In order to see the impact of null values for each row, we'll create a count of the null values for that row and store it in a `null_count` feature. Then, we can group by the `null_count` values, and see how many rows contain 0, 1, 2, 3, or more null values.

# In[5]:


# Count the number of null values that occur in each row
train["null_count"] = train.isnull().sum(axis=1)

# Group the null counts
counts = train.groupby("null_count")["claim"].count().to_dict()
null_data = {"{} Null Value(s)".format(k) : v for k, v in counts.items() if k < 6}
null_data["6 or More Null Values"] = sum([v for k, v in enumerate(counts.values()) if k > 5])

# Plot the null count results
pie, ax = plt.subplots(figsize=[20, 10])
plt.pie(x=null_data.values(), autopct="%.2f%%", explode=[0.05]*len(null_data.keys()), labels=null_data.keys(), pctdistance=0.5)
_ = plt.title("Percentage of Null Values Per Row (Train Data)", fontsize=14)


# The results from this breakdown are interesting. Of all the rows of data we have available, 37.5% of them contain a complete set of values for each feature. This means that of all the rows of data, 37.53% of them have numeric values for every feature. On the flipside, some rows may contain more than 6 null values across the 118 features. Again, we should look to see if this distribution of missing values occurs in roughly the same fashion in the test dataset.

# In[6]:


# Count the number of null values that occur in each row
test["null_count"] = test.isnull().sum(axis=1)

# Group the null counts
counts = test.groupby("null_count")["null_count"].count().to_dict()
null_data = {"{} Null Value(s)".format(k) : v for k, v in counts.items() if k < 6}
null_data["6 or More Null Values"] = sum([v for k, v in enumerate(counts.values()) if k > 5])

# Plot the null count results
pie, ax = plt.subplots(figsize=[20, 10])
plt.pie(x=null_data.values(), autopct="%.2f%%", explode=[0.05]*len(null_data.keys()), labels=null_data.keys(), pctdistance=0.5)
_ = plt.title("Percentage of Null Values Per Row (Test Data)", fontsize=14)


# It looks like our testing dataset and training dataset are roughly the same with regards to the null value distribution per row. Slighly more than 1/3rd of our datasets contain a complete set of values on a row.

# ### 2.3 Null Value Correlation with Claim
# 
# Let's check to see if there is any strong correlation with null value features and the `claim` column. To do this, we'll look at areas where `_is_null` features are strongly correlated with the target variable.

# In[7]:


correlation_features = null_features.copy()
correlation_features.append("claim")
null_correlation = train[correlation_features].corr()
null_correlation.style.background_gradient(cmap='coolwarm')

f, ax = plt.subplots(figsize=(30, 30))

# Draw the heatmap with the mask and correct aspect ratio
_ = sns.heatmap(
    null_correlation, 
    mask=np.triu(np.ones_like(null_correlation, dtype=bool)), 
    cmap=sns.diverging_palette(230, 20, as_cmap=True), 
    vmax=.3, 
    center=0,
    square=True, 
    linewidths=.5, 
    cbar_kws={"shrink": .5}
)


# As we can see, there is no standout affinity between a particular null feature and our target variable of `claim`. In other words, as an example, feature `f42` does not show any stronger correlation with the `claim` column than any other feature. No magic feature bullet here.

# ### 2.4 Null Value Count Distribution with Claim
# 
# We should also check to see how null values are distributed in regards to the target variable `claim`. 

# In[8]:


z = dict()
for (null_count, claim_status), value_count in train[["null_count", "claim"]].value_counts().to_dict().items():
    if null_count not in z:
        z[null_count] = dict()
    z[null_count][claim_status] = value_count
a = {
    "Number of Null Values": ["Not Claimed (0)", "Claimed (1)"],
}
a = []
for null_values in range(15):
    a.append([null_values, z[null_values][0], z[null_values][1]])
df = pd.DataFrame(a, columns=["Number of Null Values", "Not Claimed (0)", "Claimed (1)"])
ax = df.plot(x="Number of Null Values", y=["Not Claimed (0)", "Claimed (1)"], kind="bar", figsize=(20, 10))
_ = ax.set_title("Number of Null Values by Claim Status", fontsize=15)
_ = ax.set_ylabel("Number of Rows")


# We can see from these results that when we have 2 or more null values, there is a really good chance that the actual row should have a target `claim` value of `1`. These null values likely correspond to key indicators that a claim should be made. Adding the total number of null values per row as a feature to the dataset will likely help our classifier a lot. It stands to reason that if we could reconstruct what those null values should be, our classifier would be much more sensitive.

# # 3 Impact of Null Values and Null Value Replacement
# 
# Now that we know we have null values in roughly the same distributions in both the training and testing dataset, we should perform a few simple experiments to see how null values are actually impacting our ability to make predictions.
# 
# ### 3.1 Simple LightGBM Classifier
# 
# Let's start by building a simple LightGBM classifier. LightGBM can [intrinsically cope with null values](https://github.com/Microsoft/LightGBM/blob/master/docs/Advanced-Topics.rst#missing-value-handle), however certain other models cannot. But, before we look at filling null values, let's try a simple baseline model.

# In[9]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import gc

# Delete un-needed test data, and garbage collect to save memory
del test
gc.collect()

# Replace nulls with the column mean for each feature
new_train = train.copy()

target = train["claim"]

k_fold = StratifiedKFold(
    n_splits=3,
    random_state=2021,
    shuffle=True,
)

train_preds = np.zeros(len(train.index), )
train_probas = np.zeros(len(train.index), )

for fold, (train_index, test_index) in enumerate(k_fold.split(new_train[features], target)):
    x_train = pd.DataFrame(new_train[features].iloc[train_index])
    y_train = target.iloc[train_index]

    x_valid = pd.DataFrame(new_train[features].iloc[test_index])
    y_valid = target.iloc[test_index]

    model = LGBMClassifier(
        random_state=2021,
        metric="auc",
        n_jobs=6,
        n_estimators=16000,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        early_stopping_rounds=200,
        verbose=0,
    )

    train_oof_preds = model.predict(x_valid)
    train_oof_probas = model.predict_proba(x_valid)[:, -1]
    train_preds[test_index] = train_oof_preds
    train_probas[test_index] = train_oof_probas

    print("-- Fold {}:".format(fold+1))
    print("{}".format(classification_report(y_valid, train_oof_preds)))

print("-- Overall:")
print("{}".format(classification_report(target, train_preds)))
print("-- ROC AUC: {}".format(roc_auc_score(target, train_probas)))

# Delete unused data, and garbage collect to save space
del model
del new_train
_ = gc.collect()

train["unmodified_preds"] = train_preds
train["unmodified_probas"] = train_probas
misclassified = train[(train["claim"] != train["unmodified_preds"])]["null_count"].value_counts().to_dict()

# Show the confusion matrix
confusion = confusion_matrix(train["claim"], train["unmodified_preds"])
ax = sns.heatmap(confusion, annot=True, fmt=",d")
_ = ax.set_title("Confusion Matrix for LGB Classifier (Unmodified Dataset)", fontsize=15)
_ = ax.set_ylabel("Actual Class")
_ = ax.set_xlabel("Predicted Class")

# Plot percentage of rows impacted by feature
sns.set_style("whitegrid")
bar, ax = plt.subplots(figsize=(10, 10))
ax = sns.barplot(x=list(misclassified.keys()), y=list(misclassified.values()))
_ = ax.set_title("Number of Misclassifications by Null Values in Row (Unmodified Dataset)", fontsize=15)
_ = ax.set_xlabel("Number of Null Values in Row")
_ = ax.set_ylabel("Number of Misclassified Predictions")
for p in ax.patches:
    height = p.get_height()
    ax.text(
        x=p.get_x()+(p.get_width()/2),
        y=height,
        s=round(height),
        ha="center"
    )


# Across our three folds, we see precision and recall is very stable. This is good - it suggests that our data is fairly uniform. However, we're now seeing a discrepancy: although the number of records with 1 null value comprises only 14% of our data, those rows generate more misclassifications (72,118 misclassifications) than the rows with complete information which accounts for 37% of our data (48,553 misclassifications). This is true when there are 2 null values per row as well, although we do see a fairly steep dropoff once more than 5 values in a row are null. The question is whether we can improve on this if we substitute different values for nulls. 

# ### 3.2 Mean Fill
# 
# With our first attempt, we will mean fill missing values within each of the columns, then build a classification model. We'll look to see how the classification breaks down based on number of null values. 

# In[10]:


# Replace nulls with the column mean for each feature
new_train = train.copy()
for feature in features:
    new_train[feature].fillna(new_train[feature].mean(), inplace=True)
    
target = train["claim"]

k_fold = StratifiedKFold(
    n_splits=3,
    random_state=2021,
    shuffle=True,
)

train_preds = np.zeros(len(train.index), )
train_probas = np.zeros(len(train.index), )

for fold, (train_index, test_index) in enumerate(k_fold.split(new_train[features], target)):
    x_train = pd.DataFrame(new_train[features].iloc[train_index])
    y_train = target.iloc[train_index]

    x_valid = pd.DataFrame(new_train[features].iloc[test_index])
    y_valid = target.iloc[test_index]

    model = LGBMClassifier(
        random_state=2021,
        metric="auc",
        n_jobs=6,
        n_estimators=16000,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        early_stopping_rounds=200,
        verbose=0,
    )

    train_oof_preds = model.predict(x_valid)
    train_oof_probas = model.predict_proba(x_valid)[:, -1]
    train_preds[test_index] = train_oof_preds
    train_probas[test_index] = train_oof_probas

    print("-- Fold {}:".format(fold+1))
    print("{}".format(classification_report(y_valid, train_oof_preds)))

print("-- Overall:")
print("{}".format(classification_report(target, train_preds)))
print("-- ROC AUC: {}".format(roc_auc_score(target, train_probas)))

# Delete unused data, and garbage collect to save space
del model
del new_train
_ = gc.collect()

train["mean_preds"] = train_preds
train["mean_probas"] = train_probas
misclassified = train[(train["claim"] != train["mean_preds"])]["null_count"].value_counts().to_dict()

# Show the confusion matrix
confusion = confusion_matrix(train["claim"], train["mean_preds"])
ax = sns.heatmap(confusion, annot=True, fmt=",d")
_ = ax.set_title("Confusion Matrix for LGB Classifier (Mean Filled Null)", fontsize=15)
_ = ax.set_ylabel("Actual Class")
_ = ax.set_xlabel("Predicted Class")

# Plot percentage of rows impacted by feature
sns.set_style("whitegrid")
bar, ax = plt.subplots(figsize=(10, 10))
ax = sns.barplot(x=list(misclassified.keys()), y=list(misclassified.values()))
_ = ax.set_title("Number of Misclassifications by Null Values in Row (Mean Filled Null)", fontsize=15)
_ = ax.set_ylabel("Number of Null Values in Row")
_ = ax.set_xlabel("Number of Misclassified Predictions")
for p in ax.patches:
    height = p.get_height()
    ax.text(
        x=p.get_x()+(p.get_width()/2),
        y=height,
        s=round(height),
        ha="center"
    )


# When comparing this result to the original null value results, we can see that using mean values has hurt our model. Groups that had 1 or 2 null values are now getting misclassified at a higher rate. Interestingly, this has come in the form of more false positives (claim values of 0 are being classified as 1 at a much higher rate). This means that we're losing important signal within our mean value - likely the null values are masking outliers within our data. Clearly, substituting average values is causing a problem. 

# 
# ### 3.3 Zero Fill
# 
# With this attempt, we'll fill missing values with zeroes instead. Zero-filling is not a terribly good idea, since we haven't looked at the individual columns. A value of 0 may significantly skew the data in ways we aren't intending it to (e.g. if a column ranges from 10,000 to 100,000 entering a value of 0 will cause a significant change in the distribution of values). However, skews such as this may be detected by tree based models, and may improve our classifier's performance.
# 

# In[11]:


# Replace nulls with zeroes
new_train = train.copy()
for feature in features:
    new_train[feature].fillna(0, inplace=True)
    
target = train["claim"]

k_fold = StratifiedKFold(
    n_splits=3,
    random_state=2021,
    shuffle=True,
)

train_preds = np.zeros(len(train.index), )
train_probas = np.zeros(len(train.index), )

for fold, (train_index, test_index) in enumerate(k_fold.split(new_train[features], target)):
    x_train = pd.DataFrame(new_train[features].iloc[train_index])
    y_train = target.iloc[train_index]

    x_valid = pd.DataFrame(new_train[features].iloc[test_index])
    y_valid = target.iloc[test_index]

    model = LGBMClassifier(
        random_state=2021,
        metric="auc",
        n_jobs=6,
        n_estimators=16000,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        early_stopping_rounds=200,
        verbose=0,
    )

    train_oof_preds = model.predict(x_valid)
    train_oof_probas = model.predict_proba(x_valid)[:, -1]
    train_preds[test_index] = train_oof_preds
    train_probas[test_index] = train_oof_probas

    print("-- Fold {}:".format(fold+1))
    print("{}".format(classification_report(y_valid, train_oof_preds)))

print("-- Overall:")
print("{}".format(classification_report(target, train_preds)))
print("-- ROC AUC: {}".format(roc_auc_score(target, train_probas)))

# Delete unused data, and garbage collect to save space
del model
del new_train
gc.collect()

train["zero_preds"] = train_preds
train["zero_probas"] = train_probas
misclassified = train[(train["claim"] != train["zero_preds"])]["null_count"].value_counts().to_dict()

# Show the confusion matrix
confusion = confusion_matrix(train["claim"], train["zero_preds"])
ax = sns.heatmap(confusion, annot=True, fmt=",d")
_ = ax.set_title("Confusion Matrix for LGB Classifier (Zero Filled Null)", fontsize=15)
_ = ax.set_ylabel("Actual Class")
_ = ax.set_xlabel("Predicted Class")

# Plot percentage of rows impacted by feature
sns.set_style("whitegrid")
bar, ax = plt.subplots(figsize=(10, 10))
ax = sns.barplot(x=list(misclassified.keys()), y=list(misclassified.values()))
_ = ax.set_title("Number of Misclassifications by Null Values in Row (Zero Filled Null)", fontsize=15)
_ = ax.set_ylabel("Number of Null Values in Row")
_ = ax.set_xlabel("Number of Misclassified Predictions")
for p in ax.patches:
    height = p.get_height()
    ax.text(
        x=p.get_x()+(p.get_width()/2),
        y=height,
        s=round(height),
        ha="center"
    )


# With zero fill, we're seeing a reduction in the number of misclassified predictions for rows missing two values. The ROC AUC value is also slightly improved, although it still lags below the baseline LightGBM model that was able to use null values as part of its model building process.
# 
# ### 3.4 Iterative Imputer
# 
# Let's take a look to see how a simple imputer works. SciKit Learn's `IterativeImputer` attempts to fill in valid values for missing fields by finding other rows similar to the current one, and interpolating the missing field's value. It's a process that can take a long time to complete, depending on how many other rows it examines. To speed it up, we'll only use 5 nearest neighbors to make a value determination.

# In[12]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

new_train = train.copy()
imputer = IterativeImputer(random_state=2021, n_nearest_features=5)
new_train[features] = imputer.fit_transform(new_train[features])
    
target = train["claim"]

k_fold = StratifiedKFold(
    n_splits=3,
    random_state=2021,
    shuffle=True,
)

train_preds = np.zeros(len(train.index), )
train_probas = np.zeros(len(train.index), )

for fold, (train_index, test_index) in enumerate(k_fold.split(new_train[features], target)):
    x_train = pd.DataFrame(new_train[features].iloc[train_index])
    y_train = target.iloc[train_index]

    x_valid = pd.DataFrame(new_train[features].iloc[test_index])
    y_valid = target.iloc[test_index]

    model = LGBMClassifier(
        random_state=2021,
        metric="auc",
        n_jobs=6,
        n_estimators=16000,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        early_stopping_rounds=200,
        verbose=0,
    )

    train_oof_preds = model.predict(x_valid)
    train_oof_probas = model.predict_proba(x_valid)[:, -1]
    train_preds[test_index] = train_oof_preds
    train_probas[test_index] = train_oof_probas

    print("-- Fold {}:".format(fold+1))
    print("{}".format(classification_report(y_valid, train_oof_preds)))

print("-- Overall:")
print("{}".format(classification_report(target, train_preds)))
print("-- ROC AUC: {}".format(roc_auc_score(target, train_probas)))

# Delete unused data, and garbage collect to save space
del model
del new_train
gc.collect()

train["imputer_preds"] = train_preds
train["imputer_probas"] = train_probas
misclassified = train[(train["claim"] != train["imputer_preds"])]["null_count"].value_counts().to_dict()

# Show the confusion matrix
confusion = confusion_matrix(train["claim"], train["imputer_preds"])
ax = sns.heatmap(confusion, annot=True, fmt=",d")
_ = ax.set_title("Confusion Matrix for LGB Classifier (Iterative Imputer)", fontsize=15)
_ = ax.set_ylabel("Actual Class")
_ = ax.set_xlabel("Predicted Class")

# Plot percentage of rows impacted by feature
sns.set_style("whitegrid")
bar, ax = plt.subplots(figsize=(10, 10))
ax = sns.barplot(x=list(misclassified.keys()), y=list(misclassified.values()))
_ = ax.set_title("Number of Misclassifications by Null Values in Row (Iterative Imputer)", fontsize=15)
_ = ax.set_xlabel("Number of Null Values in Row")
_ = ax.set_ylabel("Number of Misclassified Predictions")
for p in ax.patches:
    height = p.get_height()
    ax.text(
        x=p.get_x()+(p.get_width()/2),
        y=height,
        s=round(height),
        ha="center"
    )


# We can now see some differences appear. While we have reduced the number of misclassifications with rows that have a single null value, we have fundamentally shifted the distribution of data across multiple columns. We can see this in the number of misclassifications with rows that have no missing data - suddenly we have increased the number of misclassifications above 60,000 from our initial value of 48,553. This could be due to the fact that we're not imputing a realistic value for fields that have null values. More likely, we're starting to discover that the null values are hiding important data that cannot be interpolated using the `IterativeImputer`.

# ### 3.5 Use Null Count as Feature
# 
# Rather than attempt to save the null values, we can instead just use the null counts as features in addition to the raw data. 

# In[13]:


new_train = train.copy()

target = train["claim"]

k_fold = StratifiedKFold(
    n_splits=3,
    random_state=2021,
    shuffle=True,
)

train_preds = np.zeros(len(train.index), )
train_probas = np.zeros(len(train.index), )

features.append("null_count")

for fold, (train_index, test_index) in enumerate(k_fold.split(new_train[features], target)):
    x_train = pd.DataFrame(new_train[features].iloc[train_index])
    y_train = target.iloc[train_index]

    x_valid = pd.DataFrame(new_train[features].iloc[test_index])
    y_valid = target.iloc[test_index]

    model = LGBMClassifier(
        random_state=2021,
        metric="auc",
        n_jobs=6,
        n_estimators=16000,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        early_stopping_rounds=200,
        verbose=0,
    )

    train_oof_preds = model.predict(x_valid)
    train_oof_probas = model.predict_proba(x_valid)[:, -1]
    train_preds[test_index] = train_oof_preds
    train_probas[test_index] = train_oof_probas

    print("-- Fold {}:".format(fold+1))
    print("{}".format(classification_report(y_valid, train_oof_preds)))

print("-- Overall:")
print("{}".format(classification_report(target, train_preds)))
print("-- ROC AUC: {}".format(roc_auc_score(target, train_probas)))

# Delete unused data, and garbage collect to save space
del model
del new_train
_ = gc.collect()

train["null_count_preds"] = train_preds
train["null_count_probas"] = train_probas
misclassified = train[(train["claim"] != train["null_count_preds"])]["null_count"].value_counts().to_dict()

# Show the confusion matrix
confusion = confusion_matrix(train["claim"], train["null_count_preds"])
ax = sns.heatmap(confusion, annot=True, fmt=",d")
_ = ax.set_title("Confusion Matrix for LGB Classifier (Null Count)", fontsize=15)
_ = ax.set_ylabel("Actual Class")
_ = ax.set_xlabel("Predicted Class")

# Plot percentage of rows impacted by feature
sns.set_style("whitegrid")
bar, ax = plt.subplots(figsize=(10, 10))
ax = sns.barplot(x=list(misclassified.keys()), y=list(misclassified.values()))
_ = ax.set_title("Number of Misclassifications by Null Values in Row (Null Count)", fontsize=15)
_ = ax.set_xlabel("Number of Null Values in Row")
_ = ax.set_ylabel("Number of Misclassified Predictions")
for p in ax.patches:
    height = p.get_height()
    ax.text(
        x=p.get_x()+(p.get_width()/2),
        y=height,
        s=round(height),
        ha="center"
    )


# As we can see, we get the best results doing so. In fields will null values, we have drastically reduced the number of misclassifications. Note however, that in rows with 0 null values, our misclassification rate hasn't changed. In fact, we probably could create a much simpler classifier with much the same characteristics by simply marking `claim` as `1` with any row that has missing values.

# ### 3.6 A Naive Classifier
# 
# Let's build a very naive classifier that just marks every `claim` as `1` if there is a null value in the field.

# In[14]:


z = dict()
for (null_count, claim_status), value_count in train[["null_count", "claim"]].value_counts().to_dict().items():
    if null_count not in z:
        z[null_count] = dict(total=0, percentage=0.0)
    z[null_count][claim_status] = value_count
    z[null_count]["total"] += value_count
for null_count in z.keys():
    z[null_count]["percentage"] = z[null_count][1] / float(z[null_count]["total"])
    
# Make "predictions" based on our null counts
train["simple_pred"] = train["null_count"].apply(lambda x: 1 if x > 0 else 0)
train["simple_probas"] = train["null_count"].apply(lambda x: z[x]["percentage"] if x in z else 0.5)

misclassified = train[(train["claim"] != train["simple_pred"])]["null_count"].value_counts().to_dict()
print("{}".format(classification_report(train["claim"], train["simple_pred"])))
print("-- ROC AUC: {}".format(roc_auc_score(target, train["simple_probas"])))

# Show the confusion matrix
confusion = confusion_matrix(train["claim"], train["simple_pred"])
ax = sns.heatmap(confusion, annot=True, fmt=",d")
_ = ax.set_title("Confusion Matrix for Simple Classifier", fontsize=15)
_ = ax.set_ylabel("Actual Class")
_ = ax.set_xlabel("Predicted Class")

# Plot percentage of rows impacted by feature
sns.set_style("whitegrid")
bar, ax = plt.subplots(figsize=(10, 10))
ax = sns.barplot(x=list(misclassified.keys()), y=list(misclassified.values()))
_ = ax.set_title("Number of Misclassifications by Null Values in Row (Simple Classifier)", fontsize=15)
_ = ax.set_xlabel("Number of Null Values in Row")
_ = ax.set_ylabel("Number of Misclassified Predictions")
for p in ax.patches:
    height = p.get_height()
    ax.text(
        x=p.get_x()+(p.get_width()/2),
        y=height,
        s=round(height),
        ha="center"
    )


# As we can see, this classifier - _by default_ - outperforms nearly every other classifier we've built so far, except for the default LGB model and the LGB model that made use of actual values plus the null count. What this is starting to suggest is that the actual values of our columns are very much not important when compared to whether or not we're seeing a null.

# ### 3.6 ROC Comparison of Null Value Approaches
# 
# Let's take a look at our ROC values as calculated by our various approaches.

# In[15]:


bar, ax = plt.subplots(figsize=(20, 10))
ax = sns.barplot(
    x=["Nulls Unmodified (LGB)", "Mean Filled Nulls (LGB)", "Zero Filled Nulls (LGB)", "Iterative Imputer Filled Nulls (LGB)", "Null Counts as Feature (LGB)", "Naive Classifier (no LGB)"],
    y=[
        float(roc_auc_score(target, train["unmodified_probas"])),
        roc_auc_score(target, train["mean_probas"]),
        roc_auc_score(target, train["zero_probas"]),
        roc_auc_score(target, train["imputer_probas"]),
        roc_auc_score(target, train["null_count_probas"]),
        roc_auc_score(target, train["simple_probas"]),        
    ]
)
_ = ax.set_title("ROC AUC Score Based on Null Value Approach", fontsize=15)
_ = ax.set_xlabel("Approach")
_ = ax.set_ylabel("ROC AUC Score")
for p in ax.patches:
    height = p.get_height()
    ax.text(
        x=p.get_x()+(p.get_width()/2),
        y=height,
        s="{:.4f}".format(height),
        ha="center"
    )


# # 4 Observations
# 
# Here are some generalized observations we can make about what we have seen within this EDA.
# 
# 1. *Nulls are the most informative feature*. The presence of a null value is by far, the most informative feature available within the dataset. While several machine learning models can handle nulls intrinsically - such as LightGBM - using a count of the number of nulls will provide sufficient signal to other approaches to provide good signal.
# 2. *Hard to impute proper values for nulls*. Our dataset likely doesn't contain enough signal to be able to impute a proper value for fields that are null. A quick exploration makes this fairly evident:
#     * There are 48,555 positive cases when there are 0 nulls present.
#     * With our best LightGBM model, we misclassify all of them: 

# In[16]:


null_count_0 = train[(train["null_count"] == 0)]
confusion = confusion_matrix(null_count_0["claim"], null_count_0["null_count_preds"])
ax = sns.heatmap(confusion, annot=True, fmt=",d")
_ = ax.set_title("Confusion Matrix when Null Count is 0 (using Best LGB Model)", fontsize=15)
_ = ax.set_ylabel("Actual Class")
_ = ax.set_xlabel("Predicted Class")


# 3. *Zero or mean filling nulls still provides signal, but not as much as nulls themselves*. We've seen somewhat decent results when we zero or mean fill null values. Why? Because the values we are filling in are likely creating outlier datapoints that the machine learning methods can use to segment or split our data. However, using counts of null values yields better results.
# 
# Any machine learning approach we take therefore, would best be served preserving where null values occur, either by using a machine learning approach that intrinsically handles nulls, or by using null counts or one-hot encoded null value fields. Additional approaches may want to explore detecting where anomalous nulls occur, and filter out those nulls with imputed values instead.
# 
# If you found this EDA useful, please consider upvoting it, or dropping a comment. I'm always happy to discuss and receive feedback!
