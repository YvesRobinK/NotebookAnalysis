#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:rgba(128, 0, 128, 0.6);border-radius:5px;display:fill">
#     <h1 style="text-align: center;padding: 12px 0px 12px 0px;">🚢 Titanic: Logistic Regression - 🕵️‍ Series</h1>
# </div>
# <img src="https://www.seekpng.com/png/full/200-2002376_predicting-titanic-survivors-old-photos-of-the-titanic.png" alt="Titanic" width="600"/>
# 
# 
# ## Introduction
# 
# This is the first in a series of machine learning tutorials using Kaggle.  The goal is to introduce machine learning, with a bent on Kaggle.  The intention is for these tutorials to be comprehensive, and to provide additional references for the avid learner.  Finally, I hope to be consistent in my style, naming, and conventions.  There will be several tutorials that all appear consistent.
# 
# 
# # The Case of the Unsinkable Ship
# 
# The [Sinking of the Titanic](https://en.wikipedia.org/wiki/Sinking_of_the_Titanic) was a great human tragedy.  Your agency has been called to investigate the [distribution](https://youtu.be/oI3hZJqXJuc) of the survivors.
# 
# # Machine Learning Introduction
# 
# Titanic is a [Classification Problem](https://www.analyticsvidhya.com/blog/2021/09/a-complete-guide-to-understand-classification-in-machine-learning/).  The evaluation measure is [accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy).
# 
# ## Future Articles in the 🕵️‍ Series
# 
# - [Logistic Regression](https://www.kaggle.com/mmellinger66/titanic-ml-logistic-regression) - Case of the Unsinkable Ship
# - Linear Regression - 
# - Random Forest - 
# - XGBoost - 
# - K-Means Clustering - Case of the Missing Penguin
# - LDA/QDA
# 
# 
# ## Data Science/Machine Learning Dictionary
# 
# Terms you'll see often:
# 
# - Exploratory Data Analysis - (EDA)
# - Cross Validation
# - Feature Engineering
# - Confusion Matrix
# - OOF (Out of Fold)
# - Blending
# - Stacking
# 
# ### Cross Validation
# 
# Cross validation is usually not covered at the beginning of most machine language tutorials, but in practise, and on Kaggle in particular, it's often critical for improving ML models.
# 
# - [Kaggle's 30 Days Of ML (Competition Part-1): Cross Validation & First Submission on Kaggle](https://youtu.be/t5fhRP62YdE)
# 
# ## A Little Theory
# 
# Logistic Regression
# 
# ## More on Logistic Regression
# 
# - [Logistic Regression for Machine Learning - Brownlee](https://machinelearningmastery.com/logistic-regression-for-machine-learning/)
# - [Conceptual Understanding of Logistic Regression for Data Science Beginners - ](https://www.analyticsvidhya.com/blog/2021/08/conceptual-understanding-of-logistic-regression-for-data-science-beginners/)
# - [StatQuest: Logistic Regression - video](https://www.youtube.com/watch?v=yIYKR4sgzI8)
#  - [Logistic Regression Details Pt1: Coefficients](https://www.youtube.com/watch?v=vN5cNN2-HWE)
# - [Introduction to Statistical Learning - Chapter 4](http://statlearning.com)
# 
# ## Kaggle Resources Used to Develop this Notebook
# 
# - [Tidy TitaRnic](https://www.kaggle.com/headsortails/tidy-titarnic) - [@headsortails](https://www.kaggle.com/headsortails)
# - [Pytanic](https://www.kaggle.com/headsortails/pytanic) - [@headsortails](https://www.kaggle.com/headsortails)
# - [Song Popularity EDA - Live Coding Fun](https://www.kaggle.com/headsortails/song-popularity-eda-live-coding-fun) - [@headsortails](https://www.kaggle.com/headsortails)
# - [🎢 Introduction to Exploratory Data Analysis](https://www.kaggle.com/robikscube/introduction-to-exploratory-data-analysis/) - [@robikscube](https://www.kaggle.com/robikscube)
# - [🎓 Student Writing Competition [Twitch Stream]](https://www.kaggle.com/robikscube/student-writing-competition-twitch-stream) - [@robikscube](https://www.kaggle.com/robikscube)
# - [Titanic: logistic regression with python](https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python)
# - [Titanic Model with 90% accuracy](https://www.kaggle.com/vinothan/titanic-model-with-90-accuracy)
# - [Titanic (0.83253) - Comparison 20 popular models](https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models)
# - [A Statistical Analysis & ML workflow of Titanic](https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic)
# - [A Journey through Titanic](https://www.kaggle.com/omarelgabry/a-journey-through-titanic)
# - [XGBoost model with minimalistic features](https://www.kaggle.com/aashita/xgboost-model-with-minimalistic-features) - Top?
# - [Titanic Project Example](https://www.kaggle.com/kenjee/titanic-project-example)
# - [Titanic Prediction | 90% Accuracy](https://www.kaggle.com/aminizahra/titanic-prediction-90-accuracy)
# - [Titanic WCG+XGBoost [0.84688]](https://www.kaggle.com/cdeotte/titanic-wcg-xgboost-0-84688) - [@cdeotte](https://www.kaggle.com/cdeotte/)
# - [Titantic Mega Model - [0.84210]](https://www.kaggle.com/cdeotte/titantic-mega-model-0-84210) - [@cdeotte](https://www.kaggle.com/cdeotte/)

# # Understanding the Titanic Data
# 
# ## Target - What we want to predict
# 
# For the Titantic dataset the target is:`Survived`
# 
# Binary value: 0=Didn't Survive or 1=Survived
# 
# ## Features
# 
# ### Categorical
# 
# - `Pclass` - Ticket class (1st,2nd,3rd)
# - `Sex` - (male, female)
# - `Embarked` - C = Cherbourg, Q = Queenstown, S = Southampton
# - `Cabin` - Cabin number
# 
# ### Numerical
# 
# - `Age` - Passenger's age
# - `SibSp` - # of siblings / spouses aboard the Titanic
# - `Parch` - # of parents / children aboard the Titanic
# - `Fare` - What the passenger paid for a ticket
# 
# ### Other
# 
# - `Name` - Full name
# - `Ticket` - Ticket number
# 
# 
# 
# ## Evaluation Metric
# 
# $Accuracy = \frac{True Positives (TP) + True Negatives (TN)}{True Positives (TP) + True Negatives (TN) + False Positives(FP) + False Negatives(FN)}$
# 
# - https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification
# 
# Percentage of passengers you correctly predict.  In other words, accurately predict who survived and who did not.
# 
# - https://developers.google.com/machine-learning/crash-course/classification/accuracy
# 
# # Preprocessing
# 
# ## Missing Values
# 
# ## Feature Engineering
# 
# # Cross Validation
# 
# This would be covered a little later in most books.  However, it's critical to developing good ML models.
# 
# KFold/KFoldStratification
# 
# I'm not going to explain the difference now.  However, we are going to use KFoldStratification for now.

# <div style="background-color:rgba(128, 0, 128, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Import Libraries</h1>
# </div>

# In[1]:


import os
import time
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


# Visualization Libraries
import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

from itertools import cycle

plt.style.use("ggplot")  # ggplot, fivethirtyeight


# <div style="background-color:rgba(128, 0, 128, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Configuration</h1>
# </div>

# In[2]:


# Change for every project
data_dir = Path("../input/titanic")


# ### The target/dependent variable in the dataset

# In[3]:


# Did the passenger survive?
# 0 = No, 1 = Yes
TARGET = "Survived"


# <div style="background-color:rgba(128, 0, 128, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Theme</h1>
# </div>

# In[4]:


color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
color_pal


# In[5]:


color_cycle


# In[6]:


mpl.rcParams['font.size'] = 16

theme_colors = ["#ffd670","#70d6ff","#ff4d6d","#8338ec","#90cf8e"]
theme_palette = sns.set_palette(sns.color_palette(theme_colors))
sns.palplot(sns.color_palette(theme_colors), size=1.1)
plt.tick_params(axis='both', labelsize=0, length = 0)


# In[7]:


my_colors = ["#CDFC74", "#F3EA56", "#EBB43D", "#DF7D27", "#D14417", "#B80A0A", "#9C0042"]
sns.palplot(sns.color_palette(my_colors), size=0.9)


# <div style="background-color:rgba(128, 0, 128, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Load Train/Test Data</h1>
# </div>
# 
# ## Load the following files
# 
#  - train.csv - Data used to build our machine learning model
#  - test.csv - Data used to build our machine learning model. Does not contain the `Suvived` target variable
#  - gender_submission.csv - A file in the proper format to submit test predictions

# In[8]:


train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")
sample_submission = pd.read_csv(data_dir / "gender_submission.csv")

print(f"train data: Rows={train.shape[0]}, Columns={train.shape[1]}")
print(f"test data : Rows={test.shape[0]}, Columns={test.shape[1]}")


# <div style="background-color:rgba(128, 0, 128, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Explore the Data</h1>
# </div>
# 
# Exploratory Data Analysis (EDA) is another complete project. I'll do  some basic exploration here but others have written extensive EDA's
# 
# - [10 Simple hacks to speed up your Data Analysis](https://www.kaggle.com/parulpandey/10-simple-hacks-to-speed-up-your-data-analysis)
# - [Simple Matplotlib & Visualization Tips](https://www.kaggle.com/subinium/simple-matplotlib-visualization-tips)
# - [Pytanic](https://www.kaggle.com/headsortails/pytanic) - [@headsortails](https://www.kaggle.com/headsortails)
# 

# In[9]:


train.columns


# ## Feature Analysis

# In[10]:


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))

# train["Sex"].value_counts(ascending=True).plot(kind="bar", figsize=(8, 5), color=color_pal[1], ax=axs[0])
train["Sex"].value_counts(ascending=True).plot(kind="bar", color=color_pal[1], ax=axs[0])

axs[0].set_title("Survival by Sex", fontsize=16)
axs[0].bar_label(axs[0].containers[0], label_type="edge")

# train["Pclass"].value_counts(ascending=False).plot(kind="bar", figsize=(5, 3), ax=axs[1])
train["Pclass"].value_counts(ascending=False).plot(kind="bar", color=color_pal[2], ax=axs[1])
axs[1].set_title("Survival by Passenger Class", fontsize=16)
axs[1].bar_label(axs[1].containers[0], label_type="edge")

train["Embarked"].value_counts(ascending=False).plot(kind="bar", color=color_pal[3], ax=axs[2])
axs[2].set_title("Survival by Embarked", fontsize=16)
axs[2].bar_label(axs[2].containers[0], label_type="edge")


plt.show()


# # Numerical Variables

# We can see that more women survived ...

# In[11]:


plt.figure(figsize=(15, 12))
sns.heatmap(train.corr(), annot=True, cmap="PuBuGn")
plt.show()


# ## Target Analysis

# <div style="background-color:rgba(128, 0, 128, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Feature Engineering</h1>
# </div>
# 
# - [Titanic - Advanced Feature Engineering Tutorial](https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial)
# - [Titanic Survival Predictions (Beginner)](https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner)
# - [Exploring Survival on the Titanic](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic)

# In[12]:


FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]


# In[13]:


train[FEATURES].head()


# # Extract Target and Drop Unused Columns

# In[14]:


y = train[TARGET]

train = train.drop(columns=["PassengerId"], axis=1).copy()
test = test.drop(columns=["PassengerId"], axis=1).copy()


# <div style="background-color:rgba(128, 0, 128, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Analyze</h1>
# </div>

# In[15]:


vars=[
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
#         "Embarked",
    "Sex"
]


train[vars].head()


# # Pairplot
# 
# - https://seaborn.pydata.org/generated/seaborn.pairplot.html

# In[16]:


get_ipython().run_cell_magic('time', '', 'plt.figure(figsize=(15,8))\n\n\nsns.pairplot(\n    train,\n    vars=[\n        "Pclass",\n        "Age",\n        "SibSp",\n        "Parch",\n#         "Embarked",\n#         "Sex"\n    ],\n    hue="Survived",\n)\nplt.show()\n')


# <div style="background-color:rgba(128, 0, 128, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Preprocessing</h1>
# </div>
# 
# ## Missing Values
# 
# We have 891 rows of training data. Age is the only feature, that we are using, with missing data.
# 
# Note, handling missing data is an entire subject that should be studied in detail.  Kaggle offers a [course](https://www.kaggle.com/learn/data-cleaning)
# 
# - [sklearn.impute.SimpleImputer](https://scikit-learn.org/stable/modules/impute.html)
# - https://scikit-learn.org/stable/modules/impute.html
# 
# - [A Guide to Handling Missing values in Python](https://www.kaggle.com/parulpandey/a-guide-to-handling-missing-values-in-python)

# In[17]:


missing_vals = train.isna().sum()
print(missing_vals[missing_vals > 0])


# In[18]:


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 4))

# plt.figure(figsize=(6,4))
fig.set_size_inches(4, 4)


ax = train["Age"].hist(
    bins=10,
    density=True,
    stacked=True,
    color="blue",
    alpha=0.6,
    ax=axs[0],
    figsize=(6,4)
    
)
train["Age"].plot(
    kind="density",
    color="red",
    title="Age Distribution",
    ax=axs[0]
)

# axs[0].xset(xlabel="Age")
plt.xlim(-20, 100)

# plt.figure(figsize=(12,4))
fig.set_size_inches(11.7, 4.27)
sns.boxplot(data=train, 
            x="Age",
            color=theme_colors[4],
            ax=axs[1],
#             figsize=(12,4)
           )
axs[1].set_title('Age Boxplot')
plt.show()

# train["Pclass"].value_counts(ascending=False).plot(kind="bar", color=color_pal[2], ax=axs[1])
# axs[1].set_title("Survival by Passenger Class", fontsize=16)
# axs[1].bar_label(axs[1].containers[0], label_type="edge")




# In[19]:


plt.figure(figsize=(15,4))
ax = sns.boxplot(data=train, x="Age",color=theme_colors[4])
ax.set_title('Age Boxplot')
plt.show()


# ### Embarked

# In[20]:


print(
    "Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):"
)
# print(train["Embarked"].value_counts())
sns.countplot(x="Embarked", data=train)
plt.show()


# In[21]:


sns.barplot(
    x="Embarked",
    y="Survived",
    data=train,
)
plt.show()


# In[22]:


ax = sns.scatterplot(
    x="Age",
    hue="Survived",
    y="Pclass",
    data=train,
)
ax.set_title("Age vs Pclass vs Survived")
plt.show()


# In[23]:


ax = sns.scatterplot(
    x="Pclass",
    hue="Survived",
    y="Survived",
    data=train,
)
ax.set_title("Pclass vs Survived")
plt.show()


# ## Impute Age
# 
# For educational purposes, we're going to discuss how to do this manually and how to use the sklearn library. Both give similar results.

# In[24]:


n = train["Age"].isna().sum()
print(f"Number missing: {n}")


# ### Manual Imputation of Age

# In[25]:


# train_df["Age"].fillna(train_df["Age"].mean(), inplace = True)
m = train["Age"].mean()
print(f"Mean age of person on the Titanic: {m:0.2f}")


# In[26]:


train["Age"].fillna(train["Age"].median(skipna=True), inplace=True)
train["Embarked"].fillna(train["Embarked"].value_counts().idxmax(), inplace=True)


# ### Use SimpleImputer Function for Age
# 
# Leaving the SimpleImputer code uncommented.  It should do nothing since we filled in the values above.

# In[27]:


impute_mean = SimpleImputer(missing_values=np.nan, strategy="mean", verbose=1)
m = impute_mean.fit_transform(train[["Age"]])
# mt = impute_mean.transform(test[["Age"]])

train["Age"] = impute_mean.fit_transform(train[["Age"]])
test["Age"] = impute_mean.transform(test[["Age"]])


# ### At this point we no longer have missing values

# In[28]:


# Verify no more Age values with na
n = train["Age"].isna().sum()
print(f"Number missing: {n}")


# # Encoding Categorical Features
# 
# Need to convert categorical features into numerical features.
# 
# Several ways:
# - One-hot Encode
# - Label Encode

# # One-hot Encode Categorical Features
# 
# Computers understand numbers. Embarked location must be turned into numerical values, for example.  One option would be to change S=1, C=2, and Q=3.  This would be LabelEncoding.
# 
# Another option is to encode the values as vectors.
# ```
# S = [1 0 0]
# C = [0 1 0]
# Q = [0 0 1]
# ```
# 
# 
# - One-hot encoding explanation
# - [pandas.get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)

# In[29]:


# X = pd.get_dummies(X[FEATURES], drop_first=True)
# X_test = pd.get_dummies(test[FEATURES], drop_first=True)

# X.head()


# ## Label Encoding Features
# 
# This method simply maps names to numbers.
# 
# ### Encode `Sex`
# 
# ```
# male = 0
# female = 1
# ```

# In[30]:


replacement_dict = {"female": 0, "male": 1}

train["Sex"] = train["Sex"].map(replacement_dict)
test["Sex"] = test["Sex"].map(replacement_dict)


# In[31]:


from sklearn.preprocessing import LabelEncoder


def label_encoder(train, test, columns):
    for col in columns:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        train[col] = LabelEncoder().fit_transform(train[col])
        test[col] = LabelEncoder().fit_transform(test[col])
    return train, test


# ### Encode `Embarked`
# 
# First compare what the **drop_first=True** option does.  Some machine learning models require this option while others do not.  Logitistic regression requires us to drop the value.

# In[32]:


#train, test = label_encoder(train, test, ["Embarked"])
# X_test = pd.get_dummies(test[FEATURES], drop_first=True)
train = pd.get_dummies(data=train, columns=["Embarked"], drop_first=True)
test = pd.get_dummies(data=test, columns=["Embarked"], drop_first=True)
train.head()


# In[33]:


test.head(2)


# - [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

# # Scale the Data
# 
# Linear models usually need to have their features scaled.
# 
# $z = \frac{x_i - \bar{x}}{\sigma}$
# 
# [Scaling](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling) does not change the shape of the distribution, as would happened if we [normalized](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-normalization) the data.

# # Heatmap
# 
# - https://seaborn.pydata.org/generated/seaborn.heatmap.html?highlight=heatmap

# In[34]:


plt.figure(figsize=(12, 12))

corr = train.corr()
corr.style.background_gradient(cmap="coolwarm")
# plt.show()


# In[35]:


plt.figure(figsize=(12, 8))
sns.heatmap(train.corr(), annot=True, fmt=".2f", cmap="RdBu_r")  # PuBuGn RdBu_r
shape = np.triu(train.corr())
# sns.heatmap(train.corr(), annot=True, fmt=".2f", mask=shape, cmap="RdBu_r")
plt.tight_layout()
plt.show()


# <div style="background-color:rgba(128, 0, 128, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Train Model with Train/Test Split</h1>
# </div>

# In[36]:


FEATURES.remove("Embarked")
FEATURES.extend(["Embarked_Q", "Embarked_S"])

FEATURES


# In[37]:


X = train[FEATURES].copy()


# In[38]:


# FEATURES = cat_features + num_features

y = train[TARGET]
X = train[FEATURES].copy()

X_test = test[FEATURES].copy()


# In[39]:


X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


# ## Train Model

# In[40]:


X_train.head()


# In[41]:


X_test.head()


# In[42]:


# liblinear is the default
model = LogisticRegression(
    solver="liblinear",
    #                             penalty="l1",
    random_state=42,
)

model.fit(X_train, y_train)


# ## Use Logistic Regression with L2 Regularization

# In[43]:


# model = RidgeClassifier(alpha=0.5)
# model.fit(X_train, y_train)


# <div style="background-color:rgba(128, 0, 128, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Score</h1>
# </div>
# 
# We get a score by evaluating our model on the validation data.

# In[44]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def show_scores(gt, yhat):
    accuracy = accuracy_score(gt, yhat)
    precision = precision_score(gt, yhat)
    recall = recall_score(gt, yhat)
    f1 = f1_score(gt, yhat)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"f1: {f1:.4f}")


# In[45]:


preds_valid = model.predict(X_valid)
score = accuracy_score(preds_valid, y_valid)
print(f"Accuracy: {score:0.4f}")
show_scores(y_valid, preds_valid)


# ## Confusion Matrix

# In[46]:


cmatrix = confusion_matrix(y_valid, preds_valid)
cmatrix


# ### A heatmap of the Confusion Matrix looks a little nicer

# In[47]:


plt.figure(figsize=(6, 6))

sns.heatmap(
    cmatrix,
    annot=True,
    fmt="2.0f",
)
# sns.set(font_scale=4)  # font size 4

# plt.show()


# ## Classification Report

# In[48]:


print(classification_report(y_valid, preds_valid))


# <div style="background-color:rgba(128, 0, 128, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Model Inference</h1>
# </div>
# 
# Now that we have trained the model, we can make inferences on data we have not seen before.

# In[49]:


model


# In[50]:


preds_test = model.predict(X_test)
preds_test


# <div style="background-color:rgba(128, 0, 128, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Submission File</h1>
# </div>
# 
# The sample file and our data is in the same row order.  This allows us to simply assign our prediction to the target column (`Survived`) in the sample submission.

# In[51]:


sample_submission["Survived"] = preds_test
sample_submission.to_csv("submission.csv", index=False)
sample_submission


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Extra Credit</h1>
# </div>
# 
# You have completed your first model.  The next section is optional. In it, we will train the model using cross validation.

# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Train Model with Cross Validation</h1>
# </div>
# 
# Four out five folds will be used for training. The fifth will be used for validation
# 
# Each fold will have a turn at being the validation fold
# 
# After each time through the loop
# 
# ## Concepts
# 
# - Stratification
#    - [Stratified sampling](https://en.wikipedia.org/wiki/Stratified_samplinghttps://en.wikipedia.org/wiki/Stratified_sampling)
#    - [Stratified sampling in Machine Learning](https://medium.com/analytics-vidhya/stratified-sampling-in-machine-learning-f5112b5b9cfehttps://medium.com/analytics-vidhya/stratified-sampling-in-machine-learning-f5112b5b9cfe)
#    - [Stratified Random Sampling Using Python and Pandas](https://towardsdatascience.com/stratified-random-sampling-using-python-and-pandas-1c84f0362ebchttps://towardsdatascience.com/stratified-random-sampling-using-python-and-pandas-1c84f0362ebc)
# - [KFold]
# - OOF - Out of Fold

# In[52]:


NFOLDS = 5

final_test_predictions = []
final_valid_predictions = {}
scores = []


kf = StratifiedKFold(n_splits=NFOLDS, random_state=42, shuffle=True)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X=X, y=y)):
    print(10 * "=", f"Fold={fold+1}", 10 * "=")
    start_time = time.time()

    x_train = X.loc[train_idx, :]
    x_valid = X.loc[valid_idx, :]  # Validation Features

    y_train = y[train_idx]
    y_valid = y[valid_idx]  # Validation Target

    model = LogisticRegression(C=0.12, solver="liblinear")
    model.fit(x_train, y_train)
    #     preds_valid = model.predict_proba(x_valid)[:,1]
    preds_valid = model.predict(x_valid)

    # Predictions for OOF
    print("--- Predicting OOF ---")
    final_valid_predictions.update(dict(zip(valid_idx, preds_valid)))

    accuracy = accuracy_score(y_valid, preds_valid)
    scores.append(accuracy)

    run_time = time.time() - start_time

    # Predictions for Test Data
    print("--- Predicting Test Data ---")
    test_preds = model.predict(X_test)

    final_test_predictions.append(test_preds)
    print(f"Fold={fold+1}, Accuracy: {accuracy:.8f}, Run Time: {run_time:.2f}\n")


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Scores</h1>
# </div>
# 
# CV, or Cross Validation, Score.
# 
# We average the means and the standard deviations.
# 
# The Adjusted Score is the average of the means minus the average of standard deviation. Do this to attempt to get one number to evaluate the score when comparing different models.

# In[53]:


print(
    f"Scores -> Adjusted: {np.mean(scores) - np.std(scores):.8f} , mean: {np.mean(scores):.8f}, std: {np.std(scores):.8f}"
)


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Save OOF Predictions</h1>
# </div>
# 
# This is unused for this example but needed later for [Blending](https://towardsdatascience.com/ensemble-learning-stacking-blending-voting-b37737c4f483).
# 
# **General idea**: The values will be use to create new features in a blended model.
# 
# - [Stacking and Blending — An Intuitive ExplanationStacking and Blending — An Intuitive Explanation](https://medium.com/@stevenyu530_73989/stacking-and-blending-intuitive-explanation-of-advanced-ensemble-methods-46b295da413chttps://medium.com/@stevenyu530_73989/stacking-and-blending-intuitive-explanation-of-advanced-ensemble-methods-46b295da413c)

# In[54]:


final_valid_predictions = pd.DataFrame.from_dict(
    final_valid_predictions, orient="index"
).reset_index()
final_valid_predictions.columns = ["id", "pred_1"]
final_valid_predictions.to_csv("train_pred_1.csv", index=False)


# <div style="background-color:rgba(255, 215, 0, 0.6);border-radius:5px;display:fill"><h1 style="text-align: center;padding: 12px 0px 12px 0px;">Submission File</h1>
# </div>
# 
# The sample file and our data is in the same row order.  This allows us to simply assign our prediction to the target column (`Survived`) in the sample submission.

# In[55]:


from scipy.stats import mode

mode_result = mode(np.column_stack(final_test_predictions), axis=1)
# m = np.mean(np.column_stack(final_test_predictions), axis=1)
m = mode_result[0].flatten()
m


# In[56]:


sample_submission["Survived"] = m
sample_submission.to_csv("submission_cv.csv", index=False)
sample_submission

