#!/usr/bin/env python
# coding: utf-8

# <h1 style="color: darkblue">Introduction</h1>
# 
# This notebook attempts to perform **Exploratory Data Analysis** on the Titanic dataset and eventually train a Machine Learning model on it and fine-tune the model using Randomized Search.

# In[1]:


# useful imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# for comparing classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

# for evaluating
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import classification_report


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <h3 style="color: darkblue">Knowing the Dataset</h3>
# 
# Let's first know what information the dataset contains.
# 
# The data has columns:
# - **Survival**: Whether a passenger survived or not (0 or 1)
# - **Pclass**: The socio-ecomonic class
#     - Upper: 1
#     - Middle: 2
#     - Lower: 3
# - **Sex**: Gender of the passenger (Male or Female)
# - **Age**: Age in years (Age is fractional if less than 1. If the age is estimated, it is in the form of xx.5)
# - **SibSp**: Number of siblings / spouses aboard the Titanic
# - **Parch**: Number of parents / children aboard the Titanic
# - **Ticket**: Ticket number
# - **Fare**: Passenger fare
# - **Cabin**: Cabin number
# - **Embarked**: Port of Embarkation
#     - C: Cherbourg
#     - Q: Queenstown
#     - S: Southampton

# In[3]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[4]:


train.head()


# To gain some more insight into the dataframe...

# In[5]:


train.describe()


# In[6]:


train.info()


# We observe here that some columns have missing data.
# These columns are:
# - **<span style="color: darkblue">Age</span>**
# - **<span style="color: darkblue">Cabin</span>**
# - **<span style="color: darkblue">Embarked</span>**
# 
# > Note: Rather than treating the features **Sex** and **Embarked** as `object` dtype. Let's convert them into categorical features to save some memory. Also the features **Cabin**, **Name** and **Ticket** won't be included in the final prepared training set so we can leave them as it is.

# In[7]:


attribs = ["Sex", "Embarked"]

def convert_cat(df, attrs):
    for col in attrs:
        df[col] = df[col].astype('category')
        
# Use the above function for both train and test sets
convert_cat(train, attribs)
convert_cat(test, attribs)


# Finally, let us look at the count of missing values, some columns contain.

# In[8]:


def count_na(df, col):
    print(f"Null values in {col}: ", df[col].isna().sum())
    
count_na(train, "Age")
count_na(train, "Cabin")
count_na(train, "Embarked")


# <h3 style="color: darkblue">Data Visualization</h3>
# 
# In this section, we will start visualizing the features of the dataset one by one.
# Firstly, **Univariate** feature visualization will be done, then we will move onto **Multivariate** feature visualization.
# 
# > To learn more about what **graphs** are useful for what **data-types**, check out this notebook here:
# [Statistical Data Types and Graphs (using Seaborn)](https://www.kaggle.com/code/maharshipandya/statistical-data-types-and-graphs-using-seaborn)

# In[9]:


# Setting some styles
sns.set_style("darkgrid")
sns.set_palette("viridis")


# <h3 style="color: darkblue">Univariate Analysis</h3>

# #### Analysis of Survived
# 
# A Histogram and a Pie chart will be two useful plots to analyse the `Survived` column as it is a categorical feature. Usefulness in the sense, both the plots will allow us to observe the distribution of each category in the feature. 

# In[10]:


fig, ax = plt.subplots(1, 2, figsize=(20, 7))

sns.histplot(data=train, x="Survived", stat="percent", bins=3, multiple="stack", ax=ax[0])
train["Survived"].value_counts().plot.pie(explode=[0.1, 0], autopct="%1.1f%%", shadow=True, ax=ax[1])

plt.show()


# We observe from the above plots that
# - **Only 38.4% of passengers survived the disaster**
# - **While 61.6% of passengers didn't survive!**

# #### Analysis of Sex
# 
# Similar to `Survived`, a histogram and a pie plot will provide us with distributions of categories since `Sex` is also a categorical feature.

# In[11]:


fig1, ax1 = plt.subplots(1, 2, figsize=(20, 7))

sns.histplot(data=train, x="Sex", ax=ax1[0])
train["Sex"].value_counts().plot.pie(shadow=True, autopct="%1.1f%%", explode=[0.1, 0], ax=ax1[1])

plt.show()


# - There are approximately 65% of **Male** passengers
# - Only 35.2% of passengers are **Female**

# #### Analysis of Pclass
# 
# `Pclass` is a categorical feature which is **ordinal** in nature.
# For this, **Bar charts** are useful plots.

# In[12]:


fig2, ax2 = plt.subplots(1, 2, figsize=(20, 7))

sns.countplot(data=train, x="Pclass", ax=ax2[0])
train["Pclass"].value_counts().plot.pie(shadow=True, autopct="%1.1f%%", ax=ax2[1])

plt.show()


# #### Analysis of Age
# 
# As observed, `Age` is a **Quantitative** feature. There are many plots to analyse these type of data. Histograms and Box plots are useful to know how the data is distributed.

# In[13]:


fig3, ax3 = plt.subplots(1, 2, figsize=(20, 7))

sns.histplot(data=train, x="Age", ax=ax3[0], kde=True)
sns.boxplot(data=train, x="Age", ax=ax3[1])

plt.show()


# #### Analysis on Fare
# 
# Similar to `Age`, `Fare` is also a Quantitative feature.

# In[14]:


fig4, ax4 = plt.subplots(1, 2, figsize=(20, 7))

sns.histplot(data=train, x="Fare", ax=ax4[0], kde=True)
sns.boxplot(data=train, x="Fare", ax=ax4[1])

plt.show()


# The histogram for `Fare` is quite skewed. Let us observe some facts about `Fare`.

# In[15]:


max_fare, min_fare = train["Fare"].max(), train["Fare"].min()

print(f"Number of passengers who paid ${min_fare}: ", train[train["Fare"] == min_fare].shape[0])
print(f"Number of passengers who paid ${max_fare}: ", train[train["Fare"] == max_fare].shape[0])
print(f"Fare given by maximum number of passengers: $", list(dict(train["Fare"].value_counts()).keys())[0])


# We observe that:
# 
# - **Only 3 people paid 512 dollars to be on Titanic**
# - **15 people paid no fare to be on Titanic. I wonder who they were?**
# - **Maximum people paid approximately 8 dollars**

# People who paid no fare to be on titanic:

# In[16]:


train[train["Fare"] == min_fare]


# #### Analysis on Embarked
# 
# Again like `Survived` and `Sex`, `Embarked` is also a categorical feature. So Bar plot and Pie chart is the way to go.

# In[17]:


fig5, ax5 = plt.subplots(1, 2, figsize=(20, 7))

sns.countplot(data=train, x="Embarked", ax=ax5[0])
train["Embarked"].value_counts().plot.pie(ax=ax5[1], autopct="%1.1f%%",
                                          explode=(0.1, 0, 0), shadow=True)

plt.show()


# As seen from the above plots, most passengers, approximately **72.4%**, boarded the Titanic from Southampton.

# #### Analysis on Parch and SibSp
# 
# Both of these features are Quantitative in nature but has discrete values. So Bar plots will be useful to gain insights about their structure.

# In[18]:


fig6, ax6 = plt.subplots(2, 2, figsize=(20, 10))

# SibSp
sns.countplot(data=train, x="SibSp", ax=ax6[0, 0]).set_title("Siblings and Spouses")
train["SibSp"].value_counts().plot.pie(ax=ax6[0, 1], shadow=True, title="Distribution of SibSp")

# Parch
sns.countplot(data=train, x="Parch", ax=ax6[1, 0]).set_title("Parents and Children")
train["Parch"].value_counts().plot.pie(ax=ax6[1, 1], shadow=True, title="Distribution of Parch")

plt.show()


# <h3 style="color: darkblue">Multivariate Analysis</h3>
# 
# In this section we will visualize two or more features together which comes under **Multivariate Analysis**.

# #### Analysis of Survived and Pclass
# 
# Using seaborn, we will plot a Histogram of `Pclass` having hue based on whether they survived or not.

# In[19]:


fig, ax = plt.subplots(figsize=(20, 7))

sns.histplot(data=train, x="Pclass", hue="Survived", multiple="stack", ax=ax).set_title("Classes Survival Stat")
plt.show()


# Hmm, people of the Upper class (class 1) survived more than Middle or Lower classes. Maybe Upper class people were favoured more (on the lifeboats?) over Lower and Middle classes.

# In[20]:


sur_upper = train[(train["Survived"] == 1) & (train["Pclass"] == 1)].shape[0]
sur_middle = train[(train["Survived"] == 1) & (train["Pclass"] == 2)].shape[0]
sur_lower = train[(train["Survived"] == 1) & (train["Pclass"] == 3)].shape[0]

print("Upper class survival: ", sur_upper)
print("Middle class survival: ", sur_middle)
print("Lower class survival: ", sur_lower)


# #### Analysis on Survived and Sex
# 
# Let's plot a Pie chart to show the distribution of which sex/gender survived more.

# In[21]:


df_m = train[(train["Survived"] == 1) & (train["Sex"] == "male")]
df_f = train[(train["Survived"] == 1) & (train["Sex"] == "female")]

df_sur = pd.concat([df_m, df_f])
df_sur["Sex"].value_counts().plot.pie(explode=[0, 0.1], shadow=True, autopct="%1.1f%%")
plt.show()


# From the above Pie plot, we observe that women survived the Titanic disaster more than the men. Only 32% of men survived, while almost 68% women survived.

# #### Analysis on Survived and Age
# 
# Since `Age` is a continuous feature, we plot a histogram with hue based on `Survived`.

# In[22]:


fig7, ax7 = plt.subplots(figsize=(20, 8))

sns.histplot(data=train, x="Age", hue="Survived", multiple="stack", kde=True)
plt.show()


# We see that in the age range 0-10, the passengers who survived are greater than non survivors. Assuming that the kids (or younger passengers) were favoured more for the lifeboats, first?

# In[23]:


print("Kids survived in age ranges: ")
print("Age 0-4:", train[(train["Age"] < 4) & (train["Survived"] == 1)].shape[0])
print("Age 4-7:", train[(train["Age"] >= 4) & (train["Age"] < 7) & (train["Survived"] == 1)].shape[0])
print("Age 7-10:", train[(train["Age"] >= 7) & (train["Age"] <= 10) & (train["Survived"] == 1)].shape[0])

print("\nKids NOT survived in age ranges: ")
print("Age 0-4:", train[(train["Age"] < 4) & (train["Survived"] == 0)].shape[0])
print("Age 4-7:", train[(train["Age"] >= 4) & (train["Age"] < 7) & (train["Survived"] == 0)].shape[0])
print("Age 7-10:", train[(train["Age"] >= 7) & (train["Age"] <= 10) & (train["Survived"] == 0)].shape[0])


# <h3 style="color: darkblue">Analyzing Correlations</h3>
# 
# We visualized the data to gain some insights on the data. Now its time to analyse correlations between every feature, using a correlation matrix.

# In[24]:


# The 2D correlation matrix
corr = train.corr()


# In[25]:


# Plotting the heatmap of corr

fig, ax = plt.subplots(figsize=(20, 7))
dataplot = sns.heatmap(data=corr, annot=True, ax=ax)
plt.show()


# We observe here that
# 
# - `Pclass` has considerable amount of negative correlation with `Survived`. This is true as seen above from visualizing the data i.e. Upper class people survived more than the Middle or Lower class people.
# - `Fare` has a considerable amount of positive correlation with `Survived`. As fare increases the chances of survival kinda increases.

# <h3 style="color: darkblue">Preparing the Data</h3>
# 
# Its now time to prepare the data for machine learning algorithms to train on. There are a few things we need to do.
# 
# - **Fill in the missing values (using an imputer)**
# - **Handle categorical and numerical variables (One hot encoding)**
# - **Use custom transformers (Using scikit-learn API)**
# 
# For this, we need to create different pipelines for numerical and categorical attributes.

# In[26]:


df = train.drop("Survived", axis=1)
labels = train["Survived"].copy()

print(labels)


# Using scikit-learn API, we will create some custom transformers to ease our task when creating a pipeline.
# 
# There will be 2 custom transformers in this case:
# 
# - **DataFrameSelector**: To select important features only
# - **AttribAdder**: A kind of feature engineering to create a new feature `famSize` from `Parch` and `SibSp`
# 
# > Note: This notebook does not dwell into the topics of Feature Engineering to keep things a bit simple

# In[27]:


# Custom transformer to sample only the useful attributes
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attrs):
        self.attrs = attrs
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attrs].values


# We need to drop the usless or missing features. Features like
# 
# - **Ticket**
# - **PassengerId**
# - **Cabin**
# - **Name**
# 
# fall under these.

# We create two arrays for selecting important **numerical** and **categorical** attributes.

# In[28]:


num_attrs = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
cat_attrs = ["Sex", "Embarked"]


# In[29]:


# Custom transformer to add Parch and SibSp as FamSize
sibsp_ix, parch_ix = 2, 3

class AttribAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        famSize = X[:, sibsp_ix] + X[:, parch_ix] + 1
        return np.c_[X, famSize]


# ### Numerical Pipeline
# 
# We will perform 3 tasks in this pipeline:
# 
# - **Select numerical columns**
# - **Imputer for missing values**
# - **Attributes adder**

# In[30]:


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ("selector", DataFrameSelector(num_attrs)),
    ("imputer", SimpleImputer(strategy="median")),
    ("attrib_adder", AttribAdder())
])


# ### Categorical Pipeline
# 
# We will perform 3 tasks in this pipeline:
# 
# - **Select categorical features**
# - **Impute missing values using `most frequent`**
# - **One hot encoding to convert the categories to one hot vectors**
# 
# > Know more about [One Hot Encoding](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)

# In[31]:


from sklearn.preprocessing import OneHotEncoder

cat_pipeline = Pipeline([
    ("selector", DataFrameSelector(cat_attrs)),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("label_binarize", OneHotEncoder(sparse=False))
])


# Let us combine these two pipelines, using **FeatureUnion** and fit it on the training dataframe.

# In[32]:


from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion([
    ("num", num_pipeline),
    ("cat", cat_pipeline)
])

# Use fit transfrom on full pipeline
titanic_prepared = full_pipeline.fit_transform(df)
titanic_prepared_df = pd.DataFrame(titanic_prepared, columns=[
    "Pclass", "Age", "SibSp", "Parch", "Fare", "FamSize", "Female", "Male", "C", "Q", "S"
])
titanic_prepared_df


# <h3 style="color: darkblue">Classification</h3>
# 
# Data preparation is done, now its time to run ML algorithms on the preprocessed data. We will write a function to compare few models, by a technique called **[Cross Validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))**.

# In[33]:


estimators = [SVC(), RandomForestClassifier(), KNeighborsClassifier(), SGDClassifier()]
fig, ax = plt.subplots(len(estimators), 2, figsize=(20, 20))

def run_compare(estis, cv=3):
    for esti_ix, esti in enumerate(estis):
        esti_preds = cross_val_predict(esti, titanic_prepared, labels, cv=cv)
        
        esti_pr_disp = PrecisionRecallDisplay.from_predictions(labels,
                                                               esti_preds, ax=ax[esti_ix][0], name=esti)
        esti_roc_disp = RocCurveDisplay.from_predictions(labels,
                                                         esti_preds, ax=ax[esti_ix][1], name=esti)
        
        print(f"\nClassification Report for {esti}:")
        print(classification_report(labels, esti_preds))

# Run
run_compare(estimators, cv=5)
plt.show()


# We observe that **RandomForestClassifier** performs the best out of all. So we will go forward with that.

# <h3 style="color: darkblue">Fine-tuning Random Forest Classifier</h3>
# 
# We will use **[Randomized Searching](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)** to find good hyper-parameter values for our classifier.

# In[34]:


# Parameters of random forest classifier
n_estimators = np.linspace(50, 300, int((300 - 50) / 20), dtype=int)
max_depth = [1, 5, 10, 50, 100, 200, 300]
min_samples_split = [2, 4, 6]
max_features = ["sqrt", "log2"]
bootstrap = [True, False]

distributions = {
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "max_features": max_features,
    "bootstrap": bootstrap
}


# In[35]:


# Randomised search cv
from sklearn.model_selection import RandomizedSearchCV

rfc = RandomForestClassifier()
random_search_cv = RandomizedSearchCV(
    rfc,
    param_distributions=distributions,
    n_iter=30,
    cv=5,
    n_jobs=4
)

search = random_search_cv.fit(titanic_prepared, labels)


# The results of this Randomized Search are stored in a dictionary named `cv_results_`. Let us print these results just to get an idea of what parameters were tested by our Randomized Search.

# In[36]:


cvres = search.cv_results_

for score, params, rank in zip(cvres["mean_test_score"], cvres["params"], cvres["rank_test_score"]):
    print(score, params, rank)


# Also, the best estimator out of these tested ones is stored in a variable called `best_estimator_`. We can use this estimator as our fine-tuned model.

# In[37]:


rfc_finetuned = search.best_estimator_
best_preds = cross_val_predict(rfc_finetuned, titanic_prepared, labels, cv=5)

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
PrecisionRecallDisplay.from_predictions(labels, best_preds, ax=ax[0])
RocCurveDisplay.from_predictions(labels, best_preds, ax=ax[1])

print(classification_report(labels, best_preds, digits=5))
plt.show()


# As we can see, the fine-tuned Random Forest Classifier has:
# 
# - **An average precision of 82%**
# - **An average recall of about 81%**
# 
# which is totally not bad than the previous bare-bones classifier!

# **<span style="font-size: 20px; color: darkblue">If this notebook helped you a slightest bit, do consider upvoting it and leaving a comment below! Feel free to extend this notebook with more knowledge. Thank you! ❤️</span>**
