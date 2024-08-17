#!/usr/bin/env python
# coding: utf-8

# <div style="width:100%;text-align: center;"><img align = middle src="https://cdn.wallpapersafari.com/85/81/klWnN6.jpg" style="height:500px"></div>

# # <div class = "alert alert-info"><strong>Dataset Fields</strong></div>
# - **PassengerId**: Unique Id for each passenger
# - **Survived**: Binary value for survival (0 = No, 1 = Yes)
# - **Pclass**: Ticket class for each passenger (1 = 1<sup>st</sup> Class, 2 = 2<sup>nd</sup> Class, 3 = 3<sup>rd</sup> Class)
# - **Sex**: Gender of each passenger
# - **Age**: Age of each passenger in years
# - **SibSp**: Number of siblings or spouses aboard the Titanic
# - **Parch**: Number of parents or children aboard the Titanic
# - **Ticket**: Ticket number for the passenger
# - **Fare**: Price of the ticker
# - **Cabin**: Cabin number of the passenger
# - **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

# # <div class = "alert alert-info" style = "margin:0px;"><strong>Installing the Necessary Libraries</strong></div>

# In[1]:


get_ipython().system('pip install pywaffle')


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Importing the Necessary Libraries</strong></div>

# In[2]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import graphviz
import missingno as msno
from pywaffle import Waffle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')


# <h3>Setting the color scheme</h3>

# In[3]:


custom_colors = ['#0B559F', '#88BEDC', '#BAD6EA']
custom_palette = sns.set_palette(sns.color_palette(custom_colors))
sns.palplot(sns.color_palette(custom_colors), size = 1)
plt.tick_params(axis = 'both', labelsize = 0, length = 0)


# <h3>Looking at the input files in the directory</h3>

# In[4]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Dataset Information</strong></div>

# <h3>Reading the data</h3>

# In[5]:


df = pd.read_csv('/kaggle/input/titanic/train.csv')
df


# <h3>Getting information about our dataset</h3>

# In[6]:


df.info()


# <h3>Looking at the statistical summary of our data</h3>

# In[7]:


df.describe()


# <h3>Total missing values in the dataset</h3>

# In[8]:


print("Count of the missing values")
print(30 * "-")
print(df.isna().sum())
print(30 * "-")
print("Total missing values are:", df.isna().sum().sum())
print(30 * "-")


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Exploratory Data Analysis (EDA)</strong></div>

# In[9]:


plt.figure(figsize = (15, 10))
sns.heatmap(df.isna(), yticklabels = False, cbar = False, cmap = 'Blues')
plt.title("Visualizing the Missing Data", fontsize = 20)
plt.xticks(rotation = 35, fontsize = 15)
plt.show()


# In[10]:


msno.bar(df, color = (0, 0.4, 0.8), sort = "ascending", figsize = (15, 10))
plt.show()


# In[11]:


print("Missing Data in the Cabin column =", (df['Cabin'].isna().sum() / len(df['Cabin']) * 100), "%")


# Due to a high number of missing data in the `Cabin` column, it would be better to drop the entire column, rather than try and fill all the values. Since `Age` and `Embarked` have a relatively lower number of missing values it is possible to fill them.

# In[12]:


plt.figure(figsize = (15, 10))
ax = sns.countplot(x = 'Survived', data = df)
plt.title('Survival Rates', fontsize = 20)
plt.xlabel('Survived', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
for p in ax.patches:
        ax.annotate('{:.0f} = {:.2f}%'.format(p.get_height(), (p.get_height() / len(df['Survived'])) * 100), (p.get_x() + 0.33, p.get_height() + 5))
plt.show()


# Based on the data in the `Survived` column, we observe that only 342 passengers managed to survive (38.38%).

# In[13]:


sex = df['Sex'].value_counts()

fig = plt.figure(
    FigureClass = Waffle, 
    rows = 4,
    columns = 8,
    values = sex,
    colors = ('#3274A1', '#FF7FA7'),
    labels = ['{} - {}'.format(a, b) for a, b in zip(sex.index, sex)],
    legend = {
        'loc': 'upper left', 
        'bbox_to_anchor': (1, 1), 
        'fontsize': 15, 
        'labelcolor': 'linecolor',
        'title': 'Gender',
        'title_fontsize': 15
        },
    font_size = 50, 
    icons = ['male','female'],
    icon_legend = True,
    figsize = (10, 8)
)

plt.title('Gender Distribution', fontsize = 20)
plt.show()


# In[14]:


plt.figure(figsize = (15, 10), dpi = 80)
plt.pie([(df.Sex == 'male').sum(), (df.Sex == 'female').sum()], labels = ["Male", "Female"], autopct = "%.2f", startangle = 90, explode = (0.1, 0.0))
plt.title('Percentage of Male and Female Passengers', fontsize = 18)
plt.show()


# Majority of the passengers aboard the Titanic were Male (64.76 %).<br>
# Let us now take a look at the Survival Rates for Male and Female passengers.

# In[15]:


plt.figure(figsize = (15, 10))
ax = sns.countplot(x = 'Survived', hue = 'Sex', data = df)
plt.title('Survival for Male and Female Passengers', fontsize = 20)
plt.xlabel('Survived', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
for p in ax.patches:
        ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x() + 0.17, p.get_height() + 3))
plt.show()


# **Observations**<br>
# - Most of the Male passengers have not survived.<br>
# - Majority of the Female passengers have survived.

# In[16]:


plt.figure(figsize = (15, 10))
ax = sns.countplot(x = 'Survived', hue = 'Pclass', data = df)
plt.title('Survival Based on Class', fontsize = 20)
plt.xlabel('Survival', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
for p in ax.patches:
        ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x() + 0.1, p.get_height() + 3))
plt.show()


# We notice that `Pclass` of the passenger does affect their survival odds.<br>
# Passengers in the 3<sup>rd</sup> class have a much higher mortality rate as compared to the other two classes.<br>
# The 1<sup>st</sup> class has a higher number of passengers that survived, probably because they were richer.

# In[17]:


plt.figure(figsize = (15, 10))
sns.distplot(df['Age'].dropna(), color = (0, 0.5, 1), bins = 40, kde = True)
plt.title('Age Density of the Passengers', fontsize = 20)
plt.xlabel('Age', fontsize = 15)
plt.show()


# Majority of the passengers on the Titanic were between 20 to 40 years of age

# In[18]:


plt.figure(figsize = (15, 10))
ax = sns.countplot(x = 'SibSp', data = df, palette = ['#004A93', '#017CB2', '#37ACCF', '#6BBFDB', '#85CCDE', '#77C4DA', '#A4D6E1'])
plt.title('Siblings/Spouses on Board', fontsize = 20)
plt.xlabel('SibSp', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
for p in ax.patches:
        ax.annotate('{:.0f} = {:.2f}%'.format(p.get_height(), (p.get_height() / len(df['SibSp'])) * 100), (p.get_x() + 0.15, p.get_height() + 5))
plt.show()


# `SibSp` indicates the number of Siblings or Spouses on board.<br>
# From the countplot we see that most of the passengers were travelling alone.

# In[19]:


plt.figure(figsize = (15, 10))
ax = sns.countplot(x = 'Parch', data = df, palette = ['#004A93', '#017CB2', '#37ACCF', '#6BBFDB', '#85CCDE', '#77C4DA', '#A4D6E1'])
plt.title('Parents/Children on Board', fontsize = 20)
plt.xlabel('Parch', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
for p in ax.patches:
        ax.annotate('{:.0f} = {:.2f}%'.format(p.get_height(), (p.get_height() / len(df['SibSp'])) * 100), (p.get_x() + 0.15, p.get_height() + 5))
plt.show()


# `Parch` indicates the number of Parents or Children aboard the ship<br>
# Just like in the case of `SibSp` we can observe that most of the people are travelling on their own.

# In[20]:


plt.figure(figsize = (15, 10))
sns.histplot(df['Fare'], bins = 40, kde = True)
plt.title('Fare Count for the Passengers', fontsize = 20)
plt.xlabel('Fare', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()


# In[21]:


plt.figure(figsize = (15, 10))
bp = sns.boxplot(x = 'Pclass', y = 'Age', data = df, palette = 'winter')
plt.xlabel('Pclass', fontsize = 15)
plt.ylabel('Age', fontsize = 15)
plt.show()


# Looking at the median age of the passengers based on their class.<br>
# The median age of the passengers in the 1<sup>st</sup> class is the highest.<br>
# The median age of the passengers in the 3<sup>rd</sup> class is the lowest.<br>
# Let us fill in the missing values in the Age column based on the median age of the `Pclass`.

# In[22]:


def transform_columns(column):
    Age = column[0]
    Pclass = column[1]
    
    if(pd.isna(Age)):
       if(Pclass == 1):
            return 38
       elif(Pclass == 2):
            return 29   
       else:
            return 23  
    else:
       return Age

df['Age'] = df[['Age', 'Pclass']].apply(transform_columns, axis = 1)
df


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Feature Engineering</strong></div>

# Lets create a new feature `IsAlone` that tells us if the passenger is travelling solo or with a family.

# In[23]:


df['IsAlone'] = df['SibSp'] + df['Parch']
df


# In[24]:


plt.figure(figsize = (15, 10))
ax = sns.countplot(x = 'IsAlone', data = df)
plt.xlabel('IsAlone', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x() + 0.30, p.get_height() + 5))
plt.show()


# As we can observe from the `IsAlone` feature most passengers are travelling by themselves, without any family.

# In[25]:


def convert_IsAlone(df):
    
    bins = [None] * len(df)

    for i in range(len(df)):
        if(df.IsAlone[i] in [0]):
            bins[i] = 'Alone'
        if(df.IsAlone[i] in [1, 2, 3, 4, 5, 6, 7, 10]):
            bins[i] = 'Not Alone'

    df['IsAlone'] = bins
    
convert_IsAlone(df)
df


# We created 2 groups for the `IsAlone` feature.<br>
# - The first group is named `Alone` and contains passengers travelling alone.<br>
# - The second group `Not Alone` is for passengers having one or more family member.

# In[26]:


plt.figure(figsize = (15, 10))
ax = sns.countplot(x = 'Survived', hue = 'IsAlone', data = df)
plt.title('Survival Count for the IsAlone Feature', fontsize = 20)
plt.xlabel('Survived', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
for p in ax.patches:
        ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x() + 0.17, p.get_height() + 3))
plt.show()


# - Passengers that travelled alone have a higher mortality than passengers that travelled with family.
# - The survival chances for passengers tha travelled alone and those that travelled with family is almost the same.

# In[27]:


df


# In[28]:


df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)
df


# In[29]:


msno.bar(df, color = (0, 0.4, 0.8), sort = "ascending", figsize = (15, 10))
plt.show()


# In[30]:


plt.figure(figsize = (15, 10))
sns.heatmap(df.isna(), yticklabels = False, cbar = False, cmap = 'Blues')
plt.title("Visualizing the Missing Data", fontsize = 20)
plt.xticks(rotation = 35, fontsize = 15)
plt.show()


# Our dataset no longer contains any missing values. We can now encode and scale the data to start training our ML models.

# # <div class = "alert alert-info" style = "margin:0px;"><strong>Checking for Correlation</strong></div>

# In[31]:


plt.figure(figsize = (15, 10))
sns.heatmap(df.corr(), cmap = 'Blues', square = True, annot = True)
plt.title("Visualizing Correlations", size = 20)
plt.show()


# In[32]:


numeric_features = ['Age', 'Fare']
sns.pairplot(df[numeric_features], size = 5)
plt.show()


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Encoding the Categorical Features</strong></div>
# The categorical data can be encoded using Label Encoder. It encodes labels with a value between 0 and n_classes - 1 where n is the number of distinct labels. If a label repeats it assigns the same value as assigned earlier. The categorical values can be converted into numeric values.

# In[33]:


label_encoder = LabelEncoder()

def label_encoder_converter(df):
    
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['IsAlone'] = label_encoder.fit_transform(df['IsAlone'])
    
label_encoder_converter(df)


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Scaling the Data</strong></div>
# StandardScaler standardizes a feature by subtracting the mean and then scaling it to unit variance.
# <div style="width:100%;text-align: center;"> <img align = left src="https://cdn-images-1.medium.com/max/800/0*vQEjz0mvylP--30Q.GIF" style="height:150px"></div>

# In[34]:


scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
df


# In[35]:


X = df.iloc[:, 1:]
y = df['Survived']
print(X, "\n\n\n", y)


# In[36]:


df


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Splitting the Data into Train and Test Sets</strong></div>

# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # <div class = "alert alert-info" style = "margin:0px;"><strong>What is a Confusion Matrix ?</strong></div>
# > A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known.
# Let us understand some of the terms associated with a confusion matrix:
# > - **True Negative**: You predicted a Negative and its True.
# > - **True Positive**: You predicted a Positive and its True.
# > - **False Positive**: You predicted a Positive but its False.
# > - **False Negative**: You predicted a Negative but its False.
#  
# **Credits**: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea

# In[38]:


labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
labels = np.asarray(labels).reshape(2,2)
plt.figure(figsize = (8, 6))
sns.heatmap(confusion_matrix(y_test, (LogisticRegression().fit(X_train, y_train)).predict(X_test)), 
            annot = labels, fmt = '', 
            cmap = 'Blues', 
            annot_kws={'size': 17}, 
            square = True)
plt.yticks(size = 15)
plt.xticks(size = 15)
plt.show()


# In[39]:


def get_model_results(cm_for_mod, y_test, y_pred, model_name):
        
    print('The F1 score for ' + model_name + ' is:', f1_score(y_test, y_pred))
    
    fig, axes = plt.subplots(1, 2, figsize = (15, 8))
    
    fig.suptitle('Graphs for ' + model_name, fontsize = 20)
    
    sns.heatmap(cm_for_mod, ax = axes[0], annot = True, cmap = 'Blues', annot_kws = {'size': 15}, square = True)
    axes[0].set_title('Confusion Matrix', fontsize = 15)
    
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    sns.lineplot(fpr, tpr, ax = axes[1])
    axes[1].set_title('ROC Curve (' + str(round(roc_auc, 3)) + ')', fontsize = 15)
    axes[1].plot([0, 1], [0, 1],'b--'), 2
    plt.show()


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Logistic Regression</strong></div>
# Logistic regression is a supervised learning algorithm used to predict the probability of a target variable. It is used for classification, in this case to predict whether a passenger survived or not

# In[40]:


lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
acc_lr = lr.score(X_test, y_test)
print("The training accuracy for logistic regression is:", lr.score(X_train, y_train) * 100, "%")
print("The testing accuracy for logistic regression is:", acc_lr * 100, "%")
cm_lr = confusion_matrix(y_test, y_pred)
get_model_results(cm_lr, y_test, y_pred, 'Logistic Regression')


# # <div class = "alert alert-info" style = "margin:0px;"><strong>K-Nearest Neighbors</strong></div>
# KNN works by finding the distances between a query and all the examples in the data, selecting the specified number examples (K) closest to the query, then votes for the most frequent label.

# In[41]:


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = knn.score(X_test, y_test)
print("The training accuracy for KNN is:", knn.score(X_train, y_train) * 100, "%")
print("The testing accuracy for KNN is:", acc_knn * 100, "%")
cm_knn = confusion_matrix(y_test, y_pred)
get_model_results(cm_knn, y_test, y_pred, 'K Nearest Neighbors')


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Support Vector Classifier</strong></div>
# It is used in classification problems to predict which class the target variable belongs to.

# In[42]:


svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_svc = svc.score(X_test, y_test)
print("The training accuracy for SVC is:", svc.score(X_train, y_train) * 100, "%")
print("The testing accuracy for SVC is:", acc_svc * 100, "%")
cm_svc = confusion_matrix(y_test, y_pred)
get_model_results(cm_svc, y_test, y_pred, 'Support Vector Classifier')


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Decision Tree Classifier</strong></div>
# Decision trees use multiple algorithms to decide to split a node into two or more sub-nodes. The creation of sub-nodes increases the homogeneity of resultant sub-nodes. In other words, we can say that the purity of the node increases with respect to the target variable.
# <div style="width:100%;text-align: center;"> <img align = left src="https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1545934190/1_r5ikdb.png" style="height:500px"> </div>

# In[43]:


dtc = DecisionTreeClassifier(random_state = 0)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
acc_dtc = dtc.score(X_test, y_test)
print("The training accuracy for Decision Tree Classifier is:", dtc.score(X_train, y_train) * 100, "%")
print("The testing accuracy for Decision Tree Classifier is:", acc_dtc * 100, "%")
cm_dtc = confusion_matrix(y_test, y_pred)
get_model_results(cm_dtc, y_test, y_pred, 'Decision Tree Classifier')


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Visualizing the Decision Tree Classifier</strong></div>

# In[44]:


dot_data = tree.export_graphviz(dtc, out_file = None, feature_names = X.columns, class_names = ["0", "1"], filled = True)
graph = graphviz.Source(dot_data, format = "jpg")
display(graph)


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Random Forest Classifier</strong></div>
# The random forest is a classification algorithm consisting of many decisions trees. It uses bagging and feature randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree.
# <div style="width:100%;text-align: center;"> <img align = left src="https://www.freecodecamp.org/news/content/images/2020/08/how-random-forest-classifier-work.PNG" style="height:400px"> </div>

# In[45]:


rf = RandomForestClassifier(random_state = 0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc_rf = rf.score(X_test, y_test)
print("The training accuracy for Random Forest Classifier is:", rf.score(X_train, y_train) * 100, "%")
print("The testing accuracy for Random Forest Classifier is:", acc_rf * 100, "%")
cm_rf = confusion_matrix(y_test, y_pred)
get_model_results(cm_rf, y_test, y_pred, 'Random Forest Classifier')


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Ada Boost Classifier</strong></div>
# It combines multiple classifiers to increase the accuracy of classifiers. AdaBoost is an iterative ensemble method. AdaBoost classifier builds a strong classifier by combining multiple poorly performing classifiers so that you will get high accuracy strong classifier.

# In[46]:


adc = AdaBoostClassifier()
adc.fit(X_train, y_train)
y_pred = adc.predict(X_test)
acc = adc.score(X_test, y_test)
acc_adc = adc.score(X_test, y_test)
print("The training accuracy for Ada Boost Classifier is:", adc.score(X_train, y_train) * 100, "%")
print("The testing accuracy for Ada Boost Classifier is:", acc_adc * 100, "%")
cm_adc = confusion_matrix(y_test, y_pred)
get_model_results(cm_adc, y_test, y_pred, 'Ada Boost Classifier')


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Extra Trees Classifier</strong></div>
# This is a type of ensemble learning technique which aggregates the results of multiple de-correlated decision trees collected in a “forest” to output it's classification result.

# In[47]:


etc = ExtraTreesClassifier()
etc.fit(X_train, y_train)
y_pred = etc.predict(X_test)
acc_etc = etc.score(X_test, y_test)
print("The training accuracy for Extra Trees Classifier is:", etc.score(X_train, y_train) * 100, "%")
print("The testing accuracy for Extra Trees Classifier is:", acc_etc * 100, "%")
cm_etc = confusion_matrix(y_test, y_pred)
get_model_results(cm_etc, y_test, y_pred, 'Extra Trees Classifier')


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Bagging Classifier</strong></div>
# Bagging classifier is an ensemble technique that fits base classifiers each on random subsets of the original dataset and then aggregates their individual predictions to form a final prediction.

# In[48]:


bgc = BaggingClassifier()
bgc.fit(X_train, y_train)
y_pred = bgc.predict(X_test)
acc_bgc = bgc.score(X_test, y_test)
print("The training accuracy for Bagging Classifier is:", bgc.score(X_train, y_train) * 100, "%")
print("The testing accuracy for Bagging Classifier is:", acc_bgc * 100, "%")
cm_bgc = confusion_matrix(y_test, y_pred)
get_model_results(cm_bgc, y_test, y_pred, 'Bagging Classifier')


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Gradient Boosting Classifier</strong></div>
# Gradient boosting classifiers are a group of machine learning algorithms that combine many weak learning models together to create a strong predictive model.

# In[49]:


gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)
acc_gbc = gbc.score(X_test, y_test)
print("The training accuracy for Gradient Boosting Classifier is:", gbc.score(X_train, y_train) * 100, "%")
print("The testing accuracy for Gradient Boosting Classifier is:", acc_gbc * 100, "%")
cm_gbc = confusion_matrix(y_test, y_pred)
get_model_results(cm_gbc, y_test, y_pred, 'Gradient Boosting Classifier')


# # <div class = "alert alert-info" style = "margin:0px;"><strong>XGBoost Classifier</strong></div>
# XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. XGBoost provides a highly efficient implementation of the stochastic gradient boosting algorithm and access to a suite of model hyperparameters designed to provide control over the model training process.

# In[50]:


xgbc = XGBClassifier(n_jobs = -1, silent = True, verbosity = 0)
xgbc.fit(X_train, y_train)
y_pred = xgbc.predict(X_test)
acc_xgbc = xgbc.score(X_test, y_test)
print("The training accuracy for XGBoost Classifier is:", xgbc.score(X_train, y_train) * 100, "%")
print("The testing accuracy for XGBoost Classifier is:", acc_xgbc * 100, "%")
cm_xgbc = confusion_matrix(y_test, y_pred)
get_model_results(cm_xgbc, y_test, y_pred, 'XGB Classifier')


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Cat Boost Classifier</strong></div>
# CatBoost is based on gradient boosted decision trees. During training, a set of decision trees is built consecutively. Each successive tree is built with reduced loss compared to the previous trees.

# In[51]:


cbc = CatBoostClassifier(verbose = 0)
cbc.fit(X_train, y_train)
y_pred = cbc.predict(X_test)
acc_cbc = cbc.score(X_test, y_test)
print("The training accuracy for Cat Boost Classifier is:", cbc.score(X_train, y_train) * 100, "%")
print("The testing accuracy for Cat Boost Classifier is:", acc_cbc * 100, "%")
cm_cbc = confusion_matrix(y_test, y_pred)
get_model_results(cm_cbc, y_test, y_pred, 'Cat Boost Classifier')


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Optimized Random Forest Classifier</strong></div>

# In[52]:


opt_rf = RandomForestClassifier(criterion = 'gini',
                               n_estimators = 1000,
                               max_depth = 7,
                               min_samples_split = 3,
                               min_samples_leaf = 3,
                               max_features = 'auto',
                               oob_score = True,
                               random_state = 0,
                               n_jobs = -1) 
opt_rf.fit(X_train, y_train)
y_pred = opt_rf.predict(X_test)
acc_opt_rf = opt_rf.score(X_test, y_test)
print("The training accuracy for the Optimized Random Forest Classifier is:", opt_rf.score(X_train, y_train) * 100, "%")
print("The testing accuracy for the Optimized Random Forest Classifier is:", acc_opt_rf * 100, "%")
cm_opt_rf = confusion_matrix(y_test, y_pred)
get_model_results(cm_opt_rf, y_test, y_pred, 'Optimized Random Forest Classifier')


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Creating the Submission File</strong></div>

# In[53]:


test_df = pd.read_csv('../input/titanic/test.csv')
test_df['Age'] = test_df[['Age', 'Pclass']].apply(transform_columns, axis = 1)
test_df = test_df.drop(columns = ['Cabin'], axis = 1)
test_df = test_df.fillna(df['Fare'].mean())
test_df['IsAlone'] = test_df['SibSp'] + test_df['Parch']
convert_IsAlone(test_df)
test_df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Embarked'], axis = 1, inplace = True)
label_encoder_converter(test_df)
test_df[numeric_features] = scaler.fit_transform(test_df[numeric_features])
X = test_df.iloc[:, 0:]
y_pred_opt_rf = opt_rf.predict(X)
final_pred = list(y_pred_opt_rf)
final_sub = pd.read_csv('../input/titanic/test.csv')['PassengerId']
final_sub = pd.DataFrame(final_sub)
final_sub['Survived'] = final_pred
final_sub.to_csv('submission.csv', index = False)
final_sub.head()


# # <div class = "alert alert-info" style = "margin:0px;"><strong>Some Useful References</strong></div>
# - https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python#Ensembling-&-Stacking-models
# - https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial
# - https://www.kaggle.com/ruchi798/break-the-ice
# - https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic
# - https://www.kaggle.com/ohseokkim/titanic-missing-and-small-data-are-disaster
