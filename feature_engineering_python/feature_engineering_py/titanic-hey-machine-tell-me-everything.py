#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <hr style="border: solid 3px blue;">
# 
# # Introduction
# 
# ![](https://c.tenor.com/poOBxHtWHFcAAAAC/tell-me-everything-inform-me.gif)
# 
# Picture Credit: https://c.tenor.com

# <span style="color:Blue"> **Cognitive Bias**
# 
# > A cognitive bias is a systematic pattern of deviation from norm or rationality in judgment. Individuals create their own "subjective reality" from their perception of the input. An individual's construction of reality, not the objective input, may dictate their behavior in the world. Thus, cognitive biases may sometimes lead to perceptual distortion, inaccurate judgment, illogical interpretation, or what is broadly called irrationality.
# >  1. some are due to ignoring relevant information (e.g., neglect of probability),
# >  2. some involve a decision or judgment being affected by irrelevant information (for example the framing effect where the same problem receives different responses depending on how it is described; or the distinction bias where choices presented together have different outcomes than those presented separately), and
# >  3. others give excessive weight to an unimportant but salient feature of the problem (e.g., anchoring).
#     
# Ref: https://en.wikipedia.org/wiki/Cognitive_bias
#     
# We may have cognitive biases when we first encounter datasets. For example, when looking at the features of the titanic dataset, you might think that gender and age are important without detailed analysis. Of course, such a decision could be right. However, we can receive feedback through the model to see if our thinking is correct and modify our thinking based on this. 
# 
# At this point, we once again strongly say: **Machine! Tell Me Everything!**
# 
# In this notebook, we analyze the dataset in as many ways as possible and try to understand the behavior of the models and the dataset through this.
# To do this, we would like to organize the notebook in the following order.
# 1. First, check what outliers there are and what characteristics the outliers have.
# 2. Perform detailed EDA and pre-process.
# 3. Check which features are important in various ways.
# 4. Model using deep learning (fast.ai) and understand deep learning.
# 5. Using ML (ensemble), do three modeling and understand the ensemble operation.

# ---------------------------------------------------------------
# # Setting up

# In[2]:


try:
    import pycaret
except:
    get_ipython().system('pip install pycaret')

try:
    import missingno
except:
    get_ipython().system('pip install missingno')


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, log_loss
from sklearn import preprocessing
import umap
import umap.plot
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import scipy.stats as stats

import warnings
warnings.filterwarnings('ignore')


# # Loading Dataset

# In[4]:


train_data = pd.read_csv('../input/titanic/train.csv')
train_data_org = train_data.copy()
test_data = pd.read_csv('../input/titanic/test.csv')
submission_data = pd.read_csv('../input/titanic/gender_submission.csv')
titanic_df = pd.concat([train_data, test_data], ignore_index = True, sort = False)
tr_idx = titanic_df['Survived'].notnull()
titanic_dl_df = titanic_df.copy()


# In[5]:


titanic_df.head(5).T.style.set_properties(**{'background-color': 'black',
                           'color': 'white',
                           'border-color': 'white'})


# PassengerId has nothing to do with survior. It can be removed immediately.

# In[6]:


titanic_df.drop(['PassengerId'],axis=1,inplace=True)


# <hr style="border: solid 3px blue;">
# 
# # Detecting Anomaly 
# 
# ![](https://miro.medium.com/max/697/1*O3lOgPwuHP7Vfc1T6NDRrQ.png)
# 
# Picture Credit:https://miro.medium.com
# 
# <span style="color:Blue"> **Anomaly Detection**
# > In data analysis, anomaly detection (also outlier detection) is the identification of rare items, events or observations which raise suspicions by differing significantly from the majority of the data. Typically the anomalous items will translate to some kind of problem such as bank fraud, a structural defect, medical problems or errors in a text. Anomalies are also referred to as outliers, novelties, noise, deviations and exceptions.
# 
# Ref: https://en.wikipedia.org/wiki/Anomaly_detection
#     
# Before EDA we can get a guide to anomaly (outlier) from the model. In fact, it is almost impossible to perform the task of finding outliers in a high-dimensional dataset without the help of a model.
#     
# We can look for anomaly before proceeding with EDA and gain insight into future EDA.

# In[7]:


from pycaret.anomaly import *


# --------------------------------------------
# ## Setting Up
# 
# > This function initializes the training environment and creates the transformation pipeline. Setup function must be called before executing any other function. It takes one mandatory parameter: data. All the other parameters are optional.
# 
# Ref: https://pycaret.readthedocs.io/en/latest/api/anomaly.html

# In[8]:


pycaret.anomaly.setup(
    data=train_data_org,
    silent=True)


# -----------------------------------------------
# ## Creating Model
# 
# > This function trains a given model from the model library. All available models can be accessed using the models function.
# 
# Ref: https://pycaret.readthedocs.io/en/latest/api/anomaly.html

# In[9]:


pca = pycaret.anomaly.create_model('pca')


# -------------------------------
# ## Assigning Model
# > This function assigns anomaly labels to the dataset for a given model. (1 = outlier, 0 = inlier).
# 
# Ref: https://pycaret.readthedocs.io/en/latest/api/anomaly.html

# In[10]:


pca_df = pycaret.anomaly.assign_model(pca)


# ----------------------------------------------------------
# ## Picking outliers
# **Let's pick the top 10 outliers**

# In[11]:


abnormal_data = pca_df[pca_df.Anomaly == 1].sort_values(by='Anomaly_Score', ascending=False)
print("the size of anomaly = ",len(abnormal_data))
abnormal_data.head(10).style.set_properties(**{'background-color': 'black',
                           'color': 'white',
                           'border-color': 'white'})


# <span style="color:Blue"> Observation:
# * The number of data determined as anomaly is 45.
# * Among the outliers, there is a lot of information about passengers with Pclass 1. 
# * In many cases, the information of passengers with Embarked S and C is determined to be an outlier.

# It is noteworthy that passengers with Pclass 1 are especially often judged as outliers. Will we be able to solve this mystery through EDA?

# ------------------------------
# ## Tuning Model
# 
# > This function tunes the fraction parameter of a given model.
# 
# Ref: https://pycaret.readthedocs.io/en/latest/api/anomaly.html

# In[12]:


tuned_pca = tune_model(model = 'pca', supervised_target = 'Survived')


# ---------------------------------------
# ## Plotting Model
# 
# > This function analyzes the performance of a trained model.
# 
# Ref: https://pycaret.readthedocs.io/en/latest/api/anomaly.html
# 
# 
# Both u-MAP and t-SNE enable low-dimensional visual confirmation through dimensionality reduction.
# If you are interested in dimensional reduction, you can take a look at the notebook below.
# 
# [dimensionality-dimension-reduction](https://www.kaggle.com/ohseokkim/the-curse-of-dimensionality-dimension-reduction)

# UMAP (Uniform Manifold Approximation and Projection), which is faster than t-SNE and separates the data space well, has been proposed for nonlinear dimensionality reduction. In other words, it can process very large datasets quickly and is suitable for sparse matrix data. Furthermore, compared to t-SNE, it has the advantage of being able to embed immediately when new data comes in from other machine learning models.

# In[13]:


plt.style.use("dark_background")
plot_model(tuned_pca,plot='umap')


# t-SNE is often used for visualization purposes by compressing data on a two-dimensional plane. Points that are close to the original feature space are also expressed in a two-dimensional plane after compression. Since the nonlinear relationship can be identified, the model performance can be improved by adding the compression results expressed by these t-SNEs to the original features. However, since the computation cost is high, it is not suitable for compression exceeding two or three dimensions.

# In[14]:


plot_model(tuned_pca,plot='tsne')


# <hr style="border: solid 3px blue;">
# 
# # EDA

# ## Checking Data Type

# In[15]:


titanic_df.info()


# In[16]:


plt.figure(figsize = (10,8))
with plt.rc_context({'figure.facecolor':'black'}):
    sns.set(style="ticks", context="talk",font_scale = 1)
    plt.style.use("dark_background")
    ax = titanic_df.dtypes.value_counts().plot(kind='bar',fontsize=20,color='purple')
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+ p.get_width() / 2., height + 0.1, height, ha = 'center', size = 25)
    sns.despine()


# <span style="color:Blue"> Observation:
# * There are 5 object features. It is expected with categorical features.
# * There are 3 float-type features.
# * There are 3 integer-type features.

# # Checking Target Value Imbalace

# In[17]:


colors = ['gold', 'mediumturquoise']
labels = ['Non-Suvivor','Suvivor']
values = titanic_df['Survived'].value_counts()/titanic_df['Survived'].shape[0]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='white', width=0.1)))
fig.update_layout(
    title_text="Titanic Survivor",
    title_font_color="white",
    legend_title_font_color="yellow",
    paper_bgcolor="black",
    plot_bgcolor='black',
    font_color="white",
)
fig.show()


# Although the survivor is small, the imbalance is not large enough for over/under sampling.
# If you want to know more about over/under sampling, please refer to the notebook below.
# 
# [Over/Under sampling](https://www.kaggle.com/ohseokkim/preprocessing-resolving-imbalance-by-sampling)

# ## Checking and Handling Missing Values

# In[18]:


import missingno as msno
plt.style.use("dark_background")
msno.matrix(titanic_df.drop(['Survived'],axis=1),color=(138/255,43/255,226/255),fontsize=30)


# In[19]:


isnull_series = titanic_df.loc[:,:'Cabin'].isnull().sum()
isnull_series[isnull_series > 0].sort_values(ascending=False)

plt.figure(figsize = (20,10))

ax = isnull_series[isnull_series > 0].sort_values(ascending=False).plot(kind='bar',
                                                                        grid = False,
                                                                        fontsize=20,
                                                                        color='purple')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+ p.get_width() / 2., height + 5, height, ha = 'center', size = 30)
sns.despine()


# <span style="color:Blue"> Observation:
#     
# There are missing values for Age, Cabin, Fare, and Embarked features. In particular, there are many missing values for Age and Cabin features. Let's think about how to handle these missing values.

# -------------------------------------------------------------
# # Categorical Features
# 
# ![](https://miro.medium.com/max/1400/1*wYbTRM0dgnRzutwZq63xCg.png)
# 
# Picture Credit: https://miro.medium.com
# 
# Categorical data can be classified into ordinal data and nominal data. In the case of an ordinal type, there is a difference in importance for each level. This value plays an important role in the case of regression, so encode it with care.
# 
# It is difficult to encode categorical features compared to numeric features. For ordinal data, it is more difficult.

# ------------------------------------------------------------------------
# ## Has_Cabin ( Derived variable )
# 
# **Question: Is there a difference in the survival rate between passengers with and without cabin?**

# In[20]:


titanic_df['Has_Cabin'] = titanic_df['Cabin'].isnull().astype(int)


# In[21]:


total_cnt = titanic_df['Survived'].count()
plt.figure(figsize=(12,8))
sns.set(style="ticks", context="talk",font_scale = 1)
plt.style.use("dark_background")
ax = sns.countplot(x="Has_Cabin",
                   hue="Survived", 
                   data=titanic_df,
                   palette = 'Purples_r')
ax.set_title('Survived Count/Rate')
for p in ax.patches:
    x, height, width = p.get_x(), p.get_height(), p.get_width()
    ax.text(x + width / 2, height + 10, f'{height} / {height / total_cnt * 100:2.1f}%', va='center', ha='center', size=20)
sns.despine()


# <span style="color:Blue"> Observation:
# 
# * Cases with cabins have more survivors compared to cases without cabins. It is likely that the new derived variable will be helpful in the classification of survivors.

# ---------------------------------------------
# ## Cabin_Label ( Derived variable )

# In[22]:


titanic_df['Cabin'] = titanic_df['Cabin'].fillna('N')
titanic_df['Cabin_label'] = titanic_df['Cabin'].str.get(0)

sns.set(style="ticks", context="talk",font_scale = 2)
plt.style.use("dark_background")

plt.figure(figsize=(25,10))
ax = sns.countplot(x="Cabin_label", hue="Survived", data=titanic_df,palette = 'Purples_r')
ax.set_title('Survived Rate')
plt.legend(loc = 'upper right')
for p in ax.patches:
    x, height, width = p.get_x(), p.get_height(), p.get_width()
    ax.text(x + width / 2, height + 10, f'{height / total_cnt * 100:2.1f}%',va='center', ha='center', size=25)
sns.despine()


# ________________________________________________________
# ## Name
# 
# It seems difficult to find the feature directly related to the survivor.

# In[23]:


import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

titanic_df['Title'] = titanic_df['Name'].apply(get_title)


# In[24]:


titanic_df['Title'] = titanic_df['Title'].replace(
       ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 
       'Rare')

titanic_df['Title'] = titanic_df['Title'].replace('Mlle', 'Miss')
titanic_df['Title'] = titanic_df['Title'].replace('Ms', 'Miss')
titanic_df['Title'] = titanic_df['Title'].replace('Mme', 'Mrs')
titanic_df['Title'].unique()


# In[25]:


rcParams['figure.figsize'] = 20,10
ax = sns.countplot(x='Title',hue ='Survived',data=titanic_df,palette="Purples_r")
ax.set_title('Survived Rate')
plt.legend(loc = 'upper right')
for p in ax.patches:
    x, height, width = p.get_x(), p.get_height(), p.get_width()
    ax.text(x + width / 2, height + 10, f'{height / total_cnt * 100:2.1f}%',va='center', ha='center', size=25)
sns.despine()


# <span style="color:Blue"> Observation:
# 
# The mortality rate is higher in the case of Mr. I think it will help with learning.

# In[26]:


rcParams['figure.figsize'] = 20,15
titles = titanic_df['Title'].unique()
plt.subplots_adjust(hspace=1.5)
sns.set(style="ticks", context="talk",font_scale = 1)
plt.style.use("dark_background")
idx = 1
for title in titles:
    plt.subplot(3,2,idx)
    ax = sns.histplot(x='Age',data=titanic_df[titanic_df['Title']== title],hue ='Survived',palette="Purples_r",kde=True)
    ax.set_title(title)
    sns.despine()
    idx = idx + 1


# <span style="color:Blue"> Observation:
# * In the case of Mr, the number of survivors is small.
# * In the case of Mrs and Miss, there are many survivors.
# 
# I think it will be helpful in judging survivors using this.
# However, it seems difficult to find the relationship between age and title from the above distributions. Therefore, it seems difficult to use this to fill in the missing values of age.

# ---------------------------------------
# ## Embarked

# In[27]:


rcParams['figure.figsize'] = 12,7
sns.set(style="ticks", context="talk",font_scale = 1)
plt.style.use("dark_background")
ax = sns.countplot(x='Embarked',hue = 'Survived',data=titanic_df,palette="Purples_r")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 3, f'{height / total_cnt * 100:2.1f}%', ha = 'center', size = 25)
sns.despine()


# <span style="color:Blue"> Observation:
#     
# * Many passengers on board at S port died.
# * For passengers boarding at port C, the survival rate is higher than the mortality rate.

# In[28]:


rcParams['figure.figsize'] = 12,7
sns.set(style="ticks", context="talk",font_scale = 1)
plt.style.use("dark_background")
ax = sns.countplot(x='Embarked',hue = 'Sex',data=titanic_df,palette="Purples_r")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 3, f'{height / total_cnt * 100:2.1f}%', ha = 'center', size = 25)
sns.despine()


# **Let's impute missing value for Embarked feature. The strategy for Embarked's missing values is to choose 'most_frequent'.**

# In[29]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
titanic_df[['Embarked']] = imp.fit_transform(titanic_df[['Embarked']])


# ----------------------------------------------------
# # Numerical Features
# 
# ![](https://static-assets.codecademy.com/Courses/Hypothesis-Testing/Intro_to_variable_types_4.png)
# 
# Picture Credit: https://t3.ftcdn.net

# ### An Extension To Imputation
# 
# > Imputation is the standard approach, and it usually works well. However, imputed values may be systematically above or below their actual values (which weren't collected in the dataset). Or rows with missing values may be unique in some other way. In that case, your model would make better predictions by considering which values were originally missing.
# 
# ![](https://i.imgur.com/UWOyg4a.png)
# 
# > In this approach, we impute the missing values, as before. And, additionally, for each column with missing entries in the original dataset, we add a new column that shows the location of the imputed entries.
# > 
# > In some cases, this will meaningfully improve results. In other cases, it doesn't help at all.
# 
# Ref: https://www.kaggle.com/alexisbcook/missing-values

# ----------------------------------------------------------------------
# ## SibSp ( Number of siblings/spouses )

# In[30]:


rcParams['figure.figsize'] = 20,10
sns.set(style="ticks", context="talk",font_scale = 2)
plt.style.use("dark_background")
ax = sns.countplot(x='SibSp',hue ='Survived',data=titanic_df,palette="Purples_r")
ax.set_title('Survived Rate')
plt.legend(loc = 'upper right')
for p in ax.patches:
    x, height, width = p.get_x(), p.get_height(), p.get_width()
    ax.text(x + width / 2, height + 7, f'{height / total_cnt * 100:2.1f}%',va='center', ha='center', size=20)
sns.despine()


# ------------------------------------------------
# ## Parch ( Number of parents/children )

# In[31]:


rcParams['figure.figsize'] = 20,10
sns.set(style="ticks", context="talk",font_scale = 2)
plt.style.use("dark_background")
ax = sns.countplot(x='Parch',hue ='Survived',data=titanic_df,palette="Purples_r")
ax.set_title('Survived Rate')
plt.legend(loc = 'upper right')
for p in ax.patches:
    x, height, width = p.get_x(), p.get_height(), p.get_width()
    ax.text(x + width / 2, height + 7, f'{height / total_cnt * 100:2.1f}%',va='center', ha='center', size=20)
sns.despine()


# -------------------------------------------------
# ## FamilySize ( Derived variable )

# **Question: Does the number of accompanying family members affect the survival rate??**

# In[32]:


titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1
rcParams['figure.figsize'] = 25,10
sns.set(style="ticks", context="talk",font_scale = 2)
plt.style.use("dark_background")
ax = sns.countplot(x='FamilySize',hue ='Survived',data=titanic_df,palette="Purples_r")
ax.set_title('Survived Rate')
plt.legend(loc = 'upper right')
for p in ax.patches:
    x, height, width = p.get_x(), p.get_height(), p.get_width()
    ax.text(x + width / 2, height + 7, f'{height / total_cnt * 100:2.1f}%',va='center', ha='center', size=20)
sns.despine()


# <span style="color:Blue"> Observation:
# 
# When FamilySize is 1, the survival rate is significantly lower than in other cases. I think it will be helpful when the model is learning.

# ______________________________________________
# ## Alone ( Derived variable )

# In[33]:


titanic_df['IsAlone'] = 0
titanic_df.loc[titanic_df['FamilySize'] == 1, 'IsAlone'] = 1

rcParams['figure.figsize'] = 15,10
sns.set(style="ticks", context="talk",font_scale = 1.5)
plt.style.use("dark_background")
ax = sns.countplot(x='IsAlone',hue ='Survived',data=titanic_df,palette="Purples_r")
ax.set_title('Survived Rate')
for p in ax.patches:
    x, height, width = p.get_x(), p.get_height(), p.get_width()
    ax.text(x + width / 2, height + 10, f'{height / total_cnt * 100:2.1f}%',va='center', ha='center', size=25)
sns.despine()


# <span style="color:Blue"> Observation:
# 
# Those who were alone died more than those who were not alone. The derived feature seems to be helpful for model training.

# -----------------------------------
# ## Fare      

# In[34]:


plt.figure(figsize = (10,13))
sns.set(style="ticks", context="talk",font_scale = 1.5)
plt.style.use("dark_background")
plt.subplots_adjust(hspace=0.3)
ax1 = plt.subplot(2,1,1)
sns.histplot(x="Fare", hue="Survived", data=titanic_df,palette = 'Purples_r',kde=True)
ax1.axvline(x=titanic_df['Fare'].mean(), color='g', linestyle='--', linewidth=3)
ax1.text(titanic_df['Fare'].mean(), 90, "Mean", horizontalalignment='left', size='small', color='yellow', weight='semibold')
ax1.set_title('Fare Histogram',fontsize=20)
ax2 = plt.subplot(2,1,2)
stats.probplot(titanic_df['Fare'],dist = stats.norm, plot = ax2)
ax2.set_title('Fare Q-Q plot',fontsize=20)
sns.despine()

mean = titanic_df['Fare'].mean()
std = titanic_df['Fare'].std()
skew = titanic_df['Fare'].skew()
print('Fare : mean: {0:.4f}, std: {1:.4f}, skew: {2:.4f}'.format(mean, std, skew))


# **It is skewed to one side. Consider nonlinear scaling. In this case, we will use QuantileTransformer.**
# 
# > The quantile function ranks or smooths out the relationship between observations and can be mapped onto other distributions, such as the uniform or normal distribution.
# 
# If you want to know more about Scaling, please refer to the notebook below.
# 
# [NotebooK](https://www.kaggle.com/ohseokkim/preprocessing-linear-nonlinear-scaling)

# In[35]:


from sklearn.preprocessing import QuantileTransformer
transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
titanic_df['Fare'] = transformer.fit_transform(titanic_df[['Fare']])


# In[36]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
titanic_df[['Fare']] = imp.fit_transform(titanic_df[['Fare']])


# In[37]:


plt.figure(figsize = (10,13))
plt.subplots_adjust(hspace=0.3)
sns.set(style="ticks", context="talk",font_scale = 1.5)
plt.style.use("dark_background")
ax1 = plt.subplot(2,1,1)
sns.histplot(x="Fare", hue="Survived", data=titanic_df,palette = 'Purples_r',kde=True)
ax1.axvline(x=titanic_df['Fare'].mean(), color='g', linestyle='--', linewidth=3)
ax1.text(titanic_df['Fare'].mean(), 90, "Mean", horizontalalignment='left', size='small', color='yellow', weight='semibold')
ax1.set_title('Fare Histogram',fontsize=20)
ax2 = plt.subplot(2,1,2)
stats.probplot(titanic_df['Fare'],dist = stats.norm, plot = ax2)
ax2.set_title('Fare Q-Q plot',fontsize=20)
sns.despine()

mean = titanic_df['Fare'].mean()
std = titanic_df['Fare'].std()
skew = titanic_df['Fare'].skew()
print('Fare : mean: {0:.4f}, std: {1:.4f}, skew: {2:.4f}'.format(mean, std, skew))


# <span style="color:Blue"> Observation:
# 
# OK! Skewness decreased. Even looking at the Q-Q plot, the normality improved.

# In[38]:


titanic_df['Fare_class'] = pd.qcut(titanic_df['Fare'], 5, labels=['F1', 'F2', 'F3','F4','F5' ])


# In[39]:


rcParams['figure.figsize'] = 15,10
sns.set(style="ticks", context="talk",font_scale = 1.5)
plt.style.use("dark_background")
ax = sns.histplot(x="Fare_class", hue="Survived", data=titanic_df,palette = 'Purples_r',kde=True)
sns.despine()


# In[40]:


titanic_df['Fare_class'] = titanic_df['Fare_class'].replace({'F1':1,'F2':2,'F3':3,'F4':4,'F5':5})


# --------------------------------------------------------------------------------------------
# ## Has_Age ( Derived variable )
# 
# **Question: Does the survival rate make a difference with and without age records?**

# In[41]:


titanic_df['Has_Age'] = titanic_df['Age'].isnull().astype(int)


# In[42]:


rcParams['figure.figsize'] = 10,6
sns.set(style="ticks", context="talk",font_scale = 1.5)
plt.style.use("dark_background")
ax = sns.countplot(x='Has_Age',hue ='Survived',data=titanic_df,palette="Purples_r")
plt.legend(loc = 'upper right')
ax.set_title('Survived Rate')
for p in ax.patches:
    x, height, width = p.get_x(), p.get_height(), p.get_width()
    ax.text(x + width / 2, height + 12,f'{height / total_cnt * 100:2.1f}%',va='center', ha='center', size=25)
sns.despine()


# <span style="color:Blue"> Observation:
#     
# * More than the case where Age is not missing.
# * Cases in which age is not missed have a higher survival rate than cases in which age is omitted.

# -------------------------------------------------------------------------
# ## Age

# In[43]:


rcParams['figure.figsize'] = 12,7
sns.set(style="ticks", context="talk",font_scale = 1.5)
plt.style.use("dark_background")
ax = sns.histplot(x="Age", hue="Survived", data=titanic_df,palette = 'Purples_r',kde=True)
plt.axvline(x=titanic_df['Age'].mean(), color='g', linestyle='--', linewidth=3)
plt.text(titanic_df['Age'].mean(), 60, "Mean", horizontalalignment='left', size='small', color='yellow', weight='semibold')
sns.despine()


# In[44]:


imputer = KNNImputer(n_neighbors=2, weights="uniform")
titanic_df[['Age']] = imputer.fit_transform(titanic_df[['Age']])


# In[45]:


robuster = RobustScaler()
titanic_df['Age'] = robuster.fit_transform(titanic_df[['Age']])


# ------------------------------------------------------
# # Encoding
# 
# Let's perform encoding on categorical features.
# 
# When only tree-based models are used, label encoding is sufficient. However, we will use one-hot encoding for model extension in the future.
# 
# If you want to know more about the encoding of categorical features, please refer to the notebook below.
# 
# [Notebook](https://www.kaggle.com/ohseokkim/preprocessing-encoding-categorical-data)

# In[46]:


titanic_df = pd.get_dummies(titanic_df, columns = ['Title','Sex', 'Embarked','Cabin_label'], drop_first=True)


# ---------------------------------------
# # Checking Correlation

# In[47]:


corr=titanic_df.corr().round(1)

plt.figure(figsize=(25, 20))
sns.set(style="ticks", context="talk",font_scale = 1)
plt.style.use("dark_background")
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr,annot=True,cmap='Purples',mask=mask,cbar=True)
plt.title('Correlation Plot')


# <span style="color:Blue"> Observation
#     
# * There is a large correlation between FamilySize and SibSp and Parch. Since the derived variable FamilySize is made of SibSp and Parch, SibSp and Parch are removed.
# * The relationship between Cabin and Has_Cabin is high. Therefore, the derived variable Has_Cabin is left and Cabin is removed.
# * The relationship between Fare and Fare_class is high. Fare is selected because skewness is removed by nonlinear transform of the Fare feature.
# * There are many features that are not related to the survived value.

# In[48]:


plt.figure(figsize=(15, 10))
sns.set(style="ticks", context="talk",font_scale = 1)
plt.style.use("dark_background")
abs(corr['Survived']).sort_values()[:-1].plot.barh(color='Purple')


# -------------------------------------------------------------
# # Selecting Features
# 
# Features that are not helpful in judging the above heatmap and survivors, or that have other derived variables, will be removed.

# In[49]:


def drop_features(df):
    df.drop(['Name','Ticket','SibSp','Parch','Fare_class',
             'Cabin','Cabin_label_G','Cabin_label_T',
             'Cabin_label_F','FamilySize','Embarked_Q','Title_Rare'],
            axis=1,
            inplace=True)
    return df

titanic_df = drop_features(titanic_df)


# **Let's check the correlation of each feature.**

# In[50]:


corr=titanic_df.corr().round(1)

plt.figure(figsize=(20, 15))
sns.set(style="ticks", context="talk",font_scale = 1)
plt.style.use("dark_background")
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr,annot=True,cmap='Purples',mask=mask,cbar=True)
plt.title('Correlation Plot')


# Let's check the correlation between the target value (Suvived) and other features.

# In[51]:


plt.figure(figsize=(15, 10))
sns.set(style="ticks", context="talk",font_scale = 1)
plt.style.use("dark_background")
abs(corr['Survived']).sort_values()[:-1].plot.barh(color='Purple')


# In[52]:


tr_idx = titanic_df['Survived'].notnull()
y_titanic_df = titanic_df[tr_idx]['Survived']
X_titanic_df= titanic_df[tr_idx].drop('Survived',axis=1)
X_test_df = titanic_df[~tr_idx].drop('Survived',axis=1)


# In[53]:


X_train, X_val, y_train, y_val=train_test_split(X_titanic_df, y_titanic_df, \
                                                  test_size=0.2, random_state=11)


# -------------------------------------------------------
# ## Explaining features with partial dependence
# 
# > Partial dependence plots (PDP) show the dependence between the target response and a set of input features of interest, marginalizing over the values of all other input features (the ‘complement’ features). Intuitively, we can interpret the partial dependence as the expected target response as a function of the input features of interest
# > 
# Ref: https://scikit-learn.org/stable/modules/partial_dependence.html#partial-dependence

# In[54]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import plot_partial_dependence

clf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
fig,ax = plt.subplots(figsize=(18,35))
sns.set(style="ticks", context="talk",font_scale = 1.2)
plt.style.use("dark_background")
fig.tight_layout()
plot_partial_dependence(clf, X_train, X_train.columns,ax=ax)


# <span style="color:Blue"> Observation
# * Cabin_label_B, Cabin_label_C, Has_Age, and IsAlone are not significant in the RandomForestClassifier model.

# In[55]:


from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=100).fit(X_train, y_train)
fig,ax = plt.subplots(figsize=(18,35))
sns.set(style="ticks", context="talk",font_scale = 1.2)
plt.style.use("dark_background")
fig.tight_layout()
plot_partial_dependence(clf, X_train, X_train.columns,ax=ax)


# <span style="color:Blue"> Observation
# * IsAlone, Title_Miss, Title_Mrs, Cabin_label_B, Cabin_label_C, Has_Age, and IsAlone are not significant in the GradientBoostingClassifier model.

# Looking at the two figures above, it can be seen that the rate of change of the target value for each feature value is different. In particular, it can be seen that the partial dependency on age and fare is very different. Finally, a hint from the above figure is that each model can predict biased results, and it is necessary to harmonize these results with the power of collective intelligence. In other words, you can indirectly feel the need for ensemble.

# In[56]:


all_cols = [cname for cname in X_titanic_df.columns]

sns.set(style="ticks", context="talk",font_scale = 1)
plt.style.use("dark_background")


# ----------------------------------------------------------
# ## Feature importance based on feature permutation
# > The estimator is required to be a fitted estimator. X can be the data set used to train the estimator or a hold-out set. The permutation importance of a feature is calculated as follows. First, a baseline metric, defined by scoring, is evaluated on a (potentially different) dataset defined by the X. Next, a feature column from the validation set is permuted and the metric is evaluated again. The permutation importance is defined to be the difference between the baseline metric and metric from permutating the feature column.
# > 
# Ref: https://scikit-learn.org/stable

# In[57]:


from sklearn.inspection import permutation_importance

result = permutation_importance(
    clf, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
)

forest_importances = pd.Series(result.importances_mean, index=all_cols)


# In[58]:


sorted_idx = result.importances_mean.argsort()
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(figsize=(10,8))
ax.boxplot(
    result.importances[sorted_idx].T, vert=False, labels=X_train.columns[sorted_idx]
)
ax.set_title("Permutation Importances")
fig.tight_layout()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()


# ----------------------------------------------------------------------------
# # Visualizing Training Dataset after Dimension Reduction
# 
# 
# Before training, we can check our processed datasets at a low level. If we can easily determine the boundary even with our eyes in a low dimension, it can be considered that the preprocessing has been done very well.
# 
# Don't be discouraged if you don't see a clear boundary. This is because our dataset is high-dimensional, and only because we may not be able to see it with the naked eye.

# In[59]:


X_train.shape


# The training dataset has 16 dimensions. To show the approximate distribution of the training dataset preprocessed above, let's reduce the dimension to two dimensions and draw it.

# In[60]:


mapper = umap.UMAP().fit(X_train)
umap.plot.points(mapper, labels=y_train, theme='fire')


# <span style="color:Blue"> Observation
#    
# As shown in the figure above, when viewed in two dimensions, there are quite a few areas where the survivors and the dead overlap.

# In[61]:


umap.plot.connectivity(mapper, show_points=True)


# <hr style="border: solid 3px blue;">
# 
# # Deep learning using fast.ai
# 
# ![](https://thumbs.gfycat.com/FlawedImpracticalGuillemot-size_restricted.gif)
# 
# Picture Credit: https://thumbs.gfycat.com
# 
# <span style="color:Blue"> **What is fast.ai?**
#     
# > fastai is a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains, and provides researchers with low-level components that can be mixed and matched to build new approaches. It aims to do both things without substantial compromises in ease of use, flexibility, or performance. This is possible thanks to a carefully layered architecture, which expresses common underlying patterns of many deep learning and data processing techniques in terms of decoupled abstractions. These abstractions can be expressed concisely and clearly by leveraging the dynamism of the underlying Python language and the flexibility of the PyTorch library
# 
# Ref: https://github.com/fastai/fastai
#     
# Modeling DL and setting of hyperparameters are more important compared to classical ML. However, these processes are not easy. Here, we are going to perform these processes using fast.ai. Also, check whether training is done properly using the activation map.

# ----------------------------------------
# ## Preprecessing for DL
# Compared to Classic ML, the process of Feature Engineering is simple.
# 
# In most cases, the following steps are required.
# 
# Handling missing values
# Encoding for categorical features (one-hot encoding is mainly used).
# Standard or Min-Max Scaling

# In[62]:


def drop_features(df):
    df.drop(['Ticket','PassengerId','Cabin'],
            axis=1,
            inplace=True,
            errors='ignore')
    return df

titanic_dl_df = drop_features(titanic_dl_df)
titanic_dl_df['Pclass'] = titanic_dl_df['Pclass'].astype(object)


# In[63]:


def replace_name(name):
    if "Mr." in name: return "Mr"
    elif "Mrs." in name: return "Mrs"
    elif "Miss." in name: return "Miss"
    elif "Master." in name: return "Master"
    elif "Ms.": return "Ms"
    else: return "No"
titanic_dl_df['Name'] = titanic_dl_df['Name'].apply(replace_name)


# In[64]:


imputer = KNNImputer(n_neighbors=2, weights="uniform")
titanic_dl_df[['Age']] = imputer.fit_transform(titanic_dl_df[['Age']])

transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
titanic_dl_df['Fare'] = transformer.fit_transform(titanic_dl_df[['Fare']])

mean_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
titanic_dl_df[['Fare']] = mean_imp.fit_transform(titanic_dl_df[['Fare']])

freq_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
titanic_dl_df[['Embarked']] = freq_imp.fit_transform(titanic_dl_df[['Embarked']])


# In[65]:


from fastai import *
from fastai.tabular.all import * 


# -------------------
# ## Defining TabularDataLoaders

# In[66]:


cat_cols = titanic_dl_df.select_dtypes(include = ['object', 'bool']).columns.tolist()
num_cols = titanic_dl_df.select_dtypes(exclude = ['object', 'bool']).columns.tolist()
num_cols.remove('Survived')
procs = [Categorify, FillMissing, Normalize]
y_names = 'Survived'
y_block = CategoryBlock()
splits = RandomSplitter(valid_pct=0.2)(range_of(train_data))


# In[67]:


dls = TabularDataLoaders.from_df(titanic_dl_df[tr_idx],
                                 procs=procs, 
                                 cat_names=cat_cols, 
                                 cont_names=num_cols,
                                 splits = splits,
                                 y_block = y_block,
                                 y_names=y_names                                 
                                )


# --------------------------
# ## Modeling
# 
# * Metric is determined by accuracy.
# * Set Patience to 5 and early stopping.
# * Sets a callback for stating the activation values.

# In[68]:


learn = tabular_learner(dls,layers=[16,32,32,16,8],
                        metrics=accuracy,
                        cbs = [EarlyStoppingCallback(monitor='accuracy', patience=5), 
                               ActivationStats(with_hist=True)])
learn.model


# ---------------------
# ## Finding the proper learning rate
# 
# ![](https://miro.medium.com/max/1200/1*Q-2Wh0Xcy6fsGkbPFJvMhQ.gif)
# 
# Picture Credit: https://miro.medium.com
# 
# As shown in the figure above, if the learning rate is selected too large, divergence occurs, and if the learning rate is selected too small, convergence is delayed. Learning rate is one of the important hyperparameters used in deep learning and has a great influence on performance. In general, there is a tendency to determine the learning rate through various experiments, but fast.ai provides a method to find the learning rate.
# 
# We decided to leave the troublesome task of determining the learning rate to the machine and focus on more creative work.

# In[69]:


sr = learn.lr_find()
sr.valley


# <span style="color:Blue"> Observation
#     
# Our fast.ai found an appropriate learning rate as shown in the figure above and gave us a guide.

# ---------------------
# ## Training

# In[70]:


learn.fit_one_cycle(100,sr.valley)


# ----------------------------------------------------------------
# ## Checking training results
# ![](https://miro.medium.com/max/474/1*wZg_RQHPRtn62dDp2Ez86A.jpeg)
# 
# Picture Credit:https://miro.medium.com

# In[71]:


learn.recorder.plot_loss()


# <span style="color:Blue"> Observation
#     
# Our model observed the loss values ​​of the train/valid dataset and found points that were not overfitting using early stopping.

# -----------------------------------------------------------
# ## Checking learning rate and momemtum scheduling
# ![](https://www.andreaperlato.com/img/momentum.png)
# 
# Picture Credit: https://www.andreaperlato.com
# 
# Momentum combines the direction from the gradient descent optimization algorithm obtained from the previous otimization procedure with the direction obtained from the current procedure to overcome the noisy gradient well.
# 
# If you look at the figure below, fast.ai finds an appropriate convergence point after increasing the learning rate gradually, lowering the learning rate again to find an appropriate learning rate. As opposed to learning, modem started with a large value and changed to a low value.
# 

# In[72]:


learn.recorder.plot_sched()
plt.subplots_adjust(wspace=0.5)


# ------------------------------------------------------
# ## Doing the fine tuning

# In[73]:


learn.recorder.fine_tune(epochs=10)


# -------------------------------------------
# ## Evaluating Model

# In[74]:


def plot_layer_stats(self, idx):
    plt,axs = subplots(1, 3, figsize=(15,3))
    plt.subplots_adjust(wspace=0.5)
    for o,ax,title in zip(self.layer_stats(idx),axs,('mean','std','% near zero')):
        ax.plot(o)
        ax.set_title(title)


# In[75]:


plt.style.use("dark_background")
plt.subplots_adjust(wspace=1)
plot_layer_stats(learn.activation_stats,-1)


# In[76]:


plt.style.use("dark_background")
plt.subplots_adjust(wspace=1)
plot_layer_stats(learn.activation_stats,-2)


# <span style="color:Blue"> Observation
#     
# The above two figures confirm the distribution of activation values for each layer. What you need to pay attention to here is to check whether the weight value is widely distributed close to zero. If many values are distributed at zero, it means that the learning is not done properly. When such a situation occurs, it seems that modeling and learning are necessary again.

# -------------------------------------------------
# ## Activations Histogram
# 
# ![](https://forums.fast.ai/uploads/default/original/3X/5/7/57a02a03d86a56561484aee9e88222ecbb7c1cf5.jpeg)
# 
# 
# <span style="color:Blue"> **Colorful Dimension**
# > The idea of the colorful dimension is to express with colors the mean and standard deviation of activations for each batch during training. Vertical axis represents a group (bin) of activation values. Each column in the horizontal axis is a batch. The colours represent how many activations for that batch have a value in that bin.
# 
# Ref: https://forums.fast.ai/

# In[77]:


learn.activation_stats.color_dim(-2)


# If you look at the above figure, you can see that the distribution of activation values ​​is evenly distributed. I think you can decide that the learning has been done properly.

# ----------------------------------------------------
# ## Predicting using test dataset

# In[78]:


titanic_dl_test = titanic_dl_df[~tr_idx]
titanic_dl_test = titanic_dl_test.drop('Survived',axis=1)


# In[79]:


dl = learn.dls.test_dl(titanic_dl_test)
preds = learn.get_preds(dl=dl)
results = preds[0].argmax(axis=1)
results = results.tolist()


# In[80]:


submission_data['Survived'] = results
#submission_data.to_csv('dl_submission.csv', index = False)


# <hr style="border: solid 3px blue;">
# 
# # Classic Machine Learning using Ensemble
# 
# ![](https://miro.medium.com/max/637/1*3GIDYOn2GNcv9bq4bQk5YA.jpeg)
# 
# Picture Credit: https://miro.medium.com
# 
# > Supervised learning algorithms perform the task of searching through a hypothesis space to find a suitable hypothesis that will make good predictions with a particular problem.Even if the hypothesis space contains hypotheses that are very well-suited for a particular problem, it may be very difficult to find a good one. Ensembles combine multiple hypotheses to form a (hopefully) better hypothesis. The term ensemble is usually reserved for methods that generate multiple hypotheses using the same base learner. The broader term of multiple classifier systems also covers hybridization of hypotheses that are not induced by the same base learner.
# 
# Ref: https://en.wikipedia.org/wiki/Ensemble_learning
# 
# Just as humans can have biased thinking, each model can also make biased predictions. Among the methods that do not make biased predictions while maintaining generality, ensemble is one of the best methods. Weak models will be able to obtain stable and better results while complementing each other's ideas.

# ## Setting up models
# 
# > This function trains and evaluates performance of all estimators available in the model library using cross validation. The output of this function is a score grid with average cross validated scores.
# 
# Ref: https://pycaret.readthedocs.io/en/latest/api/classification.html

# In[81]:


from pycaret.classification import *
clf1 = setup(data = titanic_df[tr_idx], 
             target = 'Survived',
             preprocess = False,
             numeric_features = all_cols,
             silent=True)


# ## Choosing top models
# 
# > This function trains and evaluates the performance of a given estimator using cross validation. The output of this function is a score grid with CV scores by fold. 
# 
# Ref: https://pycaret.readthedocs.io/en/latest/api/classification.html

# In[82]:


top5 = compare_models(sort='Accuracy',n_select = 5,
                      exclude = ['knn', 'svm','ridge','nb','dummy','qda','xgboost']
                     )


# ## Creating Models
# 
# > This function trains and evaluates the performance of a given estimator using cross validation. The output of this function is a score grid with CV scores by fold. 
# 
# Ref: https://pycaret.readthedocs.io/en/latest/api/classification.html

# In[83]:


catboost = create_model('catboost')
rf = create_model('rf')
lightgbm = create_model('lightgbm')
gbc = create_model('gbc')
mlp = create_model('mlp')
lr = create_model('lr')
dt = create_model('dt')


# # Interpreting Models
# 
# This function analyzes the predictions generated from a trained model. Most plots in this function are implemented based on the SHAP (SHapley Additive exPlanations).
# 
# > SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions
# 
# Ref: https://shap.readthedocs.io/en/latest/
# 
# **If you want to know more about feature importance and SHAP, please refer to the notebook below.**
# 
# [Notebook](https://www.kaggle.com/ohseokkim/explaning-machine-by-feature-importnace)

# In[84]:


with plt.rc_context({'figure.facecolor':'grey'}):
    interpret_model(catboost)


# In[85]:


with plt.rc_context({'figure.facecolor':'grey'}):
    interpret_model(rf)


# In[86]:


with plt.rc_context({'figure.facecolor':'grey'}):
    interpret_model(lightgbm)


# In[87]:


with plt.rc_context({'figure.facecolor':'grey'}):
    interpret_model(dt)


# <span style="color:Blue"> Observation:
# * Among the features, if you look at Fare and Age, the features are spread in a wide distribution of importance, and the colors are also spread from blue to red. 
# * Each model is learning with the importance of different features. The diversity of these models seems to increase the performance of the ensemble model.
# * Title_Mr and Sex_male play an important role in how the model learns.

# # Tuning Hyperparameters
# 
# > This function tunes the hyperparameters of a given estimator. The output of this function is a score grid with CV scores by fold of the best selected model based on optimize parameter. 
# 
# Ref: https://pycaret.readthedocs.io/en/latest/api/classification.html

# In[88]:


tuned_rf = tune_model(rf, optimize = 'Accuracy',early_stopping = True,search_library='optuna')
tuned_lightgbm = tune_model(lightgbm, optimize = 'Accuracy',early_stopping = True,search_library='optuna')
tuned_catboost = tune_model(catboost, optimize = 'Accuracy',early_stopping = True,search_library='optuna')
tuned_gbc = tune_model(gbc, optimize = 'Accuracy',early_stopping = True,search_library='optuna')
tuned_lr = tune_model(lr, optimize = 'Accuracy',early_stopping = True,search_library='optuna')
tuned_mlp = tune_model(mlp, optimize = 'Accuracy',early_stopping = True)
tuned_dt = tune_model(dt, optimize = 'Accuracy',early_stopping = True)


# **Multilayer perceptron (MLP)**
# 
# > A multilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN). The term MLP is used ambiguously, sometimes loosely to mean any feedforward ANN, sometimes strictly to refer to networks composed of multiple layers of perceptrons (with threshold activation). Multilayer perceptrons are sometimes colloquially referred to as "vanilla" neural networks, especially when they have a single hidden layer.
# 
# Ref: https://en.wikipedia.org/wiki/Multilayer_perceptron

# In[89]:


plt.figure(figsize=(10, 8))
with plt.rc_context({'figure.facecolor':'black','text.color':'white'}):
    plot_model(tuned_mlp, plot='learning')


# In[90]:


plt.figure(figsize=(8, 8))
with plt.rc_context({'figure.facecolor':'black','text.color':'black'}):
    plot_model(tuned_dt, plot='tree')


# <span style="color:Blue"> Observation:
#    
# * The greater the feature importance, the earlier the separation.

# In[91]:


plt.figure(figsize=(10, 8))
plot_model(tuned_dt, plot='learning')


# --------------------------------
# ## Stacking
# 
# ![](https://mlfromscratch.com/content/images/2020/01/model_stacking_overview-4.png)
# 
# Picture Credit: https://mlfromscratch.com

# In[92]:


stack_model = stack_models(estimator_list = [mlp,rf,lightgbm,catboost,gbc,lr], meta_model = catboost ,optimize = 'Accuracy')


# In[93]:


plt.figure(figsize=(8, 8))
plot_model(stack_model, plot='boundary')


# In machine learning, it is important to determine the boundary. In particular, in tree-based models, it is more important to determine the boundary, because the process of creating a new leaf in the tree is also the process of determining the boundary.
# Looking at the above picture again, there are many overlapping points with the green dot indicating the survior and the blue dot indicating the non-survior. Determining the boundary in this situation would be a very difficult task.
# If the feature engineer work was done well, the distribution of the two points to determine the boundary would have been well divided. However, the titanic dataset is difficult to do with some missing values and a small dataset.

# In[94]:


plt.figure(figsize=(8, 8))
plot_model(stack_model, plot = 'auc')


# In[95]:


plt.figure(figsize=(8, 8))
plot_model(stack_model, plot='confusion_matrix')


# ---------------------------------------------------
# ## Soft Voting
# 
# ![](https://miro.medium.com/max/806/1*bliKQZGPccS7ho9Zo6uC7A.jpeg)
# 
# Picture Credit: https://miro.medium.com
# 
# > This function trains a Soft Voting classifier for select models passed in the estimator_list param. The output of this function is a score grid with CV scores by fold.
# 
# Ref: https://pycaret.readthedocs.io/en/latest/api/classification.html

# In[96]:


blend_soft = blend_models(estimator_list = [mlp,rf,lightgbm,catboost,gbc,lr], optimize = 'Accuracy',method = 'soft')


# In[97]:


plt.figure(figsize=(8, 8))
plot_model(blend_soft, plot='boundary')


# It seems that the Boundary is set properly.

# In[98]:


plt.figure(figsize=(8, 8))
plot_model(blend_soft, plot = 'auc')


# In[99]:


plt.figure(figsize=(8, 8))
plot_model(blend_soft, plot='confusion_matrix')


# ---------------------------------------------------------------------------------------------------------------------------------------
# # Hard Voting
# 
# ![](https://miro.medium.com/max/428/1*XnZwlg7Th3nga25sSlanJQ.jpeg)
# 
# Picture Credit: https://vitalflux.com
# 
# 
# > This function trains a **Majority Rule classifier** for select models passed in the estimator_list param. The output of this function is a score grid with CV scores by fold.
# 
# Ref: https://pycaret.readthedocs.io/en/latest/api/classification.html

# In[100]:


blend_hard = blend_models(estimator_list = [mlp,rf,lightgbm,catboost,gbc,lr], optimize = 'Accuracy',method = 'hard')


# In[101]:


plt.figure(figsize=(8, 8))
plot_model(blend_hard, plot='boundary')


# Compared to the soft blending model, the boundary does not look clean.

# In[102]:


plt.figure(figsize=(8, 8))
plot_model(blend_hard, plot='confusion_matrix')


# -------------------------------------------------------------------------------------------------
# ## Calibrating the final model
# 
# > This function calibrates the probability of a given estimator using isotonic or logistic regression. 
# 
# Ref: https://pycaret.readthedocs.io/en/latest/api/classification.html

# In[103]:


cali_model = calibrate_model(blend_soft)


# --------------------------------------------------
# # Finalizing the last model
# > This function trains a given estimator on the entire dataset including the holdout set.
# 
# Ref: https://pycaret.readthedocs.io/en/latest/api/classification.html
# 
# The blend_soft model is selected based on the above result. Finally, the model is tuned with the entire dataset.
# 

# In[104]:


final_model = finalize_model(cali_model)


# ## Checking the final model

# In[105]:


plt.figure(figsize=(8, 8))
plot_model(final_model, plot='boundary')


# In[106]:


plt.figure(figsize=(8, 8))
plot_model(final_model, plot='confusion_matrix')


# ---------------------------------------
# # Checking Last Results

# Considering above results, the soft blending model seems appropriate among ensemble models. Therefore, we use this model to make the final prediction with the test dataset.

# In[107]:


last_prediction = final_model.predict(X_test_df)
submission_data['Survived'] = last_prediction.astype(int)
submission_data.to_csv('submission.csv', index = False)


# In[ ]:




