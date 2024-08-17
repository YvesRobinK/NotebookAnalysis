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


# In[2]:


get_ipython().system('pip install captum')
get_ipython().system('pip install pycaret-nightly')


# <hr style="border: solid 3px blue;">
# 
# # Introduction
# 
# ![](https://thumbs.gfycat.com/InexperiencedLongAmphibian-max-1mb.gif)
# 
# Picture Credit: https://thumbs.gfycat.com

# In this notebook, we will try to find out the differences between the classic machine learning method and the deep learning method through the titanic problem. In particular, I would like to focus on the deep learning method.

# <span style="color:Blue"> **Classical ML** 
# 
# > **Advantages:**
# > * More suitable for small data
# > * Easier to interpret outcomes
# > * Cheaper to perform
# > * Can run on low-end machines
# > * Does not require large computational power
# 
# > **Disadvantages:**
# > * Difficult to learn large datasets
# > * Require feature engineering
# > * Difficult to learn complex functions
# 
# <span style="color:Blue"> **Deep learning**
# 
# > **Advantages:**
# > * Suitable for high complexity problems
# > * Better accuracy, compared to classical ML
# > * Better support for big data
# > * Complex features can be learned
# 
# > **Disadvantages:**
# > * Difficult to explain trained data
# > * Require significant computational power

# For the Titanic problem, since the dataset size is small, classical ML seems to be more suitable.
# However, we will try to solve the titanic problem in two ways without any bias.
# 
# **However, please keep that in mind. The final choice is up to you.
# We look forward to making a wise choice for you.**

# -----------------------------------------------------------------------------------------------------------------------
# # Setting Up

# In[3]:


import numpy as np
import torch

from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

import matplotlib
import matplotlib.pyplot as plt

from scipy import stats
import pandas as pd
import re

from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import preprocessing
import umap
import umap.plot

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer


# In[4]:


train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
submission_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
titanic_df = pd.concat([train_data, test_data], ignore_index = True, sort = False)
tr_idx = titanic_df['Survived'].notnull()
titanic_df_copy = titanic_df.copy()


# ----------------------------------------------------------------------------------------------------------------------
# # Classic Machine Learning

# ![](https://miro.medium.com/max/1400/1*Da7wVx5j1KcSJ-I4DVFZyQ.png)
# 
# Picture Credit: https://miro.medium.com
# 
# Classic Machine Learning requires feature engineering. Also, with good feature engineering, the model can learn better. Therefore, when using classic ML, it is necessary to observe the dataset in detail and perform good preprocessing based on it.

# -----------------------------------------------------------------------------
# # Feature Engineering
# 
# > Feature engineering is the process of using domain knowledge to extract features (characteristics, properties, attributes) from raw data.
# > 
# > **The feature engineering process is:**
# > * Brainstorming or testing features;
# > * Deciding what features to create;
# > * Creating features;
# > * Testing the impact of the identified features on the task;
# > * Improving your features if needed;
# > * Repeat.
# 
# Ref: https://en.wikipedia.org/wiki/Feature_engineering
# 
# The below feature engineering process actually requires a lot of thought and time. The purpose of this notebook is to compare ML and DL, so we will omit the EDA process during the preprocessing process. If you are curious about the additional EDA process, please refer to the notebook below.
# 
# [titanic-missing-and-small-data-are-disaster](https://www.kaggle.com/ohseokkim/titanic-missing-and-small-data-are-disaster)

# -------------------------------------------------------------------------------------
# ## Preprocessing

# In[5]:


titanic_df['Has_Cabin'] = titanic_df['Cabin'].isnull().astype(int)
titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1

titanic_df['IsAlone'] = 0
titanic_df.loc[titanic_df['FamilySize'] == 1, 'IsAlone'] = 1

titanic_df['Cabin'] = titanic_df['Cabin'].fillna('N')
titanic_df['Cabin_label'] = titanic_df['Cabin'].str.get(0)

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

titanic_df['Title'] = titanic_df['Name'].apply(get_title)

titanic_df['Title'] = titanic_df['Title'].replace(
       ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 
       'Rare')

titanic_df['Title'] = titanic_df['Title'].replace('Mlle', 'Miss')
titanic_df['Title'] = titanic_df['Title'].replace('Ms', 'Miss')
titanic_df['Title'] = titanic_df['Title'].replace('Mme', 'Mrs')

titanic_df['Has_Age'] = titanic_df['Age'].isnull().astype(int)

imputer = KNNImputer(n_neighbors=2, weights="uniform")
titanic_df[['Age_knn']] = imputer.fit_transform(titanic_df[['Age']])

robuster = RobustScaler()
titanic_df['Age_knn'] = robuster.fit_transform(titanic_df[['Age_knn']])

titanic_df.drop(['Age'],axis=1,inplace=True)

transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
titanic_df['Fare'] = transformer.fit_transform(titanic_df[['Fare']])

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
titanic_df[['Fare']] = imp.fit_transform(titanic_df[['Fare']])

titanic_df['Fare_class'] = pd.qcut(titanic_df['Fare'], 5, labels=['F1', 'F2', 'F3','F4','F5' ])
titanic_df['Fare_class'] = titanic_df['Fare_class'].replace({'F1':1,'F2':2,'F3':3,'F4':4,'F5':5})

imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
titanic_df[['Embarked']] = imp.fit_transform(titanic_df[['Embarked']])


# --------------------------------------------------------------------------------------------------
# # Detecting Outliers by PCA
# 
# ![](https://miro.medium.com/max/920/1*v9VXKpHGnrtt96voOTt7nQ.gif)
# 
# Picture Credit: https://miro.medium.com
# 
# In the case of Classic ML, outliers must be identified and removed to improve performance. Therefore, outlier removal is an important process for classic ML.
# 
# The more features, the higher the dimension. When projecting to a lower dimension through PCA, new insights can be gained. PCA can effectively detect outliers.
# 
# PC 1 has the largest variance in the dataset distribution. That is, the outlier in PC 1 is very likely to be real outlier

# In[6]:


from sklearn.decomposition import PCA


# In[7]:


def apply_pca(X, standardize=True):
    # Standardize
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X)
    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    return pca, X_pca, loadings


# In[8]:


def outlier_iqr(data):
    q1,q3 = np.percentile(data,[25,75])
    iqr = q3-q1
    lower = q1-(iqr*1.5)
    upper = q3+(iqr*1.5)
    return np.where((data>upper)|(data<lower))


# In[9]:


def encode_features(dataDF,feat_list):
    for feature in feat_list:
        le = preprocessing.LabelEncoder()
        le = le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])
        
    return dataDF


# In[10]:


features = ["Sex","Age_knn","FamilySize","IsAlone",'Embarked','Cabin_label']
titanic_copy = titanic_df[tr_idx].copy()
y_copy = titanic_copy.pop("Survived")
X_copy = titanic_copy.loc[:, features]
encode_features(X_copy,['Sex', 'Embarked','Cabin_label'])
pca, X_pca, loadings = apply_pca(X_copy)
print(loadings)


# In[11]:


import plotly.express as px
fig = px.histogram(X_pca.melt(), color="variable", 
                   marginal="box",
                   barmode ="overlay",
                   histnorm ='density'
                  )  
fig.update_layout(
    title_font_color="black",
    legend_title_font_color="green",
    title={
        'text': "PCA Histogram",
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
)


# **Let's check out outliers in PC 1**

# In[12]:


pc1_outlier_idx = list(outlier_iqr(X_pca['PC1'])[0])


# In[13]:


component = "PC1"

def highlight_min(s, props=''):
    return np.where(s == np.nanmin(s.values), props, '')

train_data.iloc[pc1_outlier_idx,:].style.set_properties(**{'background-color': 'Grey',
                            'color': 'white',
                            'border-color': 'darkblack'})


# <span style="color:Blue"> Observation:
# * The Sage family started from S port, there was no Cabin, and the ages were not recorded, and they appear to be a poor and pitiful family with a pclass 3 rating.
# * All three females in this family using the Miss title have died.
# 
# **The last sad news is that the training dataset is small, so it seems difficult to remove even if the above data are outliers.    
# Nevertheless, fortunately, this problem is not a regression problem, but a classification problem. If it is a regression problem, outliers should be removed.**

# --------------------------------
# ## Encoding

# In[14]:


titanic_df = pd.get_dummies(titanic_df, columns = ['Title','Sex', 'Embarked','Cabin_label'],drop_first=True)


# ------------------------------------------------
# ## Selecting Features
# 
# In the case of classic ML, the process of selecting features is required.
# However, this process is not necessary for deep learning because important features are naturally selected through the erre backpropagation process.
# 
# 

# In[15]:


def drop_features(df):
    df.drop(['Name','Ticket','SibSp','Parch','Fare_class',
             'Cabin','Cabin_label_G','Cabin_label_T',
             'Cabin_label_F','FamilySize','Embarked_Q','Title_Rare','PassengerId'],
            axis=1,
            inplace=True)
    return df

titanic_df = drop_features(titanic_df)


# In[16]:


corr=titanic_df.corr().round(1)
sns.set(font_scale=1.15)
plt.figure(figsize=(14, 10))
sns.set_style("white")
sns.set_palette("bright")
abs(corr['Survived']).sort_values()[:-1].plot.barh()
plt.gca().set_facecolor('#FFFFFF')


# In[17]:


corr=titanic_df.corr().round(1)

sns.set(font_scale=1.15)
plt.figure(figsize=(14, 10))
sns.set_style("white")
sns.set_palette("bright")
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr,annot=True,cmap='Blues',mask=mask,cbar=True)
plt.title('Correlation Plot')


# In[18]:


sns.set(font_scale=2)
plt.figure(figsize=(14, 10))
sns.set_style("white")
sns.set_palette("bright")
sns.pairplot(titanic_df,kind = 'reg',corner = True,palette ='Blues',hue='Survived' )


# In[19]:


tr_idx = titanic_df['Survived'].notnull()
y_titanic_df = titanic_df[tr_idx]['Survived']
X_titanic_df= titanic_df[tr_idx].drop('Survived',axis=1)
X_test_df = titanic_df[~tr_idx].drop('Survived',axis=1)


# In[20]:


train_final = titanic_df[tr_idx]
all_cols = [cname for cname in X_titanic_df.columns]


# In[21]:


train_final.shape


# -----------------------------------------------------------------
# ## Visualizing Training Dataset after Dimension Reduction
# 
# Before training, let's draw a scaled down titanic dataset in 2D and 3D.

# In[22]:


mapper = umap.UMAP().fit(X_titanic_df) 
umap.plot.points(mapper, labels=y_titanic_df, theme='fire')


# In[23]:


umap.plot.connectivity(mapper, show_points=True)


# In[24]:


from umap import UMAP
import plotly.express as px

umap_3d = UMAP(n_components=3, init='random', random_state=0)
x_umap = umap_3d.fit_transform(X_titanic_df)
umap_df = pd.DataFrame(x_umap)
train_y_sr = pd.Series(y_titanic_df,name='label').astype(str)
print(type(x_umap))
new_df = pd.concat([umap_df,train_y_sr],axis=1)
fig = px.scatter_3d(
    new_df, x=0, y=1, z=2,
    color='label', labels={'color': 'label'},
    opacity=0.7
)
fig.update_traces(marker_size=1.5)
fig.show()


# If you enlarge or rotate the picture above, you can see how the preprocessed dataset is mapped in 3D. The preprocessed dataset is 17-dimensional, so we cannot visualize the preprocessed dataset.
# 
# However, it can be expected that clustering will be possible to some degree even when viewed in the above three-dimensional view.

# ----------------------------------------------------------------------------------------------
# # Modeling
# 
# ![](https://machinelearningknowledge.ai/wp-content/uploads/2019/12/Bagging-Bootstrap-Aggregation.gif)
# 
# Picture Credit: https://machinelearningknowledge.ai
# 

# In this notebook, we want to do classic ML using ensemble. Through ensemble, weak models learn differently to form collective intelligence. In this case, modeling is done using soft voting method using soft blending.
# 
# The ensemble learning method is observed in the learning method of DL in some way. DL will be explained later.

# ## Setting Up

# In[25]:


from pycaret.classification import *
clf1 = setup(data = train_final, 
             target = 'Survived',
             preprocess = False,
             numeric_features = all_cols,
             silent=True)


# ## Creating Models

# In[26]:


catboost = create_model('catboost')
rf = create_model('rf')
lightgbm = create_model('lightgbm')
gbc = create_model('gbc')
lda = create_model('lda')
lr = create_model('lr')


# ## Tuning Hyperparmaters

# In[27]:


tuned_rf = tune_model(rf, optimize = 'Accuracy',early_stopping = True)
tuned_lightgbm = tune_model(lightgbm, optimize = 'Accuracy',early_stopping = True)
tuned_catboost = tune_model(catboost, optimize = 'Accuracy',early_stopping = True)
tuned_gbc = tune_model(gbc, optimize = 'Accuracy',early_stopping = True)
tuned_lda = tune_model(lda, optimize = 'Accuracy',early_stopping = True)
tuned_lr = tune_model(lr, optimize = 'Accuracy',early_stopping = True)


# ## Interpreting Models

# In[28]:


interpret_model(lightgbm)


# In[29]:


interpret_model(catboost)


# In[30]:


interpret_model(rf)


# The operation of ensemble in ML can be thought of as using the diversity of other models as above. Looking at the above figures, different models have different feature importance. Let's also remember this picture for comparison with DL.

# ## Blending Models

# In[31]:


blend_soft = blend_models(estimator_list = [lr,rf,lightgbm,catboost,gbc,lda], optimize = 'Accuracy',method = 'soft')


# ## Calibrating Models

# In[32]:


cali_model = calibrate_model(blend_soft)


# ## Finalizing Model

# In[33]:


final_model = finalize_model(cali_model)


# In[34]:


plt.figure(figsize=(8, 8))
plot_model(final_model, plot='boundary')


# Most of the classic ML processes can be thought of as the process of determining the boundary as shown in the figure above. Keep this picture in mind for later comparison with the learning process of DL.

# In[35]:


plt.figure(figsize=(7, 7))
plot_model(final_model, plot='confusion_matrix')


# In[36]:


plt.figure(figsize=(8, 8))
plot_model(final_model, plot = 'auc')


# In[37]:


last_prediction = final_model.predict(X_test_df)
submission_data['Survived'] = last_prediction.astype(int)
submission_data.to_csv('submission.csv', index = False)


# <hr style="border: solid 3px blue;">
# 
# # Deep Learing
# 
# 
# ![](https://cdnb.artstation.com/p/assets/images/images/010/538/265/original/dane-vranes-datascience-optimized.gif?1524951477)
# 
# Picture Credit: https://cdnb.artstation.com
# 
# One of the advantages of DL is that the preprocessing process is simpler than that of classic ML.
# And, one of the disadvantages is that it is difficult to explain the model.
# In particular, this notebook summarizes the methods for explaining DL.
# 
# Compared to classic ML in DL, modeling and hyperparameter selection are more important than preprocessing.

# -----------------------------------------------------------
# ## Preprecessing for DL
# 
# Compared to Classic ML, the process of Feature Engineering is simple.
# 
# **In most cases, the following steps are required.**
# * Handling missing values
# * Encoding for categorical features (one-hot encoding is mainly used).
# * Standard or Min-Max Scaling

# In[38]:


def drop_features(df):
    df.drop(['Ticket','PassengerId','Cabin'],
            axis=1,
            inplace=True,
            errors='ignore')
    return df

titanic_dl_df = drop_features(titanic_df_copy)
titanic_dl_df['Pclass'] = titanic_dl_df['Pclass'].astype(object)


# In[39]:


def replace_name(name):
    if "Mr." in name: return "Mr"
    elif "Mrs." in name: return "Mrs"
    elif "Miss." in name: return "Miss"
    elif "Master." in name: return "Master"
    elif "Ms.": return "Ms"
    else: return "No"
titanic_dl_df['Name'] = titanic_dl_df['Name'].apply(replace_name)


# In[40]:


imputer = KNNImputer(n_neighbors=2, weights="uniform")
titanic_dl_df[['Age']] = imputer.fit_transform(titanic_dl_df[['Age']])

transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
titanic_dl_df['Fare'] = transformer.fit_transform(titanic_dl_df[['Fare']])

mean_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
titanic_dl_df[['Fare']] = mean_imp.fit_transform(titanic_dl_df[['Fare']])

freq_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
titanic_dl_df[['Embarked']] = freq_imp.fit_transform(titanic_dl_df[['Embarked']])


# --------------------------------------------
# ## Encoding and Scaling

# In[41]:


titanic_dl_df = pd.get_dummies(titanic_dl_df,drop_first=True)


# In[42]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
titanic_dl_df.loc[:,'Age':] = scaler.fit_transform(titanic_dl_df.loc[:,'Age':])


# In[43]:


train_data = titanic_dl_df[tr_idx]
test_data = titanic_dl_df[~tr_idx]


# In[44]:


test_data.drop('Survived',axis = 1,inplace=True)
test_data = test_data.to_numpy()


# ----------------------------------------------------------------------
# ## Separating Traing and Test Datasets

# In[45]:


# Set random seed for reproducibility.
np.random.seed(42)

# Convert features and labels to numpy arrays.
labels = train_data["Survived"].to_numpy().astype(bool)
train_data = train_data.drop(['Survived'], axis=1)
feature_names = list(train_data.columns)
data = train_data.to_numpy()

# Separate training and test sets using 
train_indices = np.random.choice(len(labels), int(0.8*len(labels)), replace=False)
test_indices = list(set(range(len(labels))) - set(train_indices))
train_features = data[train_indices]
train_labels = labels[train_indices]
val_features = data[test_indices]
val_labels = labels[test_indices]


# ------------------------------------------------------------------------
# ## Modeling
# 
# In neural networks, the model process is very important. An appropriate model should be designed according to the problem to be solved. In some cases, a well-trained model and related parameters are used. For the Titanic problem, we decided to use a simple model consisting of fully connected layers.
# 
# Dropout and Batch Normalization Layers are used here.

# ### Dropout
# 
# ![](https://miro.medium.com/max/1100/1*WPr205gm0CsQXGa0oltOew.gif)
# 
# Picture Credit: https://miro.medium.com
# 
# **Dropout**
# 
# > **Dilution (also called Dropout) is a regularization technique for reducing overfitting in artificial neural networks by preventing complex co-adaptations on training data.** It is an efficient way of performing model averaging with neural networks. The term dilution refers to the thinning of the weights.The term dropout refers to randomly "dropping out", or omitting, units (both hidden and visible) during the training process of a neural network. Both the thinning of weights and dropping out units trigger the same type of regularization, and often the term dropout is used when referring to the dilution of weights.
# 
# Ref: https://en.wikipedia.org/
# 
# Dropout ensures that only certain neurons are not trained, so that the diversity of the neural network is not reduced. In addition, it maintains generality by preventing the neural network from overfitting to a specific dataset.
# It seems safe to say that it creates an ensemble effect of a neural network.

# ### Batch Normalization Layers
# 
# ![](https://miro.medium.com/max/1548/1*1HNT2c2bAu37RgxNCCxFZw.gif)
# 
# Picture Credit: https://miro.medium.com
# 
# > Batch normalization (also known as batch norm) is a method used to make artificial neural networks faster and more stable through normalization of the layers' inputs by re-centering and re-scaling.
# 
# Ref: https://en.wikipedia.org/

# In[46]:


import torch
import torch.nn as nn
torch.manual_seed(1)  # Set seed for reproducibility.
class TitanicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(13, 13)
        self.BatchNorm1d1 = nn.BatchNorm1d(13, affine=False)
        self.drop1 = nn.Dropout(0.3)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(13, 13)
        self.BatchNorm1d2 = nn.BatchNorm1d(13, affine=False)
        self.drop2 = nn.Dropout(0.3)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(13, 13)
        self.BatchNorm1d3 = nn.BatchNorm1d(13, affine=False)
        self.drop3 = nn.Dropout(0.3)
        self.act3 = nn.ReLU()
        self.linear4 = nn.Linear(13, 8)
        self.BatchNorm1d4 = nn.BatchNorm1d(8, affine=False)
        self.drop4 = nn.Dropout(0.3)
        self.act4 = nn.ReLU()
        self.linear5 = nn.Linear(8, 8)
        self.BatchNorm1d5 = nn.BatchNorm1d(8, affine=False)
        self.drop5 = nn.Dropout(0.3)
        self.act5 = nn.ReLU()
        self.linear6 = nn.Linear(8, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lin1_out = self.linear1(x)
        lin1_out = self.BatchNorm1d1(lin1_out)
        lin1_out = self.drop1(lin1_out)
        act1_out = self.act1(lin1_out)
        lin2_out = self.linear2(act1_out)
        lin2_out = self.BatchNorm1d2(lin2_out)
        lin2_out = self.drop2(lin2_out)
        act2_out = self.act2(lin2_out)
        lin3_out = self.linear3(act2_out)
        lin3_out = self.BatchNorm1d3(lin3_out)
        lin3_out = self.drop3(lin3_out)
        act3_out3 = self.act3(lin3_out)
        lin4_out = self.linear4(act3_out3)
        lin4_out = self.BatchNorm1d4(lin4_out)
        lin4_out = self.drop4(lin4_out)
        act4_out4 = self.act4(lin4_out)
        lin5_out = self.linear5(act4_out4)
        lin5_out = self.BatchNorm1d5(lin5_out)
        lin5_out = self.drop4(lin5_out)
        act_out5 = self.act5(lin5_out)
        return self.softmax(self.linear6(act_out5))


# In[47]:


net = TitanicNet()
print(net)


# ----------------------------------------------------------------
# ## Training
# 
# ![](https://machinelearningknowledge.ai/wp-content/uploads/2019/10/Backpropagation.gif)
# 
# Picture Credit: https://machinelearningknowledge.ai
# 
# In the case of DL, the weight of each neuron (node) is adjusted through error backpropagation. This process is the most important part of the process of learning. Through this gradient descent process, we can understand which neurons play an important role and which features are important.

# In[48]:


criterion = nn.CrossEntropyLoss()
num_epochs = 2000

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
input_tensor = torch.from_numpy(train_features).type(torch.FloatTensor)
label_tensor = torch.from_numpy(train_labels)
for epoch in range(num_epochs):    
    output = net(input_tensor)
    loss = criterion(output, label_tensor.long())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print ('Epoch {}/{} => Loss: {:.2f}'.format(epoch+1, num_epochs, loss.item()))


# In[49]:


out_probs = net(input_tensor).detach().numpy()
out_classes = np.argmax(out_probs, axis=1)
print("Train Accuracy:", sum(out_classes == train_labels) / len(train_labels))


# In[50]:


val_input_tensor = torch.from_numpy(val_features).type(torch.FloatTensor)
out_probs = net(val_input_tensor).detach().numpy()
out_classes = np.argmax(out_probs, axis=1)
print("Test Accuracy:", sum(out_classes == val_labels) / len(val_labels))


# -----------------------------------------------------------------------------------------------------------------
# # Interpreting Neural Networks
# 
# ![](https://d1aueex22ha5si.cloudfront.net/Conference/1263/BackGround/brain-pic5_animated_synapses-1-1629371355924.gif)
# Picture Credit: https://d1aueex22ha5si.cloudfront.net
# 
# 
# **neural network/neurons**
# 
# > A neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes. Thus a neural network is either a biological neural network, made up of biological neurons, or an artificial neural network, for solving artificial intelligence (AI) problems. The connections of the biological neuron are modeled in artificial neural networks as weights between nodes. A positive weight reflects an excitatory connection, while negative values mean inhibitory connections. All inputs are modified by a weight and summed. This activity is referred to as a linear combination. Finally, an activation function controls the amplitude of the output. For example, an acceptable range of output is usually between 0 and 1, or it could be −1 and 1
# 
# Ref: https://en.wikipedia.org/wiki/Neural_network
# 

# Here we cover the following:
# * **Are features in neural networks important?**
# * **Which neuron plays an important role in a specific layer?**
# * **What are the distributional differences between important and non-significant neurons?**
# * **What are the important features for each neuron?**

# ---------------------------------------------------------------------------------------
# ## Which of the features were actually important to the model to reach this decision?
# 
# In this case, let's check what features are important for making a decision that the target value (Survived) is true in our designed neural network.

# In[51]:


ig = IntegratedGradients(net)
val_input_tensor.requires_grad_()
attr, delta = ig.attribute(val_input_tensor,target=1, return_convergence_delta=True)
attr = attr.detach().numpy()


# ## Visualizing Feature Importances

# In[52]:


def visualize_importances(feature_names, importances, title="Average Feature Importances", axis_title="Features"):
    x_pos = (np.arange(len(feature_names)))
    plt.figure(figsize=(12,6))
    sns.barplot(x_pos, importances)
    plt.xticks(x_pos, feature_names, rotation=90)
    plt.xlabel(axis_title)
    plt.title(title)
    sns.despine()
    
visualize_importances(feature_names, np.mean(attr, axis=0))


# From the feature attribution information, we obtain some interesting insights regarding the importance of various features.
# 
# <span style="color:Blue"> **Observation**:
#     
# * The Name_Mr and Sex_male features have a strong negative relationship and affect the survival rate.
# * The Fare feature has a strong positive relationship and affects the survival rate.
# * The effect on Age is not as big as I thought.
# * SibSp feature had little effect on survival rate.  

# ### Fare Attribution Distribution
# 
# Let's draw a distribution of Fare attribution that has a positive relationship with the survival rate.

# In[53]:


sns.histplot(attr[:,3]);
plt.title("Distribution of Fare Attribution Values");
sns.despine()


# <span style="color:Blue"> **Observation**:    
# * If you look at the distribution, you can see a long tail on the right.
# * This distribution is consistent with the fact that people who pay high rates are more likely to survive.

# ## Sex_male Attribution Distribution
# 
# Let's draw a distribution of Sex_male attribution that has a negative relationship with the survival rate.

# In[54]:


sns.histplot(attr[:,10]);
plt.title("Distribution of Sex_male Attribution Values");
sns.despine()


# <span style="color:Blue"> **Observation**:    
# * If you look at the distribution, you can see a long tail on the left.
# * This distribution is consistent with the fact that many of the Males died.

# ------------------------------------------------------------------
# ## Understanding the importance of all the neurons in the output of a particular layer
# 
# > An artificial neuron is a mathematical function conceived as a model of biological neurons, a neural network. Artificial neurons are elementary units in an artificial neural network.The artificial neuron receives one or more inputs (representing excitatory postsynaptic potentials and inhibitory postsynaptic potentials at neural dendrites) and sums them to produce an output (or activation, representing a neuron's action potential which is transmitted along its axon). Usually each input is separately weighted, and the sum is passed through a non-linear function known as an activation function or transfer function. The transfer functions usually have a sigmoid shape, but they may also take the form of other non-linear functions, piecewise linear functions, or step functions.
# 
# Ref: https://en.wikipedia.org/wiki/Artificial_neuron

# **In this case, we choose net.act1, the output of the first hidden layer.**

# In[55]:


cond = LayerConductance(net, net.act1)
cond_vals = cond.attribute(val_input_tensor,target=1)
cond_vals = cond_vals.detach().numpy()


# The first layer of our neural network has 13 neurons. 
# 
# **Let's visualize which neuron played an important role among the 13 neurons.**

# In[56]:


visualize_importances(range(13),np.mean(cond_vals, axis=0),
                      title="Average Neuron Importances",
                      axis_title="Neurons")


# <span style="color:Blue"> **Observation**:
#     
# * The importance of each neuron's survival rate varies with different magnitudes.

# **Let's plot the distributions of values ​​for all neurons in the first layer.**

# In[57]:


sns.set(font_scale = 1.5)
sns.set_style("white")
sns.set_palette("bright")
plt.figure(figsize=(20, 30))
plt.subplots_adjust(hspace=1)
for neuron in range(13):
    plt.subplot(7,2,neuron+1)
    mean = cond_vals[:,neuron].mean()
    std = cond_vals[:,neuron].std()
    sns.histplot(cond_vals[:,neuron],color='blue');
    plt.title(f"Neuron {neuron} : Mean = {mean:.2f}, Std = {std:.2f}")
    sns.despine()


# <span style="color:Blue"> **Observation**:
#     
# * The importance of each neuron's survival rate varies with different magnitudes.

# ----------------------------------------------------------------------------------
# ## Understanding what parts of the input contribute to activating a particular input neuron

# In[58]:


neuron_cond = NeuronConductance(net, net.act1)


# In[59]:


sns.set(font_scale = 1.5)
sns.set_style("white")
sns.set_palette("bright")
plt.figure(figsize=(20, 40))
plt.subplots_adjust(hspace=2)
for neuron in range(13):
    neuron_cond_vals = neuron_cond.attribute(val_input_tensor, neuron_selector=neuron, target=1)
    plt.subplot(7,2,neuron+1)
    x_pos = (np.arange(len(feature_names)))
    ax = sns.barplot(x_pos, neuron_cond_vals.mean(dim=0).detach().numpy())
    ax.set_xticks(x_pos, feature_names, rotation=90)
    ax.set_title(f"Average Feature Importances for Neuron {neuron}")
    sns.despine()


# Looking at the above figures, the importance of features that each neutron judges to be important are different. It looks similar to how weak models of classic ML's ensambles have different feature importance.
# 
# This diversity of neurons seems to be one of the factors that make DL strong. If each neuron does not have diversity, the performance of the neural network will be reduced. In this case, it seems that modeling needs to be done again.

# ---------------------------------------------------------------------------------------
# # Testing using Test Dataset

# In[60]:


test_input_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
out_probs = net(test_input_tensor).detach().numpy()
out_classes = np.argmax(out_probs, axis=1)
submission_data['Survived'] = out_classes
submission_data.to_csv('dl_submission.csv', index = False)


# In[61]:


submission_data


# # Conclusion
# 
# What choices did you make?
# 
# I think I will use the classic ML method for the titanic dataset. The reason for this judgment is as follows.
# * Titanic dataset is small in size. DL seems to learn generalized knowledge more effectively when the dataset is large.
# * There are many missing values in the Titanic dataset, so a feature engineering process to fill in the missing values through EDA seems to be necessary. DL requires little or no preprocessing, which is a common orthodoxy.
# 
# <hr style="border: solid 3px blue;">
