#!/usr/bin/env python
# coding: utf-8

# # 📜 Summary 📜
# 
# * This is an EDA of the [playground series data from season 3 episode 5](https://www.kaggle.com/competitions/playground-series-s3e5/). 
# * The data is synthetic and asks to predict **wine quality**, with a range of 0-10. With 10 being a high quality wine (note that our dataset only has values ranging from 3 to 8)
#  * Note: this is a Multi-class classification problem
# * The dataset for this competition (both train and test) was generated from a deep learning model trained on the original [**Wine Quality**](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset ) dataset 
#  * Feature distributions are close to, but not exactly the same, as the original i.e the **data is synthetic**
# * The metric required to be used by the competition is [**quadratic weighted kappa**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html) 

# # 🏠 Load libraries & data 🏠

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import log_loss, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RepeatedStratifiedKFold,StratifiedKFold
from scipy.stats import boxcox, median_abs_deviation
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, RFECV, SelectKBest
from sklearn.decomposition import PCA

import shap 
import lightgbm as lgb
import catboost as cat
import xgboost as xgb
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE


# ### Load data

# In[2]:


df_train = pd.read_csv("/kaggle/input/playground-series-s3e5/train.csv", index_col = 0)
df_test = pd.read_csv("/kaggle/input/playground-series-s3e5/test.csv", index_col = 0)
sub = pd.read_csv("/kaggle/input/playground-series-s3e5/sample_submission.csv",index_col = 0)


# ### Set Project Parameters
# Below we set parameters that we will use to re-run this notebook with different aspects 

# In[3]:


target = "quality" #Target column that we will be predicting, this is here for quick reference 

# Different scaling options
SCALING = True
DISTRIBUTION = True
OUTLIERS= True

# Feature engineering and model training options
ADD_DATA = True
THRESHOLD = 0.4 # For CV model removal
SMOTE_over = False
EPOCHS= 10000

# Notebook settings
sns.set_style("darkgrid")
pd.set_option('mode.chained_assignment',None)


# ### Additional data
# The synthetic data in the competition (df_train) was created from an original dataset. We can add this original data to our training data to (hopefully) improve model prediction
# * An additional column (is_generated) is added to show if this is the training or the additional data
# * We add this here to be included in our EDA

# In[4]:


if ADD_DATA:
    add_data = pd.read_csv('/kaggle/input/wine-quality-dataset/WineQT.csv', index_col = "Id")

#     df_train['is_generated'] = 1
#     df_test['is_generated'] = 1
#     add_data['is_generated'] = 0

    df_train = pd.concat([df_train, add_data],axis=0, ignore_index=True)
df_train


# # 📃 Basic Analysis 📃
# In the basic analysis section we want to get a feel of the data, the number of samples, features and target. We will refer back to this section as we explore the data more indepth

# In[5]:


print("Initial look at the data")
df_train.head()


# In[6]:


print("First 5 rows of Target")
display(df_train[target].head())


# In[7]:


print("Unique datatypes:\n",df_train.dtypes.unique())


# In[8]:


print("Column information:\n")
df_train.info()


# In[9]:


print("Statistical values of features\n")
df_train.describe()


# In[10]:


print("Null values: \nTrain:", df_train.isnull().sum().sum(), "\nTest:",df_test.isnull().sum().sum())


# In[11]:


print("Number of Duplicates in train:",df_train.duplicated().sum())
print("Number of Duplicates in test:",df_test.duplicated().sum())


# In[12]:


print("Duplicated targets:\n", df_train[df_train.duplicated()][target].value_counts())


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes on basic EDA:  </span></b>
#     
# * **No null values.** We therefore dont need to use imputation 
# * **Categorical data** ==> No Categorical data 
# * Data types are all float values excluding the target (integer)
# * Data is very small with only 2056 datapoints 
# * Using **df.describe()** is hard to understand the data with so many columns and without visualisation. We will refer back to this data in our visualisation section
# * **Duplicates**: We should drop duplicates as this will cause our model to overfit and give us overinflated scores as it will 'learn' the one duplicate and perfectly predict the other 

# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:red;font-size:1.2em;">Potential Solutions: </span></b>
#     
# #### 1. Categorical Encoding for categorical data (not required):
# * [Weight of Evidence]("https://contrib.scikit-learn.org/category_encoders/woe.html")
# * [Label encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) for ordinal data 
# * [onehotencoding](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html) for nominal data) 
# 
# #### 2. Null Value Imputation (not required):
# * [Sklearn Imputation]("https://scikit-learn.org/stable/modules/impute.html")
# 
# #### 3. Small Dataset:
#  * **Additional data** => this is possible by appending the orginal dataset
#  * **Synthetic data** ==> we can create additional data using Sampling techniques 
#      * Ovesampling with SMOTE: 
#          * I am wary of this as this data has already been created synthetically and synthetic data make reduce model preformance
#      * [Generative Adversarial Networks (GANS) for tabular data](https://towardsdatascience.com/how-to-generate-tabular-data-using-ctgans-9386e45836a6) -- this is computationally heavy and I wouldnt recommend within Kaggle 
#     
# #### 4. Duplicates
# * We will drop duplicates in the feature engineering section as we need to becareful that we dont remove too many samples as we already have a small dataset
#  

# ## 📈 Target Analysis 📈
# This is a Mutli-class classification problem, where we are expected to predict the quality of wine on a 8 rating scale.\
# From the below we can see that we only have wine quality starting from 3 --> 8
# 

# In[13]:


fig, ax = plt.subplots(figsize = (25,7))
sns.countplot(x= df_train[target])
plt.title("Wine Quality Count")

total = len(df_train)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='center')


# In[14]:


df_train[target].unique()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes Imbalanced dataset:  </span></b>
# 
# * We have certain wine qualities that are more prolific than others
# * We need to investigate our metric to understand how it is affected by imbalanced data 
# 
# <span style="color:red;font-size:1.2em;">Potential Solutions: Imbalanced Data  </span>
# 1. **Stratified Cross-validation** : we will try keep the same number of each target class in each fold. This should help the model understand the imbalance
# 2. **Class Weighting** : Certain models allow for class weightings to be added as parameters
# 3. **Additional Features** : Create features that help our model identify distingishing characteristics of certain classes
# 4. **Over /Under Sampling**: 
#  * [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) creates sythetic data by Oversampling minority classes
#      * Testing required as sythetic data could likely negatively impact score
#  * Undersampling: We can remove majority samples to even out the classes
#      * Due to the small number of samples, removing samples would be a poor choice as this reduces our dataset significantly

# # 🚀 Distribution & Skewness 🚀
# Why look at distribution:
# * Many ML algorithms are more accurate when training on data that is normally distributed. Specifically linear models like LDA, Gaussian Naive Bayes, Logistic Regression, Linear Regression as they are explicitly calculated from the assumption that the residuals are Normally (Guassian) distributed 
# * Note that Gradient boosting models can compute faster with normally distrubuted values but dont neccessarily improve with normally distributed data 
# 
# ##### **Normal Distribution (centre around the mean)**
# 
# ![image.png](attachment:531b83f0-a2dc-4e44-beaf-4502172231ba.png)

# In[15]:


numerical = [col for col in df_train.select_dtypes(["int64", "float64","int32", "float32"]).columns if col not in [target]]
categorical = [col for col in df_train.select_dtypes("object").columns if col!=target]
print("Numerical Columns:\n",numerical)
print("\n")
print("Categorical Columns:\n",categorical)


# #### Categorical Data (Count)
# Categorical data does not need to be normally distrubuted as it is discrete, we will therefore visualise the data with countplots/barplots to get a better indication of how it looks against the target

# Note: we dont have any categorical columns for this data, so the below is ignored 

# In[16]:


if len(categorical)>0:
    fig,ax = plt.subplots(4,2,figsize = (25,25))
    total = len(df_train)
    ax = np.ravel(ax)

    for i,col in enumerate(categorical):
        sns.countplot(ax = ax[i],x = df_train[col], hue=df_train[target])
        sns.countplot(ax = ax[i],x = df_train[col], hue=df_train[target])
        ax[i].tick_params(labelrotation=90)
        ax[i].set_title(f"{col}",fontsize = 12)
        ax[i].legend(title='Attrition', loc='upper right', labels=['No Attrition', 'Attrition'])
        ax[i].set(xlabel=None)

        for p in ax[i].patches:
            percentage = f'{100 * p.get_height() / total:.1f}%\n'
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax[i].annotate(percentage, (x, y), ha='center', va='center')

    fig.suptitle("Employee Attrition by categorical columns",fontsize = 20)
    plt.tight_layout()
    plt.show()


# ## Skewness & Kurtosis
# * Skewness is a measure of the lack of symmetry of the data ( how "off" the  data is from normal distrubution i.e its skewness)
#     * Skewness > 1 = highly positively skewed
#     * Skewness < -1 =  highly negatively skewed
#     * Skewness close to 0 = Normally distributed 
# * Kurtosis is a measure of whether the data is heavy-tailed or light-tailed relative to a normal distribution
# 
# ![image.png](attachment:73894680-f91e-413d-9676-ef98bc7b00fa.png)
# 
# #### **NB**: Hair et al. (2010) and Bryne (2010) argued that data is considered to be normal if skewness is between ‐2 to +2 and kurtosis is between ‐7 to +7

# In[17]:


# We concatenate test data to train to get a full view of the data as we know it
skew_df = pd.concat((df_train.drop(target,axis =1), df_test), axis =0).skew(numeric_only=True).sort_values()
print("Skewly distributed columns by skewness value:\n") 
display(skew_df)


# In[18]:


kurtosis_df = pd.concat((df_train.drop(target,axis =1), df_test), axis =0).kurtosis().sort_values()
print("Tailed columns by kurtosis value\n") 
display(kurtosis_df)


# In[19]:


fig,ax = plt.subplots(figsize=(25,7))

ax.bar(x = skew_df[(skew_df<2)& (skew_df>-2)].index, height = skew_df[(skew_df<2)& (skew_df>-2)], color = "g", label= "Semi-normal distribition")
ax.bar(x = skew_df[skew_df>2].index, height = skew_df[skew_df>2], color = "r", label = "Positively skewed features")
ax.bar(x = skew_df[skew_df<-2].index, height = skew_df[skew_df<-2], color = "b", label = "Negatively skewed features")
ax.legend()
fig.suptitle("Skewness of numerical columns",fontsize = 20)
ax.tick_params(labelrotation=90)


# In[20]:


fig,ax = plt.subplots(figsize=(25,7))

ax.bar(x = kurtosis_df[(kurtosis_df<7)& (kurtosis_df>-7)].index, height = kurtosis_df[(kurtosis_df<7)& (kurtosis_df>-7)], color = "g", label= "Semi-normal distribition")
ax.bar(x = kurtosis_df[kurtosis_df>7].index, height = kurtosis_df[kurtosis_df>7], color = "r", label = "Positive kurtosis")
ax.bar(x = kurtosis_df[kurtosis_df<-7].index, height = kurtosis_df[kurtosis_df<-7], color = "b",  label = "Negative kurtosis")
ax.legend()
fig.suptitle("Kurtosis of numerical columns",fontsize = 20)
ax.tick_params(labelrotation=90)


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes on Distribution (Skewness and Kurtosis)  </span></b>
#     
# * The degree of distibution of data further for a normal distribution will affect certain model performance (specifically linear models) 
# * As such we need to try correct this feattures that have positive / negative kurtosis and skewness
# 
# <span style="color:red;font-size:1.2em;">Potential Solution: Distribution (Guassian) Transformation </span>
# * Investigate Transformations to create normally distributed features (Log transform, Cube features, boxcox , Scaling etc)
#  * Specifically the columns: 'sulphates', 'residual sugar', 'chlorides' as these show high kurtosis and skewnes

# # 👨🏼‍🌾 Distribution Transformations: 👨🏼‍🌾
# * Log transform 
# * Scaling (Quantile) 
# * BoxCox 
# * Cube root / Square root

# In[21]:


# We note our columns that have non-normal distrubutions
non_dist_cols = ['sulphates', 'residual sugar', 'chlorides']


# In[22]:


fig,ax = plt.subplots(len(non_dist_cols),5, figsize = (30,15))
for i,col in enumerate(non_dist_cols):
    #scale
    scaler = QuantileTransformer(output_distribution="normal")
    quant_df = scaler.fit_transform(df_train[[col]])

    sns.histplot(x= df_train[col],ax= ax[i,0], color = "r")
    sns.histplot(quant_df,ax= ax[i,1] )
    sns.histplot(np.log1p(df_train[col]), ax = ax[i,2], color= "orange")
    try:
        sns.histplot(boxcox(df_train[col])[0], ax = ax[i,3], color= "orange")
    except:
        pass
    sns.histplot(np.sqrt(df_train[col]), ax = ax[i,4], color= "green")
    ax[i,0].set_title(f"Orginal ({col})")
    ax[i,0].set(xlabel=None)
    ax[i,1].set_title(f"Quantile Scaling ({col})")
    ax[i,2].set_title(f"Log transform ({col})")
    ax[i,2].set(xlabel=None)
    ax[i,3].set_title(f"Boxcox ({col})")
    ax[i,4].set_title(f"Square Root ({col})")
plt.suptitle("Distribution Transformations",fontsize = 30)
plt.tight_layout(pad = 4)
plt.show()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes:  </span></b>
#     
# * We are looking for the best process to create a normally distributed graphs
# * From the above we see that **QuantileScaling** is the best process for most of the features as this gives us a nice bell shaped curve. 
# 
# **NB** Quantile Scaling will always be a good scaling process for these distributions as we assume they are non-parametric 

# # 🧩 Correlation & Mututal Information  🧩
# * Features that contain similar (correlated / mutual) information negatively impact certain models as this can cause **overfitting** with linear models 

# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Note:  </span></b>
# 
# * Note that decision trees 'generally' arent affected by highly correlated features as they will choose the best feature that provides the highest information gain 
#  * Random forests could be afftected by highly correlated features as they might select the correlatd feature with less information gain 

# In[23]:


plt.figure(figsize = (25,12))

corr = pd.concat((df_train.drop(target,axis =1), df_test), axis =0).corr()
upper_triangle = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr,vmin = -1, vmax = 1, cmap = "Spectral", annot = True, mask = upper_triangle)
plt.title("Correlation of all features and target", fontsize= 18)
plt.show()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes on Correlation  </span></b>
#     
# * Features that contain similar (correlated / mutual) information negatively impact certain models as this can cause overfitting.
# * High correlation between a number of features  for example: 
#  * 'ph' and 'fixed acidity' have a large negative correlation 
#  * 'density' and 'fixed acidity' have a large positive correlation
# 
# 
# <span style="color:red;font-size:1.2em;">Potential Solutions: Correlation reduction</span>
#     
# This would only be needed for linear models as they will overfit with highly correlated features 
#  #### 1. Reduce Overfitting
#  * Linear models will be impacted negatively by correlated features, due to overfitting 
#   * use Lasso / Ridge models as these model regularize our feature inputs
#       * [General overfitting techniques ](https://medium.com/geekculture/how-to-stop-overfitting-your-ml-and-deep-learning-models-bb8324ace80b) 
# 
# #### 2. Dimensionality Reduction 
#  * Trial and error: **Remove columns** and run the model i.e. Drop highly correlated columns and rerun model 
#  * **Decomposition**  i.e. use Principal Component Analysis (PCA) on highly correlated features which removes correlation 
#  
#  
# ##### lets apply PCA and regularize our models with hyperparameter tuning and using specific model types

# ### 🎇 Pre-Training Feature Selection:  Mutual Information  🎇
# There are a number of processes and algorithms that can help us select features before we train our model. We will try Mutual Information. However a few others include:
# * [Recursive Feature Elimination](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV) 
# * [Model importance ](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel)
# * [Sequential Feature Selection ](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html#sklearn.feature_selection.SequentialFeatureSelector)
# 
# ### Mutual Information
# * Mutual information (MI) measures the dependecy of features to the target. 
# * Positive values are show a dependecy between the feature and the target i.e. the feature is important to the target and its predictions 
# * Values equal to zero shows independency to the target and are less important to target predictions 

# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Note:  </span></b>
#     
# We want features that have a large mutual information with our Target (i.e. positive values) and remove features with little to no mutual information. This **should** improve our model performance.
# 
# We will implement a mutual information test using Sklearn's [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html). This will apply our mutual information algorithm (to test for dependency) and select K highest features (Note: we will set k to be equal to all the features to visualise every column) 

# In[24]:


# feature selection
def select_features(df):
   df_trn = pd.get_dummies(df_train,drop_first =True) #onehotencoding needed for categorical columns 
   X_train, X_test, y_train, y_test = train_test_split(df_trn.drop(target,axis =1), df_trn[target], test_size=0.33, random_state=1) 
   
   fs = SelectKBest(score_func=mutual_info_classif, k='all')
   fs.fit(X_train, y_train)
   X_train_fs = fs.transform(X_train)
   X_test_fs = fs.transform(X_test)
   
   columns = X_train.columns
   return X_train_fs, X_test_fs, fs, columns 

# feature selection
X_train_fs, X_test_fs, fs, columns = select_features(df_train)


# In[25]:


selected_feats = pd.DataFrame(fs.scores_, index = columns).sort_values(by = 0 , ascending = True)

fig,ax = plt.subplots(figsize =(20,12))
plt.barh(y =columns, width= selected_feats[0])
plt.title("Mutual information of features to target")
plt.show()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes on Mutual information  </span></b>
# 
# * We can conclude that certain features are highly dependent on our Target(i.e. Alcohol). We should potentially look to apply feature engineering to these features, so as to further improve our model accuracy.  
# * There is little to no mutual information for 'fixed acidity' and 'volatile acidity' and could be potential for removal. However we will note these features and train our model to confirm feature importance below 

# # 🦠 Outliers / Distribution 🦠
# * Outliers will skew certain models and result in poor local optimums in trained models and reduce model performance. As such we need to assist our models in identifying outliers:
# * We will identify features with a large number of outliers through:
#     * Boxplots visualization
#     * zscore analysis (we calculate each samples standard deviation from the feature mean)

# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Note:</span></b>
#     
# * Decision Trees are robust towards outliers as they try best partition the data. Therefore outliers should be resolved for linear models 

# In[26]:


fig,ax = plt.subplots(int(np.ceil(len(numerical)/4)),4,figsize = (30,20))
ax = np.ravel(ax)

for i,col in enumerate(numerical):
    sns.boxplot(ax = ax[i], x = pd.concat((df_train.drop(target,axis =1), df_test), axis =0)[col], color= "blue")

fig.suptitle("Box plots of all data ",fontsize = 20)
plt.tight_layout(pad=3)
plt.show()


# In[27]:


from scipy.stats import zscore
df_zscores = pd.concat((df_train.drop(target,axis =1), df_test), axis =0)[numerical].apply(zscore)
print("Sample z-scores by feature:\n")
df_zscores.head()


# Lets visualise this better with boxplots of z-scores

# In[28]:


fig,ax = plt.subplots(int(np.ceil(len(numerical)/4)),4,figsize = (25,16))
ax = np.ravel(ax)

for i,col in enumerate(df_zscores.columns):
    sns.boxplot(ax = ax[i], x = df_zscores[col], color= "blue")
ax[i].set_title(f"{col}")
fig.suptitle("Z-scores ",fontsize = 20)
plt.tight_layout(pad=3)
plt.show()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes: Columns with outliers   </span></b>
# 
# * Columns with a large number of outliers will skew our models. 
# * From eye balling the above graphs we have a number of columns with outliers
#  * fixed acidity
#  * residual sugar
#  * chlorides
#  * Density
#  * ph 
#  * sulphates
# 
# <span style="color:red;font-size:1.3em;">Potential Solutions for Outliers</span>
# * Assist model in identifying outliers:
#  * **RobustScaler (scaling data):** This is a simple and effective method
#  * **Add column Z-score:** we wont do this as this will add another column of data that may negatively affect our model
# * **Outlier removal:** as stated above we dont want to reduce our sample size any further
#  * [Notebook on outlier removal ](https://www.kaggle.com/code/nareshbhat/outlier-the-silent-killer)

# In[29]:


outliers = ['fixed acidity', 'residual sugar', 'chlorides','pH' ,'sulphates']


# # 🦠Metric Understanding ([Quadratic Weighted Kappa](https://www.kaggle.com/competitions/playground-series-s3e5/overview/evaluation)) 🦠
# Quadratic weighted kappa also called [Cohen's Kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa) with Quadratic weightings, measures the degree of agreement between two raters i.e. the agreement between an models predictions and the target actuals, where the higher the agreement the higher the weight.
# 
# * The metric outputs values from -1 and 1, with 0 noting random agreement and 1 noting complete agreement
#  *  It is possible for the statistic to be negative. which can occur by chance if there is no relationship between the predictions and target
# * The formula is a complicated one such that we can rather use [sklearn's library](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html), with weights =  ‘quadratic’
# 
# 

# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes on Metric   </span></b>
#     
# * Luckily we have a fantastic notebook on this metric already which Ill note [here](https://www.kaggle.com/code/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps/notebook)
# * The most important aspects that I took from this notebook can be summed up as:
#     * ***all values lying on the diagonal are penalized the least with a penalty of 0, whereas predictions and true labels furthest away from each other are penalized the most***
#     * This part is referring the a confusion matrix of true lables vs predicted labels and essentially says that if you the prediction is further away from the actual value i.e. predict 8 and actual is 3. Then this will result in a worse score that if you predicted 4
# 
# <span style="color:red;font-size:1.3em;">Potential Solutions</span>
# * Prediction Analysis: Ensure that our predictions are close to one another
#     * If not: engineer features that will distinguish lower classes from higher classes 

# # 🎯 Feature Engineering 🎯
# * Feature engineering is the most complex part of data science as this requires certain domain knowledge as well as trial and error. 
# * This is usually an iterative process by adding/ removing features and training and analysing models 
# * We will leave leave this section bare for now 

# In[30]:


df_trn = df_train.copy(deep = True)
df_tst = df_test.copy(deep = True)


# In[31]:


# Drop duplicates
df_trn.drop_duplicates(inplace = True,ignore_index  = True)
print(df_trn.duplicated().sum())


# ## 🧬 Encoding 🧬
# 
# <span style="color:orange;font-size:1.5em;">Solution Implementation: Encoding </span>
# * Onehotencoding vs WOEencoder vs LabelEncoder
# 
# As our feautures are all numerical we can ignore encoding for this project

# In[32]:


if len(categorical)> 0:
    df_trn = pd.get_dummies(df_train,drop_first =True)
    df_tst = pd.get_dummies(df_test,drop_first =True)
df_trn.head()


# #### Fix target class: Wine Quality 
# The target has 6 classes however they start at 3. We need to change this to start at 0 as certain models will do this automatically or will fail as it expects the target to start from 0
# * We need to remember to revert the change when submitting our predictions

# In[33]:


#map the classes to start from 0
df_trn[target] = df_trn[target].map({3:0,
                    4:1,
                    5:2,
                    6:3,
                    7:4,
                    8:5})


# ## 🕵🏻Correlation reduction w/ PCA 🕵🏻
# PCA is a simple methodology to remove correlation from our dataset and as such should inhibit our models from overfitting

# In[34]:


# we will only look at certain features that are highly correlated and run pca  as incorrectly running PCA on the wrong columns could decompose important features. This is therefore a trial and error exercise 

pca_cols = ["pH","fixed acidity"]
pca_ = PCA(n_components=1 ,whiten= False)
df_trn["pca_1"] = pca_.fit_transform(df_trn[pca_cols])
df_tst["pca_1"] = pca_.fit_transform(df_tst[pca_cols])

for cols in pca_cols:
    for df in [df_trn,df_tst]:
        df.drop(cols, axis =1, inplace = True)

df_trn


# In[35]:


plt.figure(figsize = (25,12))
corr = pd.concat((df_trn.drop(target,axis =1), df_tst), axis =0).corr()
upper_triangle = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr,vmin = -1, vmax = 1, cmap = "Spectral", annot = True, mask = upper_triangle)
plt.title("Relook: Correlation of all features and target", fontsize= 18)
plt.show()


# ## Split

# In[36]:


X = df_trn.drop([target],axis =1)
y = df_trn[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# # 🧊 Basic Modelling 🧊
# We run a basic model for testing in order to:
# * Quickly confirm any changes made to our code 
# * Quickly check if additional features/processes cause large changes in our models score 
# * Evaluate feature importance
# 
# Note: I wont be scaling the values here as it makes it hard to understand our feature values in visualizations later on

# #### Below we create a custom metric function for models 

# In[37]:


# create a custom metric for lightgbm
def kappa_score(dy_true, dy_pred):
    pred_labels = dy_pred.reshape(len(np.unique(dy_true)),-1).argmax(axis=0)
    
    ks = cohen_kappa_score(dy_true, pred_labels, weights ='quadratic' )
    is_higher_better = True

    return "kappa_score", ks, is_higher_better


# In[38]:


# Note: scale_pos_weight is total majoriity / total minority class ==1477/200
# class_pos_weight =   (len(df_trn) - sum(df_trn[target]))/sum(df_trn[target])
# print("scale_pos_weight:",class_pos_weight)

print("skew columns:\n", non_dist_cols)
outliers = [col for col in outliers if col not in non_dist_cols + pca_cols]
print("outliers:\n", outliers)
scaled_cols = [col for col in df_trn.drop(target,axis =1).columns if col not in outliers + non_dist_cols+pca_cols]
print("scaled_cols:\n", scaled_cols) 

if len(scaled_cols)==0:
    SCALING = False


# State our base model paramaters 

# In[39]:


lgb_params ={'objective': 'multiclassova',#multiclassova multiclass
             "metric":"multi_logloss", 
             "boosting": "gbdt",#"dart",gbdt
             'num_class': 6,
             'is_unbalanced' : True,
#              'lambda_l1': 1.0050418664783436e-08, 
#              'lambda_l2': 9.938606206413121,
#              'num_leaves': 44,
              'feature_fraction': 0.8247273276668773,
              'bagging_fraction': 0.5842711778104962,
#             'bagging_freq': 6,
#              'min_data_in_leaf': 134,
#              'min_child_samples': 70,
#              'max_depth': 8,
               'class_weight': 'balanced', #'balanced',     weights     
             'n_estimators':EPOCHS,
             'learning_rate':0.01,
            'device':'cpu'}


# In[40]:


model= lgb.LGBMClassifier(**lgb_params)

X_train_s = X_train.copy(deep= True)
X_test_s = X_test.copy(deep= True)
test_temp = df_tst.copy(deep= True)
y_train_s = y_train.copy(deep= True)
X_temp = X.copy(deep = True)

model.fit(X_train_s,y_train_s,
          eval_set=[(X_test_s,y_test)],
          eval_metric=kappa_score,
          callbacks= [lgb.log_evaluation(-1), lgb.early_stopping(30)])

y_preds = model.predict_proba(X_test_s)
trn_preds_base = model.predict_proba(X_temp)
test_base = model.predict_proba(test_temp)

score_trn =cohen_kappa_score(y, trn_preds_base.argmax(axis =1), weights ='quadratic' )
score_val = cohen_kappa_score(y_test, y_preds.argmax(axis =1), weights ='quadratic' )

print("\nTrn kappa:",score_trn)
print("Val kappa:",score_val)
print("Val logloss", log_loss(y,trn_preds_base))


# In[41]:


fig, ax = plt.subplots(1,2, figsize = (30,7))
sns.countplot(x =test_base.argmax(axis =1)+3, label = f"baseline: test preds", color ='green', ax= ax[0])
sns.countplot(x =trn_preds_base.argmax(axis =1)+3, label = f"baseline: train preds", color ='red', ax= ax[1])
sns.countplot(x =y+3, label = f"Actual train",alpha =0.5, color ='blue', ax= ax[1])
ax[0].legend()
ax[1].legend()
plt.suptitle("Predictions vs actual", fontsize = 20)
plt.show()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes:</span></b>
#     
# * The model is overpredicting 7 and under prediction 5 and 6
# * Can we improve this with some additional features. Lets look at the feature importances and shap values to get some ideas

# # 📈 Partial Dependency Plots 📈
# * Partial dependecy plots show how changes in certain features inmpact the classification 
# * This can give us information on additional features and how our models classifies our training data

# In[42]:


from pdpbox import pdp, get_dataset, info_plots
pdp_dist = pdp.pdp_isolate(model=model, dataset=X_train_s
                           , model_features=X_train_s.columns
                           , feature='sulphates')
pdp.pdp_plot(pdp_dist, feature_name='sulphates',figsize =(25,20))
plt.tight_layout()
plt.show()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes: </span></b>
#     
# * From the above graphs we see how the target probability (y axis) changes as the values of 'sulphates' changee (note that they are scaled)
# * If we want to better predict **Class 3 and Class 2**, we can try create a feature that notes if "sulphates" are around 0.6. As this causes a spike in probability for Class 3 and a drop for Class 2

# # ☘️ Feature importance ☘️
# 
# ## 1. Intrinsic model feature importance
# * As we are using lightgbm as our base model we can get an output of the models feature importance. 

# In[43]:


importances = pd.DataFrame(data = model.feature_importances_, index = model.feature_name_).sort_values(ascending = False , by =0)

plt.figure(figsize = (20,15))
sns.barplot(x = importances[0], y= importances.index)
plt.show()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes:</span></b>
#     
# * **Personal opinion**: intrinsic feature importance of tree based models are poor determinants of the **level of importance** as it looks at information gain. This calculation determines how the data is partitioned into a target value and doesnt neccessarily the determine level of importance 
#  * Feature importance is helpful noting **0 important features** as these features werent included in the prediction at all and can be **dropped** 
#  * I would ignore the level of importance to a certain degree 

# ## 2. Shap values 
# [Shap values](https://shap.readthedocs.io/en/latest/index.html) are a good indicator of the **degree of feature importance** 
# * We use multiple visualization methods to assists when plotting shap values from our fitted model
# 
#  
# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes on Shap Values   </span></b>
#     
# Shape values determine how the prediction was impacted. A **High** Shap value means that the predictive was influenced positively:
# i.e. if the output is a probability from 0 to 1
# * A high (postive) shap value will increase the prediction towards 1 (denoted as **RED** color)
# * A low (negative) shap will decrease the prediction towards 0 (denoted as **BLUE** color

# In[44]:


shap.initjs() # for visualization 

#Get Shap values
explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X_test_s)


# <span style="color:orange;font-size:1.8em;">Target Interaction (by shap feature importance) </span>
# 
# ### Summary Plot
# The shap summary plot gives us an indication of feature importance. We need to pay specific attention to features with **high** average shap values as well as those that are zero:
# * Zero valued features are good candidates for removal 
# * High average shap values are good candidates for feature engineering. We can get ideas for feature engineering by viewing **shap.dependeceplots** and **shap.force_plots** (see below) to see the relationship of features

# In[45]:


shap.summary_plot(shap_values, X_test_s,max_display = 400, plot_size = [20,10], show= False)
plt.title("Average Shap values across all samples", fontsize = 20)
plt.show()


# <div class='alert alert-block alert-success'><b><span style="color:blue;font-size:1.2em;"> </span></b>
#     
# **NB:** remember to add 3 to each class as we dont start at 0 (we start at class 3)

# In[46]:


# We can get the shap values for each class by indexing the shap_values i.e shap_values[0] ==Class 0 (remember we need to add 3 to our classes)
fig = plt.figure()
for i in range(len(df_trn[target].unique())):
    ax = fig.add_subplot(231+i)
    shap.summary_plot(shap_values[i], X_test_s,max_display = 400, show= False, color_bar = False)
    ax.set_title(f"Class {i+3}", fontsize =12)
    
plt.gcf().set_size_inches(30,20)
plt.tight_layout()
plt.show()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes: </span></b>
#     
# * Certain features may have zero average shap value. As such they dont play a part in the models prediction. 
#     * We can test our model again without these features to see if there is a model improvement (i.e. drop  column)
# * Additionally, the graphs also give us ideas on new features we can create:
#     * i.e. Sulphates is important to all Classes and we can see in the "Class 7" graph that the "points" under sulphates are red as the shap value increases 
#         * Can we create a cutoff or filter feature that lets the model know when the sulphates are above a certain level i.e. sulphates > 1.0. This may help it classify Class 7 better
# 

# ### Force Plot
# The forceplot above looks at a number of samples and shows how certain (important) features influenced the final model prediction i.e. How these features push the model to predict the target of 0 or 1

# In[47]:


shap.force_plot(explainer.expected_value[1], shap_values[0][0:1,:], X_test_s.iloc[0:1,:])


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes: </span></b>
#     
# ##### This is plot of 1 sample within our dataset for CLASS 3: 
# * The **Output value**  is the bold value and is the Shapley value for this sample. It is **negative** so it means that the model is pushing towards a negative result (i.e. towards 0 == not likely to be  Class 3) Note: a positve value would push the model towards 1 (highly likely to be Class 3)
#     * This **Output** value is the "raw" value which is then transformed into probability space, to give you the final probability between  0 and 1 (i.e. negative shap means < 0.5 probabiltity).
# * The base value:
#  * This is the value that would be predicted if we didn’t know any features for the current instance. The base value is the average of the model output over the training dataset
# * All **Blue** colored features have a **negative** influence on the models prediction and all **Red** colored features have a **postive** influence. 
#  * The Features **closest** to the **Output value** have a higher degree of influence on the prediction (i.e the shap value) 
#  * The value associated with each feature shows the sample value (e.g. if 'free sulpher dioxide' = -1.143 that is the value for this sample in the dataset)  
# 
# ##### This sample is showing that our model is predicting that this samples is not likely to be Class 3

# <div class='alert alert-block alert-success'><b><span style="color:blue;font-size:1.2em;"> </span></b>
# Below is a force_plot of ALL the samples, use the dropdowns on the x and y axis to filter by specific features

# In[48]:


shap.force_plot(explainer.expected_value[1], shap_values[0], X_test_s)


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes: </span></b>
#     
# * If we set features on the x and y axis. We will see all the shap values for Class 3. i.e. how our model predicts class 3 and its interaction by the features 
#  * We can potentially create a new feature from these two features
#  
#  
#  ##### Ill leave it to you to create and investigate force_plots for all the classes 

# <span style="color:orange;font-size:1.8em;">Feature Interaction / Feature Dependency </span>
#     
# Below we look at the interaction and relationship **between features** 
# 
# #### Interaction Values
# Shap Interaction values are the fantastic as visualizing how features affect others in relation to the models prediction 

# In[49]:


# Get interaction values
shap_interaction_values = explainer.shap_interaction_values(X_test_s)


# ### 1. Heatmap plot of shap interaction values 
# 
# Taken from [analysing interactions with shap](https://towardsdatascience.com/analysing-interactions-with-shap-8c4a2bc11c2a):
# 1. To start we will calculate the absolute mean for each cell across all matrices. 
# 2. We take the absolute values as we do not want positive and negative SHAP values to offset each other. 
# 3. Because the interaction effects are halved, we also multiply the off diagonals by 2. 
# 4. We then display this as a heatmap.
# 
# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Note: </span></b>
# As this is a multiclass classification problem we have interaction values for each class i.e. 12 columns = 12 shap interaction values 

# #### 1a Absolute mean shap interactions 
# * We get the **absolute mean** of the shap interactions 
#  * the absolute of the values gives us the magnitude of the interaction and includes both negative and postive probability (i.e. adds them together as positives) 
#  * if we want the overall result , we must remove the absolute calculation and only calculate the mean shap interactions 

# In[50]:


# Get absolute mean of matrices
abs_mean_shap = np.abs(shap_interaction_values).mean(axis =1)


# In[51]:


fig,ax = plt.subplots(3,2, figsize = (30, 30)) 
ax = np.ravel(ax)

for i in range(len(abs_mean_shap)):
    df = pd.DataFrame(abs_mean_shap[i], index=X_test_s.columns, columns=X_test_s.columns)
    df.where(df.values == np.diagonal(df),df.values*2, inplace=True)
    sns.heatmap(df.round(decimals=3), cmap='coolwarm', annot=True, fmt='.6g', cbar=True, ax=ax[i]   )
    
    ax[i].tick_params(axis='x', labelsize=10, rotation=90)
    ax[i].tick_params(axis='y',  labelsize=10)
    ax[i].set_title(f"Class {i+3}")
plt.suptitle("Absolute Mean Shap Interaction Values (Magnitude of effect on model predictions)",fontsize=15)
plt.tight_layout(pad = 4)
plt.show()


# #### 1b Mean shap interactions 
# * Gives us the feature interaction value that and the overall push that these features have on the models prediction i.e. red = high probability of class 

# In[52]:


# Get absolute mean of matrices
mean_shap = np.array(shap_interaction_values).mean(axis =1)


# In[53]:


fig,ax = plt.subplots(3,2, figsize = (30, 30)) 
ax = np.ravel(ax)

for i in range(len(mean_shap)):
    df = pd.DataFrame(mean_shap[i], index=X_test_s.columns, columns=X_test_s.columns)
    df.where(df.values == np.diagonal(df),df.values*2, inplace=True)
    sns.heatmap(df.round(decimals=3), cmap='coolwarm', annot=True, fmt='.6g', cbar=True, ax=ax[i]   )
    
    ax[i].tick_params(axis='x', labelsize=10, rotation=90)
    ax[i].tick_params(axis='y',  labelsize=10)
    ax[i].set_title(f"Class {i+3}")
plt.suptitle("Mean Shap Interaction Values (Overall effect on model predictions)",fontsize=15)
plt.tight_layout(pad = 4)
plt.show()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes: </span></b>
#     
# * from the above graphs we can see that there is a large interaction between 'sulphates' and 'alcohol' 
#  * We could create a new feature to exploit this i.e. sulphates x alcohol

# #### Dependence plots
# * Dependence plots require a lot of investigation and testing. We look at the relationship of 2 features and how they influence the **shap values** and **shap interactions**  (i.e. the model output) 
# * Below I took a feature with a high average Shap value (see summary plot i.e. alcohol) and look at its relationship with features that interact highly with it (see heatmaps above). This will hopefully allow us to find any additional feature we can create (i.e clip values, multiply/Add/Subtract features , polynomial features etc) 

# In[54]:


fig = plt.figure()
ax0 = fig.add_subplot(2,2,1)
shap.dependence_plot("total sulfur dioxide", shap_values[4], X_test_s, display_features=X_test_s, interaction_index="alcohol", show=False,ax= ax0)
ax0.set_title("Class 8: 'total sulfur dioxide' vs 'alcohol'  (Shap Values)" )
ax1 = fig.add_subplot(2,2,2)
shap.dependence_plot( ( "total sulfur dioxide","alcohol"),  shap_interaction_values[4], X_test_s, display_features=X_test, show=False, ax= ax1)
ax1.set_title("Class 8: 'total sulfur dioxide' + 'alcohol' (Shap Interactions)" )

ax2 = fig.add_subplot(2,2,3)
shap.dependence_plot("sulphates", shap_values[4], X_test_s, display_features=X_test_s, interaction_index="alcohol", show=False,ax= ax2)
ax2.set_title("Class 8: 'sulphates' vs 'alcohol' (Shap Values)" )
ax3 = fig.add_subplot(2,2,4)
shap.dependence_plot(  ("sulphates","alcohol"),  shap_interaction_values[4], X_test_s, display_features=X_test, show=False, ax= ax3)
ax3.set_title("Class 8: 'sulphates' + 'alcohol' (Shap Interactions)" ) 

plt.gcf().set_size_inches(25,12)
plt.tight_layout()
plt.show()


# In[55]:


fig = plt.figure()
for i in range(1,7):
    axes = fig.add_subplot(2,3,i)
    shap.dependence_plot("sulphates", shap_values[4], X_test_s, display_features=X_test_s, interaction_index=i, show=False,ax= axes)
plt.gcf().set_size_inches(30,15)
fig.suptitle("Shap Dependency plots for sulphates vs other features for CLASS 7", fontsize = 20)
plt.tight_layout(pad =3)
plt.show()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;"> Notes: </span></b>
# 
# ##### We want to isolate High/Low  Shap values and clumps of color
# * As Sulphates increase this results in a high Shap value (and as such a high probability of Class 7) 
# * There doesnt seem to be any interaction affect of the above features with sulphates (i.e. no clumping of color) 
#     
# ##### We consequently dont get much from these dependency plots other than noting that higher sulphates will result in a higher chance of class 7 and class 8
#     * We could try create a new column by **binning** the sulphates values into key thresholds that will help our model classify 7 and 8
#          * i.e. sulphates < 0.8 and sulphates >0.8

# # 🧊  Muti-Model Testing 🧊 
# In this section we will test multiple models. This will allow use to understand which model would be best to optimize as well as allow use to ensemble models once optimized (note that we dont optimize the models in this notebook) 
# The models we test should be quite broad however we can lump these into 3 categories:
# * Linear i.e. Logistic regression for general linear modelling, Ridge when we dnt have optimal features selected (Ridge regularizes features) and Support Vector Machines for additional kernalization
# * Trees i.e. gradient boosting such as LightGBM and XGBoost with ensemble trees convered by Random Forests and Extra Trees  
# * Neural networks (see below)
# 
# **On Neural network exclusion**:\
# As this is a tabular problem (not an image or text based) I will assume that neural networks wont do as well for this problem. This is also quickly apparent when looking at the data which is not homogenous in nature (i.e similar distributions, scale and source) this is usually an indicator that Neural networks wont do well with this problem 
# Also due to their long run time and complexity I will note include here 
# 

# <span style="color:orange;font-size:1.5em;">Solution Implementation: Skewness,  Outliers , Scaling & Imbalanced data </span>
# * **Quantile Scaling** for non-stributed features should normalise our data  
# * **Robust Scaling** for outlier identification
# * **Standard Scaler** for linear models to assist convergence, which we will apply to all the other columns that arent outliers or skew 
# * **SMOTE** to oversample our data as our classes are imbalanced 
# 
# ##### Note: At the beggining of this notebook we set the paramaters to turn the above on or off. This is for testing purposes i.e. run this notebook with different paramaters to test model improvement

# In[56]:


def Scaling(X_train, X_test, test_df,  y_train, X= None,) : 
    
    """Scaling and Sampling Helper function: 
        Scales and oversamples training and validation dataframes 
    
    :param 
        X_train: pandas dataframe of training data , less target values
        X_test: pandas dataframe of validation data , less target values
        test_df: pandas dataframe of test data , less target values
        y_train: pandas dataframe or series of target training values
        X: pandas dataframe of training and validation data, less target values
        
    :return: scaled input paramaters as pandas dataframes 

    """
    
    test_s = test_df.copy(deep = True)
    X_train_s = X_train.copy(deep = True)
    X_test_s = X_test.copy(deep = True)
    
    if X is not None:
        X_s = X.copy(deep = True)
    
    if OUTLIERS and len(outliers)>0:
        #Scale outliers: see boxplots
        scaler = RobustScaler()
        X_train_s[outliers] = scaler.fit_transform(X_train_s[outliers])
        X_test_s[outliers]  = scaler.transform(X_test_s[outliers])
        test_s[outliers] = scaler.transform(test_s[outliers])
        if X is not None:
            X_s[outliers] = scaler.transform(X_s[outliers])
    
    if DISTRIBUTION:
        #Scale Skewness: see distribution
        scaler = QuantileTransformer(output_distribution="normal")
        X_train_s[non_dist_cols] = scaler.fit_transform(X_train_s[non_dist_cols])
        X_test_s[non_dist_cols] = scaler.transform(X_test_s[non_dist_cols])
        test_s[non_dist_cols] = scaler.transform(test_s[non_dist_cols])
        if X is not None:
            X_s[non_dist_cols] = scaler.transform(X_s[non_dist_cols])
            
    if SCALING: 
        scaler = StandardScaler()
        X_train_s[scaled_cols] = scaler.fit_transform(X_train_s[scaled_cols])
        X_test_s[scaled_cols] = scaler.transform(X_test_s[scaled_cols])
        test_s[scaled_cols] = scaler.transform(test_s[scaled_cols])
        if X is not None:
            X_s[scaled_cols] = scaler.transform(X_s[scaled_cols])
            
    if SMOTE_over:
        smt = SMOTE()
        X_train_s, y_train_s = smt.fit_resample(X_train_s, y_train)
    else:
        y_train_s = y_train.copy(deep= True)
              
    if X is not None:
        return pd.DataFrame(X_train_s, columns = X.columns )   , pd.DataFrame(X_test_s, columns = X.columns ) , pd.DataFrame(test_s, columns = test_df.columns ) , pd.DataFrame(X_s, columns = X.columns ),  y_train_s  
    else:
        return pd.DataFrame(X_train_s, columns = X.columns ), pd.DataFrame(X_test_s, columns = X.columns )  , pd.DataFrame(test_s, columns = test_df.columns ), y_train_s


# Due to an imbalanced dataset we want to create a dictionary of all our class weights and pass this to models that allow for this. Note that we use this for catboost only as the other models work better with the "balanced" parameter

# In[57]:


weights = df_trn[target].value_counts().sort_index()
weights = weights.astype(float)
for i in range(len(weights)):
    #weights[i] = len(df_trn) / (len(weights) * weights[i])
    weights[i] = 1-  (weights[i] / len(df_trn))
weights= weights.to_dict()
print("Catboost class weights:\n")
weights


# In[58]:


xgb_params = { 
    'objective' : "multi:softproba",
    'num_class' : 6,
    'n_estimators' : EPOCHS, 
    'early_stopping_rounds' :30,
    #'custom_metric':kappa_score,
    #'scale_pos_weight':weights
             }
cat_params = {'iterations':EPOCHS,
               'class_weights' : weights,
              #'eval_metric' : kappa_score,  ## Need to fix this doesnt work 
              'learning_rate': 0.01,
              'loss_function':'MultiClass'
             }
ET_params = {'max_depth':6, 'num_iterations':EPOCHS}
RF_params = {'class_weight' : 'balanced',  'max_depth':8,'n_estimators':EPOCHS,
            }


# In[59]:


# Comment out the models below to include in the Crossvalidation 

models = {
      "LogisticRegression": LogisticRegression(max_iter = EPOCHS),
    "SVC":SVC(probability=True, kernel = "rbf",class_weight='balanced',max_iter = EPOCHS),
     "lightgbm": lgb.LGBMClassifier(**lgb_params), 
    "xgboost": xgb.XGBClassifier(**xgb_params), 
    "catboost": cat.CatBoostClassifier(**cat_params),
     "ExtraTreeClassifier": ExtraTreesClassifier(),
     "RandomForestClassifier":RandomForestClassifier(**RF_params)
}


# ## Cross Validation

# In[60]:


# We use Stratified Kfold due to the class imbalance
kfold= StratifiedKFold(n_splits=5)


# In[61]:


test_preds =[]
train_preds = []
val_preds = []
model_shap_values = []

OOF_val_score =[]
OOF_val_loss =[]

for name,model in models.items():
    score_train = []
    score_val= []
    score_loss = []
    
    #Shap stuff
    list_shap_values = []
    val_index = []
    
    in_fold_trn_preds = []   
    in_fold_val_preds= []
    in_fold_preds = []
    print("\n######",name,"######")

    for fold, (train_idx,val_idx) in enumerate(kfold.split(X,y)):
        X_train,y_train = X.iloc[train_idx,:], y.iloc[train_idx]
        X_test,y_test = X.iloc[val_idx,:], y.iloc[val_idx]
        val_index.extend(val_idx)
        
        #Scaling
        if name in ["LogisticRegression",'Ridge', "SVC"]:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)
            X_temp = X.copy(deep = True)
            X_temp = scaler.transform(X_temp)
            test_temp = scaler.transform(df_tst)
            y_train_s = y_train.copy(deep = True)
        else:
            X_train_s, X_test_s , test_temp, X_temp, y_train_s = Scaling(X_train, X_test , df_tst, y_train, X )
            
                   
        #Fit
        if name in ["lightgbm"]:
            model.fit(X_train_s,y_train_s,
                      eval_set=[(X_test_s,y_test)],
                      eval_metric=kappa_score,
                      callbacks= [lgb.log_evaluation(-1), lgb.early_stopping(30)])
            
        if name in ["xgboost"]:
            model.fit(X_train_s,y_train_s,
                     eval_set=[(X_test_s,y_test)],
                      verbose= 0
                     )
        elif name in ["catboost"]:
            model.fit(X_train_s,y_train_s,
                      eval_set=[(X_test_s,y_test)],
                      #eval_metric=kappa_score(),
                      early_stopping_rounds=30,
                      verbose= 0
                     )          
        else:
            model.fit(X_train_s,y_train_s)
            
        # Predict
        y_preds = model.predict_proba(X_test_s)
        val_score= cohen_kappa_score(y_test, y_preds.argmax(axis =1), weights ='quadratic' )
        in_fold_val_preds.extend(y_preds)
        
        #remove low scoring models 
        if val_score >THRESHOLD:
            y_trn_preds = model.predict_proba(X_temp)
            score_val.append(val_score)
            score_train.append(cohen_kappa_score(y, y_trn_preds.argmax(axis =1), weights ='quadratic'))
            score_loss.append(log_loss(y_test, y_preds))
            in_fold_trn_preds.append(y_trn_preds )
            in_fold_preds.append(model.predict_proba(test_temp) )
    
    if len(score_val)>0:
        OOF_val_score.append(np.mean(score_val))
        OOF_val_loss.append(np.mean(score_loss))
        train_preds.append(np.mean(in_fold_trn_preds,axis=0))
        val_preds.append(in_fold_val_preds)
        test_preds.append(np.mean(in_fold_preds,axis=0))

    print("MEAN Trn AUC:",np.mean(score_train))
    print("MEAN Val AUC:",np.mean(score_val))
    print("MEAN logloss:",np.mean(score_loss))


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;"> Notes: Optimizing Models & Score </span></b>
# 
# * We  look at both the kappa_score score for the **validation** set (i.e. X_test) and for the **training** set (i.e. X_train) to identify  **over or underfitting**
# 
# <span style="color:red;font-size:1.3em;">Potential Solutions: Over/Underfitting </span>
# 1. **Overfitting:** The train score will be much larger than the validation score. There will be significant overfitting if the train score is close to 1 (i.e the model has over learnt the training data)
#   * Solution: [Reduce overfitting](https://medium.com/geekculture/how-to-stop-overfitting-your-ml-and-deep-learning-models-bb8324ace80b) such as implementing regularization, add data, feature engineering etc 
# 
# 2. **Underfitting:** If validation score is much lower than training. i.e there is a significant gap 
#   * Solution: Hyperparameter tuning and feature engineering (quick improvements would be to increase the number of epochs and  lower learning rate) 
# 
#     
# I will not optimize each model. I will leave this as homework for anyone to try: 
# * [Optuna optimization](https://optuna.org/)
# * [GridSearch ](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
# * Manual trial and error

# ## Setup submissions 

# In[62]:


sub[target]  = 0
sub_base =  sub.copy(deep= True)
sub_ensemble = sub.copy(deep= True)
sub_best_cv = sub.copy(deep= True)
sub_cal = sub.copy(deep = True)
mean_ensemble = sub.copy(deep = True)
mode_ensemble = sub.copy(deep = True)
hard_ensemble = sub.copy(deep = True)


# In[63]:


# Base model submission
sub_base[target]=test_base.argmax(axis = 1)
print("Base Model (lightgbm) values (inital 5 rows):")
sub_base = sub_base + 3
sub_base.to_csv("submission_base.csv")
sub_base.head()


# In[64]:


# Best CV model submission 
sub_best_cv_trn = y.copy(deep = True)
sub_best_cv_trn = train_preds[np.argmax(OOF_val_score)].argmax(axis = 1)
sub_best_cv_trn = sub_best_cv_trn+3

sub_best_cv[target] = test_preds[np.argmax(OOF_val_score)].argmax(axis = 1)
print(f"Best CV model: {list(models)[np.argmax(OOF_val_score)]}")
print("Inital 5 rows")
sub_best_cv = sub_best_cv+3
sub_best_cv.to_csv("sub_best_cv.csv")
sub_best_cv.head()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;"> Question: Why did Random Forest do well  </span></b>
# 
# Now that we can see our Cross Validation scores we can get an insight into how certain Algorithms work with our data
# * We know that Random Forest does **boostrapping** of samples and **feature selection**, therefore we can deduce that either:
#  * Certain features are negatively affecting our models performance (as all features are used in lightgbm and xgboost)
#  * Certain samples are negatively affecting our models performance i.e. outliers 
#  
# <span style="color:red;font-size:1.3em;">Potential Solutions </span>
# * Determine if Random Forest is impacted by bootstrapping or features selected
#  * **Boostrapping**: check our scaling and distribution techniques to make sure we are identifying outliers. Also add better features 
#  * **Features selected**: decomposition of features through dropping features and PCA of correlated features
#  

# # 🥣 Ensembling 🥣
# There are a number of ensembling techniques and I will state below my most used:
# 1. <span style="color:orange;font-size:1em;">Mode, Hard Voting, Soft Voting Ensemble</span>
# 2. <span style="color:orange;font-size:1em;">Weighted Average Ensemble</span> 
# 3. <span style="color:orange;font-size:1em;">Hill Climbing</span> ==> iterative process: adding predictions to others and multipling by random weights to find the best (**Note:** This can be quite complicated and I do use this process in this notebook) 
# 4. <span style="color:orange;font-size:1em;">Optimization methods</span> ==> Similar to High Climbing however it uses specialised optimization algorithms (please see link bellow on how to implement)
# 
# More on how these are implemented please check out my [notebook here ](https://www.kaggle.com/code/slythe/post-ensembling-hill-climbing-weighted-scipy)

# ### Mode vs Hard and Soft Voting
# As we are looking at a multi-class classification problem. We can ensemble a few other ways such as 
# * Mode: Get the **most frequent prediction** from all the models i.e. four models predicted the class as 3 and two models predicted the class as 1, therefore we use class 3 (most frequent) 
# * Soft-Voting (also called Mean Ensembling): Sum all the **probabilities** and divide by the number of models to get the **mean probabilities**
#  * This is only effective if the models are optimized (which ours arent) 
# * Hard Voting: Identify the **highest probability** accross all models, this is the prediction 
#     * This is usually better than Soft-voting when model are NOT optimized by less so if they models are optimized 

# In[65]:


# Get a dataframe of train and test predictions 
all_trn_preds = pd.DataFrame(index = df_trn.index)
all_tst_preds = pd.DataFrame(index = df_tst.index)

for i, preds in enumerate(train_preds):
    all_trn_preds[f"{list(models.keys())[i]}_{OOF_val_score[i]}"] = train_preds[i].argmax(axis=1)
    all_tst_preds[f"{list(models.keys())[i]}_{OOF_val_score[i]}"] = test_preds[i].argmax(axis=1)
all_tst_preds = all_tst_preds+3
all_tst_preds


# #### Mode Ensembling

# In[66]:


mode_ensemble_trn= y.copy(deep = True)
mode_ensemble_trn= all_trn_preds.mode(axis = 1)[0].values
mode_ensemble_trn = np.round(mode_ensemble_trn).astype('int32')
mode_ensemble_trn = mode_ensemble_trn+3

mode_ensemble[target]= all_tst_preds.mode(axis = 1)[0]
mode_ensemble[target] = np.round(mode_ensemble[target]).astype('int32')
mode_ensemble = mode_ensemble
mode_ensemble.to_csv("mode_ensemble.csv")
mode_ensemble


# #### Soft Voting (Mean) Ensembling

# In[67]:


# Get mean probabilities
mean_probs = pd.DataFrame(np.zeros((len(all_tst_preds), 6)))
mean_probs_trn =  pd.DataFrame(np.zeros((len(all_trn_preds), 6)))

for i,preds in enumerate(test_preds):
    mean_probs = mean_probs + test_preds[i]
    mean_probs_trn = mean_probs_trn + train_preds[i]
mean_probs = mean_probs/ len(all_tst_preds.columns)
mean_probs_trn = mean_probs_trn /len(all_tst_preds.columns)
mean_probs


# In[68]:


# Mean ensemble (train)
mean_ensemble_trn= y.copy(deep = True)
mean_ensemble_trn= mean_probs_trn.idxmax(axis = 1).values
mean_ensemble_trn = mean_ensemble_trn.astype('int32')
mean_ensemble_trn = mean_ensemble_trn+3

# Mean ensemble (test)
mean_ensemble[target]= mean_probs.idxmax(axis = 1).values
mean_ensemble[target] = mean_ensemble[target].astype('int32')
mean_ensemble= mean_ensemble+3
mean_ensemble.to_csv("mean_ensemble.csv")
mean_ensemble


# ### Hard Voting
# Get the max probability accross all model predictions 

# In[69]:


argmax_val_trn = []

for row in range(len(y)) :
    max_row_val = 0
    current_argmax = -1   
    #for each row of predictions get the max 
    for m in range(len(train_preds)) :
        #if max is greater that other models find the prediction class
        if max(train_preds[m][row]) > max_row_val:
            current_argmax = np.argmax(train_preds[m][row])
            max_row_val = np.max(train_preds[m][row])
            
    argmax_val_trn.extend([current_argmax])


# In[70]:


argmax_val = []

for row in range(len(df_tst)) :
    max_row_val = 0
    current_argmax = -1   
    #for each row of predictions get the max 
    for m in range(len(test_preds)) :
        #if max is greater that other models find the prediction class
        if max(test_preds[m][row]) > max_row_val:
            current_argmax = np.argmax(test_preds[m][row])
            max_row_val = np.max(test_preds[m][row])
            
    argmax_val.extend([current_argmax])


# In[71]:


hard_ensemble[target] = argmax_val
hard_ensemble= hard_ensemble+3
hard_ensemble.to_csv("hard_ensemble.csv")
hard_ensemble


# ## Weighted Ensembling
# * Multiple the predictions by their relative score (wieght) and divide by the sum of the scores 
# * In this notebook I will focus on **weighted ensembling** as this usually gets decent results for very little coding
# 
# ![image.png](attachment:03c76936-185b-4cd9-be46-f2b10465e1e4.png)

# In[72]:


sub_ensemble_trn = y.copy(deep = True)

for i, preds in enumerate(test_preds):
    sub_ensemble[target] = sub_ensemble[target] + (test_preds[i].argmax(axis=1) * OOF_val_score[i]) #multiply preds by their corresponding auc score
    sub_ensemble_trn= sub_ensemble_trn + (train_preds[i].argmax(axis=1) * OOF_val_score[i])
    
sub_ensemble_trn =   (sub_ensemble_trn/ sum(OOF_val_score) ).values
sub_ensemble_trn = np.round(sub_ensemble_trn).astype('int32')
sub_ensemble_trn= sub_ensemble_trn+3

sub_ensemble[target] =   (sub_ensemble[target]/ sum(OOF_val_score)).values 
sub_ensemble[target] = np.round(sub_ensemble[target]).astype('int32')
sub_ensemble= sub_ensemble+3
sub_ensemble.to_csv("sub_weighted_ensemble.csv")
sub_ensemble.head()


# # ✅ Post-Prediction Analysis & Manipulation ✅
# 
# #### i. Calibration: 
# * Our models may not be optimized and a simple linear model trained on our training predictions may be able to smooth out any inconsistencies with our model through fitting and additional linear model on our predictions 
# 
# #### ii. Prediction shifting

# In[73]:


# Create an Ensemble of the training data predictions 
sub_train = pd.DataFrame(index = y.index)
sub_train[target] = 0

for i, preds in enumerate(train_preds):
    sub_train[target] = sub_train[target] + (train_preds[i].argmax(axis=1) * OOF_val_score[i]) #multiple preds by their corresponding auc score
    
sub_train[target] =   sub_train[target]/ sum(OOF_val_score) 
sub_train[target] = np.round(sub_train[target]).astype('int32')
sub_train = sub_train+3
sub_train.to_csv("sub_train.csv")
sub_train.head()


# ### i. Calibration
# * We will calibrate our **test predictions** by fitting a linear regression model to a dataframe of our **training predictions** from each models Cross Validation. 
# * This fitted calibrated model will then predict the **test predictions** and hopefully smooth out the values and provide a final test prediction 

# In[74]:


CALIBRATION = "linear"


# In[75]:


if CALIBRATION == "linear":
    model = LogisticRegression(max_iter = 10000)
elif CALIBRATION=='gaus': 
    model = GaussianNB()
elif CALIBRATION=='CV':
    model = CalibratedClassifierCV(lin_model, method='isotonic', cv=5)
    
model.fit(mean_probs_trn, np.ravel(y))
y_cal_trn  = model.predict_proba(mean_probs_trn)
y_cal_test  = model.predict_proba(mean_probs)
y_cal_test = y_cal_test.argmax(axis =1)
y_cal_test= y_cal_test
y_cal_test


# In[76]:


sub_cal[target] = y_cal_test
print("Calibrated values (inital 5 rows)")
sub_cal.to_csv("sub_cal.csv")
sub_cal = sub_cal+3
sub_cal.head()


# ## Post-Prediction Analysis 

# In[77]:


fig, ax = plt.subplots(7,2, figsize = (30,30))
ax = np.ravel(ax)

sns.countplot(x =sub_base[target], label = f"Baseline: test preds", color ='black', ax= ax[0])
sns.countplot(x =trn_preds_base.argmax(axis =1)+3, label = f"Baseline: train preds", color ='black', ax= ax[1])
sns.countplot(x =y+3, label = f"Actual train",alpha =0.5, color ='blue', ax= ax[1])

sns.countplot(x =mean_ensemble[target], label = f"Ensemble Mean: test preds", color ='orange', ax= ax[2])
sns.countplot(x =mean_ensemble_trn, label = f"Ensemble Mean: train preds", color ='orange', ax= ax[3])
sns.countplot(x =y+3, label = f"Actual train",alpha =0.5, color ='blue', ax= ax[3])

sns.countplot(x =mode_ensemble[target], label = f"Ensemble Mode: test preds", color ='red', ax= ax[4])
sns.countplot(x =mode_ensemble_trn, label = f"Ensemble Mode: train preds", color ='red', ax= ax[5])
sns.countplot(x =y+3, label = f"Actual train",alpha =0.5, color ='blue', ax= ax[5])

sns.countplot(x =sub_ensemble[target], label = f"Ensemble weighted: test preds", color ='purple', ax= ax[6])
sns.countplot(x =sub_ensemble_trn, label = f"Ensemble weighted: train preds", color ='purple', ax= ax[7])
sns.countplot(x =y+3, label = f"Actual train",alpha =0.5, color ='blue', ax= ax[7])

sns.countplot(x =hard_ensemble[target], label = f"Ensemble Hard: test preds", color ='yellow', ax= ax[8])
sns.countplot(x =argmax_val_trn, label = f"Ensemble Hard: train preds", color ='yellow', ax= ax[9])
sns.countplot(x =y+3, label = f"Actual train",alpha =0.5, color ='blue', ax= ax[9])

sns.countplot(x =sub_best_cv[target], label = f"Best CV: {list(models)[np.argmax(OOF_val_score)]}", color ='green',ax = ax[10])
sns.countplot(x =sub_best_cv_trn, label = f"Best CV: {list(models)[np.argmax(OOF_val_score)]}", color ='green', ax= ax[11])
sns.countplot(x =y+3, label = f"Actual train",alpha =0.5, color ='blue', ax= ax[11])

sns.countplot(x =y_cal_test, label = f"Calibrated test preds", color ='pink',ax = ax[12])
sns.countplot(x =y_cal_trn.argmax(axis =1), label = f"Calibrated train preds", color ='pink', ax= ax[13])
sns.countplot(x =y+3, label = "Actual train",alpha =0.5, color ='blue', ax= ax[13])

for i in range(13):
    ax[i].legend(fontsize = 15)
    
plt.suptitle("Predictions vs actual", fontsize = 15)
plt.tight_layout(pad = 3)
plt.show()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes: Prediction Analysis </span></b>
# 
# * Im not happy with the test predictions for most of the models. We can see that only 3 classes are predicted for most of them
# * This is generally the case for highly imbalanced datasets 
# * Solutions are the same as stated previously: 
#  * Oversampling (not implemented)
#  * Additional features (not implemented)
#  * Class weighting of models (implemented)
#  
# ##### NB: Random Forest and Hard Voting Ensembling seem to be the best predictions here. We should use these for submission 

# ### Residual  Analysis 
# * Lets look at our Validation predictions and see where our best cross-validation model incorrectly predicted the classification 

# In[78]:


y_ =pd.DataFrame(y+3)
y_.loc[val_index,"val_predictions"] = np.argmax(val_preds[np.argmax(OOF_val_score)], axis =1) +3
y_["val_predictions"] = y_["val_predictions"].astype(int)


# In[79]:


plt.figure(figsize = (20,7))
true_vals = y_.loc[y_.quality ==y_.val_predictions].groupby("quality").sum()
inco_vals = y_.loc[y_.quality !=y_.val_predictions].groupby("quality").sum()
plt.bar(x = list(true_vals.index), height = true_vals.val_predictions, label = "Correct Predictions")
plt.bar(x = list(inco_vals.index), height = inco_vals.val_predictions, alpha = 0.5, color = "r", label = "Incorrect Predictions")
plt.title("Incorrect predictions for Random Forest" ,fontsize =15)
plt.legend()
plt.show()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes: </span></b>
# 
# * **Zero values of Class 3/ 4/8  were correctly predicted**: this is very poor and we would was to try improve this my helping our model identify class 3, 4 and 5 better (TIP: Looking at our shap graphs might give us a feature that we could engineer to assist)
# * **Equal amounts of Correct/ Incorrect predictions for class 6**: Same as above, however this class has a large number of values. As such we should focus on this to improve our score the most
# 
# Lets try look at the incorrect predictions and count the classes that were predicted instead 

# In[80]:


#Create a cross tab of all classes and the number of incorrect predictions by class
df_= pd.crosstab(y_.quality, y_.val_predictions)
# fill correct predictions as 0 
for row_col in list(df_.index):
    try:
        df_.loc[row_col,df_.columns==row_col] = 0
    except:
        pass
print("True values vs number of incorrect predictions\n")
df_


# In[81]:


ax = df_.plot(kind="barh", figsize =(25,12), stacked=True,)
for c in ax.containers:
    ax.bar_label(c, label_type='center')
plt.yticks(fontsize =15)
plt.xlabel("Incorrect values",fontsize =15)
plt.title("Incorrect predictions by class count for Random Forest" ,fontsize =20)
plt.tight_layout()
plt.ylabel("True Values",fontsize =15)
plt.show()


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Notes: </span></b>
#     
# * Again we can see that class 6 is highly incorrectly predicted
# * **Class 6** has **+-350** predictions that are **incorrectly predicted as class 5**
# 
# <span style="color:red;font-size:1.2em;">Potential Solutions: Incorrect predictions </span>
# 1. Additonal features (not implemented)
# 2. Threshold changes and prediction shifting (see below)

# ## ii. Prediction shifting (Also called Threshold cuttoff for binary classification)
# We now undergo a process of shifting our probabilities such that predictions of 5 are more likely to be 6:
# 1. select all test predictions that were predicted as 5 and get these indexes 
# 2. Filter our test predictions by this index 
# 3. Multiply the probability for class 6 by our percentage (see above)
# 4. We should now have a data set where we are more likely to predict 6 if the value was orginally 5
# 
# **Note**: This process can be highly inconclusive and inaccurate. There is a lot of trial and error here and it may not work as expected

# In[82]:


# 350 + 286 values of prediction==6 are incorrect. Lets see what % this is to the total predictions of 5
perc_ =  ((int(df_[df_.index ==6].sum(axis =1).values))/ (y+3==5).sum())
# lets get the total change if we increased the value 
perc_ = perc_ + 1
perc_


# The idea here is to multiply this value to our class 6 probabilities (that were predicted as 5 or 7). Ideally this will increase the number of clas 6 predictions and reduce incorrect class 5 and 7 preds

# In[83]:


# get indexes of class 6 
Index_57 = sub_best_cv.reset_index(drop = True)
Index_57 = Index_57[(Index_57.quality==6)|(Index_57.quality==7)].index

df_rf_changed = pd.DataFrame(test_preds[np.argmax(OOF_val_score)])
df_rf_changed.iloc[Index_57,3 ] = df_rf_changed.iloc[Index_57,3 ]* perc_ # multiple class 6 by the percentage 
df_rf_changed = df_rf_changed.idxmax(axis =1)
df_rf_changed = df_rf_changed+3


# In[84]:


fig, ax = plt.subplots(1,2, figsize = (25,8), sharey = True)
ax_1 = sns.countplot(x =sub_best_cv[target], label = f"RF Preds (original)", color ='blue',ax = ax[0])
ax_2 = sns.countplot(x =df_rf_changed, label = f"RF shifted preds", color ='red', ax= ax[1])

for ax in [ax_1, ax_2]:
    for c in ax.containers:
        ax.bar_label(c, label_type='center')
ax_1.legend()
ax_2.legend()
plt.show()


# Class 6 has increased, class 5 and class 7 have decreased

# In[85]:


rf_shifted = sub.copy(deep= True)
rf_shifted[target] = df_rf_changed.values
rf_shifted.to_csv("rf_shifted.csv")
rf_shifted


# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:2px;">
# <b><span style="color:blue;font-size:1.2em;">Final Notes: </span></b>
#     
# A question I get a lot is how do I improve my score. To put it simply; its a focus on 3 areas: 
# 1. <span style="color:red;font-size:1em;">Model hyperparameter optimization </span>
# 1. <span style="color:red;font-size:1em;">Feature Engineering </span>
# 1. <span style="color:red;font-size:1em;">Post processing your prediction </span>
# 
# As noted above, feature engineering & hyperparameter optimization is a lengthy process of trial and error (creating/ dropping and testing new features/parameters) as well as model optimization \
# To keep this notebook short I will leave a few things to you
# 
# #### Additional Things to Try on Your Own
# 
# * PCA columns 
# * Feature engineering
# * Model Optimization (Hyperparameter Tuning)
# * Different Ensembling methods
