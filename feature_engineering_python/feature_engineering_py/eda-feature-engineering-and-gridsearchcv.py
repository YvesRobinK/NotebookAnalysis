#!/usr/bin/env python
# coding: utf-8

# **So we would start analysing by importing all important libraries .**

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


#Importing python libraries
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model


# In[3]:


pd.set_option('display.max_rows',10)#So that we can see the whole dataset at one go


# We are making dataframe test and train to analyse the given dataset.

# In[4]:


# import train and test to play with it
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[5]:


#get the type
type(train_df)


# # Basic Structure

# Going through the complete dataset to get the jist of it.

# In[6]:


test_df.info()


# In[7]:


train_df.info()


# In[8]:


test_df['Survived'] = -888 #Adding Survived with a default value


# In[9]:


test_df.info()


# In[10]:


#test_df.head()
train_df.head()


# In[11]:


train_df = train_df[['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Survived']]


# In[12]:


#Concatinating two data frames(train and test)
df = pd.concat((train_df,test_df),axis = 0)

df = df.reset_index()


# In[13]:


df.info()


# In[14]:


df = df.drop(['index'],axis=1)

df.head()


# In[15]:


df = df.set_index('PassengerId')
df.tail()


# 
# # <h1>Various pandas functions on data set<h1>

# In[16]:


df.Name.head()


# In[17]:


df.loc[5:10]


# In[18]:


#indexing : use iloc based indexing


# In[19]:


df.iloc[5:10,3:8]


# In[20]:


#filter rows based on the condition
male_passengers = df.loc[df.Sex =='male',:]
print('Number of male passengers : {0}'.format(len(male_passengers)))


# In[21]:


#use & or | operators to build complex logic


# In[22]:


male_passengers_first_class = df.loc[((df.Sex =='male') &(df.Pclass == 1)),:]


# In[23]:


print('Number of passengers in first class:{0}'.format(len(male_passengers_first_class)))


# So far we have use numpy and pandas to analyse the whole data and going through it.
# Now we will Start exploring the whole Dataset.

# 
# # Summaty Statistics

# 
# <h2>Centrality Measure<h2>

# Various technique such as Mean and Median are used as Centrality measures
# Mean may me good option often but it is quite affected by extreme value.In such cases we can use median for analysis.

# <h2>Spread/Dispersion measure<h2>

# we use range we can see how things are packed.But it is affected by extreme values.
# So percentiles are used.These are specially used as quartiles
# 

# In[24]:


# use .describe() to get statistics for all numeric columns
df.describe()


# In[25]:


df.isnull().sum()


# In[26]:


#Numerical feature
#centrality measures


# In[27]:


print('Mean Fare : {0}'.format(df.Fare.mean()))
print('Median Fare : {0}'.format(df.Fare.median()))


# In[28]:


#dispersion measure
print('Max fare  : {0}'.format(df.Fare.max()))#max
print('Min fare  : {0}'.format(df.Fare.min()))#max
print('Fare range  : {0}'.format(df.Fare.max() - df.Fare.min()))#range
print('25 percentile  : {0}'.format(df.Fare.quantile(.25)))#25 percentile
print('50 percentile  : {0}'.format(df.Fare.quantile(.50)))#50 percentile
print('75 percentile  : {0}'.format(df.Fare.quantile(.75)))#75 percentile
print('Variance fare: {0}'.format(df.Fare.var()))#variance
print('Standard deviation  : {0}'.format(df.Fare.std()))#standard deviation




# In[29]:


# box-whiskers plot
df.Fare.plot(kind='box')


# # Categorical features
# <p>Also known as features with non integer values.(Boolean,Someone's Name or Gender)<p>

# In[30]:


#use describe to get statistics for all columns including non-numeric ones
df.describe(include='all')


# In[31]:


df.Sex.value_counts()


# In[32]:


df.Sex.value_counts(normalize = True)


# In[33]:


df[df.Survived != -888].Survived.value_counts()


# In[34]:


df.Pclass.value_counts()


# In[35]:


#Visualize Sex count,Survived and Class wise survival
df.Sex.value_counts().plot(kind='bar');


# In[36]:


df[df.Survived != -888].Survived.value_counts().plot(kind='bar');


# In[37]:


df.Pclass.value_counts().plot(kind='bar');


# In[38]:


df.Pclass.value_counts().plot(kind='bar',rot = 0,title = "Pclass count on Titanic");


# <h2>Distributions of univariate feature at a time<h1>

# In[39]:


#for univariate distributions we use Histogram and KDE
#KDE stands for Kerenl Density Estimation


# In[40]:


df.Age.plot(kind ='hist',title = 'histogram for Age' );


# In[41]:


df.Age.plot(kind ='kde',title = 'histogram for Age' );


# In[42]:


df.Fare.plot(kind ='hist',title = 'histogram for Age' );


# <h2>Bivariate Distribution<h2>

# In[43]:


#We use bivariate distribution for Scatter plot


# In[44]:


df.plot.scatter(x='Age',y='Fare',title='Scatter Plot:Age vs Fare');


# In[45]:


df.plot.scatter(x='Age',y='Fare',title='Scatter Plot:Age vs Fare',alpha = 0.5);


# In[46]:


df.plot.scatter(x='Pclass',y='Fare',title='Scatter Plot:Pclass vs Fare');


# <h2>Grouping<h2>

# In[47]:


df.groupby('Sex').Age.median()


# In[48]:


#group by
df.groupby('Pclass').Fare.median()


# In[49]:


df.groupby('Pclass').Age.median()


# In[50]:


df.groupby(['Pclass'])['Fare','Age'].median()


# In[51]:


df.groupby(['Pclass']).agg({'Fare':'mean','Age':'median'})


# In[52]:


# more complicated aggregation
aggregations ={
    'Fare': {#work on fare column
        'mean_Fare':'mean',
        'median_Fare':'median',
        'Max_Fare':max,
        'Min_Fare' :np.min
    },
    'Age': {
        'mean_Age':'mean',
        'median_Age':'median',
        'Max_Age':max,
        'Min_Age' :np.min
    }
}


# In[53]:


df.groupby(['Pclass']).agg(aggregations)


# In[54]:


df.groupby(['Pclass','Embarked']).Fare.median()


# <h2>Crosstab<h2>

# In[55]:


pd.crosstab(df.Sex,df.Pclass)


# In[56]:


pd.crosstab(df.Sex,df.Pclass).plot(kind='bar');


# <h2>Pivot Table<h2>

# In[57]:


#pivot table
df.pivot_table(index='Sex',columns = 'Pclass',values = 'Age',aggfunc='mean')


# 
# 
# 
# Similar results can be found by group by table

# In[58]:


df.groupby(['Sex','Pclass']).Age.mean().unstack()


# # Data Munging

# **Step 1:Missing value addressing**

# Feature : Embarked

# In[59]:


df.isnull().sum()


# In[60]:


train_df.isnull().sum()


# In[61]:


df.info()


# In[62]:


df[df.Embarked.isnull()]


# In[63]:


#how many people embarked at a particular points
df.Embarked.value_counts()


# In[64]:


#which embarked point has highest survival count
pd.crosstab(df[df.Survived != -888].Survived,df[df.Survived != -888].Embarked)


# In[65]:


# impute missing value with 'S'
#df.loc[df.Embarked.isnull(),'Embarked'] = S
#df.Embarked.fillna('S',inplace = True)


# In[66]:


df.groupby(['Pclass','Embarked']).Fare.median() 


# From above we get to know that the passenger who are in class 1 and paid fare of 80 has  more chances to be Embarked from C.

# In[67]:


df.Embarked.fillna('C',inplace = True)


# In[68]:


df.Embarked.isnull().sum()


# In[69]:


df.info()


# Feature : Fare

# In[70]:


df[df.Fare.isnull()]


# In[71]:


median_fare = df.loc[(df.Pclass == 3) & (df.Embarked=='S'),'Fare'].median()
print (median_fare)


# In[72]:


df.Fare.fillna(median_fare,inplace=True)


# In[73]:


df.info()


# Feature : Age

# In[74]:


df.Age.isnull().sum()


# <h1>We have three options to fill missing age value<h1>
# <h2> option 1 : replace all missing value with the mean value<h2>

# In[75]:


df.Age.plot(kind='hist',bins=20);


# In[76]:


df.Age.mean()


# Due to several 70's and 80's the mean is quite affected

# <h2>option 2: Replace by median<h2>

# In[77]:


df.groupby('Sex').Age.median()


# In[78]:


df[df.Age.notnull()].boxplot('Age','Sex');


# <h2>option 3 : replace with median age of Pclass<h2>

# In[79]:


df[df.Age.notnull()].boxplot('Age','Pclass');


# <h2>option 4 : replace with median age of title<h2>

# We all know that the' Mr' is used for old veteran and 'master' is used for young man.Similar goes with females.So we will extract titles to predict the values of the age of the passenger of titanic. 

# In[80]:


df.Name.head()


# In[81]:


def GetTitle(name):
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title


# In[82]:


#use map function to apply the function on each Name value row i
df.Name.map(lambda x : GetTitle(x))


# In[83]:


df.Name.map(lambda x : GetTitle(x)).unique()


# In[84]:


def GetTitle(name):
    title_group = {'mr' : 'Mr', 
               'mrs' : 'Mrs', 
               'miss' : 'Miss', 
               'master' : 'Master',
               'don' : 'Sir',
               'rev' : 'Sir',
               'dr' : 'Officer',
               'mme' : 'Mrs',
               'ms' : 'Mrs',
               'major' : 'Officer',
               'lady' : 'Lady',
               'sir' : 'Sir',
               'mlle' : 'Miss',
               'col' : 'Officer',
               'capt' : 'Officer',
               'the countess' : 'Lady',
               'jonkheer' : 'Sir',
               'dona' : 'Lady'
                 }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]


# In[85]:


df['Title'] = df.Name.map(lambda x : GetTitle(x))


# In[86]:


df.head()


# In[87]:


df[df.Age.notnull()].boxplot('Age','Title');


# In[88]:


#replace missing values
title_age_median = df.groupby('Title').Age.transform('median')


# In[89]:


df.Age.fillna(title_age_median,inplace = True)


# In[90]:


df.info()


# # Outliers

# Techniques to compensate for outliers
# 1. 1.Removal
# 1. 2.Transformation
# 1. 3.Binning
# 1. 4.Imputation

# In[91]:


# use histograms to get to understand the distribution
df.Age.plot(kind = 'hist' , bins = 20);


# In[92]:


df.loc[df.Age > 70]


# In[93]:


# hsitograms for fare
df.Fare.plot(kind='hist',bins = 20, title = 'Histograms for Fare')


# In[94]:


df.Fare.plot(kind='box');


# In[95]:


# look for the outliers
df.loc[df.Fare == df.Fare.max()]


# In[96]:


#try to use transformation to reduce the skewness


# In[97]:


LogFare = np.log(df.Fare +1)#adding 1 to accomalate 


# In[98]:


LogFare.plot(kind='hist',bins = 20);


# In[99]:


#binning


# In[100]:


pd.qcut(df.Fare,4)


# In[101]:


pd.qcut(df.Fare,4,labels=['very_low','low','high','very_high'])


# In[102]:


pd.qcut(df.Fare,4,labels = ['very_low','low','high','very_high']).value_counts().plot(kind='bar',rot = 0);


# In[103]:


# create fare bin feature
df['Fare_Bin']=pd.qcut(df.Fare,4,labels=['very_low','low','high','very_high'])


# # Feature Engineering

# **Feature : Age State(Adult or Child) **

# In[104]:


#Age State based on Age
df['AgeState'] = np.where(df['Age']>=18,'Adult','Child')


# In[105]:


#AgeState Counts
df['AgeState'].value_counts()


# In[106]:


pd.crosstab(df[df.Survived != -888].Survived , df[df.Survived != -888].AgeState)


# **Feature : FamilySize**

# In[107]:


df['FamilySize'] = df.Parch + df.SibSp + 1 # i for Self


# In[108]:


#explore the family feature
df['FamilySize'].plot(kind = 'hist',color = 'c');


# In[109]:


#further exploring familoy size with mjax family size
df.loc[df.FamilySize == df.FamilySize.max(),['Name','Survived','FamilySize','Ticket']]


# In[110]:


pd.crosstab(df[df.Survived != -888].Survived , df[df.Survived != -888].FamilySize)


# 
# 
# **Feature :isMother**

# In[111]:


# a lady aged 18 or more who has Parch >0 and is married 
df['IsMother'] = np.where(((df.Sex == 'female') & (df.Parch > 0) & (df.Age>18) & (df.Title != 'Miss')),1,0)


# In[112]:


#Crosstab with mother
pd.crosstab(df[df['Survived'] != -888].Survived,df[df['Survived'] != -888].IsMother )


# 
# **Feature:Deck**

# In[113]:


#explore Cabin values
df.Cabin


# In[114]:


#Getting unique cabin
df.Cabin.value_counts()


# In[115]:


#We see that T is odd one out in above observation so we can asume it is mistaken value


# In[116]:


# get the value to Nan
df.loc[df.Cabin == 'T','Cabin']=np.NaN


# In[117]:


def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')


# In[118]:


df['Deck'] = df['Cabin'].map(lambda x: get_deck(x))


# In[119]:


# check counts
df.Deck.value_counts()


# In[120]:


pd.crosstab(df[df.Survived != -888].Survived,df[df.Survived != -888].Deck)


# In[121]:


df.info()


#  <h1>Categorical feature<h1>
#  
#  

# In[122]:


#sex
df['IsMale'] = np.where(df.Sex == 'male',1,0)


# In[123]:


#columns deck,pclass,title,Agestate
df = pd.get_dummies(df,columns=['Deck','Pclass','Title','Fare_Bin','Embarked','AgeState'])


# In[124]:


df.info()


# # Drop and Reorder Columns

# In[125]:


#drop columns


# In[126]:


df.drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'],axis = 1,inplace = True)


# In[127]:


#reorder columns
columns = [column for column in df.columns if column != 'Survived']
columns = ['Survived']+columns
df = df[columns]


# In[128]:


df.info()


# In[129]:


df.to_csv('out.csv')#Saving Dataset before making predicting model
#This would be saved as output in Version folder.


# In[130]:


train_df = df.loc[0:891,:]


# In[131]:


train_df.info()


# In[132]:


train_df.tail()


# In[133]:


test_df = df.loc[892:,:]


# In[134]:


test_df.tail()


# # Feature Selection for higher accuracy

# Feature Selection Methods:
# 
# I will share 3 Feature selection techniques that are easy to use and also gives good results.
# 
# 1. Univariate Selection
# 
# 2. Feature Importance
# 
# 3. Correlation Matrix with Heatmap

# In[135]:


train_df.shape


# In[136]:


test_df.shape
test_df.info()


# 1. Univariate Selection

# Statistical tests can be used to select those features that have the strongest relationship with the output variable.
# 
# The scikit-learn library provides the SelectKBest class that can be used with a suite of different statistical tests to select a specific number of features.
# 
# The example below uses the chi-squared (chi²) statistical test for non-negative features to select  the best features

# In[137]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("Survived", axis=1).copy()


# In[138]:


X_train.shape


# In[139]:


Y_train.shape


# In[140]:


X_test.shape


# In[141]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[142]:


Xtr = X_train.copy()
Xtr.head()


# In[143]:


ytr = Y_train.copy()
#target column i.e price range
ytr.head()
Y_train.head()


# In[144]:


#apply SelectKBest class to extract top 20 best features
bestfeatures = SelectKBest(score_func=chi2, k=20)
fit = bestfeatures.fit(Xtr,ytr)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtr.columns)


# In[145]:


#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(32,'Score'))  #print 10 best features


# From the analysis we get that score of various columns vary differentely. For now using this method we can drop the feature with score less than 10 will make our dataset more cleaner and more accurate.
# Lets try another method
# 

# 2 . Feature Importance

# You can get the feature importance of each feature of your dataset by using the feature importance property of the model.
# 
# Feature importance gives you a score for each feature of your data, the higher the score more important or relevant is the feature towards your output variable.
# 
# Feature importance is an inbuilt class that comes with Tree Based Classifiers, we will be using Extra Tree Classifier for extracting the top 10 features for the dataset.

# In[146]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


# In[147]:


model = ExtraTreesClassifier()
model.fit(Xtr,ytr)


# In[148]:


print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=Xtr.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()




# 3.Correlation Matrix with Heatmap
# 
# Correlation states how the features are related to each other or the target variable.
# 
# Correlation can be positive (increase in one value of feature increases the value of the target variable) or negative (increase in one value of feature decreases the value of the target variable)
# 
# Heatmap makes it easy to identify which features are most related to the target variable, we will plot heatmap of correlated features using the seaborn library.

# In[149]:


train_df.head()


# In[150]:


corrmat = train_df.corr()
print(corrmat.Survived)
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(train_df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# Now lets remove the worst 12 features to make the dataset clearner and more perfect.
# 
# Here is the list of not important features <br>
#         Title_Master    <br>
#         Fare_Bin_low     <br>
#           Title_Lady     <br>
#        Fare_Bin_high     <br>
#               Deck_F     <br>
#            Title_Sir     <br>
#       AgeState_Adult     <br>
#               Deck_A    <br>
#            FamilySize    <br>
#               Deck_G     <br>
#        Title_Officer     <br>
#          Embarked_Q      <br>

# Deck_B    25.875581
# 3             IsMother    24.601467
# 27          Embarked_C    22.009402
# 12              Deck_Z    20.731648
# 8               Deck_D    19.489646
# 9               Deck_E    18.140638
# 31      AgeState_Child    11.827471
# 7               Deck_C    10.936730

# In[151]:


list = ['Title_Master','Fare_Bin_low','Title_Lady','Fare_Bin_high','Deck_F','Title_Sir','AgeState_Adult','Deck_A','FamilySize','Deck_G',
        'Title_Officer','Embarked_Q','Deck_B','IsMother','Embarked_C','Deck_Z','Deck_D','Deck_E','AgeState_Child']


# In[152]:


X_train.drop(list,axis=1,inplace = True)
X_test.drop(list,axis = 1,inplace = True)


# # Building Machine Learning Models

# Now we will train several Machine Learning models and compare their results. Note that because the dataset does not provide labels for their testing-set, we need to use the predictions on the training set to compare the algorithms with each other. 

# I will be using RandomDomforest Classifier for predicting the outcomes as it yeild's most result.(more detail checkout discussion panel)After that to tune the model use GridSearchCV and RandomizedSearchCV. 
# 

# In[153]:


X_train.shape


# In[154]:


Y_train.shape


# In[155]:


# Random FOrest Classifier using Grid SearchSearch CV


# In[156]:


X_test.shape


# In[157]:


rfc=RandomForestClassifier(random_state=42)


# In[158]:


param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[159]:


"""CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, Y_train)"""


# In[160]:


"""CV_rfc.best_params_"""


# In[161]:


rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 500, max_depth=7, criterion='gini')


# In[162]:


rfc1.fit(X_train, Y_train)


# In[163]:


pred=rfc1.predict(X_test)


# In[164]:


X_test.index


# In[165]:


df_result = pd.DataFrame(pred)


# In[166]:


df_result


# In[167]:


df_result['Survived'] = pred


# In[168]:


df_result.drop(0,axis =1,inplace = True)


# In[169]:


df_result['PassengerId']=X_test.index


# In[170]:


df_result.head()


# In[171]:


df_result = df_result.set_index('PassengerId')


# In[172]:


df_result.head()


# In[173]:


df_result.to_csv('output.csv')


# In[174]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,10,15,20,30]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[175]:


"""
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42,
                               n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train, Y_train)"""


# In[176]:


"""print(rf_random.best_params_)"""


# In[177]:


rf1=RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators= 200, max_depth=7, criterion='gini',
                           min_samples_split = 2, min_samples_leaf = 2, bootstrap =  False)


# In[178]:


rf1.fit(X_train, Y_train)


# In[179]:


pred1 = rf1.predict(X_test)


# In[180]:


df_result_1 = pd.DataFrame(pred1)


# In[181]:


df_result_1


# In[182]:


df_result_1['Survived'] = pred1


# In[183]:


df_result_1.head()


# In[184]:


df_result_1.drop(0,axis =1,inplace = True)


# In[185]:


df_result_1['PassengerId']=X_test.index


# In[186]:


df_result_1.head()


# In[187]:


df_result_1 = df_result_1.set_index('PassengerId')


# In[188]:


df_result_1.head()


# In[189]:


df_result_1.to_csv('output_1.csv')


# In[190]:




