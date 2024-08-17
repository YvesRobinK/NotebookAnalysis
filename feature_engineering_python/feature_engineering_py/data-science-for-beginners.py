#!/usr/bin/env python
# coding: utf-8

# Hi, if you are new to the field of Data science then this notebook might help you. In this competition we have to predict the survivors from testing data based on the training data. Let's see how we can do this with the help of data science. 

# # Importing Libraries
# Python has a number of libraries available that can save us a lot of time and number of lines of code. To use these libraries we have to import them. Every library is unique and capable to doing specific tasks.
# 

# In[1]:


import sys  #access to system parameters https://docs.python.org/3/library/sys.html

import pandas as pd #collection of functions for data processing and analysis

import matplotlib #collection of functions for scientific and publication-ready visualization

import numpy as np   #foundational package for scientific computing

import scipy as sp    #collection of functions for scientific computing and advance mathematics

import IPython 
from IPython import display #pretty printing of dataframes in Jupyter notebook

import sklearn  #collection of machine learning algorithms

import random   #generate random values

import time    #To handle time values

import warnings #ignore warnings 
warnings.filterwarnings('ignore')


#Importing modelling algorithms from sklean and xgboost
from sklearn import svm , tree, linear_model , neighbors , naive_bayes , ensemble , discriminant_analysis , gaussian_process
from xgboost import XGBClassifier


#Importing modelling helpers
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot  as plt
import matplotlib.pylab as pylab
import seaborn as sns



#Configure Visualization Defaults
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12 , 8


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))







# We have imported the libraries and their methods above that will help us throughout our code. The scikit-learn library or sklearn library contains the algorithms that are most commonly used in data science. Matplotlib and seaborn are used to visualize data i.e. plotting graphs. 

# # Importing Data
# The train and test data are already provided to us in the competiton dataset. We need to import this data and perform our analysis.

# In[2]:


#importing the training data
training_data = pd.read_csv('../input/titanic/train.csv')
# making a copy of training data
train = training_data.copy(deep=True)

#importing the testing data
test = pd.read_csv('../input/titanic/test.csv')


# Here we have imported the train and test datasets.

# In[3]:


#data cleaner -> to clean both datasets at once
data_cleaner = [train, test]


# Here I have passed both test and train datasets as reference to the data_cleaner.By doing this we will be able to clean both the datasets by using one variable(data_cleaner) only.

# In[4]:


#Viewing train data 
train.sample(10)


# In[5]:


#Viewing test data 
test.sample(10)


# # Data Cleaning
# Before training our models, we have to see is there any problem with our data. For example, we have to look whether the data is realistic or not like a human's age is given to 500 which is not a realistic value. Another example can that the dataset has some missing values, if there are missing values we have to fill them. 

# In[6]:


#Checking null values in train dataset
print('Train columns with null values: \n', train.isnull().sum())


# We can see that the age column has 177 null values, cabin has 687 null values and embarked has 2 null values. Null value means 0.

# In[7]:


# Checking null values in test dataset
print('Test columns with null values: \n', test.isnull().sum())


# We can see that the age and cabin columns have null values.

# Now let's clean our data.

# In[8]:


#Completing : complete or delete missing values in train and test dataset
for dataset in data_cleaner:
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median() , inplace = True)
    
    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0] , inplace = True)
    
    #complete missing fair with median 
    dataset['Fare'].fillna(dataset['Fare'].median() , inplace = True)
    
    
#Deleting unwanted columns from train dataset
drop_column = ['PassengerId' , 'Cabin' , 'Ticket']
train.drop(drop_column , axis = 1 , inplace = True)



print(train.isnull().sum())

    


# We can see that there is no null value in the train dataset now. 

# In[9]:


# Feature Engineering for train and test datasets
for dataset in data_cleaner:
    #discrete variables 
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
    # if the person is alone 
    dataset['IsAlone'] = 1
    
    #if the person is not alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
    
    #splitting title from name
    dataset['Title'] = dataset['Name'].str.split(', ' , expand = True)[1].str.split('.' , expand = True)[0]
    
    dataset['FareBin'] = pd.qcut(dataset['Fare'],4)
    
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int),5)
    

    


# In[10]:


#cleaning rare title names
stat_min = 10
title_names = (train['Title'].value_counts()<stat_min) #this will create a true false series with title name as index

train['Title'] = train['Title'].apply(lambda x: 'Misc' if title_names.loc[x] ==True else x )

print(train['Title'].value_counts())


# Here if title is help by less than 10(stat_min) of people that we are adding that title under miscellaneous.

# In[11]:


#previewing data
train.describe(include = 'all')


# There is no null value in train dataset.

# In[12]:


#Converting objects to category using Label Encoder for train and test datasets

# Code categorical data
label = LabelEncoder()
for dataset in data_cleaner:
    dataset['Sex_Code']= label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code']= label.fit_transform(dataset['Embarked'])
    dataset['Title_Code']= label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code']= label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code']= label.fit_transform(dataset['FareBin'])
    
#setting target Y
Target = ['Survived']


#defining x variables for original features 
train_x = ['Sex' , 'Pclass' , 'Embarked' , 'Title' , 'SibSp' , 'Parch' , 'Age' , 'Fare' , 'FamilySize' , 'IsAlone'] #original features
train_x_calc = ['Sex_Code' , 'Pclass' , 'Embarked_Code' , 'Title_Code' , 'SibSp' , 'Parch' , 'Age' , 'Fare' , 'FamilySize' , 'IsAlone'] #features for algorithm calculations
train_xy = Target + train_x  
print('Original X Y: ',train_xy,'\n')


#defining x variables for original w/bin features to remove continous variables
train_x_bin = ['Sex_Code' , 'Pclass' , 'Embarked_Code' , 'Title_Code' ,'FamilySize' , 'AgeBin_Code' , 'FareBin_Code']
train_xy_bin = Target + train_x_bin
print('Bin X Y:' , train_xy_bin , '\n')

#defining x and y for dummy features original
train_dummy = pd.get_dummies(train[train_x])
train_x_dummy = train_dummy.columns.tolist()
train_xy_dummy = Target + train_x_dummy
print('Dummy X Y: ', train_xy_dummy, '\n')
train_dummy.head()




# Double Checking Cleaned Data

# In[13]:


print('Train columns with null values: \n', train.isnull().sum())
print('\n'*3)
print
print(train.info())
print('\n'*3)

print('Test columns with null values: \n', test.isnull().sum())
print('\n'*3)
print(test.info())
print('\n'*3)


# # Performing Exploratory Analysis with Statistics
# 
# Now we will explore our data with descriptive and graphical statistics to describe and summarize our variables.

# In[14]:


#Discrete Variable correlation by survival
for i in train_x:
    if train[i].dtype != 'float64':
        print('Survival Correlation by: ', i)
        print(train[[i , Target[0]]].groupby( i , as_index = False).mean())
        print('\n'*3)
        
#using crosstab
print(pd.crosstab(train['Title'], train[Target[0]]))
    


# Here we have correlated the features to their survival. 

# In[15]:


#Graph Distribution of Quantitative Data
plt.figure(figsize = [16 ,12])

plt.subplot(231)
plt.boxplot(x = train["Fare"], showmeans = True , meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(x = train["Age"], showmeans = True , meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (years)')

plt.subplot(233)
plt.boxplot(x = train["FamilySize"], showmeans = True , meanline = True)
plt.title('Family Size  Boxplot')
plt.ylabel('Family Size (#)')

plt.subplot(234)
plt.hist(x = [train[train['Survived']==1]['Fare'] , train[train['Survived']==0]['Fare']], stacked = True , color = ['g' , 'r'] , label = ['Survived' , 'Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x = [train[train['Survived']==1]['Age'] , train[train['Survived']==0]['Age']], stacked = True , color = ['g' , 'r'] , label = ['Survived' , 'Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (years)')
plt.ylabel('# of Passengers')
plt.legend()


plt.subplot(236)
plt.hist(x = [train[train['Survived']==1]['FamilySize'] , train[train['Survived']==0]['FamilySize']], stacked = True , color = ['g' , 'r'] , label = ['Survived' , 'Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()


# In[16]:


#we will use seaborn graphics for multi-variable comparison

# Graphing individual features by survival
fig , s_axis = plt.subplots(2,3, figsize = [26,26])


sns.barplot(x = 'Embarked' , y = 'Survived' , data = train , ax= s_axis[0,0])
sns.barplot(x = 'Pclass' , y = 'Survived' , order = [1,2,3], data = train , ax = s_axis[0,1])
sns.barplot(x = 'IsAlone' , y = 'Survived' ,order = [1,0], data = train , ax = s_axis[0,2])

sns.pointplot(x = 'FareBin' , y = 'Survived' ,data = train , ax = s_axis[1,0])
sns.pointplot(x = 'AgeBin' , y = 'Survived' ,data = train , ax = s_axis[1,1])
sns.pointplot(x = 'FamilySize' , y = 'Survived' ,data = train , ax = s_axis[1,2])


# In[17]:


# Graph distribution of qualitative data : Pclass
#we know class mattered in survival, now let's compare class and a 2nd feature

fig, (axis1, axis2, axis3)  = plt.subplots(1,3,figsize = (14,14))

sns.boxplot(x = 'Pclass' , y = 'Fare' , hue = 'Survived' , data = train , ax = axis1)
axis1.set_title('Pclass vs Fare survival comparison')

sns.violinplot(x = 'Pclass' , y = 'Age' , hue = 'Survived' , data = train , split = True , ax = axis2 )
axis2.set_title('Pclass vs Age survival comparison')

sns.boxplot(x = 'Pclass' , y = 'FamilySize' , hue = 'Survived' , data = train , ax = axis3)
axis3.set_title('Pclass vs FamilySize comparison')




# In[18]:


#Graph distribution of qualitative data: Sex
#we know sex mattered in survival, now let's compare sex and a 2nd feature

fig , qaxis = plt.subplots(1,3, figsize=(14,14))

sns.barplot (x = 'Sex' , y = 'Survived' , hue = 'Embarked' , data = train , ax = qaxis[0])
axis1.set_title('Sex vs Embarked Survival comparison')

sns.barplot(x = 'Sex' , y = 'Survived' , hue = 'Pclass' , data = train , ax = qaxis[1])
axis1.set_title('Sex vs Pclass Survival comparison')

sns.barplot(x = 'Sex' , y = 'Survived' , hue = 'IsAlone' , data = train , ax = qaxis[2])
axis1.set_title('Sex vs IsAlone Survival comparison')


# In[19]:


#more side-by-side comparisons
fig, (maxis1, maxis2) = plt.subplots(1,2, figsize = (16,16))

# how does family size influences the survival of the two sexes
sns.pointplot(x = 'FamilySize' , y = 'Survived' , hue = 'Sex' , data = train , palette = {'male' : 'blue' , 'female' : 'pink'},
             markers = ['*' , 'o'], linestyles = ['-' , '--'] , ax = maxis1)

# how does Pclass influences the survival of the two sexes
sns.pointplot(x = 'Pclass' , y = 'Survived' , hue = 'Sex' , data = train , palette = {'male' : 'blue' , 'female' : 'pink'},
             markers = ['*' , 'o'], linestyles = ['-' , '--'] , ax = maxis2)


# In[20]:


# how does the port of embarkment influences the survival of two sexes with respect to Pclass
e = sns.FacetGrid(train , col = 'Embarked')
e.map(sns.pointplot, 'Pclass' , 'Survived' , 'Sex' ,  ci = 95.0 , palette = 'deep')
e.add_legend()


# In[21]:


#plot distribution of age of passengers who survived or did not survive
a = sns.FacetGrid(train , hue = 'Survived' , aspect = 4)
a.map(sns.kdeplot, 'Age' , shade = True)
a.set(xlim = (0 , train['Age'].max()))
a.add_legend()


# In[22]:


# histogram comparison of sex , class and age by survival
h = sns.FacetGrid(train , row = 'Sex' , col = 'Pclass' , hue = 'Survived')
h.map(plt.hist , 'Age' , alpha = 0.8)
h.add_legend()


# In[23]:


#pair plots of entire dataset
pp = sns.pairplot(train , hue = 'Survived' , palette = 'deep' , size = 2 , diag_kind = 'kde' , diag_kws = dict(shade = True),
                 plot_kws = dict(s=10))
pp.set(xticklabels = [])


# In[24]:


#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize = (16,16))
    colormap = sns.diverging_palette(220 , 10 , as_cmap = True)
    
    _  = sns.heatmap(
        df.corr(),
        cmap = colormap,
        square = True,
        cbar_kws = {'shrink' : 0.9},
        ax = ax,
        annot = True, 
        linewidths = 0.1 , vmax = 1.0 , linecolor = 'black',
        annot_kws = {'fontsize' : 12}
    )
    
    plt.title(' Pearson Correlation of features' , y = 1.05 , size = 15) 
    
correlation_heatmap(train)


# # Splitting Training and Testing Data
# 
# We split our training data because if we will train our model with 100% training data we will not be sure whether our model will give right predictions with the test data.For example if we split the training data into 75% and 25% and we train our model with the 75% then we can test our the predictions of our model by using the remaining 25% of the data.Testing data may or may be available to us, we have to make sure that our model is predicting the right predictions by the training data only.

# In[25]:


#split train and test data with function defaults
#random_state -> seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(train[train_x_calc], train[Target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(train[train_x_bin], train[Target] , random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(train_dummy[train_x_dummy], train[Target], random_state = 0)


print("Training Dataset Shape: {}".format(train.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))


# # Modelling the Data
# Now we will select and train machine learning models. We will use our training data to train them.

# In[26]:


#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]



#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
#note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = train[Target]

#index through MLA and save performance to table
row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, train[train_x_bin], train[Target], cv  = cv_split , return_train_score=True)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    

    #save MLA predictions - see section 6 for usage
    alg.fit(train[train_x_bin], train[Target])
    MLA_predict[MLA_name] = alg.predict(train[train_x_bin])
    
    row_index+=1

    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare


# In[27]:


#Plotting a barplot of Algorith vs Accuracy score
sns.barplot(x = 'MLA Test Accuracy Mean' , y = 'MLA Name' ,data = MLA_compare , color = 'r')
plt.title('Machine Learning Algorithm vs Accuracy Score')
plt.xlabel('Accuracy Score')
plt.ylabel('Machine Learning Algorithm')


# # Evaluating Model Performance

# In[28]:


#coin flip model with random 1/survived 0/died
#iterating over dataFrame rows as (index, Series) pairs

for index, row in train.iterrows(): 
    #random number generator: https://docs.python.org/2/library/random.html
    if random.random() > .5:     # Random float x, 0.0 <= x < 1.0    
        train.at[index, 'Random_Predict'] = 1 #predict survived/1
    else: 
        train.at[index, 'Random_Predict'] = 0 #predict died/0
#score random guess of survival. Use shortcut 1 = Right Guess and 0 = Wrong Guess
#the mean of the column will then equal the accuracy
train['Random_Score'] = 0
train.loc[(train['Survived'] == train['Random_Predict']), 'Random_Score'] = 1 #set to 1 for correct prediction
print('Coin Flip Model Accuracy: {:.2f}%'. format(train['Random_Score'].mean()*100))


#we can also use scikit's accuracy_score function to save us a few lines of code
print('Coin Flip Model Accuracy w/SciKit: {:.2f}%'.format(metrics.accuracy_score(train['Survived'], train['Random_Predict'])*100))


# In[29]:


#group by or pivot table
pivot_female = train[train.Sex=='female'].groupby(['Sex' , 'Pclass' , 'Embarked' , 'FareBin'])['Survived'].mean()
print('Survival Decision Tree w/Female Node: \n' , pivot_female)

pivot_male = train[train.Sex == 'male'].groupby(['Sex' , 'Title'])['Survived'].mean()
print('\n\nSurvival Decision Tree w/male Node: \n ', pivot_male)


# In[30]:


#Decision Tree Model Accuracy
def myTree(df):
    #initialize table to store predictions
    Model = pd.DataFrame(data = {'Predict':[]})
    male_title = ['Master'] 
    
    for index , rows in df.iterrows():
        #Question 1: Were you on the Titanic; majority died
        Model.loc[index ,'Predict'] =0
        #Question 2: Are you female; majority survived
        if (df.loc[index , 'Sex'] == 'female'):
            Model.loc[index , 'Predict'] == 1
            
         # Question 3 Female - FareBin; set anything less than .5 in female node decision tree back to 0 
        if ((df.loc[index , 'Sex'] == 'female')&
            (df.loc[index, 'Pclass'] == 3)&
            (df.loc[index , 'Embarked']== "S")&
            (df.loc[index , 'Fare'] > 8)
           
           ):
            Model.loc[index, 'Predict'] =0
            
        #Question 4 Male: Title; set anything greater than .5 to 1 for majority survived
        if ((df.loc[index, 'Sex'] == 'Male')&
           (df.loc[index , 'Title'] in male_title)
           ):
            Model.loc[index, 'Predict'] =1
    return Model
            
    #model data
Tree_Predict = myTree(train)
print('Decision Tree Model Accuracy/Precision Score: {:.2f}%\n'.format(metrics.accuracy_score(train['Survived'], Tree_Predict)*100))
    
            
print(metrics.classification_report(train['Survived'], Tree_Predict))
        
        


# In[31]:


#Plotting Accuracy Summary
import itertools
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion Matrix' , cmap = plt.cm.Blues):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1) [:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix without normalization')
        
    print(cm)
    
    plt.imshow(cm , interpolation = 'nearest' , cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks , classes , rotation = 45)
    plt.yticks(tick_marks, classes)
    
    fmt = '0.2f' if normalize else 'd'
    thresh = cm.max() /2.
    for i,j in itertools.product(range(cm.shape[0]) , range(cm.shape[1])):
        plt.text(j,i,format (cm[i,j] , fmt), horizontalalignment = 'center' , color = 'white' if cm[i,j]>thresh else 'black')
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#computing confusion matrix
cnf_matrix = metrics.confusion_matrix(train['Survived'] , Tree_Predict)
np.set_printoptions(precision = 2)
class_names = ['Dead' , 'Survived']

#Plotting a non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

#Plotting a normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix , classes = class_names ,normalize = True ,  title = 'Normalized Confusion Matrix ')

    
   
    


# # Tuning Model with Hyper - Parameters
# A hyperparameter is a parameter whose value is used to control the learning process.
# 
# 
# 
# We will tune our model using ParameterGrid, GridSearchCV, and customized sklearn scoring.

# In[32]:


#Base Model
dtree= tree.DecisionTreeClassifier(random_state = 0)
base_results = model_selection.cross_validate(dtree, train[train_x_bin] , train[Target] ,return_train_score = True, cv = cv_split )
dtree.fit(train[train_x_bin] , train[Target])

print('Before DT Parameters: ' , dtree.get_params())
print("BEFORE DT Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100 ))
print('Before DT Test w/bin score mean: {:.2f}'.format(base_results['test_score'].mean()*100))
print('Before DT Test w/bin score 3*std: +- {:.2f}'.format(base_results['test_score'].std()*100*3))
print('-'*20)

#Tuning Hyper - Parameters
param_grid = {'criterion' : ['gini' , 'entropy'] , 'max_depth':[2,4,6,8,10,None] , 'random_state':[0]}
tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier() , param_grid= param_grid , scoring ='roc_auc' ,return_train_score = True, cv = cv_split)
tune_model.fit(train[train_x_bin] , train[Target] )


print('After DT Parameters: ', tune_model.best_params_)
print('After DT Training w/bin score mean: {:.2f}' .format(tune_model.cv_results_['mean_train_score'] [tune_model.best_index_]*100))
print('After DT Test w/bin score mean: {:.2f}' .format(tune_model.cv_results_['mean_test_score'] [tune_model.best_index_]*100))
print('After DT Test w/bin score 3*std: +- {:.2f}' .format(tune_model.cv_results_['std_test_score'] [tune_model.best_index_]*100*3))


# # Tuning Model with Feature Selection
# 
# We will use recursive feature elimination (RFE) with cross validation (CV).

# In[33]:


#base model
print('Before DT RFE Training shape old: ' , train[train_x_bin].shape)
print('Before DT RFE Training columns old: ' , train[train_x_bin].columns.values)
print('Before DT RFE Training w/bin score mean: {:.2f}'.format(base_results['train_score'].mean()*100))
print('Before DT RFE Test w/bin score mean: {:.2f}'.format(base_results['test_score'].mean()*100))
print('Before DT RFE Test w/bin score 3*std: +- {:.2f}'.format(base_results['train_score'].std()*100*3))
print('-'*20)

#feature selection
dtree_rfe = feature_selection.RFECV(dtree, step = 1, scoring = 'accuracy'  ,cv=cv_split)
dtree_rfe.fit(train[train_x_bin] , train[Target])

#transforming x&y to reduced features and fit new model
X_rfe = train[train_x_bin].columns.values[dtree_rfe.get_support()]
rfe_results = model_selection.cross_validate(dtree, train[X_rfe] , train[Target] , return_train_score = True, cv=cv_split )

print('After DT RFE Training Shape new:' ,train[X_rfe].shape)
print('After DT RFE Training Columns new:' ,X_rfe)
print('After DT RFE Training w/bin score mean: {:.2f}'.format(rfe_results['train_score'].mean()*100))
print('After DT RFE Test w/bin score mean: {:.2f}'.format(rfe_results['test_score'].mean()*100))
print('After DT RFE Test w/bin score 3*std: +- {:.2f}'.format(rfe_results['train_score'].std()*100*3))
print('-' *20)

#Tuning RFE model
rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid = param_grid , scoring = 'roc_auc' , return_train_score = True, cv= cv_split)
rfe_tune_model.fit(train[X_rfe] , train[Target])

print('After DT RFE Tuned Parameters:' , rfe_tune_model.best_params_)
print('After DT RFE Tuned Training w/bin score mean: {:.2f}'.format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print('After DT RFE Tuned Test w/bin score mean: {:.2f}'.format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('After DT RFE Tuned Test w/bin score 3*std: +- {:.2f}'.format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100*3))





# In[34]:


#Graph MLA version of Decision Tree
import graphviz

dot_data  = tree.export_graphviz(dtree, out_file = None , feature_names = train_x_bin , class_names = True , filled = True, rounded = True)

graph = graphviz.Source(dot_data)
graph


# In[35]:


#compare altgorithm predictions with each other, where 1 = exactly similar and 0 = exactly opposite
correlation_heatmap(MLA_predict)


# In[36]:


#why choose one model, when you can pick them all with voting classifier
vote_est = [
    #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
    ('ada', ensemble.AdaBoostClassifier()),
    ('bc', ensemble.BaggingClassifier()),
    ('etc',ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),
     #Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc
    ('gpc', gaussian_process.GaussianProcessClassifier()),
    
    #GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ('lr', linear_model.LogisticRegressionCV()),
    
    #Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html
    ('bnb', naive_bayes.BernoulliNB()),
    ('gnb', naive_bayes.GaussianNB()),
    
    #Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
    ('knn', neighbors.KNeighborsClassifier()),
    
    #SVM: http://scikit-learn.org/stable/modules/svm.html
    ('svc', svm.SVC(probability=True)),
    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
   ('xgb', XGBClassifier())
]
#Hard Vote or majority rules
vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
vote_hard_cv = model_selection.cross_validate(vote_hard,train[train_x_bin], train[Target],return_train_score = True ,cv  = cv_split)
vote_hard.fit(train[train_x_bin], train[Target])

print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 
print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))
print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))
print('-'*10)


#Soft Vote or weighted probabilities
vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
vote_soft_cv = model_selection.cross_validate(vote_soft, train[train_x_bin], train[Target],return_train_score = True, cv  = cv_split)
vote_soft.fit(train[train_x_bin], train[Target])

print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 
print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))
print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))
print('-'*10)




# In[37]:


#tune each estimator before creating a super model
#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
grid_n_estimator = [50,100,300]
grid_ratio = [.1,.25,.5,.75,1.0]
grid_learn = [.01,.03,.05,.1,.25]
grid_max_depth = [2,4,6,None]
grid_min_samples = [5,10,.03,.05,.10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]

vote_param = [{
#            #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
            'ada__n_estimators': grid_n_estimator,
            'ada__learning_rate': grid_ratio,
            'ada__algorithm': ['SAMME', 'SAMME.R'],
            'ada__random_state': grid_seed,
    
            #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
            'bc__n_estimators': grid_n_estimator,
            'bc__max_samples': grid_ratio,
            'bc__oob_score': grid_bool, 
            'bc__random_state': grid_seed,
            
            #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
            'etc__n_estimators': grid_n_estimator,
            'etc__criterion': grid_criterion,
            'etc__max_depth': grid_max_depth,
            'etc__random_state': grid_seed,


            #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
            'gbc__loss': ['deviance', 'exponential'],
            'gbc__learning_rate': grid_ratio,
            'gbc__n_estimators': grid_n_estimator,
            'gbc__criterion': ['friedman_mse', 'mse', 'mae'],
            'gbc__max_depth': grid_max_depth,
            'gbc__min_samples_split': grid_min_samples,
            'gbc__min_samples_leaf': grid_min_samples,      
            'gbc__random_state': grid_seed,
    
            #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
            'rfc__n_estimators': grid_n_estimator,
            'rfc__criterion': grid_criterion,
            'rfc__max_depth': grid_max_depth,
            'rfc__min_samples_split': grid_min_samples,
            'rfc__min_samples_leaf': grid_min_samples,   
            'rfc__bootstrap': grid_bool,
            'rfc__oob_score': grid_bool, 
            'rfc__random_state': grid_seed,
        
            #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
            'lr__fit_intercept': grid_bool,
            'lr__penalty': ['l1','l2'],
            'lr__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'lr__random_state': grid_seed,
            
            #http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
            'bnb__alpha': grid_ratio,
            'bnb__prior': grid_bool,
            'bnb__random_state': grid_seed,
    
            #http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
            'knn__n_neighbors': [1,2,3,4,5,6,7],
            'knn__weights': ['uniform', 'distance'],
            'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'knn__random_state': grid_seed,
            
            #http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
            #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
            'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'svc__C': grid_max_depth,
            'svc__gamma': grid_ratio,
            'svc__decision_function_shape': ['ovo', 'ovr'],
            'svc__probability': [True],
            'svc__random_state': grid_seed,
    
    
            #http://xgboost.readthedocs.io/en/latest/parameter.html
            'xgb__learning_rate': grid_ratio,
            'xgb__max_depth': [2,4,6,8,10],
            'xgb__tree_method': ['exact', 'approx', 'hist'],
            'xgb__objective': ['reg:linear', 'reg:logistic', 'binary:logistic'],
            'xgb__seed': grid_seed    

        }]


# In[38]:


#Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]


grid_param = [
            [{
            #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
            'n_estimators': grid_n_estimator, #default=50
            'learning_rate': grid_learn, #default=1
            #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
            'random_state': grid_seed
            }],
       
    
            [{
            #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
            'n_estimators': grid_n_estimator, #default=10
            'max_samples': grid_ratio, #default=1.0
            'random_state': grid_seed
             }],

    
            [{
            #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
            'n_estimators': grid_n_estimator, #default=10
            'criterion': grid_criterion, #default=”gini”
            'max_depth': grid_max_depth, #default=None
            'random_state': grid_seed
             }],


            [{
            #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
            #'loss': ['deviance', 'exponential'], #default=’deviance’
            'learning_rate': [.05], #default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
            'n_estimators': [300], #default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
            #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
            'max_depth': grid_max_depth, #default=3   
            'random_state': grid_seed
             }],

    
            [{
            #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
            'n_estimators': grid_n_estimator, #default=10
            'criterion': grid_criterion, #default=”gini”
            'max_depth': grid_max_depth, #default=None
            'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
            'random_state': grid_seed
             }],
    
            [{    
            #GaussianProcessClassifier
            'max_iter_predict': grid_n_estimator, #default: 100
            'random_state': grid_seed
            }],
        
    
            [{
            #LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
            'fit_intercept': grid_bool, #default: True
            #'penalty': ['l1','l2'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs
            'random_state': grid_seed
             }],
            
    
            [{
            #BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
            'alpha': grid_ratio, #default: 1.0
             }],
    
    
            #GaussianNB - 
            [{}],
    
            [{
            #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
            'n_neighbors': [1,2,3,4,5,6,7], #default: 5
            'weights': ['uniform', 'distance'], #default = ‘uniform’
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }],
            
    
            [{
            #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
            #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
            #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [1,2,3,4,5], #default=1.0
            'gamma': grid_ratio, #edfault: auto
            'decision_function_shape': ['ovo', 'ovr'], #default:ovr
            'probability': [True],
            'random_state': grid_seed
             }],

    
            [{
            #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
            'learning_rate': grid_learn, #default: .3
            'max_depth': [1,2,4,6,8,10], #default 2
            'n_estimators': grid_n_estimator, 
            'seed': grid_seed  
             }]   
        ]



start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter
for clf, param in zip (vote_est, grid_param): #https://docs.python.org/3/library/functions.html#zip

    #print(clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm
    #print(param)
    
    
    start = time.perf_counter()        
    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, return_train_score = True , cv = cv_split, scoring = 'roc_auc')
    best_search.fit(train[train_x_bin], train[Target])
    run = time.perf_counter() - start

    best_param = best_search.best_params_
    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))
    clf[1].set_params(**best_param) 


run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes.'.format(run_total/60))

print('-'*10)


# In[39]:


#Hard Vote or majority rules w/Tuned Hyperparameters
grid_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
grid_hard_cv = model_selection.cross_validate(grid_hard, train[train_x_bin], train[Target],return_train_score = True, cv  = cv_split)
grid_hard.fit(train[train_x_bin], train[Target])

print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100)) 
print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))
print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))
print('-'*10)

#Soft Vote or weighted probabilities w/Tuned Hyperparameters
grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
grid_soft_cv = model_selection.cross_validate(grid_soft, train[train_x_bin], train[Target],return_train_score = True ,  cv  = cv_split)
grid_soft.fit(train[train_x_bin], train[Target])

print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100)) 
print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))
print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))
print('-'*10)


# In[40]:


#prepare data for modeling
print(test.info())
print("-"*10)
#data_val.sample(10)



#handmade decision tree - submission score = 0.77990
test['Survived'] = myTree(test).astype(int)

test['Survived'] = grid_hard.predict(test[train_x_bin])

#submit file
submit = test[['PassengerId','Survived']]
submit.to_csv("../working/submit.csv", index=False)

print('Validation Data Distribution: \n', test['Survived'].value_counts(normalize = True))
submit.sample(10)


# # Thank You
# If you have come this far thanks a lot for giving your time. This notebook was extremely challenging for me. Please give an upvote as it will motivate me.

# In[ ]:




