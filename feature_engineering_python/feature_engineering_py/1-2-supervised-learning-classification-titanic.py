#!/usr/bin/env python
# coding: utf-8

# ![Titanic3.jpg](attachment:b70988cf-37d2-4e06-8888-df1cab6c860d.jpg)

# ### Introduction
# I want to start my notbook with a confession. 
# 
# For a long time I could not understand why a person should study history. And only with time did I understand: history needs to be studied so that the catastrophes that were in the past do not happen in the future. 
# The disaster of the Titanic is perhaps the most infamous shipwreck in the history of mankind. It happened on April 15, 1912. During its maiden voyage, the Titanic sank after colliding with an iceberg, resulting in the deaths of 1,500 (approximate number) of the 2,224 passengers and crew. This is a very sad disaster that people still remember.
# 
# The Titanic dataset is a well-known dataset. Many newcomers to Kaggle begin their journey into data science by solving the legendary Titanic competition. In this kernel you will find my solution to this classification problem. Please consider upvoting if this is useful to you.

# # Table of contents:
# 1.Import
# 
#    1.1.Import of Required Modules
# 
#    1.2.Importing (Reading) Data
# 
# 2.Exploratory Data Analysis (EDA)
# 
#    2.1.Data Visualization
# 
# 3.Data Cleaning
# 
# 4.Feature Engineering / Feature Selection
# 
# 5.Machine Learning Models
# 
# 6.Evaluate & Interpret Results
# 
#    6.1.Cross Validation
# 
#    6.2.Grid search
# 
#    6.3.Confusion Matrix
# 
#    6.4.Additional metrics (precision, recall, F-measure)
# 
#    6.5.ROC Curve and AUC Score
# 
#    6.6.Ensembling:
#     
#    6.6.1.Random Forest
# 
#    6.6.2.Voting Classifier
# 
#    6.6.3.Bagging
# 
#    6.6.4.Boosting
# 
#    6.7.Neural networks (deep learning)
# 
#    6.8.Pipeline
# 
# 7.Creating Submission File
# 
# 8.(if necessary) Define the Question of Interest/Goal

# # 1.Import

# # 1.1.Import of Required Modules
# First, of course, we need to import several Python libraries, such as numpy, pandas, matplotlib and seaborn.

# In[1]:


# data analysis libraries 
import numpy as np 
import pandas as pd

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


# # 1.2.Importing (Reading) Data
# Let's read in training and testing data.

# In[2]:


df = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[3]:


test_df = pd.read_csv('/kaggle/input/titanic/test.csv')


# # 2.Exploratory Data Analysis (EDA)
# And now let's look at the training data.

# In[4]:


type(df)


# In[5]:


df.head(3)  # the head function returns the first (3) lines


# In[6]:


df.tail(3)  # the tail function returns the last (3) lines


# In[7]:


df.shape     # the shape function returns the size - this is a pair (number of rows, number of columns)


# In[8]:


df.dtypes    # the type function returns the types of all data from the Data Frame


# In[9]:


df.info()            # the info function provides general information about the DataFrame


# Let's look at the features in the dataset. We are interested in how complete they are.

# In[10]:


df.isnull().sum() # checking for null or missing values


# In[11]:


df.describe(include="all")


# # 2.1.Data Visualization
# Visualization is a very useful stage of our work. In the graphs, you can see what is hidden behind the numbers.

# Let's ask ourselves, are there many survivors?

# In[12]:


f,ax=plt.subplots(1,2,figsize=(12,4))
df['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=False)
ax[0].set_title('Survivors (1) and the dead (0)')
ax[0].set_ylabel('')
sns.countplot('Survived', data=df,ax=ax[1])
ax[1].set_ylabel('Quantity')
ax[1].set_title('Survivors (1) and the dead (0)')
plt.show()


# #### Let's ask ourselves, what types of data are we dealing with here?
# Categorical Features in our case: Sex, Embarked. 
# 
# Ordinal Features in our case: Pclass. 
# 
# Continuous Features in our case: Age.

# In[13]:


df.groupby(['Sex','Survived'])['Survived'].count()


# In[14]:


f,ax=plt.subplots(1,2,figsize=(12,4))
df[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survivors by sex')
sns.countplot('Sex',hue='Survived',data=df,ax=ax[1])
ax[1].set_ylabel('Quantity')
ax[1].set_title('Survived (1) and deceased (0): men and women')
plt.show()


# And what is the situation among passengers from different classes of cabins (Pclass)?

# In[15]:


f,ax=plt.subplots(1,2,figsize=(12,4))
df['Pclass'].value_counts().plot.bar(color=['purple','royalblue','orchid'],ax=ax[0])
ax[0].set_title('Total number of passengers by Pclass')
ax[0].set_ylabel('Quantity')
sns.countplot('Pclass',hue='Survived',data=df, ax=ax[1])
ax[1].set_ylabel('Quantity')
ax[1].set_title('Survived (1) and deceased (0): by Pclass')
plt.show()


# #### So, after looking at our data in EDA and Data Visualization, we can make some observations:
# There are a total of 891 passengers in our training set.
# 
# The age feature is absent in about 20% of its values. This characteristic is very important for survival. We will try to fill in these gaps.
# 
# 77% of its values are missing in the Cabin feature. This is a large number. It is difficult to fill in the missing values here. We will probably remove these values from our dataset.
# 
# The Embarked feature is missing 0.22% of its values that could be corrected.

# # 3.Data Cleaning

# #### PassengerId feature
# I think this feature is uninformative. We will delete it. Therefore, we will not do any cleaning of this feature.

# #### Survived feature
# This feature is informative. We'll keep it. This feature is already in a numerical form convenient for us. Therefore, we will not do any cleaning. There are no missing values in this feature.

# #### Pclass feature
# Here the situation is similar to Survived feature.

# #### Name feature
# This feature is a non-numeric feature. In this form, we will not be able to use this feature. There is no missing information in this feature. For now, we will leave this feature as it is.

# #### Sex feature
# Here the situation is similar to Name feature.

# #### Age feature
# The age feature has 177 zero values. We can assign such NaN values, for example, the average age of a data set. But this is not the best solution. There were many people of different ages on the Titanic. But we can look closely at the peculiarity of the name. Names have a greeting like Mr. or Mrs. We will use this information.

# In[16]:


df['Initial']=0
for i in df:
    df['Initial']=df.Name.str.extract('([A-Za-z]+)\.') # lets extract the Salutations


# We use here the Regex. It looks for strings which lie between A-Z or a-z and followed by a .(dot). We extract the Initials from the Name.

# In[17]:


pd.crosstab(df.Initial,df.Sex).T.style.background_gradient(cmap='BuPu') # Checking the Initials with the Sex


# There are some misspelled Initials like Miller or Me that stand for Miss. Let`s replace them with Miss and same thing for other values.

# In[18]:


df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                      ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],
                      inplace=True)


# In[19]:


df.groupby('Initial')['Age'].mean() # check the average age by Initials


# Filling NaN Ages

# In[20]:


# Assigning the NaN Values with the Ceil values of the mean ages
df.loc[(df.Age.isnull())&(df.Initial=='Mr'),'Age']=33
df.loc[(df.Age.isnull())&(df.Initial=='Mrs'),'Age']=36
df.loc[(df.Age.isnull())&(df.Initial=='Master'),'Age']=5
df.loc[(df.Age.isnull())&(df.Initial=='Miss'),'Age']=22
df.loc[(df.Age.isnull())&(df.Initial=='Other'),'Age']=46


# In[21]:


df.Age.isnull().any() # no null values left


# In[22]:


f,ax=plt.subplots(1,2,figsize=(12,4))
df[df['Survived']==0].Age.plot.hist(ax=ax[0],bins=20, edgecolor='black', color='firebrick')
ax[0].set_title('Survived (0)')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
df[df['Survived']==1].Age.plot.hist(ax=ax[1],color='royalblue',bins=20,edgecolor='black')
ax[1].set_title('Survived (1)')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()


# #### SibSp feature
# This feature is preserved as it is

# #### Parch feature
# This feature is preserved as it is

# #### Ticket feature
# This feature is a non-numeric feature. In this form, we will not be able to use this feature. There is no missing information in this feature. For now, we will leave this feature as it is.

# #### Fare feature
# There are no missing in this feature. This feature consists of numeric values. But they are very different. For now, we will leave this feature unchanged.

# In[23]:


print('Highest Fare was:',df['Fare'].max())
print('Lowest Fare was:',df['Fare'].min())
print('Average Fare was:',df['Fare'].mean())


# #### Cabin feature
# A lot of NaN values (687) and also many passengers have multiple cabins. We will probably need to drop this feature.

# #### Embarked feature
# There are 2 missing values. But they can be filled in.

# In[24]:


pd.crosstab([df.Embarked,df.Pclass],[df.Sex,df.Survived],margins=True).style.background_gradient(cmap='BuPu')


# As we saw that maximum passengers boarded from Port S, we replace NaN with S.

# In[25]:


df['Embarked'].fillna('S',inplace=True)


# In[26]:


df.Embarked.isnull().any()# now no NaN values


# # 4.Feature Engineering / Feature Selection
# Let's create a heat map and see how different features correlate with each other.

# In[27]:


plt.figure(figsize=(12,4))
sns.heatmap(df.corr(), annot=True, center=0, linewidths=.5, fmt='.2f', vmin=-1, vmax=1, cmap='BuPu')


# The heat map shows how these objects correlate. But in fact they are not strongly correlated. The highest correlation is observed between SibSp and Patch. From this we conclude that we can continue working with all the features.

# #### PassengerId feature
# Cannot be categorised.

# In[28]:


df.drop(['PassengerId'],axis=1,inplace=True)


# #### Survived feature
# This feature is preserved as it is.

# #### Pclass feature
# This feature is preserved as it is

# #### Name feature

# In[29]:


df['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)


# We don't need name feature as it cannot be converted into any categorical value.

# In[30]:


df.drop(['Name'],axis=1,inplace=True)


# #### Sex feature

# In[31]:


df['Sex'].replace(['male','female'],[0,1],inplace=True)


# #### Age feature

# In[32]:


df['Age_group']=0
df.loc[df['Age']<=16,'Age_group']=0
df.loc[(df['Age']>16)&(df['Age']<=32),'Age_group']=1
df.loc[(df['Age']>32)&(df['Age']<=48),'Age_group']=2
df.loc[(df['Age']>48)&(df['Age']<=64),'Age_group']=3
df.loc[df['Age']>64,'Age_group']=4
df.head(3)


# In[33]:


df['Age_group'].value_counts().to_frame().style.background_gradient(cmap='BuPu')#checking the number of passenegers in each band


# We have the Age_group feature. So we can drop the Age feature.

# In[34]:


df.drop(['Age'],axis=1,inplace=True)


# #### SibSp feature
# This feature represents whether a person is alone or with his family members.
# 
# Sibling = brother, sister, stepbrother, stepsister
# 
# Spouse = husband, wife

# In[35]:


pd.crosstab([df.SibSp],df.Survived).style.background_gradient(cmap='BuPu')


# In[36]:


pd.crosstab(df.SibSp,df.Pclass).style.background_gradient(cmap='BuPu')


# #### Parch feature

# In[37]:


pd.crosstab(df.Parch,df.Pclass).style.background_gradient(cmap='BuPu')


# #### Ticket feature
# It is any random string that cannot be categorised.

# In[38]:


df.drop(['Ticket'],axis=1,inplace=True)


# #### Fare feature

# In[39]:


df['Fare_span']=pd.qcut(df['Fare'],4)
df.groupby(['Fare_span'])['Survived'].mean().to_frame().style.background_gradient(cmap='BuPu')


# In[40]:


df['Fare_gap']=0
df.loc[df['Fare']<=7.91,'Fare_gap']=0
df.loc[(df['Fare']>7.91)&(df['Fare']<=14.454),'Fare_gap']=1
df.loc[(df['Fare']>14.454)&(df['Fare']<=31),'Fare_gap']=2
df.loc[(df['Fare']>31)&(df['Fare']<=513),'Fare_gap']=3


# We have the Fare_gap feature. So we can do this:

# In[41]:


df.drop(['Fare'],axis=1,inplace=True)


# In[42]:


df.drop(['Fare_span'],axis=1,inplace=True)


# #### Cabin feature
# A lot of NaN values and also many passengers have multiple cabins. So this is a useless feature.

# In[43]:


df.drop(['Cabin'],axis=1,inplace=True)


# #### Embarked feature

# In[44]:


df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)


# Looking at correlations of the original variables after Feature Engineering / Feature Selection:

# In[45]:


sns.heatmap(df.corr(),annot=True,cmap='BuPu',linewidths=0.2,annot_kws={'size':10})
fig=plt.gcf()
fig.set_size_inches(12,4)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# In[46]:


df


# # 5.Machine Learning Models
# Now we will predict whether the passenger will survive or not using some classification algorithms. I will use the following algorithms to create the model:
# 
# 1)K-Nearest Neighbours
# 
# 2)Logistic Regression
# 
# 3)Support Vector Machines(Linear)
# 
# 4)Kernel Support Vector Machines (kernel-SVM or rbf-SVM)
# 
# 5)Naive Bayes
# 
# 6)Decision Tree

# In[47]:


# importing the required ML packages
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.linear_model import LogisticRegression # logistic regression
from sklearn.svm import LinearSVC # Linear Support Vector Machine (linear-SVM)
from sklearn.svm import SVC # Kernel Support Vector Machines (kernel-SVM or rbf-SVM)
from sklearn.naive_bayes import GaussianNB # Naive bayes
from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.ensemble import GradientBoostingClassifier # Gradient boosting of regression trees
from sklearn.neural_network import MLPClassifier # Multilayer Perceptrons (MLP)
from sklearn.model_selection import train_test_split # training and testing data split
from sklearn import metrics # accuracy measure
from sklearn.metrics import confusion_matrix # it's for confusion matrix


# Dividing the original DataFrame by X_trail, X_test, y_train, y_test in a certain proportion (in our case 70% and 30%) from sklearn.model_selection import train_test_split : X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 11)

# In[48]:


train,test=train_test_split(df,test_size=0.3,random_state=11,stratify=df['Survived'])
X_train=train[train.columns[1:]]
y_train=train[train.columns[:1]]
X_test=test[test.columns[1:]]
y_test=test[test.columns[:1]]
X=df[df.columns[1:]]
y=df['Survived']


# #### K-Nearest Neighbours (KNN)

# In[49]:


knn=KNeighborsClassifier() 
knn.fit(X_train,y_train)
prediction_knn=knn.predict(X_test)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction_knn,y_test))


# In[50]:


test_accuracy = []
# trying n_neighbors from 1 to 30
neighbors_settings = range(1, 31)
for n_neighbors in neighbors_settings:
    # we are building a model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # we are recording the accuracy for the test set
    test_accuracy.append(knn.score(X_test, y_test))

plt.plot(neighbors_settings, test_accuracy, label="accuracy on the test set",color='royalblue')
plt.ylabel("accuracy")
plt.xlabel("k - number of neighbors")
plt.legend()
print('Accuracies for different values of k are:', test_accuracy)
print('max accuracy =',max(test_accuracy))


# #### Logistic Regression

# In[51]:


logreg = LogisticRegression()
logreg.fit(X_train,y_train)
prediction_logreg=logreg.predict(X_test)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction_logreg,y_test))


# #### Linear Support Vector Machine (linear-SVM)

# In[52]:


linear_svm=LinearSVC()
linear_svm.fit(X_train,y_train)
prediction_linear_svm=linear_svm.predict(X_test)
print('Accuracy for linear SVM on the training set is', linear_svm.score(X_train, y_train))
print('Accuracy for linear SVM on the test set is', linear_svm.score(X_test, y_test))


# #### Kernel Support Vector Machines (kernel-SVM or rbf-SVM)

# In[53]:


kerSVM=SVC(kernel='rbf', C=10, gamma=0.1, random_state=1)
kerSVM.fit(X_train,y_train)
prediction_kerSVM=kerSVM.predict(X_test)
print('Accuracy for kernal SVM is',metrics.accuracy_score(prediction_kerSVM,y_test))


# #### Gaussian Naive Bayes

# In[54]:


gnb=GaussianNB()
gnb.fit(X_train,y_train)
prediction_gnb=gnb.predict(X_test)
print('The accuracy of the NaiveBayes is',metrics.accuracy_score(prediction_gnb,y_test))


# #### Decision Tree

# In[55]:


dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
prediction_dt=dt.predict(X_test)
print('The accuracy of the DecisionTree is',metrics.accuracy_score(prediction_dt,y_test))


# # 6.Evaluate & Interpret Results

# # 6.1.Cross Validation
# Cross-validation is a statistical method of assessing generalizing ability, which is more stable and thorough than splitting data into training and test sets. In cross-validation, the data is split several times and several models are built. The most commonly used variant of cross–validation is k-fold cross-validation, in which k is a user–defined number, usually 5 or 10. When performing a five-block cross-validation, the data is first split into five parts of (approximately) the same size, called folds.
# 
# Then a sequence of models is constructed. The first model is trained using block 1 as a test set, and the remaining folds (2-5) act as a training set. The model is built on the basis of data located in folds 2-5, and then its accuracy is evaluated on the data of fold 1. Then the second model is trained, this time fold 2 is used as a test set, and the data in folds 1, 3, 4, and 5 serve as a training set. This process is repeated for folds 3, 4 and 5, which act as test suites. For each of these five splits of data into training and test sets, we calculate the accuracy. As a result, we fixed five values of accuracy. By default, cross_vol_score performs a three-fold cross-validation, returning three accuracy values. We can change the number of folds by setting a different value for the cv parameter.
# 
# The most common way to summarize the accuracy calculated during cross–validation is to calculate the average value.
# 
# In scikit-learn, cross-validation is implemented using the cross_vol_score function of the model selection module. Let's show how it works using the example of the Kernel Support Vector Machines (kernel-SVM or rbf-SVM) model.

# In[56]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(kerSVM, X, y) # We can change the number of folds by setting a different value for the cv parameter
print('cross-validation (CV) values: {}'.format(scores))


# In[57]:


print('cross-validation (CV) mean for kernel-SVM: {:.2f}'.format(scores.mean()))


# Scikit-learn allows you to configure the cross-validation process much more precisely by using the cross-validation splitter as a cv parameter. To do this, we must first import the Fold class from the model_selection module and create an instance of it by specifying the required number of folds:

# In[58]:


from sklearn.model_selection import KFold
# we can pass the fold partitioning generator as a cv parameter to the cross_vol_score function
kfold = KFold(n_splits=10,random_state=7, shuffle=True)  # 
print('cross-validation (CV) values:\n{}'.format(cross_val_score(kerSVM, X, y, cv=kfold)))


# In[59]:


print('cross-validation (CV) mean for kernel-SVM:\n{}'.format(cross_val_score(kerSVM, X, y, cv=kfold).mean()))


# We can find cross-validation (CV) mean accuracy for all models:

# In[60]:


from sklearn.model_selection import KFold             # for K-fold cross validation
from sklearn.model_selection import cross_val_score   # score evaluation
from sklearn.model_selection import cross_val_predict # prediction
# we can pass the fold partitioning generator as a cv parameter to the cross_vol_score function
kfold = KFold(n_splits=10, random_state=7, shuffle=True)   # mix and split the data into 10 equal parts
cv_mean =[]
acc_all =[]
std =[]
classifiers=['KNN','Logistic Regression','Linear SVM', 'kernel-SVM','Naive Bayes','Decision Tree']
models=[KNeighborsClassifier(n_neighbors=20), LogisticRegression(), LinearSVC(),  
     SVC(kernel='rbf', C=10, gamma=0.1, random_state=1) , GaussianNB(), DecisionTreeClassifier()]
for i in models:
    model = i
    cv_res = cross_val_score(model,X,y, cv = kfold, scoring = "accuracy")   
    cv_res = cv_res
    cv_mean.append(cv_res.mean())
    std.append(cv_res.std())
    acc_all.append(cv_res)
acc_mean =pd.DataFrame({'cross-validation (CV) mean': cv_mean,'standard deviation (Std)': std},index=classifiers)   
acc_mean


# In[61]:


acc_mean['cross-validation (CV) mean'].plot.bar(width=0.8, color='royalblue')
plt.title('Cross-Validation Mean Accuracy')
fig=plt.gcf()
fig.set_size_inches(8,5)
plt.show()


# # 6.2.Grid search
# Finding the optimal values of the key parameters of the model (that is, the values that give the best generalizing ability) is a difficult task, but it is mandatory for almost all models and datasets. Since the search for optimal parameter values is a common task, the scikit-learn library offers standard methods to solve it. The most commonly used method is grid search, which is essentially an attempt to sort through all possible combinations of parameters of interest.

# In[62]:


best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.1, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]:
        # for each combination of parameters, we train the SVC
        svm = SVC(kernel='rbf',gamma=gamma, C=C) 
        svm.fit(X_train, y_train)
        # we evaluate the quality of the SVC on the test set
        score = svm.score(X_test, y_test)
        # if we get the best accuracy value, we save the value and parameters
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}
print('The best value of accuracy: {:.6f}'.format(best_score))
print('Best parameter values: {}'.format(best_parameters))


# Let's make sure that the parameters found are the best, that is, they give the greatest accuracy:

# In[63]:


kerSVM=SVC(kernel='rbf', C=8.0, gamma=0.1, random_state=1) # C=1.5, gamma=0.1
kerSVM.fit(X_train,y_train)
prediction_kerSVM=kerSVM.predict(X_test)
print('Accuracy for kernal SVM is',metrics.accuracy_score(prediction_kerSVM,y_test))


# In[64]:


svm_max = kerSVM


# ### Grid search with cross-validation

# In[65]:


from sklearn.model_selection import GridSearchCV


# In[66]:


# we divide the data into a training + test set and a test set
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)
# we divide the training + test set into training and test sets
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)


# In[67]:


for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # for each combination of parameters,
        # we train SVC
        kerSVM=SVC(kernel='rbf', C=C, gamma=gamma, random_state=1) 
        # performing a cross-validation
        scores = cross_val_score(kerSVM, X_trainval, y_trainval, cv=5)
        # we calculate the mean accuracy of the cross-validation
        score = np.mean(scores)
        # if we get the best accuracy value, we save the value and parameters
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}
# re-building the model on the set, 
# obtained as a result of combining training and verification data
kerSVM = SVC(**best_parameters)
kerSVM.fit(X_trainval, y_trainval)


# In[68]:


param_grid = {'C': [0.001, 0.01, 0.1, 1.5, 10, 100],'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
print('Parameter Grid:\n{}'.format(param_grid))


# Now we can create an instance of the GridSearchCV class by passing the model (SVC), the grid of the desired parameters (param_grid), as well as the cross-validation strategy that we want to use (for example, a five-fold stratified cross-validation):

# In[69]:


grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)


# In[70]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[71]:


grid_search.fit(X_train, y_train)


# In[72]:


print('Accuracy on the test set: {:.2f}'.format(grid_search.score(X_test, y_test)))


# In[73]:


print('Best parameter values: {}'.format(grid_search.best_params_))
print('The best value is cross-validation. correctness: {:.2f}'.format(grid_search.best_score_))


# In[74]:


print('The best model:\n{}'.format(grid_search.best_estimator_))


# # 6.3.Confusion Matrix
# An Confusion Matrix is a table or diagram showing the accuracy of predicting a classifier with respect to two or more classes. The classifier's predictions are on the X axis, and the classes on the Y axis.
# 
# The table cells are filled with the number of classifier predictions. Correct predictions go diagonally from the upper left corner to the lower right.

# In[75]:


f,ax=plt.subplots(2,3,figsize=(13,11))
kfold = KFold(n_splits=10, random_state=7, shuffle=True)   # mix and split the data into 10 equal parts

y_pred=cross_val_predict(knn,X_test,y_test,cv=kfold)
sns.heatmap(confusion_matrix(y_test,y_pred),ax=ax[0,0],annot=True, cmap='BuPu',fmt='2.0f')  
ax[0,0].set_title('Matrix for KNN')

y_pred = cross_val_predict(logreg,X_test,y_test,cv=kfold)
sns.heatmap(confusion_matrix(y_test,y_pred),ax=ax[0,1],annot=True, cmap='BuPu',fmt='2.0f')
ax[0,1].set_title('Matrix for Logistic Regression')

y_pred = cross_val_predict(linear_svm,X_test,y_test,cv=kfold)
sns.heatmap(confusion_matrix(y_test,y_pred),ax=ax[0,2],annot=True, cmap='BuPu', fmt='2.0f')
ax[0,2].set_title('Matrix for linear-SVM')

y_pred = cross_val_predict(kerSVM,X_test,y_test,cv=kfold)
sns.heatmap(confusion_matrix(y_test,y_pred),ax=ax[1,0],annot=True, cmap='BuPu', fmt='2.0f')
ax[1,0].set_title('Matrix for kernel-SVM or rbf-SVM')

y_pred = cross_val_predict(gnb,X_test,y_test,cv=kfold)
sns.heatmap(confusion_matrix(y_test,y_pred),ax=ax[1,1],annot=True, cmap='BuPu', fmt='2.0f')
ax[1,1].set_title('Matrix for Gaussian Naive Bayes')

y_pred = cross_val_predict(dt,X_test,y_test,cv=kfold)
sns.heatmap(confusion_matrix(y_test,y_pred),ax=ax[1,2],annot=True, cmap='BuPu', fmt='2.0f')
ax[1,2].set_title('Matrix for Decision Tree')

plt.subplots_adjust(hspace=0.2,wspace=0.2)
plt.show()


# The elements of the main diagonal of the Confision Matrix correspond to the correct predictions (classification results). The upper-left forecast is called true negative (TN). The lower right forecast is called true positive (TP). the remaining elements show how many examples belonging to one class were mistakenly classified as another class. The lower left forecast is called false negative (FN). The upper right forecast is called false positive (FP).

# # 6.4.Additional metrics (precision, recall, F-measure)
# In addition to accuracy, there are other metrics for classification obtained using TP, FP, TN and FN.
# 
# Accuracy = (TP + TN)/(TP + TN + FP + FN)
# 
# Precision = TP/(TP + FP)
# 
# Recall = TP/(TP + FN)
# 
# F-measure = (2 * Precision * Recall)/(Precision + Recall) # also known as the f1-measure

# In[76]:


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score


# In[77]:


kerSVM=SVC()
kerSVM.fit(X_train, y_train)
y_pred=kerSVM.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, y_pred),
                index = [['Actual class', 'Actual class'], ['Negative', 'Positive']],
                columns = [['Predicted class', 'Predicted class'], ['Negative', 'Positive']])


# In[78]:


print('Accuracy   :', accuracy_score(y_test, y_pred))
print('Precision  :', precision_score(y_test, y_pred))
print('Recall     :', recall_score(y_test, y_pred))
print('f1-measure :', f1_score(y_test, y_pred))


# sklearn has a convenient classification_report function that returns recall, precision and F-measure for each of the classes, as well as the number of instances of each class.

# In[79]:


print(classification_report(y_test, y_pred))


# # 6.5.ROC Curve and AUC Score
# The ROC curve (Receiver Operating Characteristic curve) allows us to consider all threshold values for a given classifier, but instead of precision and recall, it shows the proportion of false positive rate (FPR) in comparison with the proportion of true positive rate (TPR). True positive rate (TPR) is just another name for recall. False positive rate (FPR) is the proportion of false positive examples from the total number of negative examples:
# 
# Recall = TPR = TP/(TP + FN)
# 
# FPR = FP /(FP + TN)

# In[80]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, kerSVM.decision_function(X_test))
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR (Recall)')
# we find the threshold value closest to zero
close_zero = np.argmin(np.abs(thresholds))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
label='threshold 0', fillstyle='none', c='purple', mew=2)
plt.legend(loc=4)


# We can calculate the Area Under Curve (AUC) - the area under ROC using the roc_auc_score function:

# In[81]:


from sklearn.metrics import roc_auc_score
kerSVM_auc = roc_auc_score(y_test, kerSVM.decision_function(X_test))
print('AUC for kerSVM: {:.3f}'.format(kerSVM_auc))


# # 6.6.Ensembling
# Ensembles are methods that combine multiple machine learning models to eventually produce a more powerful model.

# # 6.6.1.Random Forests
# In fact, a random forest is a set of decision trees, where each tree is slightly different from the others. The idea of a random forest is that each tree can predict pretty well, but most likely retrains on parts of the data. If we build many trees that work well and are retrained to varying degrees, we can reduce overtraining by averaging their results.

# In[82]:


rf=RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)
prediction_rf=rf.predict(X_test)
print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction_rf,y_test))


# #### Grid search for Random Forest

# In[83]:


best_score = 0
for n_estimators in [5, 10, 20, 40, 60, 80, 100, 120]:
    for max_depth in [2,3,7,11,15]:
        # for each combination of parameters, we train the SVC
        rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
        rf.fit(X_train, y_train)
        # evaluating the quality of Random Forest Classifier on a test set
        score = rf.score(X_test, y_test)
        # if we get the best accuracy value, we save the value and parameters
        if score > best_score:
            best_score = score
            best_parameters = {'n_estimators': n_estimators, 'max_depth': max_depth}
print('The best value of accuracy: {:.2f}'.format(best_score))
print('Best parameter values: {}'.format(best_parameters))


# # 6.6.2.Voting Classifier
# The Scikit-learn library has a VotingClassifier, which is an excellent tool for using several machine learning models that are not similar to each other at once and combining them into one classifier. This reduces the risk of retraining, as well as incorrect interpretation of the results of any one single model.
# 
# Among the VotingClassifier parameters there is a voting parameter with two possible values: 'hard' and 'soft'.

# In[84]:


from sklearn.ensemble import VotingClassifier
clf1 = KNeighborsClassifier() 
clf2 = LogisticRegression()
clf3 = SVC(kernel='linear')
clf4 = SVC(kernel='rbf')
clf5 = GaussianNB()
clf6 = DecisionTreeClassifier()
clf7 = RandomForestClassifier()

ensemble_hard=VotingClassifier(estimators=[('knn',KNeighborsClassifier(n_neighbors=20)),
                                              ('lr',LogisticRegression(C=0.05)),
                                              ('svm',SVC(kernel='linear',probability=True)),
                                              ('rbf',SVC(probability=True,kernel='rbf',C=1.5,gamma=0.1)),
                                              ('nb',GaussianNB()),
                                              ('dt',DecisionTreeClassifier(random_state=0)),
                                              ('rf',RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0))
                                            ],
                       voting='hard').fit(X_train,y_train)
print('The accuracy for ensembled model is:',ensemble_hard.score(X_test,y_test))
cross=cross_val_score(ensemble_hard,X, y, cv = 10,scoring = "accuracy")
print('The cross validated score is',cross.mean())


# # 6.6.3.Bagging
# Bagging (from Bootstrap aggregation) is a simple and very powerful ensemble method that combines predictions from multiple Machine learning methods together to predict more accurately than any single model.
# 
# Bagging is based on the statistical bootstrap method, which allows you to evaluate many statistics of complex distributions. Bootstrap aggregation is a procedure that is used to reduce the excessive variance of algorithms.

# In[85]:


# Bagging for KNN
from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=20),random_state=11,n_estimators=100)
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction,y_test))
result=cross_val_score(model,X,y,cv=10,scoring='accuracy')
print('The cross validated score for bagged KNN is:',result.mean())


# In[86]:


# Bagging for DecisionTree
model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,n_estimators=100)
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print('The accuracy for bagged Decision Tree is:',metrics.accuracy_score(prediction,y_test))
result=cross_val_score(model,X,y,cv=10,scoring='accuracy')
print('The cross validated score for bagged Decision Tree is:',result.mean())


# # 6.6.4.Boosting
# Is it possible to get one strong model from a large number of relatively weak and simple models? By weak models we mean such models whose accuracy can be only slightly higher than random guessing.
# 
# As Boosting, we will give an example of Adaboost.

# #### AdaBoost

# In[87]:


from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(n_estimators=100,random_state=0,learning_rate=0.1)
result=cross_val_score(ada,X,y,cv=10,scoring='accuracy')
print('The cross validated score for AdaBoost is:',result.mean())


# Gradient boosting of regression trees is another ensemble method that combines many trees to create a more powerful model. Despite the word "regression" in the name, these models can be used for regression and classification. Unlike a random forest, gradient boosting builds a sequence of trees in which each tree tries to correct the mistakes of the previous one.

# #### Gradient boosting of regression trees

# In[88]:


gbrt = GradientBoostingClassifier(random_state=0) 
gbrt.fit(X_train, y_train)
prediction_gbrt=gbrt.predict(X_test)
print('The accuracy of the Gradient boosting is',metrics.accuracy_score(prediction_gbrt,y_test))


# # 6.7.Neural networks (deep learning)
# The family of algorithms known as neural networks has recently experienced its resurgence under the name "deep learning". Despite the fact that deep learning promises great prospects in various fields of machine learning applications, deep learning algorithms, as a rule, are rigidly tied to specific use cases. In my notebook, I want to give an example of relatively simple methods, namely multilayer perceptrons for classification and regression. Multilayer perceptrons (MLP) are also called simple (vanilla) neural networks, and sometimes just neural networks.

# #### Multilayer Perceptrons (MLP)

# In[89]:


mlp=MLPClassifier(random_state=10)
mlp.fit(X_train,y_train)
prediction_mlp=mlp.predict(X_test)
print('The accuracy of the MLP is',metrics.accuracy_score(prediction_mlp,y_test))


# # 6.8.Pipeline
# Finally, we need to talk about the Pipeline class, a tool that allows you to combine several stages of preprocessing into one chain. In reality, machine learning projects rarely consist of just one model, most often they are a sequence of preprocessing steps. Pipelines allows you to encapsulate several stages into one python object that supports the already familiar scikit-learn interface, offering to use the fit, predict, transform methods.

# In[90]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', MinMaxScaler()), ('kerSVM', SVC())])


# In[91]:


pipe.fit(X_train, y_train)


# In[92]:


print('Accuracy on the test set: {:.2f}'.format(pipe.score(X_test, y_test)))


# #### Convenient make_pipeline function

# In[93]:


from sklearn.pipeline import make_pipeline
# standard syntax:
# pipe_long = Pipeline([('scaler', MinMaxScaler(), ('kerSVM', SVC(C=100))])
# short syntax:
pipe = make_pipeline(StandardScaler(), SVC(C=100))


# In[94]:


pipe.fit(X_train, y_train)


# In[95]:


print('Accuracy on the test set: {:.2f}'.format(pipe.score(X_test, y_test)))


# # 7.Creating Submission File

# In[96]:


test_df


# In[97]:


pessId = test_df['PassengerId']


# In[98]:


test_df.info()


# In[99]:


test_df.isnull().sum()   # we get the total number of NaN elements in X


# In[100]:


test_df['Initial']=0
for i in test_df:
    test_df['Initial']=test_df.Name.str.extract('([A-Za-z]+)\.') # lets extract the Salutations


# In[101]:


test_df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                      ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],
                      inplace=True)


# In[102]:


# Assigning the NaN Values with the Ceil values of the mean ages
test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Mr'),'Age']=33
test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Mrs'),'Age']=36
test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Master'),'Age']=5
test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Miss'),'Age']=22
test_df.loc[(test_df.Age.isnull())&(test_df.Initial=='Other'),'Age']=46


# In[103]:


test_df.Age.isnull().any() # no null values left


# In[104]:


test_df['Embarked'].fillna('S',inplace=True)


# In[105]:


test_df.Embarked.isnull().any()# now no NaN values


# In[106]:


test_df.drop(['PassengerId'],axis=1,inplace=True)


# In[107]:


test_df['Initial'].replace(['Mr','Mrs','Miss','Master','Other', 'Dona'],[0,1,2,3,4,5],inplace=True)


# In[108]:


test_df.drop(['Name'],axis=1,inplace=True)


# In[109]:


test_df['Sex'].replace(['male','female'],[0,1],inplace=True)


# In[110]:


test_df['Age_group']=0
test_df.loc[test_df['Age']<=16,'Age_group']=0
test_df.loc[(test_df['Age']>16)&(test_df['Age']<=32),'Age_group']=1
test_df.loc[(test_df['Age']>32)&(test_df['Age']<=48),'Age_group']=2
test_df.loc[(test_df['Age']>48)&(test_df['Age']<=64),'Age_group']=3
test_df.loc[test_df['Age']>64,'Age_group']=4
test_df.head(3)


# In[111]:


test_df.drop(['Age'],axis=1,inplace=True)


# In[112]:


test_df.drop(['Ticket'],axis=1,inplace=True)


# In[113]:


test_df['Fare_span']=pd.qcut(test_df['Fare'],4)


# In[114]:


test_df['Fare_gap']=0
test_df.loc[test_df['Fare']<=7.91,'Fare_gap']=0
test_df.loc[(test_df['Fare']>7.91)&(test_df['Fare']<=14.454),'Fare_gap']=1
test_df.loc[(test_df['Fare']>14.454)&(test_df['Fare']<=31),'Fare_gap']=2
test_df.loc[(test_df['Fare']>31)&(test_df['Fare']<=513),'Fare_gap']=3


# In[115]:


test_df.drop(['Fare'],axis=1,inplace=True)


# In[116]:


test_df.drop(['Fare_span'],axis=1,inplace=True)


# In[117]:


test_df.drop(['Cabin'],axis=1,inplace=True)


# In[118]:


test_df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)


# In[119]:


test_df


# In[120]:


test_X = test_df


# In[121]:


pred_y = svm_max.predict(test_X)   # the prediction method makes a prediction for the test set specified by the kaggle.com


# In[122]:


submission = pd.DataFrame({
         'PassengerId': pessId,
         'Survived': pred_y
     })
submission
submission.to_csv('titanic.csv', index=False)


# # 8.(if necessary) Define the Question of Interest/Goal
# I did not perform this item in this notebook.

# I sincerely thank you for your interest in my notebook.
# 
# I would be grateful for any feedback!
