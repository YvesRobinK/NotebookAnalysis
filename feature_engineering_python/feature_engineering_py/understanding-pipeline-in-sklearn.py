#!/usr/bin/env python
# coding: utf-8

# <div style="color:black;
#            display:fill;
#            border-radius:1px;
#            background-color:#87ceeb;
#            align:center;">
# 
# <p style="padding: 1px;
#               color:white;">
# <center>
#     
# # Understanding Pipeline in sklearn

# ## Table of Contents
# 
# * [1. Introduction to Pipeline](#1)
# * [2. Examples](#2)
#      - [2.1 : Example1: Linear Regression on Sinusoids](#3)
#      - [2.2 : Example2: Cancer dataset](#4)  
#      - [2.3 : Example3: Titanic dataset](#5)  
#        - [2.3.1 : Library and data import](#6)
#        - [2.3.2 : Define pipelines](#7)
#        - [2.3.3 : Define final pipeline and predict](#8)
#      - [2.4 : Example4: Iris dataset](#9)
#   
# * [3. References](#10)

# <div style="color:black;
#            display:fill;
#            border-radius:1px;
#            background-color:#e4f2f8;
#            align:center;">
# 
# <p style="padding: 1px;
#               color:white;">
# <center>
#     
# <a id="1"></a>
# ## 1. Introduction to Pipeline

# Machine Learning problem commonly involves two steps. First, we sequentially transform the data comprising of several steps such as feature transformation,dimensionality reduction, standardization etc. Secondly, we learn from the data by using an estimator or regressor to gain insights.
# 
# Pipeline simplifies the use of Machine learning model by combining various data transformation part with the data estimation part into single unit.In this notebook, We will illustrate the use of pipeline in Sci-Kit Learn library through examples.
# 
# 
# ![image.png](attachment:60655edc-141c-4c7e-8211-37ce4af86a24.png)

# <div style="color:black;
#            display:fill;
#            border-radius:1px;
#            background-color:#e4f2f8;
#            align:center;">
# 
# <p style="padding: 1px;
#               color:white;">
# <center>
#     
# <a id="2"></a>
# ## 2. Examples

# <a id="3"></a>
# ### 2.1: Example1: Linear Regression on Sinusoids 

# ##### Generate Training Data
# 
# - For the training data, we will use single feature as input and the response is a      
#   sinusoidal function of input feature (Y = X + sin(X))
# - We will add some noise to the response to make it realistic for later predictions

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
X_train = np.linspace(0, 10 * np.pi, num=1000); # input feature
noise = np.random.normal(scale=3, size=X_train.size); # additive noise
y_train = X_train + 10 * np.sin(X_train)+ noise; # response variable
plt.scatter(X_train,y_train,color='g'); # visualize the dataset
plt.xlabel('training feature');
plt.ylabel('training response');


# ##### Generate Test Data

# In[2]:


X_test = np.linspace(10, 15 * np.pi, num=100); # test data is again a linear array
noise = np.random.normal(scale=3, size=X_test.size); # we add noise sameway as we did for train
y_true = X_test + 10 * np.sin(X_test)+ noise; # true response desired from test data
plt.scatter(X_test,y_true,color='g'); # visualize test feature and test response
plt.xlabel('test feature');
plt.ylabel('test response');


# ##### Linear Regression (without using feature engineering)
# 
# - Now let us blindly use linear regression on the example1 data to
#   see how it fits the data

# In[3]:


from sklearn.linear_model import LinearRegression
model = LinearRegression();
model.fit(X_train.reshape(-1,1),y_train.reshape(-1,1));
y_pred = model.predict(X_test.reshape(-1,1));
plt.scatter(X_test,y_true,color = 'g');
plt.plot(X_test,y_pred,color = 'r');
plt.xlabel('test feature');
plt.ylabel('Linear regression fit');


# ##### Inference
# 
# - Now ofcourse Linear regression without feature transformation did not do
#   a great job.
# - Next we will use sine transform on the input feature and then use pipeline
#   to illustrate the linear regression on transformed feature in one go.

# #### Using Pipeline for Linear Regression on transformed feature

# In[4]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# this function transforms input feature into sine function
def sine_transform(X):
    y = X+10*np.sin(X);
    return y
    
# here we define the pipeline for transformation and estimation in one unit
pipe = Pipeline([('sine_transform',FunctionTransformer(sine_transform)),
                 ('estimator',LinearRegression())]);

pipe.fit(X_train.reshape(-1,1),y_train.reshape(-1,1));
y_pred2 = pipe.predict(X_test.reshape(-1,1));

plt.scatter(X_test,y_true,color='g');
plt.plot(X_test,y_pred2,'r');
plt.xlabel('test feature');
plt.ylabel('Fit on sine transformed feature');


# ##### Conclusion on Example1
# 
# - Here, we illustrated how we could combine feature engineering and estimators in one-go using sci-kit pipeline.
# - Sine transformation on input feature resulted in amazing fit/predictions.
# - Hence, simple linear regression can be very powerful, if we know our data and properly 
#   engineer it before the estimation stage.
#  

# <a id="4"></a>
# ### 2.2: Example2: Cancer dataset 

# In[5]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.svm import SVC

cancer = load_breast_cancer();
print("shape of cancer data :{}".format(cancer.data.shape));


# In[6]:


target = cancer.target;
ax = sns.countplot(x = target, label="Count", palette="muted")
print('Number of benign cancer: ', len(target[target==1]));
print('Number of malignant cancer: ', len(target[target==0]));
ax.set_xticks([0,1],['malignant','benign']);


# - create training and test datasets

# In[7]:


X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target, random_state=10);
print("X_train dimensions : {}".format(X_train.shape))
print("y_train dimensions : {}".format(y_train.shape))


# - we have 30 features and one target to be classified.

# ##### Building pipeline
# 
# - Here we will choose SVM estimator. Since SVM is sensitive to data scaling, we will use minmax scaler.
# - Our objective is to combine scaling with SVM as a single unit using pipeline.

# In[8]:


pipe = Pipeline([("scaler",MinMaxScaler()),("svm_classifier",SVC())]);


# - Now we have our pipeline, we can estimate the accuracy of our SVC model.

# In[9]:


pipe.fit(X_train,y_train)
test_score = pipe.score(X_test,y_test);
print("test_score: {:.3f}".format(test_score))


# #### Using pipe to tune hyperparameters of SVM
# 
# - Before we can run our gridsearch, we need to associate each parameters with the specific part of
#   our pipeline. 
# 
# - For example, if we want to optimize C and Gamma parameters, it belong to "svm_classifier"
#   part of the pipe. Hence we name it as "svm_classifier__C" in the GirdSearchCV.

# In[10]:


from sklearn.model_selection import GridSearchCV
 
# defining parameter range
param_grid = {'svm_classifier__C': [0.001,0.01,0.1,1,10,100],
              'svm_classifier__gamma': [0.001,0.01,0.1,1,10,100]};
 
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5);
grid.fit(X_train, y_train);
print("best_parameters:{}".format(grid.best_params_));


# #### Conclusion on Example2
# 
# - First the pipeline transforms the data using min_max scaler and then calls the score method on the
#   estimator on the scaled data.
# - Second, we directly used pipe to optimize parameters on specific parts of the pipe (hyperparameters
#   of SVM part of the pipeline).
# 

# <a id="5"></a>
# ### 2.3: Example3: Titanic dataset 

# <a id="6"></a>
# #### 2.3.1: Library and data import </p>

# In[11]:


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# import the data as pandas dataframe
train_data = pd.read_csv('../input/titanic/train.csv');
test_data  = pd.read_csv('../input/titanic/test.csv');
PID = test_data['PassengerId']; # save PID for competition submission

# drop irrelevant columns
train_data = train_data.drop(columns=['Name','Cabin','Ticket','PassengerId'], axis=1);
test_data = test_data.drop(columns=['Name','Cabin','Ticket','PassengerId'], axis=1);


# <a id="7"></a>
# #### 2.3.2: Define pipelines </p>
# 
# - Treat columns separately depending on numerical or categorical features.
# - For numerical we scale while categorical features are encoded.
# - Both are swept through imputer for missing values.
# - Preprocessor is a pipeline with sub-pipes inside.

# In[12]:


numerical_cols   = ['Age', 'SibSp', 'Parch', 'Fare'];
categorical_cols = ['Pclass', 'Sex', 'Embarked'];

numeric_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='mean'))
      ,('scaler', RobustScaler())
])
categorical_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='constant'))
      ,('encoder', OneHotEncoder())
])

preprocessor = ColumnTransformer(transformers=[('numeric', numeric_transformer, numerical_cols)
                               ,('categorical', categorical_transformer, categorical_cols) ]); 


# In[13]:


X = train_data.drop(columns = ['Survived'],axis=1);
y = train_data['Survived'];


# <a id="8"></a>
# #### 2.3.3: Define final pipeline and predict 

# In[14]:


#model
model = RandomForestClassifier(max_depth = 7,min_samples_split=2, n_estimators = 73,random_state=10);

#model pipeline
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

model_pipeline.fit(X, y);
y_submission = model_pipeline.predict(test_data).astype(int);
output = pd.DataFrame({'PassengerId': PID, 'Survived': y_submission})
output.to_csv('submission.csv', index=False)


# ##### Conclusion on Example3
# 
# - Here, we illustrated how we could multiple pipes and then combine them at final pipeline using titanic dataset.
# 

# <a id="9"></a>
# ### 2.4 : Example4: Iris dataset
# 
# The Iris dataset is a well-known dataset in machine learning and statistics that is often used for classification tasks. It was introduced by the British statistician and biologist Ronald Fisher in 1936, and contains measurements of the sepal length, sepal width, petal length, and petal width of three different species of iris flowers: Iris setosa, Iris versicolor, and Iris virginica.
# 
# The dataset consists of 150 instances, with 50 instances for each of the three species. For each instance, there are four features (sepal length, sepal width, petal length, and petal width) and a target variable that specifies the species of the iris (setosa, versicolor, or virginica).

# In[15]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
X, y = load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create a pipeline with three steps: scaling, PCA, and logistic regression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', LogisticRegression())
])

# Fit the pipeline to the training data
pipe.fit(X_train, y_train)

# Evaluate the pipeline on the test data
score = pipe.score(X_test, y_test)
print(f'Test accuracy: {score:.3f}')


# <a id="9"></a>
# #### 3. References
# 
# - https://pythonguides.com/scikit-learn-pipeline/
# - https://www.youtube.com/watch?v=jzKSAeJpC6s&t=385s
# - https://scikit-learn.org/stable/auto_examples/compose/plot_digits_pipe.html#sphx-glr-auto-examples-compose-plot-digits-pipe-py
