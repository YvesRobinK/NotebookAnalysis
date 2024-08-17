#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # [AI-ML-DS : Training for beginners](https://www.kaggle.com/vbmokin/ai-ml-ds-training-for-beginners-in-kaggle). Level 1 (very simple). 2020
# ## Kaggle GM, Prof. [@vbmokin](https://www.kaggle.com/vbmokin)
# ### [Vinnytsia National Technical University](https://vntu.edu.ua/), Ukraine
# #### [Chair of the System Analysis and Information Technologies](http://mmss.vntu.edu.ua/index.php/ua/)

# # The concept of training:
# * the **last version (commit)** of the notebook has:
#         * the basic tasks (after "TASK:")
#         * the additional tasks for self-execution (after "ADDITIONAL TASK:")
# * the **previuos version (commit)** of the notebook has **answers** for the basic tasks

# ## Competition [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

# ## Acknowledgements
# * [Data Science for tabular data: Advanced Techniques](https://www.kaggle.com/vbmokin/data-science-for-tabular-data-advanced-techniques)
# * [EDA for tabular data: Advanced Techniques](https://www.kaggle.com/vbmokin/eda-for-tabular-data-advanced-techniques)
# * [Titanic - Top score : one line of the prediction](https://www.kaggle.com/vbmokin/titanic-top-score-one-line-of-the-prediction)
# * [Three lines of code for Titanic Top 25%](https://www.kaggle.com/vbmokin/ai-ml-ds-training-l1-titanic-decision-tree)
# * [Three lines of code for Titanic Top 20%](https://www.kaggle.com/vbmokin/three-lines-of-code-for-titanic-top-20)

# ### It is recommended to start working with this notebook from study:
# * the [description of the data](https://www.kaggle.com/c/titanic/data)
# * the [task](https://www.kaggle.com/c/titanic) of the competition
# * the [my lecture](https://www.youtube.com/watch?v=WERtPBptOWw&list=PL4DHq-xU-ebUiB6T6vjd0SoDha4GOm8zV&index=2&t=951s) about this notebook in YouTube (in Ukrainain).

# <a class="anchor" id="0.1"></a>
# ## Table of Contents
# 
# 1. [Import libraries](#1)
# 1. [Download data](#2)
# 1. [EDA & FE](#3)
# 1. [Modeling](#4)
# 1. [Prediction & Submission](#5)

# ## 1. Import libraries<a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[1]:


# Work with Data - the main Python libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

# Modeling and Prediction
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix


# ## 2. Download data<a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[2]:


# Download training data
train = pd.read_csv('../input/titanic/train.csv')


# **TASK:** Display the first 5 rows of the training dataframe.

# In[3]:


# Display the first 5 rows of the training dataframe.


# In[4]:


# Display basic information about training data
train.info()


# In[5]:


# Download test data
test = pd.read_csv('../input/titanic/test.csv')
test.tail(7)  # Display the 7 last rows of the training dataframe


# **TASK:** Display basic information about the test data

# In[6]:


# Display basic information about the test data


# In[7]:


# Download submission sample file
submission = pd.read_csv('../input/titanic/gender_submission.csv')


# **TASK:** Display the 10 first rows of the submission dataframe

# In[8]:


# Display the 10 first rows of the submission dataframe


# ## 3. EDA & FE<a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# Let's make the following assumptions:
# - **Women** have a better chance of surviving ==> feature "**Sex**" is important
# - **Single people** have a worse chance of surviving ==> the **number of family members** is important
# - **Children** have a better chance of survival ==> **Age** is important

# In[9]:


def highlight(value):
    # The painting of the cell in different colors depending on the value: 
    # >= 0.5 - palegreen, < 0.5 - pink
    
    if value >= 0.5:
        style = 'background-color: palegreen'
    else:
        style = 'background-color: pink'
    return style


# ### Feature "Sex"

# In[10]:


# Pivot table
pd.pivot_table(train, values='Survived', index=['Sex']).style.applymap(highlight)


# ### Feature "Family_size"

# **TASK:** Correct the formula that determines the size of each passenger's family (the new feature: train['Family_size']) by the number  of his / her siblings ("SibSp") and his / her parents and descendants ("Parch")

# In[11]:


# Calculation the new feature "Family_size" as the size of each passenger's family by 
# the number of his / her siblings ("SibSp") and his / her parents and descendants ("Parch")
train['Family_size'] = (train['SibSp'] + train['Parch']) // 2  # it is wrong!


# In[12]:


# Pivot table for input features 'Sex' and 'Family_size' and output feature 'Survived'
pd.pivot_table(train, values='Survived', index=['Sex', 'Family_size']).style.applymap(highlight)


# ### Feature "Age"

# In[13]:


# Analysis missing data for feature "Age"
missing_age_values = train['Age'].isnull().sum() 


# **TASK:** Calculate what percentage of the value of missing_age_values is relative to the total number of values (dataframe length) and round it to two decimal places. Save it into missing_age_values_per_cent.

# In[14]:


# Calculate what percentage of the value of missing_age_values is relative 
# to the total number of values (dataframe length) and round it to two decimal places. 
# Save it into missing_age_values_per_cent.
missing_age_values_per_cent = missing_age_values/1000  # it is wrong!


# In[15]:


# Output missing_age_values_per_cent
print(f'Feature "Age" has {missing_age_values_per_cent}, %')


# Bad solution, but the easiest way is to replace the missing data with the average.

# **TASK:** Calculation the average (mean) value of all data of Age. Save it into mean_age.

# In[16]:


# Calculation the average (mean) value of all data of Age. Save it into mean_age.
mean_age = 30   # it is wrong!


# **TASK:** Round mean_age to two decimal places and print it.

# In[17]:


# Round mean_age to two decimal places and print it.


# In[18]:


# Statistics for training data
train.describe()


# In[19]:


# Number of unique values of age
number_age_unique_values = len(train['Age'].unique())
number_age_unique_values


# We reduce the number of unique age values by 7 times (7 because children under 14 (14 // 7 = 2) were considered a child at that time).

# **TASK:** Divide all age values train['Age'] by 7 and save in new feature train['Age_7'].

# In[20]:


# Divide all age values train['Age'] by 7 and save in new feature train['Age_7'].
train['Age_7'] = train['Age']     # it is wrong!


# In[21]:


train['Age_7'] = train['Age_7'].fillna(mean_age // 7).astype('int')


# **TASK:** Determine how many unique values the feature "Age" contains and print it.

# In[22]:


# Print the number of unique values of the feature "Age"


# **TASK:** Build a pivot table for input features 'Sex', 'Family_size' and 'Age' and output feature 'Survived'.

# In[23]:


# Pivot table for input features 'Sex', 'Family_size' and 'Age' and output feature 'Survived'


# In[24]:


# Decision trees work with numbers, not words, so we must encode feature "Sex" by numbers
train['Sex'].replace({'male': 0, 'female': 1}).head()


# The result of FE is usually combined into one function so that it is convenient to apply to different datasets

# **TASK:** Copy your commands (see above) to this function to calculate new features or process existing ones

# In[25]:


# Copy commands (see above) to this function to calculate new features or process existing ones

def df_transform(df):
    # FE for df
    
    # Number of family members - feature "Family_size"
    df['Family_size'] = (df['SibSp'] + df['Parch']) // 2  # it is wrong!
    
    # Age multiple of 7
    df['Age_7'] = df['Age']    # it is wrong!
    
    # Average age of all dataset multiple of 7
    mean_age = 30 // 7   # it is wrong!
    
    # Replace missing age values to average age and rounding them to integers
    df['Age_7'] = df['Age_7'].fillna(mean_age).astype('int')
    
    # Encoding feature "Sex" by numbers
    df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})
    
    # Select the main features
    df = df[['Family_size','Age_7','Sex']]
    
    return df


# In[26]:


# Selecting a target featute and removing it from training dataset
target = train.pop('Survived')


# In[27]:


# FE to training dataset
train = df_transform(train)

# Statistics of training dataset
train.info()


# In[28]:


train


# In[29]:


# FE to test dataset
test = df_transform(test)
test.info()


# In[30]:


test


# **It is important to make sure** that all features in the training and test datasets:
# * do not have missing values (number of non-null values = number of entries of index) 
# * all features have a numeric data type (int8, int16, int32, int64 or float16, float32, float64).

# **ADDITIONAL TASKS:**
# 1. Replace the feature "Family_size" to cabin class "Pclass" and "Age_7" to "Embarked".
# 2. Replace the value of feature "Embarked" with the numbers 0, 1 and 2 (for example: 0 => C = Cherbourg, 1 => Q = Queenstown, 2 => S = Southampton). 

# ## 4. Modeling<a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[31]:


# Select model as Decision Tree Classifier 
# "Classifier" because target has limited (integer) number of classes, in this case 2 classes = [0, 1] or ["No Survived", "Survived"]
# For a small amount of data, it is better to choose a smaller parameter max_depth - from 3 to 5, let give 4
# at a more complex Level we will teach the program to calculate it automatically
model = DecisionTreeClassifier(max_depth=4, random_state=42)

# Training model
model.fit(train, target)


# ### Decision Tree Visualization

# In[32]:


# Visualization - build a plot with Decision Tree
plt.figure(figsize=(20,12))
plot_tree(model, filled=True, rounded=True, class_names=["No Survived", "Survived"], feature_names=train.columns) 


# ### Confusion matrix

# In[33]:


# Prediction for training data
y_train = model.predict(train).astype(int)


# In[34]:


confusion_matrix(target, y_train)


# **ADDITIONAL TASKS:**
# 1. Experiment with resizing (in the "figsize = ()") the drawing with the decision tree.
# 2. Try changing the value of max_depth above so that the number of correct predictions (prediction values 0 when 0 and 1 when 1 ib the confusion matrix) is greater.

# ## 5. Prediction & Submission<a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# In[35]:


# Prediction of target for test data
y_pred = model.predict(test).astype(int)


# In[36]:


# Saving the result into submission file
submission["Survived"] = y_pred
submission.to_csv('submission.csv', index=False) # Competition rules require that no index number be saved

# Building the Histogram of predicted target values for test data
submission['Survived'].hist()


# In[37]:


# Calculation of the mean value of forecasting data
submission['Survived'].mean()


# **ADDITIONAL TASK:** 
# 1. Try changing the value of random_state (0, 1, ... 42 - choose some values) in the DecisionTreeClassifier() in section 4 and check for changes in the forecast results (in the confusion matrix, in the histogram above, and in the mean value of forecasting data).
# 2. Submit each version of the forecast (with a different list of features, with a different value of max_depth - see additional tasks above) to the competition and see if you can rise in the rankings.

# Next, you should commited this notebook, then in the "Output" section find the file "submission.csv" and press button "Submit" for the submission it to the competition.

# ## The general task (10 steps):
# I. Solve all tasks after the word "**TASK:**", entering the code in the cell under each such task.
# 
# II. Click on the top: "Run All".
# 
# III. Compare your results with the answers in the previous version of my notebook. If something is different - fix it. If you find an error or inaccuracy in comments or tasks in my notebook, please write about it in the comment to that notebook.
# 
# IV. Perform "**ADDITIONAL TASK:**" (at least a few) so that the performance results (plots, feature names and/or forecast) in your version of the notebook begin to differ from my original notebook.
# 
# V. Prepare the notebook for publication:
# 
# * rename by changing the name at the top of the window (to the left of the words "Draft saved"), for example to the following: 
# **Titanic: very simple Decision Tree with tuning**
# or 
# **AI-ML-DS : Titanic - Decision Tree - SAIT VNTU PTL**
# where at the end you write the name of your team in this competition;
# * at the beginning of your version of the notebook add 3 new cells of type "Markdown" with the following text:
# 
# -> *in the first cell* - a reference to the original source (Kaggle himself writes it on the right, but it is customary to duplicate it at the beginning to explain that this notebook is a development of another):
# 
# **Thanks to [AI-ML-DS Training. L1A: Titanic - Decision Tree](https://www.kaggle.com/vbmokin/ai-ml-ds-training-l1a-titanic-decision-tree)**
# 
# -> *in the second cell* - a shortlist of what you updated, changed, deleted, etc., for example (**be sure to edit it!**):
# 
# ### My upgrade:
# * add new plots;
# * replaced feature "Age" to "Pclass";
# * add a new feature "Fare";
# * model tuning (changed parameter "max_depth");
# * improved accuracy - LB score increase to 0.78883
# 
# -> *in the third cell* (you can combine with the second, ie immediately write what is added and why) - give an explanation of the reasons for such actions: ideas, hypotheses, experimental plan, for example:
# 
# I'm sure that class of cabin "Pclass" and fare of ticket "Fare" is a more important feature than "Age".
# 
# If your team consists of some members please write: "We sure...".
# 
# VI. "Save & Run All (Commit)" of your notebook - press the button "Save Version" in the top right corner of the window and press "Save". When done, scroll through and make sure there are no ERROR messages in any cell. Sometimes it can be executed successfully, but half of the cells are with errors, then the result of execution will be incorrect.
# 
# VII. If it's a notebook is on a certain dataset (not for competition), go to the next step right away. Or if this notebook for the Prize Kaggle Competition then after the committing of running go to the section "Output" and press "Submit". Make sure the system successfully accepted your submission and wrote your LB score. Calculation LB score can take some time (from a few seconds to 9 hours). If it outputs an error and LB score does not output, then return to the editor and correct the error.
# 
# VIII. Publish the successfully committed notebook: press the button "Sharing" in the top right corner of the window, in the form at the top press "Private" and select "Public" in the list, then press "Save". 
# 
# IX. Choose tags (optional) - press "Edit tags" and choose for example "exploratory data analysis", "feature engineering", "classification", "beginner".
# 
# X. If you are a student of my course, then please add a comment to the notebook, where you tag me, which will then allow me to quickly find it, for example, the following text:
# 
# -> bad option:
# 
# @vbmokin, please see how we solved this problem
# 
# -> best option:
# 
# The notebook was prepared by team "SAIT VNTU PTL" from the course "AI-ML-DS Training" (tutor - @vbmokin) for the competition "Titanic: Machine Learning from Disaster".
# 
# 
# 
# **Good luck!**

# I hope you find this notebook useful and enjoyable.
# 
# Your comments and feedback are most welcome.
# 
# [Go to Top](#0)
