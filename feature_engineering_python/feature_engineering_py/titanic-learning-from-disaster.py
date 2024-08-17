#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## **The Challenge**
# 
# The sinking of the Titanic is one of the most infamous shipwrecks in history.
# 
# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# 
# While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
# 
# In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
# 
# **I will continue to work on and improve this notebook. Please upvote if you find it useful!**
# 
# ## **Loading Data and Exploratory Data Analysis**
# 
# We will first load the train data and examine its structure and some basic statistics. 
# 
# The **training set** should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

# In[2]:


titanic = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[3]:


# Displaying the first few rows of the dataset
titanic.head()


# In[4]:


# Checking the dimensions of the dataset
titanic.shape


# In[5]:


# Checking the data types of each variable
titanic.dtypes


# In[6]:


# Checking some basic statistics of the dataset
titanic.describe()


# In[7]:


survived = titanic['Survived'].value_counts(normalize=True) * 100
print(f"Survival rate: {survived[1]:.2f}%")


# We can see that the survival rate for females was significantly higher than for males.

# In[8]:


gender_survived = titanic.groupby(['Sex'])['Survived'].value_counts(normalize=True) * 100
gender_survived = gender_survived.rename('Percentage').reset_index()

plt.figure(figsize=(8,6))
sns.barplot(x='Sex', y='Percentage', hue='Survived', data=gender_survived)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate (%)')
plt.show()


# Next, let's explore the relationship between age and survival rate. We can see that children (those under 18 years old) had a higher survival rate compared to other age ranges.

# In[9]:


age_survived = titanic[['Age', 'Survived']].dropna()
age_survived['Age'] = age_survived['Age'].astype(int)
age_survived['Age Range'] = pd.cut(age_survived['Age'], bins=[0, 18, 35, 60, 100], labels=['<18', '18-35', '36-60', '>60'])
age_survived = age_survived.groupby(['Age Range'])['Survived'].value_counts(normalize=True) * 100
age_survived = age_survived.rename('Percentage').reset_index()

plt.figure(figsize=(8,6))
sns.barplot(x='Age Range', y='Percentage', hue='Survived', data=age_survived)
plt.title('Survival Rate by Age Range')
plt.xlabel('Age Range')
plt.ylabel('Survival Rate (%)')
plt.show()


# Let's also explore the relationship between passenger class and survival rate. We can see that passengers in first class had a higher survival rate compared to second and third class passengers.

# In[10]:


class_survived = titanic.groupby(['Pclass'])['Survived'].value_counts(normalize=True) * 100
class_survived = class_survived.rename('Percentage').reset_index()

plt.figure(figsize=(8,6))
sns.barplot(x='Pclass', y='Percentage', hue='Survived', data=class_survived)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate (%)')
plt.show()


# Univariate analysis of each variable

# In[11]:


# Creating histograms of all numeric variables
titanic.hist(figsize=(10,10))
plt.show()


# The correlation matrix of the Titanic dataset measures the strength and direction of the linear relationship between each pair of numeric variables. The values in the matrix range from -1 to 1, where a value of 1 indicates a perfect positive correlation, a value of -1 indicates a perfect negative correlation, and a value of 0 indicates no correlation.
# 
# The heatmap generated from the correlation matrix using the Seaborn library shows the correlation coefficients for each pair of variables in a color-coded matrix. The color scale ranges from blue (negative correlation) to red (positive correlation), with white representing no correlation.
# 
# From the heatmap, we can see that there is a positive correlation between the passenger class (Pclass) and fare (Fare) variables, which makes sense since higher passenger classes usually correspond to higher fares. There is also a negative correlation between the passenger class (Pclass) and the survival outcome (Survived) variable, indicating that passengers in lower classes were less likely to survive.
# 
# Another interesting observation is the positive correlation between the number of parents/children aboard (Parch) and the number of siblings/spouses aboard (SibSp). This suggests that families tended to travel together on the Titanic.
# 
# Overall, the correlation matrix and heatmap provide a useful tool for exploring the relationships between different variables in the Titanic dataset and can help identify interesting patterns and insights.

# In[12]:


# Creating a correlation matrix of all numeric variables
corr = titanic.corr()
sns.heatmap(corr, annot=True)
plt.show()


# ## **Feature Engineering**
# 
# Feature engineering is the process of transforming raw data into a set of meaningful features that can be used to improve the performance of machine learning models. The goal of feature engineering is to create informative and discriminating features that can help the model better capture the underlying patterns in the data.
# 
# In the context of the Titanic dataset, we can perform feature engineering by creating new features based on existing ones or by transforming existing features to better represent the information they contain.
# 
# One example of feature engineering on the Titanic dataset is creating a new feature called "FamilySize" that combines the "SibSp" (number of siblings/spouses aboard) and "Parch" (number of parents/children aboard) features into a single feature representing the total size of the passenger's family.

# In[13]:


# Create a new feature 'FamilySize'
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1

# Print the first 5 rows of the updated dataset
titanic.head()


# This code creates a new column in the Titanic dataset called "FamilySize" by adding the values of "SibSp" and "Parch" for each passenger and adding 1 to account for the passenger themselves. This feature can be useful for predicting survival outcomes as larger families may have had a harder time escaping the sinking ship.
# 
# Another example of feature engineering is creating a new feature called "Title" that extracts the title of each passenger from their name. This feature can provide information about the passenger's social status and may be related to their survival outcome.

# In[14]:


# Create a new feature 'Title'
titanic['Title'] = titanic['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

# Replace rare titles with 'Rare'
titanic['Title'] = titanic['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'the Countess'], 'Rare')

# Replace common female titles with 'Miss'
titanic['Title'] = titanic['Title'].replace(['Mlle','Ms'], 'Miss')

# Replace common male titles with 'Mr'
titanic['Title'] = titanic['Title'].replace('Mme', 'Mrs')

# Print the unique titles and their counts
print(titanic['Title'].value_counts())


# This code creates a new column in the Titanic dataset called "Title" by extracting the title from the "Name" column using a lambda function. The extracted titles are then printed along with their counts. This feature can be useful for predicting survival outcomes as certain titles may be associated with higher or lower survival rates.
# 
# **Cabin Deck:** Create a new feature that extracts the deck from the cabin number. The deck may provide information about the passenger's location on the ship and their proximity to lifeboats.

# In[15]:


# Create a new feature 'Deck' based on the cabin number
titanic['Deck'] = titanic['Cabin'].str[0]

# Replace missing values with 'Unknown'
titanic['Deck'].fillna('Unknown', inplace=True)

# Print the unique decks and their counts
print(titanic['Deck'].value_counts())


# This code creates a new column in the Titanic dataset called "Deck" by extracting the first character from the "Cabin" column. The extracted decks are then printed along with their counts. This feature can be useful for predicting survival outcomes as certain decks may be associated with higher or lower survival rates.
# 
# **Age Bins:** Create a new feature that discretizes the "Age" column into age bins. The age bins may provide information about the passenger's age group and their likelihood of survival.

# In[16]:


# Create age bins
bins = [0, 12, 18, 30, 50, 100]
labels = ['Child', 'Teen', 'Adult', 'Middle-aged', 'Senior']
titanic['AgeGroup'] = pd.cut(titanic['Age'], bins=bins, labels=labels)

# Print the unique age groups and their counts
print(titanic['AgeGroup'].value_counts())


# This code creates a new column in the Titanic dataset called "AgeGroup" by discretizing the "Age" column into age bins using the pandas.cut() function. The age groups and their counts are then printed. This feature can be useful for predicting survival outcomes as certain age groups may be associated with higher or lower survival rates.
# 
# **Fare per Person:** Create a new feature that calculates the fare per person based on the "Fare" and "FamilySize" columns. The fare per person may provide information about the passenger's socio-economic status and their likelihood of survival.

# In[17]:


# Create a new feature 'FarePerPerson'
titanic['FarePerPerson'] = titanic['Fare'] / titanic['FamilySize']

# Print the mean fare per person for survivors and non-survivors
print(titanic.groupby('Survived')['FarePerPerson'].mean())


# This code creates a new column in the Titanic dataset called "FarePerPerson" by dividing the "Fare" column by the "FamilySize" column. The mean fare per person for survivors and non-survivors is then printed. This feature can be useful for predicting survival outcomes as higher fares per person may be associated with higher survival rates.
# 
# **Age * Class:** Create a new feature that multiplies the "Age" and "Pclass" columns. The age * class may provide information about the passenger's socio-economic status and their likelihood of survival.
# 

# In[18]:


# Create a new feature 'AgeClass'
titanic['AgeClass'] = titanic['Age'] * titanic['Pclass']

# Print the mean age * class for survivors and non-survivors
print(titanic.groupby('Survived')['AgeClass'].mean())


# This code creates a new column in the Titanic dataset called "AgeClass" by multiplying the "Age" and "Pclass" columns. The mean age * class for survivors and non-survivors is then printed. This feature can be useful for predicting survival outcomes as higher age * class values may be associated with higher socio-economic status and thus higher survival rates.
# 
# 
# We can see that the engineered features are correlated with the target Survived. They might be useful in predicting whether a person would survive.

# In[19]:


# Creating a correlation matrix of all numeric variables
corr = titanic.corr()
sns.heatmap(corr, annot=True)
plt.show()


# In[20]:


titanic.columns


# In[21]:


# Create subplots with 2 rows and 3 columns
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Define the variables to plot
vars_to_plot = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

# Loop through the variables and plot them for each value of 'Survived'
for i, var in enumerate(vars_to_plot):
    row, col = i // 3, i % 3
    sns.histplot(data=titanic, x=var, hue='Survived', kde=True, ax=axes[row, col])
    axes[row, col].set_title(f"{var} distribution")

plt.suptitle("Distribution of Variables by Survival Status")
plt.tight_layout()
plt.show()


# ## **Machine Learning**
# 
# We split the data into features (X) and target (y) variables by dropping the 'Survived' column from X and assigning it to y. Finally, we use the train_test_split function from scikit-learn to split the data into training and testing sets with a 80/20 split and a random state of 42.
# 
# After splitting the data, you can use the X_train and y_train variables to train your machine learning model, and the X_test and y_test variables to evaluate its performance on unseen data.

# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Apply label encoding to 'Sex' column
le = LabelEncoder()
titanic['Sex'] = le.fit_transform(titanic['Sex'])

# Calculate median age
median_age = titanic['Age'].median()

# Substitute missing values with median
titanic['Age'].fillna(median_age, inplace=True)

# Calculate median age
median_age = titanic['AgeClass'].median()

# Substitute missing values with median
titanic['AgeClass'].fillna(median_age, inplace=True)

# Split the data into features (X) and target (y) variables
X = titanic.drop(['Survived', 'PassengerId', 'Ticket', 'Name', 
                 'Cabin', 'Embarked', 'Title', 'Deck', 'AgeGroup'], axis=1)

# Create one-hot encoded features for categorical columns
#X = pd.get_dummies(X, columns=['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'AgeGroup',
                              # 'Cabin'])

print(X.shape)
print(X.columns)
print(X.isna().sum())
y = titanic['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## **Logistic Regression**
# 
# Logistic regression is a supervised learning algorithm used for binary classification problems, where the goal is to predict a binary outcome (0 or 1) based on a set of input features. It models the relationship between the input features and the probability of the binary outcome using the logistic function (also called sigmoid function), which maps any real-valued input to a value between 0 and 1.
# 
# The logistic regression model estimates the coefficients of the input features that maximize the likelihood of observing the binary outcome given the input features. These coefficients are used to predict the probability of the binary outcome for new input examples.
# 
# The logistic regression algorithm is widely used in various fields, such as finance, healthcare, marketing, and social sciences, for applications such as fraud detection, disease diagnosis, customer segmentation, and opinion mining. It is a simple yet effective algorithm that can handle both numerical and categorical input features, and it can be easily extended to handle multiclass classification problems.

# In[23]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, plot_roc_curve, plot_precision_recall_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Scale the data:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the logistic regression model:
logreg = LogisticRegression(max_iter=1000)

# Fit the model on the training data:
logreg.fit(X_train_scaled, y_train)

# Make predictions on the testing data:
y_pred = logreg.predict(X_test_scaled)

# Evaluate the performance of the model using various metrics:
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC score: ", roc_auc_score(y_test, logreg.predict_proba(X_test_scaled)[:, 1]))

# Plot the ROC curves:
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for i, (train, test) in enumerate(cv.split(X, y)):
    logreg.fit(X.iloc[train], y.iloc[train])
    viz = plot_roc_curve(logreg, X.iloc[test], y.iloc[test], name='ROC fold {}'.format(i+1), alpha=0.3, lw=1, ax=plt.gca())
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[24]:


#Plot the feature importance:
importance = logreg.coef_[0]
for i, v in enumerate(importance):
    print('Feature: %s, Score: %.5f' % (X.columns[i], v))
plt.bar([x for x in range(len(importance))], importance)
plt.xticks(range(len(X.columns)), X.columns, rotation=45, ha='right')
plt.show()


# ## **Deep Learning**
# 
# Deep learning is a type of machine learning that is used to teach artificial neural networks how to learn from vast amounts of data. These neural networks are designed to mimic the structure of the human brain, and they are composed of layers of interconnected nodes called neurons that work in concert to process and analyze information.
# 
# In the process of deep learning, the neural network is fed large amounts of data and then adjusts its parameters through a process called backpropagation to minimize the difference between its predicted output and the actual output. This allows the neural network to recognize patterns and make accurate predictions.
# 
# Deep learning has many practical applications, such as image and speech recognition, natural language processing, and self-driving cars. It has revolutionized the field of artificial intelligence and has contributed to significant advancements in research and development.

# In[25]:


import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, plot_roc_curve, plot_precision_recall_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Scale the data:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model:
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X_train_scaled.shape[1]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss and Adam optimizer:
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training data:
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=200, batch_size=64)

# Make predictions on the testing data:
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

# Evaluate the performance of the model using various metrics:
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC score: ", roc_auc_score(y_test, model.predict(X_test_scaled)))

# Plot the ROC curves:
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for i, (train, test) in enumerate(cv.split(X, y)):
    # Define the neural network model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=X.shape[1]),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model with binary cross-entropy loss and Adam optimizer:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model on the training data:
    model.fit(X.iloc[train], y.iloc[train], validation_split=0.2, epochs=100, batch_size=32, verbose=0)

    # Make predictions on the testing data:
    y_pred = model.predict(X.iloc[test])

    # Compute the ROC curve and area under the curve:
    fpr, tpr, thresholds = roc_curve(y.iloc[test], y_pred)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' %
(mean_auc, std_auc), lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
label=r'$\pm$ 1 std. dev.')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# ## **Tree Based Models**
# 
# Tree-based models are popular in machine learning for classification and regression tasks due to their ability to handle complex datasets and provide interpretable results. The algorithm creates a decision tree structure where nodes represent decisions based on specific features and edges represent possible outcomes. The algorithm selects the feature with the most information about the target variable and splits the data based on its values at each node. This process continues until a stopping criterion is met. To make predictions, the algorithm traverses the tree from the root node to a leaf node, outputting the predicted value.
# 
# Tree-based models can capture nonlinear relationships between features and target variables, handle numerical and categorical data, and missing values. They also provide insight into feature importance. Decision trees, random forests, and gradient boosting machines are common types of tree-based models that improve performance and prevent overfitting.
# 
# 
# ## **XGBoost (Extreme Gradient Boosting)**
# 
# XGBoost (Extreme Gradient Boosting) is a powerful machine learning algorithm that is widely used for classification and regression tasks. It is based on the gradient boosting technique, which involves combining many weak learners (decision trees) to create a strong learner.
# 
# The main features of XGBoost include speed and scalability, regularization, feature importance, missing values handling, and built-in cross-validation. In terms of performance, XGBoost is known for its high accuracy and has won several machine learning competitions on Kaggle.
# 
# The XGBoost classifier works by training a sequence of decision trees on the data. At each iteration, the model calculates the gradient of the loss function with respect to the predictions, and then fits a new tree to the negative gradient. This process is repeated until the loss function converges, resulting in a final prediction for each sample.
# 
# The hyperparameters of the XGBoost model can be tuned to achieve optimal performance, including the learning rate, the number of trees, the maximum depth of the trees, and the regularization parameters.
# 
# Bayesian search is a hyperparameter tuning technique that uses probability and Bayesian inference to find the optimal hyperparameters for a given machine learning algorithm. Unlike grid search, which tries every possible combination of hyperparameters, Bayesian search tries to intelligently explore the hyperparameter space by focusing on the most promising regions.
# 
# Bayesian search uses an iterative process to build a probabilistic model of the relationship between hyperparameters and performance. The algorithm starts by sampling a set of hyperparameters from the prior distribution and evaluating their performance on a validation set. It then updates its model with this new information and samples the next set of hyperparameters based on the updated model. This process continues until a stopping criterion is met, such as a maximum number of iterations or a desired level of performance.
# 
# One of the advantages of Bayesian search is that it can handle noisy and non-convex objective functions. It also tends to be more efficient than grid search because it can focus on the most promising regions of the hyperparameter space. However, it can be computationally expensive, particularly when the number of hyperparameters is large or the dataset is large.
# 
# Overall, Bayesian search is a powerful technique for hyperparameter tuning that can help improve the performance of machine learning models.
# 
# Optuna is an open source hyperparameter optimization framework to automate hyperparameter search. It offers an efficient way to find ideal hyperparameters and many visualizations of the results and search space.

# In[26]:


import xgboost as xgb
from sklearn.model_selection import cross_val_score
import optuna

# Define the objective function to optimize
def objective(trial):
    # Define the hyperparameter search space
    params = {'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
              'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
              'max_depth': trial.suggest_int('max_depth', 3, 11),
              'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
              'gamma': trial.suggest_float('gamma', 0, 0.5),
              'subsample': trial.suggest_float('subsample', 0.5, 0.9),
              'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
              'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
              'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)}

    # Create an XGBoost classifier with the current set of hyperparameters
    model = xgb.XGBClassifier(**params)

    # Evaluate the classifier using 5-fold cross-validation
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

    # Return the mean cross-validation score as the objective value to minimize
    return score.mean()

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters and the corresponding mean cross-validation score
print("Best hyperparameters:", study.best_params)
print("Best mean cross-validation score:", study.best_value)

# Fit the model on the full training set using the best hyperparameters
best_model = xgb.XGBClassifier(**study.best_params)
best_model.fit(X_train, y_train)

# Evaluate the accuracy of the best model on the test set
accuracy = best_model.score(X_test, y_test)
print("Accuracy:", accuracy)


# In[27]:


print("Best params:")
for key, value in study.best_params.items():
    print(f"\t{key}: {value}")


# **plot_optimization_history** plots optimization history of all trials in a study. The blue dots show the smape on each trial, and the red line - the best value attained.

# In[28]:


from optuna.visualization import plot_optimization_history
plot_optimization_history(study)


# The code line below plots hyperparameters importance during optimization. In this case, reg_alpha, learning rate, and col_sample by tree were the most important.

# In[29]:


from optuna.visualization import plot_param_importances
plot_param_importances(study)


# In[30]:


import xgboost as xgb
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Define the hyperparameter search space to search over
param_space = {'learning_rate': (0.01, 0.5, 'log-uniform'),
               'n_estimators': (50, 1000),
               'max_depth': (3, 11)}

# Create an XGBoost classifier
model = xgb.XGBClassifier()

# Perform a Bayesian search over the hyperparameter search space using 5-fold cross-validation
with tqdm(total=50) as pbar:
    opt = BayesSearchCV(model, param_space, cv=5, n_iter=50, scoring='accuracy')
    opt.fit(X_train, y_train)
    pbar.update(1)

# Print the best hyperparameters and the corresponding mean cross-validation score
print("Best hyperparameters:", opt.best_params_)
print("Best mean cross-validation score:", opt.best_score_)

# Fit the model on the full training set using the best hyperparameters
best_model = xgb.XGBClassifier(**opt.best_params_)
best_model.fit(X_train, y_train)

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Evaluate the accuracy of the best model on the test set
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# ## **Machine Learning Interpretability**
# 
# Machine learning interpretability refers to the ability to understand and explain how a machine learning model makes its predictions or decisions. It is important for several reasons, including improving transparency and accountability, identifying potential biases or errors, and building trust in the model.
# 
# **SHAP** (**SH**apley **A**dditive ex**P**lanations) values are a popular method for interpreting the predictions of machine learning models. **SHAP** values are based on the concept of game theory and provide an intuitive way to understand the importance of each feature in the model's predictions.
# 
# **SHAP** values measure the contribution of each feature to the difference between the actual prediction and the expected prediction. They can be used to explain the output of any machine learning model and can be calculated for individual predictions or for the overall model. By analyzing the **SHAP** values, we can determine which features are most important in driving the model's predictions and how they influence the output.
# 
# One of the advantages of **SHAP** values is that they provide a global and local interpretation of the model. A global interpretation refers to the overall importance of each feature in the model, while a local interpretation refers to the importance of each feature in a specific prediction. This makes it possible to identify which features are consistently important across all predictions and which features are important for a particular prediction.
# 
# **SHAP** values are widely used in various applications, including credit risk assessment, medical diagnosis, and image recognition, among others. They can be visualized using various plots, such as **SHAP** summary plots and individual contribution plots, to aid in the interpretation of the model.
# 
# Summary plot of the feature importances using the **SHAP** (SHapley Additive exPlanations) library. Overall, the plot shows the relative importance of each feature in the model's predictions. The features with higher SHAP values contribute more to the predictions, while those with lower SHAP values contribute less. The plot can help to identify which features have the most impact on the model's performance and can be useful for feature selection or interpretation. Sex is the most important feature, followed by Pclass, AgeClass, and FarePerPerson.

# In[31]:


import shap
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Summary plot of feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")


# The plot below provides an overview of the relative importance of each feature in the model's predictions and how each feature contributes to the model's output for each instance in the test set. It can be useful for identifying which features are most important for the model and how they affect the model's predictions.

# In[32]:


# Summary plot of SHAP values
shap.summary_plot(shap_values, X_test)


# This plot provides an overview of how the "Age" feature affects the model's predictions for each instance in the test set. It can be useful for understanding the relationship between the feature and the model's predictions and for identifying any non-linear relationships or interactions with other features.

# In[33]:


# Dependence plot of Age feature vs. SHAP value
shap.dependence_plot("Age", shap_values, X_test)


# ## **Load Test Data and Make Submission**

# In[34]:


# Load test set
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test


# We need to do the same feature engineering as before.

# In[35]:


# Create a new feature 'FamilySize'
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

# Create a new feature 'Title'
test['Title'] = test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

# Replace rare titles with 'Rare'
test['Title'] = test['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'the Countess'], 'Rare')

# Replace common female titles with 'Miss'
test['Title'] = test['Title'].replace(['Mlle','Ms'], 'Miss')

# Replace common male titles with 'Mr'
test['Title'] = test['Title'].replace('Mme', 'Mrs')

# Create a new feature 'Deck' based on the cabin number
test['Deck'] = test['Cabin'].str[0]

# Replace missing values with 'Unknown'
test['Deck'].fillna('Unknown', inplace=True)

# Create age bins
bins = [0, 12, 18, 30, 50, 100]
labels = ['Child', 'Teen', 'Adult', 'Middle-aged', 'Senior']
test['AgeGroup'] = pd.cut(test['Age'], bins=bins, labels=labels)

# Create a new feature 'FarePerPerson'
test['FarePerPerson'] = test['Fare'] / test['FamilySize']

# Create a new feature 'AgeClass'
test['AgeClass'] = test['Age'] * test['Pclass']
test


# In[36]:


test.columns


# In[37]:


# Apply label encoding to 'Sex' column
le = LabelEncoder()
test['Sex'] = le.fit_transform(test['Sex'])

# Fill missing values with median or mode
#test['Age'].fillna(test['Age'].median(), inplace=True)
#test['Fare'].fillna(test['Fare'].median(), inplace=True)

prediction = test.drop(['PassengerId', 'Ticket', 'Name', 
                 'Cabin', 'Embarked', 'Title', 'Deck', 'AgeGroup'], axis=1)
print(test.shape)


# In[38]:


# Make predictions on the test dataset
pred = best_model.predict(prediction)
pred = pd.DataFrame(pred, columns=['Survived'])
sns.displot(pred);


# In[39]:


# Calculate the median value of the 'Survived' column in the training set
median_survived = pred['Survived'].median()

# Fill the missing values in the 'Survived' column of the test set with the median value
pred['Survived'] = pred['Survived'].fillna(median_survived)
pred


# In[40]:


survived = pred['Survived'].value_counts(normalize=True) * 100
print(f"Survival rate: {survived[1]:.2f}%")


# In[41]:


test['Survived'] = pred['Survived']
test


# In[42]:


# Create a submission DataFrame with 'PassengerId' and 'Survived' columns
submission = test[['PassengerId', 'Survived']]
submission


# In[43]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




