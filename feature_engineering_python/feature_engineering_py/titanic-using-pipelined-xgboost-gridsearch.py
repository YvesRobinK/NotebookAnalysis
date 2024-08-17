#!/usr/bin/env python
# coding: utf-8

# # **Titanic - Machine Learning from Disaster**

# ## Contents:
# 1. **X, y Dataframes Creation**
#     * Import libraries
# 2. **X, y Summary**
# 3. **Data Cleaning**
#     * Remove the columns with more than half missing values
# 4. **Exploratory Data Analysis*
# 5. **Data Visualization (Original Features)**
#     * Correlation between features and target on heatmap
#     * Sex vs Survived
#     * Age vs Survived
#     * Fare vs Survived
# 6. **Feature Engineering**
#     * Create New Categorical Features
#         1. Name Prefix
#         2. Age Category
#         3. Fare Category
#         4. Family Size
#     * Data Visualization of Created Categories
#     * Imputing Age (using group means)
# 7. **Feature Selection**
# 8. **Model Creation**
#     * Preprocessing Pipelines
#     * Model Pipeline
#     * Visualize the pipeline
# 9. **Training and Testing Model**
#     * Grid Search & Cross Validation
#     * Feature Importance
#     * Confusion Matrix
#     * Classification Report
# 10. **Predicting y**

# # 1) X, y Dataframes Creation

# ## Import Data Analytics libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from math import ceil


# ## Import Machine Learning Libraries

# In[2]:


from sklearn.pipeline import Pipeline

# To perform operations on columns:
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# ML algorithms:
from xgboost import XGBClassifier

# To evaluate performance model:
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# In[3]:


# Read the data
X_y = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')
X_test = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')

# Remove rows with missing target
X_y = X_y.dropna(subset=['Survived'], axis=0)

#  Separate target y from predictors X
X = X_y.copy()
y = X.pop('Survived')


# # 2) X, y Summary

# In[4]:


X.head(2)


# In[5]:


# Creating function so that we can reuse it afterwords
def show_info(X,X_test):
    DataTypes = pd.DataFrame(X.dtypes.value_counts(),columns=['X'])
    DataTypes['X_test'] = X.dtypes.value_counts().values
    print("Number of Columns with different Data Types:\n")
    print(DataTypes,'\n')
    
    info = pd.DataFrame(X.dtypes, columns=['Dtype'])
    info['Unique_X'] = X.nunique().values
    info['Unique_X_test'] = X_test.nunique().values
    info['Null_X'] = X.isnull().sum().values
    info['Null_X_test'] = X_test.isnull().sum().values
    return info


# In[6]:


show_info(X,X_test)


# In[7]:


y.head(2)


# In[8]:


y.describe()


# # 3) Data Cleaning

# ## Remove the columns with more than half missing values

# In[9]:


# Making function so that we can reuse it in later stages as well
def show_null_values(X, X_test):
    
    # Making DataFrame for combining training and testing missing values
    null_values = pd.DataFrame(X.isnull().sum(), columns=['Train Data'])
    null_values['Test Data'] = X_test.isnull().sum().values

    # Showing only columns having missing values and sorting them
    null_values = null_values.loc[(null_values['Train Data']!=0) | (null_values['Test Data']!=0)]
    null_values = null_values.sort_values(by=['Train Data','Test Data'],ascending=False)
    
    print("Total missing values:\n",null_values.sum(),'\n',sep='')
    
    return null_values


# In[10]:


show_null_values(X, X_test)


# In[11]:


# Show columns with more than half values missing
null_columns = [col for col in X.columns if X[col].isnull().sum() > X.shape[0]/2]
null_columns


# In[12]:


# Drop the above mentioned columns
X = X.drop(null_columns, axis=1)
X_test = X_test.drop(null_columns, axis=1)


# # 4) Exploratory Data Analysis

# ### Correlation with target

# In[13]:


Xy = X.join(y)
correlation_matrix = Xy.corr()
correlation_matrix.Survived


# ### Combination of Sex, Pclass and Embarked

# In[14]:


df1 = Xy.groupby(['Sex','Pclass']).Survived.agg(['sum','count'])
df1['survival rate'] = round(df1['sum']/df1['count'],2)
df1


# **Insights:**
# * Most of the females in 1st and 2nd class survived.
# * Most of the males in 2nd and 3rd class didn't survive.

# In[15]:


df2 = Xy.groupby(['Embarked','Pclass']).Survived.agg(['sum','count'])
df2['survival rate'] = round(df2['sum']/df2['count'],2)
df2


# **Insights:**
# * Highest survival rate - 1st Pclass passenger Embarked at C
# * Lowest survival rate - 3rd Pclass passenger Embarked at S

# In[16]:


df3 = Xy.groupby(['Embarked','Sex']).Survived.agg(['sum','count'])
df3['survival rate'] = round(df3['sum']/df3['count'],2)
df3


# **Insights:**
# * Highest survival rate - female passenger Embarked at C
# * Lowest survival rate - male passenger Embarked at Q

# # 5) Data Visualization

# ## Correlation between features and target on heatmap

# In[17]:


# Returns copy of array with upper part of the triangle (which will be masked/hidden)
mask = np.triu(correlation_matrix)

plt.figure(figsize=(3, 3), dpi=120)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, mask=mask, linewidths=1, cbar=False)
plt.show()


# ### Insights from above correlation heatmap:
# 1. Survival is more linearly related to Pclass and Fare.
# 2. Pclass and Fare are negatively linear correlated. (as 1st class has more fare and 3rd class has less fare)
# 3. Parch and SibSp have more correlation. (So people which have Parents and Children with them are more likely to have with their Siblings and Spouse.)

# ## Sex vs Survived

# In[18]:


sns.catplot(data=Xy, x="Sex", y="Survived", hue="Pclass", kind="bar");


# ### Insights:
# * Survival rate of female passengers was higher than that of male passengers.
# * Survival rate was higher for upper class passengers than that of lower class passengers.

# ## Age vs Survived

# In[19]:


sns.catplot(data=Xy, y="Age", x="Survived", hue="Sex", kind="swarm", height=5, aspect=1.2);


# ### Insights:
# * Male passengers with age more than 50 had less chances of survival.
# * Passengers with age less than 10 had more chances of survival.

# ## Fare vs Survived

# In[20]:


sns.violinplot(data=Xy, y="Fare", x="Survived", hue="Sex", split=True, height=10, aspect=1);
plt.ylim(0, 300)


# ### Insights:
# * Passengers having more fare than 80 had more rate of survival.

# # 6) Feature Engineering

# ## Create New Categorical Features

# ### 1] Name Prefix

# In[21]:


X['Prefix'] = X['Name'].str.split(expand=True)[1]
X_test['Prefix'] = X_test['Name'].str.split(expand=True)[1]


# In[22]:


prefixes = X.Prefix.value_counts()
top_prefixes = prefixes[prefixes>len(X)/25].index
top_prefixes


# In[23]:


X.Prefix = X.Prefix.apply(lambda x: x if x in top_prefixes else 'other')
X_test.Prefix = X_test.Prefix.apply(lambda x: x if x in top_prefixes else 'other')


# In[24]:


df = pd.DataFrame(X['Prefix'].value_counts())
df['Prefix_test'] = X_test['Prefix'].value_counts().values
df


# ## Imputing Age (using group means)

# In[25]:


# Number of missing values in Age column
null_index=X.Age.isnull()
null_index_test=X_test.Age.isnull()
print(null_index.sum(),null_index_test.sum())


# In[26]:


avg_ages = X.groupby(['Prefix','Pclass']).Age.mean().round()
pd.DataFrame(avg_ages)


# In[27]:


for (i,k) in avg_ages.index:
    value = avg_ages.loc[i,k]
    X.loc[(X.Prefix==i) & (X.Pclass==k) & (X.Age.isnull()),'Age'] = value
    X_test.loc[(X_test.Prefix==i) & (X_test.Pclass==k) & (X_test.Age.isnull()),'Age'] = value


# In[28]:


# Number of missing age values after imputation
print(X["Age"].isnull().sum(),X_test["Age"].isnull().sum())


# In[29]:


# Rows where we imputed Age values
X[null_index].head(2)


# In[30]:


# Now, we will combine known prefixes as these categories are redundent when Sex column is present.
X.Prefix = X.Prefix.replace(['Mr.', 'Miss.', 'Mrs.', 'Master.'], 'known')
X_test.Prefix = X_test.Prefix.replace(['Mr.', 'Miss.', 'Mrs.', 'Master.'], 'known')


# In[31]:


# We do not need 'Name' column anymore.
X = X.drop(['Name'], axis=1)
X_test = X_test.drop(['Name'], axis=1)


# ### 2] Age Category

# In[32]:


def Age_categorise(df):
    df['Age_Cat'] = pd.cut(df.Age, bins = [0,10,40,60,100], labels = ['child','young','adult','senior'])
    
Age_categorise(X)
Age_categorise(X_test)
X['Age_Cat'].dtype


# In[33]:


df = pd.DataFrame(X['Age_Cat'].value_counts())
df['Age_Cat_test'] = X_test['Age_Cat'].value_counts().values
df


# ### 3] Fare Category

# In[34]:


X.Fare.describe()


# In[35]:


def Fare_categorise(df):
    df['Fare_Cat'] = pd.qcut(df.Fare, q=4, labels = ['low','medium','high','very_high'])
    
Fare_categorise(X)
Fare_categorise(X_test)
X['Fare_Cat'].dtype


# In[36]:


df = pd.DataFrame(X['Fare_Cat'].value_counts())
df['Fare_Cat_test'] = X_test['Fare_Cat'].value_counts().values
df


# ### 4] Family Size

# In[37]:


X['Family'] = X['SibSp'] + X['Parch']
X_test['Family'] = X_test['SibSp'] + X_test['Parch']

X = X.drop(['SibSp','Parch'],axis=1)
X_test = X_test.drop(['SibSp','Parch'],axis=1)


# In[38]:


X.Family.value_counts()


# In[39]:


def Categorize_Family(df):
    df['Family_Size'] = pd.cut(df.Family, bins = [-1,0,2,12], labels = ['alone', 'medium', 'large'])
    
Categorize_Family(X)
Categorize_Family(X_test)


# In[40]:


df = pd.DataFrame(X['Family_Size'].value_counts())
df['Family_Size_test'] = X_test['Family_Size'].value_counts().values
df


# Let us see newly created categorical columns.

# In[41]:


X.head(2)


# In[42]:


X_test.head(2)


# ## Data Visualization of Created Categories

# In[43]:


# Concatenating X and y for Data Visualization purpose
Xy = X.copy()
Xy['Survived'] = y.copy()


# In[44]:


# Making matplotlib parameters default
plt.rcParams.update(plt.rcParamsDefault)


# In[45]:


l=['Pclass','Sex','Age_Cat','Fare_Cat','Embarked','Family_Size']
figure, axes = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=False, dpi=140)
index = 0
axes = axes.flatten()
for axis in axes:
    sns.countplot(x = l[index], hue = "Survived", data = Xy, ax=axis)
    axis.set_title(l[index]+' vs Survived')
    index = index+1
    plt.tight_layout()
    for tick in axis.get_xticklabels():
        tick.set_rotation(45)
plt.show()


# ### Insights: 
# In each of the plots for feature vs target, we can see one category has more chance of survival than other categories as count of survived=1 is more than survived=0:
# * Pclass - 1
# * Sex - female
# * Age_Cat - 0
# * Fare_Cat - 3
# * Embarked - C
# * Family_Size - 1

# # 7) Feature Selection

# In[46]:


# Columns to choose from for model training and prediction
X.columns


# In[47]:


# We used Categorical columns Age_Cat, Fare_Cat, Family_Size for visualizations only.
# Select columns for classification model
X = X[['Age', 'Pclass', 'Sex', 'Family_Size']]
X_test = X_test[[ 'Age', 'Pclass', 'Sex', 'Family_Size']]


# In[48]:


# Select numerical columns
numerical_cols = [cname for cname in X.columns 
                  if X[cname].dtype in ['int64', 'float64']]


# In[49]:


# Select categorical columns with low cardinality (number of unique values in a column)
categorical_cols = [cname for cname in X.columns 
                    if X[cname].nunique() < 10 and
                    X[cname].dtype in ["object","category"]]


# In[50]:


# Keep selected columns only
my_cols = numerical_cols + categorical_cols
X = X[my_cols]
X_test = X_test[my_cols]


# ## Final X and X_test on which regression model will be trained

# In[51]:


print(len(categorical_cols),len(numerical_cols))


# In[52]:


X_test.head(2)


# In[53]:


show_info(X,X_test)


# # 8) Model Creation

# ## Preprocessing Pipelines

# In[54]:


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')


# In[55]:


# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[56]:


# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# ## Model Pipeline

# In[57]:


# Create object of XGBClassifier class
xgb = XGBClassifier()

# Bundle preprocessing and modeling code in a pipeline
classifier = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', xgb)
                     ])


# ## Visualize the pipeline

# In[58]:


from sklearn import set_config
set_config(display='diagram')
classifier


# # 9) Training and Testing Model

# ## Grid Search & Cross Validation

# In[59]:


param_grid = [
    {
        "model__reg_lambda": [50],
        "model__subsample": [0.7,1],
        "model__learning_rate": [0.4],
        "model__n_estimators": [5, 10, 20],
        "model__max_depth": [3, 4, 5]
    }
]
grid_search = GridSearchCV(classifier, param_grid, cv=3, verbose=1)
grid_search.fit(X, y);


# In[60]:


print("Best params:",grid_search.best_params_,sep='\t',end='\n\n')
print("Best score in grid search:",round(grid_search.best_score_,3),sep='\t')
print("Score on whole trained data:",round(grid_search.score(X, y),3),sep='\t')


# In[61]:


# Top 5 parameter combinations
df = pd.DataFrame(grid_search.cv_results_)
display(df.sort_values('rank_test_score')[:5])


# ## Feature Importance

# In[62]:


Feature_Imp = grid_search.best_estimator_.named_steps["model"].feature_importances_
Feature_Imp


# In[63]:


# Predicting y for trained data
y_pred = grid_search.predict(X)

# Converting y_pred from Array to DataFrame with appropriate index and column name
y_pred = pd.DataFrame(y_pred, index=X.index, columns=['Survived_Predicted'])


# In[64]:


# Checking trainig data without preprocessing (features, actual target and predicted target)
Xyy = pd.concat([X,y,y_pred],axis=1)
Xyy.head(2)


# In[65]:


# Preprocessed (Imputed and OneHotCoded) Training and Test data using Pipeline
column_values = ['Age','Pclass','Sex_female','Sex_male','Family_alone','Family_large','Family_medium']
X_processed = pd.DataFrame(grid_search.best_estimator_.named_steps["preprocessor"].transform(X), columns=column_values)
X_processed.head(2)


# In[66]:


ax =pd.Series(Feature_Imp,X_processed.columns).sort_values(ascending=True).plot.barh(width=0.8)
ax.set_title('Feature Importance in XgBoost')
plt.show()


# **Note:**
# * From above plot, we can see that *One Hot Encoded* column 'Sex_male' has zero importance in trained XGBoost model. This is because that same information is conveyed in 'Sex_feamle'. (If the passenger is not female, then the passenger is male.)

# ## Confusion Matrix (for training data)

# In[67]:


# Credit: 'https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea'
cm = confusion_matrix(y, y_pred)

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]

labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, xticklabels=['0','1'], yticklabels=['0','1'])
ax.set(ylabel="Actual y", xlabel="Predicted y");


# ## Classification Report (for training data)

# In[68]:


print(classification_report(y, y_pred))


# In[69]:


# Model Evaluation metrics 
print('Accuracy Score : ' + str(accuracy_score(y,y_pred).round(3)))
print('Precision Score : ' + str(precision_score(y,y_pred).round(3)))
print('Recall Score : ' + str(recall_score(y,y_pred).round(3)))
print('F1 Score : ' + str(f1_score(y,y_pred).round(3)))


# # 10) Predicting y

# In[70]:


# Generate test predictions
y_test_pred = grid_search.predict(X_test)

# Converting y_pred from Array to DataFrame with appropriate index and column name
y_test_pred = pd.DataFrame(y_test_pred, index=X_test.index, columns=['Survived_Predicted'])


# In[71]:


# Checking testing data with features and predicted target
Xy_test = pd.concat([X_test,y_test_pred],axis=1)
Xy_test.head(2)


# In[72]:


# Save output to CSV file
output = pd.DataFrame({'PassengerId': X_test.index,
                       'Survived': y_test_pred.Survived_Predicted})
output.to_csv('submission.csv', index=False)


# In[73]:


# Submit results
submission_data = pd.read_csv("submission.csv")
submission_data.head(2)

