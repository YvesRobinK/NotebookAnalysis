#!/usr/bin/env python
# coding: utf-8

# # Library & Data Import

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from category_encoders import OrdinalEncoder
from sklearn.decomposition import PCA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Load the dataframe
df = pd.read_csv('/kaggle/input/playground-series-s3e16/train.csv')




# ## Basic EDA

# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.tail()


# In[8]:


# Check for missing values
missing_values = df.isnull().sum()
print('Missing Values:')
print(missing_values)


# In[9]:


df.describe()


# In[10]:


# Distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[11]:


# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[12]:


# Scatter plot of numerical features against the target variable
numerical_features = ['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[feature], y=df['Age'])
    plt.title(f'{feature} vs Age')
    plt.xlabel(feature)
    plt.ylabel('Age')
    plt.show()


# In[13]:


# Box plots of categorical features against the target variable
categorical_features = ['Sex']
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[feature], y=df['Age'])
    plt.title(f'{feature} vs Age')
    plt.xlabel(feature)
    plt.ylabel('Age')
    plt.show()


# ## Data Imputation for 0 values that don't make sense

# In[14]:


# Check the count of zero values in the 'Height' column
zero_height_count = (df['Height'] == 0).sum()
print('Count of zero values in the Height column:', zero_height_count)

# Calculate the mean height excluding zero values
mean_height = df.loc[df['Height'] != 0, 'Height'].mean()

# Impute zero values with the mean height
df.loc[df['Height'] == 0, 'Height'] = mean_height

# Verify if zero values are imputed
zero_height_count_after = (df['Height'] == 0).sum()
print('Count of zero values in the Height column after imputation:', zero_height_count_after)



# ### Train Test Split

# In[15]:


# Split the data into features (X) and target variable (y)
X = df.drop('Age', axis=1)
y = df['Age']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)


# ## Linear Regression Baseline Model

# In[16]:


# Define the columns to be included in each preprocessing step
numeric_features = ['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']
categorical_features = ['Sex']

# Create preprocessing pipelines for numeric and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

# Apply the transformations to the columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the pipeline with preprocessing and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error (MAE): %.3f' % mae)


# ## Basic Feature Engineering

# In[17]:


# Feature engineering
df['Length_to_Diameter'] = df['Length'] / df['Diameter']
df['Volume'] = 3.1416 * (df['Diameter'] / 2) ** 2 * df['Height']
df['Shell_Density'] = df['Shell Weight'] / (df['Length'] * df['Diameter'] * df['Height'])
df['Sex_Encoded'] = df['Sex'].map({'I': 0, 'M': 1, 'F': 2})


# In[18]:


# Feature engineering functions
def cat_encoder(X_train, X_test, cat_cols, encode='label'):
    if encode == 'label':
        ## Label Encoder
        encoder = OrdinalEncoder(cols=cat_cols, handle_missing='ignore')
        train_encoder = encoder.fit_transform(X_train[cat_cols]).astype(int)
        test_encoder = encoder.transform(X_test[cat_cols]).astype(int)
        X_train[cat_cols] = train_encoder[cat_cols]
        X_test[cat_cols] = test_encoder[cat_cols]
        encoder_cols = cat_cols
    else:
        ## OneHot Encoder
        encoder = OneHotEncoder(cols=cat_cols)
        train_encoder = encoder.fit_transform(X_train[cat_cols]).astype(int)
        test_encoder = encoder.transform(X_test[cat_cols]).astype(int)
        X_train = pd.concat([X_train, train_encoder], axis=1)
        X_test = pd.concat([X_test, test_encoder], axis=1)
        X_train.drop(cat_cols, axis=1, inplace=True)
        X_test.drop(cat_cols, axis=1, inplace=True)
        encoder_cols = list(train_encoder.columns)
    return X_train, X_test, encoder_cols

def create_features(df):
    # Calculate the Length-to-Diameter Ratio
    df["Length_to_Diameter_Ratio"] = df["Length"] / df["Diameter"]
    # Calculate the Length-Minus-Height
    df["Length_Minus_Height"] = df["Length"] - df["Height"]
    # Calculate the Weight-to-Shell Weight Ratio
    # df["Weight_to_Shell_Weight_Ratio"] = df["Weight"] / (df["Shell Weight"] + 1e-15)
    return df

def add_pca_features(X_train, X_test):    
    # Select the columns for PCA
    pca_features = X_train.select_dtypes(include=['float64']).columns.tolist()
    n_components = 4 # len(pca_features)
    # Create the pipeline
    pipeline = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=n_components))])
    # Perform PCA
    pipeline.fit(X_train[pca_features])
    # Create column names for PCA features
    pca_columns = [f'PCA_{i}' for i in range(n_components)]
    # Add PCA features to the dataframe
    X_train[pca_columns] = pipeline.transform(X_train[pca_features])
    X_test[pca_columns] = pipeline.transform(X_test[pca_features])
    return X_train, X_test

# Create engineered features
df = create_features(df)


# In[19]:


df.head()


# In[20]:


# Split the data into features (X) and target variable (y)
X = df.drop('Age', axis=1)
y = df['Age']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Categorical features
cat_cols = ['Sex']

# Numerical features
num_cols = X_train.select_dtypes(include=['float64']).columns.tolist()

# Create the column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', 'passthrough', cat_cols)
    ])

# Encode categorical features and perform PCA
X_train, X_test, _ = cat_encoder(X_train, X_test, cat_cols, encode='label')
X_train, X_test = add_pca_features(X_train, X_test)


# In[21]:


X_train.head()


# ## Linear Regression including engineered features

# In[22]:


# Split the data into features (X) and target variable (y)
X = df.drop('Age', axis=1)
y = df['Age']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the columns to be included in each preprocessing step
numeric_features = ['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight',
                    'Shell Weight', 'Length_to_Diameter', 'Volume', 'Shell_Density']
categorical_features = ['Sex_Encoded']

# Create preprocessing pipelines for numeric and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

# Apply the transformations to the columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the pipeline with preprocessing and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error (MAE): %.3f' % mae)


# ## Random Forest Regression

# In[23]:


from sklearn.ensemble import RandomForestRegressor

# Create the pipeline with preprocessing and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor(n_estimators=100, random_state=420))])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error (MAE): %.3f' % mae)


# ## Feature Importance

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importances from the trained model
importances = pipeline.named_steps['regressor'].feature_importances_

# Get the feature names
numeric_feature_names = numeric_features
categorical_feature_names = pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)
feature_names = numeric_feature_names + list(categorical_feature_names)

# Create a dataframe of feature importance
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the dataframe by feature importance
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Print the top important features
print('Top Important Features:')
print(feature_importance_df.head())

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# ## Gradient Boosting Regressor

# In[25]:


from sklearn.ensemble import GradientBoostingRegressor

# Create the pipeline with preprocessing and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=420))])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error (MAE): %.3f' % mae)


# ## XGBoost, CatBoost, LightGBM

# In[26]:


import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb

# XGBoost
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', xgb.XGBRegressor(n_estimators=100, random_state=420))])

# CatBoost
catboost_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('regressor', CatBoostRegressor(iterations=100, random_state=420, verbose=0))])

# LightGBM
lgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', lgb.LGBMRegressor(n_estimators=100, random_state=420))])

# Fit the pipelines to the training data
xgb_pipeline.fit(X_train, y_train)
catboost_pipeline.fit(X_train, y_train)
lgb_pipeline.fit(X_train, y_train)

# Make predictions on the test set
xgb_pred = xgb_pipeline.predict(X_test)
catboost_pred = catboost_pipeline.predict(X_test)
lgb_pred = lgb_pipeline.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
catboost_mae = mean_absolute_error(y_test, catboost_pred)
lgb_mae = mean_absolute_error(y_test, lgb_pred)

print('XGBoost MAE: %.3f' % xgb_mae)
print('CatBoost MAE: %.3f' % catboost_mae)
print('LightGBM MAE: %.3f' % lgb_mae)


# ## Optuna Hyperparameter Search for LightGBM model

# In[27]:


import optuna
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage

# Define the LightGBM objective function for Optuna
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 0.9, 0.1),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.6, 0.9, 0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100),
        'n_estimators': 1000
    }

    # Create the pipeline with preprocessing and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', lgb.LGBMRegressor(**params))])

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Calculate the Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)

    return mae



# In[28]:


import warnings

# Suppress FutureWarning messages
warnings.simplefilter('ignore', FutureWarning)

# SQLite database configuration for Optuna study
storage_name = "sqlite:///optuna_study.db"
storage = RDBStorage(storage_name)

# Optimize the LightGBM model using Optuna with a TPESampler
sampler = TPESampler(seed=420)
study = optuna.create_study(direction='minimize', sampler=sampler, storage=storage)
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params

# Create the final LightGBM model with the best hyperparameters
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', lgb.LGBMRegressor(**best_params))])

# Fit the final pipeline to the training data
final_pipeline.fit(X_train, y_train)

# Make predictions on the test set using the final model
y_pred = final_pipeline.predict(X_test)

# Calculate the Mean Absolute Error (MAE) of the final model
mae = mean_absolute_error(y_test, y_pred)
print('Final Model MAE: %.3f' % mae)


# In[29]:


print(study.best_value,'\n') # print parameter values
display(study.best_params)


# In[30]:


fig = optuna.visualization.plot_optimization_history(study) # see a graphical plot of the study optimization trajectory

fig.show()


# ## Submission

# In[31]:


# Load the test data
test_data = pd.read_csv('/kaggle/input/playground-series-s3e16/test.csv')

# Categorical features
cat_cols = ['Sex']

# Numerical features
num_cols = X_train.select_dtypes(include=['float64']).columns.tolist()


# In[32]:


# Apply the same preprocessing steps
test_features = create_features(test_data)
test_features, _, _ = cat_encoder(test_data, test_data, cat_cols, encode='label')
test_features, _ = add_pca_features(test_data, test_data)

test_features


# In[33]:


# # Load the trained model (use the best model you obtained)
# model = lgb.LGBMRegressor()
# pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
# pipeline.fit(X_train, y_train)

# # Make predictions on the test data
# predictions = model.predict(test_data)



# In[34]:


# # Format the predictions in the required format
# submission = pd.DataFrame({'Age': predictions})
# submission.index.name = 'id'

# # Save the predictions to a CSV file
# submission.to_csv("submission.csv", index= False)


# In[35]:


# predictions.to_csv("submission.csv", index= False)

