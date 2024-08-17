#!/usr/bin/env python
# coding: utf-8

# # <div style="padding: 30px;color:white;margin:10;font-size:60%;text-align:left;display:fill;border-radius:10px;background-color:#FFFFFF;overflow:hidden;background-color:#3498db"><b>INTRODUCTION</b></div>
# 
# This study employs an **ensemble machine learning** approach with **optuna hyperparameter tuning** to predict wild blueberry yield based on factors affecting pollination efficiency. The dataset, derived from the [Wild Blueberry Pollination Simulation Model](http://https://www.kaggle.com/datasets/shashwatwork/wild-blueberry-yield-prediction-dataset), includes features such as `bee densities`, `temperature`, and `precipitation` during the bloom season. By optimizing the ensemble model, we aim to provide insights into the factors influencing wild blueberry yield
# 
# <br>
# 
# ![](https://veganonboard.com/wp-content/uploads/2018/09/DSC_0361.jpg.webp)
# 
# 
# ### <b><span style='color:#3498db'>|</span> Domain Knowledge</b>
# 
# <br>
# 
# 1. **`Clonesize` [Square Meter] :** The size of the blueberry clones could impact the overall yield because larger clones may produce more fruit. However, the relationship may not be linear, as other factors can also affect fruit production.
# 
# 2. **`Honeybee`, `Bumbles bee`, `Andrena bee`, `Osmia bee` [Square meter/mins] :** The densities of various bee species are crucial for pollination efficiency, which directly impacts fruit set and yield. Different bee species may have varying levels of effectiveness in pollinating wild blueberries.
# 
# 3. **`MaxOfUpperTRange`, `MinOfUpperTRange`, `AverageOfUpperTRange`, `MaxOfLowerTRange`, `MinOfLowerTRange`, `AverageOfLowerTRange` [Celcius] :** Temperature ranges can significantly affect plant growth and pollinator activity. Extreme temperatures can cause stress in plants or limit pollinator activity, which could negatively impact the yield.
# 
# 4. **`RainingDays`, `AverageRainingDays` [Days] :** Rainy days can impact the yield in various ways. On the one hand, adequate water availability is essential for plant growth. On the other hand, excessive rain or rain during the bloom season can reduce pollinator activity and affect fruit set.
# 
# 5. **`Fruitset`[proportion]:** Fruit set refers to the proportion of flowers that develop into fruits. This is a critical factor in determining the yield, as a higher fruit set usually results in a higher yield.
# 
# 6. **`Fruitmass` [grams] :** The mass of individual fruits is also an important factor, as larger fruits contribute more to the overall yield. Variability in fruit mass can be influenced by factors such as pollination efficiency, resource availability, and environmental conditions.
# 
# 7. **`Seeds` [unit] :** The number of seeds per fruit can have a secondary impact on yield. Although not a direct determinant of yield, a higher number of seeds might indicate better pollination, which could contribute to a higher fruit set and yield.

# # <div style="padding: 30px;color:white;margin:10;font-size:60%;text-align:left;display:fill;border-radius:10px;background-color:#FFFFFF;overflow:hidden;background-color:#3498db"><b><span style='color:#FFFFFF'>1.</span></b> <b>GETTING STARTED</b></div>
# 
# ### <b><span style='color:#3498db'> 1.1) </span> Import neccessary libraries</b>

# In[1]:


#Import library
import pandas as pd
import numpy as np
#Visualization
import seaborn as sns
import matplotlib.pyplot as plt
#Machine learning Models
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from scipy.cluster.hierarchy import linkage, dendrogram
#Import Optuna
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import r2_score
#Import warnings
import warnings
warnings.filterwarnings("ignore")
import logging
# Set the log level for the optuna package to WARNING
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ### <b><span style='color:#3498db'> 1.2) </span> Input the data</b>

# In[2]:


# Load the dataset
train_data = pd.read_csv("/kaggle/input/playground-series-s3e14/train.csv")
train_data= train_data.drop('id', axis=1)
data = train_data.copy()


# # <div style="padding: 30px;color:white;margin:10;font-size:60%;text-align:left;display:fill;border-radius:10px;background-color:#FFFFFF;overflow:hidden;background-color:#3498db"><b><span style='color:#FFFFFF'>2.</span></b> <b>EXPLORATORY DATA ANALYSIS</b></div>
# 
# ### <b><span style='color:#3498db'> 2.1) </span> Describe the Data</b>

# In[3]:


# apply styling to describe method
styled_data = train_data.describe().style \
    .background_gradient(cmap='Blues') \
    .set_properties(**{'text-align': 'center', 'border': '1px solid black'})

# display styled data
display(styled_data)


# ### <b><span style='color:#3498db'> 2.2) </span> Correlation and Importance features</b>

# In[4]:


corr = train_data.corr()
target_corr = corr['yield'].drop('yield')

# Create a heatmap of the correlations with the target column
sns.set(font_scale=1.2)
sns.set_style("white")
sns.set_palette("PuBuGn_d")
sns.heatmap(target_corr.to_frame(), cmap="coolwarm", annot=True, fmt='.2f')
plt.title('Correlation with Yield Column in Train Data')
plt.show()


# In[5]:


# Assuming 'data' is your DataFrame
correlation_matrix = train_data.corr()

# Set a correlation threshold (e.g., 0.5 or -0.5)
correlation_threshold = 0.1

# Get the correlations with the target variable (yield)
target_correlations = correlation_matrix['yield']

# Filter the features that have a correlation above the threshold (in absolute value) with the target variable
important_features = target_correlations[abs(target_correlations) >= correlation_threshold].index.tolist()

# Display the important features
print("Important features based on the Correlation threshold with the fruit yield column:", important_features)

# Create a new DataFrame with only the important features
important_data = train_data[important_features]


# ### <b><span style='color:#3498db'> 2.3) </span> Violin plot for highly correlated features</b>

# In[6]:


# Create violin plots for each feature against the target variable (yield)
for feature in important_features:
    if feature != 'yield' and feature != 'fruitset' and feature != 'fruitmass' and feature != 'seeds':
        plt.figure()
        sns.violinplot(x=feature, y='yield', data=important_data)
        plt.title(f"Violin plot of {feature} vs yield")
        plt.show()


# ### <b><span style='color:#3498db'> 2.4) </span> Scatter plots</b>

# In[7]:


# Create scatter plots for fruit set, fruit mass, and seeds against the target variable (yield)
plt.figure(figsize=(12,4))

plt.subplot(1, 3, 1)
sns.scatterplot(x='fruitset', y='yield', data=important_data)
plt.title("Scatter plot of fruit set vs yield")

plt.subplot(1, 3, 2)
sns.scatterplot(x='fruitmass', y='yield', data=important_data)
plt.title("Scatter plot of fruit mass vs yield")

plt.subplot(1, 3, 3)
sns.scatterplot(x='seeds', y='yield', data=important_data)
plt.title("Scatter plot of seeds vs yield")

plt.tight_layout()
plt.show()


# # <div style="padding: 30px;color:white;margin:10;font-size:60%;text-align:left;display:fill;border-radius:10px;background-color:#FFFFFF;overflow:hidden;background-color:#3498db"><b><span style='color:#FFFFFF'>3.</span></b> <b>FEATURE ENGINEERING</b></div>
# 
# ![](https://a-z-animals.com/media/2023/02/shutterstock_506285848-1536x1024.jpg)
# 
# 
# ### <b><span style='color:#3498db'>|</span> Feature Explanation</b>
# <br>
# 
# * **`Temperature range`:** Instead of using the upper and lower temperature ranges separately, we could create a new feature for the daily temperature range (upper - lower). This could help capture the impact of temperature fluctuations on pollination and yield.
# 
# * **`Temperature extremes`:** Create binary features indicating whether the daily temperature exceeds certain thresholds (e.g., extreme high or low temperatures). These features could help identify days with potentially harmful temperatures that could affect pollination and yield.
# 
# * **`Total bee density`:** Sum the densities of all bee species (Honeybee, Bumbles bee, Andrena bee, Osmia bee) to create a new feature for the total bee density. This could help capture the combined effect of all bee species on pollination and yield.
# 
# * **`Bee species dominance`**: Calculate the proportion of each bee species in the field, which can help capture the relative importance of each bee species in pollination.
# 
# * **`Rain intensity`:** Instead of just counting rainy days, you could calculate the average daily precipitation during the bloom season. This could provide a more accurate representation of the water availability during the bloom season.
# 
# * **`Interaction features`**

# In[8]:


# 1. Temperature range
data['TemperatureRange'] = data['MaxOfUpperTRange'] - data['MinOfLowerTRange']

# 2. Temperature extremes
threshold_high = 71.9  
threshold_low = 50    

data['ExtremeHighTemp'] = (data['AverageOfUpperTRange'] > threshold_high).astype(int)
data['ExtremeLowTemp'] = (data['AverageOfLowerTRange'] < threshold_low).astype(int)

# 3. Total bee density
data['TotalBeeDensity'] = data['honeybee'] + data['bumbles'] + data['andrena'] + data['osmia']

# 4. Bee species dominance
total_density = data['honeybee'] + data['bumbles'] + data['andrena'] + data['osmia']
data['HoneybeeDominance'] = data['honeybee'] / total_density
data['BumblesBeeDominance'] = data['bumbles'] / total_density
data['AndrenaBeeDominance'] = data['andrena'] / total_density
data['OsmiaBeeDominance'] = data['osmia'] / total_density

# 5. Rain intensity
data['RainIntensity'] = data['AverageRainingDays'] / data['RainingDays']

# 6. Interaction features
data['BeeDensity_TemperatureInteraction'] = data['TotalBeeDensity'] * data['TemperatureRange']
data['BeeDensity_RainInteraction'] = data['TotalBeeDensity'] * data['RainIntensity']


# ### <b><span style='color:#3498db'> 3.1) </span> Correlation and New importance features</b>

# In[9]:


corr = data.corr()
# Extract the correlations with the target column
target_corr = corr['yield'].drop('yield')


# Create a heatmap of the correlations with the target column
sns.set(font_scale=0.8)
sns.set_style("white")
sns.set_palette("PuBuGn_d")
sns.heatmap(target_corr.to_frame(), cmap="coolwarm", annot=True, fmt='.2f')
plt.title('Correlation with Yield Column in Data')
plt.show()


# # <div style="padding: 30px;color:white;margin:10;font-size:60%;text-align:left;display:fill;border-radius:10px;background-color:#FFFFFF;overflow:hidden;background-color:#3498db"><b><span style='color:#FFFFFF'>4.</span></b> <b>HYPER PARAMETERS TUNING</b></div>
# 
# ### <b><span style='color:#3498db'> 4.1) </span> Preprocess the data</b>

# In[10]:


# Preprocess the data
X = data.drop('yield', axis=1)
y = data['yield']


# In[11]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ### <b><span style='color:#3498db'> 4.2) </span> Optuna Optimization</b>

# In[12]:


# Define Optuna objectives for each model
def xgb_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    subsample = trial.suggest_float("subsample", 0.5, 1)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1)

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

def lgbm_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    num_leaves = trial.suggest_int("num_leaves", 31, 127)
    subsample = trial.suggest_float("subsample", 0.5, 1)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1)

    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

def catboost_objective(trial):
    iterations = trial.suggest_int("iterations", 100, 500)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2)
    depth = trial.suggest_int("depth", 3, 10)
    l2_leaf_reg = trial.suggest_int("l2_leaf_reg", 1, 9)
    subsample = trial.suggest_float("subsample", 0.5, 1)

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        subsample=subsample,
        random_state=42,
        verbose=0
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2


# In[13]:


# Run Optuna optimization for each model
sampler = TPESampler(seed=42)

xgb_study = optuna.create_study(direction="maximize", sampler=sampler)
xgb_study.optimize(xgb_objective, n_trials=50)

lgbm_study = optuna.create_study(direction="maximize", sampler=sampler)
lgbm_study.optimize(lgbm_objective, n_trials=50)

catboost_study = optuna.create_study(direction="maximize", sampler=sampler)
catboost_study.optimize(catboost_objective, n_trials=50)

# Print best hyperparameters for each model from Optuna
print("Best parameters for XGBoost from Optuna: ", xgb_study.best_params)
print("Best parameters for LightGBM from Optuna: ", lgbm_study.best_params)
print("Best parameters for CatBoost from Optuna: ", catboost_study.best_params)


# # <div style="padding: 30px;color:white;margin:10;font-size:60%;text-align:left;display:fill;border-radius:10px;background-color:#FFFFFF;overflow:hidden;background-color:#3498db"><b><span style='color:#FFFFFF'>5.</span></b> <b>ENSEMBLE MACHINE LEARNING</b></div>
# 
# ### <b><span style='color:#3498db'> 5.1) </span> Apply the Hyperparameters</b>

# In[14]:


# Use the best hyperparameters obtained from the previous GridSearchCV
xgb_best_params = {'n_estimators': 291, 'learning_rate': 0.02572640965568375, 'max_depth': 4, 'subsample': 0.552187345043806, 'colsample_bytree': 0.8272028535549523}
lgbm_best_params = {'n_estimators': 217, 'learning_rate': 0.07053452738526493, 'max_depth': 4, 'num_leaves': 86, 'subsample': 0.9281322440018367, 'colsample_bytree': 0.6325640166339597}
catboost_best_params = {'iterations': 288, 'learning_rate': 0.0537104881625418, 'depth': 9, 'l2_leaf_reg': 2, 'subsample': 0.542177159163148}

# Train multiple base models with the best parameters
xgb = XGBRegressor(**xgb_best_params, random_state=42)
lgbm = LGBMRegressor(**lgbm_best_params, random_state=42)
catboost = CatBoostRegressor(**catboost_best_params, random_state=42, verbose=0)


# In[15]:


# Feature selection using SelectKBest
selector = SelectKBest(score_func=f_regression, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)


# In[16]:


# Combine the base models to create an ensemble and assign weights
weights = [0.4, 0.3, 0.3]
# Combine the base models to create an ensemble
ensemble = VotingRegressor([('xgb', xgb), ('lgbm', lgbm), ('catboost', catboost)])
# Train the ensemble model using selected features
ensemble.fit(X_train_selected, y_train)


# ### <b><span style='color:#3498db'> 5.2) </span> Top features</b>

# In[17]:


# Train the best performing XGBoost model with the best hyperparameters from Optuna
best_xgb = XGBRegressor(**xgb_study.best_params, random_state=42)
best_xgb.fit(X_train, y_train)

# Get feature importance values
importances = best_xgb.feature_importances_

# Number of top features to visualize
top_n = 10

# Sort the features by importance and select the top N
sorted_idx = np.argsort(importances)[-top_n:]
top_features = data.columns[:-1][sorted_idx].tolist()

# Perform hierarchical clustering on the top N features
linked = linkage(np.array(importances[sorted_idx]).reshape(-1, 1), 'single')

# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked, labels=top_features, orientation='top', distance_sort='descending')
plt.title("Hierarchical Clustering of Top {} Feature Importance".format(top_n))
plt.xticks(rotation=90) 
plt.show()


# ### <b><span style='color:#3498db'> 5.3) </span> Cross Validation</b>

# In[18]:


# Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble, X_train_selected, y_train, cv=kfold, scoring='r2')
print("Cross-validation scores: ", cv_scores)
print("Mean CV R-squared: {:.2f}".format(np.mean(cv_scores)))
print("Standard Deviation of CV R-squared: {:.2f}".format(np.std(cv_scores)))


# In[19]:


# Make predictions using the selected features
y_pred = ensemble.predict(X_test_selected)

# Evaluate the performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: {:.2f}".format(mse))
print("R-squared: {:.2f}".format(r2))


# ### <b><span style='color:#3498db'> 5.4) </span> Residuals</b>

# In[20]:


# Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()


# # <div style="padding: 30px;color:white;margin:10;font-size:60%;text-align:left;display:fill;border-radius:10px;background-color:#FFFFFF;overflow:hidden;background-color:#3498db"><b><span style='color:#FFFFFF'>6.</span></b> <b>APPLY ON TEST DATASET</b></div>
# 
# ### <b><span style='color:#3498db'> 6.1) </span> Load test data</b>

# In[21]:


# Load the test dataset
test_df = pd.read_csv("/kaggle/input/playground-series-s3e14/test.csv")


# ### <b><span style='color:#3498db'> 6.2) </span> Apply feature engineering on test data</b>

# In[22]:


# 1. Temperature range
test_df['TemperatureRange'] = test_df['MaxOfUpperTRange'] - test_df['MinOfLowerTRange']

# 2. Temperature extremes
threshold_high = 72 
threshold_low = 50    

test_df['ExtremeHighTemp'] = (test_df['AverageOfUpperTRange'] > threshold_high).astype(int)
test_df['ExtremeLowTemp'] = (test_df['AverageOfLowerTRange'] < threshold_low).astype(int)

# 3. Total bee density
test_df['TotalBeeDensity'] = test_df['honeybee'] + test_df['bumbles'] + test_df['andrena'] + test_df['osmia']

# 4. Bee species dominance
total_density = test_df['honeybee'] + test_df['bumbles'] + test_df['andrena'] + test_df['osmia']
test_df['HoneybeeDominance'] = test_df['honeybee'] / total_density
test_df['BumblesBeeDominance'] = test_df['bumbles'] / total_density
test_df['AndrenaBeeDominance'] = test_df['andrena'] / total_density
test_df['OsmiaBeeDominance'] = test_df['osmia'] / total_density

# 5. Rain intensity
test_df['RainIntensity'] = test_df['AverageRainingDays'] / test_df['RainingDays']

# 6. Interaction features
test_df['BeeDensity_TemperatureInteraction'] = test_df['TotalBeeDensity'] * test_df['TemperatureRange']
test_df['BeeDensity_RainInteraction'] = test_df['TotalBeeDensity'] * test_df['RainIntensity']


# ### <b><span style='color:#3498db'> 6.3) </span> Apply the Model</b>

# In[23]:


# Get a list of all columns except 'id'
selected_features = test_df.drop('id', axis=1).columns.tolist()

# Select all columns except 'id' from test_df
X_test = test_df[selected_features]


# In[24]:


# Scale
X_test_scaled = scaler.transform(X_test)

# Apply feature selection
X_test_selected = selector.transform(X_test_scaled)

# Make predictions
y_test_pred = ensemble.predict(X_test_selected)


# ### <b><span style='color:#3498db'> 6.4) </span> OUTPUT</b>

# In[25]:


# Create a new DataFrame with ID column and predicted yields
result_df = pd.DataFrame({'id': test_df['id'], 'yield': y_test_pred})

# Save the DataFrame
result_df.to_csv('predictions.csv', index=False)


# ***
# 
# <BR>
#     
#     
# <div style="text-align: center;">
#    <span style="font-size: 4.5em; font-weight: bold; font-family: Arial;">THANK YOU!</span>
# </div>
# 
# <div style="text-align: center;">
#     <span style="font-size: 5em;">✔️</span>
# </div>
# 
# <br>
# 
# <div style="text-align: center;">
#    <span style="font-size: 1.4em; font-weight: bold; font-family: Arial; max-width:1200px; display: inline-block;">
#        If you discovered this notebook to be useful or enjoyable, I'd greatly appreciate any upvotes! Your support motivates me to regularly update and improve it. :-)
# 
#    </span>
# </div>
# 
# <br>
# 
# <br>
# 
# <div style="text-align: center;">
#    <span style="font-size: 1.2em; font-weight: bold;font-family: Arial;">@pannmie</span>
# </div>
