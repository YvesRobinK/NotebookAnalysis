#!/usr/bin/env python
# coding: utf-8

# <div style="padding: 35px;color:white;margin:10;font-size:200%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2274725/pexels-photo-2274725.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>Getting Started </span></b> </div>
# 
# <br>
# 
# The aim of this analysis is to examine a range of crab-specific characteristics and their interconnections in order to predict the age of the crab accurately. These characteristics include aspects such as `Sex`, `Length`, `Diameter`, `Height`, `Weight`, `Shucked Weight`, `Viscera Weight`, and `Shell Weight`. Through a comprehensive exploratory data analysis (EDA), we will identify significant factors that can help us predict the age of crabs with a higher degree of precision and accuracy. 
# 
# <br>
# 
# ![](https://images.pexels.com/photos/584501/pexels-photo-584501.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)
# 
# <br>
# 
# ### <b><span style='color:#16C2D5'>|</span> Domain Knowledge</b>
# 
# <br>
# 
# 1. **`Sex`:** This refers to the biological gender of the crabs and is typically categorized as male or female. Understanding the sex distribution of the population can be important for studies on reproductive behavior, life cycles, or gender-specific characteristics in crabs.
# 
# 2. **`Length`:** This is the measurement of the crab from the tip of its rostrum (the forward-most point of the head) to the rear of the carapace (the "shell" that protects the crab). Length is a primary indicator of size and growth, often correlating with other attributes like age and weight.
# 
# 3. **`Diameter`:** This is the measurement across the widest part of the crab's body. The diameter, along with length and height, helps determine the overall size and shape of the crab.
# 
# 4. **`Height`:** Height is the vertical measurement of the crab. This could be measured from the base of the crab to the top of its shell, giving an indicator of the crab's overall body shape and size.
# 
# 5. **`Weight`:** This is the overall weight of the crab, typically measured when the crab is alive and in its entirety. Weight can be an indicator of the crab's health, age, and stage in the growth cycle.
# 
# 6. **`Shucked Weight`:** This refers to the weight of the crab without its shell. The shell is a significant portion of a crab's weight, so the shucked weight can provide insights into the meat yield of the crab, which could be particularly relevant in commercial fishing or aquaculture contexts.
# 
# 7. **`Viscera Weight`:** This refers to the weight of the internal organs of the crab, commonly known as the 'guts' or 'innards'. These organs are deep inside the body and protected by the shell.
# 
# 8. **`Shell Weight`:** This is the weight of the crab's shell once it's been separated from the body. The weight and condition of the shell can provide important clues about the crab's growth, age, and overall health.
# 
# 9. **`Age`:** This refers to the age of the crab, likely measured in months. Aging crabs can be challenging, but it's an important factor in understanding the lifecycle, growth rates, and population dynamics of the species.
# 
# 
# <br>
# 
# ✔️ **Each of these measurements provides a unique piece of information about the individual crab and can be used collectively to gain a comprehensive understanding of the crab population's health, diversity, and growth.**
# 
# # <span style="color:#E888BB; font-size: 1%;">INTRODUCTION</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/15665165/pexels-photo-15665165/free-photo-of-crabs-with-claws.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>INTRODUCTION </span></b> </div>
# 
# <br>
# 
# ### <b>I <span style='color:#16C2D5'>|</span> Abstract</b> 
# 
# <br>
# 
# In this project, we conducted a comprehensive study on age prediction of crabs using a combination of advanced machine learning techniques. We leveraged a dataset composed of several features, including `gender`, `length`, `diameter`, `height`, `weight`, `shucked weight`, `viscera weight`, and `shell weight`, to predict the `age` of crabs.
# 
# Our methodology started with a detailed **<span style='color:#16C2D5'>exploratory data analysis</span>** to gain insights into the data. We **<span style='color:#16C2D5'>preprocessed the data</span>**  to handle **categorical variables** and **scaled numerical variables** to facilitate learning for the model. Subsequently, we divided the data into **<span style='color:#16C2D5'>*training and testing sets</span>** to ensure the robustness and generalizability of our models.
# 
# We employed two gradient boosting algorithms, **<mark style="background-color:#16C2D5;color:white;border-radius:5px;opacity:1.0">XGBoost</mark>** and **<mark style="background-color:#16C2D5;color:white;border-radius:5px;opacity:1.0">CatBoost</mark>** , for prediction. The effectiveness of these models highly depends on their **<span style='color:#16C2D5'>hyperparameters</span>**, so we performed hyperparameter tuning using **<span style='color:#16C2D5'>GridSearchCV</span>**  and **<span style='color:#16C2D5'>RandomizedSearchCV</span>** .
# 
# To measure the performance of our models, we calculated the **<span style='color:#16C2D5'>Root Mean Square Error (RMSE)</span>** , **<span style='color:#16C2D5'>Mean Absolute Error (MAE)</span>** , and the **<span style='color:#16C2D5'>Coefficient of Determination (R-squared)</span>** . We also examined the feature importance to identify the variables that significantly influence the models' decisions.
# 
# Moreover, we used **<span style='color:#16C2D5'>cross-validation</span>**  to evaluate the model's performance across different subsets of the data, providing us with a more comprehensive assessment. Finally, we combined the predictions from both models using ensemble learning, which helped enhance the overall predictive power.
# 
# ### <b>II <span style='color:#16C2D5'>|</span> Import libraries</b> 

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from catboost import CatBoostRegressor


# ### <b>III <span style='color:#16C2D5'>|</span> Input the data</b> 

# In[2]:


df = pd.read_csv("/kaggle/input/playground-series-s3e16/train.csv")


# In[3]:


df.head()


# # <span style="color:#E888BB; font-size: 1%;">1 | EXPLORATORY DATA ANALYSIS</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/15665165/pexels-photo-15665165/free-photo-of-crabs-with-claws.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>1 | EXPLORATORY DATA ANALYSIS </span></b> </div>
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2274725/pexels-photo-2274725.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'> 1. Data Quality</span></b> </div>
# 
# ### <b>I <span style='color:#16C2D5'>|</span> Handling Duplicates</b> 

# In[4]:


# Handle duplicates
duplicate_rows_data = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_data.shape)


# ### <b>II <span style='color:#16C2D5'>|</span> Uniqueness</b> 

# In[5]:


# Loop through each column and count the number of distinct values
for column in df.columns:
    num_distinct_values = len(df[column].unique())
    print(f"{column}: {num_distinct_values} distinct values")


# ### <b>III <span style='color:#16C2D5'>|</span> Missing Values</b> 

# In[6]:


# Checking null values
print(df.isnull().sum())


# ### <b>IV <span style='color:#16C2D5'>|</span> Describe the Data</b> 

# In[7]:


df.describe().style.format("{:.2f}")


# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2274725/pexels-photo-2274725.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'> 2. Univariate Analysis</span></b> </div>

# In[8]:


data = df.copy()
# Drop column 'ID'
data = data.drop('id', axis=1)


# In[9]:


data.hist(bins=50, figsize=(20,15))
plt.show()


# In[10]:


# Bar plot for gender
sns.countplot(x='Sex', data=df)
plt.title('Gender Distribution')
plt.show()


# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2274725/pexels-photo-2274725.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'> 2. Bivariate Analysis</span></b> </div>

# In[11]:


#Scatter plot between Length and Weight
sns.scatterplot(x='Length', y='Weight', data=df)
plt.title('Length vs Weight')
plt.show()


# In[12]:


#Box plot of Weight by Sex
sns.boxplot(x='Sex', y='Weight', data=df)
plt.title('Weight Distribution by Sex')
plt.show()


# In[13]:


#Box plot of Weight by Sex
sns.barplot(x='Sex', y='Weight', data=df, estimator=np.mean)
plt.title('Average Weight by Sex')
plt.show()


# In[14]:


#Violinplot For Each Numeric Variable Split By Sex
for column in data.select_dtypes(include=[np.number]).columns:
    sns.violinplot(x='Sex', y=column, data=df)
    plt.show()


# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2274725/pexels-photo-2274725.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'> 3. Multivariable Analysis</span></b> </div>

# In[15]:


sns.pairplot(df, vars=["Length", "Diameter", "Height", "Weight"], hue="Sex")
plt.show()


# In[16]:


sns.scatterplot(x='Length', y='Diameter', hue='Weight', data=df, palette='viridis')
plt.title('Length vs Diameter colored by Weight')
plt.show()


# In[17]:


sns.pairplot(df, vars=["Shell Weight", "Shucked Weight", "Viscera Weight", "Weight"], hue='Sex')


# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2274725/pexels-photo-2274725.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'> 4. Multivariate Analysis</span></b> </div>

# In[18]:


sns.scatterplot(x='Length', y='Weight', hue='Sex', data=df)


# In[19]:


sns.scatterplot(x='Diameter', y='Weight', hue='Sex', data=df)


# In[20]:


sns.scatterplot(x='Height', y='Weight', hue='Sex', data=df)


# In[21]:


sns.scatterplot(x='Age', y='Shell Weight', hue='Sex', data=df)


# In[22]:


sns.scatterplot(x='Age', y='Shucked Weight', hue='Sex', data=df)


# In[23]:


sns.scatterplot(x='Age', y='Viscera Weight', hue='Sex', data=df)


# In[24]:


data.head()


# # <span style="color:#E888BB; font-size: 1%;">2 | CORRELATION</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/15665165/pexels-photo-15665165/free-photo-of-crabs-with-claws.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>2 | CORRELATION </span></b> </div>
# 

# In[25]:


def perform_one_hot_encoding(df, column_name):
    # Perform one-hot encoding on the specified column
    dummies = pd.get_dummies(df[column_name], prefix=column_name)

    # Drop the original column and append the new dummy columns to the dataframe
    df = pd.concat([df.drop(column_name, axis=1), dummies], axis=1)

    return df

# Perform one-hot encoding on the gender variable
data = perform_one_hot_encoding(data, 'Sex')


# In[26]:


# Compute the correlation matrix
correlation_matrix = data.corr()

#Graph I.
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title("Correlation Matrix Heatmap")
plt.show()

#Graph II
# Create a heatmap of the correlations with the target column
corr = data.corr()
target_corr = corr['Age'].drop('Age')

# Sort correlation values in descending order
target_corr_sorted = target_corr.sort_values(ascending=False)

sns.set(font_scale=0.8)
sns.set_style("white")
sns.set_palette("PuBuGn_d")
sns.heatmap(target_corr_sorted.to_frame(), cmap="coolwarm", annot=True, fmt='.2f')
plt.title('Correlation with Age')
plt.show()


# ### <b><span style='color:#16C2D5'>|</span> Intepret the results </b>
# 
# <br>
# 
# * **`Shell Weight`** (0.66): This has the highest positive correlation with Age. This suggests that as the shell weight increases, the age of the creature also tends to increase. The correlation is fairly strong, indicating a significant relationship.
# 
# * **`Height`** (0.64): This also has a strong positive correlation with Age. As the height of the creature increases, the age also tends to increase.
# 
# * **`Diameter`** (0.62), **`Length`** (0.61), **`Weight`**  (0.60), and **`Viscera Weight`** (0.58): These all have positive correlations with Age, suggesting that as these measurements increase, the age of the creature also tends to increase. The correlations are fairly strong, indicating a significant relationship.
# 
# * **`Shucked Weight`** (0.50): This has a moderate positive correlation with Age. As the shucked weight increases, the age of the creature also tends to increase, but the relationship is not as strong as the other measurements.
# 
# * **`Sex_F`** (0.29): This suggests that females tend to be older than males, but the correlation is weak, indicating that sex is not a strong predictor of age in this case.
# 
# * **`Sex_M`** (0.22): This suggests that males tend to be younger, but again, the correlation is weak, indicating that sex is not a strong predictor of age.
# 
# * **`Sex_I`**(-0.52): This has a moderate negative correlation with Age. This suggests that individuals classified as "I" (possibly indicating "Indeterminate" or "Immature") tend to be younger. The negative correlation indicates that as the likelihood of the creature being "I" increases, the age tends to decrease.
# 
# <br>
# 
# <div style="border-radius:10px;border:#16C2D5 solid;padding: 15px;background-color:#ffffff00;font-size:100%;text-align:left">
#     💬 Please note that correlation does not imply causation. While these variables are associated with Age, it doesn't necessarily mean that changes in these variables cause changes in Age. Other factors not included in this dataset may also influence Age.
#     </div>
# 
# 
# # <span style="color:#E888BB; font-size: 1%;">3 | PREDICTIVE ANALYSIS</span>
# <div style="padding: 30px;color:white;margin:10;font-size:150%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/15665165/pexels-photo-15665165/free-photo-of-crabs-with-claws.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>3 | PREDICTIVE ANALYSIS</span></b> </div>
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2274725/pexels-photo-2274725.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'> 1.Clustering Analysis </span></b> </div>

# In[27]:


# Load the dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s3e16/train.csv')

# Create a copy of the dataframe to not alter the original
train_data_preprocessed = train_data.copy()

# Preprocessing: Label encoding for categorical variables
le = LabelEncoder()
categorical_features = ['Sex']  # Adjust this to your data
for feature in categorical_features:
    train_data_preprocessed[feature] = le.fit_transform(train_data[feature])

# Preprocessing: MinMax scaling for numerical/ratio variables
mm = MinMaxScaler()
numerical_features = ['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']
for feature in numerical_features:
    train_data_preprocessed[feature] = mm.fit_transform(train_data_preprocessed[feature].values.reshape(-1,1))

# Exclude 'Id' and 'Age' from t-SNE
train_data_preprocessed = train_data_preprocessed.drop(columns=['id', 'Age'])

# Apply t-SNE with different perplexity and learning rate
tsne = TSNE(n_components=2, random_state=42, perplexity=50, learning_rate=200)
tsne_results = tsne.fit_transform(train_data_preprocessed)

# Plotly Interactive plot
df_tsne = pd.DataFrame(data = tsne_results, columns = ['Dim_1', 'Dim_2'])
df_tsne['Age'] = train_data['Age']  # Color by 'Age'
fig = px.scatter(df_tsne, x='Dim_1', y='Dim_2', color='Age', title='t-SNE plot colored by Age')
fig.show()


# In[28]:


# Exclude 'Id' and 'Age' from clustering
X = train_data.drop(columns=['id', 'Age'])

# One-hot encoding of the categorical feature 'Sex'
X = pd.get_dummies(X)

# Standardize the data to have a mean of ~0 and a variance of 1
X_std = StandardScaler().fit_transform(X)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)  # Change to 2 components
X_tsne = tsne.fit_transform(X_std)

# Perform KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_std)

# Create DataFrame for Plotting
df_tsne = pd.DataFrame(X_tsne, columns=['Component 1', 'Component 2'])
df_tsne['Cluster'] = clusters

# Plot t-SNE components and color by clusters
fig = px.scatter(df_tsne, x='Component 1', y='Component 2',
                 color='Cluster', color_continuous_scale='Rainbow', opacity=0.5)

# Update colorbar title
fig.update_layout(coloraxis_colorbar=dict(title="Cluster"))

# Show the plot
fig.show()


# In[29]:


# Load the dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s3e16/train.csv')

# Exclude 'Id' and 'Age' from clustering
X = train_data.drop(columns=['id', 'Age'])

# One-hot encoding of the categorical feature 'Sex'
X = pd.get_dummies(X)

# Standardize the data to have a mean of ~0 and a variance of 1
X_std = StandardScaler().fit_transform(X)

# Perform t-SNE
tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X_std)

# Perform KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_std)

# Create DataFrame for Plotting
df_tsne = pd.DataFrame(X_tsne, columns=['Component 1', 'Component 2', 'Component 3'])
df_tsne['Cluster'] = clusters

# Plot t-SNE components and color by clusters
fig = px.scatter_3d(df_tsne, x='Component 1', y='Component 2', z='Component 3', 
                    color='Cluster', color_continuous_scale='Rainbow', opacity=0.5)

# Update colorbar title
fig.update_layout(coloraxis_colorbar=dict(title="Cluster"))

# Show the plot
fig.show()


# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2274725/pexels-photo-2274725.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'> 2. Prediction </span></b> </div>

# In[30]:


# Load the dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s3e16/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s3e16/test.csv')


# In[31]:


# Split the features and target variable, exclude 'Id' from features
X = train_data.drop(columns=['id', 'Age'])
y = train_data['Age']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Keep the 'Id' column of the test dataset for final submission
test_ids = test_data['id']
X_test = test_data.drop(columns=['id'])


# In[32]:


# Preprocessing
numeric_features = ['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']
categorical_features = ['Sex']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])


# In[33]:


# Define XGBRegressor and CatBoostRegressor models
xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
cat_model = CatBoostRegressor(random_state=42, silent=True)


# In[34]:


# Create pipelines
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', xgb_model)])

cat_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', cat_model)])


# In[35]:


# Define parameter grid for XGBoost
xgb_params = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__max_depth': [3, 5, 7],
    'regressor__subsample': [0.5, 0.7, 1.0],
    'regressor__colsample_bytree': [0.6, 0.8, 1.0],
}


# In[36]:


# Define parameter grid for CatBoost
cat_params = {
    'regressor__iterations': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__depth': [4, 6, 8],
    'regressor__l2_leaf_reg': [1, 3, 5],
}


# In[37]:


# Randomized search for XGBoost
xgb_random_search = RandomizedSearchCV(xgb_pipeline, xgb_params, n_iter=50, cv=5, verbose=0, n_jobs=1, random_state=42)
xgb_random_search.fit(X_train, y_train)

# Print the best parameters from the randomized search for XGBoost
print(f"Best parameters for XGBoost: {xgb_random_search.best_params_}")


# ### <b><span style='color:#16C2D5'>|</span> Intepret the results </b>
# 
# Let's break down each parameter and its selected value:
# 
# * **regressor__subsample: 0.7** - This parameter controls the fraction of the total data that is provided to any given individual tree. In this case, each tree is trained on 70% of the total data. This is done to prevent overfitting and to ensure that the model generalizes well to unseen data.
# 
# * **regressor__n_estimators: 300** - This represents the number of trees that will be built in the XGBoost model. More trees usually lead to more robust models but also increase the computational complexity. In your case, 300 trees have been found to be the optimal number.
# 
# * **regressor__max_depth: 3** - This is the maximum depth of any given tree. The depth of a tree is defined as the length of the longest path from the root of the tree to a leaf. Deeper trees can model more complex relationships by adding more splits in the data, but they also risk overfitting. In this case, a max depth of 3 was found to be optimal.
# 
# * **regressor__learning_rate: 0.1** - This parameter, also known as the "step size", controls how quickly the model learns. Lower values require more trees to be built but can often lead to more accurate models. In this case, a learning rate of 0.1 was chosen as optimal.
# 
# * **regressor__colsample_bytree: 0.8** - This parameter controls the fraction of features that are sampled for building each tree. A value of 0.8 means that 80% of the total features are sampled for each tree. Similar to subsampling, this is done to prevent overfitting and to ensure model generalization.
# 
# <div style="border-radius:10px;border:#16C2D5 solid;padding: 15px;background-color:#ffffff00;font-size:100%;text-align:left">
#     💬 The values of these parameters were likely found through a process of hyperparameter optimization. These are the values that, when used in the XGBoost model, yielded the highest cross-validated performance on the training data. Note that these values are dataset-specific and might not generalize to other datasets. Furthermore, using these values for future predictions assumes that the underlying data distribution does not change significantly.
#     </div>
# 
# 

# In[38]:


# Randomized search for CatBoost
cat_random_search = RandomizedSearchCV(cat_pipeline, cat_params, n_iter=50, cv=5, verbose=0, n_jobs=1, random_state=42)
cat_random_search.fit(X_train, y_train)

# Print the best parameters from the randomized search for CatBoost
print(f"Best parameters for CatBoost: {cat_random_search.best_params_}")


# ### <b><span style='color:#16C2D5'>|</span> Intepret the results </b>
# 
# Let's break down each of these parameters and their optimal values:
# 
# * **regressor__learning_rate: 0.1** - The learning rate is a tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function. In the context of boosting, this parameter controls how quickly the model fits the residual error using additional base learners. A smaller learning rate requires more base learners to achieve similar complexity as a larger learning rate. In this case, a learning rate of 0.1 has been found optimal, providing a balance between model complexity and speed of learning.
# 
# * **regressor__l2_leaf_reg: 3** - This parameter refers to a regularization term in the cost function of CatBoost, which helps to prevent overfitting by discouraging complex models. Specifically, the l2_leaf_reg parameter controls the L2 regularization term on the weights of the leaf values. Higher values make the model more conservative, while lower values make the model fit the data more closely. The optimal value found for your model is 3.
# 
# * **regressor__iterations: 300** - This is the maximum number of trees that will be built in the CatBoost model, similar to the n_estimators parameter in XGBoost. More trees can model more complex relationships but also increase the computational complexity and the risk of overfitting. In this case, 300 trees have been found to be the optimal number.
# 
# * **regressor__depth: 6**- This parameter controls the depth of the trees in the CatBoost model. Deeper trees can model more complex relationships by creating more splits in the data, but they also risk overfitting and increase computational complexity. In your model, a depth of 6 was found to be optimal.
# 
# <div style="border-radius:10px;border:#16C2D5 solid;padding: 15px;background-color:#ffffff00;font-size:100%;text-align:left">
#     💬 These parameters are the result of a hyperparameter optimization process, and they yielded the best performance on your training data according to a certain evaluation metric during cross-validation. Please note that these values are specific to the current dataset and may not necessarily generalize to other datasets. The optimal parameters are also dependent on the specific distribution and characteristics of the target variable . Using these parameter values for future predictions assumes that the underlying data distribution does not significantly change.
#     </div>
# 
# 

# In[39]:


# Model evaluation using cross-validation
xgb_cv_scores = cross_val_score(xgb_random_search.best_estimator_, X_train, y_train, cv=5)
cat_cv_scores = cross_val_score(cat_random_search.best_estimator_, X_train, y_train, cv=5)

print(f"Cross-Validation scores for XGBoost: {xgb_cv_scores}")
print(f"Cross-Validation scores for CatBoost: {cat_cv_scores}")


# ### <b><span style='color:#16C2D5'>|</span> Intepret the results </b>
# 
# In this case, the CatBoost model has slightly higher scores on most of the folds compared to the **<mark style="background-color:#16C2D5;color:white;border-radius:5px;opacity:1.0">XGBoost model</mark>** . It suggests that the **<mark style="background-color:#16C2D5;color:white;border-radius:5px;opacity:1.0">CatBoost model</mark>**  may perform better on this dataset, though the difference is not very large.
# 
# <div style="border-radius:10px;border:#16C2D5 solid;padding: 15px;background-color:#ffffff00;font-size:100%;text-align:left">
#     ⚠️ It's important to note that these scores are specific to the current dataset and the way it was split in this particular run of cross-validation. Different datasets or different splits might lead to different results.
#     </div>
# 
# 
# 

# In[40]:


# Plotting cross-validation scores
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(range(len(xgb_cv_scores)), xgb_cv_scores)
plt.xlabel('Fold')
plt.ylabel('R^2 Score')
plt.title('XGBoost Cross Validation Scores')

plt.subplot(1, 2, 2)
plt.bar(range(len(cat_cv_scores)), cat_cv_scores)
plt.xlabel('Fold')
plt.ylabel('R^2 Score')
plt.title('CatBoost Cross Validation Scores')

plt.tight_layout()
plt.show()


# In[41]:


# Feature importance for XGBoost and CatBoost
xgb_importances = xgb_random_search.best_estimator_.named_steps['regressor'].feature_importances_
cat_importances = cat_random_search.best_estimator_.named_steps['regressor'].get_feature_importance()

# print feature importances
print(f"Feature importances for XGBoost: {xgb_importances}")
print(f"Feature importances for CatBoost: {cat_importances}")


# ### <b><span style='color:#16C2D5'>|</span> Intepret the results </b>
# 
# <br>
# 
# **<mark style="background-color:#16C2D5;color:white;border-radius:5px;opacity:1.0">XGBoost</mark>** **Feature Importances:**
# 
# * `Length`: 0.02205274
# * `Diameter`: 0.01318688
# * `Height`: 0.14367083
# * `Weight`: 0.02896496
# * `Shucked Weight`: 0.06952686
# * `Viscera Weight`: 0.012254
# * `Shell Weight`: 0.43509355
# * `Sex_F`: 0.03038907
# * `Sex_I`: 0.23940954
# * `Sex_M`: 0.00545156
# 
# > The `Shell Weight` is the most important feature according to the **<mark style="background-color:#16C2D5;color:white;border-radius:5px;opacity:1.0">XGBoost model</mark>**, followed by `Sex_I` and `Height`. The least important features are `Sex_M`, `Viscera Weight`, and `Diameter`.
# 
# <br>
# 
# **<mark style="background-color:#16C2D5;color:white;border-radius:5px;opacity:1.0">CatBoost</mark>** **Feature Importances:**
# 
# * `Length`: 1.87162805
# * `Diameter`: 3.49696017
# * `Height`: 6.92279983
# * `Weight`: 10.15507769
# * `Shucked Weight`: 26.54986719
# * `Viscera Weight`: 3.95315652
# * `Shell Weight`: 40.3998022
# * `Sex_F`: 0.55914348
# * `Sex_I`: 5.90157306
# * `Sex_M:` 0.18999181
# 
# > According to the **<mark style="background-color:#16C2D5;color:white;border-radius:5px;opacity:1.0">CatBoost model</mark>**  model, the `Shell Weight` is again the most important feature, followed by `Shucked Weight` and `Weight`. The least important features are `Sex_M`, `Sex_F`, and `Length`.
# 
# <br>
# 
# <div style="border-radius:10px;border:#16C2D5 solid;padding: 15px;background-color:#ffffff00;font-size:100%;text-align:left">
# 📝These feature importance scores can help us understand which features are driving the predictions of our model. They can be especially useful for feature selection (removing irrelevant features can sometimes improve model performance) and for understanding the relationships in the data (important features are likely to be strongly related to the target variable).
#     </div>

# In[42]:


# Get the feature names from the pipeline
ohe_feature_names = xgb_random_search.best_estimator_.named_steps['preprocessor'].transformers_[1][1]\
.get_feature_names_out()
numeric_feature_names = xgb_random_search.best_estimator_.named_steps['preprocessor'].transformers_[0][2]

# Combine numeric and one-hot encoded feature names
xgb_feature_names = np.concatenate([numeric_feature_names, ohe_feature_names])

# Feature importance plots
plt.figure(figsize=(12, 6))
plt.barh(range(len(xgb_feature_names)), xgb_importances)
plt.yticks(range(len(xgb_feature_names)), xgb_feature_names)
plt.xlabel('Importance')
plt.title('XGBoost Feature Importances')
plt.show()


# In[43]:


# Ensemble learning
voting_regressor = VotingRegressor(estimators=[('xgb', xgb_random_search.best_estimator_), 
                                               ('cat', cat_random_search.best_estimator_)])

# Fit the VotingRegressor to the training data
voting_regressor.fit(X_train, y_train)

# Predict on the validation set and calculate metrics
y_val_pred = voting_regressor.predict(X_val)
print(f'Validation RMSE: {np.sqrt(mean_squared_error(y_val, y_val_pred))}')
print(f'Validation MAE: {mean_absolute_error(y_val, y_val_pred)}')
print(f'Validation R2 Score: {r2_score(y_val, y_val_pred)}')


# ### <b><span style='color:#16C2D5'>|</span> Intepret the result </b>
# 
# <br>
# 
# **<span style='color:#16C2D5'>The Root Mean Square Error (RMSE)</span>** for our model is **approximately 2.03**. This value tells us that, on average, our model's predictions are about 2.03 units off from the actual values. This metric gives us a good indication of how accurately our model is able to predict the age of the crab. 
# 
# > However, it's important to note that the RMSE metric squares the errors before they are averaged, which means that larger errors are given relatively more weight than smaller errors. Thus, a RMSE of 2.03 indicates a moderate amount of prediction error in our model.
# 
# **<span style='color:#16C2D5'>The Mean Absolute Error (MAE)</span>** , on the other hand, is **about 1.41**. Unlike RMSE, MAE gives equal weight to all errors, regardless of their size. 
# 
# > This means that, on average, our model's predictions deviate from the actual values by about 1.41 units in either direction. This suggests that our model's predictions are reasonably close to the actual values.
# 
# **<span style='color:#16C2D5'>The R2 score </span>** for our model is **approximately 0.59** , or 59.38% when expressed as a percentage. This tells us that about 59.38% of the variation in the age of the crab can be explained by the features included in our model. 
# 
# > While this isn't a perfect score (which would be 1.0 or 100%), it's a reasonable result and suggests that our model has a fair degree of predictive power. However, there's still room for improvement, and we might be able to increase this score by incorporating additional relevant features into our model or by tuning our model's parameters more effectively.
# 
# <br>
# 
# <div style="border-radius:10px;border:#16C2D5 solid;padding: 15px;background-color:#ffffff00;font-size:100%;text-align:left">
# 📝These results provide a good starting point, but there's definitely more work that could be done to improve our model's accuracy. In the next steps, I would consider exploring additional feature engineering opportunities, fine-tuning our model's parameters further, or even trying out different modeling approaches.
#     </div>
# 
# # <span style="color:#E888BB; font-size: 1%;">4 | RESULT AND DISCUSSION</span>
# <div style="padding: 30px;color:white;margin:10;font-size:150%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/15665165/pexels-photo-15665165/free-photo-of-crabs-with-claws.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>4 | RESULT AND DISCUSSION</span></b> </div>
# 
# ### <b><span style='color:#16C2D5'>|</span> Result </b>
# 
# Our project focused on predicting the age of crabs using a dataset with a variety of features, including `sex`, `length`, `diameter`, `height`, `weight`, `shucked weight`, `viscera weight`, and `shell weight`. We leveraged two powerful machine learning models for this task: **<mark style="background-color:#16C2D5;color:white;border-radius:5px;opacity:1.0">XGBoost and CatBoost</mark>** .
# 
# From our exploration and tuning, we identified the best **<span style='color:#16C2D5'>hyperparameters</span>** for both models: For **<mark style="background-color:#16C2D5;color:white;border-radius:5px;opacity:1.0">XGBoost</mark>**, **a learning rate of 0.1, a maximum tree depth of 3, subsample ratio of 0.7, and a column sample by tree of 0.8 performed the best.** For **<mark style="background-color:#16C2D5;color:white;border-radius:5px;opacity:1.0">CatBoost</mark>**, **a learning rate of 0.1, L2 leaf regularization of 3, iterations of 300, and tree depth of 6 were the optimal settings**.
# 
# In terms of model performance, both models achieved similar results, with **<span style='color:#16C2D5'>hyperparameters</span>**cross-validation scores for XGBoost ranging from 0.56 to 0.59, and for CatBoost, scores ranged from 0.56 to 0.59.
# 
# When examining **<span style='color:#16C2D5'>feature importances</span>** , `Shell Weight` was the most influential feature in predicting crab age for both models. This aligns with domain knowledge, as a crab's shell grows as the crab ages.
# 
# For both models, the validation **<span style='color:#16C2D5'>RMSE</span>** was approximately 2.03, the **<span style='color:#16C2D5'>MAE </span>** was approximately 1.41, and the **<span style='color:#16C2D5'>R2 score</span>** was about 0.59, indicating a moderate level of predictive accuracy.
# 
# ### <b><span style='color:#16C2D5'>|</span> Discussion </b>
# 
# From the results, it can be concluded that both the **<mark style="background-color:#16C2D5;color:white;border-radius:5px;opacity:1.0">XGBoost and CatBoost</mark>** models offer a reasonable degree of accuracy in predicting crab age from the given features. However, the results also indicate room for improvement. Shell Weight appeared as the most significant predictor, but reliance on a single feature could limit the model's capacity to capture complex patterns in the data.
# 
# In terms of the model performance metrics, the achieved **<span style='color:#16C2D5'>RMSE, MAE, and R2 scores</span>** are satisfactory but not excellent. The models are moderately accurate but do not explain all the variance in the data.
# 
# > Moving forward, there are several possible paths for improving these models. Exploring additional feature engineering techniques, such as creating interactions between features or deriving new features, could be valuable. Similarly, additional hyperparameter tuning might lead to better results. Finally, while XGBoost and CatBoost are robust models, other modeling approaches could also be worth exploring.
# 
# <br>
# 
# <div style="border-radius:10px;border:#16C2D5 solid;padding: 15px;background-color:#ffffff00;font-size:100%;text-align:left">
# 📝In conclusion, the project was successful in demonstrating the application of machine learning to predict crab age and provided insights into the key factors influencing crab age. However, as is often the case in data science, there remain further opportunities for improvement and exploration.
#     </div>
#     
# 
# # <span style="color:#E888BB; font-size: 1%;">5 | APPLY ON TEST DATA</span>
# <div style="padding: 30px;color:white;margin:10;font-size:150%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/15665165/pexels-photo-15665165/free-photo-of-crabs-with-claws.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>5 | APPLY ON TEST DATA</span></b> </div>

# In[44]:


# Predict on the test set
y_test_pred = voting_regressor.predict(X_test)

# Visualize prediction results
plt.figure(figsize=(10, 5))
sns.histplot(y_test_pred, bins=30)
plt.title('Distribution of Predicted Age')
plt.show()


# In[45]:


# Prepare final submission dataframe
submission = pd.DataFrame({
    'id': test_ids,
    'Age': y_test_pred
})
print(submission.head())


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
#    <span style="font-size: 1.6em; font-weight: bold;font-family: Arial;"><a href="https://www.kaggle.com/tumpanjawat/code">@pannmie</a></span>
# </div>
# 
