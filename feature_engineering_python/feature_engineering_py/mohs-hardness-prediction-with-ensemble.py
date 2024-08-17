#!/usr/bin/env python
# coding: utf-8

# <div style="border-radius:10px; border:#87d8de solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# <h3 align="left"><font color='#87d8de'>💡 Notebook At a Glance:</font></h3>
# 
# * **Objective of the Notebook**
#     * The primary aim of this notebook is to predict the Mohs hardness scale of various minerals. The Mohs hardness scale is a qualitative ordinal scale that characterizes the scratch resistance of various minerals through the ability of a harder material to scratch a softer material. It's a critical measure in mineralogy and is used extensively in geology and material science.
# 
# * **About Mohs Hardness**
#     * The Mohs hardness scale was created in 1812 by German geologist and mineralogist Friedrich Mohs and is one of the oldest scales for measuring hardness. The scale ranges from 1 (talc) to 10 (diamond), representing a range of minerals' resistance to scratching. This scale helps in identifying minerals and understanding their properties and applications.
# 
# * **Our Approach**
# 
#     * Exploratory Data Analysis (EDA): We will start with a thorough exploratory analysis of the dataset. This step involves understanding the distribution of data, identifying patterns, correlations, and potential outliers. EDA is crucial as it informs the subsequent steps in the modeling process.
# 
#     * Simple Ensemble Model: Post EDA, we will construct a simple ensemble model combining CatBoost, XGBoost, and LightGBM. These models are chosen for their ability to handle different types of data, robustness, and ease of use. An ensemble approach is expected to harness the strengths of each individual model, potentially leading to better predictive performance.
# 
# * **Conclusion**
#     * This notebook will serve as a comprehensive guide from data exploration to model building, with the goal of accurately predicting Mohs hardness. We anticipate that our approach will yield insightful results and contribute to a better understanding of how different factors influence the hardness of minerals.
#     
#     ![image.png](attachment:7cd0e5f8-a526-4d70-84e7-758e2c0d32e4.png)
#     *image source: gemsociety

# In[1]:


# CSS style setting
get_ipython().system('wget http://bit.ly/3ZLyF82 -O CSS.css -q')
    
from IPython.core.display import HTML
with open('./CSS.css', 'r') as file:
    custom_css = file.read()

HTML(custom_css)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from tqdm.auto import tqdm
import math
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

tqdm.pandas()

rc = {
    "axes.facecolor": "#FFF9ED",
    "figure.facecolor": "#FFF9ED",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4
}

sns.set(rc=rc)

from colorama import Style, Fore
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
mgt = Style.BRIGHT + Fore.MAGENTA
gld = Style.BRIGHT + Fore.YELLOW
res = Style.RESET_ALL


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# > #### 💦 Brief EDA

# In[3]:


# summary table function
def summary(df):
    # Print the shape of the DataFrame
    print(f'data shape: {df.shape}')  
    # Create a summary DataFrame
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    # Calculate the number of missing values
    summ['#missing'] = df.isnull().sum().values 
    # Calculate the percentage of missing values
    summ['%missing'] = df.isnull().sum().values / len(df)* 100
    # Calculate the number of unique values
    summ['#unique'] = df.nunique().values
    # Create a descriptive DataFrame
    desc = pd.DataFrame(df.describe(include='all').transpose())
    # Add the minimum, maximum, and first three values to the summary DataFrame
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['first value'] = df.loc[0].values
    summ['second value'] = df.loc[1].values
    summ['third value'] = df.loc[2].values
    
    # Return the summary DataFrame
    return summ


# In[4]:


df = pd.read_csv('/kaggle/input/playground-series-s3e25/train.csv')


# In[5]:


summary(df)


# <div style="border-radius:10px; border:#87d8de solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# <h3 align="left"><font color='#87d8de'>💡 Dataset Summary:</font></h3>
# 
# * The dataset contains a total of 10,407 entries, each represented by a row.
# * The target variable for prediction is 'Hardness'.
# * There are no missing values in any of the columns, indicating that no imputation for missing data is necessary.
# * Apart from the 'id' column, which is an identifier and of type int64, all other variables are of the float64 type.
# * The range of values across these float64 columns is quite broad, suggesting that if a non-tree-based model is used, scaling of the features would be necessary to ensure optimal model performance.

# In[6]:


# let's check the distribution of target variable

sns.histplot(df, x="Hardness", kde=True)
plt.show()


# <div style="border-radius:10px; border:#87d8de solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# <h3 align="left"><font color='#87d8de'>💡 About Target variable:</font></h3>
# 
# * The 'Hardness' values range from 1 to 10.
# * There are noticeable concentrations of data points around values 2, 4, and 6.

# In[7]:


features = [col for col in df.columns if col not in ['id', 'Hardness']]


# > #### 📊 Distributions of X varaibles

# In[8]:


# check numerical variables' distribution

features = features  # we created feature list above.
n_bins = 50
histplot_hyperparams = {
    'kde':True,
    'alpha':0.4,
    'stat':'percent',
    'bins':n_bins
}

columns = features
n_cols = 4
n_rows = math.ceil(len(columns)/n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
ax = ax.flatten()

for i, column in enumerate(columns):
    sns.kdeplot(
        df[column], 
        ax=ax[i], color='#9E3F00'
    )
    
    # titles
    ax[i].set_title(f'{column} Distribution');
    ax[i].set_xlabel(None)
    
for i in range(i+1, len(ax)):
    ax[i].axis('off')
    
fig.suptitle(f'Numerical Feature Distributions\n\n\n', ha='center',  fontweight='bold', fontsize=25)
plt.tight_layout()


# <div style="border-radius:10px; border:#87d8de solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# <h3 align="left"><font color='#87d8de'>💡 About Target variable:</font></h3>
# 
# * Skewness in Distribution: Variables like allelectrons_Total and density_Total show a left-skewed distribution. This means most of their values are clustered towards the lower end of the range.
# 
# * Outliers: Some variables in your dataset exhibit potential outliers, as indicated by the extreme values in the distributions. These outliers could significantly affect the overall dataset characteristics and model performance.

# > #### 📊 box plot distribution

# In[9]:


num_cols_count = len(features)
n_rows = num_cols_count

fig, axs = plt.subplots(n_rows, 2, figsize=(14, n_rows*5))

for idx, col in enumerate(features):

    # Plot histogram
    sns.histplot(data=df, x=col, kde=True, ax=axs[idx, 0], color='cornflowerblue', bins=30)
    axs[idx, 0].set_title(f'Histogram of {col}')
    axs[idx, 0].axvline(df[col].mean(), color='red', linestyle='--')  # mean
    axs[idx, 0].axvline(df[col].median(), color='green', linestyle='-')  # median
    axs[idx, 0].legend({'Mean':df[col].mean(), 'Median':df[col].median()})
    
    # Plot boxplot
    sns.boxplot(data=df, x=col, ax=axs[idx, 1], color='lightcoral')
    axs[idx, 1].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


# > #### 📊 correlation

# In[10]:


def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str='Dataset correlation') -> None:
    # Remove 'id' and 'log_target' columns
    df = df.drop(columns=['id'])

    corr = df.corr()
    fig, axes = plt.subplots(figsize=(20, 10))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    
    # Adjust the font size of the annotations using annot_kws
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap='YlOrRd', annot=True, annot_kws={"size": 8})
    
    plt.title(title_name)
    plt.show()

# Now, you can use this function to visualize the correlation heatmap of your desired dataset.
plot_correlation_heatmap(df, 'Dataset Correlation')


# <div style="border-radius:10px; border:#87d8de solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# <h3 align="left"><font color='#87d8de'>💡 Correlation Table:</font></h3>
# 
# * High Correlation Between Certain Features: The correlation coefficient of 0.99 between atomicweight_Average and allelectrons_Average suggests these two variables are almost identical in terms of the information they convey. This high degree of correlation implies redundancy, as both variables essentially represent the same characteristic.
# 
# * Other Highly Correlated Variables: There are also a few other variables with correlations exceeding 0.8. This indicates strong relationships between these variables, which could lead to multicollinearity issues in certain types of models.
# 
# * Low Correlation with Target Variable: The low correlation of all variables with the target variable 'Hardness' indicates that no single feature strongly predicts the target on its own. This suggests that the relationship between features and the target variable might be non-linear or complex.
# 
# * Implications for Model Choice:
#     * Tree-Based Models: For tree-based models, like Random Forest or Gradient Boosting, high correlation between features is generally less of a concern. These models can handle multicollinearity better than linear models.
#     * Other Models: If you're considering linear models or models that assume feature independence (like Naive Bayes), it's advisable to address the issue of high correlation. This could involve removing one of the highly correlated variables or using techniques like Principal Component Analysis (PCA) to reduce dimensionality.

# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">📌 baseline modeling code</p>

# <div style="border-radius:10px; border:#87d8de solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# <h3 align="left"><font color='#87d8de'>💡 About Evaluation Method - Median Absolute Error :</font></h3>
# 
# * The choice of using Median Absolute Error (MedAE) as the evaluation metric for your model's performance has specific implications:
# 
# * Focus on Median Error: Median Absolute Error evaluates the median of the absolute differences between predicted and actual values. This focus on the median makes MedAE more robust to outliers compared to mean-based metrics like Mean Absolute Error (MAE) or Mean Squared Error (MSE).
# 
# * Robustness to Outliers: Given that MedAE is less sensitive to outliers, it's a suitable metric if your dataset contains outliers or extreme values that you don't want to disproportionately impact the model's performance evaluation.
# 
# * Interpretability: MedAE is quite interpretable. It directly represents the median amount by which the predictions deviate from the actual values, making it easy to understand and explain.
# 
# * Model Optimization: When optimizing your model, you'll aim to minimize the MedAE. This means you'll be tuning your model to improve the median prediction accuracy rather than the mean, which can sometimes provide a more realistic assessment of model performance, especially in skewed datasets.
# 
# * Comparing Models: When comparing different models, those with a lower MedAE will be considered better in terms of median prediction accuracy. This is particularly useful when comparing models that might have different ways of dealing with outliers.

# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Assuming X and y are already defined, and features list excludes 'id'
X = df[features]
y = df['Hardness']

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
cat_model = CatBoostRegressor(random_state=42, silent=True)
xgb_model = XGBRegressor(random_state=42)
lgb_model = LGBMRegressor(random_state=42)

# Train models
cat_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)

# Make predictions
cat_predictions = cat_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)
lgb_predictions = lgb_model.predict(X_test)

# Ensemble predictions (simple averaging)
ensemble_predictions = (cat_predictions + xgb_predictions + lgb_predictions) / 3

# Evaluate ensemble model
ensemble_medae = median_absolute_error(y_test, ensemble_predictions)
print(f"Ensemble Model MedAE: {ensemble_medae}")


# In[12]:


import matplotlib.pyplot as plt
from sklearn.metrics import median_absolute_error

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter Plot: Predicted vs Actual
axes[0].scatter(y_test, ensemble_predictions, color='blue', alpha=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Actual Values')
axes[0].set_ylabel('Predicted Values')
axes[0].set_title('Scatter Plot: Predicted vs. Actual')

# Line Plot: Predicted vs Actual
axes[1].plot(y_test.values, color='blue', alpha=0.5, label='Actual')
axes[1].plot(ensemble_predictions, color='green', alpha=0.5, label='Predicted')
axes[1].set_title('Line Plot: Predicted vs. Actual')
axes[1].legend()

plt.tight_layout()
plt.show()



# <div style="border-radius:10px; border:#87d8de solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# <h3 align="left"><font color='#87d8de'>💡 Model traning conclusion :</font></h3>
# 
# * Model Approach: We employed an ensemble model consisting of CatBoost, XGBoost, and LightGBM. This approach leverages the strengths of each individual model, potentially offering a more robust and accurate predictive model compared to using any single one of these algorithms alone.
# 
# * Performance on Test Set: The ensemble model achieved a performance on the test set that was slightly superior to the established benchmark. This indicates the effectiveness of the ensemble strategy in capturing the complexities of the dataset and making accurate predictions.
# 
# * Opportunities for Improvement:
# 
#     * Hyperparameter Tuning: While the basic ensemble model outperformed the benchmark, there's room for improvement through hyperparameter tuning. Systematic grid search can be applied to each model in the ensemble to fine-tune their parameters for even better performance.
#     * Data Preprocessing: Further preprocessing steps such as feature scaling, handling missing values, feature engineering, or selection can contribute to model improvement. This also includes exploring different techniques for outlier handling or data transformation.
#     * Iterative Process: It's important to note that model development is iterative. Continuous evaluation and adjustment of model parameters and preprocessing steps are key to achieving optimal performance.
