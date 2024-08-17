#!/usr/bin/env python
# coding: utf-8

# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">💦 Notebook at a glance</p>

# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notebook Summary:</font></h3>
# 
# <b>Objective:</b> In this challenge, our primary aim is not predicting outcomes directly, but rather enhancing a dataset for subsequent training of a random forest model. This unique twist highlights the significance of data quality, ensuring its optimization to effectively capture underlying patterns.
# 
# <b>Dataset Origin:</b> The dataset provided for this competition originates from the "Dissolved oxygen prediction in river water" dataset, albeit synthetically modified. Participants are encouraged to creatively incorporate the original dataset, leveraging its potential for this challenge.
# 
# <b>Importance:</b> The famous saying "garbage in, garbage out" rings true in machine learning. The quality of input data profoundly influences the model's predictive prowess. By refining the dataset, we lay a sturdy groundwork for the random forest model, amplifying its predictive capabilities.
# 
# <b>Exploratory Data Analysis (EDA):</b> This EDA is meticulously designed to explore the dataset extensively, identifying potential areas for enhancement, uncovering anomalies, deciphering variable interactions, and ultimately structuring the data to harmonize with the random forest model's requirements.
# 
# <b>Desired Outcome:</b> As we wrap up this notebook, our objective is to furnish a dataset meticulously primed for effective modeling. Recommendations and insights gleaned from our analysis will drive the refining process, ensuring the data is impeccably prepared for subsequent modeling phases.
# 
# <b>Special Emphasis on Evaluation:</b> The refined data will fuel the training of a random forest model, enabling it to make predictions that dictate our competition score. This underscores the paramount significance of the data preprocessing phase.
# 
# <b>Domain Knowledge Integration:</b> Leveraging our domain expertise, we will introduce novel derived variables. These additions are poised to not only enrich the dataset but also potentially uncover latent patterns that could significantly enhance the model's performance.
# 
# <b>Best of luck to all participants in this endeavor!</b>
# 

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


# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">💦 Brief EDA</p>

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


df = pd.read_csv('/kaggle/input/playground-series-s3e21/sample_submission.csv')


# In[5]:


summary(df)


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Preprocessing Recommendations:</font></h3>
# 
# <b>Variable Scaling:</b> Due to the varied range of each variable, scaling all variables to match a consistent range can be beneficial. Options like MinMaxScaler or StandardScaler from scikit-learn can be utilized.
# 
# <b>Outlier Handling:</b> For instance, the minimum value for NO2_4 is indicated as -4, which is physically implausible and likely represents an error or outlier. Such values should be replaced or removed.
# 
# <b>Feature Aggregation:</b> For each chemical substance, statistics such as mean, median, standard deviation, max, and min values can be calculated to form new features.
# 
# <b>Correlation Analysis:</b> By analyzing the correlation between variables, those with a low correlation coefficient with the target variable can be removed. For variables with high correlations, consider merging or transforming to avoid multicollinearity.
# 
# <b>Feature Engineering:</b> New features can be constructed by combining values of various chemicals. An example could be the ratio of O2 to NO2.
# 
# <b>ID Variable Handling:</b> The 'id' variable seems to serve only as an identifier for each row, and it may be beneficial not to use it during model training.
# 
# <b>Non-linear Transformation:</b> Some variables might have a non-linear relationship with the target. Considering transformations like log or square root can be helpful in these instances.
# 
# <b>Data Visualization:</b> Visualize the distribution of each variable, their relationship with the target, and trends to gain deeper insights into the dataset.
# 
# By incorporating these preprocessing steps, one can enhance the quality of the dataset, enabling more accurate model predictions.
# 

# > ###  📊 distribution of target variable

# In[6]:


# let's check the distribution of target variable

sns.histplot(df, x="target", kde=True)
plt.show()


# > ####  the majority of the data points were clustered within the range of 3 to 18, making it challenging to discern the distribution clearly. To achieve a better representation and view of the distribution, I applied a log transformation to the target values, which not only spreads out the data points but also provides a clearer perspective on the skewness and distribution of the target variable.

# In[7]:


# Log transformation of target
df['log_target'] = np.log1p(df['target'])

# Plotting the distribution of the log-transformed target
plt.figure(figsize=(10, 6))
sns.histplot(df, x="log_target", kde=True, color='royalblue')

# Add a vertical line for the median
plt.axvline(df['log_target'].median(), color='red', linestyle='--')
plt.title("Distribution of Log-transformed Target Variable", fontsize=15)
plt.xlabel("Log-transformed Target")
plt.ylabel("Frequency")

# To display the original target values on the top axis
secax = plt.gca().secondary_xaxis('top', functions=(np.exp, np.log))
secax.set_xlabel('Original Target Values')

plt.show()


# > ###  📊 distribution of all variables

# In[8]:


features = [col for col in df.columns if col not in ['log_target', 'id', 'target']]


# In[9]:


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


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# <b>Initial Visualization:</b> Initially, we visualized the distribution of each variable to get a basic understanding of our dataset's characteristics. This initial step is essential as it provides a general overview of the data spread and potential patterns or anomalies that might exist.
# 
# <b>Multifaceted Analysis:</b> However, to delve deeper and grasp a more detailed understanding, it's beneficial to observe data from multiple perspectives. That's where the combination of histograms and boxplots comes in handy.
# 
# <b>Histograms:</b> These not only show the frequency of data points but also give insights about the central tendency (mean and median) and spread of the data. By overlaying KDE (Kernel Density Estimation), we can also visualize the probable density function of the variable.
# 
# <b>Boxplots:</b> Boxplots, on the other hand, are particularly useful for identifying outliers and understanding the variability of the data. The interquartile range (IQR) depicted by the box gives a compact view of the dataset's spread, and whiskers can show data variability outside the upper and lower quartiles. Notably, data points that lie beyond the whiskers can be potential outliers.
# 
# <b>Comprehensive Understanding:</b> Thus, by juxtaposing these two visualizations, we gain a comprehensive understanding of each variable, from its distribution and central tendency to its spread and potential outliers. This multi-faceted approach helps in ensuring we don't overlook any crucial aspect of the data during the exploratory analysis phase.
# 

# In[10]:


num_cols_count = len(features)
n_rows = num_cols_count

fig, axs = plt.subplots(n_rows, 2, figsize=(14, n_rows*5))

for idx, col in enumerate(features):

    # Plot histogram
    sns.histplot(data=df, x=col, kde=True, ax=axs[idx, 0], color='cornflowerblue', bins=30)
    axs[idx, 0].set_title(f'Histogram of {col}')
    axs[idx, 0].axvline(df[col].mean(), color='red', linestyle='--')  # 평균값 표시
    axs[idx, 0].axvline(df[col].median(), color='green', linestyle='-')  # 중앙값 표시
    axs[idx, 0].legend({'Mean':df[col].mean(), 'Median':df[col].median()})
    
    # Plot boxplot
    sns.boxplot(data=df, x=col, ax=axs[idx, 1], color='lightcoral')
    axs[idx, 1].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


# > ### 📊 correlation

# In[11]:


def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str='Dataset correlation') -> None:
    # Remove 'id' and 'log_target' columns
    df = df.drop(columns=['id', 'log_target'])

    corr = df.corr()
    fig, axes = plt.subplots(figsize=(20, 10))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    
    # Adjust the font size of the annotations using annot_kws
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap='YlOrRd', annot=True, annot_kws={"size": 6})
    
    plt.title(title_name)
    plt.show()

# Now, you can use this function to visualize the correlation heatmap of your desired dataset.
plot_correlation_heatmap(df, 'Dataset Correlation')


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 EDA Conclusion:</font></h3>
# 
# <b>Correlation Insights:</b> While our EDA unveiled certain variables with pronounced correlation, it's not widespread across the board. It's essential to note here that while high correlation might be a concern for some linear models, the random forest algorithm inherently manages multicollinearity. Hence, there may be no pressing need to address this.
# 
# <b>Scale Sensitivity:</b> Random forests are indeed less sensitive to variable scales compared to many other models. This attribute comes from their tree-based structure, where splits are based on variable thresholds rather than coefficients. Thus, the varying scales of predictors won't drastically affect its performance.
# 
# <b>Handling Outliers:</b> Another strong suit of the random forest is its resilience to outliers. Given its tree-based nature, the model creates splits based on conditions, which means it doesn't heavily rely on the assumption of normally distributed data. However, extremely pronounced outliers might still affect the quality of splits, so a thorough outlier investigation and potential treatment would still be advisable.
# 
# <b>Final Thoughts:</b> Even though the random forest is versatile and adaptive, refining our dataset is still pivotal. Properly curated data can accelerate model convergence, make training more efficient, and potentially lead to more accurate predictions. Therefore, while we might not need extensive preprocessing tailored for the random forest, ensuring the data quality and structure is still of paramount importance.

# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">💦 Feature Engineering & Evalution</p>

# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# <h3 align="left"><font color='#DEB887'>💡 feature engineering with domain knowledge:</font></h3>
# 
# **O2-to-chemical ratios:** 
# Dissolved oxygen (O2) can have relationships with other chemical compounds, especially when considering how certain chemicals can consume oxygen. Therefore, it might be insightful to consider the ratio of O2 to other chemicals:
# 
# - **O2 to NH4 ratio:** This ratio represents how much oxygen is present concerning ammonium ions. A higher ratio may suggest a lower level of pollutants or decomposition processes, while a lower ratio might hint at potential pollutants or active decomposition. 
#   $$\text{O2 to NH4 ratio} = \frac{O2}{NH4}$$
# 
# - **O2 to NO2 ratio:** Represents the amount of dissolved oxygen available concerning nitrite ions. 
#   $$\text{O2 to NO2 ratio} = \frac{O2}{NO2}$$
# 
# - **O2 to NO3 ratio:** It signifies the amount of dissolved oxygen concerning nitrate ions. Given that nitrates are often the end product of nitrification, this ratio might provide insights into the water's stage in the nitrogen cycle.
#   $$\text{O2 to NO3 ratio} = \frac{O2}{NO3}$$
# 
# - **O2 to BOD5 ratio:** Biochemical Oxygen Demand (BOD5) reflects the amount of oxygen required for microbial metabolism over a 5-day period. Thus, this ratio can offer insights into potential organic pollutant concentrations.
#   $$\text{O2 to BOD5 ratio} = \frac{O2}{BOD5}$$
# 
# **Nitrogen Cycle Indicators:** 
# Ammonia (NH4), nitrites (NO2), and nitrates (NO3) are all part of the nitrogen cycle. This cycle is vital in water ecosystems, and any deviation can be a signal of water quality issues.
#   $$\text{Total Nitrogen} = NH4 + NO2 + NO3$$
# 
# **Oxygen Demand Difference:** 
# The difference between the actual dissolved oxygen and the Biochemical Oxygen Demand (BOD5) can be a critical indicator.
#   $$\text{Oxygen Demand Difference} = O2 - BOD5$$
# 

# In[12]:


df = df.drop(columns=['id','log_target'])  # Drop 'id' column


# In[13]:


# Feature Engineering
stations = 7  # As per your clarification
for i in range(1, stations + 1):  
    # O2-to-chemical ratios
    df[f'O2_{i}_to_NH4_{i}'] = np.where(df[f'NH4_{i}'] != 0, df[f'O2_{i}'] / df[f'NH4_{i}'], df[f'O2_{i}'].mean())
    df[f'O2_{i}_to_NO2_{i}'] = np.where(df[f'NO2_{i}'] != 0, df[f'O2_{i}'] / df[f'NO2_{i}'], df[f'O2_{i}'].mean())
    df[f'O2_{i}_to_NO3_{i}'] = np.where(df[f'NO3_{i}'] != 0, df[f'O2_{i}'] / df[f'NO3_{i}'], df[f'O2_{i}'].mean())
    df[f'O2_{i}_to_BOD5_{i}'] = np.where(df[f'BOD5_{i}'] != 0, df[f'O2_{i}'] / df[f'BOD5_{i}'], df[f'O2_{i}'].mean())
    
    # Nitrogen Cycle Indicators
    df[f'Total_Nitrogen_{i}'] = df[f'NH4_{i}'] + df[f'NO2_{i}'] + df[f'NO3_{i}']
    
    # Oxygen Demand Difference
    df[f'O2_{i}_BOD5_diff'] = df[f'O2_{i}'] - df[f'BOD5_{i}']


# In[14]:


df.head()


# In[15]:


# 2. Scaling
scaler = StandardScaler()
features = df.drop(columns=['target'])
features_scaled = scaler.fit_transform(features)
df[features.columns] = features_scaled

# 3. Train-test split
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train and evaluate Random Forest
rf = RandomForestRegressor(n_estimators=1000, max_depth=7, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f'Train MAE: {mae_train}')
print(f'Test MAE: {mae_test}')


# In[16]:


# Train Data Visualization
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Train Data: Actual vs. Predicted')

# Test Data Visualization
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.7, color='r')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Test Data: Actual vs. Predicted')

plt.tight_layout()
plt.show()


# In[17]:


import matplotlib.cm as cm

# Feature importance visualization
importances = rf.feature_importances_
indices = np.argsort(importances)

# Color map
color_map = cm.get_cmap('YlOrRd')

plt.figure(figsize=(10, 15))
plt.title("Feature Importances")

plt.barh(range(X.shape[1]), importances[indices], align="center", color=color_map(importances[indices]))
plt.yticks(range(X.shape[1]), X.columns[indices])
plt.ylim([-1, X.shape[1]])
plt.tight_layout()
plt.show()



# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Feature Engineering Conclusion:</font></h3>
# 
# <b>Contributions of Engineered Variables:</b> Our exploratory data analysis shed light on the positive impact of certain derived variables on model performance. These additions to the dataset have indeed played a role in enhancing the model's predictive capabilities.
# 
# <b>Performance Discrepancy:</b> Notably, there exists a substantial disparity between the performance on the test set and the training set. While the random forest algorithm inherently manages some aspects of generalization, this discrepancy indicates the presence of overfitting, which needs to be addressed.
# 
# <b>Addressing Overfitting:</b> Given the constraints on model adjustment, our focus shifts towards rectifying overfitting through data-centric strategies. The innate adaptability of the random forest model can be leveraged, but proper data refinement remains key to achieving a balance between training and test set performance.
# 
# <b>To Be Continued:</b> As we progress, our challenge is to pinpoint effective ways to curb overfitting within the dataset, aligning with the model's limitations. By employing creative variable engineering and judicious preprocessing, we aim to bridge the gap between training and test performance.
# 
# Stay tuned for more insights in the next phase...

# In[ ]:




