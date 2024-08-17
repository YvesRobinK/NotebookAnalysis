#!/usr/bin/env python
# coding: utf-8

# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">✍️ Notebook at a glance</p>

# ![image.png](attachment:6a79b34d-aa14-438c-aa78-3e6e5d85a53d.png)

# In[1]:


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
import re as re
from collections import Counter

from tqdm.auto import tqdm
import math
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import warnings
warnings.filterwarnings('ignore')

import time
from xgboost import XGBClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
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
bold_start = Style.BRIGHT
bold_end = Style.NORMAL


# In[3]:


train = pd.read_csv('/kaggle/input/playground-series-s3e24/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e24/test.csv')


# In[4]:


# summary table function
pd.options.display.float_format = '{:,.2f}'.format
def summary(df):
    print(f'data shape: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values 
    summ['%missing'] = df.isnull().sum().values / len(df) * 100
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['average'] = desc['mean'].values
    summ['standard_deviation'] = desc['std'].values
    summ['first value'] = df.loc[0].values
    summ['second value'] = df.loc[1].values
    summ['third value'] = df.loc[2].values
    
    return summ


# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#4bce55; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">📊 EDA</p>

# In[5]:


summary(train).style.background_gradient(cmap='YlOrBr')


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Data summary:</font></h3>
# 
# - **Total Number of Variables**: 24
# - **Data Types**:
#   * **Float64**: 6 (waist, eyesight(left), eyesight(right), hemoglobin, serum creatinine)
#   * **Int64**: 18 (All other variables including the target 'smoking')
# - **Missing Values**: None (All variables have 0 missing values)
# - **Target Variable**: 'smoking' (Binary: 0 or 1)
# 
# <h4 align="left">💼 Features Overview:</h4>
# 
# 1. **id**: Unique identifier for each data point.
# 2. **age**: Age of the individual, categorized in 5-year intervals.
# 3. **height(cm)**: Height of the individual in centimeters.
# 4. **weight(kg)**: Weight of the individual in kilograms.
# 5. **waist(cm)**: Waist circumference of the individual in centimeters.
# 6. **eyesight(left/right)**: Eyesight measurements for the left and right eyes.
# 7. **hearing(left/right)**: Hearing ability for the left and right ears, represented as binary.
# 8. **systolic**: Systolic blood pressure measurement.
# 9. **relaxation**: Diastolic blood pressure measurement.
# 10. **fasting blood sugar**: Fasting blood sugar level.
# 11. **Cholesterol**: Total cholesterol level.
# 12. **triglyceride**: Triglyceride level.
# 13. **HDL**: High-density lipoprotein cholesterol level.
# 14. **LDL**: Low-density lipoprotein cholesterol level.
# 15. **hemoglobin**: Hemoglobin level in the blood.
# 16. **Urine protein**: Level of protein in urine, categorized.
# 17. **serum creatinine**: Serum creatinine level.
# 18. **AST**: Level of aspartate aminotransferase enzyme.
# 19. **ALT**: Level of alanine aminotransferase enzyme.
# 20. **Gtp**: Level of gamma-glutamyl transferase enzyme.
# 21. **dental caries**: Presence (1) or absence (0) of dental cavities.
# 22. **smoking**: Target variable indicating if the individual is a smoker (1) or not (0).
# 
# <h4 align="left">📌 Important Notes:</h4>
# 
# - There are no missing values in the dataset, so you don't have to consider imputation methods.
# - The target variable is 'smoking' which is binary. This indicates whether the individual smokes or not.
# 

# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_count(df: pd.core.frame.DataFrame, col_list: list, title_name: str='Train') -> None:
    """Draws the pie and count plots for categorical variables.
    
    Args:
        df (pd.core.frame.DataFrame): A pandas dataframe representing the data to be analyzed. 
            This could be a training set, test set, etc.
        col_list (list): A list of categorical variable column names from 'df' to be analyzed.
        title_name (str): The title of the graph. Default is 'Train'.
        
    Returns:
        None. This function produces pie and count plots of the input data and displays them using matplotlib.
    """

    # Creating subplots with 2 columns for pie and count plots for each variable in col_list
    f, ax = plt.subplots(len(col_list), 2, figsize=(12, 5))
    plt.subplots_adjust(wspace=0.3)

    for col in col_list:

        # Computing value counts for each category in the column
        s1 = df[col].value_counts()
        N = len(s1)

        outer_sizes = s1
        inner_sizes = s1/N

        # Colors for the outer and inner parts of the pie chart
        outer_colors = ['#FF6347', '#20B2AA']
        inner_colors = ['#FFA07A', '#40E0D0']

        # Creating outer pie chart
        ax[0].pie(
            outer_sizes, colors=outer_colors, 
            labels=s1.index.tolist(), 
            startangle=90, frame=True, radius=1.2, 
            explode=([0.05]*(N-1) + [.2]),
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'}, 
            textprops={'fontsize': 14, 'weight': 'bold'},
            shadow=True
        )

        # Creating inner pie chart
        ax[0].pie(
            inner_sizes, colors=inner_colors,
            radius=0.8, startangle=90,
            autopct='%1.f%%', explode=([.1]*(N-1) + [.2]),
            pctdistance=0.8, textprops={'size': 13, 'weight': 'bold', 'color': 'black'},
            shadow=True
        )

        # Creating a white circle at the center
        center_circle = plt.Circle((0,0), .5, color='black', fc='white', linewidth=0)
        ax[0].add_artist(center_circle)

        # Barplot for the count of each category in the column
        sns.barplot(
            x=s1, y=s1.index, ax=ax[1],
            palette='coolwarm', orient='horizontal'
        )

        # Customizing the bar plot
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax[1].set_ylabel('')  # Remove y label

        # Adding count values at the end of each bar
        for i, v in enumerate(s1):
            ax[1].text(v, i+0.1, str(v), color='black', fontweight='bold', fontsize=14)

        # Adding labels and title
        plt.setp(ax[1].get_yticklabels(), fontweight="bold")
        plt.setp(ax[1].get_xticklabels(), fontweight="bold")
        ax[1].set_xlabel(col, fontweight="bold", color='black', fontsize=14)

    # Setting a global title for all subplots
    f.suptitle(f'{title_name} Dataset Distribution of {col}', fontsize=20, fontweight='bold', y=1.05)

    # Adjusting the spacing between the plots
    plt.tight_layout()    
    plt.show()


# In[7]:


plot_count(train, ['smoking'], 'Train')


# In[8]:


# remove id and target variable
num_variables = train.select_dtypes(include=[np.number]).columns.tolist()
num_variables.remove('id')
num_variables.remove('smoking')

# log transform
for col in num_variables:
    train[col] = train[col].apply(lambda x: np.log(x+1))

# adjust the size of graph
plt.figure(figsize=(14, len(num_variables)*4))

for idx, column in enumerate(num_variables):
    plt.subplot(len(num_variables)//2 + len(num_variables)%2, 2, idx+1)
    sns.histplot(x=column, hue="smoking", data=train, bins=30, kde=True, palette='YlOrRd')
    plt.title(f"{column} Distribution (Log Transformed)")
    plt.tight_layout()
    
plt.show()


# <div style="border-radius:10px; border:#87d8de solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# <h3 align="left"><font color='#87d8de'>💡 Observations Post-Visualization:</font></h3>
# 
# * **Log Transformation**: From the updated visualizations, it's clear that the distribution of many variables in our dataset was skewed. This skewed distribution can hinder the efficacy of many machine learning algorithms that assume a Gaussian distribution. Hence, applying a log transformation is a common technique to handle such scenarios. After the transformation, the distributions of the variables became more centralized, making them easier to analyze.
# 
# * **Discriminative Power of Variables**: Some variables, such as `weight` and `triglyceride`, show a distinct difference in distributions when split by the 'smoking' target feature. This suggests that these variables might have a strong discriminative power and can be crucial in distinguishing between the two classes (smokers and non-smokers) in our target variable.
# 
# Such insights are invaluable as we progress with the data analysis. Recognizing the potential significance of these variables can guide feature engineering efforts and inform model selection. This understanding can also be instrumental in creating more robust and accurate predictive models.
# 

# In[9]:


def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str='Train correlation') -> None:

    corr = df.corr()
    fig, axes = plt.subplots(figsize=(12, 8))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap='RdBu_r', annot=True,annot_kws={"size": 8})
    plt.title(title_name)
    plt.show()


plot_correlation_heatmap(train.drop('id', axis=1), 'Train Dataset Correlation')


# <div style="border-radius:10px; border:#87acde solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# <h3 align="left"><font color='#87acde'>💡 Observations on Correlation Analysis:</font></h3>
# 
# * **Intuitive Correlations**: Some variable pairs in our dataset, such as `height` and `weight`, exhibit strong correlations, which align with our general understanding. These variables naturally move in tandem in many real-world scenarios, and their correlation coefficient of 0.7 in our dataset confirms this relationship. Similarly, `Cholesterol` and `LDL` also display a high correlation, which is consistent with medical knowledge since LDL is a component of total cholesterol.
# 
# * **Implications for Modeling**: While high correlation between independent variables can be problematic for some machine learning models, especially linear regression, due to multicollinearity, our choice of using tree-based models reduces this concern. Tree-based models, like decision trees and random forests, are less sensitive to multicollinearity. Thus, even if some variables are highly correlated, the tree-based models can handle them without significant performance degradation.
# 
# * **Strategic Decision**: The choice to proceed with tree-based models, given the correlations observed, is strategic. It not only mitigates the potential issues arising from multicollinearity but also leverages the non-linear decision boundaries that tree-based models can create, potentially capturing complex relationships in the data.
# 

# In[10]:


from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

def perform_hierarchical_clustering(input_data, title):
    """
    Function to perform hierarchical clustering and visualize the clusters in the form of a dendrogram.
    
    Args:
        input_data (pd.core.frame.DataFrame): The data on which to perform hierarchical clustering.
        title (str): The title for the plot.

    Returns:
        None. Displays a dendrogram plot.
    """
    
    # Compute the correlation matrix of the input data
    # It measures linear relationships between variables
    correlation_matrix = input_data.corr()

    # Convert the correlation matrix to distances
    # The distance represents dissimilarity: a value of 0 indicates perfect correlation, while a value of 1 indicates no correlation.
    distance_matrix = 1 - np.abs(correlation_matrix)

    # Perform hierarchical clustering using the "complete" method
    # The "complete" linkage method calculates the maximum distance between sets of observations at each iteration.
    Z = linkage(squareform(distance_matrix), 'complete')
    
    # Create a new figure and set the size and resolution
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=120)
    
    # Plot the dendrogram
    # The dendrogram visually shows the arrangement of the clusters produced by the corresponding executions of the hierarchical clustering.
    # The y-axis represents the distance or dissimilarity between clusters.
    # The x-axis represents the data points or variables.
    dn = dendrogram(Z, labels=input_data.columns, ax=ax, above_threshold_color='#ff0000', orientation='right', color_threshold=0.7*max(Z[:,2]))
    hierarchy.set_link_color_palette(None)  # Reset color palette to default
    
    # Add gridlines to the plot for better readability
    plt.grid(axis='x')
    
    # Set the title of the plot
    plt.title(f'{title} Hierarchical clustering, Dendrogram', fontsize=18, fontweight='bold')

    # Display the plot
    plt.show()

# Perform hierarchical clustering on the train dataset (excluding 'id' column if it exists)
input_data = train.drop(columns=['id'], errors='ignore') if 'id' in train.columns else train
perform_hierarchical_clustering(input_data, title='Train data')


# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">📌 baseline modeling code</p>

# In[11]:


def f_importance_plot(f_imp):
    fig = plt.figure(figsize=(12, 0.20*len(f_imp)))
    plt.title('Feature importances', size=16, y=1.05, 
              fontweight='bold', color='#444444')
    a = sns.barplot(data=f_imp, x='avg_imp', y='feature', 
                    palette='YlOrBr_r', linestyle="-", 
                    linewidth=0.5, edgecolor="black")
    plt.xlabel('')
    plt.xticks([])
    plt.ylabel('')
    plt.yticks(size=11, color='#444444')
    
    for j in ['right', 'top', 'bottom']:
        a.spines[j].set_visible(False)
    for j in ['left']:
        a.spines[j].set_linewidth(0.5)
    plt.tight_layout()
    plt.show()
    
def show_confusion_roc(oof: list) -> None:
    """Draws a confusion matrix and roc_curve with AUC score.
        
        Args:
            oof: predictions for each fold stacked. (list of tuples)
        
        Returns:
            None
    """
    
    f, ax = plt.subplots(1, 2, figsize=(13.3, 4))
    df = pd.DataFrame(np.concatenate(oof), columns=['id', 'preds', 'target']).set_index('id')
    df.index = df.index.astype(int)
    cm = confusion_matrix(df.target, df.preds.ge(0.5).astype(int))
    cm_display = ConfusionMatrixDisplay(cm).plot(cmap='YlOrBr_r', ax=ax[0])
    ax[0].grid(False)
    RocCurveDisplay.from_predictions(df.target, df.preds, color='#20BEFF', ax=ax[1])
    plt.tight_layout();
    
def get_mean_auc(oof_results):
    """Calculate the mean AUC from out-of-fold predictions."""
    true_values = []
    preds = []
    
    for oof in oof_results:
        true_values.extend(oof[:, 2])
        preds.extend(oof[:, 1])
    
    return roc_auc_score(true_values, preds)


# In[12]:


FOLDS = 10
SEED = 1004
xgb_models = []
xgb_oof = []
predictions = np.zeros(len(test))
f_imp = []

counter = 1
X = train.drop(columns=['smoking', 'id'])
y = train['smoking']
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    if (fold + 1) % 5 == 0 or (fold + 1) == 1:
        print(f'{"#" * 24} Training FOLD {fold + 1} {"#" * 24}')
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]
    watchlist = [(X_train, y_train), (X_valid, y_valid)]

    # XGboost model and fit with GPU support
    model = XGBClassifier(n_estimators=1000, n_jobs=-1, max_depth=4, eta=0.2, colsample_bytree=0.67, tree_method='gpu_hist')
    model.fit(X_train, y_train, eval_set=watchlist, early_stopping_rounds=300, verbose=0)

    val_preds = model.predict_proba(X_valid)[:, 1]
    val_score = roc_auc_score(y_valid, val_preds)
    best_iter = model.best_iteration

    idx_pred_target = np.vstack([val_idx, val_preds, y_valid]).T  # shape(len(val_idx), 3)
    f_imp.append({i: j for i, j in zip(X.columns, model.feature_importances_)})
    print(f'{bold_start}{" " * 20}AUC: {blu}{val_score:.5f}{res} {" " * 6}Best Iteration: {blu}{best_iter}{res}{bold_end}')

    xgb_oof.append(idx_pred_target)
    xgb_models.append(model)
    if val_score > 0.80:
        test_preds = model.predict_proba(test.drop(columns=['id']))[:, 1]
        predictions += test_preds
        counter += 1

predictions /= counter
mean_val_auc = get_mean_auc(np.array(xgb_oof))
print(f'{bold_start}{red}{"*" * 45}{res}')
print(f'Mean AUC: {red}{mean_val_auc:.5f}{res}{bold_end}')


# In[13]:


show_confusion_roc(xgb_oof)
f_imp_df = pd.DataFrame(f_imp).mean().reset_index()
f_imp_df.columns = ['feature', 'avg_imp']
f_importance_plot(f_imp_df)


# <div style="border-radius:10px; border:#4bce55 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#4bce55'>💡 Model Training: XGBoost with Stratified K-Fold Cross Validation</font></h3>
# 
# - **Overall Performance**: Your model has achieved an impressive average AUC of 0.87038. The AUC (Area Under the Curve) score ranges from 0.5 to 1, with a score closer to 1 indicating superior model performance. Your score of 0.87 showcases that the model is performing very well.
# 
# - **Consistency Across Folds**: The AUC scores across different folds range from 0.86598 to 0.87592. This consistency suggests that the model is robust and not overfitting to any specific fold of data.
# 
# - **Iterations Insights**: The 'Best Iteration' metric varies for each fold. This is common and indicates that the optimal number of boosting rounds is slightly different based on the specific subset of data in each fold.
# 
# - **Suggestions for Improvement**: While the current performance is commendable, there's always room for improvement. Consider exploring feature engineering, experimenting with different model architectures, or fine-tuning hyperparameters to potentially enhance the model's performance further.
# 
# - **Final Note**: Overall, the model demonstrates commendable performance. This provides a strong foundation for any subsequent real-world testing or further experimentation.
# 
