#!/usr/bin/env python
# coding: utf-8

# # Import necessary libraries
# 

# In[1]:


import numpy as np  # NumPy for numerical operations
import pandas as pd  # Pandas for data manipulation

pd.set_option("display.max_columns", None)  # Display all columns in Pandas DataFrames

# Import statistical and machine learning libraries
from scipy import stats  # SciPy for scientific and statistical functions

# Import data visualization libraries
import matplotlib.pyplot as plt  # Matplotlib for basic plotting
import seaborn as sns  # Seaborn for enhanced data visualization
import plotly.express as px  # Plotly Express for interactive visualizations

# Import preprocessing tools
from sklearn.preprocessing import LabelEncoder  # LabelEncoder for encoding categorical variables

# Model Selection
from sklearn.model_selection import KFold

# Import machine learning models
from sklearn.experimental import enable_hist_gradient_boosting # Enable Histogram-Based Gradient Boosting
from sklearn.ensemble import (
    HistGradientBoostingClassifier,  # Histogram-Based Gradient Boosting Classifier
    GradientBoostingClassifier,  # Gradient Boosting Classifier
    AdaBoostClassifier,  # AdaBoost Classifier
    RandomForestClassifier,  # Random Forest Classifier
    ExtraTreesClassifier,  # Extra Trees Classifier
    VotingClassifier,  # Ensemble Voting Classifier
    StackingClassifier,  # Stacking Classifier
)
from xgboost import XGBClassifier  # XGBoost Classifier
from lightgbm import LGBMClassifier # lightgbm Classifier

# Import evaluation metrics
from sklearn.metrics import f1_score  # F1-score for model evaluation

# Suppress warnings to improve code readability
import warnings

warnings.filterwarnings("ignore")


# # Custom style settings

# In[2]:


# Define a dictionary `rc` with custom style settings for plots
rc = {
    "axes.facecolor": "#F8F8F8",  # Background color for plot axes
    "figure.facecolor": "#F8F8F8",  # Background color for the entire figure
    "axes.edgecolor": "#000000",  # Color of the edges of plot axes
    "grid.color": "#EBEBE7" + "30",  # Color of grid lines with transparency
    "font.family": "serif",  # Font family used for text in the plot
    "axes.labelcolor": "#000000",  # Color of axis labels
    "xtick.color": "#000000",  # Color of x-axis tick marks
    "ytick.color": "#000000",  # Color of y-axis tick marks
    "grid.alpha": 0.4  # Transparency level for grid lines
}

# Set the style of seaborn plots using the custom style dictionary
sns.set(rc=rc)

# Define a custom color palette with a list of color codes
palette = ['#302c36', '#037d97', '#E4591E', '#C09741',
           '#EC5B6D', '#90A6B1', '#6ca957', '#D8E3E2']

# Import color styling libraries and define some text styles
from colorama import Style, Fore
blk = Style.BRIGHT + Fore.BLACK  # Bright black text
mgt = Style.BRIGHT + Fore.MAGENTA  # Bright magenta text
red = Style.BRIGHT + Fore.RED  # Bright red text
blu = Style.BRIGHT + Fore.BLUE  # Bright blue text
res = Style.RESET_ALL  # Reset text style to default


# # Loading and Preparing Data

# In[3]:


train = pd.read_csv('/kaggle/input/playground-series-s3e22/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e22/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s3e22/sample_submission.csv')
origin = pd.read_csv('/kaggle/input/horse-survival-dataset/horse.csv')

# Add a new column 'is_generated' to the 'train' DataFrame and set all values to 1
train["is_generated"] = 1

# Add a new column 'is_generated' to the 'test' DataFrame and set all values to 1
test["is_generated"] = 1

# Add a new column 'is_generated' to the 'origin' DataFrame and set all values to 0
origin["is_generated"] = 0

# Drop the 'id' column from the 'train' DataFrame
train.drop('id', axis=1, inplace=True)

# Drop the 'id' column from the 'test' DataFrame
test.drop('id', axis=1, inplace=True)

# Concatenate the 'train' and 'origin' DataFrames along rows, ignoring index, and store the result in 'train_total'
train_total = pd.concat([train, origin], ignore_index=True)

# Remove duplicate rows from the 'train_total' DataFrame, if any
train_total.drop_duplicates(inplace=True)

# Concatenate the 'train_total' and 'test' DataFrames along rows, ignoring index, and store the result in 'total'
total = pd.concat([train_total, test], ignore_index=True)

# Print the shapes of the three DataFrames: 'train', 'test', and 'total'
print('The shape of the train data:', train.shape)
print('The shape of the test data:', test.shape)
print('The shape of the total data:', total.shape)


# In[4]:


train.head()


# # Exploratory Data Analysis (EDA)
# 

# In[5]:


# Create a list 'num_var' that contains column names from 'train' where the number of unique values is greater than 10
num_var = [column for column in train.columns if train[column].nunique() > 10]

# Create a list 'bin_var' that contains column names from 'train' where the number of unique values is exactly 2 (binary variables)
bin_var = [column for column in train.columns if train[column].nunique() == 2]

# Create a list 'cat_var' that contains specific categorical column names from 'train'
cat_var = ['temp_of_extremities', 'peripheral_pulse', 'mucous_membrane', 'capillary_refill_time', 'pain',
           'peristalsis', 'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 'rectal_exam_feces',
           'abdomen', 'abdomo_appearance', 'lesion_2', 'surgery', 'age', 'surgical_lesion', 'lesion_3', 'cp_data']

# Define the target variable, which is 'outcome'
target = 'outcome'


# In[6]:


# Calculate descriptive statistics for the 'train' DataFrame, transpose the result, and apply styling
train.describe().T\
    .style.bar(subset=['mean'], color=px.colors.qualitative.G10[2])\
    .background_gradient(subset=['std'], cmap='Blues')\
    .background_gradient(subset=['50%'], cmap='BuGn')


# In[7]:


# Define a custom function 'summary' that takes a DataFrame 'df' as input
def summary(df):
    # Create a new DataFrame 'sum' to store summary information
    sum = pd.DataFrame(df.dtypes, columns=['dtypes'])  # Column 'dtypes' stores data types of columns
    sum['missing#'] = df.isna().sum()  # Column 'missing#' stores the count of missing values for each column
    sum['missing%'] = (df.isna().sum()) / len(df)  # Column 'missing%' stores the percentage of missing values
    sum['uniques'] = df.nunique().values  # Column 'uniques' stores the count of unique values for each column
    sum['count'] = df.count().values  # Column 'count' stores the count of non-missing values for each column
    # Note: The 'skew' column is commented out and not used in this version of the function

    return sum  # Return the 'sum' DataFrame containing the summary information

# Call the 'summary' function on the 'train_total' DataFrame and apply a background gradient using the 'Blues' colormap
summary(train_total).style.background_gradient(cmap='Blues')


# In[8]:


summary(train).style.background_gradient(cmap='Blues')


# # Countplots for Categorical Variables by Outcome

# In[9]:


columns_cat = [column for column in train.columns if train[column].nunique() < 10]

def plot_count(df,columns,n_cols,hue):
    
    
    n_rows = (len(columns) - 1) // n_cols + 1
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(17, 4 * n_rows))
    ax = ax.flatten()
    
    for i, column in enumerate(columns):
        sns.countplot(data=df, x=column, ax=ax[i],hue=hue)

        # Titles
        ax[i].set_title(f'{column} Counts', fontsize=18)
        ax[i].set_xlabel(None, fontsize=16)
        ax[i].set_ylabel(None, fontsize=16)
        ax[i].tick_params(axis='x', rotation=10)

        for p in ax[i].patches:
            value = int(p.get_height())
            ax[i].annotate(f'{value:.0f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                           ha='center', va='bottom', fontsize=9)

    ylim_top = ax[i].get_ylim()[1]
    ax[i].set_ylim(top=ylim_top * 1.1)
    for i in range(len(columns), len(ax)):
        ax[i].axis('off')

    # fig.suptitle(plotname, fontsize=25, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
plot_count(train,columns_cat,3,'outcome')


# # Countplots for Categorical Variables in Test Data

# In[10]:


columns_cat = [column for column in train.columns if train[column].nunique() < 10 and column != target]
plot_count(test,columns_cat,3,None)


# **Noteworthy Observations:**
# 
# 1. It is evident that the `age Counts` column predominantly comprises adults, indicating an imbalance.
# 
# 2. The `peripheral_pulse_Counts` feature exhibits a notable scarcity of instances classified as `absent` and `increased`.
# 
# 3. Notably, the `lesion_2` and `lesion_3` variables seem to lack meaningful information or relevance.
# 
# Furthermore, it's worth mentioning that certain feature labels present in the training dataset do not find a counterpart in the test dataset. To address this issue, I propose merging the two datasets.
# 

# **Now, let's turn our attention to the numerical features.**

# # Scatter Matrix with Target

# In[11]:


def plot_pair(df_train,num_var,target,plotname):
    
    g = sns.pairplot(data=df_train, x_vars=num_var, y_vars=num_var, hue=target, corner=True)
    g._legend.set_bbox_to_anchor((0.8, 0.7))
    g._legend.set_title(target)
    g._legend.loc = 'upper center'
    g._legend.get_title().set_fontsize(14)
    for item in g._legend.get_texts():
        item.set_fontsize(14)

    plt.suptitle(plotname, ha='center', fontweight='bold', fontsize=25, y=0.98)
    plt.show()

plot_pair(train,num_var,target,plotname = 'Scatter Matrix with Target')


# # Train vs. Test Data - Numerical Variables Comparison

# In[12]:


df = pd.concat([train[num_var].assign(Source = 'Train'), 
                test[num_var].assign(Source = 'Test')], 
               axis=0, ignore_index = True);

fig, axes = plt.subplots(len(num_var), 3 ,figsize = (16, len(num_var) * 4.2), 
                         gridspec_kw = {'hspace': 0.35, 'wspace': 0.3, 'width_ratios': [0.80, 0.20, 0.20]});

for i,col in enumerate(num_var):
    ax = axes[i,0];
    sns.kdeplot(data = df[[col, 'Source']], x = col, hue = 'Source', ax = ax, linewidth = 2.1)
    ax.set_title(f"\n{col}",fontsize = 9, fontweight= 'bold');
    ax.grid(visible=True, which = 'both', linestyle = '--', color='lightgrey', linewidth = 0.75);
    ax.set(xlabel = '', ylabel = '');
    ax = axes[i,1];
    sns.boxplot(data = df.loc[df.Source == 'Train', [col]], y = col, width = 0.25,saturation = 0.90, linewidth = 0.90, fliersize= 2.25, color = '#037d97',
                ax = ax);
    ax.set(xlabel = '', ylabel = '');
    ax.set_title(f"Train",fontsize = 9, fontweight= 'bold');

    ax = axes[i,2];
    sns.boxplot(data = df.loc[df.Source == 'Test', [col]], y = col, width = 0.25, fliersize= 2.25,
                saturation = 0.6, linewidth = 0.90, color = '#E4591E',
                ax = ax); 
    ax.set(xlabel = '', ylabel = '');
    ax.set_title(f"Test",fontsize = 9, fontweight= 'bold');

plt.tight_layout();
plt.show();


# # Now, let's look at the distribution of numerical features in the training set.

# In[13]:


plt.figure(figsize=(14, len(num_var) * 2.5))

for idx, column in enumerate(num_var):
    plt.subplot(len(num_var), 2, idx*2+1)
    sns.histplot(x=column, hue="outcome", data=train, bins=30, kde=True)
    plt.title(f"{column} Distribution for outcome")
    plt.ylim(0, train[column].value_counts().max() + 10)
    
plt.tight_layout()
plt.show()


# # Correlation Matrix

# In[14]:


corr_matrix = train[num_var].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='Blues', fmt='.2f', linewidths=1, square=True, annot_kws={"size": 9} )
plt.title('Correlation Matrix', fontsize=15)
plt.show()


# # Preprocessing and Feature Selection

# In[15]:


def chi_squared_test(df, input_var, target_var, significance_level=0.05):
    contingency_table = pd.crosstab(df[input_var], df[target_var])
    chi2, p, _, _ = stats.chi2_contingency(contingency_table)
    
    if p < significance_level:
        print(f'\033[32m{input_var} has a significant relationship with the target variable.\033[0m') 
    else:
        print(f'\033[31m{input_var} does not have a significant relationship with the target variable.\033[0m')  

for i in cat_var:
    chi_squared_test(train, i, target)


# # Mapping Target Labels to Numerical Values

# In[16]:


total[target] = total[target].map({'died':0,'euthanized':1,'lived':2})


# # Label Encoding for binary columns  & One-Hot Encoding for category columns

# In[17]:


def preprocessing(df, le_cols, ohe_cols):
    
    # Label Encoding for binary cols
    le = LabelEncoder()    
    for col in le_cols:
        df[col] = le.fit_transform(df[col])
    
    # OneHot Encoding for category cols
    df = pd.get_dummies(df, columns = ohe_cols)
    
    df["pain"] = df["pain"].replace('slight', 'moderate')
    df["peristalsis"] = df["peristalsis"].replace('distend_small', 'normal')
    df["rectal_exam_feces"] = df["rectal_exam_feces"].replace('serosanguious', 'absent')
    df["nasogastric_reflux"] = df["nasogastric_reflux"].replace('slight', 'none')
        
    df["temp_of_extremities"] = df["temp_of_extremities"].fillna("normal").map({'cold': 0, 'cool': 1, 'normal': 2, 'warm': 3})
    df["peripheral_pulse"] = df["peripheral_pulse"].fillna("normal").map({'absent': 0, 'reduced': 1, 'normal': 2, 'increased': 3})
    df["capillary_refill_time"] = df["capillary_refill_time"].fillna("3").map({'less_3_sec': 0, '3': 1, 'more_3_sec': 2})
    df["pain"] = df["pain"].fillna("depressed").map({'alert': 0, 'depressed': 1, 'moderate': 2, 'mild_pain': 3, 'severe_pain': 4, 'extreme_pain': 5})
    df["peristalsis"] = df["peristalsis"].fillna("hypomotile").map({'hypermotile': 0, 'normal': 1, 'hypomotile': 2, 'absent': 3})
    df["abdominal_distention"] = df["abdominal_distention"].fillna("none").map({'none': 0, 'slight': 1, 'moderate': 2, 'severe': 3})
    df["nasogastric_tube"] = df["nasogastric_tube"].fillna("none").map({'none': 0, 'slight': 1, 'significant': 2})
    df["nasogastric_reflux"] = df["nasogastric_reflux"].fillna("none").map({'less_1_liter': 0, 'none': 1, 'more_1_liter': 2})
    df["rectal_exam_feces"] = df["rectal_exam_feces"].fillna("absent").map({'absent': 0, 'decreased': 1, 'normal': 2, 'increased': 3})
    df["abdomen"] = df["abdomen"].fillna("distend_small").map({'normal': 0, 'other': 1, 'firm': 2,'distend_small': 3, 'distend_large': 4})
    df["abdomo_appearance"] = df["abdomo_appearance"].fillna("serosanguious").map({'clear': 0, 'cloudy': 1, 'serosanguious': 2})

    return df    


# In[18]:


total = preprocessing(total, le_cols = ["surgery", "age", "surgical_lesion", "cp_data"], ohe_cols = ["mucous_membrane"])


# # Feature Engineering and Missing Value Imputation
# 

# In[19]:


def features_engineering(df):
    
    data_preprocessed = df.copy()
    
    # Imputer 
    cols_with_nan = df.drop(target,axis=1).columns[df.drop(target,axis=1).isna().any()].tolist()

    for feature in cols_with_nan:
        data_preprocessed[feature].fillna(data_preprocessed[feature].mode()[0], inplace=True)
    
    return data_preprocessed

total = features_engineering(total)


# In[20]:


df_train = total[total[target].notna()]
df_test = total[total[target].isna()]
df_test.drop(target,axis=1,inplace=True)


# In[21]:


full_features = df_test.columns.tolist()
bin_features = df_test.select_dtypes('bool').columns

df_train[bin_features] = df_train[bin_features].astype('int64')
df_test[bin_features] = df_test[bin_features].astype('int64')


# In[22]:


df_train.head()


# # F1 Score Calculation Function

# In[23]:


def caculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average = 'micro')


# # Baseline Model and Feature Selection Evaluation

# In[24]:


lgbm_baseline = LGBMClassifier(n_estimators=80,
                     max_depth=4,
                     random_state=42)

f1_results = pd.DataFrame(columns=['Selected_Features', 'F1'])

def evaluation(df, select_features, note):
    global f1_results
    
    X = df[select_features]
    Y = df[target]
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]
        
        lgbm_baseline.fit(X_train, y_train)
        y_hat = lgbm_baseline.predict(X_test) 
        f1 = caculate_f1(y_test, y_hat)
        f1_scores.append(f1)
    
    average_f1 = np.mean(f1_scores)
    new_row = {'Selected_Features': note, 'F1': average_f1}
    f1_results = pd.concat([f1_results, pd.DataFrame([new_row])], ignore_index=True)

    print('====================================')
    print(note)
    print("Average F1:", average_f1)
    print('====================================')
    return average_f1
evaluation(df=df_train,select_features=full_features,note='Baseline')


# # Feature Selection: Addressing Multicollinearity with Correlation

# In[25]:


def correlation(dataset, threshold):
    col_corr = set()  
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) >= threshold: 
                colname = corr_matrix.columns[i]                  
                col_corr.add(colname)
    return col_corr      

corr_features = correlation(df_train, 0.35)
corr_features


# In[26]:


corr_features = df_test.drop(['abdominal_distention',
 'abdomo_protein',
 'capillary_refill_time',
 'cp_data',
 'lesion_3',
 'mucous_membrane_dark_cyanotic',
 'mucous_membrane_normal_pink',
 'packed_cell_volume',
 'peripheral_pulse',
 'peristalsis',
 'rectal_exam_feces',
 'respiratory_rate',
 'surgical_lesion',
 'temp_of_extremities',
 'total_protein'],axis=1).columns.tolist()


# In[27]:


evaluation(df=df_train,select_features=corr_features,note='Corr Features')


# # Features Importance

# In[28]:


def f_importance_plot(f_imp):
    fig = plt.figure(figsize=(12, 0.20*len(f_imp)))
    plt.title(f'Feature importances', size=16, y=1.05, 
              fontweight='bold')
    a = sns.barplot(data=f_imp, x='imp', y='feature', linestyle="-", 
                    linewidth=0.5, edgecolor="black",palette='GnBu')
    plt.xlabel('')
    plt.xticks([])
    plt.ylabel('')
    plt.yticks(size=11)
    
    for j in ['right', 'top', 'bottom']:
        a.spines[j].set_visible(False)
    for j in ['left']:
        a.spines[j].set_linewidth(0.5)
    plt.tight_layout()
    plt.show()


# In[29]:


clf = LGBMClassifier(n_estimators=1000,
                     max_depth=10,
                     random_state=42)
clf.fit(df_train.drop(target,axis=1), df_train[target])

f_imp_df = pd.DataFrame({'feature': df_train.drop(target,axis=1).columns, 'imp': clf.feature_importances_})
f_imp_df.sort_values(by='imp',ascending=False,inplace=True)
f_importance_plot(f_imp_df)


# In[30]:


best_feature_num = 30
best_score = 0.7392406127690802
print(f'Best feature number is Top {best_feature_num}, Best score is {best_score}')


# # Based on these results, it seems that the top 30 features deliver the best performance in cross-validation.

# In[31]:


best_features = f_imp_df.head(best_feature_num).feature.to_list()


# # Selecting Best Features for Modeling

# In[32]:


X = df_train[best_features]
y = df_train[target]

test_df = df_test[best_features]

X


# # Model Development
# 
# Now, let's delve into the development of predictive models. We will explore various algorithms and techniques to build and evaluate models for our dataset.
# 
# **Note: Training on the entire dataset and testing on the training data may lead to overfitting.**
# **High training F1 score (e.g., 100%) should be interpreted with caution, as it may indicate overfitting.**
# 

# **HistGradientBoostingClassifier**

# In[33]:


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score


hist_model = HistGradientBoostingClassifier(
    max_depth=4,           # Adjust the maximum depth of each tree
    max_iter=80,          # Adjust the number of boosting iterations
    learning_rate=0.1,     # Adjust the learning rate
    random_state=42,   
    scoring='f1_micro',          
    max_leaf_nodes = 21,
    l2_regularization = 0.1,
)

hist_model.fit(X, y)
print(f"HistGradientBoosting Model: F1 Score (Micro-Average) = {f1_score(y, hist_model.predict(X), average='micro') * 100:.2f}%")


# **When modeling with the `HistGradientBoostingClassifier`, it achieved a leaderboard score of 0.84756.**

# In[34]:


submission = hist_model.predict(test_df)
sample_submission['outcome'] = submission
sample_submission


# In[35]:


# Create a mapping dictionary
outcome_mapping = {0.0: 'died', 1.0: 'euthanized', 2.0: 'lived'}

# Map the values in the "outcome" column using the dictionary
sample_submission['outcome'] = sample_submission['outcome'].map(outcome_mapping)
sample_submission


# In[36]:


sample_submission.to_csv('submission.csv', index=False)


# # If you found this notebook helpful and insightful, please consider giving it an upvote to show your appreciation. Your support is greatly valued! 😊
# 

# In[ ]:





# In[ ]:




