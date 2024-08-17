#!/usr/bin/env python
# coding: utf-8

# <p style="background-color:#007FFF;font-family:'Open Sans'; color:#FFFFFF;font-size:300%;text-align:center;border-radius:20px 20px;">Improving a Model the Data Centric Way! PS3E22</p>
# 
# <img src="https://www.simplilearn.com/ice9/free_resources_article_thumb/Data_Visualization_Tools.jpg">
# 
# In this project, I will be implementing Feature selection, engineering and data preprocessing on the Synthetic playground Season 3 Episode 21 dataset.
# 
#    <a id='top'></a>
# <div class="list-group" id="list-tab" role="tablist">
# <p style="background-color:#007FFF;font-family:Open Sans;color:#FFF9ED;font-size:150%;text-align:center;border-radius:10px 10px;">TABLE OF CONTENTS</p>   
#     
# * [1. IMPORTING LIBRARIES & INITIALISING EVALUATION MODEL](#1)
#     
# * [2. LOADING DATA](#2)
#     
# * [3. DATASET OVERVIEW](#3)
#     
# * [4. HANDLING NA VALUES](#4)   
#     
# * [5. PREPROCESSING (SCALING & OUTLIER REMOVAL)](#5) 
#       
# * [6. PCA & SCALING](#6)
#     
# * [7. GENERATING NEW FEATURES](#7)
#     
# * [8. PERFORMANCE COMPARISONS](#8)
#    
# * [9. THE END](#9)
# 

# <a id="1"></a>
# # <p style="background-color:#007FFF;font-family:newtimeroman;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px 10px;">||||| IMPORTING LIBRARIES & INITIALISING EVALUATION MODEL|||||</p>

# All the neccesary libraries for executing the code are imported in this section!

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from colorama import Fore, Style, init
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import sklearn as sk
from sklearn.decomposition import PCA


# We will initialize the model based on the competition's specified model parameters. Our objective is to optimize the data to achieve the best possible performance with the fixed model. We cannot make any adjustments to the choice of the model itself or its parameters, as they are predetermined.

# In[2]:


rf = RandomForestRegressor(
       n_estimators=1000,
       max_depth=7,
       n_jobs=-1,
       random_state=42)


# <a id="2"></a>
# # <p style="background-color:#007FFF;font-family:newtimeroman;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px 10px;">||||| LOADING DATA |||||</p>

# Loading the data into a pandas dataframe!

# In[3]:


df = pd.read_csv("/kaggle/input/playground-series-s3e21/sample_submission.csv", index_col='id')
df_copy=df
df.head()


# In[4]:


X = df.drop('target', axis=1)
y = df['target']


# Lets first see how the model performs on the default data to get a better understanding of how our adjustments to the data are affecting model performance.

# In[5]:


print(cross_val_score(rf, X, y, cv = 5, scoring = 'neg_root_mean_squared_error'))
rf.fit(X, y)


# In[6]:


feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nsmallest(35).plot(kind='barh')

#From least to increasing usefulness
X['NH4_7'].value = 0


# Data consists of 3500 rows and 36 columns, out of which one column is our target column.

# In[7]:


df_copy.shape


# In[8]:


sns.scatterplot(data = df, x='O2_1', y = 'O2_2')


# <a id="3"></a>
# # <p style="background-color:#007FFF;font-family:newtimeroman;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px 10px;">||||| DATASET OVERVIEW |||||</p>

# In[9]:


df.shape


# Having a look at the various statistics of the features present in the data!

# In[10]:


df.describe().T.style.background_gradient(cmap="summer")


# Clearly the data will require scaling as some features like NO2_1 have a maximum value of 0.95 while other features consist of extremely high values like BOD5_5 which has a maximum value of 82.45

# In[11]:


def PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):
    print(style + color + text + Style.RESET_ALL)


# Checking for NA values in the dataset!

# In[12]:


df.isna().sum()


# No Na values present hence no requirement for imputation.

# In[13]:


df.duplicated().sum()


# No duplicates present in the data either.

# In[14]:


PrintColor(f"\nData Information", color = Fore.BLUE)
display(df.info())


# All data is of float64 type hence no requiremnt for data type conversions.

# <a id="4"></a>
# # <p style="background-color:#007FFF;font-family:newtimeroman;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px 10px;">||||| PREPROCESSING (SCALING AND OUTLIER REMOVAL) |||||</p>

# Removing Outliers from the data using a customised Inter Quantile Range Method, all values below the lower bound and above the upper bound are thus eliminated.

# In[15]:


def remove_outliers_iqr(df):
    Q1 = df.quantile(0.1)
    Q3 = df.quantile(0.9)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df >= lower_bound) & (df <= upper_bound)].dropna()
# Remove outliers from the specified column
df_no_outliers = remove_outliers_iqr(df)


# In[16]:


df = df_no_outliers


# O2_1 and O2_2 before outlier removal, notice certain data points which are clealry outliers, there is a whole cluster of these outliers which have an O2_1 value of more than 40. the noise in our data is obscuring the trend which the data is trying to indicate.

# In[17]:


sns.set_style("whitegrid")
sns.lmplot(data = df_copy, x = "O2_1", y="target", scatter_kws={"color": '#007FFF'}, line_kws = {"color": 'red'})


# O2_2 suffers from a similar case of outliers, only worse since we have a data point which has scored more than 60 on the O2_2 feature.

# In[18]:


sns.set_style("whitegrid")
sns.lmplot(data = df_copy, x = "O2_2", y="target", scatter_kws={"color": '#007FFF'}, line_kws = {"color": 'red'})


# After outlier removal, notice all the O2_1 values above 14 have been eliminated thus making the trend in our data quite clear and highlighting the positive correaltion between O2_1 and our Target variable.

# In[19]:


plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
sns.lmplot(data = df, x = "O2_1", y="target", scatter_kws={"color": '#81cdc6'}, line_kws = {"color": 'red'})


# The strong positive correlation between 'O2_2' and the target variable has become evident and is now clearly visible. Eliminating these outliers from the dataset also enhances the model's performance, as it no longer needs to adapt or overfit to these exceptional data points during the training process.

# In[20]:


plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
sns.lmplot(data = df, x = "O2_2", y="target", scatter_kws={"color": '#81cdc6'}, line_kws = {"color": 'red'})


# In[21]:


y = df['target']
X = df.drop('target', axis=1)


# Scaling the data standardizes the range of values for all features, ensuring that they fall within a consistent scale. This standardization plays a pivotal role in enhancing model accuracy, as models often tend to assign greater significance to features with larger values. Scaling mitigates this issue, preventing the model from unfairly emphasizing features solely based on their magnitude, thus promoting more accurate and equitable feature evaluation.

# In[22]:


Scaler = MinMaxScaler()
numerical_scaled = pd.DataFrame(Scaler.fit_transform(X), columns=X.columns)
numerical_scaled


# In[23]:


df_copy.head()


# The correlation heatmap helps us in easily recognising variables which have a heavy correlation and can be useful in prediction, classification and forecasting.

# In[24]:


plt.figure(figsize=(50,30))
sns.heatmap(df_copy.corr(), cmap='coolwarm', annot=True, annot_kws={"size": 12, "weight": "bold"})
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()


# Chcking which feature has the best correlation with the overall dataset.

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

corr_features = correlation(df, 0.8)
corr_features


# In[26]:


X.shape


# Now lets calculate which feature is most useful in predicting our target variable

# In[27]:


# Calculate the correlation matrix
correlation_matrix = df_copy.corr()

# Extract the correlations of the target variable with other variables
correlations_with_target = correlation_matrix['target']

# Display the correlations with the target variable
print(correlations_with_target)


# O2_1, O2_2, O2_6, NH4_5 seem to have higher correlation when compared to other features

# In[28]:


print(cross_val_score(rf, X, y, cv = 5, scoring = 'neg_root_mean_squared_error'))
rf.fit(X, y)


# Our model error has decreased by a pretty good amount!

# In[29]:


feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nsmallest(35).plot(kind='barh')

#From least to increasing usefulness
X['NH4_7'].value = 0


# Clearly O2_1 is the feature the model is assinging most importance to

# Here is the list of the top 10 variables with highest correlation to the target.

# In[30]:


sorted_corr = feat_importances.sort_values(ascending=False)

# Get the top n rows from the sorted DataFrame
top_n_rows = sorted_corr.head(10)
top_n_rows


# <a id="5"></a>
# # <p style="background-color:#007FFF;font-family:newtimeroman;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px 10px;">||||| PCA & SCALING |||||</p>

# In[31]:


top_n_rows1 = pd.DataFrame(top_n_rows)


# In[32]:


top_n_rows1.reset_index(names = "column_name", inplace=True)


# In[33]:


top_n_rows1['column_name'].unique()


# Creating 5 PCA components out of the top 10 features with the highest correlation to the target variable!

# In[34]:


n_components = 5
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)


# In[35]:


X_pca


# In[36]:


X_pca = pd.DataFrame(X_pca)


# In[37]:


X_pca


# Fitting and testing the new PCA data to see how it affects our model error.

# In[38]:


print(cross_val_score(rf, X_pca, y, cv = 5, scoring = 'neg_mean_squared_error'))
rf.fit(X_pca, y)


# Seems like PCA hurt our model pretty badly. You can expirement with different values of the n_components parameter and try to find a value which brings better results.

# Scaling the PCA generated features to check if it improves the performance.

# In[39]:


Scaler = MinMaxScaler()
X_scaled = pd.DataFrame(Scaler.fit_transform(X_pca), columns=X_pca.columns)
X_scaled


# In[40]:


print(cross_val_score(rf, X_scaled, y, cv = 5, scoring = 'neg_mean_squared_error'))
rf.fit(X_scaled, y)


# Clearly this isnt helping our model so lets try a different approach.

# In[41]:


sk.metrics.get_scorer_names()


# <a id="6"></a>
# # <p style="background-color:#007FFF;font-family:newtimeroman;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px 10px;">||||| GENERATING NEW FEATURES |||||</p>

# In[42]:


X.columns


# Creating new features by applying all four arithmetic functions on the top 5 features and only keeping the top 10.

# In[43]:


def create_and_select_arithmetic_features(dataset, target_column, num_features_to_keep=10):
    # Copy the original dataset to avoid modifying the original data
    dataset_copy = dataset.copy()
    
    # List of columns on which you want to perform arithmetic operations
    columns_to_operate_on = ['O2_1', 'O2_2', 'O2_4', 'BOD5_6', 'NO3_3']  # Replace with your column names
    
    # Initialize a list to store the newly created feature DataFrames
    new_feature_dfs = []
    
    # Create new features by applying arithmetic operations
    for col1 in columns_to_operate_on:
        for col2 in columns_to_operate_on:
            if col1 != col2:  # Avoid applying operations on the same column
                # Addition
                addition_df = dataset_copy[col1] + dataset_copy[col2]
                addition_df.rename(f'{col1}_plus_{col2}', inplace=True)
                new_feature_dfs.append(addition_df)
                
                # Subtraction
                subtraction_df = dataset_copy[col1] - dataset_copy[col2]
                subtraction_df.rename(f'{col1}_minus_{col2}', inplace=True)
                new_feature_dfs.append(subtraction_df)
                
                # Multiplication
                multiplication_df = dataset_copy[col1] * dataset_copy[col2]
                multiplication_df.rename(f'{col1}_times_{col2}', inplace=True)
                new_feature_dfs.append(multiplication_df)
                
                # Division (avoid division by zero)
                division_df = dataset_copy[col1] / (dataset_copy[col2] + 1e-5)
                division_df.rename(f'{col1}_divided_by_{col2}', inplace=True)
                new_feature_dfs.append(division_df)
    
    # Concatenate all the new feature DataFrames
    new_features = pd.concat(new_feature_dfs, axis=1)
    
    # Combine the new features with the target variable
    selected_dataset = pd.concat([new_features, dataset_copy[target_column]], axis=1)
    
    # Split the dataset into features (X) and the target variable (y)
    X = selected_dataset.drop(columns=[target_column])
    y = selected_dataset[target_column]
    
    # Use a RandomForestClassifier to get feature importances
    rf.fit(X, y)
    
    # Get feature importances
    feature_importances = rf.feature_importances_
    
    # Sort feature importances and get the indices of the top N features
    top_features_indices = feature_importances.argsort()[-num_features_to_keep:][::-1]
    
    # Select only the top N features
    selected_features = [X.columns[i] for i in top_features_indices]
    
    # Create a new dataset with only the selected features
    X_selected = X[selected_features]
    
    # Add the target variable back to the selected dataset
    selected_dataset = pd.concat([X_selected, y], axis=1)
    
    return selected_dataset


# In[44]:


df_1 = create_and_select_arithmetic_features(df, 'target')


# In[45]:


df_1.head()


# In[46]:


df_1.shape


# In[47]:


df_1 = pd.DataFrame(df_1)


# In[48]:


X1 = df_1.drop('target', axis=1)
y1 = df_1['target']


# Testing our model on these new custom created features.

# In[49]:


print(cross_val_score(rf, X1, y1, cv = 5, scoring = 'neg_mean_squared_error'))
rf.fit(X1, y1)


# a decent improvement over our previous best result!

# In[50]:


feat_importances = pd.Series(rf.feature_importances_, index=X1.columns)
feat_importances.nsmallest(35).plot(kind='barh')


# Here i noticed that some variables are merely repeated, O2_1_times_O2_2 and O2_2_times_O2_1 are the same variable and have the same values, since order of operands does not matter in addition or multiplication opeations. Hence these features should be eliminated. I will do so ahead.

# Scaling the new features to improve model performance further.

# In[51]:


Scaler = MinMaxScaler()
X_scaled = pd.DataFrame(Scaler.fit_transform(X1), columns=X1.columns)
X_scaled


# In[52]:


print(cross_val_score(rf, X_scaled, y1, cv = 5, scoring = 'neg_mean_squared_error'))
rf.fit(X_scaled, y1)


# Negligable improvement in model error.

# <a id="7"></a>
# # <p style="background-color:#007FFF;font-family:newtimeroman;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px 10px;">||||| FEATURE COMPARISON |||||</p>

# Now i will be creating a new dataframe where i will combine the old and new custom features and then train the model on the new dataframe to see if it improves the model performance.

# In[53]:


X_scaled.shape


# In[54]:


X.shape


# In[55]:


X=pd.DataFrame(X)
X_scaled = pd.DataFrame(X_scaled)


# In[56]:


X.shape


# In[57]:


X = X.reset_index(drop=True)


# In[58]:


X = X.rename_axis('ID')


# In[59]:


X.head()


# In[60]:


X_scaled = X_scaled.rename_axis('ID')


# In[61]:


X_scaled.head()


# In[62]:


X_scaled.isna().sum()


# In[63]:


X = X.merge(X_scaled, on='ID', how='left')


# In[64]:


X.shape


# In[65]:


X.isna().sum()


# Finally our new dataframe is ready!

# In[66]:


print(cross_val_score(rf, X, y, cv = 5, scoring = 'neg_mean_squared_error'))
rf.fit(X, y)


# Nice improvement in our error values, clearly our model is working better on the combined data.

# In[67]:


feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nsmallest(35).plot(kind='barh')


# All models are getting a decent amount of importance assigned to them by the model resulting in a more balanced appraoch to predicting the target variable. However some might think that the new features werent very useful.

# In[68]:


y = y.reset_index(drop=True)
y = y.rename_axis('ID')
y.head()


# In[69]:


y = pd.DataFrame(y)


# In[70]:


y.head()


# In[71]:


df_11 = X.merge(y, on='ID', how='left')


# In[72]:


df_11


# Creating a correlation heatmap to see how our new features comapare to the default ones at predicting our target variable!

# In[73]:


plt.figure(figsize=(50,30))
sns.heatmap(df_11.corr(), cmap='coolwarm', annot=True, annot_kws={"size": 12, "weight": "bold"})
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()


# In[74]:


correlation_matrix_11 = df_11.corr()

# Extract the correlations of the target variable with other variables
correlations_with_target_11 = correlation_matrix_11['target']

# Display the correlations with the target variable
print(correlations_with_target_11)


# In[75]:


correlation_matrix_11


# Here i drop the repititve new features which consisted of the same values, as i had promised before.

# In[76]:


index_to_drop = ['O2_1_times_O2_2', 'O2_1_plus_O2_2', 'O2_1_minus_O2_4', 'NO3_3_plus_O2_1']
correlations_with_target_11 = correlations_with_target_11.drop(index_to_drop)


# Now to get the top 10 features with highest correlation we will take the absolute values since otherwise features with a very negative correlation wont make it to the list despite the fact that they can be very useful to our model.

# In[77]:


top_10_corr_vars = correlations_with_target_11.abs().sort_values(ascending=False).head(11)


# In[78]:


top_10_corr_vars


# These are the top 11 features with highest correlation to our target variable, the reason ive take 11 features is because obviously we cant count the target variable which has a correlaton of 1, we will drop the target variable from this list soon.

# In[79]:


top_10_corr_vars = pd.DataFrame(top_10_corr_vars)


# In[80]:


top_10_corr_vars


# In[81]:


index_to_drop = 'target'
top_10_corr_vars = top_10_corr_vars.drop(index_to_drop)


# Now we have dropped the target variable and converted this data into a dataframe.

# In[82]:


top_10_corr_vars


# Out of the top 10 best correlating features to our target variable, 5 are new features which we custom generated thus proving that they are quite useful!

# Since we now have the top 10 features lets give the negative correlating features their orginal values again.

# In[83]:


row_index = 'O2_4_minus_O2_1'
column_name = 'target'
new_value = -0.500574

top_10_corr_vars.at[row_index, column_name] = new_value


# In[84]:


top_10_corr_vars


# In[85]:


row_index = 'O2_2_divided_by_O2_1'
column_name = 'target'
new_value = -0.246726

top_10_corr_vars.at[row_index, column_name] = new_value


# In[86]:


top_10_corr_vars.head()


# In[87]:


top_10_corr_vars


# Creating a heatmap to visualise the importance of the top 10 features!

# In[88]:


plt.figure(figsize=(30,30))
heatmap = sns.heatmap(top_10_corr_vars[['target']].sort_values(by='target', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG', annot_kws={'fontsize': 18})
heatmap.set_xticklabels(heatmap.get_xticklabels(), size=20)
heatmap.set_yticklabels(heatmap.get_yticklabels(), size=20)
plt.yticks(rotation=45)
for text in heatmap.texts:
    text.set_weight('bold')
heatmap.set_title('Features Correlating with Target Variable', fontdict={'fontsize':26}, pad=16);


# In[89]:


tcv = top_10_corr_vars


# In[90]:


tcv.index


# In[91]:


top_10_df = df_11[['O2_2_times_O2_1', 'O2_1', 'O2_2_plus_O2_1', 'O2_4_minus_O2_1', 'O2_2',
       'O2_2_divided_by_O2_1', 'O2_1_plus_NO3_3', 'O2_1_divided_by_O2_2',
       'O2_6', 'O2_3']]


# In[92]:


top_10_df.head()


# Lets visualise more clearly what these correaltions indicate and how the trend visualises on a scatter plot or lmplot.

# In[93]:


plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
for col in top_10_df:    # Create an lmplot for the current column
    sns.lmplot(data=df_11, x=col, y="target", scatter_kws={"color": '#81cdc6'}, line_kws={"color": 'red'})    
plt.tight_layout()
plt.show()


# Retrieving our top 10 features to train our model and test whether it significatnly impacts our model error.

# In[94]:


top_10_df = df_11[['O2_2_times_O2_1', 'O2_1', 'O2_2_plus_O2_1', 'O2_4_minus_O2_1', 'O2_2',
       'O2_2_divided_by_O2_1', 'O2_1_plus_NO3_3', 'O2_1_divided_by_O2_2',
       'O2_6', 'O2_3', 'target']]


# In[95]:


Xtop = top_10_df.drop('target', axis=1)
ytop = top_10_df['target']


# In[96]:


Xtop.shape


# In[97]:


print(cross_val_score(rf, Xtop , ytop, cv = 5, scoring = 'neg_mean_squared_error'))
rf.fit(Xtop, ytop)
feat_importances = pd.Series(rf.feature_importances_, index=Xtop.columns)
feat_importances.nsmallest(35).plot(kind='barh')


# Good improvement in our model errors!

# <a id="9"></a>
# # <p style="background-color:#007FFF;font-family:newtimeroman;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px 10px;">||||| PERFORMANCE COMPARISONS |||||</p>

# * The Error of the Original Dataset:[-2.61056522 -1.01930614 -1.15758718 -1.0183318  -1.19130388]
# 
# * The Error after Outlier removal and Scaling: [-0.86609707 -0.92933693 -0.93719733 -0.90125726 -0.88352335]
# 
# * The Error after PCA and Scaling: [-1.48985677 -1.50225027 -1.62627489 -1.51329632 -1.34094797]
# 
# * The Error after Creation of New Features and Scaling: [-0.75320029 -0.89398753 -0.93544193 -0.83154944 -0.78563043]
# 
# * **The Error after Adding New Features to Old Features: [-0.73833044 -0.8590105  -0.86274661 -0.81257984 -0.77009859]**
# 
# * The Error after using only Top 10 Features: [-0.73336648 -0.88913631 -0.89773962 -0.83270577 -0.76347312]

# I have highlighted the best approach in **Bold**!

# In[98]:


#def create_submission(dataset):
    #submitted_dataset = dataset.to_csv('submission.csv', index = False)
    
#create_submission(removed_lows)


# <a id="10"></a>
# # <p style="background-color:#007FFF;font-family:newtimeroman;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px 10px;">||||| THE END |||||</p>
