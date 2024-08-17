#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wget http://bit.ly/3ZLyF82 -O CSS.css -q')

from IPython.core.display import HTML
with open('./CSS.css', 'r') as file:
    custom_css = file.read()


HTML(custom_css)


# ## <p style = "font-weight:bold ; letter-spacing:2px ; color: ; font-size:100% ; text-align:center ; padding: 0px ; border-bottom : 5px solid #f0ad4e ; background-color: #CCCCCC; padding: 10px;">MACHINE FAILURE</p>
# ![](https://media.istockphoto.com/vectors/machine-failure-drawing-vector-id858191982)

# <div style = 'border : 3px solid brown; background-color:lightyellow;padding:10px'><h3>Details about the dataset:</h3>
# 
# <li><b>Product ID</b>: Product ID, which represents categorical data, is a key feature used to distinguish the type of product processed and consists of a letter Low (50%), medium (30%), High (20%) as product quality variants.</li>
# <li><b>Air temperature (K)</b>: Air temperature, which represents numerical data, refers to the temperature of the environment (between 2 K and 300 K after normalization).</li>
# <li><b>Process temperature (K)</b>: Process temperature, which represents numerical data, refers to the temperature of the production process.</li>
# <li><b>Rotational speed (rpm)</b>: Rotational speed, which represents numerical data, refers to the rotational speed of the main shaft.</li>
# <li><b>Torque (Nm)</b>: Torque represents a type of numerical data and is generally equal to 40 Nm where ε = 10 and no negative values.</li>
# <li><b>Tool wear (min)</b>: Tool wear, which represents numerical data, refers to the tool operation time.</li>
# The six equipment fault features of the data points are as follows:
# 
# <li><b>Tool wear failure (TWF)</b>: Tool wear failure causes a process failure.</li>
#     
# <li><b>Heat dissipation failure (HDF)</b>: Heat dissipation causes a process failure.</li>
# 
# <li><b>Power failure (PWF)</b>: Power failure causes a process failure.</li>
# 
# <li><b>Overstrain failure (OSF)</b>: OSF refers to the failure caused by overstrain in the production process.</li>
# 
# <li><b>Random failures (RNF)</b>: RNFs are failures whose cause cannot be determined. Their occurrence probability in the production process is 0.1%.</li>
# 
# <li><b>Machine failure</b>: The original two-category label (0 represents normal, and 1 represents failure)</li>
# 
# 
# </div> 

# <div id = 'h1'  class = 'alert alert-block alert-info' style="border-bottom: 5px solid #f0ad4e; background-color: #CCCCCC; padding: 10px;">
# <h2>😉👉 Importing Libraries:</h2>
# </div>

# In[2]:


pip install colorama


# In[3]:


import pandas as pd
import numpy as np
import math
import re
from pandas_profiling import ProfileReport
from sklearn.impute import KNNImputer

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import missingno as msno
from colorama import Fore, Style
from prettytable import PrettyTable ,ALL
from tabulate import tabulate
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

import warnings 
warnings.filterwarnings('ignore')


# In[4]:


df_train = pd.read_csv('/kaggle/input/playground-series-s3e17/train.csv',index_col='id')
df_orginal = pd.read_csv('/kaggle/input/machine-failure-predictions/machine failure.csv')
df_orginal.drop(['UDI'],axis=1,inplace=True)
df_test = pd.read_csv('/kaggle/input/playground-series-s3e17/test.csv',index_col='id')
df_submission = pd.read_csv('/kaggle/input/playground-series-s3e17/sample_submission.csv',index_col='id')


# In[5]:


get_ipython().run_cell_magic('time', '', '\ndef Print(text: str, color=Fore.MAGENTA, style=Style.BRIGHT):\n    print(style + color + text + Style.RESET_ALL)\n# original.drop([\'UDI\'] , axis = 1 , inplace = True)    \n    \ndef Preprocess(df, name):\n    # Define color styles for the table\n    if name == \'Train\':\n        styles = [\n            {\'selector\': \'th\', \'props\': [(\'border\', \'1px solid black\'),(\'background-color\', \'lightblue\'), (\'text-align\', \'center\')]},\n            {\'selector\': \'td\', \'props\': [(\'border\', \'1px solid black\'),(\'background-color\', \'lightyellow\'), (\'text-align\', \'center\')]},\n\n        ]\n        \n    else  :\n        styles = [\n            {\'selector\': \'th, td\', \'props\': [(\'border\', \'1px solid black\'), (\'text-align\', \'center\')]},\n            {\'selector\': \'th\', \'props\': [(\'background-color\', \'lightgreen\')]},\n        ]\n\n    ####################\n    #   First 5 Rows   #\n    ####################\n    Print(f\'\\n---------- Data Preprocessing ----------\\n\', Fore.RED)\n\n    Print(f\'\\n----------->{name}.head()\\n\')\n    display(df.head())\n    \n    ###################\n    #   Describing    #\n    ###################\n\n    Print(f\'\\n----------->{name} Description\\n\')\n    if name == \'Original\':\n        table = df.describe().T.style.bar(subset=[\'mean\'], color=\'orange\') \\\n                               .background_gradient(subset=[\'std\'], cmap=\'Reds\') \\\n                               .background_gradient(subset=[\'min\'], cmap=\'Blues\') \\\n                               .background_gradient(subset=[\'max\'], cmap=\'YlOrRd\') \\\n                               .set_table_styles(styles)\n                             \n        display(table)\n    else:\n        table = df.describe().T.style.bar(subset=[\'mean\'], color=\'green\') \\\n                               .set_table_styles(styles)\n        display(table)\n        \n    ##################\n    #   Information  #\n    ##################\n        \n    Print(f\'\\n---------->{name} Information\\n\')\n    info = df.info()     \n        \n   ######################\n   #   Null values      #\n   ######################\n\n    Print(f\'\\n----------> Sum of null values in {name}\\n\')\n    if name == \'Original\':    \n        null_counts = df.isnull().sum()\n\n        table = PrettyTable()\n        table.field_names = [\'Column\', \'Null Count\']\n        for col, null_count in null_counts.items():\n            table.add_row([col, null_count])\n\n        table.hrules = ALL  # Add lines between each line\n        print(table)\n        \n    else :\n        null_counts = df.isnull().sum()\n\n        table = []\n        for col, null_count in null_counts.items():\n            table.append([col, null_count])\n\n        headers = [\'Column\', \'Null Count\']\n        print(tabulate(table, headers, tablefmt=\'grid\'))\n        \n    ##################\n    #  Unique Values  #\n    ##################\n    Print(f\'\\n-----------> Sum of unique values in {name}\\n\')\n            \n    unique_counts = df.nunique().to_frame().rename(columns={0: \'Unique Value Count\'}).transpose()\n\n    # Display the unique value counts table\n    table = PrettyTable()\n    table.field_names = [\'Column\', \'Unique Value Count\']\n    for col, unique_count in unique_counts.items():\n        table.add_row([col, unique_count])\n\n    table.hrules = ALL  # Add lines between each line\n    print(table)\n        \n    ##################\n    #   Zero Values  #\n    ##################\n            \n    Print(f\'\\n-----------> Sum of zero values in {name}\\n\',Fore.RED)\n    if name == \'Original\':\n        \n        zeros = df.isin({0}).sum()\n\n        table = PrettyTable()\n        table.field_names = [\'Column\', \'Zero Count\']\n        for col, zero_count in zeros.items():\n            table.add_row([col, zero_count])\n\n        table.hrules = ALL  # Add lines between each line\n        print(table)\n        \n    else :\n        \n        df_zero_count = df.isin({0}).sum()\n        df_zero_count_vertical = df_zero_count.to_frame()\n\n        # Define border styles for the DataFrame\n        border_styles = [\n            {\'selector\': \'th, td\', \'props\': [(\'border\', \'1px solid black\'), (\'text-align\', \'center\')]},\n        ]\n\n        # Apply border styles to the DataFrame\n        styled_df = df_zero_count_vertical.style.set_table_styles(border_styles)\n\n        # Display the styled DataFrame\n        display(styled_df)\n        \n\nPrint(f"\\n{\'-\'*15}> Running Done...^-^ \\n")\n')


# In[6]:


get_ipython().run_cell_magic('time', '', "Preprocess(df_train , 'Train')\n")


# In[7]:


get_ipython().run_cell_magic('time', '', "Preprocess(df_orginal , 'Original')\n")


# In[8]:


print('Shape of Train dataset is :',df_train.shape)
print('Size of Train dataset is  :',df_train.size)


# In[9]:


print(f'This train dataset has {df_train.shape[0]} instances with the {df_train.shape[1]-1} features and 1 output variable')


# In[10]:


df_train.columns


# In[11]:


# Identify duplicate records
duplicates = df_train.duplicated()

# Count the number of duplicate records
count_duplicates = duplicates.sum()

# Print the count of duplicate records
print("Number of duplicate records:", count_duplicates)

# Drop duplicate records
df_train = df_train.drop_duplicates()


# <div style = 'border:3px solid brown;background-color:lightyellow;padding:10px'><h4> Analysis:</h4>
# 
# <li>From the above report we can see that there are no null values in the dataset</li> 
#     <li> There are duplicates records in the train data we need to remove them</li>
#     <li> The data is imbalanced such as there are 98% Non machine failures and only 2% machine failures are there.</li>
#    
# </div>

# <div id = 'h2'  class = 'alert alert-block alert-info' style="border-bottom: 5px solid #f0ad4e; background-color: #CCCCCC; padding: 10px;">
# <h2>😉👉 Exploratory Data Analysis(EDA):</h2>
# </div>

# In[12]:


target_col = 'Machine failure'

num_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
]

binary_cols = [
    'TWF',
    'HDF',
    'PWF',
    'OSF',
    'RNF'
]

cat_cols = 'Type'


# In[13]:


# Heirachical clustering

def plot_dendrogram(data, name):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=120)
    correlations = data.corr()
    converted_corr = 1 - np.abs(correlations)
    Z = linkage(squareform(converted_corr), 'complete')

    dn = dendrogram(Z, labels=data.columns, ax=ax, above_threshold_color='#FF092A', orientation='right')
    hierarchy.set_link_color_palette(None)
    plt.grid(axis='x')
    plt.title(f'{name} Hierarchical clustering, Dendrogram', fontsize=18, fontweight='bold',color='Green')
    plt.show()

plot_dendrogram(df_train[num_cols+binary_cols], name='Train')
plot_dendrogram(df_test[num_cols+binary_cols], name='Test')


# <div style = 'border:3px solid brown;background-color:lightyellow;padding:10px'><h4> Analysis:</h4>
# 
# <li>We used Dendrogram for finding the  relationships between numeric columns and binary columns based on their similarities or distances.</li>
#    
# </div>
# 

# In[14]:


# Visualize the datatypes
plt.rcParams.update({'font.size': 10})

df_train.dtypes.value_counts().plot.pie(explode=[0.1, 0.1,0.1],
                                       autopct='%1.2f%%',
                                       shadow=True)
plt.title('Data Type',
          color='Green',
          loc='center',
          font='Times New Roman');


# In[15]:


get_ipython().system('pip install missingno')


# In[16]:


# Visualize the Null values in train data

msno.bar(df_train)
plt.show()


# In[17]:


# Visual
import matplotlib.pyplot as plt
import seaborn as sns

def plot_type_distribution(train_data, original_data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot for train data
    ax1 = axes[0, 0]
    ax1 = sns.countplot(x='Type', data=train_data, ax=ax1)
    ax1.bar_label(ax1.containers[0])
    ax1.set_title("Type (Train Data)", fontsize=16)

    ax2 = axes[0, 1]
    ax2 = train_data['Type'].value_counts().plot.pie(explode=[0.1, 0.1, 0.1], autopct='%1.2f%%', shadow=True, ax=ax2)
    ax2.set_title("Distribution of Types (Train Data)", fontsize=16)

    # Plot for original data
    ax3 = axes[1, 0]
    ax3 = sns.countplot(x='Type', data=original_data, ax=ax3)
    ax3.bar_label(ax3.containers[0])
    ax3.set_title("Type (Original Data)", fontsize=16)

    ax4 = axes[1, 1]
    ax4 = original_data['Type'].value_counts().plot.pie(explode=[0.1, 0.1, 0.1], autopct='%1.2f%%', shadow=True, ax=ax4)
    ax4.set_title("Distribution of Types (Original Data)", fontsize=16)

    plt.tight_layout()
    plt.show()

# Example usage with df_train and df_original DataFrames
plot_type_distribution(df_train, df_orginal)


# <div style = 'border:3px solid brown;background-color:lightyellow;padding:10px'><h4> Analysis:</h4>
# 
# <li>From the above count plot we can clearly see there are more light machines than the other 2 </li>
#     <li> L-> Light </li>
#     <li> M -> Medium </li>
#     <li> H -> Heavy </li>
#    
# </div>

# In[18]:


#function to visualize the binary machine failures
def plot_binary_machine_failures(dataframe, column):
    # Plot the machine failures
    plt.figure(figsize=(10, 6))

    # Countplot for Machine failure
    ax = plt.subplot(1, 2, 1)
    ax = sns.countplot(x=column, data=dataframe)
    ax.bar_label(ax.containers[0])
    plt.title(column + " Failure", fontsize=20)

    # Pie chart for Outcome
    ax = plt.subplot(1, 2, 2)
    outcome_counts = dataframe[column].value_counts()
    ax = outcome_counts.plot.pie(explode=[0.1, 0.1], autopct='%1.2f%%', shadow=True)
    ax.set_title("Outcome", fontsize=20, color='Red', font='Lucida Calligraphy')

    # Display the plot
    plt.tight_layout()
    plt.show()


# In[19]:


# Visualize the machine failure
plot_binary_machine_failures(df_train, 'Machine failure')


# In[20]:


plot_binary_machine_failures(df_train, 'TWF')


# In[21]:


plot_binary_machine_failures(df_train, 'HDF')


# In[22]:


plot_binary_machine_failures(df_train, 'PWF')


# In[23]:


plot_binary_machine_failures(df_train, 'OSF')


# In[24]:


plot_binary_machine_failures(df_train, 'RNF')


# <div style = 'border:3px solid brown;background-color:lightyellow;padding:10px'><h4> Analysis:</h4>
# 
# <li>I will add this data(the records of Machine failure==1) to the train data  </li>
# 
# </div>

# In[25]:


# checking the count of total failures ==1 & Machine failure ==0 in Train data
TotalFt_1 = df_train[(df_train[binary_cols] == 1).any(axis=1) & (df_train[target_col] == 0)]
counts = len(TotalFt_1)
print(counts)


# In[26]:


# checking the count of total failures ==1 & Machine failure ==0 in orginal
TotalFOrg_1 = df_orginal[(df_orginal[binary_cols] == 1).any(axis=1) & (df_orginal[target_col] == 0)]
counts = len(TotalFOrg_1)
print(counts)


# In[27]:


df_orginal.loc[(df_orginal[binary_cols] == 1).any(axis=1) & (df_orginal[target_col] == 0), target_col] = 1


# In[28]:


# Filter the original dataframe based on Machine failure equals 1
df_machine_failures_org = df_orginal[df_orginal['Machine failure'] == 1]

# Print the separate dataframe
df_machine_failures_org


# In[29]:


merged_data = pd.concat([df_train, df_machine_failures_org], ignore_index=True)
merged_data


# In[30]:


# checking the count of total failures ==1 & Machine failure ==0 in merged data
TotalF_1 = merged_data[(merged_data[binary_cols] == 1).any(axis=1) & (merged_data[target_col] == 0)]
counts = len(TotalF_1)
print(counts)


# In[31]:


# Changing the improper Machine failures to the proper ones
merged_data.loc[(merged_data[binary_cols] == 1).any(axis=1) & (merged_data[target_col] == 0), target_col] = 1


# In[32]:


# Check the count of machine failures for each type
failure_counts = merged_data.loc[merged_data['Machine failure'] == 1, 'Type'].value_counts()

# Display the counts for each type
for typ, count in failure_counts.items():
    print(f"{typ} -> Count of machine failures: {count}")


# In[33]:


# Visualize the count of machine failures for each type
failure_counts = merged_data.loc[merged_data['Machine failure'] == 1, 'Type'].value_counts()

# Get the unique types and their counts
types = failure_counts.index
counts = failure_counts.values

# Create a colormap
cmap = plt.get_cmap('rainbow')
colors = cmap(np.linspace(0, 1, len(types)))

# Plot the counts for each type
plt.figure(figsize=(8, 6))
bars = plt.bar(types, counts, color=colors)

# Add count labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom')

plt.xlabel('Type')
plt.ylabel('Count of Machine Failures')
plt.title('Machine Failures by Type')
plt.show()


# In[34]:


# Function to visualize the numerical distributions
def plot_numerical_distributions(dataframe, columns_to_plot, num_rows, num_cols):
    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 10))

    # Flatten the axes array
    axes = axes.flatten()

    # distribution plots for each feature with respect to 'Type'
    for i, column in enumerate(columns_to_plot):
        sns.histplot(data=dataframe, x=column, hue='Type', kde=True, multiple='stack', ax=axes[i])
        axes[i].set_title(f"Distribution Plot: {column}")
        axes[i].set_xlabel(column)
        axes[i].legend(title='Type', labels=['L', 'M', 'H'])

    # Remove any unused subplots
    if len(columns_to_plot) < num_rows * num_cols:
        for j in range(len(columns_to_plot), num_rows * num_cols):
            fig.delaxes(axes[j])

    # Adjust the spacing between subplots
    fig.suptitle("Distribution Plots for Numerical Features", fontsize=24, fontweight='bold', y=1.10)
    fig.tight_layout()

    # Display the subplots
    plt.show()


# In[35]:


columns_to_plot = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
num_rows = 2
num_cols = 3

plot_numerical_distributions(df_train, columns_to_plot, num_rows, num_cols)


# <div style = 'border:3px solid brown;background-color:lightyellow;padding:10px'><h4> Handling Data:</h4>
# 
# <li> Tool wear</li>
# 
# </div>

# In[36]:


# Function to check the count of tool wear has a value 0 wrt to machine failures
def count_tool_wear_zeros(merged_data):
    Toolwear_df_0 = merged_data[(merged_data['Machine failure'] == 0) & (merged_data['Tool wear [min]'] == 0)]
    count_0 = len(Toolwear_df_0)
    print("Number of 0 in Tool wear with Machine failure = 0:", count_0)

    Toolwear_df_1 = merged_data[(merged_data['Machine failure'] == 1) & (merged_data['Tool wear [min]'] == 0)]
    count_1 = len(Toolwear_df_1)
    print("Number of 0 in Tool wear with Machine failure = 1:", count_1)
    
# call the function
print("--------Merged Data----------")
count_tool_wear_zeros(merged_data)
print('*'*50)
# here just i am checking the count of the Tool wear with the orginal data
print("--------Orginal Data----------")
count_tool_wear_zeros(df_orginal)


# In[37]:


# Create a scatter plot with hue
sns.scatterplot(data=merged_data, x='Tool wear [min]',y='Air temperature [K]', hue='Machine failure')

plt.xlabel('Tool wear [min]')
plt.ylabel('Air temperature [K]')
plt.title('Scatter Plot: Tool wear [min] vs. Air temperature [K]')
plt.show()


# In[38]:


# Using Knn imputer to change the Tool wear Zero(0) Values based on Machine failure == 1
merged_data.loc[merged_data['Machine failure']==1,'Tool wear [min]']=merged_data.loc[merged_data['Machine failure']==1,'Tool wear [min]'].replace(0,np.nan)

mask_1= merged_data['Machine failure'] == 1

 # Perform KNN imputation on the Tool wear column for the selected rows
imputer = KNNImputer(n_neighbors=5)  # You can adjust the number of neighbors as needed

merged_data.loc[mask_1,'Tool wear [min]'] = imputer.fit_transform(merged_data.loc[mask_1,'Tool wear [min]'].values.reshape(-1,1))

 # Print the updated DataFrame
merged_data.head()


# In[39]:


# # Filter rows with Machine failure == 0 and Tool wear [min] == 0
# mask = (merged_data['Machine failure'] == 0) & (merged_data['Tool wear [min]'] == 0)
# rows_to_fill = merged_data[mask]

# # Iterate over the rows to fill the 0 values
# for index, row in rows_to_fill.iterrows():
#     # Compare with other features and update the Tool wear [min] value accordingly
#     if row['Type'] == 'L' and row['Air temperature [K]'] > 300:
#         merged_data.at[index, 'Tool wear [min]'] = 100
#     elif row['Type'] == 'M' and row['Air temperature [K]'] < 300:
#         merged_data.at[index, 'Tool wear [min]'] = 50
#     else:
#         merged_data.at[index, 'Tool wear [min]'] = 75

# # Print the count after filling the values
# count_filled = len(merged_data[(merged_data['Machine failure'] == 0) & (merged_data['Tool wear [min]'] == 0)])
# print("Number of 0 in Tool wear with Machine failure = 0 (after filling):", count_filled)


# <div style = 'border:3px solid brown;background-color:lightyellow;padding:10px'><h4> Analysis:</h4>
# 
# <li>If the machine failure happens then the tool wear value not be 0 but when it comes to non machine failure the tool wear may be 0 so i will try to impute those values with KNN </li>
#     <li> I have checked with the orginal data there </li>
#    </div>

# <div style = 'border:3px solid brown;background-color:lightyellow;padding:10px'><h4> Handling Data:</h4>
# 
# <li> TWF,HDF,PWF,OSF,RNF</li>
# 
# </div>

# In[40]:


reason_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Filter the data for machine failures
failures_data = merged_data[merged_data['Machine failure'] == 1]

# Count the occurrences of each reason for machine failures
reason_counts = failures_data[reason_columns].sum()

# Create a colormap
cmap = plt.get_cmap('rainbow')
colors = cmap(np.linspace(0, 1, len(reason_columns)))

# Plot the reason counts
plt.figure(figsize=(8, 6))
bars = plt.bar(reason_columns, reason_counts, color=colors)

# Add count labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom')

plt.xlabel('Reasons for Machine Failures')
plt.ylabel('Count')
plt.title('Reasons for Machine Failures (Machine failure = 1)')
plt.show()


# In[41]:


df_1=merged_data[merged_data['Machine failure']==1]
df_1


# In[42]:


features_index=list(df_1.index)
columns=list(df_1.iloc[:,-5:].columns)
# df_1.loc[features_index,columns]=failures_df_1.loc[features_index,columns]
merged_data.loc[features_index,columns]=df_1.loc[features_index,columns]


# In[43]:


failures_df_1 = df_1[(df_1['TWF']==0) & (df_1['HDF']==0) & (df_1['PWF']==0) & (df_1['OSF']==0) & (df_1['RNF']==0) ]
for x in failures_df_1.iloc[:,-5:].columns:
    failures_df_1[x].replace(0,np.nan,inplace=True)


# In[44]:


failures_df_1


# In[45]:


features_index=list(failures_df_1.index)
columns=list(failures_df_1.iloc[:,-5:].columns)
df_1.loc[features_index,columns]=failures_df_1.loc[features_index,columns]
# merged_data.loc[features_index,columns]=df_1.loc[features_index,columns]


# In[46]:


# Create a KNNImputer object
imputer = KNNImputer(n_neighbors=10)  # Specify the number of neighbors to consider

# Impute NaN values in the DataFrame
df_imputed = imputer.fit_transform(df_1.iloc[:,2:])

# Convert the imputed array back to a DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=df_1.iloc[:,2:].columns)
df_imputed


# In[47]:


# Specify the failure columns
failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Apply the condition and replace values in the failure columns
df_imputed[failure_columns] = df_imputed[failure_columns].applymap(lambda x: 0 if x < 0.1 else 1)
df_imputed


# In[48]:


features_index=list(df_imputed.index)
columns=list(df_imputed.iloc[:,-5:].columns)
# df_1.loc[features_index,columns]=failures_df_1.loc[features_index,columns]
merged_data.loc[features_index,columns]=df_imputed.loc[features_index,columns]


# In[49]:


merged_data


# <div style = 'border:3px solid brown;background-color:lightyellow;padding:10px'><h4> Analysis:</h4>
# 
# <li>The merged dataset contains a total of 2797 instances of machine failures. These failures are categorized into types such as TWF, HDF, PWF, OSF, and RNF. Normally, when a machine failure occurs, the corresponding type will have a value of 1. However, it is puzzling that out of the 2797 machine failures, 512 instances do not have any type specified (i.e., they have a value of 0). To address this issue and make the data more meaningful, I imputed these 0 values using a KNN imputer.</li> 
#    
# </div>

# In[50]:


columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
           'Tool wear [min]', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
# Calculate correlation matrix
corr_matrix = merged_data[columns].corr()

# Plot correlation matrix
plt.figure(figsize=(18,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)

plt.title('Correlation Matrix')
plt.show()


# <div style = 'border:3px solid brown;background-color:lightyellow;padding:10px'><h4> Analysis: Correlation Matrix</h4>
# 
# <li>Air temperature and process temperature have a strong positive correlation of 0.856. This indicates that as the air temperature increases, the process temperature tends to increase as well.</li>
#     <li>Rotational speed and torque exhibit a significant negative correlation of -0.779. This suggests that higher rotational speed is associated with lower torque values.</li>
#     <li>Machine failure and HDF (Hydraulic component failures) have a relatively high correlation of 0.355. This indicates that there is some relationship between machine failures and hydraulic component failures.</li> 
#    
# </div>

# <div id = 'h3'  class = 'alert alert-block alert-info' style="border-bottom: 5px solid #f0ad4e; background-color: #CCCCCC; padding: 10px;">
# <h2>😉👉 Outliers Detection:</h2>
# </div>

# In[51]:


# Select the numerical columns to plot
numerical_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# Create boxplots for each numerical column
plt.figure(figsize=(12, 8))
for i, column in enumerate(numerical_columns):
    plt.subplot(2, 3, i+1)
    sns.boxplot(data=merged_data[column])
    plt.title(f'Boxplot: {column}')
    plt.xlabel(column)
plt.tight_layout()
plt.show()


# <div id = 'h4'  class = 'alert alert-block alert-info' style="border-bottom: 5px solid #f0ad4e; background-color: #CCCCCC; padding: 10px;">
# <h2>😉👉 Feature Engineering:</h2>
# </div>

# In[52]:


# Create new features by comparing other ones
def New_Features(df):
    df['TemperatureDifference'] = df['Process temperature [K]'] - df['Air temperature [K]']
    
    df['TemperatureRatio'] = df['Process temperature [K]'] / df['Air temperature [K]']
    
    df['TemperatureVariability'] = df[['Air temperature [K]', 'Process temperature [K]']].std(axis=1)
    
    df['TemperatureChangeRate'] = df['TemperatureDifference'] / df['Tool wear [min]']
    
    df['TemperatureChangeRate'] = np.where(df['TemperatureChangeRate']== float('inf'),1, df['TemperatureChangeRate'])
    
    df['Power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
    
    df['ToolWearRate'] = df['Tool wear [min]'] / df['Tool wear [min]'].max()
    
    df['TF'] = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum(axis=1)
    
    
    return df


# <div style = 'border:3px solid brown;background-color:lightyellow;padding:10px'><h4> Why we created new features :</h4>
# 
# <p>The function <b>"New_Features"</b> introduces new derived features to the dataset, such as 'TemperatureDifference' and 'TemperatureRatio', which capture the temperature relationship between the process and air temperature. This additional information may provide valuable insights for the machine learning model to make more accurate predictions.</p>
#     <p>The function also calculates the 'Power' feature by multiplying the 'Torque [Nm]' and 'Rotational speed [rpm]', which can potentially capture the combined effect of these two variables on the machine's performance. By including this feature, the model may benefit from a more comprehensive representation of the relationship between torque, rotational speed, and power consumption.</p>
# </div> 

# In[53]:


# Add the new features to the train and test data , call the function
merged_data = New_Features(merged_data)
df_test = New_Features(df_test)


# In[54]:


merged_data


# In[55]:


df_test


# In[56]:


from sklearn.preprocessing import RobustScaler

def apply_robust_scaler(data, columns):
    scaler = RobustScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data


scaler_column = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]","TWF", "HDF", "PWF", "OSF", "RNF", "Power","TemperatureDifference", "TemperatureVariability", "TemperatureRatio", "ToolWearRate", "TemperatureChangeRate", "TF"]

merged_data = apply_robust_scaler(merged_data, scaler_column)
test = apply_robust_scaler(df_test, scaler_column)


# <div style = 'border:3px solid brown;background-color:lightyellow;padding:10px'><h4> Why we use Robust Scaler :</h4>
# 
# >RobustScaler is applied to standardize numerical features by removing the median and scaling the data according to the interquartile range (IQR). This scaling technique is robust to outliers and is particularly useful when the data contains extreme values or non-normal distributions. It helps to bring the features to a similar scale, making them more comparable and improving the performance of certain machine learning algorithms.
# </div> 
# 

# In[57]:


# Define a function to remove special characters from column names

def remove_special_characters(column_name):
    # Remove non-alphanumeric characters
    pattern = r'[^A-Za-z0-9_]+'
    return re.sub(pattern, '', column_name)

# Rename columns of the train DataFrame
train = merged_data.rename(columns=lambda x: remove_special_characters(x))

# Rename columns of the test DataFrame
test = test.rename(columns=lambda x: remove_special_characters(x))


# In[58]:


feature_names = train.drop('Machinefailure', axis = 1).columns.tolist()


# <div id = 'h5'  class = 'alert alert-block alert-info' style="border-bottom: 5px solid #f0ad4e; background-color: #CCCCCC; padding: 10px;">
# <h2>😉👉 Model Building:</h2>
# </div>

# <div style = 'border:3px solid brown;background-color:lightyellow;padding:10px'><h4>CatBoost:</h4>
#     <p>CatBoost is a gradient boosting algorithm specifically designed to handle categorical variables in machine learning tasks. It employs an innovative approach called ordered boosting, which incorporates the natural ordering of categorical features to improve accuracy. CatBoost automatically handles missing values and incorporates advanced techniques like target encoding and feature combinations. It provides fast and scalable training, and its powerful handling of categorical data makes it a popular choice for classification tasks.</p>

# In[59]:


# Split the data into features (X) and target variable (y)
X = train.drop(columns=['Machinefailure'])
y = train['Machinefailure']

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=13)

# Create the Pool objects with categorical features
pool_train = Pool(X_train, y_train, feature_names=feature_names, cat_features=['ProductID', 'Type'])
pool_valid = Pool(X_valid, y_valid, feature_names=feature_names, cat_features=['ProductID', 'Type'])

# Set the best parameters obtained from Optuna
cb_parames = {
    'n_estimators': 3261,
    'max_depth': 5,
    'learning_rate': 0.025184406348245668,
    'max_bin': 306,
    'random_strength': 0.23,
    'grow_policy': 'Lossguide',
    'bootstrap_type': 'Bayesian',
    'objective':'Logloss',
    "loss_function": "AUC",
    'eval_metric': "AUC",
    'l2_leaf_reg':  0.06609180403841047,
    'min_child_samples': 140,
    'random_state': 13,
    'silent': True,
    'task_type': 'GPU',
}

# Build the model with the best parameters
model = CatBoostClassifier(**cb_parames)

# Fit the model to the training data
model.fit(pool_train)

# Make predictions on the validation data
y_pred = model.predict(pool_valid)


# <div style = 'border:3px solid brown;background-color:lightyellow;padding:10px'><h4>Fine tuning parameters Values:</h4>
#     <p>Optuna is a hyperparameter optimization framework in Python. It provides a flexible and efficient framework for automating the parameter optimization process and improving the performance of machine learning models.</p>

# In[60]:


# Calculate and print the classification report
print("Classification Report:")
print(classification_report(y_valid, y_pred))

# Calculate and print the confusion matrix
cm = confusion_matrix(y_valid, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Calculate the probabilities for each class
y_prob = model.predict_proba(pool_valid)[:, 1]


# In[61]:


# Calculate the area under the ROC curve
roc_auc = roc_auc_score(y_valid, y_prob)
print("Area under ROC curve:", roc_auc)

# Compute the false positive rate and true positive rate
fpr, tpr, thresholds = roc_curve(y_valid, y_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[62]:


# Get the feature importance
feature_importance = model.get_feature_importance(pool_train)

# Create a dataframe with feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

# Sort the dataframe by importance scores in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)  # Sort in ascending order

# Create a vertical bar plot of feature importance
plt.figure(figsize=(8, 10))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()


# In[63]:


# Create a dataframe of actual and predicted values
df_results = pd.DataFrame({'Actual': y_valid, 'Predicted': y_pred})
# Print the dataframe
df_results.T


# In[64]:


# predict on the test data 
y_predict_catboost = model.predict(test)


# In[65]:


# Make a submission 
df_submission['Machine failure']=y_predict_catboost
df_submission


# In[66]:


# save it into csv
df_submission.to_csv('submission_catboost(F).csv')


# <div style = 'border:3px solid brown;background-color:lightyellow;padding:10px'><h4> Conclusion:</h4>
# 
# <li>Previous score was 0.97774, and after performing in-depth feature handling, specifically considering types of failures, the new score improved to 0.97832.</li>
#     <li> I will try using pipelines and various classifiers tomorrow.</li>
#                   <h2>If you appreciate my work, please consider upvoting it 👍</h2>
# </div>

# In[ ]:




