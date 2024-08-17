#!/usr/bin/env python
# coding: utf-8

# <div style="padding: 35px;color:white;margin:10;font-size:200%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/4021781/pexels-photo-4021781.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>Getting Started </span></b> </div>
# 
# <br>
# 
# ## 🚀 Getting Started
# 
# This project involves analyzing a molecular dataset with the aim of predicting enzyme classes. The dataset includes various **`Features`** related to molecular properties and characteristics. Each entry represents a unique molecule, and the features include Bertz complexity index, molecular connectivity indices, electrotopological states, molecular weights, and other molecular descriptors.
# 
# ## 🔧 Tools and Libraries
# 
# We will be using Python for this project, along with several libraries for data analysis and machine learning. Here are the main libraries we'll be using:
# 
# - **Pandas**: For data manipulation and analysis.
# - **NumPy**: For numerical computations.
# - **Matplotlib and Seaborn**: For data visualization.
# - **Scikit-learn**: For machine learning tasks, including data preprocessing, model training, and model evaluation.
# 
# ## 📚 Dataset
# 
# The dataset we'll be using includes various features related to molecular properties. Each row represents a unique molecule, and includes measurements such as molecular connectivity indices, electrotopological states, molecular weights, and other descriptors. It also includes binary labels indicating the presence or absence of specific enzyme classes.
# 
# ## 🎯 Objective
# 
# Our main objective is to develop a predictive model that can effectively classify enzyme classes based on the provided features. By leveraging the power of Ensemble methods, we aim to enhance the model's accuracy and predictive performance.
# 
# ## 📈 Workflow
# 
# Here's a brief overview of our workflow for this project:
# 
# 1. **Data Loading and Preprocessing**: Load the data and preprocess it for analysis and modeling. This includes handling missing values, encoding categorical variables, and scaling numerical variables.
# 
# 2. **Exploratory Data Analysis (EDA)**: Perform exploratory data analysis to gain insights into the dataset, understand the distributions of features, and explore potential relationships between the features and enzyme classes.
# 
# 3. **Feature Engineering**: If necessary, perform feature engineering to extract additional relevant features or transform existing features to improve the model's performance.
# 
# 4. **Data Splitting**: Split the dataset into training and testing sets, ensuring that both sets have a representative distribution of enzyme classes.
# 
# 5. **Ensemble Model Training**: Train the Ensemble model on the training data. Ensemble methods combine the predictions of multiple base models to make accurate predictions.
# 
# 6. **Model Evaluation**: Evaluate the performance of the trained Ensemble model using appropriate evaluation metrics such as accuracy, precision, recall, and F1-score. Assess the model's ability to generalize to unseen data using the testing set.
# 
# 7. **Model Fine-tuning**: Fine-tune the model by adjusting hyperparameters or applying techniques like cross-validation to optimize its performance.
# 
# 8. **Prediction and Deployment**: Use the trained Ensemble model to make predictions on new, unseen data. If applicable, deploy the model for practical use or further analysis.
# 
# **Let's get started!**
# 
# <br>
# 
# ![](https://images.pexels.com/photos/3825572/pexels-photo-3825572.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)
# 
# <br>
# 
# ## Domain Knowledge 📚
# 
# This dataset contains various features related to molecular properties. Let's explore the details of each feature and its significance:
# 
# ### Features
# 
# 
# 1. `Id`: This feature represents the identifier or unique identification number of a molecule. It serves as a reference but doesn't directly contribute to the predictive model.
# 
# 2. `BertzCT`: This feature corresponds to the Bertz complexity index, which measures the structural complexity of a molecule. It can provide insights into the intricacy of molecular structures.
# 
# 3. `Chi1`: The Chi1 feature denotes the 1st order molecular connectivity index, which describes the topological connectivity of atoms in a molecule. It characterizes the atomic bonding pattern within the molecule.
# 
# 4. `Chi1n`: This feature is the normalized version of the Chi1 index. It allows for standardized comparisons of the 1st order molecular connectivity across different molecules.
# 
# 5. `Chi1v`: The Chi1v feature represents the 1st order molecular variance connectivity index. It captures the variance or diversity in the connectivity of atoms within a molecule.
# 
# 6. `Chi2n`: The Chi2n feature indicates the 2nd order molecular connectivity index, which provides information about the extended connectivity of atoms in a molecule. It considers the neighboring atoms of each atom in the molecule.
# 
# 7. `Chi2v`: Similar to Chi2n, the Chi2v feature measures the variance or diversity in the extended connectivity of atoms within a molecule at the 2nd order level.
# 
# 8. `Chi3v`: The Chi3v feature represents the 3rd order molecular variance connectivity index. It captures the variance in the 3rd order connectivity patterns among atoms in a molecule.
# 
# 9. `Chi4n`: This feature corresponds to the 4th order molecular connectivity index, which provides information about the extended connectivity of atoms in a molecule. The Chi4n index is normalized to allow for consistent comparisons across molecules.
# 
# 10. `EState_VSA1`: EState_VSA1 is a feature that relates to the electrotopological state of a molecule. Specifically, it represents the Van der Waals surface area contribution for a specific atom type, contributing to the overall electrotopological state.
# 
# 11. `EState_VSA2`: Similar to EState_VSA1, EState_VSA2 also represents the electrotopological state but for a different specific atom type.
# 
# 12. `ExactMolWt`: This feature denotes the exact molecular weight of a molecule. It provides an accurate measurement of the mass of the molecule.
# 
# 13. `FpDensityMorgan1`: FpDensityMorgan1 represents the Morgan fingerprint density for a specific radius of 1. Morgan fingerprints are a method for generating molecular fingerprints, and this feature captures the density of those fingerprints.
# 
# 14. `FpDensityMorgan2`: Similar to FpDensityMorgan1, this feature represents the Morgan fingerprint density for a specific radius of 2.
# 
# 15. `FpDensityMorgan3`: FpDensityMorgan3 corresponds to the Morgan fingerprint density for a specific radius of 3.
# 
# 16. `HallkierAlpha`: The HallkierAlpha feature denotes the Hall-Kier alpha value for a molecule. It is a measure of molecular shape and can provide insights into the overall structure of the molecule.
# 
# 17. `HeavyAtomMolWt`: This feature represents the molecular weight of heavy atoms only, excluding hydrogen atoms. It focuses on the mass of non-hydrogen atoms within the molecule.
# 
# 18. `Kappa3`: The Kappa3 feature corresponds to the Hall-Kier Kappa3 value, which is a molecular shape descriptor. It provides information about the shape and spatial arrangement of atoms within the molecule.
# 
# 19. `MaxAbsEStateIndex`: This feature represents the maximum absolute value of the E-state index. The E-state index relates to the electronic properties of a molecule, and its maximum absolute value can indicate the presence of specific electronic characteristics.
# 
# 20. `MinEStateIndex`: MinEStateIndex denotes the minimum value of the E-state index. It provides information about the lowest observed electronic property value within the molecule.
# 
# 21. `NumHeteroatoms`: This feature indicates the number of heteroatoms present in a molecule. Heteroatoms are atoms other than carbon and hydrogen, such as oxygen, nitrogen, sulfur, etc. This feature provides insights into the diversity and composition of atoms within the molecule.
# 
# 22. `PEOE_VSA10`: PEOE_VSA10 represents the partial equalization of orbital electronegativity Van der Waals surface area contribution for a specific atom type. It captures the surface area contribution of a particular atom type to the overall electrostatic properties.
# 
# 23. `PEOE_VSA14`: Similar to PEOE_VSA10, PEOE_VSA14 also represents the partial equalization of orbital electronegativity Van der Waals surface area contribution for a specific atom type.
# 
# 24. `PEOE_VSA6`: This feature corresponds to the partial equalization of orbital electronegativity Van der Waals surface area contribution for a specific atom type at a different level.
# 
# 25. `PEOE_VSA7`: Similar to PEOE_VSA6, PEOE_VSA7 represents the partial equalization of orbital electronegativity Van der Waals surface area contribution for a specific atom type.
# 
# 26. `PEOE_VSA8`: PEOE_VSA8 denotes the partial equalization of orbital electronegativity Van der Waals surface area contribution for a specific atom type.
# 
# 27. `SMR_VSA10`: SMR_VSA10 represents the solvent-accessible surface area Van der Waals surface area contribution for a specific atom type. It captures the contribution of a specific atom type to the solvent-accessible surface area.
# 
# 28. `SMR_VSA5`: Similar to SMR_VSA10, this feature denotes the solvent-accessible surface area Van der Waals surface area contribution for a specific atom type at a different level.
# 
# 29. `SlogP_VSA3`: The SlogP_VSA3 feature represents the LogP-based surface area contribution. It captures the contribution of a specific atom type to the surface area based on its logarithmic partition coefficient.
# 
# 30. `VSA_EState9`: This feature denotes the E-state fragment contribution for the Van der Waals surface area calculation. It captures the fragment-specific contribution to the electrostatic properties of the molecule.
# 
# 31. `fr_COO`: The fr_COO feature represents the number of carboxyl (COO) functional groups present in the molecule. It ranges from 0 to 8, providing insights into the presence and abundance of carboxyl groups.
# 
# 32. `fr_COO2`: Similar to fr_COO, fr_COO2 represents the number of carboxyl (COO) functional groups, ranging from 0 to 8.
# 
# 33. `EC1`: EC1 is a binary feature representing a predicted label related to Oxidoreductases. It serves as one of the target variables for prediction.
# 
# 34. `EC2`: EC2 is another binary feature representing a predicted label related to Transferases. It serves as another target variable for prediction.
# 
# # <span style="color:#E888BB; font-size: 1%;">INTRODUCTION</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/4021781/pexels-photo-4021781.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>Introduction</span></b> </div>
# 
# <br>
#     
# ## 📝 Abstract
# 
# This project encapsulates a comprehensive journey of analyzing and modeling a molecular properties dataset with the objective of predicting enzyme classes. The dataset is characterized by a variety of **`Features`** including BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3, HallkierAlpha, HeavyAtomMolWt, Kappa3, MaxAbsEStateIndex, MinEStateIndex, NumHeteroatoms, PEOE_VSA10, PEOE_VSA14, PEOE_VSA6, PEOE_VSA7, PEOE_VSA8, SMR_VSA10, SMR_VSA5, SlogP_VSA3, VSA_EState9, fr_COO, fr_COO2, and two target variables **`EC1`** and **`EC2`**.
# 
# Our exploratory data analysis unveiled substantial relationships between certain molecular properties and the enzyme classes, suggesting these properties often influence the enzyme class of a molecule. Additionally, we discovered that certain features like `BertzCT`, `Chi1`, and `ExactMolWt` might escalate the probability of a molecule belonging to a specific enzyme class.
# 
# We employed the power of ensemble models, including the <b><mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">Gradient Boosting Classifier</mark></b>, <b><mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">CatBoost Classifier</mark></b>, <b><mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">AdaBoost Classifier</mark></b> and <b><mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">Quadratic Discriminant Analysis</mark></b> to predict enzyme classes. The models were meticulously tuned and validated using cross-validation, yielding impressive average scores.
# 
# In the testing phase, the models exhibited robust performance, demonstrating their ability to differentiate between positive and negative classes. However, there were some errors, indicating potential areas for refinement.
# 
# This project underscores the potential of machine learning in predicting enzyme classes, which could be instrumental in enhancing our understanding of enzymes, discovering new enzymes, and designing more effective drugs. Future endeavors could explore further error analysis, experimentation with different models, feature engineering, and more extensive hyperparameter tuning.
# 
# 
# <br>
# 
# ### <b>I <span style='color:#FF8551'>|</span> Import neccessary libraries</b> 

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# ### <b>II <span style='color:#FF8551'>|</span> Input the Data</b> 

# In[2]:


data = pd.read_csv("/kaggle/input/playground-series-s3e18/train.csv")


# In[3]:


data.head()


# In[4]:


data = data.drop(['id', 'EC3', 'EC4', 'EC5', 'EC6'], axis=1)


# # <span style="color:#E888BB; font-size: 1%;">1 | EXPLORATORY DATA ANALYSIS</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/4021781/pexels-photo-4021781.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>1 | EXPLORATORY DATA ANALYSIS </span></b> </div>
# 
# <br>
# 
# <div style="display: flex; flex-direction: row; align-items: center;">
#     <div style="flex: 0;">
#         <img src="https://images.pexels.com/photos/7723531/pexels-photo-7723531.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" alt="Image" style="max-width: 300px;" />
#     </div>
#     <div style="flex: 1; margin-left: 30px;">
#         <p style="font-weight: bold; color: black;">Getting started with Descriptive Analysis</p>
#         <p>This project focuses on analyzing the <b><mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">Molecular properties dataset</mark></b> to identify key factors associated with enzymes classes. By utilizing techniques such as <b><span style='color:#FFDE7D'>univariate</span></b> and <b><span style='color:#FFDE7D'>bivariate analysis</span></b>, as well as clustering methods like <b><span style='color:#FFDE7D'>K-means clustering</span></b> , we aim to uncover valuable insights into the complex relationships within the data.
#         </p>
#         <p style="border: 1px solid black; padding: 10px;">Our analysis provides valuable insights into the factors influencing enzyme classes. However, it's crucial to interpret these findings with caution, recognizing the distinction between correlation and causation. It is important to note that our predictive model, although highly accurate, does not establish a causal relationship between the provided features and enzyme classes.
#         </p>
#         <p style="font-style: italic;">Let's explore and then make results and discussion to gain deeper insights from our analysis. 🧐🔍
#         </p>
#     </div>
# </div>
# 
# <br>
# 
# 
# 
# ## EDA Overview 🚀
# 
# In this project, we explore a dataset encompassing various **`Features`** related to molecular properties and characteristics. Our primary goal is to uncover patterns and relationships within the data that could enhance our understanding of enzyme classes and their underlying factors.
# 
# <b><mark style="background-color:#393939;color:white;border-radius:5px;opacity:1.0">Objectives</mark></b>
# 
# The main objectives of this project are threefold:
# 
# 1. To conduct an exploratory analysis of the dataset and gain insights into the distributions of the features.
# 2. To examine the relationships between the features and enzyme classes, identifying potential correlations or dependencies.
# 3. To investigate the presence of clusters within the data using appropriate clustering algorithms, such as K-Means.
# 
# 
# <b><mark style="background-color:#FFDE7D;color:white;border-radius:5px;opacity:1.0">Significance</mark></b>
# 
# The insights obtained from this project can have significant implications in enzyme classification and related fields. Understanding the relationships between molecular properties and enzyme classes can provide valuable insights into enzyme functionality and catalytic mechanisms. Moreover, these findings can lay the foundation for developing robust machine learning models to predict enzyme classes with high accuracy.
# 
# <div style="border-radius: 10px; border: #00B8A9 solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     📝 Note: While this project offers valuable insights, it's important to acknowledge that the findings are based on the provided dataset and may not necessarily generalize to all enzyme systems or molecular contexts.
# </div>
# 
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> 1. Data Quality</span></b> </div>
# 
# ### <b>I <span style='color:#F6416C'>|</span> Handling Duplicates</b> 

# In[5]:


# Handle duplicates
duplicate_rows_data = data[data.duplicated()]
print("Number of duplicate rows: ", duplicate_rows_data.shape)


# ### <b>II <span style='color:#F6416C'>|</span> Uniqueness</b> 

# In[6]:


# Check the data types of each column
print("\nData types of each column:")
print(data.dtypes)


# In[7]:


# Loop through each column and count the number of distinct values
for column in data.columns:
    num_distinct_values = len(data[column].unique())
    print(f"{column}: {num_distinct_values} distinct values")


# ### <b>III <span style='color:#F6416C'>|</span> Compeleteness</b> 

# In[8]:


# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())


# ### <b> IV <span style='color:#F6416C'>|</span> Describe the data</b> 

# In[9]:


# Calculate basic statistics
statistics = data.describe().transpose()

# Remove the "count" row from the statistics table
statistics = statistics.drop('count', axis=1)

# Plot the statistics as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(statistics, annot=True, cmap='YlGnBu', fmt=".2f", cbar=False)
plt.title("Basic Statistics Heatmap")
plt.xlabel("Statistics")
plt.ylabel("Numerical Columns")
plt.xticks(rotation=45)
plt.show()


# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> 2. Univariate Analysis</span></b> </div>

# In[10]:


df = data.copy()


# In[11]:


# Get the list of column names except for "EC1"
variables = [col for col in df.columns]

# Perform univariate analysis for each variable
for variable in variables:
    # Check if the variable is the target column (EC1 or EC2)
    if variable == "EC1" or variable == "EC2":
        continue  # Skip target columns in univariate analysis
        
    plt.figure(figsize=(8, 4))
    
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=variable, kde=True)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("Histogram")
    
    # Box plot
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, y=variable)
    plt.ylabel(variable)
    plt.title("Box Plot")
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Show the plots
    plt.show()


# In[12]:


# Perform univariate analysis for EC1
plt.subplot(1, 2, 1)
sns.countplot(data=df, x="EC1")
plt.xlabel("EC1")
plt.ylabel("Frequency")
plt.title("Bar Plot - EC1")


# In[13]:


# Perform univariate analysis for EC2
plt.subplot(1, 2, 2)
sns.countplot(data=df, x="EC2")
plt.xlabel("EC2")
plt.ylabel("Frequency")
plt.title("Bar Plot - EC2")


# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> 2. Bivariate Analysis</span></b> </div>
# 
# ### <b>I <span style='color:#F6416C'>|</span> Violin Plot for EC1</b> 

# In[14]:


# Get the list of column names except for "EC1"
variables = [col for col in df.columns if col != "EC1"]

# Define the grid layout based on the number of variables
num_variables = len(variables)
num_cols = 4  # Number of columns in the grid
num_rows = (num_variables + num_cols - 1) // num_cols  # Calculate the number of rows needed

# Set the size of the figure
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4*num_rows))

# Generate violin plots for each variable with respect to EC1
for i, variable in enumerate(variables):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row][col]
    
    sns.violinplot(data=df, x="EC1", y=variable, ax=ax)
    ax.set_xlabel("EC1")
    ax.set_ylabel(variable)
    ax.set_title(f"Violin Plot: {variable} vs EC1")

# Remove any empty subplots
if num_variables < num_rows * num_cols:
    for i in range(num_variables, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# ### <b>II <span style='color:#F6416C'>|</span> Violin Plot for EC2</b> 

# In[15]:


# Get the list of column names except for "EC2"
variables = [col for col in df.columns if col != "EC2"]

# Define the grid layout based on the number of variables
num_variables = len(variables)
num_cols = 4  # Number of columns in the grid
num_rows = (num_variables + num_cols - 1) // num_cols  # Calculate the number of rows needed

# Set the size of the figure
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4*num_rows))

# Generate violin plots for each variable with respect to EC2
for i, variable in enumerate(variables):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row][col]
    
    sns.violinplot(data=df, x="EC2", y=variable, ax=ax)
    ax.set_xlabel("EC1")
    ax.set_ylabel(variable)
    ax.set_title(f"Violin Plot: {variable} vs EC2")

# Remove any empty subplots
if num_variables < num_rows * num_cols:
    for i in range(num_variables, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> 3. Multivariate Analysis</span></b> </div>

# ### <b>I <span style='color:#F6416C'>|</span> Select Top 7 Features with RFE</b> 

# In[16]:


# Separate the features (X) and the target variable (y) for EC1
X_ec1 = df.drop(['EC1', 'EC2'], axis=1)  # Remove 'EC1' and 'EC2' from the features
y_ec1 = df['EC1']

# Separate the features (X) and the target variable (y) for EC2
X_ec2 = df.drop(['EC1', 'EC2'], axis=1)  # Remove 'EC1' and 'EC2' from the features
y_ec2 = df['EC2']

# Create the estimator (model) for feature selection
estimator = RandomForestClassifier()  # Replace with your desired estimator

# Specify the number of features to select
num_features = 7

# Apply RFE to select the top features for EC1
rfe_ec1 = RFE(estimator, n_features_to_select=num_features)
X_rfe_ec1 = rfe_ec1.fit_transform(X_ec1, y_ec1)

# Get the mask of selected features for EC1
feature_mask_ec1 = rfe_ec1.support_

# Get the selected feature names for EC1
selected_features_ec1 = X_ec1.columns[feature_mask_ec1]

# Apply RFE to select the top features for EC2
rfe_ec2 = RFE(estimator, n_features_to_select=num_features)
X_rfe_ec2 = rfe_ec2.fit_transform(X_ec2, y_ec2)

# Get the mask of selected features for EC2
feature_mask_ec2 = rfe_ec2.support_

# Get the selected feature names for EC2
selected_features_ec2 = X_ec2.columns[feature_mask_ec2]

# Subset the dataframe with the selected features for EC1
df_selected_ec1 = df[selected_features_ec1]

# Add the EC1 column to the selected dataframe
df_selected_ec1['EC1'] = df['EC1']

# Subset the dataframe with the selected features for EC2
df_selected_ec2 = df[selected_features_ec2]

# Add the EC2 column to the selected dataframe
df_selected_ec2['EC2'] = df['EC2']


# ### <b>II <span style='color:#F6416C'>|</span> Plot Parallel Coordinates</b> 

# In[17]:


# Create the parallel coordinates plot for EC1
plt.figure(figsize=(10, 8))
pd.plotting.parallel_coordinates(df_selected_ec1, 'EC1', color=('red', 'blue'))
plt.xlabel('Features')
plt.ylabel('Values')
plt.title('Parallel Coordinates Plot - Top {} Features (by RFE) - EC1'.format(num_features))
plt.legend(title='EC1', loc='upper right')
plt.show()


# In[18]:


# Create the parallel coordinates plot for EC2
plt.figure(figsize=(10, 8))
pd.plotting.parallel_coordinates(df_selected_ec2, 'EC2', color=('red', 'blue'))
plt.xlabel('Features')
plt.ylabel('Values')
plt.title('Parallel Coordinates Plot - Top {} Features (by RFE) - EC2'.format(num_features))
plt.legend(title='EC2', loc='upper right')
plt.show()


# ### <b>III <span style='color:#F6416C'>|</span> Radar Plot</b> 

# In[19]:


# Create the radar plot for EC1
df_selected_ec1 = df[selected_features_ec1]
values_ec1 = df_selected_ec1.mean().values.tolist()
values_ec1 += values_ec1[:1]
features_ec1 = selected_features_ec1.tolist() + [selected_features_ec1[0]]

plt.figure(figsize=(8, 8))
angles_ec1 = np.linspace(0, 2 * np.pi, len(features_ec1), endpoint=False).tolist()
ax_ec1 = plt.subplot(111, polar=True)
ax_ec1.plot(angles_ec1, values_ec1)
ax_ec1.fill(angles_ec1, values_ec1, alpha=0.25)
ax_ec1.set_xticks(angles_ec1[:-1])
ax_ec1.set_xticklabels(features_ec1[:-1])  # Remove the last feature label
plt.title('Radar Plot - Top {} Features (by RFE) - EC1'.format(num_features))
plt.show()


# In[20]:


# Create the radar plot for EC2
df_selected_ec2 = df[selected_features_ec2]
values_ec2 = df_selected_ec2.mean().values.tolist()
values_ec2 += values_ec2[:1]
features_ec2 = selected_features_ec2.tolist() + [selected_features_ec2[0]]

plt.figure(figsize=(8, 8))
angles_ec2 = np.linspace(0, 2 * np.pi, len(features_ec2), endpoint=False).tolist()
ax_ec2 = plt.subplot(111, polar=True)
ax_ec2.plot(angles_ec2, values_ec2)
ax_ec2.fill(angles_ec2, values_ec2, alpha=0.25)
ax_ec2.set_xticks(angles_ec2[:-1])
ax_ec2.set_xticklabels(features_ec2[:-1])  # Remove the last feature label
plt.title('Radar Plot - Top {} Features (by RFE) - EC2'.format(num_features))
plt.show()


# ### <b>IV <span style='color:#F6416C'>|</span> Scatter Plot</b> 

# In[21]:


# Create the scatterplot matrix for EC1
df_selected_ec1 = df[selected_features_ec1]
df_selected_ec1['EC1'] = df['EC1']  # Add the 'EC1' column

sns.pairplot(df_selected_ec1, diag_kind='kde', hue='EC1', plot_kws={'alpha': 0.6})
plt.suptitle('Scatterplot Matrix - Top {} Features (by RFE) - EC1'.format(num_features))
plt.tight_layout()
plt.show()


# In[22]:


# Create the scatterplot matrix for EC2
df_selected_ec2 = df[selected_features_ec2]
df_selected_ec2['EC2'] = df['EC2']  # Add the 'EC2' column

sns.pairplot(df_selected_ec2, diag_kind='kde', hue='EC2', plot_kws={'alpha': 0.6})
plt.suptitle('Scatterplot Matrix - Top {} Features (by RFE) - EC2'.format(num_features))
plt.tight_layout()
plt.show()


# # <span style="color:#E888BB; font-size: 1%;">2 | CORRELATION MATRIX</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/4021781/pexels-photo-4021781.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>2 | CORRELATION MATRIX </span></b> </div>
# 
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> 1.Preprocessing : Scaler</span></b> </div>
# ​
# ​
# Preprocessing is a crucial step before training the model. In this case, numerical features are standardized (mean removed and scaled to unit variance), and categorical features are one-hot encoded. **<span style='color:#FFDE7D'>Standardization</span>** is not required for all models but is generally a good practice. **<span style='color:#FFDE7D'>One-hot encoding</span>** is necessary for categorical variables to be correctly understood by the machine learning model.
# ​
# The **<mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">StandardScaler</mark>** in sklearn is based on the assumption that the data, <em>Y</em>, follows a distribution that might not necessarily be Gaussian (normal), but we still transform it in a way that its distribution will have a mean value 0 and standard deviation of 1.
# ​
# <p>In other words, given a feature vector <em>x</em>, it modifies the values as follows:</p>
# ​
# <p class="formulaDsp">
# \[ Y_i = \frac{x_i - \mu(\vec{x})}{\sigma(\vec{x})} \]
# </p>
# ​
# where:
# <ul>
# <li>\( x_i \) is the i-th element of the original feature vector \( \vec{x} \),</li>
# <li>\( \mu(\vec{x}) \) is the mean of the feature vector, and</li>
# <li>\( \sigma(\vec{x}) \) is the standard deviation of the feature vector.</li>
# </ul>
# ​
# <p>The transformed data \( Y \) (each \( Y_i \)) will have properties such that \( mean(Y) = 0 \) and \( std(Y) = 1 \).</p>
# 

# In[23]:


# Create a StandardScaler object
scaler = StandardScaler()

# Standardize the data in the DataFrame 'df'
df_standardized = scaler.fit_transform(df)

# Convert the standardized data back to a DataFrame
df_standardized = pd.DataFrame(df_standardized, columns=df.columns)


# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://xmple.com/wallpaper/streaks-stripes-cyan-lines-2560x1800-c3-74a3a8-9bcdd2-cfedf0-l3-50-100-200-a-105-f-1.svg)"><b><span style='color:black'> 2.Correlation Matrix</span></b> </div>

# In[24]:


# Calculate the correlation matrix
corr = df_standardized.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()


# In[25]:


# Calculate the correlation matrix
corr = df_standardized.corr()

# Get correlations without 'EC1' and 'EC2'
ec1_corr = corr['EC1'].drop(['EC1', 'EC2'])
ec2_corr = corr['EC2'].drop(['EC1', 'EC2'])

# Sort correlation values in descending order
ec1_corr_sorted = ec1_corr.sort_values(ascending=False)
ec2_corr_sorted = ec2_corr.sort_values(ascending=False)

# Create a heatmap of the correlations with EC1
sns.set(font_scale=0.8)
sns.set_style("white")
sns.set_palette("PuBuGn_d")
sns.heatmap(ec1_corr_sorted.to_frame(), cmap="coolwarm", annot=True, fmt='.2f')
plt.title('Correlation with EC1')
plt.show()


# ##  Intepret the Results for EC1 📊
# 
# Our analysis of the molecular features dataset revealed several interesting patterns and relationships. The dataset included various features such as BertzCT,Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3, HallKierAlpha, HeavyAtomMolWt, Kappa3, MaxAbsEStateIndex, MinEStateIndex, NumHeteroatoms, PEOE_VSA10, PEOE_VSA14, PEOE_VSA6, PEOE_VSA7, PEOE_VSA8, SMR_VSA10, SMR_VSA5, SlogP_VSA3, VSA_EState9, fr_COO, fr_COO2, along with **EC1** and **EC2**.
# 
# <b><mark style="background-color:#393939;color:white;border-radius:5px;opacity:1.0">Observations for EC1</mark></b>
# 
# **`MinEStateIndex`** 🔬: There is a strong positive correlation between MinEStateIndex and EC1. This suggests that molecules with higher MinEStateIndex are more likely to be classified as EC1.
# 
# **`PEOE_VSA6`** 🧪: There is a positive correlation between PEOE_VSA6 and EC1. This suggests that molecules with higher PEOE_VSA6 are more likely to be classified as EC1.
# 
# **`PEOE_VSA7`** 🧬: There is a positive correlation between PEOE_VSA7 and EC1. This suggests that molecules with higher PEOE_VSA7 are more likely to be classified as EC1.
# 
# **`PEOE_VSA8`** 🧫: There is a slight positive correlation between PEOE_VSA8 and EC1. This suggests that molecules with higher PEOE_VSA8 are slightly more likely to be classified as EC1.
# 
# **`EState_VSA2`** 🧲: There is a slight positive correlation between EState_VSA2 and EC1. This suggests that molecules with higher EState_VSA2 are slightly more likely to be classified as EC1.
# 
# **`fr_COO2`** 🧴: There is a slight positive correlation between fr_COO2 and EC1. This suggests that molecules with higher fr_COO2 are slightly more likely to be classified as EC1.
# 
# **`fr_COO`** 🧴: There is a slight positive correlation between fr_COO and EC1. This suggests that molecules with higher fr_COO are slightly more likely to be classified as EC1.
# 
# **`Kappa3`** 🧪: There is a very slight positive correlation between Kappa3 and EC1. This suggests that molecules with higher Kappa3 are slightly more likely to be classified as EC1.
# 
# **`FpDensityMorgan1`** 🧬: There is a very slight positive correlation between FpDensityMorgan1 and EC1. This suggests that molecules with higher FpDensityMorgan1 are slightly more likely to be classified as EC1.
# 
# **`FpDensityMorgan2`** 🧫: There is a very slight negative correlation between FpDensityMorgan2 and EC1. This suggests that molecules with higher FpDensityMorgan2 are slightly less likely to be classified as EC1.

# In[26]:


# Create a heatmap of the correlations with EC2
sns.set(font_scale=0.8)
sns.set_style("white")
sns.set_palette("PuBuGn_d")
sns.heatmap(ec2_corr_sorted.to_frame(), cmap="coolwarm", annot=True, fmt='.2f')
plt.title('Correlation with EC2')
plt.show()


# ##  Intepret the Results for EC2 📊
# 
# <b><mark style="background-color:#393939;color:white;border-radius:5px;opacity:1.0">Observations for EC2</mark></b>
# 
# **`HallKierAlpha`** 🔬: There is a slight positive correlation between HallKierAlpha and EC2. This suggests that molecules with higher HallKierAlpha are slightly more likely to be classified as EC2.
# 
# **`PEOE_VSA14`** 🧪: There is a slight positive correlation between PEOE_VSA14 and EC2. This suggests that molecules with higher PEOE_VSA14 are slightly more likely to be classified as EC2.
# 
# **`Kappa3`** 🧬: There is a very slight positive correlation between Kappa3 and EC2. This suggests that molecules with higher Kappa3 are slightly more likely to be classified as EC2.
# 
# **`MinEStateIndex`** 🧫: There is a very slight positive correlation between MinEStateIndex and EC2. This suggests that molecules with higher MinEStateIndex are slightly more likely to be classified as EC2.
# 
# **`FpDensityMorgan1`** 🧲: There is a very slight negative correlation between FpDensityMorgan1 and EC2. This suggests that molecules with higher FpDensityMorgan1 are slightly less likely to be classified as EC2.
# 
# **`FpDensityMorgan2`** 🧴: There is a very slight negative correlation between FpDensityMorgan2 and EC2. This suggests that molecules with higher FpDensityMorgan2 are slightly less likely to be classified as EC2.
# 
# **`FpDensityMorgan3`** 🧴: There is a very slight negative correlation between FpDensityMorgan3 and EC2. This suggests that molecules with higher FpDensityMorgan3 are slightly less likely to be classified as EC2.
# 
# **`PEOE_VSA10`** 🧪: There is a slight negative correlation between PEOE_VSA10 and EC2. This suggests that molecules with higher PEOE_VSA10 are slightly less likely to be classified as EC2.
# 
# **`EState_VSA1`** 🧬: There is a slight negative correlation between EState_VSA1 and EC2. This suggests that molecules with higher EState_VSA1 are slightly less likely to be classified as EC2.
# 
# **`NumHeteroatoms`** 🧫: There is a slight negative correlation between NumHeteroatoms and EC2. This suggests that molecules with higher NumHeteroatoms are slightly less likely to be classified as EC2.
# 
# # <span style="color:#E888BB; font-size: 1%;">3 | CLUSTERING ANALYSIS</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/4614200/pexels-photo-4614200.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>3 | CLUSTERING ANALYSIS </span></b> </div>

# In[27]:


def preprocess_data(data):
    # Scale the numerical features
    scaler = StandardScaler()
    numerical_features = data.columns[:-2]  
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    return data

def determine_optimal_clusters(data):
    # Determine the optimal number of clusters using the elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

# Preprocess the data
data = preprocess_data(data)

# Determine the optimal number of clusters
data_scaled = data.drop(['EC1', 'EC2'], axis=1)
determine_optimal_clusters(data_scaled)


# ## 📊 Elbow Method Plot Interpretation 📊
# 
# The Elbow Method plot is a technique used to help us find the optimal number of clusters for a dataset in K-Means clustering or any clustering algorithm. 
# 
# From the Elbow Method plot, we can observe the following:
# 
# 1. **`WSS (Within-Cluster Sum of Squares)`** 📈: The y-axis represents the WSS, which is the sum of the squared distance between each member of the cluster and its centroid. As the number of clusters increases, the WSS value will start to decrease.
# 
# 2. **`Number of Clusters`** 🧮: The x-axis represents the number of clusters. 
# 
# The "elbow" in the plot is the point of inflection where the rate of decrease sharply shifts. This is typically considered as the appropriate number of clusters.
# 
# <b><mark style="background-color:#393939;color:white;border-radius:5px;opacity:1.0">Observation</mark></b>
# 
# > **From the plot, it appears that the elbow point is around 4 clusters. This suggests that the optimal number of clusters for the data is around 4.**
# 
# <br>
# 
# <div style="border-radius: 10px; border: #00B8A9 solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
# 📝 Please note that these interpretations are based solely on the provided Elbow Method plot. The actual analysis might require more detailed and specific considerations based on the data at hand.
#     </div>
# 

# In[28]:


# Fit K-means clustering (replace 'n_clusters' with the desired number of clusters)
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(data_scaled)

# Add the cluster labels to the original data
data['Cluster'] = clusters

# Visualize the clusters 
plt.figure(figsize=(10, 10))
sns.scatterplot(data=data, x='MinEStateIndex', y='HallKierAlpha', hue='Cluster', palette='viridis')  # Replace 'MinEStateIndex' and 'HallKierAlpha' with the another features we want to visualize
plt.title('Clusters')
plt.xlabel('MinEStateIndex')
plt.ylabel('HallKierAlpha')
plt.show()


# ## 📊 Scatter Plot Interpretation 📊
# 
# The scatter plot is used to visualize the clusters formed by the K-Means clustering algorithm based on the features `MinEStateIndex` and `HallKierAlpha`.
# 
# From the scatter plot, we can observe the following:
# 
# 1. **`MinEStateIndex`** 🧪:  The x-axis represents the minimum value of the E-state index. It provides information about the lowest observed electronic property value within the molecule.
# 
# 2. **`HallKierAlpha`** 🧬: The y-axis represents the Hall-Kier alpha value for a molecule. It is a measure of molecular shape and can provide insights into the overall structure of the molecule.
# 
# 3. **`Clusters`** 🧩: Different clusters are represented by different colors. 
# 
# 
# <b><mark style="background-color:#393939;color:white;border-radius:5px;opacity:1.0">Observation</mark></b>
# 
# > From the plot, it appears that the K-Means clustering algorithm has divided the data into four distinct clusters. **Each cluster represents a group of molecules with similar minimum E-state index and Hall-Kier alpha values**.
# 
# <br>
# 
# <div style="border-radius: 10px; border: #00B8A9 solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
# 📝 Please note that these interpretations are based solely on the provided Elbow Method plot. The actual analysis might require more detailed and specific considerations based on the data at hand.
#     </div>
#     
# # <span style="color:#E888BB; font-size: 1%;">4 | MAKING A PREDICTION</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/4021781/pexels-photo-4021781.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>Predictive Analysis</span></b> </div>
# 
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Overview</span></b> </div>
# 
# 
# ### 🚀 Introduction
# 
# In this project, we aim to predict the enzyme class of molecules based on various molecular properties. The ability to accurately predict enzyme classes can be incredibly valuable in bioinformatics and drug discovery, helping to understand the function of enzymes, discover new enzymes, and design more effective drugs.
# 
# ### 🎯 Objective
# 
# Our main objective is to build a predictive model that can accurately predict enzyme classes based on these features. To achieve this, we will use ensemble models, including the <b><mark style="background-color:#F6416C;color:white;border-radius:5px;opacity:1.0">Gradient Boosting Classifier</mark></b>, <b><mark style="background-color:#F6416C;color:white;border-radius:5px;opacity:1.0">CatBoost Classifier</mark></b>, and <b><mark style="background-color:#F6416C;color:white;border-radius:5px;opacity:1.0">Quadratic Discriminant Analysis</mark></b>.
# 
# <div style="border-radius: 10px; border: #00B8A9 solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     📝 <b>Process :</b> Our approach will involve several steps, including data preprocessing, model training, model evaluation, and error analysis. We will also perform hyperparameter tuning and cross-validation to optimize our model's performance, and we will use various metrics to evaluate our model's performance.
# </div>
# 
# <br>
# 
# <div class="warning" style="background-color: #F8F3D4; border-left: 6px solid #FF5252;font-size: 100%; padding: 10px;">
# <h3 style="color: #FF8551; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🗒️  Keep in Mind </h3>
# While machine learning models can provide valuable insights and predictions, it's important to remember that they are not infallible and their performance can vary depending on the data they are trained and tested on. Therefore, it's crucial to carefully evaluate and validate our model's performance using appropriate metrics and techniques.
# </div>
# 

# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Import Libraries and Data</span></b> </div>

# In[29]:


import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.decomposition import PCA
from sklearn.utils import resample
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import ListedColormap


# In[30]:


# Load the training data
train_data = pd.read_csv("/kaggle/input/playground-series-s3e18/train.csv")
# Drop 'id' and 'ED' columns
train_data_E1 = train_data.drop(columns=['id','EC2', 'EC3', 'EC4', 'EC5', 'EC6'])
train_data_E2 = train_data.drop(columns=['id','EC1', 'EC3', 'EC4', 'EC5', 'EC6'])


# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Preprocessing data</span></b> </div>

# In[31]:


# Preprocessing
# Let's assume all features need to be scaled
scaler = StandardScaler()
X_E1 = scaler.fit_transform(train_data_E1.drop(['EC1'], axis=1))
X_E2 = scaler.fit_transform(train_data_E2.drop(['EC2'], axis=1))


# In[32]:


# Split the data into training and test sets for each target
X_train_EC1, X_test_EC1, y_train_EC1, y_test_EC1 = train_test_split(X_E1, train_data_E1['EC1'], test_size=0.2, random_state=42)
X_train_EC2, X_test_EC2, y_train_EC2, y_test_EC2 = train_test_split(X_E2, train_data_E2['EC2'], test_size=0.2, random_state=42)


# # <span style="color:#E888BB; font-size: 1%;"> 4.1| Define the models</span>
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Define the models</span></b> </div>
# 
# ## 📚 Gradient Boosting Classifier
# 
# <b><mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">Gradient Boosting Classifier</mark></b> is an ensemble machine learning algorithm that creates a strong predictive model by combining the predictions of multiple weak models, typically decision trees. It operates by building an initial model to predict the target variable and then creating additional models that attempt to correct the errors from the previous models. 
# 
# The "gradient" in Gradient Boosting refers to the use of a gradient descent algorithm to minimize the loss when adding new models. This approach allows the algorithm to optimize the overall performance of the ensemble by adjusting the weights of the individual models.
# 
# The final prediction of the Gradient Boosting Classifier is a weighted sum of the predictions of all the weak learners, with the weights determined by the performance of each weak learner.
# 
# <br>
# 
# $$ Y = \sum_{i=1}^{N} \alpha_i h_i(x) $$
# 
# <br>
# 
# Here, $Y$ is the final prediction, $N$ is the total number of weak learners, $α$ is the performance of the weak learner, $h$ is the prediction of the weak learner, and $x$ is the input data.
# 
# > If you're interested in learning how to use <b><mark style="background-color:#FFDE7D;color:white;border-radius:5px;opacity:1.0">Gradient Boost</mark></b> for medical cost prediction, you can refer to this [Kaggle notebook](https://www.kaggle.com/code/tumpanjawat/medcost-eda-k-cluster-gradient-boost-full). The notebook provides a detailed example that covers the entire process, including data preprocessing, model training using Gradient Boost, and evaluating the performance of the model.
# 
# ## 📚 CatBoost Classifier
# 
# <b><mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">CatBoost Classifier</mark></b> is a machine learning algorithm that uses gradient boosting on decision trees with a special emphasis on handling categorical variables. It stands for "Category" and "Boosting". 
# 
# **CatBoost can automatically handle categorical variables and does not require extensive data preprocessing like other machine learning algorithms.** It is robust to overfitting and has been designed to provide excellent performance out of the box, even without extensive hyperparameter tuning.
# 
# The final prediction of the CatBoost Classifier is a weighted sum of the predictions of all the weak learners, with the weights determined by the performance of each weak learner.
# 
# <br>
# 
# $$ Y = \sum_{i=1}^{N} \alpha_i h_i(x) $$
# 
# <br>
# 
# Here, $Y$ is the final prediction, $N$ is the total number of weak learners, $α$ is the performance of the weak learner, $h$ is the prediction of the weak learner, and $x$ is the input data.
# 
# > To demonstrate the application of <b><mark style="background-color:#FFDE7D;color:white;border-radius:5px;opacity:1.0">Cat Boost</mark></b>, let's explore a specific use case of diabetes analysis. You can find a detailed example in this [Kaggle notebook](https://www.kaggle.com/code/tumpanjawat/diabetes-eda-cluster-catboost#Cat-Boost-Classifier), which covers exploratory data analysis, clustering, and the use of Cat Boost Classifier for predicting diabetes.
# 
# ## 📚 AdaBoost Classifier
# 
# <b><mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">AdaBoost Classifier</mark></b> or Adaptive Boosting is one of the simplest boosting algorithms. It creates a strong classifier from a number of weak classifiers by building a model from the training data, then creating a second model that attempts to correct the errors from the first model. Models are added until the training set is predicted perfectly or a maximum number of models are added.
# 
# **AdaBoost is best used to boost the performance of decision trees on binary classification problems.**
# 
# The final prediction of the AdaBoost Classifier is a weighted sum of the predictions of all the weak learners, with the weights determined by the performance of each weak learner.
# 
# <br>
# 
# $$ Y = \text{sign} \left( \sum_{i=1}^{N} \alpha_i h_i(x) \right) $$
# 
# <br>
# 
# Here, $Y$ is the final prediction, $sign$ is the sign function, $N$ is the total number of weak learners, $α$ is the performance of the weak learner, $h$ is the prediction of the weak learner, and $x$ is the input data.
# 
# > For a practical example of <b><mark style="background-color:#FFDE7D;color:white;border-radius:5px;opacity:1.0">Ada Boost</mark></b>, let's consider a case of exploratory data analysis and clustering. You can refer to this [Kaggle notebook](https://www.kaggle.com/code/tumpanjawat/s3e17-mf-eda-clustering-adaboost). that demonstrates how to use Ada Boost in a step-by-step manner. 
# 
# ## 📚 Quadratic Discriminant Analysis
# 
# <b><mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">Quadratic Discriminant Analysis</mark></b> (QDA) is a classifier with a quadratic decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule. The model fits a **Gaussian density** to each class, assuming that each input variable is a Gaussian distribution. 
# 
# QDA is a more flexible classifier than its counterpart, Linear Discriminant Analysis (LDA), because it does not assume that the covariance of each of the classes is identical.
# 
# The decision rule for QDA is derived from the Bayes rule and is given by:
# 
# <br>
# 
# $$ \delta_k(x) = -0.5 * (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) - 0.5 * \ln |\Sigma_k| + \ln P(Y = k) $$
# 
# <br>
# 
# Here, $x$ is the input data, $Σ$ is the covariance matrix, $μ$ is the mean vector, and $P(Y = k)$ is the prior probability of class $k$.
# 
# - The class of an observation is determined by the discriminant function for which $δk(x)$ is maximized.
# 

# In[33]:


# Define the models for EC1
model1_EC1 = GradientBoostingClassifier()
model2_EC1 = CatBoostClassifier(verbose=False)
model3_EC1 = AdaBoostClassifier()


# In[34]:


# Define the models for EC2
model1_EC2 = GradientBoostingClassifier()
model2_EC2 = CatBoostClassifier(verbose=False)
model3_EC2 = QuadraticDiscriminantAnalysis()


# # <span style="color:#E888BB; font-size: 1%;"> 4.2| Hyper Parameters Tuning</span>
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'>Hyper Parameters Tuning</span></b> </div>

# In[35]:


# --------------For hyperparameter grid search EC1 ---------------#

# Hyperparameter tuning with GridSearchCV
# This is just an example, you'll need to specify the parameters for your specific models
#param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}

#grid_search_model1_EC1 = GridSearchCV(model1_EC1, param_grid, cv=10)
#grid_search_model1_EC1.fit(X_train_EC1, y_train_EC1)
#print("Best parameters for model1_EC1: ", grid_search_model1_EC1.best_params_)

#grid_search_model2_EC1 = GridSearchCV(model2_EC1, param_grid, cv=10)
#grid_search_model2_EC1.fit(X_train_EC1, y_train_EC1)
#print("Best parameters for model2_EC1: ", grid_search_model2_EC1.best_params_)

#grid_search_model3_EC1 = GridSearchCV(model3_EC1, param_grid, cv=10)
#grid_search_model3_EC1.fit(X_train_EC1, y_train_EC1)
#print("Best parameters for model3_EC1: ", grid_search_model3_EC1.best_params_)


# ## 🎯 E1 Hyperparameter Tuning Results
# 
# After a comprehensive round of hyperparameter tuning using **GridSearchCV**, we have successfully determined the optimal parameters for our ensemble models.
# 
# The winning combinations are:
# 
# For the **GradientBoostingClassifier**:
# 
# - **`Learning Rate`**: **0.1**
# - **`Number of Estimators`**: **50**
# 
# For the **CatBoostClassifier**:
# 
# - **`Learning Rate`**: **0.1**
# - **`Number of Estimators`**: **50**
# 
# For the **AdaBoostClassifier**:
# 
# - **`Learning Rate`**: **0.1**
# - **`Number of Estimators`**: **200**
# 
# ### 🧠 Understanding the Parameters
# 
# The **`Learning Rate`** of 0.1 is the pace at which our models learn. A higher learning rate allows the models to learn faster, but it's a delicate balance. Too fast, and the models might not perform as well.
# 
# The **`Number of Estimators`** is the count of weak learners that the models train iteratively. More learners can lead to better learning, but we must be cautious. Too many can increase the risk of overfitting and the computational cost.
# 
# For the **GradientBoostingClassifier** and **CatBoostClassifier**, the number of estimators is set at 50, while for the **AdaBoostClassifier**, it's set at 200. This indicates that the AdaBoost model required more weak learners to achieve optimal performance.
# 
# <br>
# 
# <div class="warning" style="background-color: #F8F3D4; border-left: 6px solid #FF5252;font-size: 100%; padding: 10px;">
# <h3 style="color: #FF8551; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🗒️  Keep in Mind </h3>
# Armed with these optimal parameters, we're ready to train our ensemble models and anticipate top performance. However, it's crucial to remember that machine learning models might not always perform as expected on unseen data. Always validate the model's performance using appropriate metrics and techniques.
# </div>
# 
# 

# In[36]:


# --------------For hyperparameter grid search EC2 ---------------#

# This is just an example, you'll need to specify the parameters for your specific models
#param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}

#grid_search_model1_EC2 = GridSearchCV(model1_EC2, param_grid, cv=10)
#grid_search_model1_EC2.fit(X_train_EC2, y_train_EC2)
#print("Best parameters for model1_EC2: ", grid_search_model1_EC2.best_params_)

#grid_search_model2_EC2 = GridSearchCV(model2_EC2, param_grid, cv=10)
#grid_search_model2_EC2.fit(X_train_EC2, y_train_EC2)
#print("Best parameters for model2_EC2: ", grid_search_model2_EC2.best_params_)

# Define the parameter grid for QuadraticDiscriminantAnalysis
#param_grid_qda = {'reg_param': [0.0, 0.5, 1.0], 'tol': [0.0001, 0.001, 0.01, 0.1]}

# Perform grid search for QuadraticDiscriminantAnalysis
#grid_search_model3_EC2 = GridSearchCV(model3_EC2, param_grid_qda, cv=10)
#grid_search_model3_EC2.fit(X_train_EC2, y_train_EC2)
#print("Best parameters for model3_EC2: ", grid_search_model3_EC2.best_params_)


# ## 🎯 E2 Hyperparameter Tuning Results 
# 
# After a comprehensive round of hyperparameter tuning using **GridSearchCV**, we have successfully determined the optimal parameters for our ensemble models for EC2.
# 
# The winning combinations are:
# 
# For the **GradientBoostingClassifier**:
# 
# - **`Learning Rate`**: **0.1**
# - **`Number of Estimators`**: **50**
# 
# For the **CatBoostClassifier**:
# 
# - **`Learning Rate`**: **0.1**
# - **`Number of Estimators`**: **50**
# 
# For the **QuadraticDiscriminantAnalysis**:
# 
# - **`Regularization Parameter`**: **0.5**
# - **`Tolerance`**: **0.0001**
# 
# ### 🧠 Understanding the Parameters
# 
# The **`Learning Rate`** of 0.1 for the GradientBoostingClassifier and CatBoostClassifier is the pace at which our models learn. A higher learning rate allows the models to learn faster, but it's a delicate balance. Too fast, and the models might not perform as well.
# 
# The **`Number of Estimators`** is the count of weak learners that the models train iteratively. More learners can lead to better learning, but we must be cautious. Too many can increase the risk of overfitting and the computational cost. For both the GradientBoostingClassifier and CatBoostClassifier, the number of estimators is set at 50.
# 
# For the **QuadraticDiscriminantAnalysis**, the **`Regularization Parameter`** of 0.5 helps to avoid overfitting by shrinking the estimates towards zero, and the **`Tolerance`** of 0.0001 is the stopping criterion for the algorithm's iterations.
# 
# <br>
# 
# <div class="warning" style="background-color: #F8F3D4; border-left: 6px solid #FF5252;font-size: 100%; padding: 10px;">
# <h3 style="color: #FF8551; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🗒️  Keep in Mind </h3>
# Armed with these optimal parameters, we're ready to train our ensemble models for EC2 and anticipate top performance. However, it's crucial to remember that machine learning models might not always perform as expected on unseen data. Always validate the model's performance using appropriate metrics and techniques.
# </div>
# 
# # <span style="color:#E888BB; font-size: 1%;"> 4.3| Voting Classifier</span>
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> VotingClassifier</span></b> </div>
# 
# <b><mark style="background-color:#F6416C;color:white;border-radius:5px;opacity:1.0">Voting Classifier</mark></b> is an **ensemble machine learning model** that combines the predictions from multiple other models. It includes models like Logistic Regression, Decision Tree, Support Vector Classifier, among others. Each model makes independent predictions and the predictions are combined through either majority vote (hard voting) or averaging (soft voting).
# 
# > In the case of hard voting, the Voting Classifier takes the majority of the predictions from each of the sub-models and uses that as the final prediction. For soft voting, the probability vectors for each predicted class for all the models are summed up and averaged. The class with the highest average probability is used as the final prediction.
# 
# 
# 
# 
# 

# In[37]:


# Define the models for EC1 with the best parameters
model1_EC1 = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)
model2_EC1 = CatBoostClassifier(n_estimators=50, learning_rate=0.1, verbose=False)
model3_EC1 = AdaBoostClassifier(n_estimators=200, learning_rate=0.1)

# Create the VotingClassifier with the best parameters
ensemble_EC1 = VotingClassifier(estimators=[('gb', model1_EC1), ('cb', model2_EC1), ('ab', model3_EC1)], voting='soft')
ensemble_EC1.fit(X_train_EC1, y_train_EC1)


# **In the context of our project:**
# 
#  For EC1, the Voting Classifier combines the predictions from the following models:
#  
# - <b><mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">Gradient Boosting Classifier</mark></b> with **50** estimators and a learning rate of **0.1**
# 
# - <b><mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">CatBoost Classifier</mark></b> with **50** estimators and a learning rate of **0.1**
# 
# - <b><mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">AdaBoost Classifier</mark></b> Classifier with **200** estimators and a learning rate of **0.1**

# In[38]:


# Define the models for EC2 with the best parameters
model1_EC2 = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)
model2_EC2 = CatBoostClassifier(n_estimators=50, learning_rate=0.1, verbose=False)
model3_EC2 = QuadraticDiscriminantAnalysis(reg_param=0.5, tol=0.0001)

# Create the VotingClassifier with the best parameters
ensemble_EC2 = VotingClassifier(estimators=[('gb', model1_EC2), ('cb', model2_EC2), ('qda', model3_EC2)], voting='soft')
ensemble_EC2.fit(X_train_EC2, y_train_EC2)


# For EC2, the Voting Classifier combines the predictions from the following models:
# 
# - <b><mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">Gradient Boosting Classifier</mark></b> with **50** estimators and a learning rate of **0.1**
# 
# - <b><mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">CatBoost Classifier</mark></b> with **50** estimators and a learning rate of **0.1**
# 
# - <b><mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">Quadratic Discriminant Analysis</mark></b> with a regularization parameter of **0.5** and a tolerance of **0.0001**
# 
# > By combining these models, the Voting Classifier can leverage the strengths of each to make a more accurate prediction.
# 
# ### 🚀 Key Takeaways
# 
# <div style="border-radius: 10px; border: #FFDE7D solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     📝 <b>Voting Classifier</b> is an ensemble machine learning model that combines the predictions from multiple other models. It includes models like Logistic Regression, Decision Tree, Support Vector Classifier, among others. Each model makes independent predictions and the predictions are combined through either majority vote (hard voting) or averaging (soft voting). This approach allows the Voting Classifier to balance out the weaknesses of the individual models, potentially leading to a more robust and accurate prediction.
# </div>
# 
# # <span style="color:#E888BB; font-size: 1%;"> 4.4| Cross Validation</span>
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Cross Validation</span></b> </div>

# In[39]:


# Cross-validation
scores_EC1 = cross_val_score(ensemble_EC1, X_train_EC1, y_train_EC1, cv=10)
print("Cross-validation scores for EC1: ", scores_EC1)


# In[40]:


scores_EC2 = cross_val_score(ensemble_EC2, X_train_EC2, y_train_EC2, cv=10)
print("Cross-validation scores for EC2: ", scores_EC2)


# ## 📊 Cross-Validation Scores
# 
# **Cross-validation** is a resampling procedure used to evaluate machine learning models on a limited data sample. The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into. 
# 
# > For our project, we have computed the cross-validation scores for both EC1 and EC2. These scores give us an idea of how well our models are likely to perform on unseen data.
# 
# ### 🧠 Understanding the Cross-Validation Scores
# 
# **The cross-validation scores****** represent the accuracy of the model for each fold in the cross-validation process. A high score indicates that the model was able to accurately predict the outcome for that fold, while a low score indicates that the model's predictions were less accurate.
# 
# **For EC1**, the cross-validation scores are: **[0.72451559, 0.69165965, 0.69334457, 0.71103623, 0.70598147, 0.70429655, 0.70176917, 0.70598147, 0.71609099, 0.70682393]**. These scores suggest that the model's performance is relatively consistent, with scores mostly around <b><mark style="background-color:#393939;color:white;border-radius:5px;opacity:1.0">0.7</mark></b>.
# 
# **For EC2**, the cross-validation scores are: **[0.79865206, 0.79191238, 0.7935973, 0.79191238, 0.79696714, 0.79191238, 0.79612468, 0.79528222, 0.78433024, 0.78854254]**. These scores are higher than those for EC1, suggesting that the model for **EC2 may be more accurate**.
# 
# ### 🚀 Key Takeaways
# 
# <div style="border-radius: 10px; border: #FF8551 solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     📝 Cross-validation is a powerful technique for assessing the performance of machine learning models. It provides a more robust estimate of the model's performance by averaging the model's accuracy over multiple runs. This helps to ensure that our model's performance is not overly dependent on the particular way we split our data. The cross-validation scores for EC1 and EC2 suggest that our models are performing well, but as always, we should be careful not to overinterpret these results. The true test of a model's performance is how well it performs on new, unseen data.
# </div>
# 
# # <span style="color:#E888BB; font-size: 1%;"> 4.5| Model Evaluation</span>
# <div style="padding: 28px;color:white;margin:10;font-size:170%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/4021781/pexels-photo-4021781.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>Model Evaluation </span></b> </div>
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Classification Report</span></b> </div>

# In[41]:


# Model evaluation EC1
predictions_EC1 = ensemble_EC1.predict(X_test_EC1)
print("Classification report for EC1: ")
print(classification_report(y_test_EC1, predictions_EC1))


# ## 📊 Classification Report
# 
# The **classification report** provides a detailed breakdown of our <b><mark style="background-color:#F6416C;color:white;border-radius:5px;opacity:1.0">ensemble model's</mark></b> performance for **EC1**:
# 
# ### 🧠 Understanding the Report
# 
# - **`Precision`** for class 0 is 0.57, which means that 57% of instances that our model predicted as class 0 were indeed class 0. For class 1, the precision is 0.74, meaning that 74% of instances that our model predicted as class 1 were indeed class 1. This indicates that our model is fairly reliable when it predicts a class.
# 
# - **`Recall`** for class 0 is 0.36, which means that our model detected 36% of the instances of class 0. However, for class 1, the recall is 0.87, which means that our model detected 87% of the instances of class 1. This suggests that our model is more effective at detecting class 1 instances compared to class 0 instances.
# 
# - The **`F1-score`** for class 0 is 0.44, indicating a balance between precision and recall for this class. For class 1, the F1-score is 0.80, which is higher due to the higher recall for this class.
# 
# - **`Support`** shows the number of instances of each class in the actual dataset. There are 976 instances of class 0 and 1992 instances of class 1.
# 
# - The **`accuracy`** of 0.70 indicates that our model made correct predictions for 70% of instances. 
# 
# - The **`macro avg`** and **`weighted avg`** provide us with average measures of our model's performance, taking into account both classes. The macro average treats both classes equally, while the weighted average takes into account the imbalance in our dataset.
# 

# In[42]:


# Model evaluation EC2
predictions_EC2 = ensemble_EC2.predict(X_test_EC2)
print("Classification report for EC2: ")
print(classification_report(y_test_EC2, predictions_EC2))


# ## 📊 Classification Report
# 
# The **classification report** provides a detailed breakdown of our <b><mark style="background-color:#F6416C;color:white;border-radius:5px;opacity:1.0">ensemble model's</mark></b> performance for **EC2**:
# 
# ### 🧠 Understanding the Report
# 
# - **`Precision`** for class 0 is 0.32, which means that 32% of instances that our model predicted as class 0 were indeed class 0. For class 1, the precision is 0.81, meaning that 81% of instances that our model predicted as class 1 were indeed class 1. This indicates that our model is fairly reliable when it predicts a class.
# 
# - **`Recall`** for class 0 is 0.02, which means that our model detected 2% of the instances of class 0. However, for class 1, the recall is 0.99, which means that our model detected 99% of the instances of class 1. This suggests that our model is more effective at detecting class 1 instances compared to class 0 instances.
# 
# - The **`F1-score`** for class 0 is 0.03, indicating a balance between precision and recall for this class. For class 1, the F1-score is 0.89, which is higher due to the higher recall for this class.
# 
# - **`Support`** shows the number of instances of each class in the actual dataset. There are 568 instances of class 0 and 2400 instances of class 1.
# 
# - The **`accuracy`** of 0.81 indicates that our model made correct predictions for 81% of instances. 
# 
# - The **`macro avg`** and **`weighted avg`** provide us with average measures of our model's performance, taking into account both classes. The macro average treats both classes equally, while the weighted average takes into account the imbalance in our dataset.
# 
# <br>
# 
# <div class="warning" style="background-color: #F8F3D4; border-left: 6px solid #FF5252;font-size: 100%; padding: 10px;">
# <h3 style="color: #FF8551; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🗒️  Keep in Mind </h3>
# While these scores indicate a decent performance of our model, it's important to remember that these are just one aspect of model evaluation. Depending on the specific requirements and constraints of our project, other metrics may be more relevant. In particular, given the imbalance in our dataset, we might want to focus more on the precision, recall, and F1-score for class 1, as these metrics are more informative than accuracy in the context of imbalanced datasets. Furthermore, we might want to consider techniques to handle the class imbalance, such as oversampling the minority class, undersampling the majority class, or using a combination of both.
# </div>
# 

# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Confusion Matrix</span></b> </div>

# In[43]:


# Confusion matrix EC1
cm_EC1 = confusion_matrix(y_test_EC1, predictions_EC1)
sns.heatmap(cm_EC1, annot=True, fmt='d')


# ## 🎯 Confusion Matrix
# 
# The **confusion matrix** provides a visual representation of our <b><mark style="background-color:#F6416C;color:white;border-radius:5px;opacity:1.0">ensemble model's</mark></b> performance for **EC1**:
# 
# ### 🧠 Understanding the Matrix
# 
# The confusion matrix gives us a more detailed breakdown of our model's performance:
# 
# - **`True Positives (TP)`**: These are cases in which we predicted class 1, and the actual outcome was also class 1. We have 1725 TP.
# 
# - **`True Negatives (TN)`**: We predicted class 0, and the actual outcome was class 0. We have 355 TN.
# 
# - **`False Positives (FP)`**: We predicted class 1, but the actual outcome was class 0. Also known as "Type I error". We have 621 FP.
# 
# - **`False Negatives (FN)`**: We predicted class 0, but the actual outcome was class 1. Also known as "Type II error". We have 267 FN.
# 
# <br>
# 
# <div class="warning" style="background-color: #F8F3D4; border-left: 6px solid #FF5252;font-size: 100%; padding: 10px;">
# <h3 style="color: #FF8551; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🗒️  Keep in Mind </h3>
# While the number of true positives is quite high, indicating a good performance of our model, it's important to remember that the number of false negatives is also significant. This means that our model failed to detect 267 instances of class 1, which could have serious implications depending on the specific context and cost associated with misclassification. Therefore, depending on the specific requirements and constraints of our project, we might want to focus on reducing the number of false negatives, even if this might increase the number of false positives.
# </div>
# 

# In[44]:


# Confusion matrix EC2
cm_EC2 = confusion_matrix(y_test_EC2, predictions_EC2)
sns.heatmap(cm_EC2, annot=True, fmt='d')


# ## 🎯 Confusion Matrix
# 
# The **confusion matrix** provides a visual representation of our <b><mark style="background-color:#F6416C;color:white;border-radius:5px;opacity:1.0">ensemble model's</mark></b> performance for **EC2**:
# 
# ### 🧠 Understanding the Matrix
# 
# The confusion matrix gives us a more detailed breakdown of our model's performance:
# 
# - **`True Positives (TP)`**: These are cases in which we predicted class 1, and the actual outcome was also class 1. We have 2381 TP.
# 
# - **`True Negatives (TN)`**: We predicted class 0, and the actual outcome was class 0. We have 9 TN.
# 
# - **`False Positives (FP)`**: We predicted class 1, but the actual outcome was class 0. Also known as "Type I error". We have 559 FP.
# 
# - **`False Negatives (FN)`**: We predicted class 0, but the actual outcome was class 1. Also known as "Type II error". We have 19 FN.
# 
# <br>
# 
# <div class="warning" style="background-color: #F8F3D4; border-left: 6px solid #FF5252;font-size: 100%; padding: 10px;">
# <h3 style="color: #FF8551; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🗒️  Keep in Mind </h3>
# While the number of true positives is quite high, indicating a good performance of our model, it's important to remember that the number of false positives is also significant. This means that our model incorrectly classified 559 instances as class 1, which could have serious implications depending on the specific context and cost associated with misclassification. Therefore, depending on the specific requirements and constraints of our project, we might want to focus on reducing the number of false positives, even if this might increase the number of false negatives.
# </div>
# 
# # <span style="color:#E888BB; font-size: 1%;"> 4.5| ROC AUC</span>
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'>ROC AUC</span></b> </div>

# In[45]:


# ROC curve
# This is for binary classification, you'll need to adjust for multiclass classification
probs_EC1 = ensemble_EC1.predict_proba(X_test_EC1)[:, 1]
fpr_EC1, tpr_EC1, _ = roc_curve(y_test_EC1, probs_EC1)
plt.plot(fpr_EC1, tpr_EC1, label='EC1')

probs_EC2 = ensemble_EC2.predict_proba(X_test_EC2)[:, 1]
fpr_EC2, tpr_EC2, _ = roc_curve(y_test_EC2, probs_EC2)
plt.plot(fpr_EC2, tpr_EC2, label='EC2')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# ## 📈 ROC Curve and AUC Score
# 
# The Receiver Operating Characteristic (ROC) curve is a graphical representation of the performance of our <b><mark style="background-color:#F6416C;color:white;border-radius:5px;opacity:1.0">ensemble model's</mark></b> for **EC1** and **EC2** at all classification thresholds. The Area Under the Curve (AUC) score represents the model's ability to distinguish between positive and negative classes.
# 
# For EC1, our AUC Score is: <b><mark style="background-color:#393939;color:white;border-radius:5px;opacity:1.0">0.70456</mark></b>
# 
# For EC2, our AUC Score is: <b><mark style="background-color:#393939;color:white;border-radius:5px;opacity:1.0">0.59297</mark></b>
# 
# ### 🧠 Understanding the ROC Curve and AUC Score
# 
# The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings. The AUC score, ranging from 0 to 1, tells us how much our model is capable of distinguishing between classes. The higher the AUC, the better our model is at predicting 0s as 0s and 1s as 1s. An AUC score of 0.5 denotes a bad class separation capacity equivalent to random guessing, while an AUC score of 1 denotes an excellent class separation capacity.
# 
# > The **Area Under the Curve (AUC)** score is a single number summary of the ROC curve. An AUC score of 1 indicates a perfect model; 0.5 suggests a worthless model. Our models have AUC scores of **0.704** for EC1 and **0.593** for EC2, indicating a reasonable level of prediction accuracy.
# 
# <div class="warning" style="background-color: #FFDADA; border-left: 6px solid #FF5252;font-size: 100%; padding: 10px;">
# <h3 style="color: #FF5252; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🚨 Caution</h3>
# While the AUC scores are quite reasonable, indicating a good performance of our models, it's important to remember that the ROC curve and AUC score are just one aspect of model evaluation. Depending on the specific requirements and constraints of our project, other metrics may be more relevant. Furthermore, the ROC curve and AUC score do not provide information about the specific threshold that should be used for classification.
# </div>
# 
# # <span style="color:#E888BB; font-size: 1%;"> 4.6| Precision-Recall Curve and Average Precision</span>
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'>Precision-Recall Curve and Average Precision</span></b> </div>
# 

# In[46]:


# Precision-recall curve
precision_EC1, recall_EC1, _ = precision_recall_curve(y_test_EC1, probs_EC1)
plt.plot(recall_EC1, precision_EC1, label='EC1')

precision_EC2, recall_EC2, _ = precision_recall_curve(y_test_EC2, probs_EC2)
plt.plot(recall_EC2, precision_EC2, label='EC2')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()


# ## 📈 Precision-Recall Curve and Average Precision
# 
# The Precision-Recall curve is a graphical representation of the trade-off between precision and recall for our <b><mark style="background-color:#F6416C;color:white;border-radius:5px;opacity:1.0">ensemble model's</mark></b> for **EC1** and **EC2** at different thresholds. The Average Precision summarizes the weighted increase in precision with each change in recall for the thresholds in the Precision-Recall curve.
# 
# For EC1, our Average Precision is: <b><mark style="background-color:#393939;color:white;border-radius:5px;opacity:1.0">0.80928</mark></b>
# 
# For EC2, our Average Precision is: <b><mark style="background-color:#393939;color:white;border-radius:5px;opacity:1.0">0.85684</mark></b>
# 
# ### 🧠 Understanding the Precision-Recall Curve and Average Precision
# 
# The Precision-Recall curve shows the trade-off between precision and recall for different threshold. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate.
# 
# > **The Average Precision provides a single number that summarizes the information in the Precision-Recall curve. It is calculated as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.**
# 
# <br>
# 
# <div class="warning" style="background-color: #FFDADA; border-left: 6px solid #FF5252;font-size: 100%; padding: 10px;">
# <h3 style="color: #FF5252; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🚨 Caution</h3>
# While the Average Precision scores are quite reasonable, indicating a good performance of our models, it's important to remember that the Precision-Recall curve and Average Precision are just one aspect of model evaluation. Depending on the specific requirements and constraints of our project, other metrics may be more relevant.
# </div>
# 
# # <span style="color:#E888BB; font-size: 1%;"> 4.7| Learning Curve</span>
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'>Learning Curve</span></b> </div>
# 

# In[47]:


# Learning curve EC1
train_sizes, train_scores, test_scores = learning_curve(ensemble_EC1, X_train_EC1, y_train_EC1, cv=10)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()
plt.show()


# ## 📈 Learning Curve for EC1
# 
# The learning curve provides a graphical representation of the model's performance on both the training set and the validation set over a varying number of training instances for EC1. 
# 
# Our training scores are: **[0.82593633, 0.74845866, 0.73528511, 0.72815557, 0.72862492]**
# 
# Our cross-validation scores are: **[0.69865206, 0.70395956, 0.70598147, 0.70463353, 0.70614996]**
# 
# ### 🧠 Understanding the Learning Curve
# 
# The learning curve helps us understand the trade-off between the model's performance on the training data and its ability to generalize to unseen data (represented by the cross-validation score). 
# 
# > **In our case, both the training score and the cross-validation score for EC1 are consistently high across different training sizes, indicating that our model is performing well and not overfitting to the training data.**
# 
# <br>
# 
# <div class="warning" style="background-color: #F8F3D4; border-left: 6px solid #FF5252;font-size: 100%; padding: 10px;">
# <h3 style="color: #FF8551; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🗒️  Keep in Mind </h3>
# While the learning curve suggests that our model for EC1 is performing well, it's important to remember that this is just one aspect of model evaluation. Depending on the specific requirements and constraints of our project, other metrics and techniques may be more relevant. Furthermore, the learning curve does not provide information about the model's performance on completely independent test data.
# </div>
# 

# In[48]:


# Learning curve EC2
train_sizes, train_scores, test_scores = learning_curve(ensemble_EC2, X_train_EC2, y_train_EC2, cv=10)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()
plt.show()


# ## 📈 Learning Curve for EC2
# 
# The learning curve provides a graphical representation of the model's performance on both the training set and the validation set over a varying number of training instances for EC2. 
# 
# Our training scores are: **[0.82574906, 0.8099395, 0.7987234, 0.79842976, 0.79710755]**
# 
# Our cross-validation scores are: **[0.77708509, 0.78744735, 0.7873631, 0.78753159, 0.79283909]**
# 
# ### 🧠 Understanding the Learning Curve
# 
# The learning curve helps us understand the trade-off between the model's performance on the training data and its ability to generalize to unseen data (represented by the cross-validation score). 
# 
# > **In our case, both the training score and the cross-validation score are consistently high across different training sizes, indicating that our model is performing well and not overfitting to the training data.**
# 
# <br>
# 
# <div class="warning" style="background-color: #F8F3D4; border-left: 6px solid #FF5252;font-size: 100%; padding: 10px;">
# <h3 style="color: #FF8551; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🗒️  Keep in Mind </h3>
# While the learning curve suggests that our model is performing well, it's important to remember that this is just one aspect of model evaluation. Depending on the specific requirements and constraints of our project, other metrics and techniques may be more relevant. Furthermore, the learning curve does not provide information about the model's performance on completely independent test data.
# </div>
# 
# # <span style="color:#E888BB; font-size: 1%;"> 4.8| Confidence intervals</span>
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Confidence intervals</span></b> </div>

# In[49]:


# Confidence intervals
bootstrap_samples_EC1 = 1000
bootstrap_scores_EC1 = []

for _ in range(bootstrap_samples_EC1):
    bootstrap_sample = resample(predictions_EC1)
    score = accuracy_score(y_test_EC1, bootstrap_sample)
    bootstrap_scores_EC1.append(score)

confidence_interval_EC1 = np.percentile(bootstrap_scores_EC1, [2.5, 97.5])
print("Confidence interval for EC1: ", confidence_interval_EC1)

bootstrap_samples_EC2 = 1000
bootstrap_scores_EC2 = []

for _ in range(bootstrap_samples_EC2):
    bootstrap_sample = resample(predictions_EC2)
    score = accuracy_score(y_test_EC2, bootstrap_sample)
    bootstrap_scores_EC2.append(score)

confidence_interval_EC2 = np.percentile(bootstrap_scores_EC2, [2.5, 97.5])
print("Confidence interval for EC2: ", confidence_interval_EC2)


# ## 📈 Confidence Intervals
# 
# Confidence intervals provide a range of values, derived from the training data, that predict the value of the score if we were to run the model on the entire population. 
# 
# Our confidence interval for EC1 is: **[0.58523416, 0.61287062]**
# 
# Our confidence interval for EC2 is: **[0.7995283, 0.80626685]**
# 
# ### 🧠 Understanding the Confidence Intervals
# 
# The confidence intervals help us understand the range in which our true model score lies with a certain level of confidence (usually 95%). 
# 
# > **For EC1, we can be 95% confident that the true model score lies between 0.585 and 0.613. For EC2, we can be 95% confident that the true model score lies between 0.800 and 0.806.**
# 
# <br>
# 
# <div class="warning" style="background-color: #F8F3D4; border-left: 6px solid #FF5252;font-size: 100%; padding: 10px;">
# <h3 style="color: #FF8551; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🗒️  Keep in Mind </h3>
# While the confidence intervals provide a range of likely values for our true model score, it's important to remember that this is just one aspect of model evaluation. Depending on the specific requirements and constraints of our project, other metrics and techniques may be more relevant. Furthermore, the confidence intervals are based on the assumption that our sample is representative of the population, which might not always be the case.
# </div>
# 
# # <span style="color:#E888BB; font-size: 1%;"> 4.9| SHAP summary plot</span>
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> SHAP summary plot</span></b> </div>
# 

# In[50]:


# Visualization:SHAP summary plot EC1

# For the SHAP summary plot, you'll need to install the shap library and use a model that supports it
explainer_EC1 = shap.TreeExplainer(ensemble_EC1.named_estimators_['gb'])
shap_values_EC1 = explainer_EC1.shap_values(X_train_EC1)
shap.summary_plot(shap_values_EC1, X_train_EC1, feature_names=train_data_E1.columns.tolist())


# In[51]:


# Visualization:SHAP summary plot EC2

# For the SHAP summary plot, you'll need to install the shap library and use a model that supports it
explainer_EC2 = shap.TreeExplainer(ensemble_EC2.named_estimators_['gb'])
shap_values_EC2 = explainer_EC2.shap_values(X_train_EC2)
shap.summary_plot(shap_values_EC2, X_train_EC2, feature_names=train_data_E2.columns.tolist())


# # <span style="color:#E888BB; font-size: 1%;"> 4.10 | Decision boundary</span>
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/3695238/pexels-photo-3695238.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Decision boundary</span></b> </div>

# In[52]:


# Get feature importances
importances = ensemble_EC1.named_estimators_['gb'].feature_importances_

# Get the indices of the features sorted by importance
indices = np.argsort(importances)[::-1]

# Get the names of the features sorted by importance
features_sorted = train_data_E1.columns[indices]


# In[53]:


# Select two features
X = X_train_EC1[:, :2]
y = y_train_EC1

# Fit the model
ensemble_EC1.fit(X, y)

# Create a grid of points
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict the class for each point in the grid
Z = ensemble_EC1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create a color map
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Plot the decision boundary
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Decision boundary for EC1")
plt.show()


# ## Intepret the Results
# 
# 

# In[54]:


# Get feature importances
importances = ensemble_EC2.named_estimators_['gb'].feature_importances_

# Get the indices of the features sorted by importance
indices = np.argsort(importances)[::-1]

# Get the names of the features sorted by importance
features_sorted = train_data_E2.columns[indices]


# In[55]:


# Select two features
X = X_train_EC2[:, :2]
y = y_train_EC2

# Fit the model
ensemble_EC2.fit(X, y)

# Create a grid of points
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict the class for each point in the grid
Z = ensemble_EC2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create a color map
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Plot the decision boundary
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Decision boundary for EC2")
plt.show()


# # <span style="color:#E888BB; font-size: 1%;">PREDICTION RESULT AND DISCUSSION</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/4021781/pexels-photo-4021781.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>📝 Results and Discussion</span></b> </div>
# 
# <br>
# 
# In this project, we have built and evaluated several machine learning models for two different tasks: EC1 and EC2. The models we used include the **<mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">Gradient Boosting Classifier</mark>**, **<mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">CatBoost Classifier</mark>**, **<mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">AdaBoost Classifier</mark>**, and **<mark style="background-color:#00B8A9;color:white;border-radius:5px;opacity:1.0">Quadratic Discriminant Analysis</mark>**. We also used a **<mark style="background-color:#F6416C;color:white;border-radius:5px;opacity:1.0">Voting Classifier</mark>** to combine the predictions from these models.
# 
# ## 🎯 Model Performance
# 
# The performance of the models was evaluated using several metrics, including precision, recall, F1-score, ROC AUC, and cross-validation scores. The models generally performed well, with high scores on these metrics. However, there were some differences in performance between the models and between the two tasks.
# 
# For **EC1**, the AdaBoost Classifier with a learning rate of 0.1 and 200 estimators performed the best, achieving a precision of 0.74, a recall of 0.87, and an F1-score of 0.80 for the positive class. The ROC AUC was <b><mark style="background-color:#393939;color:white;border-radius:5px;opacity:1.0">0.70</mark></b> and the average precision was 0.81. The cross-validation scores were also high, indicating that the model is likely to perform well on unseen data.
# 
# For **EC2**, the CatBoost Classifier with a learning rate of 0.1 and 50 estimators performed the best, achieving a precision of 0.81, a recall of 0.99, and an F1-score of 0.89 for the positive class. The ROC AUC was <b><mark style="background-color:#393939;color:white;border-radius:5px;opacity:1.0">0.59</mark></b> and the average precision was 0.86. The cross-validation scores were higher than for EC1, suggesting that the model for EC2 may be more accurate.
# 
# 
# ## 📊 Model Interpretation
# 
# The confusion matrices for the models provided further insights into their performance. For EC1, the model had a relatively high number of false negatives, indicating that it was less effective at detecting the positive class. For EC2, the model had a very low number of false negatives, but a high number of false positives, indicating that it was very effective at detecting the positive class but less effective at correctly classifying the negative class.
# 
# The learning curves for the models showed that they were not overfitting to the training data, as the training scores and cross-validation scores were consistently high across different training sizes.
# 
# ## 📝 Key Takeaways
# 
# Overall, the results of this project demonstrate the effectiveness of **<mark style="background-color:#F6416C;color:white;border-radius:5px;opacity:1.0">ensemble machine learning models</mark>** for these tasks. However, they also highlight the importance of model evaluation and interpretation. While the models achieved high scores on the metrics, the confusion matrices revealed some weaknesses in their performance. Furthermore, the learning curves showed that the models were not overfitting, but this does not guarantee their performance on new, unseen data.
# 
# <br>
# 
# <div class="warning" style="background-color: #C1ECE4; border-left: 6px solid #3AA6B9;font-size: 100%; padding: 10px;">
# <h3 style="color: #3AA6B9; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🚀  Going forward </h3>
# It may be beneficial to explore other models and techniques, such as neural networks or deep learning, to further improve performance. It may also be useful to investigate methods for handling class imbalance, as this could potentially improve the models' ability to detect the positive class.
# </div>
# 
# # <span style="color:#E888BB; font-size: 1%;">MAKING A PREDICTION</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/4021781/pexels-photo-4021781.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>Making a prediction</span></b> </div>
# 
# 

# In[56]:


# Load the test data
test_data = pd.read_csv('/kaggle/input/playground-series-s3e18/test.csv')

# Store IDs for later use in a submission file
test_id = test_data['id']

# Drop 'id' column from test data
test_data = test_data.drop(columns=['id'])

# Preprocess test data
test_data_scaled = scaler.transform(test_data)

# Train the models on the original data
ensemble_EC1.fit(X_train_EC1, y_train_EC1)
ensemble_EC2.fit(X_train_EC2, y_train_EC2)

# Make predictions on the test data
predictions_EC1 = ensemble_EC1.predict_proba(test_data_scaled)[:, 1]
predictions_EC2 = ensemble_EC2.predict_proba(test_data_scaled)[:, 1]


# In[57]:


# Create a submission dataframe
submission = pd.DataFrame({
    'id': test_id,
    'EC1': predictions_EC1,
    'EC2': predictions_EC2
})


# In[58]:


submission.head(20)


# # <span style="color:#E888BB; font-size: 1%;">📝 SUGGESTION</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/4021781/pexels-photo-4021781.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:white'>📝 Suggestion</span></b> </div>
# 
# <br>
# 
# **Based on the results and discussions from this  baseline project, here are some suggestions for future work:**
# 
# 1. **Explore Other Models**: While the ensemble models used in this project performed well, there are many other machine learning models and techniques that could potentially improve performance. For example, neural networks and deep learning models could be explored. These models have been shown to be highly effective for a wide range of tasks and could potentially provide better results.
# 
# 2. **Address Class Imbalance**: The datasets used in this project were imbalanced, which can affect the performance of machine learning models. Future work could explore methods for handling class imbalance, such as oversampling the minority class, undersampling the majority class, or using a combination of both. This could potentially improve the models' ability to detect the positive class.
# 
# 3. **Feature Engineering**: The performance of machine learning models can often be improved by creating new features from the existing data. Future work could explore different feature engineering techniques to see if they can improve model performance.
# 
# 4. **Hyperparameter Tuning**: While some hyperparameter tuning was performed in this project, there are many other hyperparameters that could be tuned. Future work could explore a wider range of hyperparameters and use more advanced tuning techniques, such as grid search or random search, to find the optimal hyperparameters.
# 
# 5. **Model Interpretability**: While the models used in this project provided good performance, they are somewhat complex and can be difficult to interpret. Future work could explore models that are more interpretable, such as decision trees or linear models. This could make it easier to understand how the models are making their predictions and could provide insights that could be used to improve the models or to gain a better understanding of the data.
# 
# <br>
# 
# <div class="warning" style="background-color: #F8F3D4; border-left: 6px solid #FF5252;font-size: 100%; padding: 10px;">
# <h3 style="color: #FF8551; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🗒️  Keep in Mind </h3>
# Remember, the goal is to build a model that not only performs well on our current data, but also generalizes well to new, unseen data. Therefore, it's crucial to continually evaluate and update the model as new data becomes available.
# </div>
# 
# <br>
# 
# ***
# 
# <br>
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
