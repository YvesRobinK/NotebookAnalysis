#!/usr/bin/env python
# coding: utf-8

# # 10 Step Exploratory Data Analysis (EDA)

# # 1. Importing Libraries and Loading Data:

# The first step is to import the necessary Python libraries, such as Pandas, NumPy, and Matplotlib.

# In[1]:


# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mstats
import scipy.stats as stats
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler


# Next, load the dataset into a Pandas DataFrame, which allows for easy manipulation and analysis of the data.

# In[2]:


# Load the dataset
data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')


# # 2. Data Inspection:

# We start by gaining a preliminary understanding of the data by inspecting its structure. We use DataFrame methods like head(), shape(), info(), and describe() to view a sample of the data, check the data types, and get summary statistics.

# In[3]:


# View the first few rows of the data
pd.set_option('display.max_columns', None) # This is so we can view all columns
data.head()


# In[4]:


# Get the shape of the data (number of rows, number of columns)
print(data.shape)


# In[5]:


# Get information about the columns and data types
data.info()


# In[6]:


# Get summary statistics of the numerical columns
data.describe()


# In[7]:


# Assign categorical and numeric columns based on data types

categorical_cols = data.select_dtypes(include='object')

numeric_cols = data.select_dtypes(include=['int', 'float']).columns


# In[8]:


# Check the unique values in each categorical column
for col in categorical_cols:
    print(data[col].unique())


# # 3. Finding & Handling Missing Data:

# Next we identify and handle missing data, as they can impact the analysis. We use methods like isnull(), fillna(), or dropna() to handle missing values appropriately based on the context of the data. We can also impute the mean or median values for numeric data, or the mode for categorical data.

# In[9]:


# Check for missing values in the dataset
pd.set_option('display.max_rows', None) # This is so we can view all rows

data.isnull().sum()




# In[10]:


# Handle missing values based on the context of the data
data['LotFrontage'].fillna(data['LotFrontage'].mean(), inplace=True)
data['MasVnrArea'].fillna(0, inplace=True)
data['GarageYrBlt'].fillna(data['GarageYrBlt'].median(), inplace=True)
data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)


# In[11]:


object_columns_with_missing_values = data.select_dtypes(include='object').columns[data.select_dtypes(include='object').isnull().any()]
object_columns_with_missing_values


# In[12]:


# Iterate over object columns with missing values
for column in object_columns_with_missing_values:
    mode_value = data[column].mode().iloc[0]  # Compute the mode for the column
    
    # Replace missing values with the mode
    data[column].fillna(mode_value, inplace=True)


# In[13]:


# Drop columns with more than 70% missing data
threshold = 0.7  # Set the threshold for missing data percentage
missing_data_percentage = data.isnull().mean()  # Compute the missing data percentage for each column
columns_to_drop = missing_data_percentage[missing_data_percentage > threshold].index  # Get columns with missing data above the threshold
data.drop(columns_to_drop, axis=1, inplace=True)  # Drop the columns with more than 70% missing data


# In[14]:


data.isnull().sum()


# In[15]:


# If we have to, we can drop the rest of the rows with missing values
data.dropna(inplace=True)


# # 4. Data Cleaning and Preprocessing:

# We further clean the data by addressing inconsistencies, outliers, and irrelevant information. We perform tasks like identifying and removing duplicates, correcting data types, standardizing values, and transforming data when required.

# In[16]:


# Check for duplicates in the dataset
duplicates = data.duplicated()
print("Number of duplicates:", duplicates.sum())


# In[17]:


# Show the duplicate rows (none here)
duplicate_rows = data[duplicates]
print("Duplicate rows:")
print(duplicate_rows)


# In[18]:


# If there had been any duplicates, this is how to remove them
data.drop_duplicates(inplace=True)


# In[19]:


# # Convert data types
# data['column_name'] = data['column_name'].astype('int')


# # 5. Data Visualization:

# Visualization plays a vital role in EDA. Python offers a plethora of libraries for creating insightful visualizations. We can make use of Matplotlib, Seaborn, or Plotly to generate various plots, including histograms, scatter plots, box plots, and bar charts. Visualizations can reveal patterns, trends, and relationships within the data.

# In[20]:


#  Histogram
plt.hist(data['SalePrice'], bins=50)
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.title('Histogram of Sale Price')
plt.show()




# In[21]:


sns.boxplot(x=data['OverallQual'], y=data['SalePrice'])
plt.xlabel('Overall Quality')
plt.ylabel('Sale Price')
plt.title('Box Plot of Sale Price by Overall Quality')
plt.show()


# In[22]:


# Scatter Plot
plt.scatter(data['GrLivArea'], data['SalePrice'])
plt.xlabel('Above Ground Living Area (sqft)')
plt.ylabel('Sale Price')
plt.title('Scatter Plot: Above Ground Living Area vs. Sale Price')
plt.show()


# # 6. Univariate Analysis:

# Analyze individual variables through univariate analysis. Explore the distribution of numerical variables using measures like mean, median, and standard deviation. For categorical variables, examine their frequencies and proportions through frequency tables, bar charts, or pie charts.

# In[23]:


# Kernel Density Estimation (KDE) plot
sns.kdeplot(data['GrLivArea'], fill=True)
plt.xlabel('Above Ground Living Area (sqft)')
plt.ylabel('Density')
plt.title('KDE Plot: Above Ground Living Area')
plt.show()




# In[24]:


# Box Plot with outliers shown
plt.boxplot(data['GrLivArea'], showfliers=True)
plt.xlabel('Above Ground Living Area (sqft)')
plt.title('Box Plot with Outliers: Above Ground Living Area')
plt.show()


# # 7. Multivariate Analysis:

# Investigate relationships between two or more variables using multi-variate analysis. Explore correlations between numerical variables using scatter plots or correlation matrices. For categorical variables, use contingency tables and stacked bar charts to understand associations and dependencies. Utilize techniques like heatmaps, pair plots, and parallel coordinates to analyze relationships among multiple variables. Cluster analysis can help identify natural groupings within the data.

# In[25]:


# Multivariate Analysis
plt.bar(data['MSZoning'].value_counts().index, data['MSZoning'].value_counts().values)
plt.xlabel('MSZoning')
plt.ylabel('Count')
plt.title('Bar Chart of MSZoning')
plt.show()


# In[26]:


# Box Plot with different colors for each category
sns.boxplot(x='MSZoning', y='GrLivArea', data=data, hue='MSZoning')
plt.xlabel('MSZoning')
plt.ylabel('Above Ground Living Area (sqft)')
plt.title('Box Plot with Different Colors: MSZoning vs. Above Ground Living Area')
plt.show()


# In[27]:


# Swarm Plot
sns.swarmplot(x='MSZoning', y='GrLivArea', data=data)
plt.xlabel('MSZoning')
plt.ylabel('Above Ground Living Area (sqft)')
plt.title('Swarm Plot: MSZoning vs. Above Ground Living Area')
plt.show()


# In[28]:


sns.stripplot(x='MSZoning', y='GrLivArea', data=data)
plt.xlabel('MSZoning')
plt.ylabel('Above Ground Living Area (sqft)')
plt.title('Stripplot: MSZoning vs. Above Ground Living Area')
plt.show()


# In[29]:


# Violin plots
sns.violinplot(x='MSZoning', y='GrLivArea', data=data)
plt.xlabel('MSZoning')
plt.ylabel('Above Ground Living Area (sqft)')
plt.title('Violin Plot: MSZoning vs. Above Ground Living Area')
plt.show()


# In[30]:


# Correlation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))  # Adjust the values to increase or decrease the figure size
plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Correlation Matrix')
plt.show()




# In[31]:


# Stacked Bar Chart
cross_tab = pd.crosstab(data['MSZoning'], data['SaleCondition'])
cross_tab.plot(kind='bar', stacked=True)
plt.xlabel('MSZoning')
plt.ylabel('Count')
plt.title('Stacked Bar Chart: MSZoning vs. SaleCondition')
plt.show()


# In[32]:


sns.scatterplot(x='LotArea', y='SalePrice', data=data)
plt.xlabel('Lot Area')
plt.ylabel('Sale Price')
plt.title('Scatter Plot of Lot Area vs. Sale Price')
plt.show()


# In[33]:


# Heatmap
plt.figure(figsize=(10, 8))  # Adjust the values to increase or decrease the figure size
sns.heatmap(data.corr(), annot=False, cmap='coolwarm')
plt.title('Heatmap')
plt.show()


# In[34]:


# # Pair Plot
# sns.pairplot(data[numeric_cols])
# plt.title('Pair Plot')
# plt.show()


# In[35]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')

# Select the three variables for the 3D scatter plot
x = data['LotArea']
y = data['OverallQual']
z = data['SalePrice']

ax.scatter(x, y, z)

ax.set_xlabel('Lot Area')
ax.set_ylabel('Overall Quality')
ax.set_zlabel('Sale Price')

plt.title('3D Scatter Plot: Lot Area, Overall Quality, Sale Price')
plt.show()


# # 8. Feature Engineering:

# Create new variables or transform existing ones to extract additional information. Feature engineering techniques include creating dummy variables, scaling features, deriving new variables from existing ones, and encoding categorical variables.

# In[36]:


# Feature Engineering
data['TotalBath'] = data['FullBath'] + data['HalfBath']
data['HasGarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)


# In[37]:


# from sklearn.preprocessing import StandardScaler

# # Select numeric columns for scaling
# numeric_columns = data.select_dtypes(include='number').columns

# # Create a StandardScaler object
# scaler = StandardScaler()

# # Scale the numeric columns
# data[numeric_columns] = scaler.fit_transform(data[numeric_columns])


# In[38]:


# # We can also Normalize Features using MinMaxScaler. This transforms the values between 0 and 1. Do either Standardization (zscoring) OR Normalization, NOT both.

# # Select numeric columns for scaling
# numeric_columns = data.select_dtypes(include='number').columns

# scaler = MinMaxScaler()
# data[numeric_columns] = scaler.fit_transform(data[numeric_columns])


# In[39]:


# Select categorical columns for encoding
categorical_columns = data.select_dtypes(include='object').columns

# Perform one-hot encoding
encoded_data = pd.get_dummies(data, columns=categorical_columns)

# # Print the encoded data
# print(encoded_data)


# # 9. Statistical Analysis:

# Conduct statistical tests to derive meaningful insights. Perform tests such as t-tests, chi-square tests, ANOVA, or regression analysis to validate hypotheses or identify significant differences between groups.

# In[40]:


group1 = data[data['OverallQual'] >= 7]['SalePrice']
group2 = data[data['OverallQual'] < 7]['SalePrice']
t_statistic, p_value = ttest_ind(group1, group2)
print('T-Statistic:', t_statistic)
print('P-Value:', p_value)




# In[41]:


# Chi-Square Test
chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(data['MSZoning'], data['CentralAir']))
print('Chi-Square:', chi2)
print('P-Value:', p)


# # 10. Outlier Detection:

# Identify outliers, which can impact the analysis and models. Use statistical techniques like z-score, box plots, or IQR (interquartile range) to detect and handle outliers appropriately. Winsorizing is one way to deal with outliers.

# In[42]:


columns_to_check = ['LotArea', 'OverallQual', 'YearBuilt', 'GrLivArea']

plt.figure(figsize=(10, 8))  # Adjust the figure size if needed

for i, column in enumerate(columns_to_check):
    plt.subplot(2, 2, i+1)  # Create subplots for each column
    sns.boxplot(data[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)

plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# In[43]:


# Handling outliers by winsorizing
from scipy.stats import mstats
data['LotArea'] = mstats.winsorize(data['LotArea'], limits=[0.05, 0.05])


# In[44]:


sns.boxplot(data['LotArea'])
plt.xlabel('Lot Area')
plt.title('Box Plot of Lot Area')
plt.show()

