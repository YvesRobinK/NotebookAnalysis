#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# **Hello again!** I'm back in a new notebook, tackling the **well-known and beginner-friendly Titanic Passenger Survival dataset**. This time around, my goal isn't to build models; instead, I'll be diving into Exploratory Data Analysis (EDA) and Feature Engineering. I'll provide explanations along the way.
# 
# I've actually already done a complete run-through, including modeling, tuning, and other processes, in a different notebook. However, I'm finding that it might be more effective to dedicate a separate notebook solely to EDA and Feature Engineering. This way, I can focus more intensely. I'm even considering the possibility of splitting EDA analysis and modeling in the future. This approach would make the notebook simpler and more specific in purpose.
# 
# 
# # EDA-Based Feature Engineering
# 
# Here's the plan: We'll start by **exploring the data** and **fixing any inconsistencies** we find. After that, we'll dive into **understanding the distribution of data** in each feature and uncover the **relationships between these features**. Our main focus will be on **how these features relate to the target** feature, which will be the heart of our future modeling process.
# 
# The goal of this is **to gain insights** from the data and **identify opportunities for performing feature engineering** on each data feature. This process can be described as EDA-based feature engineering. By conducting a thorough analysis, we'll likely discover various ways to enhance the data, ultimately providing a **significant boost to our model's performance** down the line. This thorough exploration is a key step to ensure we're leveraging the data effectively for our model's success.

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Load Data

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')

titanic_palette = ["#42424c", "#fba94b", "#ca4a3f", "#e0e0e0", "#88b7b5", "#0f8b8d", "#3f84e5", "#d0e3cc"]
sns.set_theme(style="white", palette=sns.color_palette(titanic_palette))


# In[4]:


df = pd.read_csv("/kaggle/input/titanic/train.csv")
df


# In[5]:


df.shape


# # Data Pre-analysis
# 
# Let's carefully examine the data to understand the nature of each feature. We need to identify whether a feature is categorical or numerical and determine its data type. Additionally, we should investigate if there are any features that contain a combination of different data types (mixed type). It's also crucial to locate missing values within the data and identify any duplicated entries. By conducting this thorough analysis, we can gain a better understanding of our dataset and ensure the integrity of our data.

# In[6]:


def summary(df):
    column = []
    d_type = []
    uniques = []
    n_uniques = []

    for col in df.columns:
        column.append(col)
        d_type.append(df[col].dtypes)
        uniques.append(df[col].unique()[:10])
        n_uniques.append(df[col].nunique())

    return pd.DataFrame({'Column': column, 'd_type': d_type, 'n_uniques': n_uniques, 'unique_sample': uniques})


# In[7]:


summary(df)


# ## Data Type
# 
# The training data consists of 12 columns, including 11 features and 1 target variable, with different data types:
# 
# * **Object**/String: Name, Sex, Ticket, Cabin, Embarked
# * **Float**/Real: Age and Fare
# * **Integer**: PassengerId, Survived (target), Pclass, SibSp, Parch
# 
# Let's focus on the Name, Ticket, and Cabin features. These features have an object data type and contain a large number of unique values. Typically, such features are unlikely to be useful for training machine learning models. However, we can observe that the Ticket and Cabin features have mixed alphanumeric data. The values in the **Name** and **Cabin** features may have **common patterns** that **could potentially be extracted and utilized in the modeling process**. On the other hand, the Ticket data does not provide specific information that we can derive from it. Therefore, in the next step, we will either drop or disregard this feature.

# ## Categorical and Numerical Features

# In[8]:


df.describe(include="O")


# Among the object/string data types, Name, Sex, Ticket, Cabin, and Embarked have the potential to be categorical features. However, it is advisable to consider only those object features with a **relatively small number of unique values as categorical features**. Features with a large number of unique values may not have a significant impact on the machine learning model. Therefore, for now, the categorical features included are only **Sex** and **Embarked**. These specific features are considered **nominal features**, indicating **non-ordered categorical variables**.
# 
# Regarding the other features, Name, Ticket, and Cabin, they have a large number of unique values. As a result, we may consider dropping these features or further analyzing them to determine their potential usefulness. It is worth noting that the Name feature is unique for each observation, which requires careful consideration during the analysis process.

# In[9]:


df.describe()


# The features mentioned above have numerical values:
# 
# * **PassengerId** is merely an ID and will be **dropped** later as it does not provide any meaningful information.
# * **Survived** is the **target** feature used for prediction.
# * **Pclass** is a categorical feature with ordered numerical values, also known as an **ordinal feature**.
# * **SibSp** and **Parch** are numerical features with discrete values. In this case, they **may also be considered as ordinal features**.
# * **Age** and **Fare** are both **numerical features with continuous values**. We can potentially **transform** these continuous numerical features **into ordinal features by binning** or grouping the values based on the data distribution and the relationship between the features and the target variable.

# ## Missing Value
# 
# Handling missing values is crucial to ensure that the model works with clean and complete data. The approach for handling missing values depends on the analysis results, and we can consider either dropping the missing values or imputing them with specific values.

# In[10]:


# display number of missing values
def count_missing_values(data):
    missing = data.isna().sum()
    df = pd.DataFrame({'Count':missing, 'Percentage':np.round(missing/len(data)*100, 2)})
    return df[df['Count'] > 0]

# plot missing values
def plot_missing_values(data):
    missing_df = count_missing_values(data)
    cmap = 'Greys' if len(missing_df) == 0 else 'Greys_r'

    plt.figure(figsize=(12,5))
    sns.heatmap(data.isna().transpose(), cmap=cmap, vmin=0, vmax=1);


# In[11]:


plot_missing_values(df)


# In[12]:


count_missing_values(df)


# In the training data, there are several features with missing values: Age, Cabin, and Embarked.
# * The **Embarked** feature has only a small number of missing values, so we can handle them by **imputing a specified value**.
# * The **Age** feature, on the other hand, has a larger proportion of missing values, accounting for **approximately 20%** of the data. One option is to drop this feature, but we could also **consider filling in the missing values** using an appropriate imputation method.
# * The **Cabin** feature has a significantly high percentage of missing values, **close to 78%**. For now, it might be best to **drop this feature**. However, we could explore filling in the missing values using certain methods in the future to potentially improve the model's accuracy.
# 
# In addition to the mentioned features, it's important to consider the possibility of missing values in other features within the test data. Although we are not checking them at the moment, **it is probable that the unseen data might contain missing values as well**. Therefore, it is crucial to **establish an appropriate imputation method for all features** that will be used in the modeling process. This ensures that our model can effectively handle missing values and make accurate predictions on new, unseen data.
# 
# **Attention!**
# For our ongoing analysis, we won't be tackling the missing values at this point. Instead, we'll handle data imputation during the preprocessing phase, before constructing a machine learning model. It's important to avoid filling in the missing data now, as it could impact the accuracy of our EDA analysis. Any filled data would be an estimate, not a concrete fact, and this could skew our insights.

# ## Duplicated Data
# 
# We need to check whether the train data contains any duplicate rows. If there are none, then we can proceed with the next step. However, if duplicates are found, we will need to drop them before continuing further.

# In[13]:


df[df.duplicated()]


# # Exploratory Data Analysis
# 
# ## Class Distribution
# 
# Examining the distribution of data in the target features offers valuable insights into potential data issues. For instance, we can determine whether we're dealing with a **classification or regression problem**. We can also identify if there's an **imbalance in the data** – a situation where one class heavily outweighs the others. In extreme cases of imbalance, it might even indicate a **detection problem**.
# 
# Addressing these questions **helps us select the appropriate methods and techniques for building our machine learning models**. By understanding the nature of the data and its characteristics, we can make informed decisions about how to proceed and ensure the success of our modeling endeavors.

# In[14]:


df.Survived.value_counts(normalize=True)


# Since the data isn't evenly divided between the classes (it's not a perfect 50:50 split), we can think of it as an "imbalance problem." However, the difference between the class amounts isn't too large (~24%). Also, it's not a situation where we're trying to detect something specific. So, we don't have to do complicated things like making the data more balanced through resampling. Instead, a simpler approach would be to give more weight to the classes that are less common during training. And when we divide the data later on, we should make sure to use a method that maintains the same class distribution as the original data (this is called "**stratify** mode").

# ## Data Distribution
# 
# Analyzing the data distribution is a valuable step in understanding the features better, as it allows us to **gain insights** and **determine the most effective preprocessing methods** before proceeding to the training phase. By examining the distribution of the data, we can uncover patterns, identify outliers, assess the skewness of the variables, and make informed decisions about data transformations or scaling techniques. This analysis aids in optimizing the data preprocessing steps and ensures that the data is prepared appropriately for the subsequent training phase.

# In[15]:


columns = ["Age", "Fare"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for col, ax in zip(columns, axes.flatten()):
    sns.histplot(x=col, data=df, kde=True, ax=ax)

plt.suptitle('Numerical Data Distribution')
plt.show();


# By examining the data distribution, we can gather valuable information about how the data is spread and identify the highest and lowest values within each feature. Specifically, for the numerical features Age and Fare, we observe a **right-skewed (positive-skew) distribution**, indicating that the **mean is greater than the median**. In such cases, considering the **median as the fill-in value** for missing data can help reduce the skewness and align the data distribution towards a more normal direction.
# 
# Here's another approach we can take: let's **transform the data** to make it look more like a normal distribution. One way to do this is by using a **power transformer**. This transformation can help our data become more balanced and easier for the model to work with.

# In[16]:


columns = ["Survived", "Pclass", "Sex", "SibSp", "Parch", "Embarked"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for col, ax in zip(columns, axes.flatten()):
    sns.countplot(x=col, data=df, ax=ax)

plt.suptitle('Categorical Data Distribution')
plt.show();


# Regarding categorical data, the distribution allows us to identify the highest and lowest scores within each category, as observed in the visualization. However, it is important to note that **this distribution alone is insufficient**. To gain more useful insights, we need to analyze how the passenger data is distributed by **considering their survival** status. Comparing the data distribution between survived and non-survived passengers enables us to uncover additional valuable insights and patterns that can inform our modeling and decision-making processes.

# ## Features by Target

# In[17]:


plt.figure(figsize=(12, 4))
sns.histplot(data=df, x='Age', hue='Survived', kde=True)
plt.title('Survival by Passenger Age');


# Based on the visualization, it appears that **young passengers in the baby or kid age range have a higher likelihood of survival** compared to passengers in other age ranges. We can further group these age values into specified ranges to observe the survival probability within each age group. This approach can facilitate classification and enhance interpretability, as we group ages based on common knowledge.

# In[18]:


plt.figure(figsize=(12, 4))
sns.histplot(data=df, x='Fare', hue='Survived', kde=True)
plt.title('Survival by Ticket Fare');


# Wow! There is a wide range of fares observed, with a majority of lower fare values. It seems that **higher fare values correlate with a higher probability of survival**, although it's important to note that correlation does not imply causation.

# In[19]:


plt.figure(figsize=(12, 4))
sns.histplot(data=df[df['Fare'] <= 100], x='Fare', hue='Survived', kde=True)
plt.title('Survival by Ticket Fare');


# Within the specific fare range of 0-100, **passengers with the cheapest tickets (Fare < 20) tend to have a lower chance of survival**, while those with **more expensive tickets (Fare > 50) exhibit a higher probability of survival**. However, for passengers with medium-fare tickets, it is difficult to draw definitive conclusions based on fare alone.

# In[20]:


columns = ["Survived", "Pclass", "Sex", "SibSp", "Parch", "Embarked"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for col, ax in zip(columns, axes.flatten()):
    sns.countplot(x=col, data=df, hue="Survived", ax=ax)


# Furthermore, the analysis reveals that a significant number of **unsurvived passengers** (500+ people) belonged to **Class 3**, were **male**, **traveled alone** without siblings/spouses or parents/children, or **embarked from Southampton (S port)**. Passengers falling into any of these categories generally have a lower chance of survival. On the other hand, passengers in **Class 1, females, or those traveling with a small number of siblings/spouses/parents/children are more likely to survive**.
# 
# The features related to siblings/spouses and parents/children show **similarities in distribution**. **Combining these features** to create a new feature could be explored to potentially improve the accuracy of the prediction.

# ## Correlation Analysis
# 
# Analyzing correlations can shed light on the connections between different features. or numerical features, a **Fcorrelation matrix** is often employed. This matrix shows the correlation coefficient between features, revealing how strongly and in which direction two numbers are related. If the coefficient is near 1, they rise together (positive correlation). If it's closer to 0, there's no clear link (weak correlation). Around -1 indicates an inverse relationship – as one goes up, the other goes down.
# 
# For non-numeric features, we rely on data visualization and Exploratory Data Analysis (EDA), similar to what we did previously. Alternatively, we can use the **chi-squared test** on pairs of categorical features. This test helps measure their association and calculate **Cramer's V**, which falls between 0 (no association) and 1 (complete association). This multi-pronged approach helps us gain deeper insights into the relationships within the data.

# In[21]:


def plot_correlation_matrix(data, columns, figsize=(8,4)):
    corr = data[columns].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap='RdGy_r', mask=mask, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Passenger Data (Numerics only)')
    plt.yticks(rotation=0)
    plt.show()


# In[22]:


num_cols = "Pclass Age SibSp Parch Fare Survived".split()
plot_correlation_matrix(df, num_cols)


# Based on the correlation matrix, the feature that **correlates the most with survivability is Pclass** (passenger class), followed by **Fare**. The correlation indicates that as the class number decreases (e.g., Pclass=1 representing a higher social class), and as the fare increases, the chances of survival also tend to increase. In other words, we can tentatively conclude that in this scenario, there is a correlation between a passenger's wealth and their likelihood of survival. However, it's important to note that this **correlation analysis does not establish a causal relationship**, and further investigation is needed.
# 
# Another interesting correlation is observed **between SibSp (number of siblings/spouses) and Parch (number of parents/children)**. These features show a correlation. It suggests the **possibility of combining these features** to create a new feature that captures the overall family size or presence. Let's see what will happen. Exploring this combination may yield valuable insights.

# In[23]:


from scipy.stats import chi2_contingency

def plot_categorical_correlation_matrix(data, columns, figsize=(8, 4)):
    corr = pd.DataFrame(index=columns, columns=columns, dtype=np.float64)
    
    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                corr.loc[col1, col2] = 1.0
            else:
                cross_tab = pd.crosstab(data[col1], data[col2])
                chi2 = chi2_contingency(cross_tab)[0]
                min_dim = min(cross_tab.shape) - 1
                corr.loc[col1, col2] = np.sqrt(chi2 / (chi2 + data.shape[0] * min_dim))
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap='Reds', mask=mask, vmin=0, vmax=1)
    plt.title('Categorical Association Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.show()


# In[24]:


cat_cols = ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch', 'Survived']
plot_categorical_correlation_matrix(df, cat_cols)


# So, what if we want to explore how all the features relate to each other, not just looking at the separation between categorical and numerical features?
# 
# I've got an idea that might not be totally valid, but it can give us a hint about the feature relationships. There are two approaches:
# 
# * First, we could convert all features into numerical values (encoded categorical features). After that, we can use a correlation matrix to see the relationships.
# 
# * The second approach involves converting all features into categorical values (binning numerical features) and then using an association matrix. However, please note that this method isn't ideal due to its limitations. It's more of an illustration, as treating categorical and numerical features the same way isn't fully reliable.

# In[25]:


drop_cols = ['PassengerId', 'Name', 'Cabin', 'Ticket']

# encode the categorical features then show correlaion matrix for all features (now all numerics)
df2 = pd.get_dummies(df.drop(columns=drop_cols), columns=['Sex', 'Embarked'])
plot_correlation_matrix(df2, df2.columns, figsize=(12,6))


# From the visual representation above, we can glean several insights based on the correlations among data features. Let's break them down:
# 
# * **Passenger Safety Levels**: Features like Pclass, Sex, and Fare exhibit higher correlations compared to others. This suggests that they might be stronger predictors of passenger safety. Specifically, females, passengers with higher Fare tickets, and those with lower Pclass codes (indicating a higher social class) tend to have a better chance of survival. Interestingly, passengers who embarked from Cherbourg, France (C port) have a higher likelihood of survival, whereas those from Southampton, England (S port) are more likely not to survive. However, the correlation between Queenstown, Ireland (Q port) and passenger safety is less clear.
# 
# * **C Port and Survival**: The higher survival rate among passengers from the C port might be attributed to the fact that many affluent or socially higher-status individuals departed from there. The data does show a significant correlation between C embarked, Pclass, and Fare, which supports this relationship. The visualization suggests that passengers from port C often belong to lower Pclass codes (higher social classes) and purchase higher-fare tickets, potentially contributing to their improved safety. It's essential to note that this is a factual observation, not a statement of causality.
# 
# * **Fare and Pclass Relationship**: A notable and intriguing observation is the inverse correlation between Fare and Pclass. In other words, passengers with lower Pclass codes tend to have higher Fare tickets, indicating that higher social classes pay more for their tickets.
# 
# * **Age and Passenger Safety**: The correlation between age and passenger safety appears to be weak, which might impact predictions related to safety.
# 
# * **SibSp and Parch**: As seen in the distribution analysis stage, there is a correlation between SibSp (sibling/spouse count) and Parch (parent/child count), highlighting their interconnectedness.
# 
# These insights offer valuable clues about feature relationships and can guide us as we proceed with our analysis and potential feature engineering. Keep in mind that these observations are based on correlations within the data and don't imply causal relationships.

# # Feature Engineering
# 
# Upon revisiting our previous data analysis, we've identified some tweaks that could potentially enhance our model's accuracy. These adjustments involve **categorizing numeric features** like Age and Fare, where we group values into different ranges. This grouping transforms continuous numbers into categorical or ordinal values, helping the model grasp the underlying data patterns better.
# 
# Our next move is to **extract new features based on existing ones** by pulling out common values. For instance, we can extract Titles from the Name feature and Decks from the Cabin feature. Additionally, we can **combine features** such as SibSp and Parch into a single feature called Family.
# 
# By implementing these changes, our goal is to supercharge our model's predictive capability and improve its accuracy in capturing the subtle intricacies of the dataset. We're on a mission to boost our model's performance!

# In[26]:


# new dataframe for results of feature engineering
df_fe = df.copy()


# ## Categorizing Age
# 
# To ensure reliability and accuracy, we will create **age groups based on the original age data** before performing any imputation. This approach avoids potential issues that may arise from using artificially generated data during the imputation process.

# In[27]:


plt.figure(figsize=(14,4))
sns.histplot(data=df_fe, x='Age', hue='Survived', bins=40, kde=True);


# In[28]:


# group age based on bins division
bins = [-np.Inf, 1, 6, 14, 19, 55, np.Inf]
labels = ["Baby","Toddler","Kid","Teenage","GrownUp","Elder"]
df_fe["Age_Cat"] = pd.cut(x=df_fe["Age"], bins=bins, labels=labels, include_lowest=True)


# In[29]:


sns.countplot(x="Age_Cat", data=df_fe, hue="Survived");


# ## Categorizing Fare

# In[30]:


plt.figure(figsize=(14,4))
sns.histplot(data=df_fe, x='Fare', hue='Survived', bins=40, kde=True);


# In[31]:


bins = [-np.Inf, 20, 50, np.Inf]
labels = ["Low","Medium","High"]
df_fe["Fare_Cat"] = pd.cut(x=df_fe["Fare"], bins=bins, labels=labels, include_lowest=True)


# In[32]:


sns.countplot(x="Fare_Cat", data=df_fe, hue="Survived");


# ## Extracting Title from Name

# In[33]:


df_fe["Name"]


# In[34]:


def find_title(name):
    return (name.split(", ")[1]).split(".")[0]

df_fe["Name"].apply(find_title).value_counts()


# Given the numerous titles present in the dataset, we will focus on extracting the most significant titles: **Mr**, **Miss**, **Mrs**, and **Master**. These titles are widely recognized and can provide meaningful insights into the passenger's characteristics. However, for the remaining less common titles, we will group them together and categorize them as "**Others**." This approach allows us to simplify the analysis while still capturing the important information conveyed by the passengers' titles.

# In[35]:


def extract_title(name):
    title = find_title(name)
    return title if title in "Mr Miss Mrs Master".split(" ") else "Others"

df_fe["Title"] = df_fe["Name"].apply(extract_title)
df_fe.head()


# ## Extracting Deck Group from Cabin Code

# In[36]:


def find_deck(cabin):
    return np.nan if cabin is np.nan else cabin[0]

df_fe["Deck"] = df_fe["Cabin"].apply(find_deck)
df_fe["Deck"].value_counts()


# Considering the significant number of decks identified, it is important to address the missing values in the Cabin column and determine an appropriate approach for filling them. Additionally, it would be beneficial to reduce the number of deck groups for easier analysis.
# 
# To explore potential groupings, we can start by **examining if the decks can be grouped based on the passenger class (Pclass)** feature. By analyzing the relationship between Pclass and the available deck information, we can identify patterns and potentially assign the missing deck values based on the corresponding passenger class.

# In[37]:


deck_div = df_fe[["Deck", "Pclass"]].copy()
deck_div.drop_duplicates().sort_values(["Deck","Pclass"])


# How is the passenger deck accommodation based on Pclass? This is roughly how it is:
# 
# * Deck **A,B,C** are occupied by Pclass **1** --> grouped into Deck **ABC**
# * Deck **D** is occupied by Pclass **1,2** --> grouped into Deck **DEF**
# * Deck **E** is occupied by Pclass **1,2,3** --> grouped into Deck **DEF**
# * Deck **F** is occupied by Pclass **2,3** --> grouped into Deck **DEF**
# * Deck **G** is occupied by Pclass **3** --> stays the same
# * Deck **T** is occupied by Pclass **1** --> moved into Deck **ABC**, since it has similar properties
# 
# Here we can't divide the passengers' deck based on their Pclass only. From the correlation analysis before we knew that **Pclass is correlated with Fare**. So let's analyze more about deck divisions using Fare features!
# 
# But before that, we'll try to do feature engineering to Fare feature by banding the values into categoric.

# In[38]:


deck_div = df_fe[["Deck", "Pclass", "Fare", "Fare_Cat"]].copy()
deck_div["count"] = deck_div.groupby(["Deck"])["Deck"].transform("count")
deck_div[~pd.isna(deck_div["Deck"])].drop(columns=["Fare"]).drop_duplicates().sort_values(["Deck","Pclass", "Fare_Cat"])


# In[39]:


plt.figure(figsize=(14,4))
sns.scatterplot(data=df_fe[(df_fe["Fare"] < 250)].sort_values(by=["Deck"]),
                x="Fare", y="Deck", hue="Pclass", palette=titanic_palette[:3]);


# Next, how are we going to split up these decks?
# 
# Based on grouping table and the scatter plot above, It looks like it would be better if we group the Decks by Pclass. Fare also can be used to determine the overlapping class. This will also determine how we will **fill in the missing data**. Here is the final strategy that we're going to use for determining **passengers' Deck, depends on Pclass and Fare** categories:
# 
# * Pclass **1** will be categorized to Deck **ABC**
# * Pclass **2** will be categorized to Deck **DEF**
# * Pclass **3** with **High** and **Medium** ticket fare will be categorized to Deck **DEF**
# * Pclass **3** with **Low** ticket fare will be categorized to Deck **G**

# In[40]:


def regroup_deck(deck):
    if deck in ["A", "B", "C", "T"]:
        return "ABC"
    elif deck in ["D", "E", "F"]:
        return "DEF"
    else:
        return deck

def impute_deck(data):
    if data["Deck"] is np.nan:
        if data["Pclass"] == 1:
            return "ABC"
        elif data["Pclass"] == 2:
            return "DEF"
        elif data["Pclass"] == 3:
            if data["Fare_Cat"] == "Low":
                return "G"
            else:
                return "DEF"
    else:
        return data["Deck"]


# In[41]:


# reorganized Deck group
df_fe["Deck"] = df_fe["Deck"].apply(regroup_deck)

# filling missing Deck
df_fe["Deck"] = df_fe.apply(impute_deck, axis=1)


# In[42]:


plt.figure(figsize=(14,4))
sns.scatterplot(data=df_fe[(df_fe["Fare"] < 250)].sort_values(by=["Deck"]),
                x="Fare", y="Deck", hue="Pclass", palette=titanic_palette[:3]);


# ## Combining SibSp and Parch into Family
# 
# Here we're going to combine two features by addition resulting new feature Family, the total number of family sailed together with each passenger.

# In[43]:


def count_family(data):
    return data["SibSp"] + data["Parch"] + 1

df_fe["Family"] = df_fe.apply(count_family, axis=1)


# In[44]:


df_fe.head()


# ## Categorizing Family Size
# 
# Let's group the number of family members into categories. By grouping the family sizes, we can examine if there are any patterns or trends that influence the survival outcomes.

# In[45]:


plt.figure(figsize=(8,4))
sns.histplot(data=df_fe, x='Family', hue='Survived', kde=True);


# In[46]:


# group family based on bins division
bins = [-np.Inf, 1, 4, np.Inf]
labels = ["Single","Small","Big"]
df_fe["Family_Cat"] = pd.cut(x=df_fe["Family"], bins=bins, labels=labels)


# In[47]:


df_fe.head()


# In[48]:


sns.countplot(x="Family_Cat", data=df_fe, hue="Survived");


# # Save The Modified Data
# 
# In comparing our initial data to the current state after feature engineering, it's clear that we've **added new dimensions to the dataset**. Looking ahead, binning categorical data could potentially replace numerical data in creating a classification model. **Different machine learning algorithms might respond variably to this change** – some might show improved accuracy, while others could see a decrease.
# 
# For now, we'll consolidate everything into the same dataframe to create a fresh dataset stemming from our EDA-based Feature Engineering. This dataset will serve as a foundation for our subsequent modeling efforts. Important to mention is that our data **still contains missing values**. As we move forward, we'll explore various approaches during preprocessing to address this issue. One consideration is performing imputation within the data pipeline to prevent any data leakage. This careful handling will ensure the integrity and accuracy of our analysis and modeling.

# In[49]:


df.head()


# In[50]:


df_fe.head()


# In[51]:


df_fe.to_csv('train_fe.csv', index=False)


# # Conclusion
# 
# * **Purpose and Scope**: This report delved into the Titanic passenger dataset, focusing on exploratory data analysis (EDA) and feature engineering. Our aim was to improve the accuracy of our machine learning model's predictions for passenger survival.
# 
# * **Key Findings**: Our analysis uncovered insights into data characteristics, feature distributions, correlations, and their impact on survival. We identified opportunities to categorize continuous features like Age and Fare, leading to enhanced predictive capabilities.
# 
# * **New Features**: By extracting Title and Deck information from existing features, we added valuable dimensions to our dataset. Combining related features, such as SibSp and Parch, into a Family feature allowed us to understand the influence of family size on survival.
# 
# * **Next Steps**: While this report focused on EDA and feature engineering, our next phase involves training machine learning models on this enriched dataset. We anticipate notable improvements in model performance and prediction accuracy.
# 
# * **Future Enhancements**: There's room to enhance our analysis further by incorporating domain expertise to create custom features. Exploring feature selection techniques can help identify the most impactful feature subset for modeling.
