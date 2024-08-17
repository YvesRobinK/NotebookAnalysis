#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing


# # Introduction
# 
# This is a fun competition. Repeating the Titanic dataset artificially. I wrote this notebook without reading anybody else's work so I would start from a fresh position with fresh ideas. This will be my baseline to progress further with. 

# # Feature Engineering
# 
# The aim here is to deal with the object columns and make them some form of numerical so that we can apply ML models to them. 

# **Load Data**

# In[2]:


import numpy as np 
import pandas as pd 

df = pd.read_csv("../input/spaceship-titanic/train.csv")
test_df = pd.read_csv("../input/spaceship-titanic/test.csv")
df.dtypes


# **Detect NaNs in Dataset**

# In[3]:


def detect_NaNs(df_temp): 
    print('NaNs in data: ', df_temp.isnull().sum().sum())
    print('******')
    count_nulls = df_temp.isnull().sum().sum()
    if count_nulls > 0:
        for col in df_temp.columns:
            print('NaNs in', col + ": ", df_temp[col].isnull().sum().sum())
    print('******')
    print('')
detect_NaNs(df)
detect_NaNs(test_df)


# **Change Transported to 0 and 1**

# In[4]:


df["Transported"] = df["Transported"].astype(int)


# **Detect Duplicates**

# In[5]:


def detect_duplicates(df_temp): 
    print('Duplicates in data: ', df.duplicated().sum())
    return df.duplicated().sum()
detect_duplicates(df)


# **Split Passenger Id**
# 
# Passenger Id looks like it might have a class at the end. Let's get that class seperated

# In[6]:


def seperate_passenger_id(df_temp):
    passenger_class = []
    for idx, row in df_temp.iterrows():
        passengerid = str(row["PassengerId"])
        if "_" in passengerid:
            passenger_class.append(int(passengerid.split("_")[1]))
        else:
            passenger_class.append(0)
    df_temp["Passenger Class"] = passenger_class
    return df_temp
df = seperate_passenger_id(df)
test_df = seperate_passenger_id(test_df)


# **Cabin Details Seperated**

# In[7]:


def seperate_cabin(df_temp):
    letters = []
    numbers = []
    final_letters = []
    for idx, row in df_temp.iterrows():
        cabin = str(row["Cabin"])
        if "/" in cabin:
            letters.append(cabin.split("/")[0])
            numbers.append(cabin.split("/")[1])
            final_letters.append(cabin.split("/")[2])
        else:
            letters.append(None)
            numbers.append(-1)
            final_letters.append(None)
    df_temp["letters"] = letters
    df_temp["numbers"] = numbers
    df_temp["final_letters"] = final_letters
    return df_temp
df = seperate_cabin(df)
test_df = seperate_cabin(test_df)
df = df.drop(columns="Cabin")
test_df = test_df.drop(columns="Cabin")


# In[8]:


df["numbers"] = pd.to_numeric(df["numbers"], errors = 'ignore')
test_df["numbers"] = pd.to_numeric(test_df["numbers"], errors = 'ignore')


# In[9]:


df.dtypes


# **Gender from Name**
# 
# The gender could be important. So I will attempt to classify the gender based on name.

# In[10]:


get_ipython().system('pip install gender_guesser')


# In[11]:


import gender_guesser.detector as gender
def predict_gender(df):
    d = gender.Detector()
    gender_predicted = []
    for idx, row in df.iterrows():
        name = str(row["Name"])
        if " " in name:
            predicted = d.get_gender(name.split(" ")[0])
            if predicted == "mostly_male":
                predicted = "male"
            elif predicted == "mostly_female":
                predicted = "female"
            gender_predicted.append(predicted)
        else:
            gender_predicted.append("unknown")
    df["gender"] = gender_predicted
    df = pd.get_dummies(df, columns = ["gender"])
    return df

df = predict_gender(df)
test_df = predict_gender(test_df)


# It appears some names have been obscured by adding a last letter to them so I will go through and reattempt to classify with its removal

# In[12]:


import gender_guesser.detector as gender
def predict_gender_remove_last_letter(df):
    d = gender.Detector()
    gender_predicted = []
    for idx, row in df.iterrows():
        if row["gender"] == "unknown":
            name = str(row["Name"])
            if " " in name:
                predicted = d.get_gender(name.split(" ")[0][:-1])
                if predicted == "mostly_male":
                    predicted = "male"
                elif predicted == "mostly_female":
                    predicted = "female"
                gender_predicted.append(predicted)
            else:
                gender_predicted.append("unknown")
        else:
            gender_predicted.append(row["gender"])
    df["gender"] = gender_predicted
    df = pd.get_dummies(df, columns = ["gender"])
    return df

# df = predict_gender_remove_last_letter(df)
# test_df = predict_gender_remove_last_letter(test_df)


# **Change the last names**
# 
# The last name could be useful for us, since it would imply families

# In[13]:


def last_names(df):
    Last_Names = []
    for idx, row in df.iterrows():
        name = str(row["Name"])
        if " " in name:
            Last_Names.append(name.split(" ")[-1])
        else:
            Last_Names.append(None)
    df["Name"] = Last_Names
    return df
df = last_names(df)
test_df = last_names(test_df)


# **Count Number of Family Members On Ship**

# In[14]:


df_temp = pd.concat([df.copy(), test_df.copy()], ignore_index=True)
df_temp['Num_Family_Members'] = df_temp.groupby(['Name'])['PassengerId'].transform('nunique')
df['Num_Family_Members'] = df_temp['Num_Family_Members'][:8693].values
test_df['Num_Family_Members'] = df_temp['Num_Family_Members'][8693:].values


# **Remove the PassengerId Column**
# 
# It has no predictive power and will naturally lead to overfitting

# In[15]:


df = df.drop(columns=["PassengerId"])
test_df = test_df.drop(columns=["PassengerId"])


# **Encode columns to one hot or numerical encoding based on number of unique values**

# In[16]:


def encode_columns(df, columns, test_df = None):
    for col in columns:
        le = preprocessing.LabelEncoder()
        le.fit(df[col].astype(str))
        if len(le.classes_) < 6:
            df = pd.get_dummies(df, columns = [col])
            if test_df is not None:
                test_df = pd.get_dummies(test_df, columns = [col])
        else:
            check_col = df.copy()[col]
            df[col] = le.transform(df[col].astype(str))
            if test_df is not None:
                #Clean out unseen labels
                inputs = []
                for idx, row in test_df.iterrows():
                    if row[col] in pd.unique(check_col):
                        inputs.append(row[col])
                    else:
                        inputs.append(None)
                test_df[col] = inputs
                test_df[col] = le.transform(test_df[col].astype(str))
    return df, test_df
#encode_columns(df, ["HomePlanet", "CryoSleep", "Destination", "VIP", "Name", "letters", "final_letters"], test_df)
df, test_df = encode_columns(df, ["HomePlanet", "CryoSleep", "Destination", "VIP", "Name", "letters", "final_letters"], test_df)


# **test the crossover for the name column**

# In[17]:


test_df["Name"].nunique()


# **test the crossover for the letters column**

# In[18]:


test_df["letters"].nunique()


# **Fill in NaNs**
# 
# I will also record where there was a NaN in case that proves useful

# In[19]:


Age_Recorded = []
def fillna_create_column(df_temp, columns, value = 0):
    """
    Fill na of provided columns and create columns to signify they weren't there
    """
    for col in columns:
        temp_col = []
        for idx, row in df_temp.iterrows():
            if row[col] != row[col]:
                temp_col.append(0)
            else:
                temp_col.append(1)
        df_temp[col + "_exists"] = temp_col
        df_temp[col] = df_temp[col].fillna(0)
    return(df_temp)
df = fillna_create_column(df, ["Age","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck","Num_Family_Members"])
test_df = fillna_create_column(test_df, ["Age","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck","Num_Family_Members"])


# Check that there are NaNs still in data

# In[20]:


def detect_NaNs(df_temp): 
    print('NaNs in data: ', df_temp.isnull().sum().sum())
    count_nulls = df_temp.isnull().sum().sum()
    if count_nulls > 0:
        print('******')
        for col in df_temp.columns:
            print('NaNs in', col + ": ", df_temp[col].isnull().sum().sum())
        print('******')
    print('')
detect_NaNs(df)
detect_NaNs(test_df)


# **Create Interactions**

# In[21]:


import itertools
def create_interactions(df_temp, column_list):
    # Cross wise interactions
    for x in itertools.combinations(column_list, 2):
        df_temp[x[0]+"+"+x[1]] = df_temp[x[0]]+df_temp[x[1]]
    # Iterative Totals
    iterative_total = 0
    i = 0
    for j in (column_list):
        iterative_total = iterative_total + df_temp[j]
        if i > 0:
            df_temp["A" + str(i) + "_iter_score"] = iterative_total
        i = i + 1
    return df_temp
df = create_interactions(df, ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"])
test_df = create_interactions(test_df, ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"])


# Rename the auto generated total to an easier to read column name

# In[22]:


df["TotalSpend"] = df["A4_iter_score"]
df = df.drop(columns="A4_iter_score")
test_df["TotalSpend"] = test_df["A4_iter_score"]
test_df = test_df.drop(columns="A4_iter_score")


# Older people tend to have more money so it would be good to divide TotalSpend by Age

# In[23]:


def spend_by_age(df_temp):
    spending_by_age = []
    for idx, row in df_temp.iterrows():
        if row["Age"] != 0:
            spending_by_age.append((row["TotalSpend"] / row["Age"]))
        else:
            spending_by_age.append(0)    
    return spending_by_age
df["spending_by_age"] = spend_by_age(df)
test_df["spending_by_age"] = spend_by_age(test_df)


# In[24]:


def create_interactions_based_on_total(df_temp, column_list, total_col_name):
    """
    Determine ratio of columns based on a total
    """
    # Cross wise interactions
    for j in (column_list):
        df_temp[j + " per " + total_col_name] = df_temp[j] / df_temp[total_col_name]
        df_temp[j + " per " + total_col_name] = df_temp[j + " per " + total_col_name].replace([np.inf, -np.inf], np.nan)
        df_temp[j + " per " + total_col_name] = df_temp[j + " per " + total_col_name].fillna(0)
    
    return df_temp
df = create_interactions_based_on_total(df, ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"], "TotalSpend")
test_df = create_interactions_based_on_total(test_df, ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"], "TotalSpend")


# In[25]:


df = create_interactions_based_on_total(df, ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"], "Age")
test_df = create_interactions_based_on_total(test_df, ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"], "Age")


# **Anomaly Detection**
# 
# We can assign scores to each row to determine if it is an outlier.

# In[26]:


get_ipython().system('pip install pycaret')


# In[27]:


df_temp = pd.concat([df.copy(), test_df.copy()], ignore_index=True)
df_temp = df_temp.drop(columns="Transported")


# In[28]:


import pycaret.anomaly as anomaly
anomaly.setup(df_temp, session_id = 123, silent=True)
display()


# In[29]:


iforest = anomaly.create_model('iforest')


# In[30]:


iforest_results = anomaly.assign_model(iforest)
iforest_results.head()


# Visualise whether the anomalys appear to present real outliers by performing PCA

# In[31]:


from yellowbrick.features import PCA as yellowPCA

y = iforest_results["Anomaly"]
X = iforest_results.drop(columns=["Anomaly"])

visualizer = yellowPCA(scale=True, projection=2, alpha=0.4)
visualizer.fit_transform(X, y)
visualizer.show()


# We can see here that outliers have been pretty well chosen by the method and therefore this is useful information. So I will add it to the dataframe.

# In[32]:


df["Anomaly"] = iforest_results["Anomaly"][:8693].values
df["Anomaly_Score"] = iforest_results["Anomaly_Score"][:8693].values
test_df["Anomaly"] = iforest_results["Anomaly"][8693:].values
test_df["Anomaly_Score"] = iforest_results["Anomaly_Score"][8693:].values


# **Add PCA Features**
# 
# The above graph shows that PCA with anomaly could be used to make difficult classification decisions.

# In[33]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

df_temp = pd.concat([df.copy(), test_df.copy()], ignore_index=True)
y = df_temp["Transported"]
X = df_temp.drop(columns="Transported", axis=1)
X_scaled = MinMaxScaler().fit_transform(X)
pca = PCA(n_components=3)
X_p = pca.fit(X_scaled).transform(X_scaled)
df["PCA_0"] = X_p[:8693,0]
df["PCA_1"] = X_p[:8693,1]
df["PCA_2"] = X_p[:8693,2]
test_df["PCA_0"] = X_p[8693:,0]
test_df["PCA_1"] = X_p[8693:,1]
test_df["PCA_2"] = X_p[8693:,2]


# **Reduce Memory Usage**
# 
# Reducing memory usage can speed things up.

# In[34]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[35]:


df = reduce_mem_usage(df)
test_df = reduce_mem_usage(test_df)


# # EDA

# **Check Class Balance**

# In[36]:


def class_balance(df_temp, target_col):
    sns.countplot(x=df_temp[target_col])
    column_values = df_temp[target_col].values.ravel()
    unique_values = pd.unique(column_values)
    unique_values = np.sort(unique_values)
    for value in unique_values:
        print(value,":",(len(df_temp.loc[df_temp[target_col] == value]) / len(df_temp)) * 100, "%")
class_balance(df, "Transported")


# **Plot Data**

# In[37]:


pltdf = df.copy()
pltdf = pltdf.sample(frac=1, random_state=42).reset_index(drop=True)
pltdf.iloc[:50, :28].plot(subplots=True, layout=(7,4), figsize=(15,10))

plt.show()


# **Pivot Table**

# In[38]:


def pivot_table(df_temp, target_col):
    df_temp = df_temp.copy()
    y = df_temp[target_col]
    X = df_temp.drop(columns=target_col, axis=1)
    X_scaled = MinMaxScaler().fit_transform(X)
    df_temp = pd.DataFrame(X_scaled)  
    df_temp.columns = X.columns
    df_temp[target_col] = y
    table = pd.pivot_table(data=df_temp,index=[target_col]).T
    sns.heatmap(table, annot=True, cmap="Blues")
    return table


# In[39]:


plt.figure(figsize=(10,15))
table = pivot_table(df, "Transported")


# Cryo sleep present quite a divergence

# **See Important Correlations**

# In[40]:


from sklearn import preprocessing

def calculate_correlations(df_temp, target_col, ratio, verbose=1):
    df_temp = df_temp.copy()
    cols = []
    cols_done = []
    if df_temp[target_col].dtype == object:
        le = preprocessing.LabelEncoder()
        df_temp[target_col] = le.fit_transform(df_temp[target_col])
    df_temp[target_col] = MinMaxScaler().fit_transform(df_temp[target_col].values.reshape(-1, 1))
    if verbose == 1:
        print("Correlations with",target_col + ":")
    for col_one in df_temp.iloc[:,:].columns:
        correlation_value =  abs(df_temp[col_one].corr(df_temp[target_col]))
        if verbose == 1:
            print(col_one, ":", df_temp[col_one].corr(df_temp[target_col]))
        if correlation_value > ratio:
            cols.append(col_one)
        cols_done.append(col_one)
    corrdf = df_temp.copy()
    corrdf = corrdf[cols].corr()
    sns.heatmap(abs(corrdf), cmap="Blues")
    return cols


# In[41]:


transported = df.pop('Transported')
df["Transported"] = transported
correlation_cols = calculate_correlations(df, "Transported", 0.2, 0)


# **Plot Bar Plots**

# In[42]:


def bar_plots(df, columns_to_plot, target_col):
    df_temp = df.copy()
    fig, axs = plt.subplots(len(columns_to_plot), 1, figsize=(10, 15))
    i = 0 
    for col in columns_to_plot:
        sns.barplot(ax=axs[i], x=target_col, y=col, data=df)# .set_title(col + " X " + target_col)
        axs[i].set_title = "test"
        i = i + 1
    # fig.subplots_adjust(hspace=0.4)
bar_plots(df, ["RoomService", "Spa", "VRDeck", "CryoSleep_True", "spending_by_age", "gender_female"], "Transported")


# As I believe was the case in the original titanic. It does gender matters. If it is possible a better gender classification would potentially help. 
# 
# Spending by age also appears to be pretty significant. 

# **Swarm Plot**

# In[43]:


df_temp = df.copy()
sns.swarmplot(x="variable", y="value", data=pd.melt(df_temp[["Age"]][:400]))


# **Pair Grid**

# In[44]:


def pair_grid_plot(df, cols):
    g = sns.PairGrid(df[cols].iloc[:500,:], diag_sharey=False)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)
    
pair_grid_plot(df, ["RoomService", "Spa", "VRDeck", "CryoSleep_True", "Transported"])


# **Dimensionality Reduction**

# In[45]:


from sklearn.decomposition import PCA

def pca_dimension_reduction_info(df_temp, target_col):
    df_temp = df_temp.copy()
    y = df_temp[target_col]
    X = df_temp.drop(columns=target_col, axis=1)
    X_scaled = MinMaxScaler().fit_transform(X)
    print(str(len(X_scaled[0])) + " initial feature components")
    pca = PCA(n_components=0.95)
    X_p = pca.fit(X_scaled).transform(X_scaled)
    print("95% variance explained by " + str(len(X_p[0])) + " components by principle component analysis")
    pca = PCA(n_components=3)
    X_p = pca.fit(X_scaled).transform(X_scaled)
    print(str(round(pca.explained_variance_ratio_.sum() * 100)) + "% variance explained by 3 components by principle component analysis")
    pca = PCA(n_components=2)
    X_p = pca.fit(X_scaled).transform(X_scaled)
    print(str(round(pca.explained_variance_ratio_.sum() * 100)) + "% variance explained by 2 components by principle component analysis")

pca_dimension_reduction_info(df.drop(columns=["PCA_0","PCA_1","PCA_2"]), "Transported")    


# In[46]:


from yellowbrick.features import PCA as yellowPCA

y = df["Transported"]
X = df.drop(columns=["Transported","PCA_0","PCA_1","PCA_2"])

visualizer = yellowPCA(scale=True, projection=2, alpha=0.4)
visualizer.fit_transform(X, y)
visualizer.show()
visualizer = yellowPCA(scale=True, projection=3, alpha=0.4, size=(700,700))
visualizer.fit_transform(X, y)
visualizer.show()


# **Decision Tree**
# 
# Simple decision trees can be easily interpreted for knowledge about the data

# In[47]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

def decision_tree(df_temp, depth, target_col):
    tree_set = df_temp.copy()
    target = tree_set[target_col]
    tree_set.drop([target_col], axis=1, inplace=True)
    tree_clf = DecisionTreeClassifier(max_depth=depth, random_state=1)
    tree_clf.fit(tree_set, target)
    text_representation = tree.export_text(tree_clf, feature_names=tree_set.columns.tolist())
    print("accuracy: " + str(tree_clf.score(tree_set, target)))    
    plt.figure(figsize=(18,18))
    # tree.plot_tree(tree_clf, feature_names=tree_set.columns, filled=True)
    class_column_values = df_temp[target_col].values.ravel()
    class_unique_values = pd.unique(class_column_values)
    class_unique_values = np.sort(class_unique_values)
    class_unique_values = class_unique_values.astype('str')
    le = preprocessing.LabelEncoder()
    target = le.fit_transform(target)
    dot_data = tree.export_graphviz(tree_clf, out_file=None, 
                                    feature_names=tree_set.columns,  
                                    class_names=class_unique_values,
                                    filled=True)
    display(graphviz.Source(dot_data, format="png")) 


# In[48]:


decision_tree(df, 2, "Transported")


# It appears that if someone stayed in Cryosleep they were less likely to be Transported. Spending may be a product of them being more active on the ship. 

# # Clean Up
# 
# Some of the dataset columns could stand to be cut away. Experiments have made me feel some features are not proving to be helpful for now. 

# In[49]:


pycaret_df = df.drop(columns=["gender_andy","gender_unknown", "gender_male"])
pycaret_test_df = test_df.drop(columns=["gender_andy","gender_unknown", "gender_male"])


# # Pycaret
# 
# I will go through here and determine the best model to classify on this dataset. I have excluded some models to keep the time to run this notebook on kaggle under control.

# In[50]:


from pycaret.classification import *
from sklearn import preprocessing

setup(data = pycaret_df.copy(), 
             target = "Transported",
             silent = True, normalize = True, session_id=1, data_split_stratify=True, categorical_features=["Passenger Class"])
display()


# In[51]:


top3 = compare_models(n_select=3, exclude=["xgboost","catboost","gbc","lr"])


# In[52]:


lightgbm = create_model("lightgbm")


# In[53]:


ensemble = ensemble_model(lightgbm, n_estimators=2)


# In[54]:


ensemble = finalize_model(ensemble)


# In[55]:


plot_model(ensemble, "confusion_matrix")


# In[56]:


plot_model(ensemble, "error")


# In[57]:


plot_model(ensemble, "boundary")


# # Create Submission

# In[58]:


predictions = predict_model(ensemble, data=pycaret_test_df)


# In[59]:


predictions.head(25)


# In[60]:


predictions["Label"] = predictions["Label"].astype(bool)


# In[61]:


submission = pd.read_csv("../input/spaceship-titanic/sample_submission.csv")
submission["Transported"] = predictions["Label"]
submission.to_csv("submission.csv", index=False)

