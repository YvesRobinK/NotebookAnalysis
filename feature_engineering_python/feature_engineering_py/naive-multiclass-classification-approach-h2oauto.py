#!/usr/bin/env python
# coding: utf-8

# # Problem Statement: To analyze the data and predict the status of the animals when they leave the welfare center

# The Animal Welfare Center (AWC) is one of the oldest animal shelters in the United States that provide care and shelter to over 15,000 animals each year. To boost its effort to help and care for animals in need, the organization makes available its accumulated data and statistics as part of its Open Data Initiative. The data contains information about the intake and discharge of animals entering the Animal Welfare Center from the beginning of October 2013 to the present day.
# 
# 
# The AWC wants to make use of this data to help uncover useful insights that have the potential to save these animals’ lives. To make better decisions in the future regarding animal safety, AWC wants to analyze this data and predict the status of the animals when they leave the welfare center.

# **Problem Type**- Multiclass classification
# 
# **Chosen workflow Stage-**
# 
# 1) Loading required libraries
# 
# 2) Loading the data
# 
# 3) Missing Value check
# 
# 4) Quick EDA and data cleaning
# 
# 5) Analyse, identify patterns, explore the data
# 
# 6) Feature Engineering
# 
# 7) Encoding the categorical data
# 
# 8) Modelling, tuning and prediction(Multiple iterations and subject to changes)
# 
# 9) Submission of results

# ** EVALUATION METRIC: f1-score**

# **DATA PREPROCESSING**

# # 1. Loading required libraries
# 
# I have used general libraries like pandas, malplotlib  for assembling, cleaning the data and most importantly libraries containing all the models

# In[1]:


#LOADING REQUIRED LIBRARIES
#Data Visualization libraries
import pandas as pd
import seaborn as sns
import numpy as np

#Machine learning classification model libraries
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')


#Accuracy Check
from sklearn.model_selection import cross_val_score

#Encoders
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Scalers
from sklearn.preprocessing import StandardScaler

#Metrices import
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Importing train, test split library
from sklearn.model_selection import train_test_split

# Libraries to test model on different thresholds
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#Metrices import
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
from numpy import loadtxt
from numpy import sort

#Import warnings
import warnings
warnings.filterwarnings("ignore")

#Importing custom packages
from sklearn.base import TransformerMixin

#For proper display of all columns
from IPython.display import display
pd.options.display.max_columns = None


# ** Defining some customer functions for specific use**

# In[205]:


#Imputing the remaining missing variables
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, A, b=None):

        self.fill = pd.Series([A[c].value_counts().index[0]
            if A[c].dtype == np.dtype('O') else A[c].mean() for c in A],
            index=A.columns)

        return self

    def transform(self, A, b=None):
        return A.fillna(self.fill)


# # 2. Loading the dataset

# Lets have a quick look at the dataset:

# In[234]:


ml_dataset_train= pd.read_csv('...input/Animal State Prediction - dataset/train.csv')
ml_dataset_test= pd.read_csv('...input/Animal State Prediction - dataset/test.csv')


# In[235]:


#Drawing a count plot of dependent variable to check the biasness of prediction in data present
ax = ml_dataset_train['outcome_type'].value_counts().plot(kind='bar',figsize=(10,5),title="Frequency of each outcome type")
ax.set_xlabel("Outcome Type")
ax.set_ylabel("Frequency")


# In[236]:


print("Percent Class Distribution")
print(ml_dataset_train['outcome_type'].value_counts(normalize= True)*100)


# Data is highly biased. We need to select our train and test split carefully.

# In[237]:


#Loading the dataset
X= pd.read_csv('/Users/apple/Documents/Animal_State_Prediction_dataset/train.csv')


# # 3. Missing Value Check

# In[238]:


#Missing data check
total = X.isnull().sum().sort_values(ascending=False)
percent = (X.isnull().sum()/X.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent * 100], axis=1, keys=['Total', 'Percent(%)'])
missing_data.head(5)


# Hence we will be dropping outcome_datetime as more than 15% that i selected as a threshold is missing.
# 
# ** "outcome_datetime" needs to be dropped remaining two missing values we can impute**

# In[239]:


X = X.drop(['outcome_datetime'], axis  = 1)


# In[240]:


X = DataFrameImputer().fit_transform(pd.DataFrame(X))


# In[241]:


print('Missing value in training set:', X.isnull().sum().max())


# # 4. Quick EDA and data cleaning

# **Description of Numeric variable:**

# In[242]:


display(X.describe())


# Couple of quick observations:
# 
# 1. The variable **"count"** is having std 0 and min and max 1, implies it is a constant column having digit 1 hence we will be **dropping** this variable.
# 
# 2. Also the statistics of **"intake_number"** and **"outcome_number"** looks similar hence lets quickly check and take a call.

# In[243]:


qck_check= pd.DataFrame(X.groupby(['intake_number', 'outcome_number']).size())
display(qck_check)


# This implies **"intake_number"** and **"outcome_number"** are **duplicate columns** hence we will **drop** anyone of the variable, **lets drop "outcome_number"** .

# Applying quick eda observations i.e. to drop outcome_number and count column:

# In[244]:


X = X.drop(['count', 'outcome_number'], axis = 1)
X.shape


# **Description of Categorical variable**:

# In[245]:


X.describe(include = ['O'])


# Looking at the above dataset it looks like **"animal_id_outcome"** is not unique. Maybe same animals were repeated brought to the shelter for some or the other reasons. Anyways we will be dropping it as it makes very little or no help in our model predictions.

# In[247]:


X = X.drop(['animal_id_outcome'], axis = 1)


# # 5. Analyse, identify patterns, explore the data

# Next step would be to analyse, identify and explore the data to take actions based upon the understanding.

# **A. Few quick points to be noted:**
# 1. age_upon_intake_(days), age_upon_intake_(years)/age_group are extracted from age_upon_intake and there is no point using all of them. Hence we will keep age_upon_intake_(days) as it is more specific. Hence **dropping age_upon_intake**, **age_upon_intake_(years), age_upon_intake_age_group** similary **dropping age_upon_outcome, age_upon_outcome_(years), age_upon_outcome_age_group
# 
# 2. dob_year, dob_month is extracted from date_of_birth hence will be using dob_year to be more specific and ** dropping date_of_birth ** and **dob_month**  as they are useless for our case.
# 3. According to my research once the animal or pet is surrendender by owner the animal/pet cannot be or vary rarely returned back to owner again with some conditions hence **intake_type** is an **important predictor**.
# 4. animal_type and breed type is specific to the category of animal hence are correlated. We need to take a call whether to use both or use one of them according to the varibale importance curve
# 5. time_in_shelter_(days) is derived from time_in_shelter hence will be **dropping time_in_shelter** and keeping **time_in_shelter_(days)**
# 

# In[248]:


X = X.drop(['age_upon_intake', 'age_upon_intake_(years)', 'age_upon_intake_age_group', 'age_upon_outcome', 'age_upon_outcome_(years)', 'age_upon_outcome_age_group', 'date_of_birth','dob_month', 'time_in_shelter'], axis = 1)


# **B. Checking categorical variables correlation with outcome_type**

# In[250]:


sns.countplot(x="animal_type", hue="outcome_type", data= X)


# In[251]:


X_Bird = X[X['animal_type']=='Bird']


# In[252]:


sns.countplot(x= 'animal_type', hue="outcome_type", data= X_Bird)


# In[253]:


X_Other= X[X['animal_type']=='Other']
sns.countplot(x= 'animal_type', hue="outcome_type", data= X_Other)


# In[254]:


#Checking animal_type with outcome_type
X.groupby(["breed", 'outcome_type']).size().head(20)


# Hence we get this insights that dogs are mostly adopted and returned back to owner, cats are either mostly been adopted or transfered, on the other hand bird are either adopted or transfered and are rarely "returned back to owner"(which has high occurence in the dataset). Last other animals are mostly euthonised or transfered.
# 
# **NOTE** We will be holding color variable as it is as there more than 400 unique colors and will be checking later by variable importance plot.

# In[255]:


#Checking intake-condition 
X.groupby(['intake_type','outcome_type']).size()


# Hence this along with intake_condition can be useful for model to help predict minority of the dependent variable such as died, disposal, missing and relocate hence our hypothesis of these variable to be important for model is proving true.

# In[256]:


X.groupby(['sex_upon_intake','sex_upon_outcome', 'outcome_type']).size()


# Hence we will be keeping on i.e for now as we can clearly see Intact male/female who were first intact and later neutered/spayed have more chances of adoption about 20 folds. hence this can prove to be an important variable.

# One more observation is that we will be dropping intake_datetime as we already have intake month, year

# In[257]:


X['intake_day'] = pd.DatetimeIndex(X['intake_datetime']).day


# **Note: The above is a new feature that is generated**

# In[258]:


X = X.drop(['intake_datetime'], axis = 1)


# **Feature modification of intake/outcome year month:**

# In[ ]:


X['intake_monthyear'] = X['intake_year'] * 100 + X['intake_month']
X['outcome_monthyear'] = X['outcome_year']* 100 + X['outcome_month']


# My hypothesis regarding outcomeWeekdays should be that there will more adoptions on sundays and saturdays as people have holiday and they can visit the facility for official adoption. I dont think that intakeWeekdays will have much significance to our prediction, we will check and discard it later if required.

# In[260]:


sns.countplot(x= 'outcome_weekday', hue="outcome_type", data= X)


# The results are as expected there are relatively more adoptions on saturday and sundays. Hence we will keep these variables for now and test it in our variable importance plot.

# In[261]:


pyplot.figure(figsize=(15,5))
sns.boxplot(X['outcome_type'], X['intake_number'])


# Higher number of intake_number has more chances of returning to owner, adoption and missing. Hence an important variable

# Lets have a quick check on variable importance and model performance so that we can tweak it accordingly.

# ** Converting the data into training and validation set **

# # Feature encoding(initial iteration)

# In[263]:


X_check = X.copy()


# In[264]:


def onehotdataframe(X_check):    
    one_hot_animal_type=pd.get_dummies(X_check.animal_type)

    one_hot_intake_condition = pd.get_dummies(X_check.intake_condition)
    one_hot_intake_condition.columns = [str(col) + '_intake' for col in one_hot_intake_condition.columns]

    one_hot_intake_type = pd.get_dummies(X_check.intake_type)
    one_hot_intake_type.columns = [str(col) + '_intake' for col in one_hot_intake_type.columns]

    one_hot_sex_upon_intake = pd.get_dummies(X_check.sex_upon_intake)
    one_hot_sex_upon_intake.columns = [str(col) + '_intake' for col in one_hot_sex_upon_intake.columns]

    one_hot_intake_weekday = pd.get_dummies(X_check.intake_weekday)
    one_hot_intake_weekday.columns = [str(col) + '_intake' for col in one_hot_intake_weekday.columns]

    one_hot_sex_upon_outcome = pd.get_dummies(X_check.sex_upon_outcome)
    one_hot_sex_upon_outcome.columns = [str(col) + '_outcome' for col in one_hot_sex_upon_outcome.columns]

    one_hot_outcome_weekday = pd.get_dummies(X_check.outcome_weekday)
    one_hot_outcome_weekday.columns = [str(col) + '_outcome' for col in one_hot_outcome_weekday.columns]     

    X_check = X_check.drop(['animal_type','intake_type', 'intake_condition', 'sex_upon_intake', 'intake_weekday', 'sex_upon_outcome', 'outcome_weekday'], axis = 1) 

    #Merging one hot encoded features with our dataset 'data' 
    X_check=pd.concat([X_check, one_hot_animal_type,one_hot_intake_condition,one_hot_intake_type,one_hot_sex_upon_intake,one_hot_intake_weekday,one_hot_sex_upon_outcome,one_hot_outcome_weekday],axis=1)  
    
    return(X_check)


# In[265]:


X_check = onehotdataframe(X_check)
X_check['outcome_type'] = X_check['outcome_type'].map({'Adoption' : 8, 'Transfer': 7, 'Return to Owner' : 6, 'Euthanasia' : 5, 'Died' : 4, 'Missing' : 3, 'Relocate' : 2, 'Rto-Adopt' : 1, 'Disposal' : 0}) 
y_check = X_check['outcome_type']
X_check= X_check.drop(['outcome_type'], axis = 1)


# ** Applying label encoding: **

# In[106]:


le1= LabelEncoder()
le2= LabelEncoder()

le1.fit(X_check['breed'])
#print(le1.classes_)

X_check['breed'] = le1.transform(X_check['breed'])

le2.fit(X_check['color'])
#print(le2.classes_)

X_check['color'] = le2.transform(X_check['color'])


# In[107]:


# Splitting the dataset into the Training set and Test set.
X_train_check, X_val_check, y_train_check, y_val_check = train_test_split(X_check, y_check, test_size = 0.2, random_state = 20)
#Split is 80%-20% for testing. Here X_test~Validation set.


print(X_train_check.shape, X_val_check.shape, y_train_check.shape, y_val_check.shape)


# In[108]:


#Drawing a count plot of dependent variable to check the biasness of prediction in data present
bx = pd.DataFrame(y_train_check)['outcome_type'].value_counts().plot(kind='bar',figsize=(10,5),title="Frequency of each outcome type")
bx.set_xlabel("Outcome Type")
bx.set_ylabel("Frequency")


# In[109]:


print("Class distribution count")
print(pd.DataFrame(y_train_check)['outcome_type'].value_counts())
print("Class distribution percent(%)")
print(pd.DataFrame(y_val_check)['outcome_type'].value_counts(normalize = True)*100)


# This implies we are good to go with this training data set as it is showing same stats as that of combined dataset.

# In[110]:


y_train_check.unique()


# # Modelling(Initial iterations)

# ** Starting with lightgbm test**

# ** Test 1**

# In[61]:


import lightgbm as lgb


# In[62]:


check_train_data=lgb.Dataset(X_train_check,label=y_train_check)


# In[63]:


params = {'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class':9,
    'metric': 'multi_logloss',
    'learning_rate': 0.0023,
    'max_depth': 7,
    'num_leaves': 16,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.6,
    'bagging_freq': 17 }


# In[64]:


lgb_cv = lgb.cv(params, check_train_data, num_boost_round=10000, nfold=3, shuffle=True, stratified=True, verbose_eval=20, early_stopping_rounds=100)

nround = lgb_cv['multi_logloss-mean'].index(np.min(lgb_cv['multi_logloss-mean']))
print(nround)

model = lgb.train(params, check_train_data, num_boost_round=nround)


# In[65]:


check_y_pred = model.predict(X_val_check)


# In[70]:


best_preds_check = np.asarray([np.argmax(line) for line in check_y_pred])


# In[72]:


from sklearn.metrics import f1_score
f1_score(best_preds_check, y_val_check , average = 'micro')


# ** Test 2**

# In[80]:


params2 = {'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class':9,
    'metric': 'multi_logloss',
    'learning_rate': 0.0023,
    'max_depth': 7,
    'num_leaves': 34,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.6,
    'bagging_freq': 17 }


# In[81]:


lgb_cv = lgb.cv(params2, check_train_data, num_boost_round=10000, nfold=3, shuffle=True, stratified=True, verbose_eval=20, early_stopping_rounds=200)

nround2 = lgb_cv['multi_logloss-mean'].index(np.min(lgb_cv['multi_logloss-mean']))
print(nround)

model2 = lgb.train(params, check_train_data, num_boost_round=nround2)


# In[82]:


check_y_pred2 = model2.predict(X_val_check)


# In[83]:


check_best_pred2 = np.asarray([np.argmax(line) for line in check_y_pred2])


# In[84]:


from sklearn.metrics import f1_score
f1_score(check_best_pred2, y_val_check , average = 'micro')


# **Test 3**

# In[ ]:


model3 = XGBClassifier(learning_rate =0.1,n_estimators=250,max_depth=5 ,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'multi:softmax',nthread=4, scale_pos_weight=1, seed=1234)


# In[97]:


model3.fit(X_train_check, y_train_check, eval_metric = 'mlogloss')


# In[75]:


check_y_pred3 = model3.predict(X_val_check)


# In[78]:


from sklearn.metrics import f1_score
f1_score(check_y_pred3, y_val_check , average = 'micro')


# **Test 4**

# In[91]:


model4 = XGBClassifier(learning_rate =0.05,n_estimators=1000,max_depth=5 ,min_child_weight=1, gamma=5, subsample=0.8, colsample_bytree=0.8,objective= 'multi:softmax',nthread=4, scale_pos_weight=1, seed=1234)


# In[92]:


model4.fit(X_train_check, y_train_check, eval_metric = 'mlogloss')


# In[93]:


check_y_pred4 = model4.predict(X_val_check)


# In[130]:


from sklearn.metrics import f1_score
f1_score(check_y_pred4, y_val_check.iloc[:] , average = 'micro')


# Model3 has max f1 score , now lets plot the feature importance curve

# In[104]:


# plot feature importance
from xgboost import plot_importance
plot_importance(model3, max_num_features= 30)
pyplot.figure(figsize=(20,10))
pyplot.show()


# In[113]:


modelcheck(model=model3, X_train_encoded=X_train_check,X_test_encoded = X_val_check,y_train = y_train_check, y_test = y_val_check, model_with_param = XGBClassifier(learning_rate =0.1,n_estimators=250,max_depth=5 ,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'multi:softmax',nthread=4, scale_pos_weight=1, seed=1234))


# This clearly gives an idea that at n=42 we achieve maximun accuracy. Hence we need to discard more variables for better prediction in the future.
# 
# But we need to be very careful as some of the less important feature can contribute in prediction of minority class.

# In[122]:


sorted_idx = np.argsort(model3.feature_importances_)[::-1]
for index in sorted_idx:
    print([X_train_check.columns[index], model3.feature_importances_[index]]) 


# We need to drop some of the weak predictor variable to tweak our model performance also we need to create some more variables for better performance of model. As per our hypothesis we will be eliminating outcome_year as it is a weak predictor.

# Note i performed the above check to quickly drop some useless predefined variables. I will be dropping the lower predicting dummy variables later. Hence dropping **outcome_year**

# In[ ]:


X = X.drop(['outcome_year'], axis  = 1)


# # 6. Feature Engineering:

# Lets try to create some more informative features
# 

# In[268]:


import re
def breed_check(breed):
    r1 = re.search(r'\/', breed)
    if r1:
        check = str(r1.group())
    else:
        check = 'not_found'
    r2 = re.search(r'Mix', breed)
    if r2:
        check2 = str(r2.group())
    else:
        check2 = 'not_found'
    if check=='/':
        breed_type = 'specific_mix' # AS NOT MENTIONED IN PROBLEM STATEMENT THIS IS WHAT MY ASSUMPTION IS.
    elif check2 == 'Mix':
        breed_type = 'unknown_mix'
    else:
        breed_type = 'pure'
    return breed_type   


# In[269]:


X['breed_bucket'] = X.breed.apply(breed_check)


# In[270]:


sns.countplot(x= 'breed_bucket', hue="outcome_type", data= X)


# This gives us an idea that that unknown mix bread type are either more adopted or transfered, pure bread type animal is either 

# In[271]:


#Trying to create some strengthing variable for the minority class
def adopt_conditions(df):
    if (df['outcome_type']=='Adoption'):      #made changes here removed !=Others
        return df['breed']
    
adoptable_dict = pd.DataFrame(X.apply(adopt_conditions, axis = 1), columns = ['adoptable_breeds'])
cleaned_adoptable_dict = adoptable_dict[adoptable_dict['adoptable_breeds'].notnull()]
cleaned_adoptable_dict = pd.DataFrame(cleaned_adoptable_dict['adoptable_breeds'].unique(), columns = ['unique_adoptable_breeds'])
adoptable_list = cleaned_adoptable_dict['unique_adoptable_breeds'].tolist()


# In[273]:


#Trying to create some strengthing variable for the minority class
def rto_adopt_conditions(df):
    if (df['outcome_type']== 'Rto-Adopt'):      #made changes here removed !=Others
        return df['breed']
    
rto_adoptable_dict = pd.DataFrame(X.apply(rto_adopt_conditions, axis = 1), columns = ['rto_adoptable_breeds'])
rto_cleaned_adoptable_dict = rto_adoptable_dict[rto_adoptable_dict['rto_adoptable_breeds'].notnull()]
rto_cleaned_adoptable_dict = pd.DataFrame(rto_cleaned_adoptable_dict['rto_adoptable_breeds'].unique(), columns = ['unique_rto_adoptable_breeds'])
rto_adoptable_list = rto_cleaned_adoptable_dict['unique_rto_adoptable_breeds'].tolist()


# In[274]:


def disposal_conditions(df):
    if (df['outcome_type']=='Disposal'):
        return df['breed']
    
disposal_dict = pd.DataFrame(X.apply(disposal_conditions, axis = 1), columns = ['disposal_breeds'])
cleaned_disposal_dict = disposal_dict[disposal_dict['disposal_breeds'].notnull()]
cleaned_disposal_dict = pd.DataFrame(cleaned_disposal_dict['disposal_breeds'].unique(), columns = ['unique_disposal_breeds'])
disposal_list = cleaned_disposal_dict['unique_disposal_breeds'].tolist()


# In[276]:


def lookup_adotable_breeds(breed):
    if breed in adoptable_list:
        return 'adoptible_likely'          #made changes returns here
    else:
        return 'adoptible_unlikely'        #made changes returns here
    
def lookup_rto_adoptabale_breeds(breed):
    if breed in rto_adoptable_list:
        return 'rto_adoptible_likely'          #made changes returns here
    else:
        return 'rto_adoptible_unlikely'        #made changes returns here
        
def lookup_disposable_breeds(breed):
    if breed in disposal_list:
        return 'high_disposability'
    else:
        return 'low_disposability'


# In[277]:


X['adoptability'] = X.breed.apply(lookup_adotable_breeds)
X['disposability'] = X.breed.apply(lookup_disposable_breeds)
X['rto_adoptability']= X.breed.apply(lookup_rto_adoptabale_breeds)


# In[278]:


#We are excluding dogs because only 2 dogs are disposed.
X.loc[X['animal_type']=='Dog', 'disposability'] = 'low_disposability'


# # Feature encoding(final iteration)

# In[293]:


X_final = X.copy()


# In[294]:


#FINAL ONE HOT ENCODING
def onehotdataframe_final(dfs):  
    one_hot_animal_type= pd.get_dummies(dfs.animal_type)

    one_hot_intake_condition = pd.get_dummies(dfs.intake_condition)
    one_hot_intake_condition.columns = [str(col) + '_intake' for col in one_hot_intake_condition.columns]

    one_hot_intake_type = pd.get_dummies(dfs.intake_type)
    one_hot_intake_type.columns = [str(col) + '_intake' for col in one_hot_intake_type.columns]

    one_hot_sex_upon_intake = pd.get_dummies(dfs.sex_upon_intake)
    one_hot_sex_upon_intake.columns = [str(col) + '_intake' for col in one_hot_sex_upon_intake.columns]

    one_hot_intake_weekday = pd.get_dummies(dfs.intake_weekday)
    one_hot_intake_weekday.columns = [str(col) + '_intake' for col in one_hot_intake_weekday.columns]

    one_hot_sex_upon_outcome = pd.get_dummies(dfs.sex_upon_outcome)
    one_hot_sex_upon_outcome.columns = [str(col) + '_outcome' for col in one_hot_sex_upon_outcome.columns]

    one_hot_outcome_weekday = pd.get_dummies(dfs.outcome_weekday)
    one_hot_outcome_weekday.columns = [str(col) + '_outcome' for col in one_hot_outcome_weekday.columns]     
    
    one_hot_breed_bucket = pd.get_dummies(dfs.breed_bucket)
    one_hot_breed_bucket.columns = [str(col) + '_breed_category' for col in one_hot_breed_bucket.columns]  
    
    dfs = dfs.drop(['animal_type','intake_type', 'intake_condition', 'sex_upon_intake', 'intake_weekday', 'sex_upon_outcome', 'outcome_weekday', 'breed_bucket'], axis = 1) 

    #Merging one hot encoded features with our dataset 'data' 
    df_encoded=pd.concat([dfs, one_hot_animal_type,one_hot_intake_condition,one_hot_intake_type,one_hot_sex_upon_intake,one_hot_intake_weekday,one_hot_sex_upon_outcome,one_hot_outcome_weekday, one_hot_breed_bucket],axis=1)  
    
    return(df_encoded)


# In[295]:


X_final = onehotdataframe_final(dfs = X_final)


# In[297]:


X_final['adoptability'] = X_final['adoptability'].map({'adoptible_unlikely' : 0, 'adoptible_likely' : 1 })
X_final['disposability'] = X_final['disposability'].map({'low_disposability' : 0, 'high_disposability' : 1 })
X_final['rto_adoptability'] = X_final['rto_adoptability'].map({'rto_adoptible_unlikely' : 0, 'rto_adoptible_likely' : 1 })
X_final['outcome_type'] = X_final['outcome_type'].map({'Adoption' : 8, 'Transfer': 7, 'Return to Owner' : 6, 'Euthanasia' : 5, 'Died' : 4, 'Missing' : 3, 'Relocate' : 2, 'Rto-Adopt' : 1, 'Disposal' : 0})

y_final = X_final['outcome_type']
X_final= X_final.drop(['outcome_type'], axis = 1)

X_final_copy = X_final.copy()
X_final_copy = X_final_copy.drop(['breed', 'color'], axis = 1)
X_final_copy['intake_monthyear'] = X_final_copy.intake_monthyear.astype(int)
X_final_copy['outcome_monthyear'] = X_final_copy.outcome_monthyear.astype(int)


# ** Dropping breed and color because i already tried to bucket these categorical features with "adoptablity","disposability" , "breed_bucket"**

# In[302]:


print(X_final_copy.shape, y_final.shape)


# In[303]:


print('Missing value in training set:', X_final_copy.isnull().sum().max())


# ** Converting into train and validation set **

# In[304]:


# Splitting the dataset into the Training set and Test set.
X_train, X_val, y_train, y_val = train_test_split(X_final_copy, y_final, test_size = 0.2, random_state = 20)
#Split is 80%-20% for testing. Here X_test~Validation set.


print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)


# In[305]:


X_train['intake_monthyear'] = X_train.intake_monthyear.astype(int)
X_val['intake_monthyear'] = X_val.intake_monthyear.astype(int)
X_train['outcome_monthyear'] = X_train.outcome_monthyear.astype(int)
X_val['outcome_monthyear'] = X_val.outcome_monthyear.astype(int)


# # TEST 5 64 leaves LIGHTGBM

# In[306]:


import lightgbm as lightgbmmodel
from lightgbm import LGBMClassifier as lgb


# In[212]:


model5 = lgb(task =  'train',boosting_type =  'gbdt',  objective = 'multiclass',num_class = 9,metric = 'multi_logloss',learning_rate =  0.0026, max_depth=  20, num_leaves =  64, feature_fraction =  0.4, bagging_fraction =  0.6, bagging_freq =  17, n_estimators = 2500, random_state =1234) 
model5.fit(X_train, y_train)


# In[213]:


y_pred1 = model5.predict(X_val)


# In[232]:


from sklearn.metrics import f1_score
f1_score(y_pred1, y_val.iloc[:] , average = 'micro')
#lgb(task =  'train',boosting_type =  'gbdt',  objective = 'multiclass',num_class = 9,metric = 'multi_logloss',learning_rate =  0.0026, max_depth=  20, num_leaves =  64, feature_fraction =  0.4, bagging_fraction =  0.6, bagging_freq =  17, n_estimators = 10000, random_state =1234)
#dropping color


# # TEST 6 Xgboost

# In[305]:


model6 = XGBClassifier(learning_rate =0.1, n_estimators= 500,max_depth= 7 ,min_child_weight=1, gamma=3, subsample=0.8, colsample_bytree=0.8,objective= 'multi:softmax',nthread=4, scale_pos_weight=1, seed=1234)
model6.fit(X_train, y_train, eval_metric = 'mlogloss')


# In[306]:


y_pred2 = model6.predict(X_val)


# In[295]:


from sklearn.metrics import f1_score
f1_score(y_pred2, y_val.iloc[:] , average = 'micro')
#(learning_rate =0.1,n_estimators=500,max_depth= 7 ,min_child_weight=1, gamma=3, subsample=0.8, colsample_bytree=0.8,objective= 'multi:softmax',nthread=4, scale_pos_weight=1, seed=1234)
#gamma reduced to 3
#keeping breed and removing color
#increasing max_depth to 3


# In[ ]:


xgbcheck = XGBClassifier(learning_rate =0.1,n_estimators=500,max_depth= 7 ,min_child_weight=1, gamma=3, subsample=0.8, colsample_bytree=0.8,objective= 'multi:softmax',nthread=4, scale_pos_weight=1, seed=1234)
xgbcheck.fit(X_train, y_train)
sorted_idx = np.argsort(model3.feature_importances_)[::-1]
for index in sorted_idx:
    print([X_train_check.columns[index], model8.feature_importances_[index]])


# Observations:
# 
# 1. Unknown_intake and unknown_outcome are same column hence dropping one, lets drop unknown_intake
# 2. Also dropping intake_year and Thursday_intake as they are poor predictors as expected. 

# In[308]:


X_train_final = X_train.drop(['Unknown_intake', 'intake_year', 'Thursday_intake'], axis = 1)
X_val_final = X_val.drop(['Unknown_intake' , 'intake_year', 'Thursday_intake'], axis = 1)


# In[311]:


model8 = XGBClassifier(learning_rate =0.1,n_estimators=500,max_depth= 7 ,min_child_weight=1, gamma=3, subsample=0.8, colsample_bytree=0.8,objective= 'multi:softmax',nthread=4, scale_pos_weight=1, seed=1234)
model8.fit(X_train_final, y_train)
y_pred4 = model8.predict(X_val_final)
predictions = [round(value) for value in y_pred4]
accuracy = accuracy_score(y_val, predictions)
print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (0.0048, X_train_final.shape[1], accuracy*100.0))


# **Best xgboost model = model8**
# 
# **Best lightgbm model = model5**

# **Comparison with other models:**

# MODEL3 - LOGISTIC REGRESSION:

# In[316]:


#For other models feature scaling is necessary
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_trainf = sc_X.fit_transform(X_train_final)
X_testf= sc_X.transform(X_val_final)
y_trainf= y_train

lr= LogisticRegression(penalty = 'l2', random_state =1234, solver ='newton-cg', multi_class= 'multinomial' )
lr.fit(X_trainf, y_trainf)
y_predlr= lr.predict(X_testf)


# In[317]:


predictions = [round(value) for value in y_predlr]
accuracy = accuracy_score(y_val, predictions)
print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (0.0048, X_train_final.shape[1], accuracy*100.0))


# MODEL4- RANDOM FOREST

# In[318]:


rf =  RandomForestClassifier(n_estimators= 1000 , criterion = 'entropy' , random_state = 0, bootstrap = True)
rf.fit(X_train_final, y_train)
y_predrf = rf.predict(X_val_final)
predictions = [round(value) for value in y_predrf]
accuracy = accuracy_score(y_val, predictions)
f_score = f1_score(y_predrf, y_val.iloc[:] , average = 'micro')
print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (0.0048, X_train_final.shape[1], accuracy*100.0))
print("f1 -score : ",f_score )


# # Predictions :

# ** Preparation of test set for predictions **

# In[332]:


test = pd.read_csv('/Users/apple/Documents/Animal State Prediction - dataset/test.csv')


# In[335]:


test = test.drop(['outcome_datetime'], axis  = 1)
test = DataFrameImputer().fit_transform(pd.DataFrame(test))
test = test.drop(['count', 'outcome_number'], axis = 1)
test = test.drop(['animal_id_outcome'], axis = 1)
test = test.drop(['age_upon_intake', 'age_upon_intake_(years)', 'age_upon_intake_age_group', 'age_upon_outcome', 'age_upon_outcome_(years)', 'age_upon_outcome_age_group', 'date_of_birth','dob_month', 'time_in_shelter'], axis = 1)
test['intake_day'] = pd.DatetimeIndex(test['intake_datetime']).day
test = test.drop(['intake_datetime'], axis = 1)
test['intake_monthyear'] = test['intake_year'] * 100 + X['intake_month']
test['outcome_monthyear'] = test['outcome_year']* 100 + X['outcome_month']
test = test.drop(['outcome_year'], axis  = 1)
test['breed_bucket'] = test.breed.apply(breed_check)
test['adoptability'] = test.breed.apply(lookup_adotable_breeds)
test['disposability'] = test.breed.apply(lookup_disposable_breeds)
test['rto_adoptability'] = test.breed.apply(lookup_rto_adoptabale_breeds)
#We are excluding dogs because only 2 dogs are disposed.
test.loc[test['animal_type']=='Dog', 'disposability'] = 'low_disposability'
test['intake_monthyear'] = test.intake_monthyear.astype(int)
test['outcome_monthyear'] = test.outcome_monthyear.astype(int)
test_final = test.copy()
test_final = onehotdataframe_final(dfs = test_final)
test_final.shape

test_final['adoptability'] = test_final['adoptability'].map({'adoptible_unlikely' : 0, 'adoptible_likely' : 1 })
test_final['disposability'] = test_final['disposability'].map({'low_disposability' : 0, 'high_disposability' : 1 })
test_final['rto_adoptability'] = test_final['rto_adoptability'].map({'rto_adoptible_unlikely' : 0, 'rto_adoptible_likely' : 1 })

#testing
test_final = test_final.drop(['breed','color','Unknown_intake', 'intake_year', 'Thursday_intake'], axis = 1)


# ** Below are iterative models that I used to check the performance of my different models on out of time validation set(i.e. test set) and select models accordingly **

# In[313]:


#Testing subject to changes
X_train_final_check = X_train_final.drop(['intake_monthyear'], axis = 1)
X_val_final_check = X_val_final.drop(['intake_monthyear'], axis = 1)


# In[320]:


#Including intake_monthyear
model5 = lgb(task = 'train',boosting_type =  'gbdt',  objective = 'multiclass',num_class = 9,metric = 'multi_logloss',learning_rate =  0.0026, max_depth=  -1, num_leaves =  64, feature_fraction =  0.4, bagging_fraction =  0.6, bagging_freq =  17, n_estimators = 2500, random_state =1234,min_split_gain = 0, reg_alpha = 0, reg_lambda = 0 ) 
model5.fit(X_train_final, y_train)
y_pred4 = model5.predict(X_val_final)
predictions = [round(value) for value in y_pred4]
accuracy = accuracy_score(y_val, predictions)
f_score = f1_score(y_pred4, y_val.iloc[:] , average = 'micro')
print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (0.0048, X_train_final.shape[1], accuracy*100.0))
print("f1 -score : ",f_score )


# In[401]:


model5 = lgb(task = 'train',boosting_type =  'gbdt',  objective = 'multiclass',num_class = 9,metric = 'multi_logloss',learning_rate =  0.0026, max_depth=  -1, num_leaves =  64, feature_fraction =  0.4, bagging_fraction =  0.6, bagging_freq =  17, n_estimators = 2500, random_state =1234,min_split_gain = 0, reg_alpha = 0, reg_lambda = 0 ) 
model5.fit(X_train_final_check, y_train)
y_pred4 = model5.predict(X_val_final_check)
predictions = [round(value) for value in y_pred4]
accuracy = accuracy_score(y_val, predictions)
f_score = f1_score(y_pred4, y_val.iloc[:] , average = 'micro')
print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (0.0048, X_train_final_check.shape[1], accuracy*100.0))
print("f1 -score : ",f_score )


# In[428]:


print('Missing value in training set:', X_train_final_check.isnull().sum().max())


# In[429]:


#Excluding intake_monthyear
rf =  RandomForestClassifier(n_estimators= 1000 , criterion = 'entropy' , random_state = 0, bootstrap = True)
rf.fit(X_train_final_check, y_train)
y_pred4 = rf.predict(X_val_final_check)
predictions = [round(value) for value in y_pred4]
accuracy = accuracy_score(y_val, predictions)
f_score = f1_score(y_pred4, y_val.iloc[:] , average = 'micro')
print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (0.0048, X_train_final.shape[1], accuracy*100.0))
print("f1 -score : ",f_score )


# In[319]:


#Including intake_monthyear
xgb = XGBClassifier(learning_rate =0.1, n_estimators= 500,max_depth= 7 ,min_child_weight=1, gamma=3, subsample=0.8, colsample_bytree=0.8,objective= 'multi:softmax',nthread=4, scale_pos_weight=99, seed=1234)
xgb.fit(X_train_final, y_train, eval_metric = 'mlogloss')
y_pred4 = xgb.predict(X_val_final)
predictions = [round(value) for value in y_pred4]
accuracy = accuracy_score(y_val, predictions)
f_score = f1_score(y_pred4, y_val.iloc[:] , average = 'micro')
print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (0.0048, X_train_final.shape[1], accuracy*100.0))
print("f1 -score : ",f_score )


# In[432]:


xgb = XGBClassifier(learning_rate =0.1, n_estimators= 500,max_depth= 7 ,min_child_weight=1, gamma=3, subsample=0.8, colsample_bytree=0.8,objective= 'multi:softmax',nthread=4, scale_pos_weight=99, seed=1234)
xgb.fit(X_train_final_check, y_train, eval_metric = 'mlogloss')
y_pred4 = xgb.predict(X_val_final_check)
predictions = [round(value) for value in y_pred4]
accuracy = accuracy_score(y_val, predictions)
f_score = f1_score(y_pred4, y_val.iloc[:] , average = 'micro')
print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (0.0048, X_train_final.shape[1], accuracy*100.0))
print("f1 -score : ",f_score )


# Hence we will keep the intake_monthyear variable

# In[322]:


estimators =[]
model1= model5
estimators.append(('lightgbm', model1))
model2=rf
estimators.append(('random_forest', model2))
model3=xgb
estimators.append(('xgboost', model3))
'''Ensembling now'''
ensemble = VotingClassifier(estimators, voting = 'soft', weights = [2,1,2])   #[2,1,2] best till now
ensemble.fit(X_train_final, y_train)


# In[324]:


y_pred_ensemble = ensemble.predict(X_val_final)
predictions = [round(value) for value in y_pred_ensemble]
accuracy = accuracy_score(y_val, predictions)
f_score = f1_score(y_pred_ensemble, y_val.iloc[:] , average = 'micro')
print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (0.0048, X_train_final.shape[1], accuracy*100.0))
print("f1 -score : ",f_score )


# In[328]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred_ensemble, y_val)


# In[327]:


from sklearn.metrics import roc_auc_score
sns.heatmap(cm,annot=True,fmt="d", cbar = False)
print(classification_report(y_pred_ensemble, y_val))


# In[406]:


final_predict1 = ensemble.predict(test_final)


# In[407]:


predictions = pd.DataFrame(ml_dataset_test['animal_id_outcome'])
y_pred_df = pd.DataFrame(final_predict1, columns=['outcome_type'])
results = pd.concat([predictions, y_pred_df], axis=1)
results['outcome_type']= results['outcome_type'].map({ 8 :'Adoption', 7: 'Transfer', 6: 'Return to Owner', 5 : 'Euthanasia' , 4 : 'Died' , 3 : 'Missing' , 2 : 'Relocate' ,  1 : 'Rto-Adopt' , 0 : 'Disposal' }).astype(str)


# In[408]:


results.to_csv('/Users/apple/Documents/Animal_State_Prediction_dataset/pred5_lgbm_64leavesestimator2_feb9_3.csv', index = False)


# # Fitting entire dataset for best predictions for external test set

# ** Fitting on complete dataset i.e. X_final_copy**

# In[329]:


X_final_copy = X_final_copy.drop(['Unknown_intake', 'intake_year', 'Thursday_intake'], axis = 1)
X_final_copy.shape


# In[330]:


ensemble.fit(X_final_copy, y_final)


# In[344]:


y_predfull = ensemble.predict(test_final)


# In[345]:


predictions = pd.DataFrame(ml_dataset_test['animal_id_outcome'])
y_pred_df = pd.DataFrame(y_predfull, columns=['outcome_type'])
results = pd.concat([predictions, y_pred_df], axis=1)
results['outcome_type']= results['outcome_type'].map({ 8 :'Adoption', 7: 'Transfer', 6: 'Return to Owner', 5 : 'Euthanasia' , 4 : 'Died' , 3 : 'Missing' , 2 : 'Relocate' ,  1 : 'Rto-Adopt' , 0 : 'Disposal' }).astype(str)


# In[346]:


results.to_csv('/Users/apple/Documents/Animal_State_Prediction_dataset/voting_feb9_full_withmnthyear.csv', index = False)


# **Miscellaneous Models**

# In[626]:


train_h2o = X_train_final.copy()
test_h2o = X_val_final.copy()
train_h2o_df = pd.concat([train_h2o, y_train], axis = 1)
test_h2o_df  = pd.concat([test_h2o, y_val], axis = 1)


# In[630]:


#FITTING AUTO ML
#NOTE IT TAKES LOT OF TIME 
import h2o
from h2o.automl import H2OAutoML

h2o.init()

train_h2o_final = h2o.H2OFrame(train_h2o_df)
test_h2o_final = h2o.H2OFrame(test_h2o_df)

# Identify predictors and response
x = train_h2o_final.columns
y_h2o = "outcome_type"
x.remove(y_h2o)

# For binary classification, response should be a factor
train_h2o_final[y_h2o] = train_h2o_final[y_h2o].asfactor()
test_h2o_final[y_h2o] = test_h2o_final[y_h2o].asfactor()

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y_h2o, training_frame = train_h2o_final)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

# The leader model is stored here
aml.leader


# In[631]:


test_h2o_dataset = h2o.H2OFrame(test_final)


# In[632]:


preds_h2o = aml.predict(test_h2o_dataset)
preds_df = preds_h2o.as_data_frame()
preds_matrix = preds_df.iloc[:,0]


# In[681]:


predictions = pd.DataFrame(ml_dataset_test['animal_id_outcome'])
y_pred_df = pd.DataFrame(preds_matrix.values, columns=['outcome_type'])


# In[682]:


results = pd.concat([predictions, y_pred_df], axis=1)
results['outcome_type']= results['outcome_type'].map({ 8 :'Adoption', 7: 'Transfer', 6: 'Return to Owner', 5 : 'Euthanasia' , 4 : 'Died' , 3 : 'Missing' , 2 : 'Relocate' ,  1 : 'Rto-Adopt' , 0 : 'Disposal' }).astype(str)


# In[683]:


y_pred_df = pd.DataFrame(preds_matrix.values, columns=['outcome_type'])


# In[684]:


results.to_csv('/Users/apple/Documents/Animal_State_Prediction_dataset/pred6_gbm_automl.csv', index = False)


# **Final results:**
# 
# Best Model: Voting Classifier(Ensemble of lightgbm, random forest and xgboost)
# 
# Best F1_score on validation set :  ~66
