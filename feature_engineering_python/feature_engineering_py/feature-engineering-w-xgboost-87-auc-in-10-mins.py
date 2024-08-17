#!/usr/bin/env python
# coding: utf-8

# # Employee Attrition Prediction 💼
# 
# <code style="background:yellow;color:red;font-size:15px;"><b>NOTE.</b></code> <span style="color:green;font-size:15px;"> The data we will use is synthetic data looking at Employee Attrition. You can find this data and its related attribute information from :<a href="https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset">IBM HR Analytics</a></span>
# 
# ***
# 
# 
# ### ✅Important Points to have in mind.
# 
# * The data is replicate of deep learning trained model on Employee Attririon.
# * Its a **Binary Classification** Task
# * The Measure for merit is **ROC-AUC Curve**

# ## Importing Important Libraries 

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.calibration import CalibratedClassifierCV
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from scipy.stats import boxcox, median_abs_deviation
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, RFECV, SelectKBest , f_classif
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier , XGBRFClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from scipy import stats
import statsmodels.api as sm
import pylab as py
from scipy.stats import norm
from sklearn.metrics import precision_recall_curve, auc, roc_curve

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
sns.set_style("darkgrid")
pd.set_option('mode.chained_assignment',None)


# In[2]:


df_train = pd.read_csv('/kaggle/input/playground-series-s3e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s3e3/test.csv')
df_train.head()


# In[3]:


df_train.shape , df_test.shape


# * *Here we can see that no of rows wrt columns are very less therefore there is good chances of overfitting*
# * *Therefore while selecting ML model we have to accompany Regularisation Technique*

# In[4]:


# First we will check where our class are balance or imbalance in nature 
fig , ax = plt.subplots(figsize =(20,7))
sns.countplot(x=df_train['Attrition'])
plt.title('Employee Attrition Count')

total = len(df_train)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='center')


# * *We can see that Class 0 contain `88.1%` out of all both class other being Class 1 for attrition*
# * *Simple Take away : Imbalance Data* 
# 
# <code style="background:yellow;color:red;font-size:15px;">⚠️ Whenever dealing with Imbalance data , be sure which metrics to target</code>
# 
# <div class="alert alert-block alert-info">
# <b>Tip:</b> <span style="font-size:15px;">Go and walkthrough some YT tutorials/Blogs on how to deal with Imbalance Data , which metrics to use and use case with proper example of these metrics. Please have a look on this amazing blog which , I've attched below hope you guys will find helpful.</span> 
#     
# <a href="https://www.analyticsvidhya.com/blog/2021/06/5-techniques-to-handle-imbalanced-data-for-a-classification-problem/">Tackling Imbalance Data!😉😉</a>
# </div>
# 
# 
# 

# # Exploratory Data Analysis

# ## Univariate Analysis of each variable 📈

# In[5]:


# Here we will first seperate Categorical and Numerical Data 
# Attrition being target variabke isnt considered 
# ID being unique identifeir 
# Employeee Count has only one value which means zero variance therefore dropping it 
numerical_feat = [col for col in df_train.select_dtypes('int64').columns if col not in ['Attrition','EmployeeCount','id']]
categorical_feat = [col for col in df_train.select_dtypes('object').columns if col != 'Attrition']


# **1. For Categorical data , no need for looking its distribution pattern i.e. whether or not Gaussian or not.**
# 
# **2. Use Count and Bar plot to represent these type of data.**
# 
# **3. Categorical Data are of two type : Ordinal and Categorical.**

# In[6]:


fig , ax = plt.subplots(4,2,figsize=(25,25))
total = len(df_train)
ax =  np.ravel(ax)

for i , col in enumerate(categorical_feat):
  sns.countplot(ax=ax[i],x=df_train[col],hue=df_train['Attrition'])
  ax[i].tick_params(labelrotation=45)
  ax[i].set_title(f"{col}",fontsize=15)
  ax[i].legend(title='Attrition', loc='upper right', labels=['No Attrition', 'Attrition'])
  ax[i].set(xlabel=None)
  
  for p in ax[i].patches:
    percentage = f'{100*p.get_height()/total:.2f}%\n'
    x = p.get_x() + p.get_width()/2
    y = p.get_height()
    ax[i].annotate(percentage, (x,y),ha='center',va='center')


fig.suptitle("Employee Attrition by categorical columns",fontsize = 20)
fig.tight_layout(pad = 3)
plt.show()
  


# ***From these subplot it is clearly visible and inferable that how each category relates to our Target variable***
# 
# <code style="background:white;color:black;font-size:15px;"><b><i>For Eg.</i></b></code> 
# 
# <ol>
#   <li>
#       In Overtime feature those who are not doing over time tend to leave current organisation viz-a-viz those who does overtime.
# <!--               <ul> -->
# <!--                   <li>Reason</li> -->
# <!--             </ul> -->
#     </li>
#   <li>In Male and Female , Male tend to leave organization more than Female.</li>
#   <li>Among Sinle , Married and Divorsed indivisual , person who is single tend to leave organization more followed by Married & Divorced </li>
# </ol>

# In[7]:


fig,ax = plt.subplots(6,4,figsize = (50,50))
ax = np.ravel(ax)

for i,col in enumerate(numerical_feat):
    sns.histplot(ax = ax[i], x = df_train[col], label = "Train", color= "green")
    sns.histplot(ax = ax[i], x = df_test[col], label = "Test")

    ax[i].legend(title='Attrition', loc='upper right', labels=['Train', 'Test'])
    
fig.suptitle("Train vs Test histograms",fontsize = 20)
plt.tight_layout(pad=3)
plt.show()


# In[8]:


fig,ax = plt.subplots(6,4,figsize = (50,50))
ax = np.ravel(ax)

for i,col in enumerate(numerical_feat):
    sns.kdeplot(ax = ax[i], x = df_train[col], label = "Train", color= "green",warn_singular=False)
    sns.kdeplot(ax = ax[i], x = df_test[col], label = "Test",warn_singular=False)
    
    ax[i].set_title(f"{col}",fontsize=20)
#     ax[i].xlabel(fontsize=18)

    ax[i].legend(title='Attrition', loc='upper right', labels=['Train', 'Test'])
    
fig.suptitle("Train vs Test histograms",fontsize = 20)
plt.tight_layout(pad=3)
plt.show()


# ***From these subplot it is clearly visible and inferable that some of the variables are positively skewed , except Monthly rate which follows negative skew trend.***
# 
# <code style="background:yellow;color:red;font-size:15px;"><b>NOTE.</b></code> 
# 
# <ol>
#   <li>Since these distributions are not gaussian normal form , using linear models wont perform well over these variable to predict target variable.</li>
#   <li>Using Random forest , XGBoost or in general any gradient boosting technique will result in overfitting because no of sample wrt no of features is very less.</li>
# </ol>

# # Building Baseline Model 
# 
# <code style="background:green;color:black;font-size:15px;"><b>WHY.</b></code> 
# 
# <ol>
#   <li><code style="font-size:16px;">This model will serve as a baseline for comparing model which we will going to use in Future.</code></li>
#   <li><code style="font-size:16px;">The characteristic of this model is that, it randomly predicts target variable without any assumption.</code></li>
#   <li><code style="font-size:16px;">Therefore our model couldn't be worse than predicting random target value , if so we should'nt look forward to use it in production environment.</code></li>
# </ol>
# 

# In[9]:


y_true = df_train['Attrition']
data = df_train.drop(['Attrition','id'],axis=1)
X_train,X_test, y_train, y_test = train_test_split(data, y_true, stratify=y_true, test_size=0.3)


# In[10]:


print("-"*10, "Distribution of output variable in train data", "-"*10)
train_distr = Counter(y_train)
train_len = len(y_train)
print("Class 0: ",int(train_distr[0])/train_len,"Class 1: ", int(train_distr[1])/train_len)
print("-"*10, "Distribution of output variable in test data", "-"*10)
test_distr = Counter(y_test)
test_len = len(y_test)
print("Class 0: ",int(test_distr[0])/test_len, "Class 1: ",int(test_distr[1])/test_len)


# In[11]:


predicted_y = np.zeros((test_len,2))
np.random.seed = 42
# np.random.seed(10)
for i in range(test_len):
    rand_probs = np.random.rand(1,2)
    predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

    
predicted_y =np.argmax(predicted_y, axis=1)
print("AUC Score on Test Data using Random Model",roc_auc_score(y_true=y_test,y_score=predicted_y))

# plot_confusion_matrix(y_test, predicted_y)


# ***Therefore any model which we will now consider should score on AUC more than our Random Model. Therefore this step is called building Baseline model🎢***

# <div class="dropdown">
#   <button>ATTENTION.⚠️</button></div>
#     
# > ***Before we get into modelling , first we have to decide which ML Algorithm to choose. Once Choosen then we'll dirty our hand over that ML Algorithm in order to deduce best result.***

# # Exploring ML Algorithm:

# 1. **Logistic Regression.**
# 2. **Random Forest.**
# 3. **XGBoost Classifier.**

# In[12]:


# Lets have a look on over our data
df_train.head()


# In[13]:


# Seperating Target variable (Y) and predictors variables (X)
y_true = df_train['Attrition']
data = df_train.drop('Attrition',axis=1)


# In[14]:


#Seperating Numerical Columns and Categorical Columns 
numerical_feat = [col for col in data.select_dtypes('int64').columns if col not in ['Attrition','EmployeeCount','id']]
categorical_feat = [col for col in data.select_dtypes('object').columns if col != 'Attrition']


# In[15]:


# Showcasing Categorical Data
data[categorical_feat]


# In[16]:


# Here we will Categorical Encode our Data.
sample = pd.get_dummies(data, columns = categorical_feat)
data = sample.drop('id',axis=1)


# <code style="background:yellow;color:red;font-size:15px;">⚠️ We had used <b>One Hot Encoding Technique</b> for Encoding Categorical Data.</code>

# In[17]:


X_train,X_test, y_train, y_test = train_test_split(data, y_true, stratify=y_true, test_size=0.4 , random_state=42)
X_val,X_test, y_val , y_test = train_test_split(X_test, y_test, stratify=y_test, test_size=0.2 , random_state=24)


# In[18]:


print("-"*10, "Distribution of output variable in train data", "-"*10)
print("Shape of Train Data:",X_train.shape)
train_distr = Counter(y_train)
train_len = len(y_train)
print("Class 0: ",int(train_distr[0])/train_len,"Class 1: ", int(train_distr[1])/train_len)
print("-"*10, "Distribution of output variable in test data", "-"*10)
print("Shape of Val Data:",X_val.shape)#,'\t',y_train.shape)
val_distr = Counter(y_val)
val_len = len(y_val)
print("Class 0: ",int(val_distr[0])/val_len, "Class 1: ",int(val_distr[1])/val_len)
print("-"*10, "Distribution of output variable in train data", "-"*10)
print("Shape of Test Data:",X_test.shape)#),'\t',y_train.shape)
test_distr = Counter(y_test)
test_len = len(y_test)
print("Class 0: ",int(test_distr[0])/test_len, "Class 1: ",int(test_distr[1])/test_len)


# In[19]:


# generating list which will contain tuples with model name and Model instance.
models = []
models.append(('LR',LogisticRegression()))
models.append(('RF',RandomForestClassifier()))
models.append(('XGB',XGBClassifier()))


# In[20]:


np.random.seed
fig , ax = plt.subplots(figsize =(20,7))
for name , model in models:
  clf = model
  clf.fit(X_train,y_train)
  clf_sigmoid = CalibratedClassifierCV(clf, cv=3, method='sigmoid')
  clf_sigmoid.fit(X_val, y_val)
  # clf_sigmoid.fit(X_train, y_train)
  y_test_predict_proba = clf_sigmoid.predict_proba(X_test)[:, 1]
  # y_test_predict_proba = clf.predict_proba(X_test)[:, 1]
  fpr, tpr, _ = metrics.roc_curve(y_test,  y_test_predict_proba)
  auc = metrics.roc_auc_score(y_test, y_test_predict_proba)
  plt.plot(fpr,tpr,label=name+"--> "+str(auc))
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.legend(loc=4)
plt.show()


# <code style="background:green;color:black;font-size:15px;"><b>Take Away.💡</b></code> 
# 
# <ol>
#   <li><code style="font-size:16px;">Among all models , performance over ROC AUC XGB > RF > LR .</code></li>
#   <li><code style="font-size:16px;">Among XGBoost and Random Forest the scores are closely tied , reason for that is both use decision tree , trained over boostraped sample of data.</code></li>
#   <li><code style="font-size:16px;">Therefore our focus of intrest will be XGBoost.</code></li>
# </ol>
# 

# <code style="background:white;color:black;font-size:15px;"><b>Why Calibrating ? 🤔.</b></code> 
# 
# <ol>
#   <li><code style="font-size:16px;">In Problem Statement of Regression Analysis , we have squared distance replicating the probabilities of predicted and actual value , less the distance more confident we are in our prediction and vice versa.</code></li>
#   <li><code style="font-size:16px;">Same thing is not available for Classification model , accuracy is just no of correct and incorrect class predicted viz-a-viz actual class , thats why CalibratedClassifierCV comes into picture.</code></li>
#   <li><code style="font-size:16px;">Calibrating a classification model means adjusting it so that its predicted probabilities match reality.</code></li>
#   <li><code style="font-size:16px;">This is important because these predicted probabilities are often used to make decisions..</code></li>
#   <li><code style="font-size:16px;">Calibration helps to improve the accuracy and fairness of these decisions.</code></li>
# </ol>
# 
# 
# 
# <code style="background:yellow;color:red;font-size:15px;">😕 If Confused while discovering about Calibration of Classification Model , feel free to ping me on <a href="https://www.linkedin.com/in/zuber-ahmed-ansari-8b5962147/">my linkedIn profile</a> , I will be more than happy to help you out.</code>
# 
# 
# 
# 
# 
# 
# 

# <code style="background:white;color:black;font-size:15px;"><b> <i> Reference Links for Learning Calibrated Classifier CV 😄.</i></b></code> 
# 
# <ol>
#   <li><code style="font-size:16px;"><a href="https://www.youtube.com/watch?v=AunotauS5yI&ab_channel=ritvikmath">Probability Calibration : Data Science Concepts</a>.</code></li>
#   <li><code style="font-size:16px;"><a href="https://www.youtube.com/watch?v=5zbV24vyO44&ab_channel=CodeEmporium">Why Logistic Regression DOESN'T return probabilities?!</a>.</code></li>
#   <li><code style="font-size:16px;"><a href="https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/">How and When to Use a Calibrated Classification Model with scikit-learn!</a>.</code></li>
# </ol>
# 
# <code style="background:white;color:black;font-size:15px;"><i>👍 Lets be nice and kind for liking their video if refering them.</i></code>
# 
# 
# 

# # Capitalizing Over XGB

# In[21]:


# Now Creating our base XGBClassifier Model so that , any feature transformation done its result should be better than this 
model = XGBClassifier()
eval_set = [(X_train,y_train),(X_val,y_val)]
model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=True)
#calibrating our model
sig_clf = CalibratedClassifierCV(model,cv=3,method='sigmoid')
#Fitting this calibrated model
sig_clf.fit(X_val, y_val)
# make predictions for test data
predictions = sig_clf.predict_proba(X_test)[:,1]
# make predictions for test data
# predictions = model.predict_proba(X_test)[:,1]
# evaluate predictions
auc = metrics.roc_auc_score(y_test, predictions)
print("AUC Obtained: %.2f%%" % (auc * 100.0))
# # Presenting XGBoost 
# model = XGBClassifier()
# model.fit(X_train, y_train,verbose=True)
# #calibrating our model
# sig_clf = CalibratedClassifierCV(model,cv=5,method='sigmoid')
# #Fitting this calibrated model
# sig_clf.fit(X_val, y_val)
# # make predictions for test data
# predictions = model.predict_proba(X_test)[:,1]
# # evaluate predictions
# auc = metrics.roc_auc_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (auc * 100.0))


# ## DATA TXFXN

# In[22]:


fig,ax = plt.subplots(6,4,figsize = (25,35))
ax = np.ravel(ax)

for i,col in enumerate(numerical_feat):

    sns.distplot(ax = ax[i], x = df_train[col], label = "Train", color= "green",bins=10)#,warn_singular=False)
    sns.distplot(ax = ax[i], x = df_test[col], label = "Test",bins=10)#,warn_singular=False)
    
    ax[i].set_xlabel(col)

    ax[i].legend(title='Attrition', loc='upper right', labels=['Train', 'Test'])
    
fig.suptitle("Train vs Test histograms",fontsize = 20)
plt.tight_layout(pad=3)
plt.show()


# In[23]:


# We concatenate test data to train to get a full view of the data as we know it
skew_df = pd.concat((df_train.drop('Attrition',axis =1), df_test), axis =0).skew(numeric_only=True).sort_values()
print("Skewly distributed columns by skewness value\n") 
display(skew_df)


# In[24]:


fig,ax = plt.subplots(figsize=(25,7))

ax.bar(x = skew_df[(skew_df<1)].index, height = skew_df[(skew_df<1)], color = "g", label= "Low skewed features")
ax.bar(x = skew_df[skew_df>1].index, height = skew_df[skew_df>1], color = "r", label = "Highly Positive skewed features")
ax.bar(x = skew_df[skew_df<-1].index, height = skew_df[skew_df<-1], color = "b",label = "Highly negative skewed features")
ax.legend()
fig.suptitle("Skewness of numerical columns",fontsize = 20)
ax.tick_params(labelrotation=90)
for bars in ax.containers:
    ax.bar_label(bars)


# <code style="background:white;color:black;font-size:15px;"><b>Let's Handle these Skewed Columns 😎.</b></code> 
# 
# <ol>
#   <li><code style="font-size:16px;">Categorical values must be excluded from this ambit of correcting skewed data i.e. all categorical data(in our case all categorical_feat).</code></li>
#   <li><code style="font-size:16px;">Therefore marching forward to attempt to make these non gaussian towards more gaussian like feature. And to expect for increase in accuracy.</code></li>
# <!--   <li><code style="font-size:16px;">So feature in which we will be intrested is ['DailyRate','DistanceFromHome','HourlyRate','MonthlyIncome','MonthlyRate','PercentSalaryHike'].</code></li> -->
# <!--   <li><code style="font-size:16px;">.</code></li> -->
# <!--   <li><code style="font-size:16px;">.</code></li> -->
# </ol>
# 
# <code style="background:yellow;color:black;font-size:15px;"><b>Quick Tip.💡</b></code> 
# <code style="background:yellow;color:red;font-size:15px;">😕 If Confused while variable's nature i.e. Numerical or Categorical.Just use Kdeplot , histplot , barchart . And look for in which among the following your feature gets well defined.</code>
#     <ol>
#         <li><code style="font-size:16px;">KDEplot/histplot - numerical , also support categorical but numerical variable will be presented in much more better way which could be trivially understood.</code></li>
#         <li><code style="font-size:16px;">barplot - Categirical Feature.</code></li>
#     </ol>
# 
# 
# 

# ### There are Bunch of mathematical transformation which may help us to transform our skewed distribution to mimic gaussian distribution. These are as follows:
# #### Reduding Right Skewness :
# * Square Root
# * Cube Root
# * Logaritmics
# * Reciprocal
# #### Reducing Left Skewness :
# * Squares
# * Cubes 
# * Highpowers
# 
# <code style="background:yellow;color:black;font-size:15px;"><b>More on this in detail.⭐</b></code> 
# <a href="https://www.kaggle.com/ashishbarvaliya">Ashish Barvaliya's Discussion on <a href="https://www.kaggle.com/discussions/getting-started/110134">Data Skewness Reducing Techniques.</a></a>

# In[25]:


fig,ax = plt.subplots(len(numerical_feat),5, figsize = (30,70))
for i,col in enumerate(numerical_feat):
    
    #scale
    scaler = QuantileTransformer(output_distribution="normal")
    quant_df = scaler.fit_transform(df_train[[col]])

    sns.distplot(x= df_train[col],ax= ax[i,0], color = "r")
    sns.distplot(quant_df,ax= ax[i,1] )
    sns.distplot(np.log1p(df_train[col]), ax = ax[i,2], color= "orange")
    try:
        sns.distplot(boxcox(df_train[col])[0], ax = ax[i,3], color= "orange")
    except:
        pass
    sns.distplot(np.sqrt(df_train[col]), ax = ax[i,4], color= "green")
    # sns.histplot(np.reciprocal(df_train[col]), ax = ax[i,5], color= "green")
    ax[i,0].set_title(f"Orginal ({col})")
    ax[i,0].set(xlabel=None)
    ax[i,1].set_title(f"Quantile Scaling ({col})")
    ax[i,2].set_title(f"Log transform ({col})")
    ax[i,2].set(xlabel=None)
    ax[i,3].set_title(f"Boxcox ({col})")
    ax[i,4].set_title(f"Square Root ({col})")
    # ax[i,5].set_title(f"Reciprocal of ({col})")
plt.suptitle("Distribution Transformations",fontsize = 20)
plt.tight_layout(pad = 6)
plt.show()


# <code style="background:white;color:black;font-size:15px;"><b>Inference👓.</b></code> 
# 
# <ol>
#   <li><code style="font-size:16px;">Among all tranformation steps , Quantile transforms turns out to be the best converting feature more into Gaussian Like.</code></li>
#   <li><code style="font-size:16px;">Therefore moving forward with Quantile Transformation , and also cross checking it with PP Plot.</code></li>
# </ol>

# In[26]:


# here we can see that 
fig,ax = plt.subplots(len(numerical_feat),4, figsize = (30,70))
for i,col in enumerate(numerical_feat):
    
    #scale
    scaler = QuantileTransformer(output_distribution="normal",ignore_implicit_zeros=True,n_quantiles=500)
    quant_df = scaler.fit_transform(df_train[[col]])

    sns.distplot(x= df_train[col],ax= ax[i,0])
    stats.probplot(df_train[col],plot=ax[i,1])
    
    sns.distplot(quant_df,ax= ax[i,2])
    stats.probplot(quant_df.reshape(-1),plot=ax[i,3])

    ax[i,0].set_title(f"original ({col})")
    ax[i,0].set(xlabel=None)
    ax[i,1].set_title(f"PP Plot of respective original ({col})")
    ax[i,1].set(xlabel=None)
    ax[i,2].set_title(f"Quantile Transformed ({col}) distplot")
    ax[i,2].set(xlabel=None)
    ax[i,3].set_title(f"Quantile Transformed ({col}) PP-Plot")
    ax[i,3].set(xlabel=None)
plt.tight_layout(pad = 4)
plt.show()


# In[27]:


# Taking copy of our data over quantiled_data for quantile transformation and then feed into our model.
quantiled_data = data.copy()
quantiled_data.head()


# In[28]:


scaler = QuantileTransformer(output_distribution="normal")
quant_df = scaler.fit_transform(quantiled_data[numerical_feat])
quant_df


# In[29]:


quantiled_data[numerical_feat] = quant_df


# In[30]:


# Creating val , test and train sets
X_train,X_test, y_train, y_test = train_test_split(quantiled_data, y_true, stratify=y_true, test_size=0.4,random_state=42)
X_val,X_test, y_val , y_test = train_test_split(X_test, y_test, stratify=y_test, test_size=0.2,random_state=21)


# In[31]:


# np.random.seed(0)
model = XGBClassifier()
eval_set = [(X_train, y_train), (X_val, y_val)]
model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=True)
#calibrating our model
sig_clf = CalibratedClassifierCV(model,cv=3,method='sigmoid')
#Fitting this calibrated model
sig_clf.fit(X_val, y_val)
# make predictions for test data
predictions = model.predict_proba(X_test)[:,1]
# make predictions for test data
# predictions = model.predict_proba(X_test)[:,1]
# evaluate predictions
auc = metrics.roc_auc_score(y_test, predictions)
print("Accuracy: %.2f%%" % (auc * 100.0))
# # Presenting XGBoost 
# model = XGBClassifier()
# model.fit(X_train, y_train,verbose=True)
# #calibrating our model
# sig_clf = CalibratedClassifierCV(model,cv=5,method='sigmoid')
# #Fitting this calibrated model
# sig_clf.fit(X_val, y_val)
# # make predictions for test data
# predictions = model.predict_proba(X_test)[:,1]
# # evaluate predictions
# auc = metrics.roc_auc_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (auc * 100.0))


# In[32]:


quantiled_data


# ## Risk Factor Calculations [Feature Engineering ]

# ### Now we deduce Risk Factor of some attribute to get feature which could helps us increase our XGBoost Output

# In[33]:


# We will formulate consolidated data for this process , explaination of same will be explained later in this section
consolidated_data = pd.concat([df_train, df_test],axis=0)


# In[34]:


fig,ax = plt.subplots(6,4,figsize = (25,35))
sns.set_style("dark")
ax = np.ravel(ax)
feats = list(consolidated_data[numerical_feat].columns)

label_0 = consolidated_data[consolidated_data['Attrition']==0]
label_1 = consolidated_data[consolidated_data['Attrition']==1]


for i,col in enumerate(feats):
  sns.kdeplot(x=label_0[col], shade=True, label="0",ax=ax[i])
  sns.kdeplot(x=label_1[col], shade=True, label="1",ax=ax[i])
#     sns.distplot(ax = ax[i], x = df_train[col], , color= "green",bins=10)#,warn_singular=False)
#     # sns.distplot(ax = ax[i], x = df_test[col], label = "Test",bins=10)#,warn_singular=False)
    
  ax[i].set_xlabel(col)

  ax[i].legend(title='Attrition', loc='upper right', labels=['A-->0', 'A-->1'])
    
fig.suptitle("Plot for finding the cutoff",fontsize = 20)
plt.tight_layout(pad=3)
plt.show()


# <code style="background:white;color:black;font-size:15px;"><b>Inference👓.</b></code> 
# 
# <ol>
#   <li><code style="font-size:16px;">Here we can see that , some feature clearly distinguished between both the class at some point in X-axis.</code></li>
#   <li><code style="font-size:16px;">The reason for taking consolidated data is that it has both train and test data in it , hence the distribution which we will get will be more generalised in nature.</code></li>
# </ol>

# In[35]:


df_train = pd.read_csv('/kaggle/input/playground-series-s3e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s3e3/test.csv')
df_train.head()


# In[36]:


df_train['Age_Risk'] = (df_train['Age']<34).astype(int)
df_train['DistanceFromHome_Risk'] = (df_train['DistanceFromHome']>=20).astype(int)
df_train['HourlyRate_Risk'] = (df_train['HourlyRate']<60).astype(int)
df_train['StockOptionLevel_Risk'] = (df_train['StockOptionLevel']<1).astype(int)
df_train['TotalWorkingYears_Risk'] = (df_train['TotalWorkingYears']<7).astype(int)
df_train['YearsAtCompany_Risk'] = (df_train['YearsAtCompany']<4).astype(int)
# df_train['YearsInCurrentRole_Risk'] = (df_train['YearsInCurrentRole']<5).astype(int)
# df_train['YearsSinceLastPromotion_Risk'] = (df_train['YearsSinceLastPromotion']<=5).astype(int)
# df_train['YearsWithCurrManager_Risk'] = (df_train['YearsWithCurrManager']<=5).astype(int)
# df_train['PercentSalaryHike_Risk'] = (df_train['PercentSalaryHike']<=12).astype(int)


# In[37]:


df_train['Attribution_Risk']=df_train['Age_Risk']+df_train['DistanceFromHome_Risk']+df_train['HourlyRate_Risk']+df_train['YearsAtCompany_Risk']\
  +df_train['StockOptionLevel_Risk']+df_train['TotalWorkingYears_Risk']


# In[38]:


y_true = df_train['Attrition']
data = df_train.drop('Attrition',axis=1)
numerical_feat = [col for col in data.select_dtypes('int').columns if col not in ['Attrition','EmployeeCount','id']]
categorical_feat = [col for col in data.select_dtypes('object').columns if col != 'Attrition']
sample = pd.get_dummies(data, columns = categorical_feat)
data = sample.drop('id',axis=1)


# In[39]:


# We also have to quantile transform our newly deduce data as well
quantiled_data = data.copy()
quantiled_data.head()


# In[40]:


scaler = QuantileTransformer(output_distribution="normal")
quant_df = scaler.fit_transform(quantiled_data[numerical_feat])
quant_df


# In[41]:


# Creating val , test and train sets
X_train,X_test, y_train, y_test = train_test_split(quantiled_data, y_true, stratify=y_true, test_size=0.4,random_state=42)
X_val,X_test, y_val , y_test = train_test_split(X_test, y_test, stratify=y_test, test_size=0.2,random_state=21)


# In[42]:


# np.random.seed(0)
model = XGBClassifier()
eval_set = [(X_train, y_train), (X_val, y_val)]
model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=True)
#calibrating our model
sig_clf = CalibratedClassifierCV(model,cv=3,method='sigmoid')
#Fitting this calibrated model
sig_clf.fit(X_val, y_val)
# make predictions for test data
predictions = model.predict_proba(X_test)[:,1]
# make predictions for test data
# predictions = model.predict_proba(X_test)[:,1]
# evaluate predictions
auc = metrics.roc_auc_score(y_test, predictions)
print("Accuracy: %.2f%%" % (auc * 100.0))
# # Presenting XGBoost 
# model = XGBClassifier()
# model.fit(X_train, y_train,verbose=True)
# #calibrating our model
# sig_clf = CalibratedClassifierCV(model,cv=5,method='sigmoid')
# #Fitting this calibrated model
# sig_clf.fit(X_val, y_val)
# # make predictions for test data
# predictions = model.predict_proba(X_test)[:,1]
# # evaluate predictions
# auc = metrics.roc_auc_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (auc * 100.0))


# <code style="background:yellow;color:black;font-size:15px;"><b>Conclusion.📈</b></code> 
#     <ol>
#         <li><code style="font-size:16px;">Simply by just Gaussian Tranformation and by adding new features we got spike in 4.77 % in AUC-Score.</code></li>
#     </ol>
# <code style="background:yellow;color:black;font-size:15px;"><b>Courtesy for Risk Factor Analysis.📈</b></code>
# 
# <a href="https://www.kaggle.com/competitions/playground-series-s3e3/discussion/380920">Bill Cruise's Discussion on this competition , Bill also had topped this competition leader board.</a>
# 
# <a href="https://www.kaggle.com/competitions/playground-series-s3e2/discussion/377609">Tilli's Discussion on , How Risk Factor are determined.</a>

# In[ ]:




