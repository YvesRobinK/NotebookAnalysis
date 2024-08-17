#!/usr/bin/env python
# coding: utf-8

# <div align="center">
#     <h1> Diabetes Mellitus Prediction </h1>
# </div>

# 

# Hi,
# This is my second notebook in this series. The first one is an exploratory analysis which forms the basis of this notebook here:
# https://www.kaggle.com/kritidoneria/beginner-automl-wids21-eda-starter
# 
# The problem statement at hand here is:
# 
# This year's challenge will focus on models to determine whether a patient admitted to an ICU has been diagnosed with a particular type of diabetes, Diabetes Mellitus. **Using data from the first 24 hours of intensive care** , individuals and teams will explore labeled training data for model development.
# 
# Feature engineering is what separates Top ranking Kaggle users from the others,among other things. Hence it is importnat to talk to our data in a language that it is intended.
# Here, I shall be creating some features ,from Domain knowledge ( as I have work experience in Healthcare) and using insights from my EDA.I wont be using auto feature generation at this stage.
# 
# 
# **If you fork it or find this useful, Please upvote the notebook and leave comments**

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_colwidth', -1)
import os


# # Reading the Data

# In[2]:


data=pd.read_csv('../input/widsdatathon2021/TrainingWiDS2021.csv',index_col=0)
data.head()


# # Reading the data dictionary and typecasting the data accordingly

# In[3]:


pd.set_option('display.max_rows', 200)
df_dict=pd.read_csv('../input/widsdatathon2021/DataDictionaryWiDS2021.csv')
df_dict

#Checking if all columns in dictionary exist in data
set(df_dict['Variable Name'])-set(data.columns)

#This appears to be an extra column. Lets remove it from the dictionary.

df_dict=df_dict[df_dict['Variable Name'].isin(data.columns)]
df_dict.shape

data.shape
#This is aligned now

# Typecasting variables according to data dictionary

#Notice how the intended Datatype has been given. Let's look at how many variables are there of each type and then typecast accordingly.


df_dict['Data Type'].value_counts()

#Binary variables are interesting. let's see which ones are these
df_dict[df_dict['Data Type']=='binary']

#Let's create a new mapping for columns
datatypes_dict={
'numeric':float,
'binary':int,
'string':str,
'integer':int
}

df_dict['Data Type_edited']=df_dict['Data Type'].map(datatypes_dict)
df_dict['Data Type_edited'].value_counts()


#This looks more workable. Now let's create a dictionary and use it to change datatypes of all columns at once.

datatype_edited_dict=dict(zip(df_dict['Variable Name'],df_dict['Data Type_edited']))
datatype_edited_dict

data=data.fillna(-999).astype(datatype_edited_dict)


# # Diabetes Classification
# 
# <h2> Type 1 </h2>is a chronic condition when the body doesnt produce enough insulin. Diagn
# <h2> Type 2 </h2>is a chronic condition that disrupts the way a body uses insulin
# <h2> Type 3 </h2>is when blood sugar spikes during pregnancy.

# <h1>Visualizing features that differ most for people diagnosed and not diagnosed</h1>

# In[4]:


((data[data['diabetes_mellitus']==0].mean()-data[data['diabetes_mellitus']==1].mean())/data[data['diabetes_mellitus']==0].mean()).sort_values(ascending=False)


# <h1> Creating features from min and max</h1>
# 
# There are many columns that have min, max values. It indicates the number of times a reading has been taken,but we can use these to create our own features.
# The difference gives us a range,and the ratio gives us the jump.
# In healthcare,measurements are often taken more than once for precision, so I'll also create flags if the measurement is taken more than once.

# In[5]:


min_feat=[x for x in data.columns if '_min' in x]
max_feat=[x for x in data.columns if '_max' in x]


# In[6]:


len(min_feat)
len(max_feat)


# In[7]:


#let's create _diff and ratio values
for i in min_feat:
    a=i
    b = i.replace('_min','_max')
#     print(a,b)
    new_col=(str(i).replace('_min',''))+'_diff'
    data[new_col]=data[b]-data[a]
    data[new_col+"flag"]=np.where(data[a]!=data[b],1,0)
    new_col_ratio=(str(i).replace('_min',''))+'_ratio'
    data[new_col_ratio]=data[b]/data[a]


# In[8]:


#seeing new added columns. This increases the feature size to above 300.
data.head()


# <h1> Creating flags for High values of certain markers </h1>
# 
# A quick Google search shows People without diabetes rarely have blood sugar levels over 140 mg/dL after a meal, unless it’s really large. Now blood sugar tests are measures as fasting and after meals,hence the range becomes important.
# 
# *Remember while dealing with healthcare data, UNITS ARE IMPORTANT!!*
# 
# (1 mg/dL = 0.0555 mmol/L)

# In[9]:


data['glucose_apache'].describe()


# In[10]:


#putting it on minimum since on an average glucose levels for a diabetic are higher than a non-diabetic
data['d1_glucose_min_flag']=np.where(data['d1_glucose_min']>(120*.0555),1,0)
data.groupby(['diabetes_mellitus'])['d1_glucose_min_flag'].mean()


# Hello. Here, I'll use some features created in an awesome notebook I came across (mentioned in references)

# In[11]:


data['age_type'] = data.age.fillna(0).apply(lambda x: 10 * (round(int(x)/10)))
data['apache_2_diagnosis_split1'] = np.where(data['apache_2_diagnosis'].isna() , np.nan , data['apache_2_diagnosis'].apply(lambda x : x % 10)  )
data['apache_2_diagnosis_type'] = data.apache_2_diagnosis.round(-1).fillna(-100).astype('int32')
data['apache_2_diagnosis_x'] = data['apache_2_diagnosis'].astype('str').str.split('.',n=1,expand=True)[0]
data['apache_3j_diagnosis_split1'] = np.where(data['apache_3j_diagnosis'].isna() , np.nan , data['apache_3j_diagnosis'].astype('str').str.split('.',n=1,expand=True)[1]  )
data['apache_3j_diagnosis_type'] = data.apache_3j_diagnosis.round(-2).fillna(-100).astype('int32')
data['apache_3j_diagnosis_x'] = data['apache_3j_diagnosis'].astype('str').str.split('.',n=1,expand=True)[0]
data['bmi_type'] = data.bmi.fillna(0).apply(lambda x: 5 * (round(int(x)/5)))
data['comorbidity_score'] = data['aids'].values * 23 + data['cirrhosis'] * 4  + data['hepatic_failure'] * 16 + data['immunosuppression'] * 10 + data['leukemia'] * 10 + data['lymphoma'] * 13 + data['solid_tumor_with_metastasis'] * 11
data['comorbidity_score'] = data['comorbidity_score'].fillna(0)
data['gcs_sum'] = data['gcs_eyes_apache']+data['gcs_motor_apache']+data['gcs_verbal_apache']
data['gcs_sum'] = data['gcs_sum'].fillna(0)
data['gcs_sum_type'] = data.gcs_sum.fillna(0).apply(lambda x: 2.5 * (round(int(x)/2.5))).divide(2.5)
data['height_type'] = data.height.fillna(0).apply(lambda x: 5 * (round(int(x)/5)))
data['weight_type'] = data.weight.fillna(0).apply(lambda x: 5 * (round(int(x)/5)))


# In[ ]:





# Similar flags can be created for creatinin and albumin,two proteins which are widely used in kidney disease analysis and Diabetes is one of the leading causes of Kidney disease.

# # Next steps
# 
# **The next steps are selecting features from the ones generated above.
# I will update this notebook soon with the same.**

# In[ ]:





# # THANKS!!

# # References
# 1. https://www.kaggle.com/kritidoneria/beginner-automl-wids21-eda-starter
# 2. https://www.kaggle.com/learn/feature-engineering
# 3. https://www.endocrineweb.com/conditions/type-1-diabetes/type-1-diabetes#:~:text=The%20primary%20screening%20test%20for,hemoglobin%20test%2C%20or%20A1C%20test.
# 4. https://www.kaggle.com/muhammadmelsherbini/jane-street-extensive-eda-pca-starter#PCA-&-Clustering
# 5. https://www.webmd.com/diabetes/guide/diabetes-hyperglycemia#:~:text=High%20blood%20sugar%2C%20or%20hyperglycemia,for%20at%20least%208%20hours.
# 6. https://www.kaggle.com/siavrez/2020fatures

# In[ ]:




