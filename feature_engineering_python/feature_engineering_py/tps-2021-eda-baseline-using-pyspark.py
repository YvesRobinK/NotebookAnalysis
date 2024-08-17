#!/usr/bin/env python
# coding: utf-8

# # Intro Of This Notebook:
# 
# I will go through in this notebook the whole process of **EDA** & Creating a **Machine Learning Model** on the famous **Titanic** dataset, which is used by many people all over the world.
# 
# The **goal** of this notebook is to show **EDA** and **how to build an ml model** using **PySpark**.

# ## What is PySpark?
# 
# **Spark** is the name of the engine, that realizes cluster computing while **PySpark** is the Python’s library to use Spark.
# 
# **PySpark** is a great language for performing exploratory data analysis at scale, building machine learning pipelines, and creating ETLs for a data platform. If you’re already familiar with Python and libraries such as Pandas, then PySpark is a great language to learn in order to create more scalable analyses and pipelines.

# ## How to install PySpark? 
# 
# **PySpark** installing process is very easy as like other python’s packages. (eg. Pandas, Numpy,scikit-learn). Want to know more then you can read [my article on **PySpark**](https://towardsdatascience.com/first-time-machine-learning-model-with-pyspark-3684cf406f54)

# In[1]:


get_ipython().system(' pip install pyspark')


# ## About the Problem:
# Using the machine learning tools, we need to analyze the information about the passensgers of Titanic and predict which passenger has survived. This problem has been published by Kaggle and is widely used for learning basic concepts of Machine Learning

# ## Exploring The Data:
# 
# ### Data Dictionary
# ![Imgur](https://i.imgur.com/bkNeXxE.png)
# 
# ## Variable Notes
# **pclass**: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# **age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# **sibsp**: The dataset defines family relations in this way...
# 
# **Sibling** = brother, sister, stepbrother, stepsister
# 
# **Spouse** = husband, wife (mistresses and fiancés were ignored)
# 
# **parch**: The dataset defines family relations in this way...
# 
# **Parent** = mother, father
# 
# **Child** = daughter, son, stepdaughter, stepson
# 
# Some children travelled only with a nanny, therefore parch=0 for them.

# ## Load Libraries

# In[2]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import isnull, when, count, col
from pyspark.ml.feature import VectorAssembler,StringIndexer
from pandas.plotting import scatter_matrix
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import numpy as np


# ## Load Data

# In[3]:


spark = SparkSession.builder.appName('tps-2021-ml-with-pyspark').getOrCreate()
df = spark.read.csv('../input/tabular-playground-series-apr-2021/train.csv', header = True, inferSchema = True)
df.printSchema()


# ## Show Dataset
# Have a peek of the first five observations. In PySpark you can show the data using **show()**

# In[4]:


df.show(5)


# Now, Data grouping by Survived for checking the classes are perfectly balanced!!

# In[5]:


df.groupby('Survived').count().show()


# **oh no!!Survived class not balanced!!**

# ## Summary statistics for numeric variables

# In[6]:


numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']
df.select(numeric_features).describe().show()


# ## Correlations
# 
# Checking Correlations between independent variables
# 

# In[7]:


numeric_data = df.select(numeric_features).toPandas()

axs = scatter_matrix(numeric_data, figsize=(8, 8));

# Rotate axis labels and remove axis ticks
n = len(numeric_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())


# ## Data preparation and feature engineering
# In this part, we will remove unnecessary columns and fill the missing values. Finally, we will select features for ml models. These features will be divided into two parts: train and test.
# 
# Let’s starting the mission 👨‍🚀 

# ### 1. Missing Data Handling:
# 

# In[8]:


df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()


# Oh Shit!! We got lots of missing data!! 
# 
# * In Age columns we got 3292 missing data
# * In Ticket columns we got 4623 missing data
# * In Fare columns we got 134 missing data
# * In Cabin columns we got 67866 missing data
# * In Embarked columns we got 250 missing data
# 
# Now our mission for handling the missing data.

# In[9]:


dataset = df.replace('null', None)\
        .dropna(how='any')


# **Again check missing data**

# In[10]:


dataset.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()


# Wow!! 👌 That’s great now datasets haven’t any missing values.😀

# ## 2. Unnecessary columns dropping
# 

# In[11]:


dataset = dataset.drop('PassengerId')
dataset = dataset.drop('Ticket')
dataset = dataset.drop('Fare')
dataset = dataset.drop('Cabin')
dataset = dataset.drop('Embarked')
dataset = dataset.drop('Name')
dataset.show()


# ## 3. Features Convert into Vector

# In[12]:


sex_in = StringIndexer(inputCol="Sex", outputCol="Sex_encoding")
df1 = sex_in.fit(dataset).transform(dataset)
df1.show()


# In[13]:


required_features = ['Survived',
                    'Pclass',
                    'SibSp',
                    'Parch',
                     'Age',
                     'Sex_encoding'
                   ]

assembler = VectorAssembler(inputCols=required_features, outputCol='features')

transformed_data = assembler.transform(df1)


# In[14]:


transformed_data.show()


# Done!!✌️ Now features are converted into a vector. 🧮

# ## 4. Train and Test Split
# Randomly split data into train and test sets, and set seed for reproducibility.

# In[15]:


(training_data, test_data) = transformed_data.randomSplit([0.8,0.2])


# In[16]:


print("Training Dataset Count: " + str(training_data.count()))
print("Test Dataset Count: " + str(test_data.count()))


# # Machine learning Model Building
# 

# ## 1. Random Forest Classifier
# 
# Random forest is a supervised learning algorithm which is used for both classification and regression cases, as well. But however, it is mainly used for classification problems. As we know that a forest is made up of trees and more trees mean more robust forests, in a similar way, random forest algorithm creates decision trees on data samples and then gets the prediction from each of them and finally selects the best solution by means of voting. It is an ensemble method which is better than a single decision tree because it reduces the over-fitting by averaging the result.

# In[17]:


rf = RandomForestClassifier(labelCol='Survived', 
                            featuresCol='features',
                            maxDepth=5)
model = rf.fit(training_data)
rf_predictions = model.transform(test_data)


# **Evaluate our Random Forest Classifier model.**

# In[18]:


multi_evaluator = MulticlassClassificationEvaluator(labelCol = 'Survived', metricName = 'accuracy')
print('Random Forest classifier Accuracy:', multi_evaluator.evaluate(rf_predictions))


# ## 2. Decision Tree Classifier
# 
# Decision trees are widely used since they are easy to interpret, handle categorical features, extend to the multiclass classification setting, while not requiring feature scaling and are able to capture non-linearities and feature interactions.

# In[19]:


dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'Survived', maxDepth = 3)
dtModel = dt.fit(training_data)
dt_predictions = dtModel.transform(test_data)
dt_predictions.select('Sex_encoding', 'Pclass','SibSp', 'Parch', 'Age', 'Survived').show(10)


# **Evaluate our Decision Tree model.**

# In[20]:


multi_evaluator = MulticlassClassificationEvaluator(labelCol = 'Survived', metricName = 'accuracy')
print('Decision Tree Accuracy:', multi_evaluator.evaluate(dt_predictions))


# ## 3. Gradient-boosted Tree classifier Model
# 
# Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.

# In[21]:


gb = GBTClassifier(labelCol = 'Survived', featuresCol = 'features')
gbModel = gb.fit(training_data)
gb_predictions = gbModel.transform(test_data)


# **Evaluate our Gradient-Boosted Tree Classifier.**

# In[22]:


multi_evaluator = MulticlassClassificationEvaluator(labelCol = 'Survived', metricName = 'accuracy')
print('Gradient-boosted Trees Accuracy:', multi_evaluator.evaluate(gb_predictions))


# # Conclusion
# 
# PySpark is a great language for data scientists to learn because it enables scalable analysis and ML pipelines. If you’re already familiar with Python and Pandas, then much of your knowledge can be applied to Spark. To sum it up, we have learned **EDA & how to build a machine learning application using PySpark**. We tried two algorithms and **gradient boosting performed best on our data set.**

# # Make Sumbission file

# In[23]:


df_pd = gb_predictions.toPandas()


# In[24]:


df_pd.head()


# In[25]:


df_pd.to_csv("gbModel.csv",index = False)


# In[26]:


gb_df = pd.read_csv("./gbModel.csv")


# In[27]:


gb_df.head()


# In[28]:


submission  = pd.read_csv("../input/tabular-playground-series-apr-2021/sample_submission.csv")
submission = submission.drop("Survived",axis=1)
submission.head()


# In[29]:


results = pd.concat([submission,gb_df.Survived],axis = 1)


# In[30]:


results.to_csv("submission.csv", index = False)


# > Machine learning models sparking when **PySpark** gave the accelerator gear like the **Need for Speed** gaming cars.

# # References:
# 1. [Building a Machine Learning (ML) Model with PySpark](https://towardsdatascience.com/first-time-machine-learning-model-with-pyspark-3684cf406f54)
# 2. [Apache Spark 3.1.1](http://spark.apache.org/docs/latest/ml-guide.html)
# 3. [A Must-Read Guide on How to Work with PySpark on Google Colab for Data Scientists!](https://www.analyticsvidhya.com/blog/2020/11/a-must-read-guide-on-how-to-work-with-pyspark-on-google-colab-for-data-scientists/)
# 4. [Learning basic ML by Titanic Survival Prediction](https://www.kaggle.com/harunshimanto/learning-basic-ml-by-titanic-survival-prediction)
