#!/usr/bin/env python
# coding: utf-8

# # Awesome Data Science for Beginners with Roadmap and Titanic Exploration
# 
# ### (Start Here) (TBU)
# 
#     This kernel has the curated list of Awesome Data Science Beginners's Resources with Roadmap and Some Exploratory Data Analysis of Titanic Disaster 
#     
# If you want to know more about Data Science but don't know where to start this list is for you!
# 
# No previous knowledge required but Python and statistics basics will definitely come in handy. These resources have been used successfully for many beginners at Data Science student groups.
#  
# > #### **Credits**: Thanks to **Practical AI - Goku Mohandas**, **Data Science University** and other contributers for such wonderful work!
# 
# ### Here are some of *my kernel notebooks* for **Machine Learning and Data Science** as follows, ***Upvote*** them if you *like* them
# 
# > * [Awesome Deep Learning Basics and Resources](https://www.kaggle.com/arunkumarramanan/awesome-deep-learning-resources)
# > * [Data Science with R - Awesome Tutorials](https://www.kaggle.com/arunkumarramanan/data-science-with-r-awesome-tutorials)
# > * [Data Science and Machine Learning Cheetcheets](https://www.kaggle.com/arunkumarramanan/data-science-and-machine-learning-cheatsheets)
# > * [Awesome ML Frameworks and MNIST Classification](https://www.kaggle.com/arunkumarramanan/awesome-machine-learning-ml-frameworks)
# > * [Tensorflow Tutorial and House Price Prediction](https://www.kaggle.com/arunkumarramanan/tensorflow-tutorial-and-examples)
# > * [Data Scientist's Toolkits - Awesome Data Science Resources](https://www.kaggle.com/arunkumarramanan/data-scientist-s-toolkits-awesome-ds-resources)
# > * [Awesome Computer Vision Resources (TBU)](https://www.kaggle.com/arunkumarramanan/awesome-computer-vision-resources-to-be-updated)
# > * [Machine Learning and Deep Learning - Awesome Tutorials](https://www.kaggle.com/arunkumarramanan/awesome-deep-learning-ml-tutorials)
# > * [Data Science with Python - Awesome Tutorials](https://www.kaggle.com/arunkumarramanan/data-science-with-python-awesome-tutorials)
# > * [Awesome TensorFlow and PyTorch Resources](https://www.kaggle.com/arunkumarramanan/awesome-tensorflow-and-pytorch-resources)
# > * [Awesome Data Science IPython Notebooks](https://www.kaggle.com/arunkumarramanan/awesome-data-science-ipython-notebooks)
# > * [Machine Learning Engineer's Toolkit with Roadmap](https://www.kaggle.com/arunkumarramanan/machine-learning-engineer-s-toolkit-with-roadmap) 
# > * [Hands-on ML with scikit-learn and TensorFlow](https://www.kaggle.com/arunkumarramanan/hands-on-ml-with-scikit-learn-and-tensorflow)
# > * [Practical Machine Learning with PyTorch](https://www.kaggle.com/arunkumarramanan/practical-machine-learning-with-pytorch)
# > * [Awesome Data Science for Beginners with Titanic Exploration](https://kaggle.com/arunkumarramanan/awesome-data-science-for-beginners)
# 

# # Titanic Exploration
# In this notebook, we'll do some exploratory data analysis with Titanic Disaster Competition.
# 
# ### Uploading the data
# 
# We're first going to get some data to play with. We're going to load the titanic dataset from the getting started competition below.
# 
# ### Loading the data
# 
# Now that we have some data to play with, let's load into a Pandas dataframe. Pandas is a great python library for data analysis.****

# In[1]:


import pandas as pd


# In[2]:


# Read from CSV to Pandas DataFrame
df = pd.read_csv("../input/train.csv", header=0)


# In[3]:


# First five items
df.head()


# These are the diferent features: 
# * pclass: class of travel
# * name: full name of the passenger
# * sex: gender
# * age: numerical age
# * sibsp: # of siblings/spouse aboard
# * parch: number of parents/child aboard
# * ticket: ticket number
# * fare: cost of the ticket
# * cabin: location of room
# * emarked: port that the passenger embarked at (C - Cherbourg, S - Southampton, Q = Queenstown)
# * survived: survial metric (0 - died, 1 - survived)

# ### Exploratory Dats Analysis EDA
# 
# We're going to explore the Pandas library and see how we can explore and process our data.

# In[4]:


# Describe features
df.describe()


# In[5]:


# Histograms
df["Age"].hist()


# In[6]:


# Unique values
df["Embarked"].unique()


# In[7]:


# Selecting data by feature
df["Name"].head()


# In[8]:


# Filtering
df[df["Sex"]=="female"].head() # only the female data appear


# In[9]:


# Sorting
df.sort_values("Age", ascending=False).head()


# In[10]:


# Grouping
sex_group = df.groupby("Survived")
sex_group.mean()


# In[11]:


# Selecting row
df.iloc[0, :] # iloc gets rows (or columns) at particular positions in the index (so it only takes integers)


# In[12]:


# Selecting specific value
df.iloc[0, 1]


# In[13]:


# Selecting by index
df.loc[0] # loc gets rows (or columns) with particular labels from the index


# ### Data Preprocessing

# In[14]:


# Rows with at least one NaN value
df[pd.isnull(df).any(axis=1)].head()


# In[15]:


# Drop rows with Nan values
df = df.dropna() # removes rows with any NaN values
df = df.reset_index() # reset's row indexes in case any rows were dropped
df.head()


# In[16]:


# Dropping multiple rows
df = df.drop(["Name", "Cabin", "Ticket"], axis=1) # we won't use text features for our initial basic models
df.head()


# In[17]:


# Map feature values
df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df["Embarked"] = df['Embarked'].dropna().map( {'S':0, 'C':1, 'Q':2} ).astype(int)
df.head()


# ### Feature Engineering

# In[18]:


# Lambda expressions to create new features
def get_family_size(sibsp, parch):
    family_size = sibsp + parch
    return family_size

df["Family_Size"] = df[["SibSp", "Parch"]].apply(lambda x: get_family_size(x["SibSp"], x["Parch"]), axis=1)
df.head()


# In[19]:


# Reorganize headers
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Family_Size', 'Fare', 'Embarked', 'Survived']]
df.head()


# ### Saving data

# In[20]:


# Saving dataframe to CSV
df.to_csv("processed_titanic.csv", index=False)


# In[21]:


# See your saved file
get_ipython().system('ls -l')


# ### End of the exploration

# # Awesome Data Science for Beginners with Roadmap
# 
# ## Contents
# 
# - [What is Data Science?](#what-is-data-science)
# - [Common Algorithms and Procedures](#common-algorithms-and-procedures)
# - [Data Science using Python](#data-science-using-python)
# - [Data Science Challenges for Beginners](#data-science-challenges-for-beginners)
# - [Data Science and Engineering your way](#data-science-and-engineering-your-way)
# - [More advanced resources and lists](#more-advanced-resources-and-lists)
# 
# ## What is Data Science?
# 
# - ['What is Data Science?' on Quora](https://www.quora.com/What-is-data-science)
# - [Explanation of important vocabulary](https://www.quora.com/What-is-the-difference-between-Data-Analytics-Data-Analysis-Data-Mining-Data-Science-Machine-Learning-and-Big-Data-1?share=1) - Differentiation of Big Data, Machine Learning, Data Science.
# 
# ## Common Algorithms and Procedures
# 
# - [Supervised vs unsupervised learning](https://stackoverflow.com/questions/1832076/what-is-the-difference-between-supervised-learning-and-unsupervised-learning) - The two most common types of Machine Learning algorithms. 
# - [9 important Data Science algorithms and their implementation](https://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.05-Naive-Bayes.ipynb) 
# - [Cross validation](https://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.03-Hyperparameters-and-Model-Validation.ipynb) - Evaluate the performance of your algorithm / model.
# - [Feature engineering](https://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.04-Feature-Engineering.ipynb) - Modifying the data to better model predictions.
# - [Scientific introduction to 10 important Data Science algorithms](http://www.cs.umd.edu/%7Esamir/498/10Algorithms-08.pdf)
# - [Model ensemble: Explanation](https://www.analyticsvidhya.com/blog/2017/02/introduction-to-ensembling-along-with-implementation-in-r/) - Combine multiple models into one for better performance.
# 
# ## Data Science using Python
# This list covers only Python, as many are already familiar with this language. [Data Science tutorials using R](https://github.com/ujjwalkarn/DataScienceR).
# 
# ### Learning Python
# 
# - [YouTube tutorial series by sentdex](https://www.youtube.com/watch?v=oVp1vrfL_w4&list=PLQVvvaa0QuDe8XSftW-RAxdo6OmaeL85M)
# - [Interactive Python tutorial website](http://www.learnpython.org/)
# 
# ### numpy
# [numpy](http://www.numpy.org/) is a Python library which provides large multidimensional arrays and fast mathematical operations on them.
# 
# - [Numpy tutorial on DataCamp](https://www.datacamp.com/community/tutorials/python-numpy-tutorial#gs.h3DvLnk)
# 
# ### pandas
# [pandas](http://pandas.pydata.org/index.html) provides efficient data structures and analysis tools for Python. It is build on top of numpy.
# 
# - [Introduction to pandas](http://www.synesthesiam.com/posts/an-introduction-to-pandas.html)
# - [DataCamp pandas foundations](https://www.datacamp.com/courses/pandas-foundations) - Paid course, but 30 free days upon account creation (enough to complete course).
# - [Pandas cheatsheet](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf) - Quick overview over the most important functions.
# 
# ### scikit-learn
# [scikit-learn](http://scikit-learn.org/stable/) is the most common library for Machine Learning and Data Science in Python.
# 
# - [Introduction and first model application](https://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.02-Introducing-Scikit-Learn.ipynb)
# - [Rough guide for choosing estimators](http://scikit-learn.org/stable/tutorial/machine_learning_map/)
# - [Scikit-learn complete user guide](http://scikit-learn.org/stable/user_guide.html)
# - [Model ensemble: Implementation in Python](http://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/)
# 
# ### Jupyter Notebook
# [Jupyter Notebook](https://jupyter.org/) is a web application for easy data visualisation and code presentation.
# 
# - [Downloading and running first Jupyter notebook](https://jupyter.org/install.html)
# - [Example notebook for data exploration](https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-instacart)
# - [Seaborn data visualization tutorial](https://elitedatascience.com/python-seaborn-tutorial) - Plot library that works great with Jupyter.
# 
# 
# ### Various other helpful tools and resources
# 
# - [Template folder structure for organizing Data Science projects](https://github.com/drivendata/cookiecutter-data-science)
# - [Anaconda Python distribution](https://www.continuum.io/downloads) - Contains most of the important Python packages for Data Science.
# - [Natural Language Toolkit](http://www.nltk.org/) - Collection of libraries for working with text-based data.
# - [LightGBM gradient boosting framework](https://github.com/Microsoft/LightGBM) - Successfully used in many Kaggle challenges.
# - [Amazon AWS](https://aws.amazon.com/) - Rent cloud servers for more timeconsuming calculations (r4.xlarge server is a good place to start).
# 
# ## Data Science Challenges for Beginners
# Sorted by increasing complexity.
# 
# - [Walkthrough: House prices challenge](https://www.dataquest.io/blog/kaggle-getting-started/) - Walkthrough through a simple challenge on house prices.
# - [Blood Donation Challenge](https://www.drivendata.org/competitions/2/warm-up-predict-blood-donations/) - Predict if a donor will donate again.
# - [Titanic Challenge](https://www.kaggle.com/c/titanic) - Predict survival on the Titanic.
# - [Water Pump Challenge](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) - Predict the operating condition of water pumps in Africa.
# 
# ## Data Science and Engineering your way
# 
# ##### An introduction to different Data Science and Engineering concepts and Applications using Python and R  
# 
# These series of tutorials on Data Science engineering will try to compare how different concepts in the discipline can be implemented in the two dominant ecosystems nowadays: R and Python.  
# 
# We will do this from a neutral point of view. Our opinion is that each environment has good and bad things, and any data scientist should know how to use both in order to be as prepared as posible for job market or to start personal project.    
# 
# To get a feeling of what is going on regarding this hot topic, we refer the reader to [DataCamp's Data Science War](http://blog.datacamp.com/r-or-python-for-data-analysis/) infographic. Their infographic explores what the strengths of **R** are over **Python** and vice versa, and aims to provide a basic comparison between these two programming languages from a data science and statistics perspective.  
# 
# Far from being a repetition from the previous, our series of tutorials will go hands-on into how to actually perform different data science taks such as working with data frames, doing aggregations, or creating different statistical models such in the areas of supervised and unsupervised learning.  
# 
# We will use real-world datasets, and we will build some real data products. This will help us to quickly transfer what we learn here to actual data analysis situations.  
# 
# If your are interested in Big Data products, then you might find interesting our series of [tutorials on using Apache Spark and Python](https://github.com/jadianes/spark-py-notebooks) or [using R on Apache Spark (SparkR)](https://github.com/jadianes/spark-r-notebooks).  
# 
# ## Tutorials
# 
# This is a growing list of tutorials explaining concepts and applications in Python and R. 
# 
# ### [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)
# 
# Machine Learning ML Crash Course with TensorFlow APIs is highly recommended by Google as it's developed by googlers.
# 
# ### [Introduction to Data Frames](https://github.com/jadianes/data-science-your-way/blob/master/01-data-frames/README.md)  
# 
# An introduction to the basic data structure and how to use it in Python/Pandas and R.  
# 
# ### [Exploratory Data Analysis](https://github.com/jadianes/data-science-your-way/blob/master/02-exploratory-data-analysis/README.md)    
# 
# About this important task in any data science engineering project.  
# 
# ### [Dimensionality Reduction and Clustering](https://github.com/jadianes/data-science-your-way/blob/master/03-dimensionality-reduction-and-clustering/README.md)    
# About using Principal Component Analysis and k-means Clustering to better represent and understand our data.  
# 
# ### [Text Mining and Sentiment Classification](https://github.com/jadianes/data-science-your-way/blob/master/04-sentiment-analysis/README.md)    
# 
# How to use text mining techniques to analyse the positive or non-positive sentiment of text documents using just *linear methods*.  
# 
# ## Applications  
# 
# These are some of the applications we have built using the concepts explained in the tutorials.  
# 
# ### [A web-based Sentiment Classifier using R and Shiny](https://github.com/jadianes/data-science-your-way/blob/master/apps/sentimentclassifier/README.md)  
# 
# How to build a web applications where we can upload text documents to be sentiment-analysed using the R-based framework [Shiny](http://shiny.rstudio.com/).  
# 
# ### [Building Data Products with Python](https://github.com/jadianes/data-science-your-way/blob/master/apps/winerama/README.md)  
# 
# Using a [wine reviews and recommendations website](http://jadianes.koding.io:8000/reviews/) as a leitmotif, this series of tutorials, with [its own separate repository](https://github.com/jadianes/winerama-recommender-tutorial) tagged by lessons, digs into how to use Python technologies such as Django, Pandas, or Scikit-learn, in order to build data products.   
# 
# ### [Red Wine Quality Data analysis with R](https://github.com/jadianes/data-science-your-way/blob/master/apps/wine-quality-data-analysis/README.md)  
# 
# Using R and ggplot2, we perform Exploratory Data Analysis of this reference dataset about wine quality.    
# 
# ### [Information Retrieval algorithms with Python](https://github.com/jadianes/data-science-your-way/blob/master/apps/information-retrieval/README.md)  
# 
# Where we show our own implementation of a couple of Information Retrieval algorithms: vector space model, and tf-idf.  
# 
# ### [Kaggle - The Analytics Edge (Spring 2015)](https://github.com/jadianes/data-science-your-way/blob/master/apps/kaggle-analytics-edge-15/)  
# 
# My solution to this Kaggle competition. It was part of the edX MOOC [The Analitics Edge](https://www.edx.org/course/analytics-edge-mitx-15-071x-0). I highly recommend this on-line course. It is one of the most applied I have ever taken about using R for data anlysis and machine learning.  
# 
# ## More advanced resources and lists
# 
# - [Awesome Data Science with Python](https://www.kaggle.com/arunkumarramanan/data-science-with-python-awesome-tutorials)
# - [Awesome Data Science with R](https://www.kaggle.com/arunkumarramanan/data-science-with-r-awesome-tutorials)
# - [Machine Learning & Deep Learning Awesome Tutorials](https://www.kaggle.com/arunkumarramanan/ml-and-deep-learning-awesome-tutorials)
# - [Awesome Data Science Resources](https://www.kaggle.com/arunkumarramanan/awesome-data-science-resources)
# 
# # Data Science University (Awesome Data Science MOOCs)
#    An Open Source Society University: 📊 Path to a free self-taught education in Data Science! 
#      
#      Data Science University - Awesome MOOCs (Massive Open Online Courses)
#     
# ## Contents
# 
# - [About](#about)
# - [Motivation & Preparation](#motivation--preparation)
# - [Curriculum](#curriculum)
# - [How to use this guide](#how-to-use-this-guide)
# - [Prerequisite](#prerequisite)
# - [References](#references)
# 
# ## About
# 
# This is a **solid path** for those of you who want to complete a **Data Science** course on your own time, **for free**, with courses from the **best universities** in the World.
# 
# In our curriculum, we give preference to MOOC (Massive Open Online Course) style courses because these courses were created with our style of learning in minds.
# 
# ## Motivation & Preparation
# 
# Here are two interesting links that can make **all** the difference in your journey.
# 
# The first one is a motivational video that shows a guy that went through the "MIT Challenge", which consists of learning the entire **4-year** MIT curriculum for Computer Science in **1 year**.
# 
# - [MIT Challenge](https://www.scotthyoung.com/blog/myprojects/mit-challenge-2/)
# 
# The second link is a MOOC that will teach you learning techniques used by experts in art, music, literature, math, science, sports, and many other disciplines. These are **fundamental abilities** to succeed in our journey.
# 
# - [Learning How to Learn](https://www.coursera.org/learn/learning-how-to-learn)
# 
# **Are you ready to get started?**
# 
# ## Curriculum
# 
# - [Linear Algebra](#linear-algebra)
# - [Single Variable Calculus](#single-variable-calculus)
# - [Multivariable Calculus](#multivariable-calculus)
# - [Python](#python)
# - [Probability and Statistics](#probability-and-statistics)
# - [Introduction to Data Science](#introduction-to-data-science)
# - [Machine Learning](#machine-learning)
# - [Project](#project)
# - [Convex Optimization](#convex-optimization)
# - [Data Wrangling](#data-wrangling)
# - [Big Data](#big-data)
# - [Database](#database)
# - [Deep Learning](#deep-learning)
# - [Natural Language Processing](#natural-language-processing)
# - [Capstone Project](#capstone-project)
# - [Specializations](#specializations)
# 
# 
# ---
# 
# ### Linear Algebra
# 
# Courses | Duration | Effort
# :-- | :--: | :--:
# [Linear Algebra - Foundations to Frontiers](https://www.edx.org/course/linear-algebra-foundations-frontiers-utaustinx-ut-5-04x#!)| 15 weeks | 8 hours/week
# [Applications of Linear Algebra Part 1](https://www.edx.org/course/applications-linear-algebra-part-1-davidsonx-d003x-1)| 5 weeks | 4 hours/week
# [Applications of Linear Algebra Part 2](https://www.edx.org/course/applications-linear-algebra-part-2-davidsonx-d003x-2)| 4 weeks | 5 hours/week
# 
# ### Single Variable Calculus
# Courses | Duration | Effort
# :-- | :--: | :--:
# [Calculus 1A: Differentiation](https://www.edx.org/course/calculus-1a-differentiation-mitx-18-01-1x)| 13 weeks | 6-10 hours/week
# [Calculus 1B: Integration](https://www.edx.org/course/calculus-1b-integration-mitx-18-01-2x)| 13 weeks | 5-10 hours/week
# [Calculus 1C: Coordinate Systems & Infinite Series](https://www.edx.org/course/calculus-1c-coordinate-systems-infinite-mitx-18-01-3x)| 13 weeks | 6-10 hours/week
# 
# ### Multivariable Calculus
# Courses | Duration | Effort
# :-- | :--: | :--:
# [MIT OCW Multivariable Calculus](http://ocw.mit.edu/courses/mathematics/18-02sc-multivariable-calculus-fall-2010/index.htm)| 15 weeks | 8 hours/week
# 
# ### Python
# Courses | Duration | Effort
# :-- | :--: | :--:
# [Introduction to Computer Science and Programming Using Python](https://www.edx.org/course/introduction-computer-science-mitx-6-00-1x-7)| 9 weeks | 15 hours/week
# [Introduction to Computational Thinking and Data Science](https://www.edx.org/course/introduction-computational-thinking-data-mitx-6-00-2x-3)| 10 weeks | 15 hours/week
# [Introduction to Python for Data Science](https://prod-edx-mktg-edit.edx.org/course/introduction-python-data-science-microsoft-dat208x-1)| 6 weeks | 2-4 hours/week
# [Programming with Python for Data Science](https://www.edx.org/course/programming-python-data-science-microsoft-dat210x)| 6 weeks | 3-4 hours/week
# 
# ### Probability and Statistics
# Courses | Duration | Effort
# :-- | :--: | :--:
# [Introduction to Probability](https://www.edx.org/course/introduction-probability-science-mitx-6-041x-1#.U3yb762SzIo)| 16 weeks | 12 hours/week
# [Statistical Reasoning](https://lagunita.stanford.edu/courses/OLI/StatReasoning/Open/about)| - weeks | - hours/week
# [Introduction to Statistics: Descriptive Statistics](https://www.edx.org/course/introduction-statistics-descriptive-uc-berkeleyx-stat2-1x)| 5 weeks | - hours/week
# [Introduction to Statistics: Probability](https://www.edx.org/course/introduction-statistics-probability-uc-berkeleyx-stat2-2x)| 5 weeks | - hours/week
# [Introduction to Statistics: Inference](https://www.edx.org/course/introduction-statistics-inference-uc-berkeleyx-stat2-3x)| 5 weeks | - hours/week
# 
# ### Introduction to Data Science
# Courses | Duration | Effort
# :-- | :--: | :--:
# [Introduction to Data Science](https://www.coursera.org/course/datasci)| 8 weeks | 10-12 hours/week
# [Data Science - CS109 from Harvard](http://cs109.github.io/2015/)| 12 weeks | 5-6 hours/week
# [The Analytics Edge](https://www.edx.org/course/analytics-edge-mitx-15-071x-2)| 12 weeks | 10-15 hours/week
# 
# ### Machine Learning
# Courses | Duration | Effort
# :-- | :--: | :--:
# [Learning From Data (Introductory Machine Learning)](https://www.edx.org/course/learning-data-introductory-machine-caltechx-cs1156x)    [[caltech]](http://work.caltech.edu/lectures.html) | 10 weeks | 10-20 hours/week
# [Statistical Learning](https://lagunita.stanford.edu/courses/HumanitiesSciences/StatLearning/Winter2016/about)| - weeks | 3 hours/week
# [Stanford's Machine Learning Course](https://www.coursera.org/learn/machine-learning)| - weeks | 8-12 hours/week
# [Google Machine Learning Crash Course with TensorFlow APIs](https://developers.google.com/machine-learning/crash-course/)| - weeks | 15 hours
# 
# ### Project
# Complete Kaggle's Getting Started and Playground Competitions
# 
# 
# ### Convex Optimization
# Courses | Duration | Effort
# :-- | :--: | :--:
# [Convex Optimization](https://lagunita.stanford.edu/courses/Engineering/CVX101/Winter2014/about)| 9 weeks | 10 hours/week
# 
# ### Data Wrangling
# Courses | Duration | Effort
# :-- | :--: | :--:
# [Data Wrangling with MongoDB](https://www.udacity.com/course/data-wrangling-with-mongodb--ud032)| 8 weeks | 10 hours/week
# 
# ### Big Data
# Courses | Duration | Effort
# :-- | :--: | :--:
# [Intro to Hadoop and MapReduce](https://www.udacity.com/course/intro-to-hadoop-and-mapreduce--ud617)| 4 weeks | 6 hours/week
# [Deploying a Hadoop Cluster](https://www.udacity.com/course/deploying-a-hadoop-cluster--ud1000)| 3 weeks | 6 hours/week
# 
# ### Database
# Courses | Duration | Effort
# :-- | :--: | :--:
# [Stanford's Database course](https://lagunita.stanford.edu/courses/DB/2014/SelfPaced/about)| - weeks | 8-12 hours/week
# 
# ### Natural Language Processing
# Courses | Duration | Effort
# :-- | :--: | :--:
# [Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/)| - weeks | - hours/week
# 
# ### Deep Learning
# Courses | Duration | Effort
# :-- | :--: | :--:
# [Deep Learning](https://www.udacity.com/course/deep-learning--ud730)| 12 weeks | 8-12 hours/week
# 
# ### Capstone Project
# - Participate in Kaggle competition
# - List down other ideas
# 
# 
# ### Specializations
# 
# After finishing the courses above, start your specializations on the topics that you have more interest.
# You can view a list of available specializations [here](https://github.com/open-source-society/data-science/blob/master/extras/specializations.md).
# 
# ![keep learning](http://i.imgur.com/REQK0VU.jpg)
# 
# ## How to use this guide
# 
# ### Order of the classes
# 
# This guide was developed to be consumed in a linear approach. What does this mean? That you should complete one course at a time.
# 
# The courses are **already** in the order that you should complete them. Just start in the [Linear Algebra](#linear-algebra) section and after finishing the first course, start the next one.
# 
# **If the course isn't open, do it anyway with the resources from the previous class.**
# 
# ### Should I take all courses?
# 
# **Yes!** The intention is to conclude **all** the courses listed here!
# 
# ### Duration of the project
# 
# It may take longer to complete all of the classes compared to a  regular Data Science course, but I can **guarantee** you that your **reward** will be proportional to **your motivation/dedication**!
# 
# You must focus on your **habit**, and **forget** about goals. Try to invest 1 ~ 2 hours **every day** studying this curriculum. If you do this, **inevitably** you'll finish this curriculum.
# 
# > See more about "Commit to a process, not a goal" [here](http://jamesclear.com/goals-systems).
# 
# ### Project Based
# 
# Here in **Data Science University**, you do **not** need to take exams, because we are focused on **real projects**!
# 
# In order to show for everyone that you **successfully** finished a course, you should create a **real project**.
# 
# > "What does it mean?"
# 
# After finish a course, you should think about a **real world problem** that you can solve using the acquired knowledge in the course. You don't need to create a big project, but you must create something to **validate** and **consolidate** your knowledge, and also to show to the world that you are capable to create something useful with the concepts that you learned.
# 
# The projects of all students will be listed in [this](https://github.com/open-source-society/data-science/blob/master/PROJECTS.md) file. Submit your project's information in that file after you conclude it.
# 
# **You can create this project alone or with other students!**
# 
# #### Project Suggestions
# 
# 
# And you should also...
# 
# ### Be creative!
# 
# This is a **crucial** part of your journey through all those courses.
# 
# You **need** to have in mind that what you are able to **create** with the concepts that you learned will be your certificate **and this is what really matters**!
# 
# In order to show that you **really** learned those things, you need to be **creative**!
# 
# Here are some tips about how you can do that:
# 
# - **Articles**: create blog posts to synthesize/summarize what you learned.
# - **GitHub repository**: keep your course's files organized in a GH repository, so in that way other students can use it to study with your annotations.
# 
# ### Which programming languages should I use?
# 
# Python and R are heavily used in Data Science community and our courses teach you both, but...
# 
# The **important** thing for each course is to **internalize** the **core concepts** and to be able to use them with whatever tool (programming language) that you wish.
# 
# [Be creative](#be-creative) in order to show your progress! :smile:
# 
# ### Stay tuned
# 
# UPVOTE for futures improvements and general information.
# 
# ## Prerequisite
# 
# The **only things** that you need to know are how to use **Git** and **GitHub**. Here are some resources to learn about them:
# 
# **Note**: Just pick one of the courses below to learn the basics. You will learn a lot more once you get started!
# 
# - [Try Git](https://try.github.io/levels/1/challenges/1)
# - [Git - the simple guide](http://rogerdudler.github.io/git-guide/)
# - [GitHub Training & Guides](https://www.youtube.com/playlist?list=PLg7s6cbtAD15G8lNyoaYDuKZSKyJrgwB-)
# - [GitHub Hello World](https://guides.github.com/activities/hello-world/)
# - [Git Immersion](http://gitimmersion.com/index.html)
# - [How to Use Git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775)

# ## Credits (Reference)
# 
# > - [practicalAI - Goku Mohandas](https://github.com/GokuMohandas/practicalAI/)
# > - [Data Science University](https://github.com/ossu/data-science)
# > - [GitHub Awesome Lists Topic](https://github.com/topics/awesome)
# > - [Awesome Learn Data science](https://github.com/siboehm/awesome-learn-datascience)
# 
# ## License
# 
# [![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
# 
# ### Please ***UPVOTE*** my kernel if you like it or wanna fork it.
# 
# ##### Feedback: If you have any ideas or you want any other content to be added to this curated list, please feel free to make any comments to make it better.
# #### I am open to have your *feedback* for improving this ***kernel***
# ###### Hope you enjoyed this kernel!
# 
# ### Thanks for visiting my *Kernel* and please *UPVOTE* to stay connected and follow up the *further updates!*
