#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}\n\n')


# <div align='center'><font size="5" color="#1E8449">An analysis of the 2020 Kaggle ML and DS Survey for the adoption of Automated Machine Learning in the industry</font></div>
# 
# <hr>
# 
# <img src='https://image.freepik.com/free-vector/dashboard-consolidating-metrics-computer-screen-business-intelligence-dashboard-business-analytics-tool-business-intelligence-metrics-concept_335657-1890.jpg' width=600>
# <div align="center"><font size="2">Source: <a href="https://www.freepik.com/vectors/computer">Computer vector created by vectorjuice - www.freepik.com</a></font></div>  
# 

# >  ### *AI and machine learning is still a field with high barriers to entry that requires expertise and resources that few companies can afford on their own: Fei-Fei Li*
# 
# 
# Artificial Intelligence(AI) has the potential to make a big difference in every facet of our lives, especially in areas like healthcare, education, and environmental conservation. However, many enterprises still struggle to deploy machine learning solutions effectively. It is primarily due to the issues of talent, time, and trust, which is prevalent in many businesses.
# 
# ![](https://cdn-images-1.medium.com/max/800/1*ykOYaXTmv8Mrb8uYYESMfA.png)
# 
# 
# * The traditional machine learning(ML) process heavily relies on human expertise. As a result, before starting on the ML journey, a company needs to invest in expert data scientists, researchers, and mathematicians. Unfortunately, there is a considerable **talent gap** with an [acute shortage of experienced and seasoned data scientists in the industry today](http://https://www.theverge.com/2017/12/5/16737224/global-ai-talent-shortfall-tencent-report). 
# * Secondly, **time** is of the essence here. When machine learning solutions drive business decisions, it is crucial to get the results quickly. Some of the current ML solutions take months to deploy, which affects their outcomes. Also, due to the heavy manual dependence, there are chances of errors creeping in the workflow. 
# * Finally, it is imperative to tackle the issue of **trust**. A lot of companies fail to translate model predictions into understandable terms for stakeholders. Although there are systems in place for interpretability and explainability in conventional ML systems, lack of knowledge and experience makes the implementation hard.
# 
# 
# AutoML is an effort towards democratizing machine learning by making its power available to everybody, rather than a select few. AutoML enables people with diverse skillsets to work on ML problems. By automating the repetitive tasks, it allows data scientists to focus on essential aspects of ML pipeline like data gathering and model deployment. 
# 
# 
# <font color='#1E8449' size=6>Motivation and Methodology</font><br>
# 
# This notebook is a deep dive into the current AutoML solutions and preactises. The idea is to:
# * Utilise the Kaggle survey data to see how the AutoML solutions have penetrated the AI ecosystem and what the future holds for this industry. 
# * Secondly, **I work as a Data Science Evangelist for H2O.ai** - An AutoML company,Hence, this analysis will be a way for me to understand how Kagglers' in particluar engage with AutoML tools. 
# 
# In this notebook, we shall analyze the AutoML usage and adoption in the survey under six different categories, namely:
# ![](https://imgur.com/oQTjHhe.png)
# *****
# 
# <div id="101"></div>
# 
# <font color='#1E8449' size=6>Contents</font><br>
# 
# <a href="#1">1. Automated Machine Learning : An Introduction</a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#11">- 1.1 Background and History</a>   
# 
# <a href="#2">2. An Overview of the Respondents</a>  
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#21">- 2.1 Total number of Survey respondents</a>   
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#22">- 2.2 People who responded to the AutoML question in 2020</a>   
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#23">- 2.3 AutoML usage by respondents</a> 
# 
# <a href="#3">3. Common categories and types of AutoML tools</a>  
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#31">- 3.1 AutoML - Category wise usage in 2020</a>   
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#32">- 3.2 AutoML Categories frequently used together</a>   
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#33">- 3.3 AutoML tools and their usage</a> 
# 
# <a href="#4">4. Analysis of AutoML Users</a>  
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#41">- 4.1 Current Job Titles</a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#411">- 4.1.1 Job titles vs AutoML Categories usage</a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#412">- 4.1.2 Job titles vs AutoML Tools usage</a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#42">- 4.2 Machine Learning Experience</a>  
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#421">- 4.2.1 Machine Learning experience vs AutoML Tools usage</a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#422">- 4.2.2 Machine Learning experience vs Job Title</a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#43">- 4.3 Coding Experience</a> 
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#431">- 4.2.1 Coding experience vs AutoML Tools usage</a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#432">- 4.2.2 Coding experience vs Machine Learning experience </a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#44">- 4.4 Job Role</a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#45">- 4.5 AutoML Users Personas</a>
# 
# <a href="#5">5. Companywise AutoML usage analysis</a>  
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#51">- 5.1 Company size of the AutoML users</a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#52">- 5.2 Machine learning maturity in business</a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#53">- 5.3 Data Science Team Size</a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#531">- 5.3.1 Relationship between Company Size and Data Science Team Size</a>
# 
# <a href="#6">6. Infrastructure</a> 
# <br>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#61">- 6.1 Money spent on Cloud Computing in 2020</a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#611">- 6.1.1 A comparison between expenditure in 2020 and 2019</a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#62">- 6.2 Money spent vs Company size</a>
# 
# <a href="#7">7. AutoML users around the globe</a>  
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#71">- 7.1 AutoML - Countrywise adoption in 2020</a>   
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#72">- 7.2 AutoML - Regionwise adoption in 2020</a>   
# 
# <a href="#8">8. Social Media Analysis : Twitter and Google Trends</a>  
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#81">- 8.1 Google Trends data analysis</a> 
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#811">- 8.1.1 Interest over time</a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#812">- 8.1.2 Interest over Region</a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#813">- 8.1.3 Related Topics and Queries related to AutoML</a> 
# <br>&nbsp;&nbsp;&nbsp;&nbsp;<a href="#82">- 8.2 Twitter data analysis</a> 
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#821">- 8.2.1 Common Words Found in Tweets</a>
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#822">- 8.2.2 Exploring Co-occurring Words (Bigrams)</a>
# 
# <a href="#9">9. Conclusion</a> 
# 
# <a href="#10">10. References</a> 
# 

# *****
# 
# <div id="1"></div>
# 
# <font color='#1E8449' size=6 >1. Automated Machine Learning : An Introduction</font><br> 
# 
# Automated Machine Learning, also known as AutoML, is the process of automating the end to end process of applying machine learning to real-world problems. A typical machine learning process consists of several steps, including ingesting and preprocessing data, feature engineering, model training, and deployment. In conventional machine learning, every step in this pipeline is monitored and executed by humans. [Tools for automatic machine learning (AutoML) aims to automate one or more stages of these machine learning pipelines](https://arxiv.org/abs/2002.04803) making it easier for non-experts to build machine learning models while removing repetitive tasks and enabling seasoned
# machine learning engineers to build better models faster.
# 
# ![](https://opendatascience.com/wp-content/uploads/2019/07/Screen-Shot-2019-07-15-at-11.43.38-AM.png)

# <div id="11"></div>
# 
# <font color='#1E8449' size=5 >1.1 Background and History</font><br>  
# 
# AutoML solutions have been around for quite some time now. The early AutoML solutions like [AutoWeka](https://www.cs.ubc.ca/labs/beta/Projects/autoweka/) [originated in academia](https://arxiv.org/pdf/1908.05557.pdf) in 2013, followed by [Auto-sklearn](https://automl.github.io/auto-sklearn/master/) and [TPOT](http://automl.info/tpot/). This triggered a new wave of machine learning and the coming years saw many other AutoML solutions including [Auto-ml](https://github.com/ClimbsRocks/auto_ml), and [Auto-Keras](https://autokeras.com/) hitting the market. Simultaneously startups like [H2O.ai](https://www.h2o.ai/) and [DataRobot](https://www.datarobot.com/) came out with their own versions of automated solutions. More recently, companies like [Amazon](https://aws.amazon.com/sagemaker/autopilot/), [Google](https://cloud.google.com/automl), and [Microsoft](https://azure.microsoft.com/en-us/services/machine-learning/automatedml/) have also joined the bandwagon.
# 
# Some of the solutions like AutoWeka, Auto-Sklearn, TPOT, H2OAutoML are fully open-sourced while DataRobot, Amazon Sagemaker, Google’s AutoML, and DriverlessAI are enterprise-based. There are some other automl solutions also like Uber’s [Ludwig](https://uber.github.io/ludwig/) and Saleforces’s [TransmogrifAI](https://docs.transmogrif.ai/en/stable/), which are also open-source. This isn’t an exhaustive list but covers the ones that are being used commonly today.
# 
# 
# 
# 

# In[2]:


# import the necessary libraries
from IPython.display import display, HTML
import numpy as np 
import pandas as pd 
from datetime import datetime
import itertools
import collections
from wordcloud import WordCloud

# Visualisation libraries
#matplotlib & Seaborn
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

#Plotly
from plotly.offline import init_notebook_mode, iplot 
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px
import pycountry
py.init_notebook_mode(connected=True)

#Folium
import folium 
from folium import plugins

#Twitter data
get_ipython().system('pip install tweepy -q')
import tweepy as tw
import nltk
from nltk import bigrams
from nltk.corpus import stopwords
import re
import string
import networkx

# sklearn 
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# Graphics in retina format 
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


#Disable warnings
import warnings
warnings.filterwarnings("ignore")


# In[3]:


plt.rcParams.update({'font.size': 13})

names = ['AutoWeka', 'Auto-sklearn', ' TPOT', 'Auto-ml', 'Auto-Keras', 
        ' Datarobot', ' H2O-Automl', 'H2O-DriverlessAI', 'Darwin', 'Google Cloud Automl', 
        'Microsoft AzureML', ' TransmogrifAI' ,'Ludwig','MLjar']
dates = ['2013', '2014', '2015', '2016','2017', '2015', '2016',
        '2017', '2018', '2017', '2018', '2018' ,'2019','2018']
 
# Convert date strings (e.g. 2014-10-18) to datetime
dates = [datetime.strptime(d, "%Y") for d in dates]
#dates = [d.year for d in dates]
 
# Choose some nice levels
levels = np.tile([-5, 5, -3, 3, -1, 1],
                 int(np.ceil(len(dates)/6)))[:len(dates)]

 
# Create figure and plot a stem plot with the date
fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
ax.set(title="AutoML Tools history")
markerline, stemline, baseline = ax.stem(dates, levels,
                                         linefmt="C3-", basefmt="k-",
                                         use_line_collection=True)
plt.setp(markerline, mec="k", mfc="w", zorder=3)
 
# Shift the markers to the baseline by replacing the y-data by zeros.
markerline.set_ydata(np.zeros(len(dates)))
 
# annotate lines
vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
for d, l, r, va in zip(dates, levels, names, vert):
    ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3),
                textcoords="offset points", va=va, ha="left")
# format xaxis 
#ax.get_xaxis().set_major_locator(mdates.YearLocator())
#ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
 
# remove y axis and spines
ax.get_yaxis().set_visible(False)
for spine in ["left", "top", "right"]:
    ax.spines[spine].set_visible(True)
ax.margins(y=0.1)

plt.show()

print('This timeline chart has been created in Matplotlib 😃')


# *****
# <a href="#101">Back to Top</a>
# 

# <div id="2"></div>
# 
# <font color='#1E8449' size=6 >2. An Overview of the Respondents</font><br>  
# Let's begin by analyzing the 2020 survey's dataset to get a big picture. We shall begin by importing the dataset and the necessary libraries for the analysis. 
# 
# 

# In[4]:


# Importing the 2017,2018 and 2019 and 2020 survey dataset

#Importing the 2020 Dataset
df_2020 = pd.read_csv("../input/kaggle-survey-2020/kaggle_survey_2020_responses.csv",low_memory=False)
#df_2020.columns = df_2020.iloc[0]
#df_2020=df_2020.drop([0])

#Importing the 2019 Dataset
df_2019 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv',low_memory=False)
#df_2019.columns = df_2019.iloc[0]
#df_2019=df_2019.drop([0])

#Importing the 2018 Dataset
df_2018 = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv',low_memory=False)
#df_2018.columns = df_2018.iloc[0]
#df_2018=df_2018.drop([0])

#Importing the 2017 Dataset
df_2017=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1',low_memory=False)



# Helper functions( Source: https://www.kaggle.com/paultimothymooney/2020-kaggle-data-science-machine-learning-survey?select=kaggle_survey_2020_responses.csv)


def count_then_return_percent(dataframe,column_name):
    '''
    A helper function to return value counts as percentages.
    
    '''
    counts = dataframe[column_name].value_counts(dropna=False)
    percentages = round(counts*100/(dataframe[column_name].count()),1)
    return percentages

def count_then_return_percent_for_multiple_column_questions(dataframe,list_of_columns_for_a_single_question,dictionary_of_counts_for_a_single_question):
    '''
    A helper function to convert counts to percentages.
    
    '''
    
    df = dataframe
    subset = list_of_columns_for_a_single_question
    df = df[subset]
    df = df.dropna(how='all')
    total_count = len(df) 
    dictionary = dictionary_of_counts_for_a_single_question
    for i in dictionary:
        dictionary[i] = round(float(dictionary[i]*100/total_count),1)
    return dictionary 

def create_dataframe_of_counts(dataframe,column,rename_index,rename_column,return_percentages=False):
    '''
    A helper function to create a dataframe of either counts 
    or percentages, for a single multiple choice question.
     
    '''
    df = dataframe[column].value_counts().reset_index() 
    if return_percentages==True:
        df[column] = (df[column]*100)/(df[column].sum())
    df = pd.DataFrame(df) 
    df = df.rename({'index':rename_index, 'Q3':rename_column}, axis='columns')
    return df

def sort_dictionary_by_percent(dataframe,list_of_columns_for_a_single_question,dictionary_of_counts_for_a_single_question): 
    ''' 
    A helper function that can be used to sort a dictionary.
    
    It is an adaptation of a similar function
    from https://www.kaggle.com/sonmou/what-topics-from-where-to-learn-data-science.
    
    '''
    dictionary = count_then_return_percent_for_multiple_column_questions(dataframe,
                                                                list_of_columns_for_a_single_question,
                                                                dictionary_of_counts_for_a_single_question)
    dictionary = {v:k    for(k,v) in dictionary.items()}
    list_tuples = sorted(dictionary.items(), reverse=False) 
    dictionary = {v:k for (k,v) in list_tuples}   
    return dictionary



# <div id="21"></div>
# 
# <font color='#1E8449' size=5>2.1 Total number of Survey respondents</font><br>
# 
# Let's first look at the total respondents who have been participating in the survey since 2017. This should give us an idea of the participants pool.

# In[5]:


# https://www.kaggle.com/paultimothymooney/2020-kaggle-data-science-machine-learning-survey?select=kaggle_survey_2020_responses.csv

# lists of answer choices and dictionaries of value counts (for the multiple choice multiple selection questions)

# Questions where respondents can select more than one answer choice have been split into multiple columns.
# These dictionaries contain value counts for every answer choice for every multiple-column question.

responses_df = df_2020


q23_dictionary_of_counts = {
    'Analyze and understand data to influence product or business decisions' : (responses_df['Q23_Part_1'].count()),
    'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data': (responses_df['Q23_Part_2'].count()),
    'Build prototypes to explore applying machine learning to new areas' : (responses_df['Q23_Part_3'].count()),
    'Build and/or run a machine learning service that operationally improves my product or workflows' : (responses_df['Q23_Part_4'].count()),
    'Experimentation and iteration to improve existing ML models' : (responses_df['Q23_Part_5'].count()),
    'Do research that advances the state of the art of machine learning' : (responses_df['Q23_Part_6'].count()),
    'None of these activities are an important part of my role at work' : (responses_df['Q23_Part_7'].count()),
    'Other' : (responses_df['Q23_OTHER'].count())
}


q26a_dictionary_of_counts = {
    'Amazon Web Services (AWS)' : (responses_df['Q26_A_Part_1'].count()),
    'Microsoft Azure': (responses_df['Q26_A_Part_2'].count()),
    'Google Cloud Platform (GCP)' : (responses_df['Q26_A_Part_3'].count()),
    'IBM Cloud / Red Hat' : (responses_df['Q26_A_Part_4'].count()),
    'Oracle Cloud' : (responses_df['Q26_A_Part_5'].count()),
    'SAP Cloud' : (responses_df['Q26_A_Part_6'].count()),
    'Salesforce Cloud' : (responses_df['Q26_A_Part_7'].count()),
    'VMware Cloud' : (responses_df['Q26_A_Part_8'].count()),
    'Alibaba Cloud' : (responses_df['Q26_A_Part_9'].count()),
    'Tencent Cloud' : (responses_df['Q26_A_Part_10'].count()),
    'None' : (responses_df['Q26_A_Part_11'].count()),
    'Other' : (responses_df['Q26_A_OTHER'].count())
}

q26b_dictionary_of_counts = {
    'Amazon Web Services (AWS)' : (responses_df['Q26_B_Part_1'].count()),
    'Microsoft Azure': (responses_df['Q26_B_Part_2'].count()),
    'Google Cloud Platform (GCP)' : (responses_df['Q26_B_Part_3'].count()),
    'IBM Cloud / Red Hat' : (responses_df['Q26_B_Part_4'].count()),
    'Oracle Cloud' : (responses_df['Q26_B_Part_5'].count()),
    'SAP Cloud' : (responses_df['Q26_B_Part_6'].count()),
    'Salesforce Cloud' : (responses_df['Q26_B_Part_7'].count()),
    'VMware Cloud' : (responses_df['Q26_B_Part_8'].count()),
    'Alibaba Cloud' : (responses_df['Q26_B_Part_9'].count()),
    'Tencent Cloud' : (responses_df['Q26_B_Part_10'].count()),
    'None' : (responses_df['Q26_B_Part_11'].count()),
    'Other' : (responses_df['Q26_B_OTHER'].count())
}

q27a_dictionary_of_counts = {
    'Amazon EC2' : (responses_df['Q27_A_Part_1'].count()),
    'AWS Lambda': (responses_df['Q27_A_Part_2'].count()),
    'Amazon Elastic Container Service' : (responses_df['Q27_A_Part_3'].count()),
    'Azure Cloud Services' : (responses_df['Q27_A_Part_4'].count()),
    'Microsoft Azure Container Instances' : (responses_df['Q27_A_Part_5'].count()),
    'Azure Functions' : (responses_df['Q27_A_Part_6'].count()),
    'Google Cloud Compute Engine' : (responses_df['Q27_A_Part_7'].count()),
    'Google Cloud Functions' : (responses_df['Q27_A_Part_8'].count()),
    'Google Cloud Run' : (responses_df['Q27_A_Part_9'].count()),
    'Google Cloud App Engine' : (responses_df['Q27_A_Part_10'].count()),
    'No / None' : (responses_df['Q27_A_Part_11'].count()),
    'Other' : (responses_df['Q27_A_OTHER'].count())
}

q27b_dictionary_of_counts = {
    'Amazon EC2' : (responses_df['Q27_B_Part_1'].count()),
    'AWS Lambda': (responses_df['Q27_B_Part_2'].count()),
    'Amazon Elastic Container Service' : (responses_df['Q27_B_Part_3'].count()),
    'Azure Cloud Services' : (responses_df['Q27_B_Part_4'].count()),
    'Microsoft Azure Container Instances' : (responses_df['Q27_B_Part_5'].count()),
    'Azure Functions' : (responses_df['Q27_B_Part_6'].count()),
    'Google Cloud Compute Engine' : (responses_df['Q27_B_Part_7'].count()),
    'Google Cloud Functions' : (responses_df['Q27_B_Part_8'].count()),
    'Google Cloud Run' : (responses_df['Q27_B_Part_9'].count()),
    'Google Cloud App Engine' : (responses_df['Q27_B_Part_10'].count()),
    'No / None' : (responses_df['Q27_B_Part_11'].count()),
    'Other' : (responses_df['Q27_B_OTHER'].count())
}

q28a_dictionary_of_counts = {
    'Amazon SageMaker' : (responses_df['Q28_A_Part_1'].count()),
    'Amazon Forecast': (responses_df['Q28_A_Part_2'].count()),
    'Amazon Rekognition' : (responses_df['Q28_A_Part_3'].count()),
    'Azure Machine Learning Studio' : (responses_df['Q28_A_Part_4'].count()),
    'Azure Cognitive Services' : (responses_df['Q28_A_Part_5'].count()),
    'Google Cloud AI Platform / Google Cloud ML Engine' : (responses_df['Q28_A_Part_6'].count()),
    'Google Cloud Video AI' : (responses_df['Q28_A_Part_7'].count()),
    'Google Cloud Natural Language' : (responses_df['Q28_A_Part_8'].count()),
    'Google Cloud Vision AI' : (responses_df['Q28_A_Part_9'].count()),
    'No / None' : (responses_df['Q28_A_Part_10'].count()),
    'Other' : (responses_df['Q28_A_OTHER'].count())
}

q28b_dictionary_of_counts = {
    'Amazon SageMaker' : (responses_df['Q28_B_Part_1'].count()),
    'Amazon Forecast': (responses_df['Q28_B_Part_2'].count()),
    'Amazon Rekognition' : (responses_df['Q28_B_Part_3'].count()),
    'Azure Machine Learning Studio' : (responses_df['Q28_B_Part_4'].count()),
    'Azure Cognitive Services' : (responses_df['Q28_B_Part_5'].count()),
    'Google Cloud AI Platform / Google Cloud ML Engine' : (responses_df['Q28_B_Part_6'].count()),
    'Google Cloud Video AI' : (responses_df['Q28_B_Part_7'].count()),
    'Google Cloud Natural Language' : (responses_df['Q28_B_Part_8'].count()),
    'Google Cloud Vision AI' : (responses_df['Q28_B_Part_9'].count()),
    'No / None' : (responses_df['Q28_B_Part_10'].count()),
    'Other' : (responses_df['Q28_B_OTHER'].count())
}




q33a_dictionary_of_counts = {
    'Automated data augmentation (e.g. imgaug, albumentations)' : (responses_df['Q33_A_Part_1'].count()),
    'Automated feature engineering/selection (e.g. tpot, boruta_py)': (responses_df['Q33_A_Part_2'].count()),
    'Automated model selection (e.g. auto-sklearn, xcessiv)' : (responses_df['Q33_A_Part_3'].count()),
    'Automated model architecture searches (e.g. darts, enas)' : (responses_df['Q33_A_Part_4'].count()),
    'Automated hyperparameter tuning (e.g. hyperopt, ray.tune, Vizier)' : (responses_df['Q33_A_Part_5'].count()),
    'Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)' : (responses_df['Q33_A_Part_6'].count()),
    'No / None' : (responses_df['Q33_A_Part_7'].count()),
    'Other' : (responses_df['Q33_A_OTHER'].count())
}

q33b_dictionary_of_counts = {
    'Automated data augmentation (e.g. imgaug, albumentations)' : (responses_df['Q33_B_Part_1'].count()),
    'Automated feature engineering/selection (e.g. tpot, boruta_py)': (responses_df['Q33_B_Part_2'].count()),
    'Automated model selection (e.g. auto-sklearn, xcessiv)' : (responses_df['Q33_B_Part_3'].count()),
    'Automated model architecture searches (e.g. darts, enas)' : (responses_df['Q33_B_Part_4'].count()),
    'Automated hyperparameter tuning (e.g. hyperopt, ray.tune, Vizier)' : (responses_df['Q33_B_Part_5'].count()),
    'Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)' : (responses_df['Q33_B_Part_6'].count()),
    'No / None' : (responses_df['Q33_B_Part_7'].count()),
    'Other' : (responses_df['Q33_B_OTHER'].count())
}

q34a_dictionary_of_counts = {
    'Google Cloud AutoML' : (responses_df['Q34_A_Part_1'].count()),
    'H20 Driverless AI': (responses_df['Q34_A_Part_2'].count()),
    'Databricks AutoML' : (responses_df['Q34_A_Part_3'].count()),
    'DataRobot AutoML' : (responses_df['Q34_A_Part_4'].count()),
    'Tpot' : (responses_df['Q34_A_Part_5'].count()),
    'Auto-Keras' : (responses_df['Q34_A_Part_6'].count()),
    'Auto-Sklearn' : (responses_df['Q34_A_Part_7'].count()),
    'Auto_ml' : (responses_df['Q34_A_Part_8'].count()),
    'Xcessiv' : (responses_df['Q34_A_Part_9'].count()),
    'MLbox' : (responses_df['Q34_A_Part_10'].count()),
    'No / None' : (responses_df['Q34_A_Part_11'].count()),
    'Other' : (responses_df['Q34_A_OTHER'].count())
}

q34b_dictionary_of_counts = {
    'Google Cloud AutoML' : (responses_df['Q34_B_Part_1'].count()),
    'H20 Driverless AI': (responses_df['Q34_B_Part_2'].count()),
    'Databricks AutoML' : (responses_df['Q34_B_Part_3'].count()),
    'DataRobot AutoML' : (responses_df['Q34_B_Part_4'].count()),
    'Tpot' : (responses_df['Q34_B_Part_5'].count()),
    'Auto-Keras' : (responses_df['Q34_B_Part_6'].count()),
    'Auto-Sklearn' : (responses_df['Q34_B_Part_7'].count()),
    'Auto_ml' : (responses_df['Q34_B_Part_8'].count()),
    'Xcessiv' : (responses_df['Q34_B_Part_9'].count()),
    'MLbox' : (responses_df['Q34_B_Part_10'].count()),
    'No / None' : (responses_df['Q34_B_Part_11'].count()),
    'Other' : (responses_df['Q34_B_OTHER'].count())
}




# Questions where respondents can select more than one answer choice have been split into multiple columns.
# These lists delineate every sub-column for every multiple-column question.



q23_list_of_columns = ['Q23_Part_1',
                       'Q23_Part_2',
                       'Q23_Part_3',
                       'Q23_Part_4',
                       'Q23_Part_5',
                       'Q23_Part_6',
                       'Q23_Part_7',
                       'Q23_OTHER']

q26a_list_of_columns = ['Q26_A_Part_1',
                        'Q26_A_Part_2',
                        'Q26_A_Part_3',
                        'Q26_A_Part_4',
                        'Q26_A_Part_5',
                        'Q26_A_Part_6',
                        'Q26_A_Part_7',
                        'Q26_A_Part_8',
                        'Q26_A_Part_9',
                        'Q26_A_Part_10',
                        'Q26_A_Part_11',
                        'Q26_A_OTHER']

q26b_list_of_columns = ['Q26_B_Part_1',
                        'Q26_B_Part_2',
                        'Q26_B_Part_3',
                        'Q26_B_Part_4',
                        'Q26_B_Part_5',
                        'Q26_B_Part_6',
                        'Q26_B_Part_7',
                        'Q26_B_Part_8',
                        'Q26_B_Part_9',
                        'Q26_B_Part_10',
                        'Q26_B_Part_11',
                        'Q26_B_OTHER']

q27a_list_of_columns = ['Q27_A_Part_1',
                        'Q27_A_Part_2',
                        'Q27_A_Part_3',
                        'Q27_A_Part_4',
                        'Q27_A_Part_5',
                        'Q27_A_Part_6',
                        'Q27_A_Part_7',
                        'Q27_A_Part_8',
                        'Q27_A_Part_9',
                        'Q27_A_Part_10',
                        'Q27_A_Part_11',
                        'Q27_A_OTHER']

q27b_dictionary_of_counts = ['Q27_B_Part_1',
                             'Q27_B_Part_2',
                             'Q27_B_Part_3',
                             'Q27_B_Part_4',
                             'Q27_B_Part_5',
                             'Q27_B_Part_6',
                             'Q27_B_Part_7',
                             'Q27_B_Part_8',
                             'Q27_B_Part_9',
                             'Q27_B_Part_10',
                             'Q27_B_Part_11',
                             'Q27_B_OTHER']

q28a_list_of_columns = ['Q28_A_Part_1',
                        'Q28_A_Part_2',
                        'Q28_A_Part_3',
                        'Q28_A_Part_4',
                        'Q28_A_Part_5',
                        'Q28_A_Part_6',
                        'Q28_A_Part_7',
                        'Q28_A_Part_8',
                        'Q28_A_Part_9',
                        'Q28_A_Part_10',
                        'Q28_A_OTHER']

q28b_list_of_columns = ['Q28_B_Part_1',
                        'Q28_B_Part_2',
                        'Q28_B_Part_3',
                        'Q28_B_Part_4',
                        'Q28_B_Part_5',
                        'Q28_B_Part_6',
                        'Q28_B_Part_7',
                        'Q28_B_Part_8',
                        'Q28_B_Part_9',
                        'Q28_B_Part_10',
                        'Q28_B_OTHER']



q33a_list_of_columns = ['Q33_A_Part_1',
                        'Q33_A_Part_2',
                        'Q33_A_Part_3',
                        'Q33_A_Part_4',
                        'Q33_A_Part_5',
                        'Q33_A_Part_6',
                        'Q33_A_Part_7',
                        'Q33_A_OTHER']

q33b_list_of_columns = ['Q33_B_Part_1',
                        'Q33_B_Part_2',
                        'Q33_B_Part_3',
                        'Q33_B_Part_4',
                        'Q33_B_Part_5',
                        'Q33_B_Part_6',
                        'Q33_B_Part_7',
                        'Q33_B_OTHER']

q34a_list_of_columns = ['Q34_A_Part_1',
                        'Q34_A_Part_2',
                        'Q34_A_Part_3',
                        'Q34_A_Part_4',
                        'Q34_A_Part_5',
                        'Q34_A_Part_6',
                        'Q34_A_Part_7',
                        'Q34_A_Part_8',
                        'Q34_A_Part_9',
                        'Q34_A_Part_10',
                        'Q34_A_Part_11',
                        'Q34_A_OTHER']

q34b_list_of_columns = ['Q34_B_Part_1',
                        'Q34_B_Part_2',
                        'Q34_B_Part_3',
                        'Q34_B_Part_4',
                        'Q34_B_Part_5',
                        'Q34_B_Part_6',
                        'Q34_B_Part_7',
                        'Q34_B_Part_8',
                        'Q34_B_Part_9',
                        'Q34_B_Part_10',
                        'Q34_B_Part_11',
                        'Q34_B_OTHER']




# In[6]:


df_all_surveys = pd.DataFrame(
    data=[len(df_2017), len(df_2018)-1, len(df_2019)-1, len(df_2020)-1],
    columns=["Number of responses"],
    index=["2017", "2018", "2019", "2020"]
)
df_all_surveys.index.names = ["Year of Survey"]
df = df_all_surveys.reset_index(level=0)


x = df['Number of responses'].index
y = df['Number of responses'].values

trace0 = go.Bar(
            x=['Year 2017','Year 2018','Year 2019','Year 2020'],
            y=y,
            text=y,
            width=0.4,
            textposition='auto',
            marker_color='#48C9B0')

trace1 = go.Scatter(
            x=['Year 2017','Year 2018','Year 2019','Year 2020'],
            y=y,
            text=y,
            marker_color='black')

data = [trace0,trace1]

layout = go.Layout(yaxis=dict(title='Number of Respondents'),width=700,height=500,showlegend=False,
                  title='Total number of respondents over the years',title_x=0.5,plot_bgcolor='white',
                  xaxis=dict(title='Survey Year'))

figure = go.Figure(data=data, layout = layout)
figure.show()


# 
# <div id="22"></div>
# 
# <font color='#1E8449' size=5>2.2 People who responded to the AutoML question in 2020.</font><br>
# 
# So there are **20036 respondents** this year. Let's look at how many of these use the AutoML tool in some form. For this, *I analysed the Question 33_A_Part_7: Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis* ? since it gives a direct understanding into the use of AutoML.

# In[7]:


# survey 2020
start_index_2020 = df_2020.columns.get_loc("Q33_A_Part_1")
end_index_2020 = df_2020.columns.get_loc("Q33_A_OTHER")
automl_categories_2020 = df_2020.iloc[:,start_index_2020:end_index_2020+1]
automl_categories_2020 = automl_categories_2020.drop([0])

respondents_who_have_answered_2020 = automl_categories_2020.dropna(how='all')
respondents_who_have_answered_2020['Q33_A_Part_7'] = respondents_who_have_answered_2020['Q33_A_Part_7'].replace(np.nan, 'Use')
respondents_not_using_automl_2020 = automl_categories_2020['Q33_A_Part_7'].value_counts(dropna=False)['No / None']
respondents_using_automl_2020 = len(respondents_who_have_answered_2020) - respondents_not_using_automl_2020
respondents_who_havenot_answered_at_all_2020 = automl_categories_2020['Q33_A_Part_7'].value_counts(dropna=False)[0] - respondents_using_automl_2020

respondents_who_have_answered_2020['Q33_A_Part_7'] = respondents_who_have_answered_2020['Q33_A_Part_7'].replace(np.nan, 'Use')

# survey 2019
start_index_2019 = df_2019.columns.get_loc("Q25_Part_1")
end_index_2019 = df_2019.columns.get_loc("Q25_Part_8")
automl_categories_2019 = df_2019.iloc[:,start_index_2019:end_index_2019+1]
automl_categories_2019 = automl_categories_2019.drop([0])


respondents_who_have_answered_2019 = automl_categories_2019.dropna(how='all')
respondents_who_have_answered_2019['Q25_Part_7'] = respondents_who_have_answered_2019['Q25_Part_7'].replace(np.nan, 'Use')
respondents_not_using_automl_2019 = automl_categories_2019['Q25_Part_7'].value_counts(dropna=False)['None']
respondents_using_automl_2019 = len(respondents_who_have_answered_2019) - respondents_not_using_automl_2019
respondents_who_havenot_answered_at_all_2019 = automl_categories_2019['Q25_Part_7'].value_counts(dropna=False)[0] - respondents_using_automl_2019

respondents_who_have_answered_2019['Q25_Part_7'] = respondents_who_have_answered_2019['Q25_Part_7'].replace(np.nan, 'Use')




colors = ['mediumturquoise','gold' ]
colors1 = ['#F4D03F','#82E0AA', "#F1948A",]



labels = ['Didnot Answer','Use some form of AutoML','Donot use AutoML']
values1 = [respondents_who_havenot_answered_at_all_2019,respondents_using_automl_2019,respondents_not_using_automl_2019 ]
values2 = [respondents_who_havenot_answered_at_all_2020,respondents_using_automl_2020,respondents_not_using_automl_2020 ]

"""
# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=labels, values=values2,name="2020"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=values1, name="2019"),
              1, 2)

#Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.4, hoverinfo='label+value', textinfo='percent', textfont_size=15,
                  marker=dict(colors=colors1, line=dict(color='#000000', width=1)))

fig.update_layout(
    title_text="Participants who responded to AutoML question",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='2020', 
                      x=0.18, 
                      y=0.5, 
                      font_size=15, 
                      showarrow=False),
                
                
                  dict(text='2019', 
                       x=0.81, 
                       y=0.5, 
                       font_size=15, 
                       showarrow=False)])

fig.show()
"""


# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values2, hole=.4)])

fig.update_traces(hoverinfo='label+value', textinfo='percent', textfont_size=15,
                  marker=dict(colors=colors1, line=dict(color='#000000', width=1)))

fig.update_layout(
    title_text="Respondents for the AutoML question ",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Year 2020', x=0.5, y=0.5, font_size=20, showarrow=False)])
fig.show()


# <div id="23"></div>
# 
# <font color='#1E8449' size=5>2.3 AutoML usage by respondents</font><br>
# So only about 10% of the total survey respondents use AutoML in some form.A lot people left this question blank.This is because a lot of them were not showed this question based on their earlier responses. Therefore to get a better perspective, let's look at the percentage of users out of the people who answered the AutoML question.
# 
# 

# In[8]:


#colors = ['mediumturquoise','gold' ]
colors = ["#82E0AA", "#F1948A"]

labels = ['Use some form of AutoML','Donot use AutoML']
#values1 = [respondents_using_automl_2019,respondents_not_using_automl_2019 ]
values2 = [respondents_using_automl_2020,respondents_not_using_automl_2020 ]





# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values2, hole=.4)])

fig.update_traces(hoverinfo='label+value', textinfo='percent', textfont_size=15,
                  marker=dict(colors=colors, line=dict(color='#000000', width=1)))

fig.update_layout(
    title_text="Respondents for the AutoML question ",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Year 2020', x=0.5, y=0.5, font_size=20, showarrow=False)])
fig.show()


# > **📌 Key Points :**
# >
# > *  Even though there are more than [five million registered users on Kaggle](https://www.kaggle.com/general/164795), the survey numbers are quite less.
# > 
# > * The resuts show that around 67% of the total respondents in 2020 didnot answer the question pertaining to AutoML. Out of the 33% who answered the AutoML question, only about 10% actually use AutoML tools in some form.
# >
# > * Another way to look at this is that out of the people who answered the AutoML question, **30% use AutoML tools in some form**. In the next part of the analysis, we'll focus our attenton only on the AutoML users.
# 
# 
# *****
# <a href="#101">Back to Top</a>
# 

# <div id="3"></div>
# 
# <font color='#1E8449' size=6>3. Common categories and types of AutoML tools</font><br>
# Let's now focus our analysis on the six key areas -  AutoML Tools, Users, Location, Company & Teams, Age, Infrastructure and social media analysis wrt the year 2020
# 
# We'll start by going deeper into the analyses of the different categories and types of AutoML tools given in the survey, But before that it is important to understand what each of the categories  means.      

# In[9]:


# subsetting the dataset for people who responded to the AutoML question in 2020

df_2020 = pd.read_csv("../input/kaggle-survey-2020/kaggle_survey_2020_responses.csv",low_memory=False)
automl_users = df_2020.loc[(df_2020['Q33_A_Part_1'].notnull()) | (df_2020['Q33_A_Part_2'].notnull())  | (df_2020['Q33_A_Part_3'].notnull()) | (df_2020['Q33_A_Part_4'].notnull()) | (df_2020['Q33_A_Part_5'].notnull()) | (df_2020['Q33_A_Part_6'].notnull()) | (df_2020[1:]['Q33_A_OTHER'].notnull())] 
automl_users['Q33_A_Part_7'] = automl_users['Q33_A_Part_7'].replace(np.nan,'Use')     


donot_use_automl = df_2020[1:][df_2020[1:]['Q33_A_Part_7']=='No / None']

automl_2020_respondents = pd.concat([automl_users, donot_use_automl], axis=0)



automl_users_2019 = df_2019.loc[(df_2019['Q25_Part_1'].notnull()) | (df_2019['Q25_Part_2'].notnull()) | (df_2019['Q25_Part_3'].notnull()) | (df_2019['Q25_Part_4'].notnull()) | (df_2019['Q25_Part_5'].notnull()) | (df_2019['Q25_Part_6'].notnull()) | (df_2019[1:]['Q25_Part_8'].notnull())] 
automl_users_2019['Q25_Part_7'] = automl_users_2019['Q25_Part_7'].replace(np.nan,'Use')     


donot_use_automl_2019 = df_2019[1:][df_2019[1:]['Q25_Part_7']=='None']

automl_2019_respondents = pd.concat([automl_users_2019, donot_use_automl_2019], axis=0)


# In[10]:


values = [['Automated data augmentation', 
           'Automated feature engineering/selection', 
           'Automated model selection', 
           'Automated model architecture searches',
           'Automated hyperparameter tuning',
           'Automation of full ML pipelines'], 
          
          ["This process includes techniques that enhance the size and quality of training datasets eg Imgaug and Albumentationsl.",
           "This process involves creating new feature sets iteratively until the ML model achieves a satisfactory accuracy score. Typical examples include:",
           "This process involves automatically searching for the right learning algorithm for a new machine learning dataset",
           "This process involves automatically identifying architectures that are superior to hand-designed ones.",
           "The process of tuning machine learning hyperparameters automatically",
           "The process of automating the complete Machine Learning Pipeline"]]

fig = go.Figure(data=[go.Table(
  columnorder = [1,2],
  columnwidth = [80,400],
  header = dict(
    values = [['<b>AutoML Category</b>'],
                  ['<b>DESCRIPTION</b>']],
    line_color='darkslategray',
    fill_color='#82E0AA',
    align=['left','center'],
    font=dict(color='black', size=12),
    height=40
  ),
  cells=dict(
    values=values,
    line_color='darkslategray',
    fill=dict(color=['#ABEBC6', 'white']),
    align=['left', 'left'],
    font_size=12,
    height=30)
    )
])
fig.show()


# <div id="31"></div>
# 
# <font color='#1E8449' size=5>3.1 AutoML -  Category wise usage in 2020</font><br>
# First, let's have a look at the overall AutoML usage i.e AutoML usage with respect to all the respondents.

# In[11]:


responses_df = automl_2020_respondents[automl_2020_respondents['Q33_A_Part_7']=='Use']
q33a_dictionary_of_counts = {
    'Automated data augmentation (e.g. imgaug, albumentations)' : (responses_df['Q33_A_Part_1'].count()),
    'Automated feature engineering/selection (e.g. tpot, boruta_py)': (responses_df['Q33_A_Part_2'].count()),
    'Automated model selection (e.g. auto-sklearn, xcessiv)' : (responses_df['Q33_A_Part_3'].count()),
    'Automated model architecture searches (e.g. darts, enas)' : (responses_df['Q33_A_Part_4'].count()),
    'Automated hyperparameter tuning (e.g. hyperopt, ray.tune, Vizier)' : (responses_df['Q33_A_Part_5'].count()),
    'Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)' : (responses_df['Q33_A_Part_6'].count()),
    'Other' : (responses_df['Q33_A_OTHER'].count())
}

dictionary_of_counts = sort_dictionary_by_percent(responses_df,
                                                  q33a_list_of_columns,
                                                  q33a_dictionary_of_counts)
title_for_chart = 'Most common categories of AutoML (or partial AutoML) tools'
title_for_y_axis = '% of respondents'
orientation_for_chart = 'h'
  
automl_categories1 = pd.DataFrame(dictionary_of_counts.items(),columns=['Most common categories of AutoML (or partial AutoML) tools', '% of respondents'])    

data = automl_categories1

trace = go.Bar(
                    #y = automl_categories["Most common categories of AutoML (or partial AutoML) tools"],
                    y = ['Others',
                           'Auto-model<br> architecture searches',
                           'Automated<br> feature engineering',
                           'Automation of<br> full ML pipelines',
                           'Automated<br> data augmentation',
                           'Automated<br> hyperparameter tuning',
                           'Automated<br> model selection'],
                    x = automl_categories1['% of respondents'] ,
                    orientation='h',
                    marker=dict(color='#17A589', opacity=0.6),
                    #line=dict(color='black',width=1)),
                    )
data = [trace]
layout = go.Layout(barmode = "group",width=1000, height=500, 
                       #xaxis= dict(title='No of times ranked higest'),
                       #yaxis=dict(autorange="reversed"),
                       showlegend=False)

    
fig = go.Figure(data = data, layout = layout)
fig.update_traces(texttemplate='%{x:.2s}', textposition='outside')
#fig.update_layout(uniformtext_minsize=5.5, uniformtext_mode='hide')
fig.update_layout(plot_bgcolor='white',title='AutoML usage category wise')
fig.update_xaxes(showgrid=False, zeroline=False, title="% of AutoML Users ")
fig.update_yaxes(showgrid=False, zeroline=False)
fig.show()    


# So, **Automatic model selection is used by almost 40% of the AutoML users**. The next in line is Automatic hyperparameter tuning and Automatic data Augmentation.
# 
# <div id="32"></div>
# 
# <font color='#1E8449' size=5>3.2  AutoML Categories frequently used together</font><br>
# Let's see which categories are used together often. This will give an idea of the preferences of the users.
# 
# 

# In[12]:


get_ipython().run_cell_magic('html', '', "\n<div class='tableauPlaceholder' id='viz1609687289227' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Au&#47;AutoML&#47;UsageofDifferenttools&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='AutoML&#47;UsageofDifferenttools' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Au&#47;AutoML&#47;UsageofDifferenttools&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1609687289227');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>\n")


# This is interesting. Automatic data Augmentation and Automatic Hyprparameter tuning are used together by most of the users.Again, Automatic data augmentaton is mstly used with Automatic model selection. This shows there isn't a clear pattern and the usage largely depends upon the experience and task at hand.
# 
# 
# <div id="33"></div>
# 
# <font color='#1E8449' size=5>3.3  AutoML tools and their usage</font><br>
# Having looked at the categories, we'll now look at the specific AutoML tools and  their adoption among the respondents of the survey. The tools are fairly known known names in the data science space but if here I have included links to their source incase you want to study more about them.

# In[13]:


from IPython.display import display, HTML

tools = pd.DataFrame({'AutoML_Tools': ['Google Cloud AutoML', 'H2O DriverlessAI','Databricks AutoML' ,'DataRobot AutoML','Xcessiv' ,'MLbox','Tpot', 'Auto_ml','Auto-Keras','Auto-Sklearn'],
     'Type': ['Enterprise', 'Enterprise', 'Enterprise', 'Enterprise','Open-Source','Open-Source','Open-Source','Open-Source','Open-Source','Open-Source'],
     'Website/Github link': ['https://cloud.google.com/automl',
                             'https://www.h2o.ai/products/h2o-driverless-ai/',
                            'https://databricks.com/product/automl-on-databricks',
                            'https://www.datarobot.com/platform/automated-machine-learning/',
                            'https://github.com/reiinakano/xcessiv',
                            'https://github.com/AxeldeRomblay/MLBox',
                            'https://github.com/EpistasisLab/tpot',
                            'https://github.com/ClimbsRocks/auto_ml',
                            'https://autokeras.com/',
                             'https://automl.github.io/auto-sklearn/master/']})

# render dataframe as html
html = tools.to_html(render_links=True, index=False).replace('<th>','<th style = "background-color: #48c980">')
# write html to file 
text_file = open("AutoML_Tools.html", "w") 
text_file.write(html) 
text_file.close() 
HTML('AutoML_Tools.html')


# In[14]:


question_named = 'Q34-A'
dictionary_of_counts = sort_dictionary_by_percent(df_2020[1:],
                                                  q34a_list_of_columns,
                                                  q34a_dictionary_of_counts)
title_for_chart = 'Most common AutoML (or partial AutoML) tools'
title_for_y_axis = '% of respondents'
orientation_for_chart = 'h'
  
automl_tools = pd.DataFrame( q34a_dictionary_of_counts.items(),columns=['Most common  AutoML (or partial AutoML) tools', '% of respondents_2020'])    
automl_tools = automl_tools.sort_values('% of respondents_2020',ascending=True)
data = automl_tools

trace = go.Bar(
                    y = data["Most common  AutoML (or partial AutoML) tools"],
                    x = data['% of respondents_2020'] ,
                    orientation='h',
                    marker=dict(color='#48C9B0', opacity=0.6),
                    #line=dict(color='black',width=1)),
                    )
data = [trace]
layout = go.Layout(barmode = "group",width=1000, height=500, 
                       #xaxis= dict(title='No of times ranked higest'),
                       #yaxis=dict(autorange="reversed"),
                       showlegend=False)

    
fig = go.Figure(data = data, layout = layout)
fig.update_traces(texttemplate='%{x:.2s}', textposition='outside')
#fig.update_layout(uniformtext_minsize=5.5, uniformtext_mode='hide')
fig.update_layout(plot_bgcolor='white',title='AutoML usage in 2020')
fig.update_xaxes(showgrid=False, zeroline=False, title="% of AutoML respondents in 2020")
fig.update_yaxes(showgrid=False, zeroline=False)
fig.show()


# Among the specific AutoML tools, Auto-Sklearn shows the maximum usage followed by another opensource tool called Autokeras.This is the findings for the year 2020. it'll also be helpful to look at the pattern for year 2019 to see which tool has witnessed the maximum growth in terms of usage.

# In[15]:


#AutoML usage in 2019

question_named = 'Q33'

q33_2019_list_of_columns = [ 'Q33_Part_1',
                         'Q33_Part_2',
                         'Q33_Part_3',
                         'Q33_Part_4',
                         'Q33_Part_5',
                         'Q33_Part_6',
                         'Q33_Part_7',
                         'Q33_Part_8',
                         'Q33_Part_9',
                         'Q33_Part_10',
                         'Q33_Part_11',
                         'Q33_Part_12']

q33_2019_dictionary_of_counts = {
    'Google Cloud AutoML' : (df_2019['Q33_Part_1'].count()),
    'H20 Driverless AI': (df_2019['Q33_Part_2'].count()),
    'Databricks AutoML' : (df_2019['Q33_Part_3'].count()),
    'DataRobot AutoML' : (df_2019['Q33_Part_4'].count()),
    'Tpot' : (df_2019['Q33_Part_5'].count()),
    'Auto-Keras' : (df_2019['Q33_Part_6'].count()),
    'Auto-Sklearn' : (df_2019['Q33_Part_7'].count()),
    'Auto_ml' : (df_2019['Q33_Part_8'].count()),
    'Xcessiv' : (df_2019['Q33_Part_9'].count()),
    'MLbox' : (df_2019['Q33_Part_10'].count()),
    'No / None' : (df_2019['Q33_Part_11'].count()),
    'Other' : (df_2019['Q33_Part_12'].count())
}

dictionary_of_counts_2019 = sort_dictionary_by_percent(df_2019[1:],
                                                  q33_2019_list_of_columns,
                                                  q33_2019_dictionary_of_counts)
title_for_chart = 'Most common AutoML (or partial AutoML) tools'
title_for_y_axis = '% of respondents'
orientation_for_chart = 'h'
  
automl_tools_2019 = pd.DataFrame(q33_2019_dictionary_of_counts.items(),columns=['Most common  AutoML (or partial AutoML) tools', '% of respondents_2019'])    
automl_tools_2019 = automl_tools_2019.sort_values('% of respondents_2019',ascending=True)
data = automl_tools_2019



# In[16]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go


fig = make_subplots(
    rows=1, cols=1,
    specs=[[{"type": "xy"}]])
 



fig.add_trace(go.Bar(
                    y = automl_tools_2019["Most common  AutoML (or partial AutoML) tools"][:-1],
            
                    
                    x = automl_tools_2019['% of respondents_2019'] ,
                    orientation='h',
                    name="2019",
                    marker=dict(color='#F1C40F', opacity=0.6)),
                    row=1, col=1)


fig.add_trace(go.Bar(
                    y = automl_tools["Most common  AutoML (or partial AutoML) tools"][:-1],
                 
                    
                    x = automl_tools['% of respondents_2020'] ,
                    orientation='h',
                    name="2020",
                    marker=dict(color='#48C9B0', opacity=0.6)),
                    row=1, col=1)


fig.update_layout(plot_bgcolor='white',height=700, showlegend=True,
                  title='AutoML Tools usage in 2020 vs 2019 ',
                 legend = dict(orientation = "h", x = 0.1, y = 1.11))


fig.show()


# AutoSklearn and Google Cloud AutoML show maximum increase in usage from 2019. Infact all the tools show an incraese in adoption and usage. Ths will be quite evident from the line chart below.

# In[17]:


dfg = pd.concat([automl_tools, automl_tools_2019], axis=1)
column_numbers = [x for x in range(dfg.shape[1])]  # list of columns' integer indices

column_numbers .remove(2) 
dfg = dfg.iloc[:, column_numbers] #return all columns except the 0th column
dfg = dfg[:-2]

dfg.set_index(dfg.columns[0]).iplot(kind='scatter',
                                    mode='markers+lines',
                                    #annotations = {6 : 'maximum<br> rise in adoption'},
                                    title='Growth in AutoML Tools Adoption(2019 to 2020)',
                                    xTitle='AutoML Tools',
                                    yTitle='Percentage',
                                    theme='white',
                                    colors = ['#F1C40F','#48C9B0'],
                                    gridcolor='white')


# > **📌 Key Points :**
# * Automatic model selection is most used followed closely by Automatic Hyerparameter and Automatic data Augmentation. 
# * The usage of Automatic feature engineering is less as compared to others. One of the plausible reasons could be that since feature engineering relies heavily on domain expertise, people prefer to do it manually.
# * Automatic data Augmentation and Automatic Hyprparameter tuning are used together by most of the users.
# * The Year 2020 has seen a better adoption of the AutoML tools as compared to 2019.
# * The adoption of opensource AutoML tools is higher than enterprise AutoML tools. AutoSklearn has shown maximum rise in adoption. In the enterprise domain, Google Cloud gained about 11% growth in adoption and 4% by H2O Driverless AI. 
# 
# *****
# <a href="#101">Back to Top</a>

# 
# <div id="4"></div>
# 
# <font color='#1E8449' size=6>4. Analysis of AutoML Users</font><br> 
# 
# We now have an idea about the different kinds of AutoML tools and techniques being used by respondents.Let's now analyse the AutoML users under the following heads:
# * Job titles, 
# * Machine learning & Coding Experience, and their
# * Job Role
# 
# A comprehensive relationship analysis has been done between various attribute of AutoML users and the graphs below are self explanatory.Click on the legend in the graphs to isolate a particluar category. Here is a demo screengrab to demonstrate the process:
# ![](https://imgur.com/mFUTL7I.gif)
# 
# 
# <div id="41"></div>
# 
# <font color='#1E8449' size=5>4.1 Current Job Titles</font><br> 
# 
# Analysing the Job Titles of the AutoML users and comparing it with the usage of different AutoML categories should give us good insights about the prederence of the users. We'll first look at the job titles of the users and then see if there is a relation between the titles and the usage of various AutoML tools. Specifically we want to look if there is relation between the two.

# In[18]:


#Q5: Select the title most similar to your current role (or most recent title if retired): 

columns = ['Automated data<br> augmentation','Automated feature<br> engineering','Automated model<br> selection','Automated model<br> architecture searches','Automated<br> hyperparameter tuning','Automation of<br> full ML pipelines']
order = ['Data Scientist',
            'Machine Learning Engineer',
            'Data Analyst',
            'Data Engineer',
            'DBA/Database Engineer',
            'DBA/Database Engineer',
            'Research Scientist',
            'Statistician',
            'Business Analyst',
            'Product/Project Manager',
            'Other']

title_per = automl_users[1:]['Q5'].value_counts(normalize=True)*100
"""
title_per.round(1).sort_values(ascending=True).round(1).iplot(kind='barh',theme='white',
                                                     title = 'Job title of AutoML users',
                                                     xTitle='% of AutoML Users in 2020',
                                                     gridcolor='white', color='#48c980')

"""
df = title_per.to_frame().round(1)
fig = px.pie(df, values=df['Q5'].values, names=df.index, color_discrete_sequence=px.colors.sequential.YlGnBu,title='Job title of AutoML users')
fig.update_traces(textposition='inside', textinfo='percent')
fig.show()


# 
# <div id="411"></div>
# 
# <font color='#1E8449' size=4>4.1.1 Job titles vs AutoML Categories usage</font><br> 

# In[19]:


title1 = automl_users[1:].groupby(['Q5'])['Q33_A_Part_1','Q33_A_Part_2','Q33_A_Part_3','Q33_A_Part_4','Q33_A_Part_5','Q33_A_Part_6'].count()
title_per1 = ((title1.T/title1.T.sum().values))*100


import plotly.figure_factory as ff
z=title_per1.round(0)
x= ['Automated data<br> augmentation',
       'Automated feature<br> engineering',
       'Automated model<br> selection',
       'Automated model<br> architecture searches',
       'Automated<br> hyperparameter tuning',
       'Automation of<br> full ML pipelines']
y = ['Business Analyst', 'DBA/Database Engineer', 'Data Analyst',
       'Data Engineer', 'Data Scientist', 'Machine Learning Engineer',
       'Other', 'Product/Project Manager', 'Research Scientist',
       'Software Engineer', 'Statistician']
fig = ff.create_annotated_heatmap(z=z.values,y=x,x=y,colorscale='gnbu')
fig.show()






# 
# <div id="412"></div>
# 
# <font color='#1E8449' size=4>4.1.2 Job titles vs AutoML Tools usage</font><br> 

# In[20]:


title11 = automl_users[1:].groupby(['Q5'])['Q34_A_Part_1','Q34_A_Part_2','Q34_A_Part_3','Q34_A_Part_4','Q34_A_Part_5','Q34_A_Part_6','Q34_A_Part_7','Q34_A_Part_8','Q34_A_Part_9','Q34_A_Part_10'].count()
title_per11 = ((title11.T/title11.T.sum().values))*100
title_per11.index = ['Google Cloud AutoML',
                    'H20 Driverless AI',
                    'Databricks AutoML',
                    'DataRobot AutoML',
                    'Tpot',
                    'Auto-Keras',
                    'Auto-Sklearn',
                    'Auto_ml',
                    'Xcessiv',
                    'MLbox']
title_per11.T.round(1).iplot(kind='barh',gridcolor='white',theme='white',
                              barmode = 'stack',
                              title='Job titles vs AutoML Tools usage',
                              xTitle='% of AutoML Users in 2020')


# > **📌 Key Points :**
# * Maximum adoption of AutoML is by Data Scientists and Machine Learning Engineers.This shows tha AutoML usage is now becoming an important part of the ML workflow.
# * Adoption of Automated Model Architecture search tools is comparatively less.
# * Both Enterprise based and Open source AutoML tools are used with Auto-Sklearn showing a massive adoption followed by Google AutoML.

# <div id="42"></div>
# 
# <font color='#1E8449' size=5>4.2 Machine Learning Experience</font><br>
# 
# We looked at the various titles of the AutoML users, it now time to look their experiences wrt Machine Learning and see how it relates to AutoML use in general.

# In[21]:


theme='white'

responses_in_order = ['I do not use machine learning methods',
                      'Under 1 year','1-2 years','2-3 years',
                      '3-4 years','4-5 years','5-10 years',
                     '10-20 years', '20 or more years']

exp_usage = automl_users[1:].groupby(['Q15'])['Q33_A_Part_7'].count().T
exp_usage_per = ((exp_usage.T/exp_usage.T.sum()).T)*100
exp_usage_per[responses_in_order].sort_values(ascending=True).round(1).iplot(kind='barh',theme='white',
                                                         title='Machine Learning experience of AutoML users',
                                                         xTitle='% of AutoML Users in 2020',
                                                         gridcolor='white',color='#48c980')





# <div id="421"></div>
# <font color='#1E8449' size=4>4.2.1 Machine Learning experience vs AutoML Tools usage</font><br>

# In[22]:


exp_usage11 = automl_users[1:].groupby(['Q15'])['Q34_A_Part_1','Q34_A_Part_2','Q34_A_Part_3','Q34_A_Part_4','Q34_A_Part_5','Q34_A_Part_6','Q34_A_Part_7','Q34_A_Part_8','Q34_A_Part_9','Q34_A_Part_10'].count()
exp_usage_per11 = ((exp_usage11.T/exp_usage11.T.sum().values))*100
exp_usage_per11.index = ['Google Cloud AutoML',
                    'H20 Driverless AI',
                    'Databricks AutoML',
                    'DataRobot AutoML',
                    'Tpot',
                    'Auto-Keras',
                    'Auto-Sklearn',
                    'Auto_ml',
                    'Xcessiv',
                    'MLbox']
exp_usage_per11.T.round(1).iplot(kind='bar',theme=theme,gridcolor='white',
                              #barmode = 'stack',
                              title='Machine Learning experience vs AutoML Tools usage',
                              xTitle='% of AutoML Users in 2020')


# <div id="422"></div>
# <font color='#1E8449' size=4>4.2.2 Machine learning experience vs Job Title</font><br>

# In[23]:


pd.crosstab(index=automl_users[1:]['Q5'], columns=automl_users[1:]['Q15']).T.iplot(kind='scatter',
                                    mode='markers',
                                    
                                    title='Machine learning experience vs Job Title',
                                    xTitle='Machine Learning Experience',
                                    yTitle='AutoML users',
                                    theme='white',size=15,
                                    #colors = ['orange','lightpink'],
                                    gridcolor='white')


# > **📌 Key Points :**
# * Most AutoML users have 1-2 years or less than an year of machine learning experence.
# * Another unique thing to notice here is that people donot use machine learning methods also use AutoML methods.
# * Most of the Data Scientists have ether 1-2 years of experience followed by 5-10 years.
# 
# 

# 
# <div id="4"></div>
# 
# <font color='#1E8449' size=5>4.3. Coding Experience</font><br>
# 
# Having an idea about the ML experience is good, but let's look at how the coding experience of AutoML users affect their choice of usage of AutoML tools.
# 

# In[24]:


coding_usage_per = automl_users[1:]['Q6'].value_counts(normalize=True)*100
coding_usage_per.sort_values(ascending=True).round(1).iplot(kind='barh',theme='white',
                                                            title= 'Coding experience of AutoML users',
                                                            xTitle='% of AutoML Users',
                                                            gridcolor='white',bargap = 0.3, color='#48c980' )



# <div id="431"></div>
# <font color='#1E8449' size=4>4.3.1 Coding experience vs AutoML Tools usage</font><br>

# In[25]:


coding_usage = automl_users[1:].groupby(['Q6'])['Q34_A_Part_1','Q34_A_Part_2','Q34_A_Part_3','Q34_A_Part_4','Q34_A_Part_5','Q34_A_Part_6','Q34_A_Part_7','Q34_A_Part_8','Q34_A_Part_9','Q34_A_Part_10'].count()
coding_usage_per = ((coding_usage.T/coding_usage.T.sum().values))*100
coding_usage_per.index = ['Google Cloud AutoML',
                    'H20 Driverless AI',
                    'Databricks AutoML',
                    'DataRobot AutoML',
                    'Tpot',
                    'Auto-Keras',
                    'Auto-Sklearn',
                    'Auto_ml',
                    'Xcessiv',
                    'MLbox']
coding_usage_per.T.round(1).iplot(kind='barh',theme='white',gridcolor='white',
                              barmode = 'stack',
                              title='Coding experience vs AutoML Tools usage',
                              xTitle='% of AutoML Users in 2020')


# <div id="432"></div>
# <font color='#1E8449' size=4>4.3.2 Coding experience vs ML experience</font><br>

# In[26]:


plt.rcParams.update({'font.size': 12})

coding_vs_ml_exp = pd.crosstab(index=automl_users[1:]['Q15'], columns=automl_users[1:]['Q6'])
coding_vs_ml_exp_per = (((coding_vs_ml_exp.T/coding_vs_ml_exp.T.sum()).T)*100).astype('int')

plt.figure(figsize=(10,6))

sns.heatmap(coding_vs_ml_exp_per,
            cmap = "YlGnBu", annot=True, cbar=False)

plt.xlabel('Coding experience')
plt.ylabel('ML experience')
print('All values are in percent')





# > **📌 Key Points :**
# * Most AutoML users have 3-5 years  of coding experence.
# * A large percent of AutoML users without any ML experience have sufficient coding experience.
# * For others the ML experience correlated with the coding experience.

# 
# <div id="44"></div>
# <font color='#1E8449' size=5>4.4 Job Roles</font><br>
# 
# In this section we'll see at different kinds of activities that the AutoML users are involved in as part of their job role.

# In[27]:


question_name = 'Q23-A'

q23_automl_list_of_columns = [ 'Q23_Part_1',
                         'Q23_Part_2',
                         'Q23_Part_3',
                         'Q23_Part_4',
                         'Q23_Part_5',
                         'Q23_Part_6',
                         'Q23_Part_7',
                         'Q23_Other']

q23_automl_dictionary_of_counts = {
    'Analyze and understand data to influence product or business decisions' : (automl_users['Q23_Part_1'].count()),
    'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data': (automl_users['Q23_Part_2'].count()),
    'Build prototypes to explore applying machine learning to new areas' : (automl_users['Q23_Part_3'].count()),
    'Build and/or run a machine learning service that operationally improves my product or workflows' : (automl_users['Q23_Part_4'].count()),
    'Experimentation and iteration to improve existing ML models' : (automl_users['Q23_Part_5'].count()),
    'Do research that advances the state of the art of machine learning' : (automl_users['Q23_Part_6'].count()),
    'None of these activities are an important part of my role at work' : (automl_users['Q23_Part_7'].count()),
    'Other' : (automl_users['Q23_OTHER'].count())
}


dictionary_of_counts = sort_dictionary_by_percent(automl_users[1:],
                                                  q23_list_of_columns,
                                                  q23_dictionary_of_counts)

title_for_chart = ' Activities that make up an important part of your role at work'
title_for_y_axis = '% of respondents'
orientation_for_chart = 'h'
  
job_role = pd.DataFrame(dictionary_of_counts.items(),columns=[' activities that make up an important part of your role at work', '% of respondents'])    

data = job_role

trace = go.Bar(
                   #y = data[" activities that make up an important part of your role at work"],
                    y =   ['Other',
                           'None of these activities<br> are an important part<br> of my role at work',
                          'Do research that advances<br> the state of the<br> art of machine learning' ,
                          'Build and/or run a machine learning<br> service that operationally improves<br> my product or workflows',  
                          'Experimentation and iteration<br> to improve existing ML models',
                          'Build and/or run<br> the data infrastructure that my<br> business uses for storing, analyzing,<br> and operationalizing data',
                          'Build prototypes to explore<br> applying machine learning<br> to new areas',                
                          'Analyze and understand<br> data to influence product<br> or business decisions'],
                    x = data['% of respondents'] ,
                    orientation='h',
                    marker=dict(color='#48c980', opacity=0.6),
                    #line=dict(color='black',width=1)),
                    )
data = [trace]
layout = go.Layout(barmode = "group",width=1500, height=700, 
                       #xaxis= dict(title='No of times ranked higest'),
                       #yaxis=dict(autorange="reversed"),
                       showlegend=False)

    
fig = go.Figure(data = data, layout = layout)
fig.update_traces(texttemplate='%{x:.2s}', textposition='outside')
#fig.update_layout(uniformtext_minsize=5.5, uniformtext_mode='hide')
fig.update_layout(plot_bgcolor='white')
fig.update_xaxes(showgrid=False, zeroline=False, title="% of Respondents")
fig.update_yaxes(showgrid=False, zeroline=False)
fig.show()







# > **📌 Key Points :**
# * Most AutoML users analyze and understand data to influence product or business decisions. This means ultimately the use of AutoML is tied to ROI on business and providing results for stakeholders.
# 
# 

# <div id="45"></div>
# 
# <font color='#1e8449' size=5>4.5 AutoML Users Personas</font><br>
# 
# Based on the analysis in this section, four major types of personas can be identified amongst AutoML users who participated in the survey. These personas dhoul give a good idea about the kind of people usng the AutoML tools and their preferences could help the companies designing and building such tools.
# 
# <img src='https://imgur.com/KV7LhNW.png' width=1000>
# 
# 
# <a href="#101">Back to Top</a>
# 
# *****

# 
# <div id="5"></div>
# 
# <font color='#1E8449' size=6>5. Companywise AutoML usage analysis</font><br>
# 
# <div id="51"></div>
# 
# <font color='#1E8449' size=5>5.1 Company size of the AutoML users</font><br>
# 
# We've analysed the usage and the kind of AutoML tools preferred by AutoML users. However, looking at users in isolation without looking at the companies that they work for isn't advisable. In this section we look at the overall companies' size and then the Data Science Team size to see if there is a relationship bwtween the two. Just like in the previous section, we'll use these details to identify the most preferred company portfolio b the AutoML users.
# 

# In[28]:


responses_in_order = ['0-49 employees',
                      '50-249 employees','250-999 employees',
                      "1000-9,999 employees","10,000 or more employees"]


size = automl_users[1:].groupby(['Q20'])['Q33_A_Part_7'].count().T
size = size[responses_in_order]
size_per = ((size.T/size.T.sum()).T)*100
size_per.round(1).iplot(kind='barh',xTitle='% of AutoML Users',gridcolor='rgb(250, 242, 242)',
                        bargap = 0.4,color='#48c980',
                        theme='white',
                        bgcolor='white',
                        title='Company size of AutoML users ')


# > **📌 Key Points :**
# * It appears most of the AutoML users work for smaller 'Startup' type companies having a strength of less than 50 employees.

# 
# <div id="52"></div>
# 
# <font color='#1E8449' size=5>5.2 Machine learning maturity in business</font><br>
# 
# So, what is kind of machine learning work that these companies are engaged in? Let's find out.

# In[29]:


ML_in_company = automl_users[1:].groupby(['Q22'])['Q33_A_Part_7'].count().T
ML_in_company.index = ["don't know",'No','exploring<br> ML methods',' well established<br> ML methods','recently started<br> using ML methods','use ML methods for<br> generating insights (but do not<br> put working models into production)'] 
ML_in_company_per = ((ML_in_company.T/ML_in_company.T.sum()).T)*100
ML_in_company_per.sort_values(ascending=True).round(1).iplot(kind='barh',
                                                             xTitle='% of AutoML Users',
                                                             gridcolor='white',color='#48c980',
                                                             theme='white')



# > **📌 Key Points :**
# * It is interesting to see that although most of AutoML users work for smaller companies, the ML methods used in these companies  are pretty mature as compared to their counterparts.

# 
# <div id="53"></div>
# 
# <font color='#1E8449' size=5>5.3 Data Science Team Size</font><br>
# 
# It is also important to look at the the strength of Data Science Teams and whether they correlate with the company size in general.
# 

# In[30]:


responses_in_order = ['0','1-2','3-4','5-9','10-14','15-19','20+']
ds_team = automl_users[1:].groupby(['Q21'])['Q33_A_Part_7'].count().T
ds_team = ds_team.reindex(index=responses_in_order)
ds_team_per = ((ds_team.T/size.T.sum()).T)*100
#ds_team_per.index = responses_in_order
ds_team_per.round(1).iplot(kind='barh',xTitle='% of AutoML Users',
                                        gridcolor='white',color='#48c980',
                                        bargap = 0.3,
                                         theme='white')


# > **📌 Key Points :**
# * The data shows that most of the AutoML users work in Data Science teams having more than 20 members or 1-2 members.One of the possible reasons for these two extreme numbers could be that the DS teams depend on the size of the company too. Let's analyse it further.

# <div id="531"></div>
# 
# <font color='#1E8449' size=4>5.3.1 Relationship between Company Size and Data Science Team Size</font><br>
# 
# The heatmap below shows the relationship between the Company size and the Data Science Team size in terms of percentage of AutoML users.
# 

# In[31]:


plt.figure(figsize=(10,6))
company_vs_teamsize= pd.crosstab(index=automl_users[1:]['Q20'], columns=automl_users[1:]['Q21'])
company_vs_teamsize_per = (((company_vs_teamsize.T/company_vs_teamsize.T.sum()).T)*100).astype('int')
sns.heatmap(company_vs_teamsize_per,
            cmap="YlGnBu", annot=True, cbar=False)
plt.xlabel('Data Science Team size')
plt.ylabel('Size of the entire company')
print('All values are in percent')
plt.show()


# > **📌 Key Points :**
# * Well,this shows a trend. Companies having greater than 10,000 employees tend to have larger DS teams and vice versa. 

# <div id="54"></div>
# 
# <font color='#1E8449' size=5>5.4 AutoML Users' Company Profile</font><br>
# 
# Based on our analysis from the given data, we obtain a company profile which related to most of the AutoML users.

# ![](https://imgur.com/Ydtf8b2.png)
# 
# 
# *****
# <a href="#101">Back to Top</a>
# 
# 

# <div id="6"></div>
# 
# <font color='#1E8449' size=6>6. Infrastructure</font><br>
# 
# Over the past two decades, the cloud computing model has changed the way that most enterprise organizations manage their information technology systems and resources.Cloud computing has created the opportunity for organizations to access the data storage and computing capabilities that they require, on an as-needed basis[[Source](https://www.sumologic.com/glossary/cloud-infrastructure/)]. Keeping this in mind, we'll see how the AutoML users or the companies they work for utilise the cloud capabilities and their expenditure on them.
# 
# 
# <div id="61"></div>
# 
# <font color='#1E8449' size=5>6.1 Money spent on Cloud Computing in 2020</font><br>
# 
# We'll start by looking at an overview of the money spent either individually or via team/company on the cloud computing products in general.

# In[32]:


# source: https://www.kaggle.com/shivamb/spending-for-ms-in-data-science-worth-it
d = {
        '0 USD': 2,
        '100-999 USD': 24,
        '1000-9,999 USD': 28,
        '1-99 USD': 16.8,
        '10,000-99,999 USD': 18,
        '100,000 or more USD': 12
}

xx = ["0 USD",'1-99 USD',"100-999 USD", "1000-9,999 USD", "10,000-99,999 USD", "100,000 or more USD"]
xx = [_ + "<br>(" +str(d[_])+ "%)" for _ in xx]
yy = [""]*len(d)
zz = [11, 40, 80, 95, 50, 30]
cc = ['red', 'green', 'purple', 'orange', "blue", "cyan"] 

trace1 = go.Scatter(x = xx, y = [""]*len(d), mode='markers', name="", marker=dict(color=cc, opacity=0.4, size = zz))
layout = go.Layout(barmode='stack', height=300, margin=dict(l=100), title='Money you (or your team) spent on machine learning and/or cloud computing services',
                   legend = dict(orientation="h", x=0.1, y=1.15), plot_bgcolor='#fff', paper_bgcolor='#fff', 
                   showlegend=False)

fig = go.Figure(data=[trace1], layout=layout)
iplot(fig)


# Most of the companies/users spent around 1000-10000 USD on cloud compute in 2020.Even A lot of companies or users also didnot spend anything too. 
# 
# 
# <div id="611"></div>
# 
# <font color='#1E8449' size=4>6.1.2. A comparison between expenditure in 2020 and 2019</font><br>
# 
# 
# * The AutoML is definitely maturing seeing the usage, but has effet also penetrated into the money being spent on cloud infrastructure?

# In[33]:


#2020

automl_users[1:]['Q25'] = automl_users[1:]['Q25'].str.replace('$','')
money_spent_cloud  = automl_users[1:]['Q25'].value_counts(normalize=True)*100

#2019
automl_users_2019['Q11'].replace({'> 100,000 (USD)':'100,000 or more (USD)'},inplace=True)
automl_users_2019[1:]['Q11'] = automl_users_2019[1:]['Q11'].str.replace('$','')
money_spent_cloud_2019  = automl_users_2019[1:]['Q11'].value_counts(normalize=True)*100

cloud_spending_comparison = pd.concat([money_spent_cloud_2019,money_spent_cloud], axis=1)
cloud_spending_comparison.rename(columns={'Q25': 2020, 'Q11': 2019},inplace=True)
cloud_spending_comparison.round(0).iplot(kind='scatter',mode='lines+markers',
                            
                                    
                                    title='Money spent on cloud computing products(2020 vs 2019)',
                                     xTitle='Money spent(USD)',
                                     yTitle='% of AutoML users',
                                    theme='white',
                                    colors = ['#f1c40f','#48c980'],
                                    gridcolor='white')


# 
# The pattern for spend remains the same in 2020 and 2019 except that there are two important things to gain from this analysis:
# * More people/companies are spending on cloud compute in 2020 as compared to 2019
# * In 2019 around 24% people didnot spend anything on cloud compute. The percentage has fallen to 2 in 2020.
# 
# 
# <div id="62"></div>
# 
# <font color='#1E8449' size=5>6.2 Money spent vs Company size</font><br>
# 
# Do bigger companies spend more?

# In[34]:


automl_users[1:]['Q25'] = automl_users[1:]['Q25'].str.replace('$','')
money_spent_cloud  = automl_users[1:]['Q25'].value_counts(normalize=True)*100

responses_in_order1 = ['0-49 employees',
                      '50-249 employees','250-999 employees',
                      "1000-9,999 employees","10,000 or more employees"]

responses_in_order2 = ['0(USD)','1-99','100-999','1000-9,999','10,000-99,999','100,000 or more (USD)']

xc = pd.crosstab(index=automl_users[1:]['Q25'], columns=automl_users[1:]['Q20']).T
xc_per = ((xc.T/xc.T.sum()).T)*100
xc_per = xc_per.reindex(index=responses_in_order1)
xc_per.round(0).iplot(kind='bar',barmode='stack',
         #mode='markers+lines',
         title='Money spent on cloud computing products vs Company size',
         yTitle='Money spent(USD)',
         xTitle='Company size',
         theme='white',symbol='circle-open-dot',
         #colors = ['orange','lightpink'],
         gridcolor='white')


# More users in larger companies spend above 100000 USD on cloud compute as compared to other categories.Users working in mid size companies tend to spend under 10000 USD, generally. 
# 
# > **📌 Key Points :**
# * Most users/copmanies using AutoML spend less than 10,000 USD on cloud compute.
# * The proportion of AutoML users spending on cloud has definitely increased in 2020 as compared to 2019 with fewer not spending a single dollar.
# * People working in larger companies tend to spend more as compared to their counterparts in smaller companies. Hoowever, there isn't a clear demarcation and people with different spending habits are found in all categories.
# 
# *****
# <a href="#101">Back to Top</a>

# <div id="7"></div>
# 
# <font color='#1E8449' size=6 >7. AutoML users around the globe</font><br> 
# 
# 
# We have seen how different companies and teams are adopting and using AutoML tools and technologies.However, a geographical analysis is also important to understand where the different users are located in the world. We'll be assesing **countrywise** as well as **region wise** adoption of AutoML. This could be particularly useful from business and marketing point of view.
# 
# <div id="71"></div>
# 
# <font color='#1E8449' size=5 >7.1 AutoML - Countrywise adoption in 2020</font><br> 
# 

# In[35]:


automl_countries = automl_users[1:]['Q3'].value_counts().to_frame().reset_index().rename(columns={'index':'Country','Q3':'Respondents'})
automl_countries['Percentage'] = (automl_countries['Respondents']*100)/len(automl_users['Q3'])
automl_countries['Percentage'] = automl_countries['Percentage'].round(1)

# Replacing the ambigious countries name with Standard names
automl_countries['Country'].replace({'United States of America':'United States',
                                    'Viet Nam':'Vietnam',
                                    "People 's Republic of China":'China',
                                    "United Kingdom of Great Britain and Northern Ireland":'United Kingdom',
                                     "Republic of Korea":'South Korea',
                                     "Iran, Islamic Republic of...":'Iran',
                                    "Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)


worldmap = [dict(type = 'choropleth', locations = automl_countries['Country'], locationmode = 'country names',
                 z = automl_countries['Percentage'], colorscale = "Greens", reversescale = True, 
                 marker = dict(line = dict( width = 0.5)), 
                 colorbar = dict(autotick = False, title = '% of AutoML users'))]

layout = dict(title = 'Countrywise adoption of AutoML', geo = dict(showframe = False, showcoastlines = True, 
                                                                projection = dict(type = 'Mercator')))

fig = dict(data=worldmap, layout=layout)
py.iplot(fig, validate=False)


# India and the US shine when it comes to usage. This is expected since most of the respondents are fro these two countries.

# <div id="72"></div>
# 
# <font color='#1E8449' size=5 >7.2 AutoML - Regionwise adoption in 2020</font><br> 
# 
# While a contrywise usage analysis is useful, focussing on regions than countries could give us more insight if we look at it from a business point of view. Here the countries have been grouped into five major regions i.e
# 
# * Asia & Pacific or APAC
# * North America
# * Europe & Middle East or EMEA
# * Latin America or LATAM
# * Africa
# 

# In[36]:


automl_countries['Region'] = np.where(automl_countries['Country'].isin(['Argentina','Brazil','Chile','Colombia','Mexico','Peru'])
                                      ,'South/Latin America',automl_countries['Country'])

automl_countries['Region'] = np.where(automl_countries['Country'].isin(['Ghana','Kenya','Nigeria','South Africa','Morocco','Tunisia'])
                                      ,'Africa',automl_countries['Region'])
automl_countries['Region'] = np.where(automl_countries['Country'].isin(['Australia','Bangladesh','China','India','Israel','Indonesia','Japan','Malaysia','Nepal','Pakistan',
                                                                        'Philippines','South Korea','Singapore','Republic of Korea','Sri Lanka','Taiwan','Thailand','Vietnam'])
                                      ,'Asia & Pacific(APAC)',automl_countries['Region'])
automl_countries['Region'] = np.where(automl_countries['Country'].isin(['Belarus','Belgium','France','Germany','Greece','Ireland','Italy','Netherlands',
                                                                        'Poland','Portugal','Romania','Russia','Spain','Sweden','Switzerland','Turkey',
                                                                        'Ukraine','United Kingdom','Egypt','Iran','Saudi Arabia','United Arab Emirates'])
                                      ,'Europe & Middle East(EMEA)',automl_countries['Region'])
automl_countries['Region'] = np.where(automl_countries['Country'].isin(['Canada','United States'])
                                      ,'North America',automl_countries['Region'])


# In[37]:


# Importing the world_coordinates dataset
world_coordinates = pd.read_csv('../input/iso-country-codes-global/wikipedia-iso-country-codes.csv')
world_coordinates.rename(columns={'English short name lower case':'Country'},inplace=True)

# Merging the coordinates dataframe with original dataframe
automl_countries = pd.merge(automl_countries, world_coordinates,on='Country')






# In[38]:


fig = px.scatter_geo(automl_countries, locations="Alpha-3 code", color="Region",
                    size="Percentage",text='Country',size_max = 35,title='Region wise adoption of AutoML in 2020',
                     projection="natural earth")
fig.show()


# > **📌 Key Points :**
# * As per the available data, most of the AutoML users belong to India and the US. This is true since most of the respondents are also from these two countries only.
# * Another way to look at this data is from a Region wise perspective. The Asia Pacific or the APAC Region shows maximum adoption followed by North America & European market. Interestingly, the same results are observed during a market research study done by [psmarketresearch.com.](https://www.prnewswire.com/) Here is a [link](https://www.psmarketresearch.com/market-analysis/automated-machine-learning-market) to their report. The figure below has been borrowed from the same report.
# 
# ![](https://www.psmarketresearch.com/img/GLOBAL-AUTOMATED-MACHINE-LEARNING-MARKET.png)
# 
# ###### source: https://www.prnewswire.com/news-releases/automated-machine-learning-automl-market-301035747.html
# 
# ******
# 
# 
# <a href="#101">Back to Top</a>

# <div id="8"></div>
# 
# <font color='#1E8449' size=6 >8. Social Media Analysis : Twitter and Google Trends</font><br> 
# 
# No analysis today is complete without having a look at what the social media handles and the Google trends say.Not only does this data give an idea about the general sentiment of the public, but it also offers a way to understand how a particluar tool or product is received by the public. For this particular I used the data from two sources:
# * Twitter data pertaining to AutoML for the year 2020, and
# * Google trends data for the year 2020 for AutoML and related searches.
# 
# The dataset has been uploaded on Kaggle and made publically available here and here. More details into collecting the data has been mentioned in the description of each dataset. Hence, in this notebook I shall limit myself to only the analysis of the data to see if we can actually discover something useful.
# 
# 
# <div id="81"></div>
# 
# <font color='#1E8449' size=5 >8.1 Google Trends data analysis</font><br> 
# 
# Google Trends is a website by Google that analyzes the popularity of top search queries in Google Search across various regions and languages. The website uses graphs to compare the search volume of different queries over time.[[Wikipedia](https://en.wikipedia.org/wiki/Google_Trends)]
# 
# The dataset consists of four filess:
# 
# * Timeline Data
# * GeoMap data
# * Related Entities data
# * Related Queries data
# 
# Let's look at what each of them signifies.
# 
# <div id="811"></div>
# 
# <font color='#1E8449' size=4.5 >8.1.1 Interest over time </font><br> 
# 
# To gauge AutoML searches interest over time, I looked at data for the past five years for the search term :  AutoML and Automated machine learning. The numbers below represent search interest relative to the highest point on the chart for the given region and time. A value of 100 is the peak popularity for the term while 50 means that the term is half as popular. A score of 0 means that there was not enough data for this term. 
# 

# In[39]:


timeline= pd.read_csv('../input/automl-google-trends-data/Searches.csv')
geomap= pd.read_csv('../input/automl-google-trends-data/geoMap.csv')
related_entities = pd.read_csv('../input/automl-google-trends-data/relatedEntities.csv')
related_queries = pd.read_csv('../input/automl-google-trends-data/relatedQueries.csv')

# Convert string to datetime64
timeline['Date'] = timeline['Date'].apply(pd.to_datetime)
timeline.set_index('Date',inplace=True)


# In[40]:


# Image source: https://support.google.com/trends/answer/4359550?hl=en&ref_topic=4365530

from PIL import Image
import urllib.request
#url = 'http://www.example.com/my_image_is_not_your_image.png'


image_path = '../input/interest-over-time/interest_over_time.png'
trace = go.Scatter(
                    x = timeline.index,
                    y = timeline['Search Interest'] ,
                    orientation='v',
                    marker = dict(color='#48c980',opacity=0.6,
                                 line=dict(color='red',width=6)
                                ),
                    )
data = [trace]
layout= go.Layout(title= " AutoML searches over the years",
                  xaxis=dict(title='Years'),
                  yaxis=dict(title="Search Interest"),
                  plot_bgcolor='white',
                  images= [dict(
                          source=Image.open(image_path),
                          xref= "paper",
                          yref= "paper",
                          x= 0.4,
                          y= 1,
                          sizex= 0.4,
                          sizey= 0.4,
                          
              
                          opacity= 1,
                          xanchor= "right", 
                          yanchor="top",
                          layer= "above")])
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# 
# When we look at the interest in autoML, we observe an increasing trend since the beginning of 2017. The advances in AI algorithms and the increasing popularity of automation technologies might be the reasons for this growth. 

# <div id="812"></div>
# 
# <font color='#1E8449' size=4.5 >8.1.2. Interest by Region in 2020 </font><br> 
# 
# Let's see in which location the above terms were most popular during the specified time frame. Again the values are calculated on a scale from 0 to 100, where 100 is the location with the most popularity as a fraction of total searches in that location. A value of 50 indicates a location which is half as popular. A value of 0 indicates a location where there was not enough data for this term.
# 

# In[41]:


worldmap = [dict(type = 'choropleth', locations = geomap['Country'], locationmode = 'country names',
                 z = geomap['Number of Searches'], colorscale = "Greens", reversescale = True,autocolorscale=False, 
                 marker = dict(line = dict( width = 1)), 
                 colorbar = dict(autotick = False, title = 'Search Interest'))]

layout = dict(title = 'AutoML searches trend on Google', geo = dict(showframe = False, showcoastlines = True, 
                                                                projection = dict(type = 'equirectangular')))

fig = dict(data=worldmap, layout=layout)
py.iplot(fig, validate=False)


# The Google trends show that the word AutoML/Automated Machine learning has been very popular in **China**. 
# 
# <div id="813"></div>
# 
# <font color='#1E8449' size=4.5 >8.1.3. Related Topics and Queries related to AutoML in 2020 </font><br> 
# 
# Finally, we look at two more metrics:
# * Related topics - Users searching for your term also searched for these topics
# * Related queries - Users searching for your term also searched for these queries

# In[42]:


fig, (ax1, ax2,) = plt.subplots(1, 2, figsize=[26, 8])
sns.set_color_codes("pastel")

font = '../input/quicksandboldttf/Quicksand-Bold.ttf'
wordcloud = WordCloud(font_path=font,
                       width=1000,
                       height=800,
                       colormap='PuRd', 
                       margin=0,
                       max_words=2000, # Maximum numbers of words we want to see 
                       min_word_length=3, # Minimum numbers of letters of each word to be part of the cloud
                       max_font_size=80, min_font_size=2,  # Font size range
                       background_color="white").generate((" ").join(related_entities['Related Entities'].to_list()))


ax2= sns.barplot(x="Search Interest", y="Related Queries", data=related_queries,
            label="Total")
ax2.set_ylabel('')  
ax2.set_title('Queries related to AutoML');



ax1.imshow(wordcloud)
ax1.axis('off')
ax1.set_title('Topics related to AutoML');




# 
# <div id="82"></div>
# 
# <font color='#1E8449' size=5 >8.2 Twitter data analysis</font><br> 
# 
# Finally, we are going to use of Twitter as a source of information to  understand the sentiment around AutoML. Since this can be a vast source of information, we'll limit the data around the popular hasthags i.e #automl, #AutoML, #automl and #AutomatedML. The tweets have been collected from 1st Jan 2020 uptill 31st Dec 2020.You can access this data [here](https://www.kaggle.com/parulpandey/tweets-related-to-automl-in-2020).
# 
# 
# <div id="821"></div>
# 
# <font color='#1E8449' size=4.5 >8.2.1 Common Words Found in Tweets</font><br> 
# 
# Let's start by analysing the common words in the tweets. For this we'll need to clean the data and get rid of characters like - RT,#,@ etc. Additionally, we'll also remove the stop words. The cleaned form of tweets look like this:
# 
# 

# In[43]:


stop_words = set(stopwords.words('english'))

# text preprocessing helper functions

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def text_preprocessing(text):
    """
    Cleaning and parsing the text.

    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(remove_stopwords)
    return combined_text

# Applying the cleaning function to both test and training datasets

tweet_df = pd.read_csv('../input/tweets-related-to-automl-in-2020/tweets_data.csv')
tweet_df['text'] = tweet_df['text'].apply(str).apply(lambda x: text_preprocessing(x))


words_in_tweet = [tweet.lower().split() for tweet in tweet_df['text']]


# List of all words across tweets
all_words = list(itertools.chain(*words_in_tweet))

# Create counter for common words
counts_words = collections.Counter(all_words)

clean_tweets = pd.DataFrame(counts_words.most_common(20),
                             columns=['words', 'count'])


# Plot horizontal bar graph
clean_tweets.sort_values(by='count').iplot(kind='barh',x='words',
                      y='count',title='Top 20 most Common Words Found in Tweets',
                      gridcolor='white',color='#48c980', theme='white',xTitle='count')








# As expected words like automl, datascience, machinelearning and ai occur quite frequently in the tweets.

# 
# <div id="822"></div>
# 
# <font color='#1E8449' size=4.5 >8.2.2 Exploring Co-occurring Words (Bigrams)</font><br> 
# 
# Let's now explore certain words which occuur together in the tweets. Such words are called bigrams.[A bigram or digram is a sequence of two adjacent elements from a string of tokens, which are typically letters, syllables, or words](https://en.wikipedia.org/wiki/Bigram#:~:text=A%20bigram%20or%20digram%20is,%2Dgram%20for%20n%3D2.)
# 
# 

# In[44]:


# Create list of lists containing bigrams in tweets
terms_bigram = [list(bigrams(tweet)) for tweet in words_in_tweet]

# Flatten list of bigrams in clean tweets
bigrams_all = list(itertools.chain(*terms_bigram))

# Create counter of words in clean bigrams
bigram_counts = collections.Counter(bigrams_all)

bigram_df = pd.DataFrame(bigram_counts.most_common(40),
                             columns=['bigram', 'count'])

# Plot horizontal bar graph
fig, ax = plt.subplots(figsize=(10, 8))

bigram_df[:20].sort_values(by='count').plot.barh(x='bigram',
                      y='count',
                      ax=ax,
                      color='#48c980')

plt.show()


# There are some pretty obvious bigrams like datascience and python; python and r; machinelearning and deeplearning etc but there are also words like nlp and nosql and nosql and iot which occur together. This is petty interesting.

# <div id="8231"></div>
# 
# <font color='#1E8449' size=3.5 >8.2.2.1 Visualizing Networks of Co-occurring Word</font><br> 
# 
# We shall now use these bigrams to visualize the top occurring bigrams as networks using the Python package NetworkX. 

# In[45]:


import networkx as nx

# Create dictionary of bigrams and their counts
d = bigram_df.set_index('bigram').T.to_dict('records')

# Create network plot 
G = nx.Graph()

# Create connections between nodes
for k, v in d[0].items():
    G.add_edge(k[0], k[1], weight=(v * 5))

#G.add_node("automl", weight=100)

fig, ax = plt.subplots(figsize=(18, 10))

pos = nx.spring_layout(G, k=2)

# Plot networks
nx.draw_networkx(G, pos,
                 font_size=16,
                 width=3,
                 edge_color='grey',
                 node_color='green',
                 with_labels = False,
                 ax=ax)

# Create offset labels
for key, value in pos.items():
    x, y = value[0]+.135, value[1]+.045
    ax.text(x, y,
            s=key,
            bbox=dict(facecolor='yellow', alpha=0.25),
            horizontalalignment='center', fontsize=13)
    
plt.show()


# > **📌 Key Points :**
# >
# The social media analysis releals some interesting insights. Let's summarize the important ones:
# 
# * There has been increase in the search interest for the term AutoML, since 2017.China shows maximum searches related to the word AutoML. 
# * As far as twitter data analysis is concerned, we got some interesting words linked to #automl like machinelearning, datascience, iot, serverless and kubernetes etc.
# 
# *****
# 
# <div id="9"></div>
# 
# <font color='#1E8449' size=6 >9. Conclusion & Key Findings</font><br> 
# 
# In this study, I have presented the findings related to AutoML usage and adoption as per the data available in the 2020 Kaggle survey.There is no doubt that the results reflect the general perception of the Kaggle users, but since the survey pool is large we can definitely get some useful insights for the overall AutoML users' population as well.
# 
# 
# 🔑 The Year 2020 has seen a better adoption of the AutoML tools as compared to 2019. AutoSklearn and AutoKeras lead the opensource brigade while Google AutoML and Driverless AI lead in enterprise versions.
# 
# 🔑Maximum adoption of AutoML is by Data Scientists who mostly use machine learning to analyze and understand data to influence product or business decisions. Multiple years of experience increases the likelihood of AutoML adoption.
# 
# 🔑 Most AutoML users work for Startups or small-sized companies having well-established Machine Learning methods.
# 
# 🔑 The proportion of AutoML users spending on cloud computing has seen an increase in 2020 as compared to 2019, with the average spend being around 10,000 USD.
# 
# 🔑Overall, the APAC Region shows maximum AutoML adoption followed by North America & EMEA markets.
# 
# 🔑 There has been an increasing trend in the search interest in AutoML since the beginning of 2017. The advances in AI algorithms and the increasing popularity of automation technologies might be the reasons for this growth.
# 
# *****

# <div id="10"></div>
# 
# <font color='#1E8449' size=6 >10. References & Acknowledgement</font><br> 
# 
# 
# * Title Image: <a href="http://www.freepik.com">Designed by Freepik</a>
# * [2020 Kaggle Data Science & Machine Learning Survey](https://www.kaggle.com/paultimothymooney/2020-kaggle-data-science-machine-learning-survey) by Paul Mooney
# * [Spending $$$ for MS in Data Science - Worth it ?](https://www.kaggle.com/shivamb/spending-for-ms-in-data-science-worth-it) by Shivam Bansal
# * https://arxiv.org/pdf/1908.05557.pdf
# * [what-topics-from-where-to-learn-data-science](https://www.kaggle.com/sonmou/what-topics-from-where-to-learn-data-science)  
# * [Machine Learning in Python: Main developments and technology trends in data science, machine learning, and artificial intelligence](https://arxiv.org/abs/2002.04803)
# * https://www.prnewswire.com/news-releases/automated-machine-learning-automl-market-301035747.html
# * https://trends.google.com/trends/explore?q=AutoML
# 
# *****
# 

# ![](https://chart-studio.plotly.com/create/?_ga=2.198211114.44164853.1608599147-14683959.1602473809&fid=parulnith:10&fid=parulnith:9)
