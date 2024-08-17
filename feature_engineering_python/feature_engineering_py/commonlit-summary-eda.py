#!/usr/bin/env python
# coding: utf-8

# # CommonLit EDA and Predictive Modeling
# 
# ![reading](https://images.pexels.com/photos/3278757/pexels-photo-3278757.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)
# 
# Welcome to this notebook on Exploratory Data Analysis (EDA) and predictive modeling for the summaries dataset. In this notebook, we will be exploring the dataset provided by the nonprofit technology organization, CommonLit. Our main goal is to predict the correctness and wording.
# 
# ## Sections
# 
# 1. **Entry Data:** We will start by loading and understanding the dataset. Exploring the columns, data types, and gaining insights into the structure of the data.
# 
# 2. **Results Distribution:** Here, we will analyze the distribution of the target to understand the balance of the dataset.
# 
# 3. **Feature Engineering and Comparison:** In this section, we will explore the existing features and potentially create new features to enhance the model's predictive power such as count of stopwords and total length.
# 
# 4. **Data Transformation - Linear Regression:** We will perform data preprocessing and use linear regression as a simple baseline model to predict the correctness of summaries.
# 
# 5. **Final Remarks:** We will conclude the notebook with key takeaways from our EDA and initial modeling.
# 
# This notebook is based on my previous notebook on [Twitter Sentiment Analysis](https://www.kaggle.com/code/kevinmorgado/twitter-sentiment-analysis-logistic-regression) using Logistic Regression for those interested in exploring NLP tasks further.
# 
# I appreciate your time in reading this notebook and I am open to any advice or feedback you may have.

# # 1. Entry Data
# 
# Firstly, the main libraries were loaded, using pandas and seaborn for Data Analysis and nltk and sklearn for building the prediction model

# In[1]:


import numpy as np # Data analysis
import pandas as pd # Data analysis
import os
import seaborn as sns # Plotting tool
import matplotlib.pyplot as plt # Plotting tool
import nltk # Natural Language Toolkit
from nltk import word_tokenize # Tokenizer
import re #Regular expressions for filtering
from sklearn.feature_extraction.text import CountVectorizer # Vectorization approach
from sklearn.model_selection import train_test_split # Test - Train Split
from sklearn.preprocessing import RobustScaler # Data Transformation
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# The first table in alphabetical order corresponds to the correct formatting of the submission file. It only consists of 4 rows as it is not the final dataset. That will be loaded on the submission process:

# In[2]:


df_samplesub=pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/sample_submission.csv")
df_samplesub.head()


# In[3]:


df_samplesub.shape


# Then, the second file corresponds to the prompt text, showing that there are 4 different questions to answer on this task.

# In[4]:


df_train_p=pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/prompts_train.csv")
df_train_p.head()


# In[5]:


df_train_p.shape


# At the same time, the summaries test is an auxiliary table for use for the predictions. It will be replaced with the actual data when submitting the scoring. For this, it has only four rows.

# In[6]:


df_test_text=pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/summaries_test.csv")
df_test_text.head()


# In[7]:


df_test_text.shape


# Then, the summaries_train is the table that was used for training the modeling, containing two extra rows that represent the two objective variables: `content` and `wording`

# In[8]:


df_train_text=pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/summaries_train.csv")
df_train_text.head()


# In[9]:


df_train_text.shape


# Finally, the prompts test table shows a similar structure to the prompts_train. However, in this implementation, it won't be used.

# In[10]:


df_test_p=pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/prompts_test.csv")
df_test_p.head()


# In[11]:


df_test_p.shape


# # 2. Result distribution
# 
# With this data, The distribution plot of content is shown. At first glance, it doesn't have a normal trend and the data is centered towards 0

# In[12]:


sns.kdeplot(data=df_train_text, x="content",color="b", alpha=.5, fill=True)
plt.title("Content distribution")
plt.grid()


# The wording column shows a similar trend, but it is not bell-shaped as the previous one. Also, the values are centered toward 0.5.

# In[13]:


sns.kdeplot(data=df_train_text, x="wording",color="r", alpha=.5, fill=True)
plt.title("Wording distribution")
plt.grid()


# Both data is between -3 and 5, and as it was considered a linear regression model it will be important to make a data transformation to improve the modeling performance.

# # 3. Feature engineering and comparison
# 
# First, the total number of words was counted by a Lambda Function and the split function, treating each summary as a string variable and iterating through all the data.
# 
# Then, from the NLTK library, the main English stopwords were loaded.

# In[14]:


#Column for the total number of words
df_train_text["count"]=df_train_text.apply(lambda x: len(x["text"].split(' ')),axis=1)
#Baseline for stopwords
stopwords=nltk.corpus.stopwords
stopwords=stopwords.words('english')


# As an example, the first text is presented, showing the counting of total words.

# In[15]:


print("*"*30+"Text"+"*"*30)
print(df_train_text.text[0])
print("*"*30+"Function Example"+"*"*30)
len(df_train_text.text[0].split(' '))


# Later on, I created a function for counting the total stopwords of a text. In this case, as the texts are short I used a simple for comparison

# In[16]:


def count_stopwords(text,stopwords):
    #Splitting text
    split=text.lower().split(' ')
    count=0
    #Counting if the word is in the stopwords category
    for i in split:
        if i in stopwords:
            count+=1
    return count
#Example
count_stopwords(df_train_text.text[0],stopwords)


# In[17]:


#Creating stopwords column
df_train_text["stopwords"]=df_train_text.text.apply(count_stopwords,stopwords=stopwords)


# With a regex expression, I counted the total number of punctuation to see if there is a relation between the scoring and text structure.

# In[18]:


print("*"*30+"Sample Text"+"*"*30)
print(df_train_text.text[4])
# Regex expression to see total punctuation
print("*"*30+"Count of punctuation"+"*"*30)
print(len(re.findall(r'[\.,\n]', df_train_text.text[4])))
#Creating column for model comparison
df_train_text["punctuation"]=df_train_text.apply(lambda x: len(re.findall(r'[\.,\n]', x.text)),axis=1)


# Finally, the data was transformed to lower and without punctuation, to be able to create the proper modeling.

# In[19]:


#Text transformation
df_train_text["lower"]=df_train_text.text.str.lower() #lowercase
df_train_text["lower"]=[str(data) for data in df_train_text.lower] #converting all to string
df_train_text["lower"]=df_train_text.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x))


# In[20]:


df_train_text


# # 4. Plotting Features
# 
# Then, I created extra kdeplots to show if there was a relation between the objective variables and the new ones. Firstly, with the `count` feature it is presented that almost all the data has between 0 and 300 words and that the data is centered towards -2 and 1.

# In[21]:


sns.kdeplot(data=df_train_text, x="wording", y="count",color="r", alpha=.9, fill=True)
plt.title("Wording distribution vs total of words")
plt.ylabel("Total words")
plt.grid()


# In[22]:


sns.kdeplot(data=df_train_text, x="content", y="count",color="b", alpha=.9, fill=True)
plt.title("Content distribution vs total of words")
plt.ylabel("Total words")
plt.grid()


# Them, with the total number of stopwords and punctuation follows the same trend, showing a possible correlation between this new features:

# In[23]:


sns.kdeplot(data=df_train_text, x="wording", y="stopwords",color="r", alpha=.9, fill=True)
plt.title("Wording distribution vs total of stopwords")
plt.ylabel("Total words")
plt.grid()


# In[24]:


sns.kdeplot(data=df_train_text, x="content", y="stopwords",color="b", alpha=.9, fill=True)
plt.title("Content distribution vs total of stopwords")
plt.ylabel("Total stopwords")
plt.grid()


# In[25]:


sns.kdeplot(data=df_train_text, x="wording", y="punctuation",color="r", alpha=.9, fill=True)
plt.title("Wording distribution vs total of punctuation")
plt.ylabel("Total punctuation")
plt.grid()


# In[26]:


sns.kdeplot(data=df_train_text, x="content", y="punctuation",color="b", alpha=.9, fill=True)
plt.title("Content distribution vs total punctuation")
plt.ylabel("Total punctuation")
plt.grid()


# # 5 Data transformation and prediction
# 
# ![Image 2](https://images.pexels.com/photos/590041/pexels-photo-590041.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)
# 
# 
# Finally, for creating an training the model I adapted the pipeline from my original notebook to tokenize the words, beginning with a simple count of total features. There are more than 12k words in this data:

# In[27]:


#Text splitting
tokens_text = [word_tokenize(str(word)) for word in df_train_text.lower]
#Unique word counter
tokens_counter = [item for sublist in tokens_text for item in sublist]
print("Number of tokens: ", len(set(tokens_counter)))


# Then, I loaded the main English stopwords to build a Bag of words, a concept related to improving the model performance, as with previous versions I had bad final performances.

# In[28]:


#Choosing english stopwords
stopwords_nltk = nltk.corpus.stopwords
stop_words = stopwords_nltk.words('english')
stop_words[:5]


# In[29]:


#Initial Bag of Words
bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    stop_words=stop_words, #English Stopwords
    ngram_range=(1, 2) #analysis of two words
)


# Later, The data was splitted on 80% training and 20% test with `train_test_split` of sklearn.

# In[30]:


#Train - Test splitting
reviews_train, reviews_test = train_test_split(df_train_text, test_size=0.2, random_state=0)


# With this, the main data was divided by content and wording to create the two models, also transforming the variables with a robust scaler to improve the performance:

# In[31]:


# Scalers for the data
sc1=RobustScaler()
sc2=RobustScaler()
#Labels for train and test encoding
#Content model
y_train_bow = sc1.fit_transform(reviews_train[['content']])
y_test_bow = sc1.transform(reviews_test[['content']])
#Wording model
y_train_bow_2 = sc2.fit_transform(reviews_train[['wording']])
y_test_bow_2 = sc2.transform(reviews_test[['wording']])


# The text was transformed by the Bag of Words encoding and two linear regression models were trained:

# In[32]:


#Creation of encoding related to train dataset
X_train_bow = bow_counts.fit_transform(reviews_train.lower)
#Transformation of test dataset with train encoding
X_test_bow = bow_counts.transform(reviews_test.lower)


# In[33]:


from sklearn.linear_model import LinearRegression
# Linear regression for content
model1 = LinearRegression()
model1.fit(X_train_bow, y_train_bow)


# ## Model 1 performance

# In[34]:


# Prediction
from sklearn.metrics import mean_squared_error
test_pred = model1.predict(X_test_bow)
print("RSME: ", np.sqrt(mean_squared_error(y_test_bow, test_pred)))


# In[35]:


get_ipython().run_cell_magic('time', '', 'model2 = LinearRegression()\nmodel2.fit(X_train_bow, y_train_bow_2)\n')


# ## Model 2 performance

# In[36]:


test_pred = model1.predict(X_test_bow)
print("RSME: ", np.sqrt(mean_squared_error(y_test_bow_2, test_pred)))


# Then, the test data was transformed to create the final submission.

# In[37]:


#Text transformation
df_test_text["lower"]=df_test_text.text.str.lower() #lowercase
df_test_text["lower"]=[str(data) for data in df_test_text.lower] #converting all to string
df_test_text["lower"]=df_test_text.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x))


# In[38]:


#Transformation of test dataset with train encoding
X_test_bow_fin = bow_counts.transform(df_test_text.lower)


# In[39]:


#Prediction and transformation with standard scaler
df_samplesub["content"] = sc1.inverse_transform(model1.predict(X_test_bow_fin))
df_samplesub["wording"] = sc2.inverse_transform(model2.predict(X_test_bow_fin))
df_samplesub


# In[40]:


#Final Submission
df_samplesub.to_csv("submission.csv",index=False)


# # Final Remarks
# 
# In conclusion, our approach using simple Linear Regression demonstrated promising results in predicting the correctness of summaries. The code implementation, based on Sentiment Analysis encoding, achieved a reasonable Root Mean Squared Error (RSME) of 0.67, indicating that the initial model has some predictive power.
# 
# Moreover, it's worth noting that despite the prediction variables not exhibiting a normal trend, the combination of Linear Regression with Bag of Words still yielded favorable outcomes.
# 
# To further improve the model's performance, we can explore more advanced Natural Language Processing (NLP) techniques, such as employing Neural Networks, which are capable of capturing complex patterns in textual data.
# 
# As a next step, I would investigate the impact of the newly generated features on the overall model performance. Feature engineering plays a crucial role in improving predictive models, and it's essential to assess how these engineered features contribute to the model's accuracy measured on RSME.
# 
# It's worth highlighting that the content-based model, which relies on keywords used in each topic, might have distinct strengths and weaknesses compared to the wording-based model, which focuses on the manner in which words are written.
# 
# To address potential issues with word typos or misspellings, it will be considered the incorporation of external Python libraries for spell-checking and error detection during the wording modeling process.

# ![Image 3](https://images.pexels.com/photos/193821/pexels-photo-193821.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)
# 
# Thanks and give an upvote if you liked this notebook!
