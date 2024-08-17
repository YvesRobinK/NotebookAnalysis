#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python -m textblob.download_corpora')


# In[2]:


pip install '/kaggle/input/pyphen-0100'


# In[3]:


import pandas as pd
import numpy as np


# In[4]:


prompts_train_df = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/prompts_train.csv")
summaries_train_df = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/summaries_train.csv")


# In[5]:


summaries_train_df.head(10)


# In[6]:


import pandas as pd

# Getting general statistics
unique_students = summaries_train_df['student_id'].nunique()
unique_prompts = summaries_train_df['prompt_id'].nunique()

# Getting descriptive statistics for content and wording scores
content_score_stats = summaries_train_df['content'].describe()
wording_score_stats = summaries_train_df['wording'].describe()

unique_students, unique_prompts, content_score_stats, wording_score_stats


# Content Score: The mean content score is slightly below zero, indicating that a large number of students have received negative scores for their summaries. However, the maximum score of 3.9 shows that there are summaries with high content quality. The wide range of scores from -1.73 to 3.9 suggests a diverse set of responses in terms of content quality.
# 
# Wording Score: Similar to the content score, the mean wording score is also slightly below zero, indicating a tendency towards lower scores. The distribution is slightly wider for the wording score, with scores ranging from -1.96 to 4.31, implying a larger variability in the wording quality of the summaries. This suggests a wide variation in the linguistic proficiency displayed in the students' summaries.

# In[7]:


import numpy as np

# Calculating word count and average word length for the summaries and prompt texts
summaries_train_df['summary_word_count'] = summaries_train_df['text'].str.split().str.len()
summaries_train_df['summary_avg_word_length'] = summaries_train_df['text'].str.split().apply(lambda x: np.mean([len(word) for word in x]) if isinstance(x, list) else None)

# Merging the summaries and prompts dataframes to get prompt texts corresponding to each summary
merged_df = summaries_train_df.merge(prompts_train_df[['prompt_id', 'prompt_text']], on='prompt_id')

# Calculating word count and average word length for the prompt texts
merged_df['prompt_word_count'] = merged_df['prompt_text'].str.split().str.len()
merged_df['prompt_avg_word_length'] = merged_df['prompt_text'].str.split().apply(lambda x: np.mean([len(word) for word in x]) if isinstance(x, list) else None)

# Getting the descriptive statistics for the new columns
summary_word_count_stats = merged_df['summary_word_count'].describe()
summary_avg_word_length_stats = merged_df['summary_avg_word_length'].describe()
prompt_word_count_stats = merged_df['prompt_word_count'].describe()
prompt_avg_word_length_stats = merged_df['prompt_avg_word_length'].describe()

summary_word_count_stats, summary_avg_word_length_stats, prompt_word_count_stats, prompt_avg_word_length_stats


# Observations:
# Summaries:
# Students' summaries contain an average of around 75 words, with a fairly large spread in word count as indicated by the standard deviation.
# The average word length in summaries is around 4.56 characters, with a relatively small variability.
# Prompts:
# The prompts have a much higher word count compared to the summaries, averaging at 688 words.
# The average word length in the prompts is slightly higher than in the summaries, with a mean value of 4.68 characters.

# In[8]:


# Calculating the correlation between content and wording scores
correlation_matrix = summaries_train_df[['content', 'wording']].corr()

# Calculating the average content and wording scores per prompt
average_scores_per_prompt = summaries_train_df.groupby('prompt_id')[['content', 'wording']].mean().reset_index()

correlation_matrix, average_scores_per_prompt


# Observations:
# There is a high positive correlation (0.75138) between the content and wording scores. This suggests that summaries that received a high content score also tended to receive a high wording score, and vice versa.

# The prompt with ID "814d6b" has the highest average scores for both content and wording, indicating that students performed the best on this prompt.
# The prompt with ID "ebad26" has the lowest average wording score, whereas the prompt with ID "39c16e" has the lowest average content score.
# There is variability in the average scores across different prompts, suggesting that some prompts might be easier or harder for the students.

# In[9]:


import matplotlib.pyplot as plt

# Setting up the figure and axes
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# Plotting histograms
ax[0, 0].hist(summaries_train_df['content'], bins=30, color='skyblue', edgecolor='black')
ax[0, 0].set_title('Content Score Distribution')
ax[0, 0].set_xlabel('Content Score')
ax[0, 0].set_ylabel('Frequency')

ax[0, 1].hist(summaries_train_df['wording'], bins=30, color='skyblue', edgecolor='black')
ax[0, 1].set_title('Wording Score Distribution')
ax[0, 1].set_xlabel('Wording Score')
ax[0, 1].set_ylabel('Frequency')

ax[1, 0].hist(merged_df['summary_word_count'], bins=30, color='skyblue', edgecolor='black')
ax[1, 0].set_title('Summary Word Count Distribution')
ax[1, 0].set_xlabel('Word Count')
ax[1, 0].set_ylabel('Frequency')

ax[1, 1].hist(merged_df['summary_avg_word_length'], bins=30, color='skyblue', edgecolor='black')
ax[1, 1].set_title('Summary Average Word Length Distribution')
ax[1, 1].set_xlabel('Average Word Length')
ax[1, 1].set_ylabel('Frequency')

# Adjusting the layout to prevent overlap
plt.tight_layout()

# Displaying the plots
plt.show()


# Content Score Distribution
# 
# The distribution is somewhat normal but slightly left-skewed, with a peak a bit below zero. This indicates that a significant number of summaries have received negative content scores.
# 
# Wording Score Distribution
# 
# Similar to the content score distribution, the wording score distribution is also somewhat normal but slightly left-skewed, with most scores being below zero.
# Summary Word Count Distribution
# 
# The distribution is highly right-skewed, indicating that most summaries contain a relatively small number of words, with a few summaries having a very high word count.
# Summary Average Word Length Distribution
# 
# The distribution is fairly normal, centered around a mean of approximately 4.5 characters per word

# In[10]:


# Setting up the figure and axes
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# Plotting box plots for scores and text metrics per prompt
merged_df.boxplot(column='content', by='prompt_id', ax=ax[0, 0], grid=False)
ax[0, 0].set_title('Content Score by Prompt')
ax[0, 0].set_xlabel('Prompt ID')
ax[0, 0].set_ylabel('Content Score')

merged_df.boxplot(column='wording', by='prompt_id', ax=ax[0, 1], grid=False)
ax[0, 1].set_title('Wording Score by Prompt')
ax[0, 1].set_xlabel('Prompt ID')
ax[0, 1].set_ylabel('Wording Score')

merged_df.boxplot(column='summary_word_count', by='prompt_id', ax=ax[1, 0], grid=False)
ax[1, 0].set_title('Summary Word Count by Prompt')
ax[1, 0].set_xlabel('Prompt ID')
ax[1, 0].set_ylabel('Word Count')

merged_df.boxplot(column='summary_avg_word_length', by='prompt_id', ax=ax[1, 1], grid=False)
ax[1, 1].set_title('Summary Average Word Length by Prompt')
ax[1, 1].set_xlabel('Prompt ID')
ax[1, 1].set_ylabel('Average Word Length')

# Adjusting the layout to prevent overlap
plt.tight_layout()

# Displaying the plots
plt.show()


# Content Score by Prompt
# 
# The prompt with ID "814d6b" has the highest median content score, indicating that students generally performed well on this prompt in terms of content.
# The interquartile range (IQR) is quite wide for all prompts, showing a substantial variation in the content scores received by different summaries for each prompt.
# Wording Score by Prompt
# 
# Similar to the content score distribution, the prompt "814d6b" also has the highest median wording score.
# The prompt "ebad26" has the lowest median wording score and shows a substantial number of outliers with very low scores, indicating that many students struggled with wording for this prompt.
# Summary Word Count by Prompt
# 
# The word count distribution varies significantly across different prompts.
# The prompt "39c16e" has a lower median word count compared to others, indicating that summaries for this prompt tend to be shorter.
# Summary Average Word Length by Prompt
# 
# The average word length is fairly consistent across different prompts, with slight variations in the median values.
# The box plots show a few outliers, indicating that there are summaries with unusually high or low average word lengths.

# In[11]:


# Setting up the figure and axes
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plotting scatter plots for content and wording scores against summary word count
ax[0].scatter(merged_df['summary_word_count'], merged_df['content'], alpha=0.5, color='blue')
ax[0].set_title('Content Score vs Summary Word Count')
ax[0].set_xlabel('Summary Word Count')
ax[0].set_ylabel('Content Score')

ax[1].scatter(merged_df['summary_word_count'], merged_df['wording'], alpha=0.5, color='green')
ax[1].set_title('Wording Score vs Summary Word Count')
ax[1].set_xlabel('Summary Word Count')
ax[1].set_ylabel('Wording Score')

# Adjusting the layout to prevent overlap
plt.tight_layout()

# Displaying the plots
plt.show()

# Calculating the correlation between summary word count and scores
word_count_content_corr = merged_df['summary_word_count'].corr(merged_df['content'])
word_count_wording_corr = merged_df['summary_word_count'].corr(merged_df['wording'])

word_count_content_corr, word_count_wording_corr


# Observations:
# Content Score vs Summary Word Count:
# 
# There is a clear positive correlation between the word count of the summaries and the content scores, indicating that longer summaries tend to receive higher content scores.
# The correlation coefficient is approximately
# 0.793
# 0.793.
# Wording Score vs Summary Word Count:
# 
# There is a positive correlation between the word count of the summaries and the wording scores, though it is less strong compared to the correlation with the content scores.
# The correlation coefficient is approximately
# 0.536
# 0.536.
# Conclusions:
# Summaries with a higher word count generally tend to receive higher scores, both in terms of content and wording. However, this correlation is stronger for content scores compared to wording scores.
# It is important to note that while there is a correlation, it does not imply causation. A longer summary is not guaranteed to have a higher score, as the quality of the content and wording also play crucial roles.

# In[12]:


# Calculating the number of unique words in each summary
merged_df['unique_word_count'] = merged_df['text'].str.split().apply(lambda x: len(set(x)) if isinstance(x, list) else None)

# Setting up the figure and axes
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plotting scatter plots for content and wording scores against unique word count
ax[0].scatter(merged_df['unique_word_count'], merged_df['content'], alpha=0.5, color='blue')
ax[0].set_title('Content Score vs Unique Word Count')
ax[0].set_xlabel('Unique Word Count')
ax[0].set_ylabel('Content Score')

ax[1].scatter(merged_df['unique_word_count'], merged_df['wording'], alpha=0.5, color='green')
ax[1].set_title('Wording Score vs Unique Word Count')
ax[1].set_xlabel('Unique Word Count')
ax[1].set_ylabel('Wording Score')

# Adjusting the layout to prevent overlap
plt.tight_layout()

# Displaying the plots
plt.show()

# Calculating the correlation between unique word count and scores
unique_word_count_content_corr = merged_df['unique_word_count'].corr(merged_df['content'])
unique_word_count_wording_corr = merged_df['unique_word_count'].corr(merged_df['wording'])

unique_word_count_content_corr, unique_word_count_wording_corr


# Observations:
# Content Score vs Unique Word Count:
# 
# There is a strong positive correlation between the number of unique words used in the summaries and the content scores, indicating that summaries with a greater variety of words tend to receive higher content scores.
# The correlation coefficient is approximately
# 0.807
# 0.807.
# Wording Score vs Unique Word Count:
# 
# A positive correlation exists between the number of unique words and the wording scores, albeit less strong compared to the content scores. This suggests that a higher variety of words in the summaries tends to be associated with higher wording scores.
# The correlation coefficient is approximately
# 0.544
# 0.544.
# 

# In[ ]:





# In[13]:


pip install "/kaggle/input/pyphen/pyphen-0.13.0-py3-none-any.whl"


# In[14]:


import nltk


# In[15]:


import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from textblob import TextBlob, Word
import pyphen
import nltk

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Simulated Semantic Similarity using TF-IDF
def simulated_semantic_similarity(source, summary):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([source, summary])
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return cosine_sim

# N-gram overlap function
def ngram_overlap(source, summary, n=2):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    ngrams = vectorizer.fit_transform([source, summary])
    overlap = ngrams.toarray()[0] * ngrams.toarray()[1]
    return sum(overlap > 0) / len(vectorizer.get_feature_names_out())

# Extract frequently used words from high-scoring summaries
def extract_better_words_from_high_scoring_summaries(data, threshold=0.8):
    high_score_summaries = data[(data['content'] >= threshold) & (data['wording'] >= threshold)]['text']
    word_list = [word for summary in high_score_summaries for word in summary.split()]
    word_freq = Counter(word_list)
    better_words = [word for word, freq in word_freq.most_common(100)]
    return better_words

def better_words_proportion(text, word_list):
    text_words = text.split()
    better_words_count = sum(1 for word in text_words if word in word_list)
    return better_words_count / len(text_words)

def unique_better_words_proportion(text, word_list):
    text_words = set(text.split())
    unique_better_words_count = sum(1 for word in text_words if word in word_list)
    return unique_better_words_count / len(text_words)


# Initialize Pyphen
dic = pyphen.Pyphen(lang='en')

def count_syllables(word):
    hyphenated_word = dic.inserted(word)
    return len(hyphenated_word.split('-'))

def flesch_reading_ease_manual(text):
    total_sentences = len(TextBlob(text).sentences)
    total_words = len(TextBlob(text).words)
    total_syllables = sum(count_syllables(word) for word in TextBlob(text).words)

    if total_sentences == 0 or total_words == 0:
        return 0

    flesch_score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    return flesch_score

def sentiment_polarity(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Feature Engineering Function
def enhanced_feature_engineering(dataframe, better_words_list):
    dataframe['simulated_semantic_similarity'] = dataframe.apply(
        lambda row: simulated_semantic_similarity(row['prompt_text'], row['text']),
        axis=1
    )
    dataframe['bigram_overlap'] = dataframe.apply(
        lambda row: ngram_overlap(row['prompt_text'], row['text'], n=2),
        axis=1
    )
    dataframe['better_words_proportion'] = dataframe['text'].apply(lambda x: better_words_proportion(x, better_words_list))
    dataframe['unique_better_words_proportion'] = dataframe['text'].apply(lambda x: unique_better_words_proportion(x, better_words_list))
    dataframe['flesch_reading_ease'] = dataframe['text'].apply(flesch_reading_ease_manual)
    dataframe['sentiment_polarity'] = dataframe['text'].apply(sentiment_polarity)
    return dataframe

# Enhanced preprocessing function
def preprocess_data(data):
    merged_df = pd.merge(data['prompts_df'], data['summaries_df'], on="prompt_id")
    text_columns = ['prompt_question', 'prompt_title', 'prompt_text', 'text']
    for column in text_columns:
        merged_df[column] = merged_df[column].apply(clean_text)
    merged_df['prompt_length'] = merged_df['prompt_text'].apply(len)
    merged_df['summary_length'] = merged_df['text'].apply(len)
    merged_df['prompt_unique_words'] = merged_df['prompt_text'].apply(lambda x: len(set(x.split())))
    merged_df['summary_unique_words'] = merged_df['text'].apply(lambda x: len(set(x.split())))
    return merged_df

# Training function
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def train_models(data):
    # Data Preprocessing: Handling missing values, encoding categorical features, etc.
    # For now, we will fill missing values with the median of the column
    data = data.fillna(data.median(numeric_only=True))
    
    # Getting the features and target variables
    features = data.drop(columns=['prompt_id', 'student_id', 'prompt_question', 'prompt_title', 'prompt_text', 'text', 'content', 'wording'])
    y_content = data['content']
    y_wording = data['wording']

    # Splitting the data into training and validation sets (only once)
    X_train, X_val, y_train_content, y_val_content = train_test_split(features, y_content, test_size=0.2, random_state=42)
    _, _, y_train_wording, y_val_wording = train_test_split(features, y_wording, test_size=0.2, random_state=42)

    # Creating a pipeline with a scaler and the regressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Defining the parameter grid for GridSearchCV
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [None, 10, 20, 30],
    }

    # Initializing GridSearchCV and finding the best hyperparameters for the content model
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train_content)
    rf_regressor_content = grid_search.best_estimator_

    # Finding the best hyperparameters for the wording model
    grid_search.fit(X_train, y_train_wording)
    rf_regressor_wording = grid_search.best_estimator_

    # Storing the trained models in a dictionary
    models = {
        "rf_regressor_content": rf_regressor_content,
        "rf_regressor_wording": rf_regressor_wording,
    }

    # Evaluate the models on the validation set and report performance metrics
    y_pred_content = rf_regressor_content.predict(X_val)
    y_pred_wording = rf_regressor_wording.predict(X_val)
    
    metrics_content = {
        "MSE": mean_squared_error(y_val_content, y_pred_content),
        "R2": r2_score(y_val_content, y_pred_content),
    }
    
    metrics_wording = {
        "MSE": mean_squared_error(y_val_wording, y_pred_wording),
        "R2": r2_score(y_val_wording, y_pred_wording),
    }

    return models, metrics_content, metrics_wording


# Define the make_predictions function
def make_predictions(test_data, trained_models):
    # Extracting the models dictionary from the tuple
    models_dict = trained_models[0]

    features = test_data.drop(columns=['prompt_id', 'student_id', 'prompt_question', 'prompt_title', 'prompt_text', 'text'])
    
    predictions_content = models_dict['rf_regressor_content'].predict(features)
    predictions_wording = models_dict['rf_regressor_wording'].predict(features)
    
    results_df = pd.DataFrame({
        'content': predictions_content,
        'wording': predictions_wording
    })

    return results_df




# Load and preprocess training data
prompts_train_df = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/prompts_train.csv")
summaries_train_df = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/summaries_train.csv")
data = {
    'prompts_df': prompts_train_df,
    'summaries_df': summaries_train_df
}
preprocessed_train_data = preprocess_data(data)
better_words_from_data = extract_better_words_from_high_scoring_summaries(preprocessed_train_data)
preprocessed_train_data = enhanced_feature_engineering(preprocessed_train_data, better_words_from_data)
trained_models = train_models(preprocessed_train_data)

# Load and preprocess test data
prompts_test_df = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/prompts_test.csv")
summaries_test_df = pd.read_csv("/kaggle/input/commonlit-evaluate-student-summaries/summaries_test.csv")
data = {
    'prompts_df': prompts_test_df,
    'summaries_df': summaries_test_df
}
preprocessed_test_data = preprocess_data(data)
preprocessed_test_data = enhanced_feature_engineering(preprocessed_test_data, better_words_from_data)
predictions_df = make_predictions(preprocessed_test_data, trained_models)
predictions_df['student_id'] = preprocessed_test_data['student_id']
predictions_df = predictions_df[['student_id', 'content', 'wording']]
predictions_df.to_csv("submission.csv", index=False)
predictions_df.head()

