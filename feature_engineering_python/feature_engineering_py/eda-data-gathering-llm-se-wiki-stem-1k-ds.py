#!/usr/bin/env python
# coding: utf-8

# # <center style="font-family: consolas; font-size: 32px; font-weight: bold;"> Kaggle - LLM Science Exam</center>
# <p><center style="color:#949494; font-family: consolas; font-size: 20px;">Use LLMs to answer difficult science questions</center></p>
# 
# ***

# # <center style="font-family: consolas; font-size: 32px; font-weight: bold;">(ಠಿ⁠_⁠ಠ) Overview</center>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ The goal of the competition is to answer difficult science-based questions written by a Large Language Model (LLM).</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ The competition aims to help researchers understand the ability of LLMs to test themselves and explore the potential of LLMs that can be run in resource-constrained environments.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ The scope of large language model capabilities is expanding, and researchers are using LLMs to characterize themselves.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ Many existing natural language processing benchmarks have become trivial for state-of-the-art models, so there is a need to create more challenging tasks to test increasingly powerful models.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ The dataset for the competition was generated by providing snippets of text on various scientific topics to the gpt3.5 model and asking it to write multiple choice questions (with known answers). Easy questions were filtered out.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ Sign language recognition AI for text entry lags far behind voice-to-text or even gesture-based typing, as robust datasets didn't previously exist.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ The largest models currently run on Kaggle have around 10 billion parameters, while gpt3.5 has 175 billion parameters.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ The competition aims to explore whether a question-answering model more than 10 times smaller than gpt3.5 can effectively answer questions written by gpt3.5. The results will shed light on the benchmarking and self-testing capabilities of LLMs.</p>

# <p style="font-family: consolas; font-size: 16px;">🔴 Description and implementation of the data collection algorithm is described in the second section of this notebook.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">🔴 An implementation of this approach on the deberta-v3-large model can be found in this notebook -- <a href="https://www.kaggle.com/code/leonidkulyk/lb-0-709-llm-se-deberta-v3-large-i-1k-wiki"><strong>[LB: 0.709] LLM-SE ~ deberta-v3-large -i | 1k Wiki</strong></a>. Training notebook with weights and logs -- <a href="https://www.kaggle.com/code/leonidkulyk/lb-0-709-llm-se-deberta-v3-large-t-1k-wiki"><strong>[LB: 0.709] LLM-SE ~ deberta-v3-large -t | 1k Wiki</strong></a>.</p>

# <p style="font-family: consolas; font-size: 16px;">🔴 UPDATE (see previous versions):</p>
# 
# - <p style="font-family: consolas; font-size: 16px;">More accurate selection of STEM categories.</p>
# - <p style="font-family: consolas; font-size: 16px;">Randomizing STEM topics with a emphasis on "S".</p>

# #### <a id="top"></a>
# # <div style="box-shadow: rgb(60, 121, 245) 0px 0px 0px 3px inset, rgb(255, 255, 255) 10px -10px 0px -3px, rgb(31, 193, 27) 10px -10px, rgb(255, 255, 255) 20px -20px 0px -3px, rgb(255, 217, 19) 20px -20px, rgb(255, 255, 255) 30px -30px 0px -3px, rgb(255, 156, 85) 30px -30px, rgb(255, 255, 255) 40px -40px 0px -3px, rgb(255, 85, 85) 40px -40px; padding:20px; margin-right: 40px; font-size:30px; font-family: consolas; text-align:center; display:fill; border-radius:15px; color:rgb(60, 121, 245);"><b>Table of contents</b></div>
# 
# <div style="background-color: rgba(60, 121, 245, 0.03); padding:30px; font-size:15px; font-family: consolas;">
# <ul>
#     <li><a href="#0" target="_self" rel=" noreferrer nofollow">0. Import all dependencies</a></li>
#     <li><a href="#1" target="_self" rel=" noreferrer nofollow">1. Data overview</a>
#         <ul>
#             <li><a href="#1.1" target="_self" rel=" noreferrer nofollow">1.1 train.csv</a></li>
#             <li><a href="#1.2" target="_self" rel=" noreferrer nofollow">1.2 test.csv</a></li>
#         </ul>
#     </li>
#     <li><a href="#2" target="_self" rel=" noreferrer nofollow">2. Data gathering</a>
#         <ul>
#             <li><a href="#2.1" target="_self" rel=" noreferrer nofollow">2.1 Form a list of STEM topics</a></li>
#             <li><a href="#2.2" target="_self" rel=" noreferrer nofollow">2.2 Randomly select a category or a page</a></li>
#             <li><a href="#2.3" target="_self" rel=" noreferrer nofollow">2.3 Extract the text from selected page</a></li>
#             <li><a href="#2.4" target="_self" rel=" noreferrer nofollow">2.4 Compose a message to the LLM model</a></li>
#             <li><a href="#2.5" target="_self" rel=" noreferrer nofollow">2.5 Сombine all elements of the pipeline</a></li>
#         </ul>
#     </li>
#     <li><a href="#3" target="_self" rel=" noreferrer nofollow">3. Data gathering enhancements</a></li>
# </ul>
# 
# </div>

# <a id="0"></a>
# # <div style="box-shadow: rgba(0, 0, 0, 0.16) 0px 1px 4px inset, rgb(51, 51, 51) 0px 0px 0px 3px inset; padding:20px; font-size:32px; font-family: consolas; text-align:center; display:fill; border-radius:15px;  color:rgb(34, 34, 34);"> <b> 0. Import all dependencies </b></div>

# In[1]:


get_ipython().system('pip install openai itables Wikipedia-API  -q')


# In[2]:


import os
import random

import openai
import requests
import wikipediaapi
import itables
import numpy as np
import pandas as pd
import plotly.express as px
from kaggle_secrets import UserSecretsClient


# In[3]:


# specify OpenAI API key in Kaggle's secrets add-ons.
user_secrets = UserSecretsClient()
openai.api_key = user_secrets.get_secret("openai_api")


# <a id="1"></a>
# # <div style="box-shadow: rgba(0, 0, 0, 0.16) 0px 1px 4px inset, rgb(51, 51, 51) 0px 0px 0px 3px inset; padding:20px; font-size:32px; font-family: consolas; text-align:center; display:fill; border-radius:15px;  color:rgb(34, 34, 34);"> <b> 1. Data overview</b></div>

# <p style="font-family: consolas; font-size: 16px;">⚪ The dataset for this competition consists of multiple-choice questions generated by a Large Language Model (LLM).</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ The questions are accompanied by options labeled A, B, C, D, and E, and each question has a correct answer labeled "answer".</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ The goal is to predict the top three most probable answers given a question prompt. </p>
# 

# <a id="1.1"></a>
# ## <div style="box-shadow: rgba(0, 0, 0, 0.18) 0px 2px 4px inset; padding:20px; font-size:24px; font-family: consolas; text-align:center; display:fill; border-radius:15px; color:rgb(67, 66, 66)"> <b> 1.1 train.csv</b></div>

# <p style="font-family: consolas; font-size: 16px;">⚪ The train.csv file contains <b>200 questions</b> with their corresponding correct answers.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ Each question consists of a prompt (the question text) and options <b>A, B, C, D, and E</b>.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ The correct answer is indicated by the <code>answer</code> column, which contains the label of the most correct answer, as defined by the generating LLM. </p>
# 

# In[4]:


train_df = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/train.csv")


# In[5]:


table = itables.show(train_df, table_options=dict(pageLength=10))
table


# <p style="font-family: consolas; font-size: 16px;">⚪ Let's display the distributions of word counts for each of the fields.</p>

# In[6]:


fig = px.histogram([len(x.split(" ")) for x in train_df.prompt], nbins=40, color_discrete_sequence=['goldenrod'])
fig.update_layout(
    showlegend=False,
    xaxis_title="Number of words",
    title={
        'text': "Distribution of the number of words in prompts",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)
fig.show()


# In[7]:


fig = px.histogram([len(x.split(" ")) for x in train_df.A], nbins=40, color_discrete_sequence=['darkgreen'])
fig.update_layout(
    showlegend=False,
    xaxis_title="Number of words",
    title={
        'text': "Distribution of the number of words in option A",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)
fig.show()


# In[8]:


fig = px.histogram([len(x.split(" ")) for x in train_df.B], nbins=40, color_discrete_sequence=['cornflowerblue'])
fig.update_layout(
    showlegend=False,
    xaxis_title="Number of words",
    title={
        'text': "Distribution of the number of words in option B",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)
fig.show()


# In[9]:


fig = px.histogram([len(x.split(" ")) for x in train_df.C], nbins=40, color_discrete_sequence=['darkslateblue'])
fig.update_layout(
    showlegend=False,
    xaxis_title="Number of words",
    title={
        'text': "Distribution of the number of words in option C",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)
fig.show()


# In[10]:


fig = px.histogram([len(x.split(" ")) for x in train_df.D], nbins=40)
fig.update_layout(
    showlegend=False,
    xaxis_title="Number of words",
    title={
        'text': "Distribution of the number of words in option D",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)
fig.show()


# In[11]:


fig = px.histogram([len(x.split(" ")) for x in train_df.E], nbins=40, color_discrete_sequence=['darkolivegreen'])
fig.update_layout(
    showlegend=False,
    xaxis_title="Number of words",
    title={
        'text': "Distribution of the number of words in option E",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)
fig.show()


# <a id="1.2"></a>
# ## <div style="box-shadow: rgba(0, 0, 0, 0.18) 0px 2px 4px inset; padding:20px; font-size:24px; font-family: consolas; text-align:center; display:fill; border-radius:15px; color:rgb(67, 66, 66)"> <b> 1.2 test.csv</b></div>

# <p style="font-family: consolas; font-size: 16px;">⚪ The test.csv file contains the test set for the competition.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ <b>The task is to predict the top <code>3</code> most probable answers</b> for each question prompt.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ The format of the test set is the same as the training set, with questions, options (A, B, C, D, and E), and the prompt text.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ The test set has approximately 4,000 different prompts, which may differ in subject matter from the training set.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ <b>NOTE</b>: The test data you see here just a copy of the training data without the answers.</p>
# 

# In[12]:


test_df = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/test.csv")


# In[13]:


test_df.head()


# <a id="2"></a>
# # <div style="box-shadow: rgba(0, 0, 0, 0.16) 0px 1px 4px inset, rgb(51, 51, 51) 0px 0px 0px 3px inset; padding:20px; font-size:32px; font-family: consolas; text-align:center; display:fill; border-radius:15px;  color:rgb(34, 34, 34);"> <b> 2. Data gathering</b></div>

# <p style="font-family: consolas; font-size: 16px;">⚪ The most important part of this competition is data collection. Therefore, it is important to understand how the test dataset was formed and how to reproduce the method of its collection.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ According to the competition description, the test dataset was formed based on pages from Wikipedia. In other words, a page on the subject of Science, Technology, Engineering, and Mathematics (with the emphasis on "S") was selected, an excerpt was taken from it, and it was passed to the GPT3.5 model. The task of the model was to generate extensive multiple-choice questions based on the text. The output of the model is a multiple-choice question with options and a correct answer.</p>
# 
# 
# <p style="font-family: consolas; font-size: 16px;">🔴 To reproduce the collection of the competition's test data, the following steps need to be taken:</p>
# 
#    1. <p style="font-family: consolas; font-size: 16px;">Form a list of STEM categories for which the corresponding page will be searched to extract the test from it.</p>
#    2. <p style="font-family: consolas; font-size: 16px;">Randomly select a category or a page related to the corresponding topic or subtopic.</p>
#    3. <p style="font-family: consolas; font-size: 16px;">After selecting the page, extract the text from it.</p>
#    4. <p style="font-family: consolas; font-size: 16px;">Compose a message to the LLM model specifying what needs to be done and provide the extracted text.</p>
#    5. <p style="font-family: consolas; font-size: 16px;">Parse the model's output and perform an automatic check for compliance with the output format.</p>

# <p style="font-family: consolas; font-size: 16px;">🔴 If you are only interested in a dataset of <b>1,000 questions for 250 Wikipedia pages</b>, you can find it by following this <a href="https://www.kaggle.com/datasets/leonidkulyk/wikipedia-stem-1k"><strong>link</strong></a>. Below, the algorithm will be described, and the code will be provided to create or supplement this dataset.</p>

# <a id="2.1"></a>
# ## <div style="box-shadow: rgba(0, 0, 0, 0.18) 0px 2px 4px inset; padding:20px; font-size:24px; font-family: consolas; text-align:center; display:fill; border-radius:15px; color:rgb(67, 66, 66)"> <b> 2.1 Form a list of STEM topics</b></div>

# <p style="font-family: consolas; font-size: 16px;">⚪ Each article on Wikipedia has a list of categories to which it belongs.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ Categories are intended to group together pages on similar subjects. They are implemented by a MediaWiki feature that adds any page with a text like [[Category:XYZ]] in its wiki markup to the automated listing that is the category with name XYZ. Categories help readers to find, and navigate around, a subject area, to see pages sorted by title, and to thus find article relationships.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ Categories are normally found at the bottom of an article page. Clicking a category name brings up a category page listing the articles (or other pages) that have been added to that particular category. There may also be a section listing the subcategories of that category. The subcategorization feature makes it possible to organize categories into tree-like structures to aid navigation.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">🔴 More information about Wikipedia categories -- <a href="https://en.wikipedia.org/wiki/Help:Category"><strong>link 1</strong></a>, <a href="https://en.wikipedia.org/wiki/Category:Subfields_by_academic_discipline"><strong>link 2</strong></a>.</p>
# 

# <p style="font-family: consolas; font-size: 16px;">⚪ Thus, you can create a list of topics on which the page will be searched.</p>

# In[14]:


# probabilities: S -> 0.294; T,E,M -> 0.235
STEM_WEIGHTS = [1.25, 1, 1, 1]

STEM = {
    "S": ["Category:Applied_sciences", "Category:Biotechnology", "Category:Biology", "Category:Natural_history"],
    "T": [
        "Category:Technology_strategy", "Category:Technical_specifications", "Category:Technology_assessment", 
        "Category:Technology_hazards", "Category:Technology_systems", "Category:Hypothetical_technology", 
        "Category:Mobile_technology", "Category:Obsolete_technologies", "Category:Philosophy_of_technology", 
        "Category:Real-time_technology", "Category:Software", "Category:Technology_development", 
        "Category:Computing", "Category:Artificial_objects", "Category:Technological_change", 
        "Category:Technical_communication", "Category:Technological_comparisons"
    ],
    "E": ["Category:Engineering_disciplines", "Category:Engineering_concepts", "Category:Industrial_equipment", "Category:Manufacturing"],
    "M": ["Category:Fields_of_mathematics", "Category:Physical_sciences"]
}

EXCLUDE_CATEGORIES = set([
    "Category:Technology", "Category:Mathematics", "Category:Works about technology", 
    "Category:Technology evangelism", "Category:Artificial objects", "Category:Fictional physical scientists"
])


# <p style="font-family: consolas; font-size: 16px;">⚪ For example, here is what the list of categories for Category:Subfields by academic discipline looks like.</p>
# 
# <p style="text-align:center;"><img src="https://github.com/leo27heady/flask-basics/assets/45982614/fd2b0504-a8fc-44a3-b28a-a528fb4f7910" width="90%" height="90%"></p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ Now let's choose the category Category:Fields of mathematics and see what is inside it.</p>
# 
# <p style="text-align:center;"><img src="https://github.com/leo27heady/flask-basics/assets/45982614/23ba4f13-29b9-46b5-b213-31e039f14b5e" width="90%" height="90%"></p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ For the category Category: Fields of mathematics, we have not only subcategories but also pages. This means that we can go deeper into the subcategories or stop at a fairly high level and choose from the available pages.</p>

# <a id="2.2"></a>
# ## <div style="box-shadow: rgba(0, 0, 0, 0.18) 0px 2px 4px inset; padding:20px; font-size:24px; font-family: consolas; text-align:center; display:fill; border-radius:15px; color:rgb(67, 66, 66)"> <b> 2.2 Randomly select a category or a page</b></div>

# <p style="font-family: consolas; font-size: 16px;">🔴 The pipeline for selecting a random article looks as follows:</p>
# 
#    1. <p style="font-family: consolas; font-size: 16px;">Randomly choose a STEM topic, i.e., either S or T or E or M.</p>
#    2. <p style="font-family: consolas; font-size: 16px;">Based on the selected STEM topic, choose a random category from its list.</p>
#    3. <p style="font-family: consolas; font-size: 16px;">Take all the subcategories and pages of the chosen category.</p>
#    4. <p style="font-family: consolas; font-size: 16px;">Randomly select a list of subcategories or pages.</p>
#    5. <p style="font-family: consolas; font-size: 16px;">If the selected list is a list of subcategories, then choose a random subcategory and go to Step 3.</p>
#    6. <p style="font-family: consolas; font-size: 16px;">If the selected list is a list of pages, then choose a random page.</p>
#    7. <p style="font-family: consolas; font-size: 16px;">End.</p>

# <p style="font-family: consolas; font-size: 16px;">⚪ Python library <a href="https://pypi.org/project/Wikipedia-API/"><strong>Wikipedia-API</strong></a> will be used to interact with Wikipedia.</p>

# In[15]:


def split_category_members(members):
    category_list, page_list= [], []

    for member_name, member_page in members:
        if member_name.startswith('Category') and member_name not in EXCLUDE_CATEGORIES:
            category_list.append((member_name, member_page))
        else:
            page_list.append((member_name, member_page))
    
    return category_list, page_list

def get_wiki_random_page(deep_subcategories=True):
    stem_label, stem_categories = random.choices(list(STEM.items()), weights=STEM_WEIGHTS, k=1)[0]
    category = random.choice(stem_categories)
    category_page = wiki_wiki.page(category)
    while True:
        chosen_list = list(category_page.categorymembers.items())
        if deep_subcategories:
            category_list, page_list = split_category_members(chosen_list)
            chosen_list = []
        else:
            category_list, page_list = [], []

        # 50% change to select category or page list if one of them isn't empty
        # helps to go deeper into subcategories because there're more pages than categories
        if not (category_list or page_list) and not chosen_list:
            continue
        elif not category_list:
            chosen_list = page_list
        elif not page_list:
            chosen_list = category_list
        else:
            chosen_list = random.choice([category_list, page_list])

        # select random page from chosen list
        selected_page_name, selected_page = random.choice(chosen_list)

        if not selected_page_name.startswith("Category"):
            break
        
        category_page = selected_page
    
    return selected_page, stem_label


# <a id="2.3"></a>
# ## <div style="box-shadow: rgba(0, 0, 0, 0.18) 0px 2px 4px inset; padding:20px; font-size:24px; font-family: consolas; text-align:center; display:fill; border-radius:15px; color:rgb(67, 66, 66)"> <b> 2.3 Extract the text from selected page</b></div>

# <p style="font-family: consolas; font-size: 16px;">⚪ Text extraction is performed as follows: if a page is long enough (> 6 sentences), the first 7 sentences are taken from the page.</p>

# In[16]:


def get_wiki_text(seen_pages, min_page_length=6, sentences_include=3):
    while True:
        wiki_page, stem_label = get_wiki_random_page()

        if wiki_page.pageid in seen_pages:
            continue

        page_sentences = wiki_page.text.split(". ")
        
        # check is the page is long enought
        if len(page_sentences) >= min_page_length:
            # main information about the topic usualy described within first 3 sentences
            wiki_text = ". ".join(page_sentences[:sentences_include]) + "."
            break
    
    return wiki_text, wiki_page.pageid, wiki_page.title, stem_label


# <a id="2.4"></a>
# ## <div style="box-shadow: rgba(0, 0, 0, 0.18) 0px 2px 4px inset; padding:20px; font-size:24px; font-family: consolas; text-align:center; display:fill; border-radius:15px; color:rgb(67, 66, 66)"> <b> 2.4 Compose a message to the LLM model</b></div>

# <p style="font-family: consolas; font-size: 16px;">⚪ The message to the LLM should be precise and clearly convey the essence of the task. It is also good practice to specify the expected output format of the model, as it makes parsing much easier.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ Additionally, it is important to provide any necessary instructions or specifications regarding the desired output. This could include specifying the number of multiple-choice questions to generate, the length or format of the answer options, or any other specific requirements.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ Furthermore, it is recommended to include any relevant context or constraints that should be considered during the text generation process. This could involve providing guidelines on the desired style, tone, or level of complexity for the questions and answers.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ By providing a comprehensive and detailed message to the LLM, the model can better understand and fulfill the requirements, leading to more accurate and tailored results.</p>

# In[17]:


options_set = set(("option_1", "option_2", "option_3", "option_4", "option_5"))
response_keys_set = set(("question", "option_1", "option_2", "option_3", "option_4", "option_5", "answer"))

delimiter = "####"
system_message = f"""
You will be provided with TEXT from wikipedia. \
The TEXT will be delimited with {delimiter} characters.
Output a python list of 5 dict objects, where each object is \
a multiple choice question whom answers should be in \
the given TEXT and that has 5 choices each and has the following format:
    'question': <question on the TEXT>
    'option_1': <question answer option>
    'option_2': <question answer option>
    'option_3': <question answer option>
    'option_4': <question answer option>
    'option_5': <question answer option>
    'answer': <answer option key label>

You should tell me which one of your proposed options is right \
by assigning the corresponding option's key label in the 'answer' field.

The question, the answer and question answer options should be broad, \
challenging, long, detailed and based on the TEXT provided.

Only output the list of objects, with nothing else.
"""


# In[18]:


def get_completion_messages(wiki_text):
    return [  
        {
            'role':'system', 
            'content': system_message
        },    
        {
            'role':'user', 
            'content': f"{delimiter}{wiki_text}{delimiter}"
        },  
    ]

def get_completion_from_messages(
    messages, 
    model="gpt-3.5-turbo", 
    temperature=0.8, 
    max_tokens=3000
):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    return response.choices[0].message["content"]


# <a id="2.5"></a>
# ## <div style="box-shadow: rgba(0, 0, 0, 0.18) 0px 2px 4px inset; padding:20px; font-size:24px; font-family: consolas; text-align:center; display:fill; border-radius:15px; color:rgb(67, 66, 66)"> <b> 2.5 Сombine all elements of the pipeline</b></div>

# <p style="font-family: consolas; font-size: 16px;">⚪ In the function gather_multiple_choice_question_dataset, we specify the number of Wikipedia pages for which we want to generate 5 multiple choice question.</p>
# 
# <p style="font-family: consolas; font-size: 16px;">⚪ We also set the number of generation attempts per page. If the number of attempts exceeds the specified limit, the generation attempts for that page will be stopped, and another page will be selected.</p>

# In[19]:


def is_correctly_formatted(mcq) -> bool:
    return all([
        len(el) == len(response_keys_set) and response_keys_set == set(list(el.keys()))
        for el in mcq
    ])

def gather_multiple_choice_question_dataset(
    pages_count: int,
    max_completion_attempts: int = 10,
    seen_pages: list = []
):
    
    attempts_list = []
    multiple_choice_questions = []

    generated_count = 0
    while generated_count < pages_count:
        wiki_text, page_id, page_title, stem_label = get_wiki_text(seen_pages, sentences_include=7)
        print(f"\nStart multiple choice questions generation: page_id={page_id}, page_title={page_title}, stem_label={stem_label}")
        
        messages = get_completion_messages(wiki_text)

        attempts_counter = 0
        while True:
            try:
                chatgpt_response = get_completion_from_messages(messages)
                mcq = eval(chatgpt_response)

                if not isinstance(mcq, list) or len(mcq) < 5 or not is_correctly_formatted(mcq):
                    raise Exception

                for i in range(len(mcq)):
                    mcq[i]["wiki_text"] = wiki_text
                    mcq[i]["page_id"] = page_id
                    mcq[i]["page_title"] = page_title
                    mcq[i]["stem_label"] = stem_label

                    if mcq[i]["answer"] in options_set:
                        continue
                    else:
                        # index method will raise an error if answer isn't in list
                        answ_indx = [v.lower() for v in mcq[i].values()].index(mcq[i]["answer"].lower())
                        mcq[i]["answer"] = list(mcq[i].keys())[answ_indx]

                multiple_choice_questions += mcq
                seen_pages.append(page_id)
                generated_count += 1
                print("Generated count:", generated_count)
                break
            except Exception:
                attempts_counter += 1
                print("Attempts count:", attempts_counter)
                attempts_list.append(attempts_counter)
                if attempts_counter > max_completion_attempts:
                    break
    
    return multiple_choice_questions, seen_pages, attempts_list


# <p style="font-family: consolas; font-size: 16px;">⚪ 
# Let's try generating some example multiple choice questions for 2 pages (2 * 5 = 10 questions in total).</p>

# In[20]:


pages_count = 2
max_completion_attempts = 10

wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')


# In[23]:


multiple_choice_questions, seen_pages, attempts_list = gather_multiple_choice_question_dataset(
    pages_count, max_completion_attempts
)


# <p style="font-family: consolas; font-size: 16px;">⚪ Let's examine the output provided by the GPT3.5-turbo model.</p>

# In[24]:


df_mcq = pd.DataFrame.from_records(multiple_choice_questions)
table = itables.show(df_mcq)
table


# <p style="font-family: consolas; font-size: 16px;">⚪ To convert <code>df_mcq</code> to competition format - use this code snippet:</p>

# In[ ]:


def conver_df_to_compet_format(df):
    df_compet = df.copy(deep=True)
    df_compet.insert(0, "id", list(range(len(df_compet))))
    df_compet.rename(
        columns = {
            'question': 'prompt', 
            'option_1': 'A', 
            'option_2': 'B', 
            'option_3': 'C', 
            'option_4': 'D', 
            'option_5': 'E'
        }, 
        inplace = True
    )

    answer_subjects = {
        'option_1': 'A', 
        'option_2': 'B', 
        'option_3': 'C', 
        'option_4': 'D', 
        'option_5': 'E'
    }
    df_compet["answer"] = df_compet["answer"].map(answer_subjects)
    df_compet = df_compet.drop(columns=["wiki_text", "page_id", "page_title", "stem_label"])

    return df_compet


# In[ ]:


df_compet = conver_df_to_compet_format(df_mcq)
df_compet.to_csv("stem_dataset.csv", index=False)


# <a id="3"></a>
# # <div style="box-shadow: rgba(0, 0, 0, 0.16) 0px 1px 4px inset, rgb(51, 51, 51) 0px 0px 0px 3px inset; padding:20px; font-size:32px; font-family: consolas; text-align:center; display:fill; border-radius:15px;  color:rgb(34, 34, 34);"> <b> 3. Data gathering enhancements</b></div>

# <p style="font-family: consolas; font-size: 16px;">⚪ The most prioritized areas for improving the data collector are:</p>
# 
# - <p style="font-family: consolas; font-size: 16px;">Enhancing the output structure of the LLM model and the overall prompt.</p>
# - <p style="font-family: consolas; font-size: 16px;">More accurate selection of STEM categories.</p>
# - <p style="font-family: consolas; font-size: 16px;">Filtering out simple questions.</p>
# - <p style="font-family: consolas; font-size: 16px;">Improving the algorithm for randomization to delve deeper into subcategories.</p>
# - <p style="font-family: consolas; font-size: 16px;">Randomizing STEM topics with a emphasis on "S".</p>
# - <p style="font-family: consolas; font-size: 16px;">"Smarter" selection of excerpts from pages.</p>
# - <p style="font-family: consolas; font-size: 16px;">Check the length of the requested prompt tokens and reduce it if needed.</p>
# - <p style="font-family: consolas; font-size: 16px;">Try another open source LLM for multiple choice question text completion.</p>

# # <div style="box-shadow: rgba(240, 46, 170, 0.4) -5px 5px inset, rgba(240, 46, 170, 0.3) -10px 10px inset, rgba(240, 46, 170, 0.2) -15px 15px inset, rgba(240, 46, 170, 0.1) -20px 20px inset, rgba(240, 46, 170, 0.05) -25px 25px inset; padding:20px; font-size:30px; font-family: consolas; display:fill; border-radius:15px; color: rgba(240, 46, 170, 0.7)"> <b> ༼⁠ ⁠つ⁠ ⁠◕⁠‿⁠◕⁠ ⁠༽⁠つ Thank You!</b></div>
# 
# <p style="font-family:verdana; color:rgb(34, 34, 34); font-family: consolas; font-size: 16px;"> 💌 Thank you for taking the time to read through my notebook. I hope you found it interesting and informative. If you have any feedback or suggestions for improvement, please don't hesitate to let me know in the comments. <br><br> 🚀 If you liked this notebook, please consider upvoting it so that others can discover it too. Your support means a lot to me, and it helps to motivate me to create more content in the future. <br><br> ❤️ Once again, thank you for your support, and I hope to see you again soon!</p>
