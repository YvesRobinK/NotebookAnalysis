#!/usr/bin/env python
# coding: utf-8

# Replace `language-tool-python` with T5 deobfuscator

# In[1]:


import spacy

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from collections import Counter

from tqdm import tqdm


@torch.no_grad()
def clean_essay(text):
    # top_chars = set([c for c, _ in Counter(text).most_common(10)])
    # if "t" in top_chars and "s" in top_chars:
    #     return text
    doc = nlp(text)
    inputs = tokenizer([s.text for s in doc.sents], truncation=True, padding=True, return_tensors="pt")
    outputs = deobfuscator.generate(inputs.input_ids.to(DEVICE), max_length=300)
    sents = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return " ".join([s.strip() for s in sents])


MODEL_PATH = "/kaggle/input/essay-gec/deobfuscator-v1"
DEVICE = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
deobfuscator = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE).eval()

nlp = spacy.load("en_core_web_sm")


# The following code is based on [LLM-Detect AI-GT Sub](https://www.kaggle.com/code/siddhvr/llm-detect-ai-gt-sub) notebook.

# * This is worth mentioning that the Public LB for this competition is highly overfitted since you can see the test set is so small. Even including a text preprocessing (which is a standard process) pipeline is decreasing the score on the LB. So I would like to advise all the people who are directly forking and making the submissions to not completely rely on these notebooks. Try making more robust models with text cleaning and preprocessing functions, better text encoders like Word2Vec, BERT and better models like LSTMs (Sequential) or GNNs (Graph-Based) so that you have a good score in the Private LB as well.
# 

# * The next few notebooks I'll publish will be having better models and text preprocessing pipelines. Just a heads up, I have made some submissions with a private notebook and the scores are not good, but we'll see that these notebooks will score higher on the Private LB.

# # Importing library

# In[2]:


import regex as re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB


# In[3]:


train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv")
external_train = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")
external_train.rename(columns={"generated": "label"}, inplace=True)


# In[4]:


def seed_everything(seed=202):
    import random
    random.seed(seed)
    np.random.seed(seed)

seed_everything()


# # Data Imports and Feature Engineering

# In[5]:


not_persuade_df = train[train["source"] != "persuade_corpus"]
persuade_df = train[train["source"] == "persuade_corpus"]
sampled_persuade_df = persuade_df.sample(n=6000, random_state=42)

all_human = set(list("".join(sampled_persuade_df.text.to_list())))
other = set(list("".join(not_persuade_df.text.to_list())))
chars_to_remove = "".join([x for x in other if x not in all_human])
print(chars_to_remove)

translation_table = str.maketrans("", "", chars_to_remove)
def remove_chars(s):
    return s.translate(translation_table)


# In[6]:


train = pd.concat([train, external_train])
test = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv")


# In[7]:


clean_texts = []
for i, r in tqdm(test.iterrows(), total=len(test)):
    clean_texts.append(clean_essay(r.text))
test["text"] = clean_texts

del deobfuscator
torch.cuda.empty_cache()


# In[8]:


train["text"] = train["text"].apply(remove_chars)
train["text"] = train["text"].str.replace("\n", "")
test["text"] = test["text"].str.replace("\n", "")
test["text"] = test["text"].apply(remove_chars)


# In[9]:


vectorizer = TfidfVectorizer(
    ngram_range=(3, 5),
    tokenizer=lambda x: re.findall(r"[^\W]+", x),
    token_pattern=None,
    strip_accents="unicode",
)
test_x = vectorizer.fit_transform(test["text"])
train_x = vectorizer.transform(train["text"])


# # Models

# In[10]:


lr = LogisticRegression()
clf = MultinomialNB(alpha=0.02)
sgd_model = SGDClassifier(max_iter=5000, tol=1e-3, loss="modified_huber")   
sgd_model2 = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber", class_weight="balanced") 
sgd_model3 = SGDClassifier(max_iter=10000, tol=5e-4, loss="modified_huber", early_stopping=True)


# # Voting Classifier

# In[11]:


ensemble = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("mnb", clf),
        ("sgd", sgd_model),
        ("sgd2", sgd_model2),
        ("sgd3", sgd_model3),
    ],
    voting="soft",
)
ensemble.fit(train_x, train.label)


# In[12]:


test["generated"] = ensemble.predict_proba(test_x)[:, 1]


# # Submission

# In[13]:


test[["id", "generated"]].to_csv("submission.csv", index=False)

