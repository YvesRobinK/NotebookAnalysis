#!/usr/bin/env python
# coding: utf-8

# #### Lots of manual feature engineering for readability and text features!
# 
# 
# #### textstacy:
# * https://textacy.readthedocs.io/en/0.12.0/api_reference/text_stats.html#textacy.text_stats.api.TextStats
# 
# #### textstat:
# * `textstat`  - https://github.com/kupolak/textstat/blob/master/README.md
# * Seems to overlap with textacy, but may be easier to use? 
# * https://www.kaggle.com/code/yhirakawa/textstat-how-to-evaluate-readability 
# 
# 
# TODO:
# * More features, e.g. parsing based:
# * https://www.kaggle.com/code/sharrpie/commonlit-readability-eda-fe-topic-modelling
# * Warning - slow!!  (Even slower if using `en_core_web_lg`)
# 
# Example of dropping correlated feats: https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on

# In[1]:


get_ipython().system(' pip install -q language_tool_python textacy textstat spacy # -U')


# In[2]:


# ! pip install textstat > /dev/null
# ! pip install gensim==3.8.3 > /dev/null
# ! pip install pyLDAvis==2.1.2 > /dev/null
# ! pip install spacy==2.3.0 > /dev/null
# ! pip install textacy > /dev/null
# ! python -m spacy download en_core_web_lg

# ! python -m spacy download en_core_web_md


# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import language_tool_python
import textacy
from textacy import text_stats
import textstat
import matplotlib.pyplot as plt

import spacy
nlp = spacy.load('en_core_web_sm',exclude=["ner","entity_linker","textcat"]) ## "lemmatizer")#('en_core_web_md') #('en_core_web_lg')


# In[4]:


FASTRUN = False
# FASTRUN = True


# In[5]:


tool = language_tool_python.LanguageTool('en-US')


# In[6]:


import re

# define a precompiled regular expression
RE_SUSPICIOUS = re.compile(r'[&#<>{}\[\]\\]')

def impurity(text, min_len=10):
    """
    Returns the share of suspicious characters in a text.
    Very short texts (less than min_len) are ignored.
    """
    if text == None or len(text) < min_len:
        return 0
    else:
        return len(RE_SUSPICIOUS.findall(text))/len(text)


# In[7]:


df = pd.read_csv('../input/feedback-prize-english-language-learning/train.csv')
df.drop(columns=["text_id"],inplace=True) # not used
df["mean_score"] = df.select_dtypes("number").mean(axis=1).round(2) # extra target
# df["var_score"] = df.select_dtypes("number").var(axis=1).round(3)  # extra target
df


# In[8]:


if FASTRUN:
    df = df.sample(15)


# In[9]:


### note: needs to run as submission in order to work
df_test = pd.read_csv("../input/feedback-prize-english-language-learning/test.csv")
df_test


# ##### Grammar Score Analysis
# See if the grammar score is correlated with the number of errors found by a standard spellchecker (https://pypi.org/project/language-tool-python/). (Based on notebook: 
# 

# In[10]:


# %%time
# ## slow - 5 min for max 500
MAX_LEN = 550
df['errors_per_unit'] = df.full_text.apply(lambda x: len(tool.check(x[:MAX_LEN])) / len(x[:MAX_LEN]))
df_test['errors_per_unit'] = df_test.full_text.apply(lambda x: len(tool.check(x[:MAX_LEN])) / len(x[:MAX_LEN]))

df['errors_sum'] = df['errors_per_unit']*(df.full_text.str.split().str.len())
df_test['errors_sum'] = df_test['errors_per_unit']*(df_test.full_text.str.split().str.len()) ## not quite accurate but close enough


# In[11]:


df[['errors_sum','errors_per_unit']]


# In[12]:


get_ipython().run_cell_magic('time', '', 'df["impurity"] = df["full_text"].apply(impurity, min_len=9)\ndf_test["impurity"] = df_test["full_text"].apply(impurity, min_len=9)\n\ndf["newline_count"] = df["full_text"].str.count("\\n")\ndf_test["newline_count"] = df_test["full_text"].str.count("\\n")\n')


# In[13]:


import seaborn as sns
try:
    sns.jointplot(x=df['errors_per_unit'], y=df['grammar'], kind='hist')

    print('correlation={}'.format(df[['errors_per_unit', 'grammar']].corr().values[0][1]))
except:()


# In[14]:


get_ipython().run_cell_magic('time', '', 'def get_word_count(text):\n    return textstat.lexicon_count(text, removepunct=True)\n\n# https://github.com/textstat/textstat\ndef get_textstat_features(df):\n    df["flesch_reading"] = df["full_text"].apply(textstat.flesch_reading_ease)\n    df["smog_index"] = df["full_text"].apply(textstat.smog_index)\n    df["flesch_kincaid"] = df["full_text"].apply(textstat.flesch_kincaid_grade) # like others, can also get from textacy\n    df["coleman_liau"] = df["full_text"].apply(textstat.coleman_liau_index)\n    df["automated_readability_index"] = df["full_text"].apply(textstat.automated_readability_index)\n    df["dale_chall_readability"] = df["full_text"].apply(textstat.dale_chall_readability_score)\n    df["difficult_words"] = df["full_text"].apply(textstat.difficult_words)\n    \n    df[\'szigriszt_pazos\'] = df.full_text.apply(textstat.szigriszt_pazos)\n    df[\'fernandez_huerta\'] = df.full_text.apply(textstat.fernandez_huerta)\n    df[\'gutierrez_polini\'] = df.full_text.apply(textstat.gutierrez_polini)\n    df[\'crawford\'] = df.full_text.apply(textstat.crawford)\n    df["linsear_write_formula"] = df["full_text"].apply(textstat.linsear_write_formula)\n    df["gunning_fog"] = df["full_text"].apply(textstat.gunning_fog)\n    df["text_standard"] = df["full_text"].apply(textstat.text_standard, float_output=True)\n    df["mcalpine_eflaw"]= df["full_text"].apply(textstat.mcalpine_eflaw)\n    df["reading_time"]= df["full_text"].apply(textstat.reading_time)\n    \n    df["lexicon_count"]= df["full_text"].apply(textstat.lexicon_count)\n    \n    ## all of the following are probably redundnat to the textACY features later: \n    df["syllable_count"]= df["full_text"].apply(textstat.syllable_count)\n    df["sentence_count"]= df["full_text"].apply(textstat.sentence_count)\n    df["n_chars"]= df["full_text"].apply(textstat.char_count)\n    df["letter_count"]= df["full_text"].apply(textstat.letter_count)\n    df["polysyllabcount"]= df["full_text"].apply(textstat.polysyllabcount) \n    df["monosyllabcount"]= df["full_text"].apply(textstat.monosyllabcount)\n\n    df[\'word_count\'] = df["full_text"].apply(get_word_count) # can be removed (in favor of ts) barring feature berlow\n    df[\'difficult_word_ratio\'] = df[\'difficult_words\'] / df[\'word_count\'] # https://www.kaggle.com/code/sharrpie/commonlit-readability-eda-fe-topic-modelling\n    return df\n')


# In[15]:


df = get_textstat_features(df)
df_test = get_textstat_features(df_test)


# #### TextAcy features
# 
# * May want to check if they are identical to textstats version, e.g. flesch kincaid

# In[16]:


### https://textacy.readthedocs.io/en/latest/api_reference/text_stats.html#textacy.text_stats.api.TextStats
def textacy_row(row):
    doc = textacy.make_spacy_doc(row["full_text"], lang="en_core_web_sm")
    ts = textacy.text_stats.TextStats(doc)
    row["n_words"] = ts.n_words # duplicate feat
    row["n_sents"] = ts.n_sents
#     row["n_chars"] = ts.n_chars

    row["n_unique_words"] = ts.n_unique_words
    row["n_long_words"] = ts.n_long_words
    row["n_monosyllable_words"] = ts.n_monosyllable_words
    row["n_polysyllable_words"] = ts.n_polysyllable_words
    row["frac_long_words"] = row["n_long_words"]/row["n_words"]
    
    row["frac_monosyllable"] = row["n_monosyllable_words"]/row["n_words"]
    row["frac_polysyllable"] = row["n_polysyllable_words"]/row["n_words"]
    row["frac_long_unique"] = row["n_long_words"]/row["n_unique_words"] # could try other combinations
    
    row["av_words_per_sent"] = row["n_words"]/row["n_sents"]
    row["av_char_per_sent"] = row["n_chars"]/row["n_sents"]
    z = ts.n_chars_per_word
    row["mean_n_chars_per_word"] = np.mean(z)
    row["max_n_chars_per_word"] = np.max(z)
    z = ts.n_syllables_per_word
    row["mean_n_syllables_per_word"] = np.mean(z)
    row["max_n_syllables_per_word"] = np.max(z)
    
#     row["t_flesch_kincaid_grade_level"] = ts.flesch_kincaid_grade_level ## this is redundnat, only used to validate it being identical to textstats outputs!
    row["smog"] = ts.smog_index #  intended as a substitute for gunning_fog
    row["t_entropy"] = ts.entropy # we calc entropy in another way befoer
    row["lix"] = ts.lix # Readability test for both English- and non-English-language texts
    row["mu_legibility_index"] = ts.mu_legibility_index # Readability test for Spanish texts based on number of words and the mean and variance of characters lengths
    row["perspicuity_index"] = ts.perspicuity_index  # Readability for Spanish; very similar to the Spanish-specific flesch_reading_eas
    row["wiener_sachtel"] = ts.wiener_sachtextformel # german readability
    return row


# In[17]:


get_ipython().run_cell_magic('time', '', 'df = df.apply(textacy_row,axis=1)\ndisplay(df)\n')


# In[18]:


get_ipython().run_cell_magic('time', '', 'df_test = df_test.apply(textacy_row,axis=1)\n')


# #### check features correlation/redundnancy

# In[19]:


# plot the heatmap
corr = df.iloc[:,7:].corr() # drop most target cols
plt.figure(figsize=(20,12))

sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns, )


# In[20]:


df.iloc[:,7:].corrwith(df["flesch_kincaid"]).sort_values() # validate identical values for feature from sources.
# note - very high correlation! 


# ### Parse tree features
# * Source:  https://www.kaggle.com/code/sharrpie/commonlit-readability-eda-fe-topic-modelling#Feature-Engineering-
# 

# In[21]:


from collections import Counter
def tree_height(root):
    if not list(root.children):
        return 1
    else:
        return 1 + max(tree_height(x) for x in root.children)


def get_average_height(paragraph):
    if type(paragraph) == str:
        doc = nlp(paragraph)
    else:
        doc = paragraph
    roots = [sent.root for sent in doc.sents]
    return np.mean([tree_height(root) for root in roots])


def count_subtrees(root):
    if not list(root.children):
        return 0
    else:
        return 1 + sum(count_subtrees(x) for x in root.children)


def get_mean_subtrees(paragraph):
    if type(paragraph) == str:
        doc = nlp(paragraph)
    else:
        doc = paragraph
    roots = [sent.root for sent in doc.sents]
    return np.mean([count_subtrees(root) for root in roots])


def get_averge_noun_chunks(paragraph):
    if type(paragraph) == str:
        doc = nlp(paragraph)
    else:
        doc = paragraph
    return len(list(doc.noun_chunks))
    
def get_noun_chunks_size(paragraph):
    if type(paragraph) == str:
        doc = nlp(paragraph)
    else:
        doc = paragraph
    noun_chunks_size = [len(chunk) for chunk in doc.noun_chunks]
    return np.mean(noun_chunks_size)


def get_pos_freq_per_word(paragraph, tag):
    if type(paragraph) == str:
        doc = nlp(paragraph)
    else:
        doc = paragraph
    pos_counter = Counter(([token.pos_ for token in doc]))
    pos_count_by_tag = pos_counter[tag]
    total_pos_counts = sum(pos_counter.values())
    return pos_count_by_tag / total_pos_counts


# In[22]:


def get_pos_tree_feats(df,TEXT_COL="doc"):
    df[TEXT_COL] = df["full_text"].apply(nlp) # new, save on tokenizing each time... 
    df['avg_parse_tree_height'] = df[TEXT_COL].apply(get_average_height)
    df['mean_parse_subtrees'] = df[TEXT_COL].apply(get_mean_subtrees)
    df['noun_chunks'] = df[TEXT_COL].apply(get_averge_noun_chunks)
    df['avg_noun_chunks'] = df['noun_chunks'] / df['sentence_count']
    df['noun_chunk_size'] = df[TEXT_COL].apply(get_noun_chunks_size)
    df['mean_noun_chunk_size'] = df['noun_chunk_size'] / df['avg_noun_chunks']
    
    ### 60% of time is following funcs:
    df['nouns_per_word'] = df[TEXT_COL].apply(lambda x: get_pos_freq_per_word(x, 'NOUN'))
    df['proper_nouns_per_word'] = df[TEXT_COL].apply(lambda x: get_pos_freq_per_word(x, 'PROPN'))
    df['pronouns_per_word'] = df[TEXT_COL].apply(lambda x: get_pos_freq_per_word(x, 'PRON'))
    df['adj_per_word'] = df[TEXT_COL].apply(lambda x: get_pos_freq_per_word(x, 'ADJ'))
    df['adv_per_word'] = df[TEXT_COL].apply(lambda x: get_pos_freq_per_word(x, 'ADV'))
    df['verbs_per_word'] = df[TEXT_COL].apply(lambda x: get_pos_freq_per_word(x, 'VERB'))
    df['cconj_per_word'] = df[TEXT_COL].apply(lambda x: get_pos_freq_per_word(x, 'CCONJ'))
    df['sconj_per_word'] = df[TEXT_COL].apply(lambda x: get_pos_freq_per_word(x, 'SCONJ'))
#     df['conj_per_word'] = df[TEXT_COL].apply(lambda x: get_pos_freq_per_word(x, 'CONJ')) ## added - does it work?  all 0s? 
    df.drop(columns=TEXT_COL,inplace=True,errors="ignore")
    return df


# In[ ]:





# In[23]:


df


# In[24]:


get_ipython().run_cell_magic('time', '', '## very slow - ~ 1 second per row if using all POS feats - faster following improvements to make a spacy doc\ndf = get_pos_tree_feats(df)\n')


# In[25]:


df_test = get_pos_tree_feats(df_test)


# In[26]:


df


# ### Export outputs
# * TODO: Train model and add more features

# In[27]:


df.to_parquet("english-learning_train.parquet",index=False)

