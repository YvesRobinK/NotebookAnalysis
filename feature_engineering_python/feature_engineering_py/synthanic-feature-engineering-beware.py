#!/usr/bin/env python
# coding: utf-8

# # Synthanic feature engineering: Beware!
# One of the keys to success, besides [overfitting and underfitting the Titanic](https://www.kaggle.com/carlmcbrideellis/overfitting-and-underfitting-the-titanic), is feature selection and feature engineering, *i.e.* "*...[the process of using domain knowledge to extract features from raw data](https://en.wikipedia.org/wiki/Feature_engineering)*". In this short notebook we look at a few examples where the "story telling" that was applicable to the *Titanic* competition cannot be directly translated to the *Synthanic* competition.
# 
# For example here is a simple comparison with the *Titanic* data:
# 
# ### Only the women survived: *Synthanic* Public Score = 0.78505
# This is akin to the `gender_submission.csv` file that is associated with the *Titanic* competition

# In[1]:


import pandas as pd
import numpy  as np

# read in the data
train_data = pd.read_csv('../input/tabular-playground-series-apr-2021/train.csv')
test_data  = pd.read_csv('../input/tabular-playground-series-apr-2021/test.csv')
all_data   = pd.concat([train_data, test_data], axis=0)


# In[2]:


# assume that nobody survived
predictions = np.zeros((test_data.shape[0]), dtype=int)

# except for the women
survived_df = test_data[(test_data["Sex"]== 'female')]
for i in survived_df.index:
    predictions[i] = 1 # the 1's are now the survivors


# ### Only the women from 1*st* and 2*nd* class survived: *Synthanic* Public Score = 0.74045

# In[3]:


# assume that nobody survived
predictions = np.zeros((test_data.shape[0]), dtype=int)

# except for the women in 1st and 2nd class
survived_df = test_data[((test_data["Pclass"] ==1)|(test_data["Pclass"] ==2)) & (test_data["Sex"]== 'female')]
for i in survived_df.index:
    predictions[i] = 1 # the 1's are now the survivors


# ### Comparison with the Titanic data
# 
# |  | Titanic | Synthanic |
# | --- | --- | --- |
# | All women survived|  0.76555 | 0.78505 |
# | Women (1st & 2nd) | 0.77512 | 0.74045 |
# 
# we can see a clear reversal in the Public Leaderboard scores, indicating that the survival dynamics have changed.
# # Family ties
# When creating synthetic data it is very tricky to maintain the relationships between columns. These are known as [constraints](https://sdv.dev/SDV/user_guides/single_table/constraints.html#single-table-constraints). For example, imagine creating two columns of synthetic data, `Country` and `City`. When creating synthetic data it is easy to populate the `Country` column by randomly selecting countries from a list of all countries, and the same goes for `City`. However, in real world data one would expect that there are certain cities that belong in certain countries. If one does not apply such a constraint, this relationship between the two columns is lost.
# 
# In a recent post ["*Family Relationship Broken? Are Real World Features Possible?*"](https://www.kaggle.com/c/tabular-playground-series-apr-2021/discussion/230190) by [Neil Cosgrove](https://www.kaggle.com/neilcosgrove) it has been observed that the *Synthanic* data does not have the same family constraints between columns as in the *Titanic* data. 
# 
# ### Ticket
# For example, let us take a look at `Ticket` number 10867

# In[4]:


all_data.query('Ticket == "10867"')


# we can see that in the data there are 14 people with the very same Ticket number embarked at Cherbourg, Queenstown and Southampton, and the passenger class (`Pclass`) was variously 1st, 2nd or 3rd, and indeed many of them were seemingly traveling alone. We can also see that the fare for Rober Basinski's 1st class ticket was £5.28, whilst Eric Ferguson forked out an eye watering £81.99 to travel in 3rd class (adjusted for inflation that would be nearly £10,000 today). Meanwhile, over on the *Titanic* everyone paid the same price for the same ticket, which was much more 'fair' (excuse the pun).
# 
# ### Cabin
# This time let us look at Cabin C11139

# In[5]:


all_data.query('Cabin == "C11139"')


# we can see that 7 people from all three ports chose to share this spacious first class cabin. Walter Yancy and Tony Lower paid £221.85 and £264.83 each. However, Edward Bowman had a great deal, paying only £26.67 (but probably had to sleep on the floor...)
# 
# ## Family size
# A classic example of feature engineering applied to the *Titanic* dataset is to create the new feature `family_size`

# In[6]:


all_data['family_size'] = all_data['SibSp'] + all_data['Parch'] + 1


# and when doing this we find an example of a family of size 18 is aboard the *Synthanic*, this seems somewhat large but it is a very big boat after all (and in the Titanic data there was the Sage family with 11 members). 
# 
# However, when we look more closely there are actually only four people in this family of 18 in the data. Evidently the `SibSp` and `Parch` features are not self-consistent:

# In[7]:


all_data.query('family_size == 18')


# Unraveling the `SibSp/Parch` notation system is somewhat convoluted (see for example the notebook ["*Extracting family relationships on Titanic: SibSp*"](https://www.kaggle.com/ailuropus/extracting-family-relationships-on-titanic-sibsp) by [ailuropus](https://www.kaggle.com/ailuropus)) however the numbers seem to indicate that these four passengers, despite their ages, are all children belonging to a polygamous family. Basically, creating the `family_size` feature for the *Synthanic* will probably not be as useful as it was for the *Titanic*.
# 
# ## Unaccompanied children
# On the *Titanic* there were nine unaccompanied children on board, representing $\approx 0.7\%$ of the passengers. On the *Synthanic* there were well over four thousand such children

# In[8]:


all_data[((all_data["family_size"] ==1) & (all_data["Age"] < 16 ))].shape[0]


# making up $\approx 2.4\%$ of the passengers. Given the grim social conditions in the post-Victorian era, perhaps the *Synthanic* was also relocating an orphanage to a better life in The Land of Opportunity?...
# 
# # Conclusion
# We can see here in this small example that the 'story' that the women from the 1st and 2nd class had  better survival that including women form all classes has changed between the *Titanic* dataset and the *Synthanic* dataset. We can also see that some of the feature engineering that works for the Titanic does not work for the Synthanic. So beware when it comes to Synthanic feature engineering; whilst the story behind the *Titanic* became a huge movie that broke all the box office records, the story behind the *Synthanic* has some serious '[plot holes](https://en.wikipedia.org/wiki/Plot_hole)'.

# # Related reading
# * [*KISS: Small and simple Titanic models*](https://www.kaggle.com/carlmcbrideellis/kiss-small-and-simple-titanic-models)
# * [*Basic Feature Engineering with the Titanic Data*](https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/)
# * [CTGAN](https://sdv.dev/SDV/user_guides/single_table/ctgan.html) user guide
# * [Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni "*Modeling Tabular Data using Conditional GAN*", arXiv:1907.00503 (2019)](https://arxiv.org/pdf/1907.00503.pdf)
# 
# #### Here is how to save a predictions file:

# In[9]:


# write out the submission data
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

