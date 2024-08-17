#!/usr/bin/env python
# coding: utf-8

# # Titanic ship's log 
# 
# This my ongoing, autodidactive tutorial. I visited the Titanic several times but I can still find some exploratory secrets and topics to improve myself. Be careful with forking... as:  
# 
# **...you can't make an omelette without breaking eggs...** 

# <img src="https://cdn.pixabay.com/photo/2018/05/14/20/46/ship-3401500_1280.jpg" width="900px">
# 
# 
# # Table of contents
# 
# 1. [Exploratory data analysis](#eda)
# 2. [Further feature engineering](#engineering)
# 3. [Preprocessing](#preprocessing)

# Before we can start we need to load our packages:

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from os import listdir

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# # Exploratory data analysis <a class="anchor" id="eda"></a>
# 
# 1. [Loading data and first impressions](#load)
# 2. [Duplicated rows](#duplicates)
# 3. [Missing values](#nans)
# 4. [Survival](#survival)

# ## Loading data and first impressions <a class="anchor" id="load"></a>

# In[2]:


base_path = "../input/titanic/"
listdir(base_path)


# In[3]:


train = pd.read_csv(base_path + "train.csv", index_col=0)
train.head()


# ### Insights
# 
# * Our target "Survived" is binary.
# * Perhaps we can use the Name and the Ticket feature to create some features to explain groups.
# * To explain family structures we could try to use SibSp and Parch.
# * We encounter NaNs and have to deal with missing values.

# In[4]:


test = pd.read_csv(base_path + "test.csv", index_col=0)
test.head(2)


# In[5]:


train.shape[0] / test.shape[0]


# The train dataset is roughly twice as big as the test data. This makes it more difficult for us as we do not have much data to generalize well. Overfitting could be an always-present companion. ;-)

# In[6]:


submission = pd.read_csv(base_path + "gender_submission.csv")
submission.head(2)


# The goal is clear. For a given test ID we need to predict the survival with 0 (no) or 1 (yes). As accuracy is our evaluation metric it won't help much to use predicted probabilities.

# In[7]:


target = train.Survived.values
combined = train.drop("Survived", axis=1).append(test, sort=False)
combined.head()


# By using a combined dataset for train and test features (without target), we can make life much easier forfeature engineering and preprocessing.  

# ## Duplicated rows <a class="anchor" id="duplicates"></a>

# In[8]:


duplicated_rows = combined.duplicated()
duplicated_rows[duplicated_rows==True]


# There are no rows in the data that are completely identical. But what about the Name? It's a bit unrealistic to have the same Name for a person on the Titanic...

# In[9]:


same_name = combined.Name.value_counts()
same_name[same_name > 1]


# Ui! Two suspicious names were found!!

# In[10]:


combined.loc[combined.Name=="Kelly, Mr. James"]


# Hmm, they look very different and perhaps James Kelly was a common name during that time. Let's keep them!

# In[11]:


combined.loc[combined.Name=="Connolly, Miss. Kate"]


# Seems to be ok as well.

# ## Missing values <a class="anchor" id="nans"></a>
# 
# Ok, next topic! Which features have missing values? Does the "missingness" depend on the passenger ID? How many missing values per passenger are most common?

# In[12]:


missing_percentage = combined.isnull().sum().sort_values(ascending=False) / combined.shape[0]
missing_vals = missing_percentage.loc[missing_percentage > 0] * 100

fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.barplot(missing_vals.index, missing_vals.values, ax=ax[0], palette="Blues_r");
sns.countplot(combined.isnull().sum(axis=1), palette="Set2", ax=ax[1]);
ax[1].set_xlabel("Number of missing values \n per passenger");
ax[0].set_title("... per feature")
ax[0].set_ylabel("% with NaNs")
ax[1].set_title("... per passenger");


# In[13]:


plt.figure(figsize=(25,5))
sns.heatmap(combined.loc[:, ["Cabin", "Age", "Embarked", "Fare"]].isnull().transpose(), cmap="binary");


# ### Insights
# 
# * The cabin has the highest % of missing values. With more than 75% this feature is likely to be useless. 
# * For both features - age and cabin - we can see an interesting pattern: Some adjacent passenger ids show a clutted, similar NaN-structures. But this could also be a pure random effect! 
# 
# Let's take a look at one example:

# In[14]:


combined.loc[965:967]


# In this case the passengers 966 and 967 have indeed the same ticket number. This might explain why we know both cabin numbers. But 965 seems to be not related to them. Consequently our pattern could still be random somehow and probably we are not able to create a useful feature using NaNs.

# ## Survival <a class="anchor" id="survival"></a>
# 
# To analyse our target together with some features, we can only rely on the train data:

# In[15]:


fig, ax = plt.subplots(2,3,figsize=(20,10))
sns.countplot(train.Survived, palette="Set2", ax=ax[0,0])
sns.countplot(train.Sex, hue=train.Survived, ax=ax[0,1], palette="Reds")
sns.countplot(train.Pclass, hue=train.Survived, ax=ax[0,2], palette="Greens")
sns.countplot(train.Embarked, hue=train.Survived, ax=ax[1,0], palette="Oranges")
sns.countplot(train.SibSp, hue=train.Survived, ax=ax[1,1], palette="Purples")
sns.countplot(train.Parch, hue=train.Survived, ax=ax[1,2], palette="Purples");


# ### Insights
# 
# * Our target is imbalanced and we need to deal with it when applying machine learning models.
# * We can already observe that the sex and the passenger class are very predictive features. The Pclass is ordinal -  the higher the class number, the more passengers died within that group. 
# * In contrast the Embarked, SibSp and Parch features are not ordinal but categorical! One can't say: "The higher, the smaller or higher the probability to survive". For Embarked we could use a mapping like {C:1, Q:2, S:3} to encode with numbers, this way we would introduce a target dependent order. We could also use target encoding directly. Doing so there is still a danger to overfit to the train data as this frequency patterns must not be similar for the test data. This danger becomes higher the less frequent the category compared to the others and we should not trust it too much.  
# * We can see very less frequent categorical levels in the SibSp and Parch features: All numbers are seldom above 2! For this reason it could be useful to fuse levels.

# In[16]:


fig, ax = plt.subplots(1,3,figsize=(20,5))
sns.swarmplot(train.Survived, train.Fare, ax=ax[0], hue=train.Pclass);
sns.swarmplot(train.Survived, train.Fare.apply(np.log), ax=ax[1], hue=train.Pclass);
sns.swarmplot(train.Embarked, train.Fare.apply(np.log), ax=ax[2], hue=train.Pclass);


# ### Insights
# 
# * The fare looks strange!!!
# * It looks like many passengers of the 3rd class paid as much as some of the 1st class. This does not make much sense.
# * The same holds for the embarkation. There should at least be a small pattern here!

# In[17]:


fig, ax = plt.subplots(2,3,figsize=(20,10))
sns.swarmplot(train.Survived, train.Age, ax=ax[0,0], palette="Set2")
sns.violinplot(train.Pclass, train.Age, hue=train.Survived, ax=ax[0,1], split=True, palette="Greens")
sns.violinplot(train.Sex, train.Age, hue=train.Survived, ax=ax[0,2], split=True, palette="Reds");
sns.swarmplot(train.Parch, train.Age, hue=train.Survived, ax=ax[1,0], palette="Set2");
sns.swarmplot(train.SibSp, train.Age, hue=train.Survived, ax=ax[1,1], palette="Set2");
sns.swarmplot(train.Embarked, train.Age, hue=train.Survived, ax=ax[1,2], palette="Set2");


# ### Insights
# 
# * In the 2nd class almost all children survived!
# * The older the passengers the more likely is it that they belong to the upper classes.
# * Even though being a female was advantageous to survive more female children died than male. 
# * It looks like having a family was better to survive than being alone. 
# * Many people that have C as embarkation survived!

# In[18]:


fig, ax = plt.subplots(2,1,figsize=(25,12))
survival = np.copy(train.Survived.values)
sns.heatmap(train.Survived.values.reshape(1,-1), cmap="binary", ax=ax[0]);
ax[0].set_title("Original survival values per passenger id");

np.random.shuffle(survival)
sns.heatmap(survival.reshape(1,-1), cmap="binary", ax=ax[1]);
ax[1].set_title("Shuffled survival values per passenger id");


# ### Insights
# 
# * The original values look accumulated sometimes, but after comparing them with a random shuffle we can see that this occurs naturally. 
# * Consequently we should not expect a pattern here!

# # Feature engineering <a class="anchor" id="engineering"></a>
# 
# 1. [Missing value features](#mval_features)
# 2. [Family size](#family_size)
# 3. [Lifting the secret of the fare](#faresecret)
# 4. [What about the cabin and the deck?](#cabin)
# 5. [The ticket group](#ticketgroup)

# ## Missing value features <a class="anchor" id="mval_features"></a>
# 
# Let's count the number of missing entries per passenger and create a feature whether the passenger has missing values or not. 

# In[19]:


combined["num_missing_vals"] = combined.isnull().sum(axis=1)
combined["has_missing_vals"] = np.where(combined.isnull().sum(axis=1) >= 1, 1, 0)


# In[20]:


test = combined.iloc[train.shape[0]::].copy()
train = combined.iloc[0:train.shape[0]].copy()
train["Survived"] = target


# In[21]:


fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.countplot(train.num_missing_vals, hue=train.Survived, ax=ax[0], palette="Set2")
sns.countplot(train.has_missing_vals, hue=train.Survived, ax=ax[1], palette="Set2");


# We can clearly observe that passengers without missing values were more likely to survive. In my opinion this makes sense as these people had the opportunity to share and report further personal information after their surival. 

# ## Family size <a class="anchor" id="family_size"></a>

# In[22]:


combined["FamilySize"] = combined.Parch + combined.SibSp + 1

test = combined.iloc[train.shape[0]::].copy()
train = combined.iloc[0:train.shape[0]].copy()
train["Survived"] = target


# Let's try to find some relationships of the family size and some important features:

# In[23]:


fig, ax = plt.subplots(2,2,figsize=(20,10))
sns.countplot(train.FamilySize, hue=train.Survived, palette="Purples", ax=ax[0,0])
sns.swarmplot(train.FamilySize, train.Fare.apply(np.log), hue=train.Pclass, ax=ax[0,1])
ax[0,1].set_ylabel("log Fare")
sns.swarmplot(train.FamilySize, train.Age, hue=train.Survived, ax=ax[1,0])
sns.swarmplot(train.FamilySize, train.Age, hue=train.Pclass, ax=ax[1,1]);


# ### Insights
# 
# * A family size above 3-4 is very seldom and it seems that having such a big family was not advantegous to survive.
# * **Very strange!!! The fare seems to have a linear dependency with the family size!** The higher the number of family members the higher the fare! Could it be that the fare is given as a price per family and not per person?
# * In addition we can see that travelling as a family with a size of 2-4 was better than travelling alone.
# * Families with size greater than 4 often belonged to the thrid class. 

# ## Lifting the secret of the fare <a class="anchor" id="faresecret"></a>

# Let's take a look at the fare of one family:

# In[24]:


combined[combined.FamilySize==8].head(5)


# We can see that all members of the Goodwin family show the same fare and the same ticket! Perhaps the fare is given as price per ticket! Let's check this assumption:

# In[25]:


combined.groupby("Ticket").Fare.std().value_counts()


# Indeed! There is only one case where this idea is not true! Now we can use this knowledge to compute the price per person and the eliminate the linear dependency with the family size:

# In[26]:


combined["TicketGroupSize"] = combined.Ticket.map(combined.groupby("Ticket").size())
combined["SinglePrice"] = combined.Fare / combined.TicketGroupSize


# We still need to take a look at the anomaly:

# In[27]:


fare_stds = combined.groupby("Ticket").Fare.std()
fare_stds[fare_stds > 0]


# In[28]:


combined[combined.Ticket == "7534"]


# Two young men travelling with the same ticket and similar fares. But are these values given per person and does this fit to the thrid class members that travelled alone to S?

# In[29]:


test = combined.iloc[train.shape[0]::].copy()
train = combined.iloc[0:train.shape[0]].copy()
train["Survived"] = target

singles = train.loc[train.FamilySize==1].copy()
fig, ax = plt.subplots(1,3,figsize=(20,5))
sns.distplot(singles[(singles.Pclass==3) & (singles.Embarked=="S")].SinglePrice, kde=False, ax=ax[0]);
sns.distplot(singles[(singles.Pclass==2) & (singles.Embarked=="S")].SinglePrice, kde=False, ax=ax[1]);
sns.distplot(singles[(singles.Pclass==1) & (singles.Embarked=="S")].SinglePrice, kde=False, ax=ax[2]);
ax[0].set_title("3rd class")
ax[1].set_title("2nd class")
ax[2].set_title("1st class")
for n in range(3):
    ax[n].set_ylabel("Frequency")


# Ok, both fare values are valid and meaningful for members that travelled alone in the 3rd class to S. Interestingly we have some single prices close to 0?! Uhuhi! We have to take a look at these strange outliers!

# In[30]:


combined.loc[139, "SinglePrice"] = combined.loc[139].Fare
combined.loc[877, "SinglePrice"] = combined.loc[877].Fare


# In[31]:


test = combined.iloc[train.shape[0]::].copy()
train = combined.iloc[0:train.shape[0]].copy()
train["Survived"] = target


# In[32]:


fig, ax = plt.subplots(2,3,figsize=(20,12))
sns.swarmplot(train.Survived, train.SinglePrice, ax=ax[0,0], hue=train.Pclass);
sns.swarmplot(train.Survived, train.SinglePrice.apply(np.log), ax=ax[0,1], hue=train.Pclass);
ax[0,1].set_ylabel("Log single price")
sns.swarmplot(train.Embarked, train.SinglePrice.apply(np.log), ax=ax[0,2], hue=train.Pclass);
ax[0,2].set_ylabel("Log single price");
sns.swarmplot(train.Survived, train.Fare, ax=ax[1,0], hue=train.Pclass);
sns.swarmplot(train.Survived, train.Fare.apply(np.log), ax=ax[1,1], hue=train.Pclass);
sns.swarmplot(train.Embarked, train.Fare.apply(np.log), ax=ax[1,2], hue=train.Pclass);
ax[1,1].set_ylabel("Log single price")
ax[1,1].set_ylabel("Log single price");
for n in range(3):
    ax[0,n].set_title("New price pattern...")
    ax[1,n].set_title("compared to old fare pattern!");


# ### Insights
# 
# * Even though we find strange values close to zero for single prices, we were able to separate the fare from the family size. Now the plot makes sense!
#     * The price depends on the Pclass. The upper the class the higher was the price. 
#     * Furthermore the embarkation price groups dependent on the Pclass are clearer than before! But again, we find strange outliers here!

# ## What about the cabin and the deck? <a class="anchor" id="cabin"></a>
# 
# Probably it's not useful but nonetheless let's have a look at it:

# In[33]:


combined["Deck"] = combined.Cabin.str.extract(r'([a-zA-Z])')


# In[34]:


test = combined.iloc[train.shape[0]::].copy()
train = combined.iloc[0:train.shape[0]].copy()
train["Survived"] = target

fig, ax = plt.subplots(2,2,figsize=(20,10))
sns.countplot(combined.Deck, ax=ax[0,0], palette="rainbow");
sns.countplot(train.Deck, hue=train.Survived, ax=ax[0,1], palette="Set2");
sns.swarmplot(combined.Deck, combined.SinglePrice.apply(np.log), hue=combined.Pclass, ax=ax[1,0]);
sns.swarmplot(train.Deck, train.SinglePrice.apply(np.log), hue=train.Survived, ax=ax[1,1], palette="Set2");


# ### Insights
# 
# * As so many cabins are unknown the counts are very low. 
# * Some decks like G and F seemed to be only for the 2nd and 3rd class. 
# * As we have so many nan-values it's too difficult to say something about survival depending on the deck. 

# ## The ticket group <a class="anchor" id="ticketgroup"></a>
# 
# We have found that the fare was related to the ticket and now I'm curious whether all passengers that travelled with one ticket number also belong to one family. 

# In[35]:


def occupy_ticket(l):
    if l < 0:
        return "FamilyIsSplitted"
    if l > 0:
        return "MultipleFamilies"
    else:
        return "OneFamily"


# In[36]:


combined["TicketGroupSize"] = combined.Ticket.map(combined.groupby("Ticket").size())
combined["TicketGroupSize_FamilySize"] = combined.TicketGroupSize - combined.FamilySize
combined["TicketOccupation"] = combined.TicketGroupSize_FamilySize.apply(lambda l: occupy_ticket(l))
combined["OnePersonFamily"] = np.where(combined.FamilySize == 1, 1, 0)


# In[37]:


test = combined.iloc[train.shape[0]::].copy()
train = combined.iloc[0:train.shape[0]].copy()
train["Survived"] = target

fig, ax = plt.subplots(2,2,figsize=(20,10))
sns.countplot(combined.TicketGroupSize_FamilySize, ax=ax[0,0])
ax[0,0].set_xlabel("TicketGroupSize - FamilySize");
sns.swarmplot(combined.TicketGroupSize_FamilySize, combined.Age, ax=ax[0,1]);
sns.violinplot(train.TicketOccupation, train.Age, hue=train.Survived, split=True, ax=ax[1,0], palette="Purples")
sns.violinplot(train.TicketOccupation, train.Age, hue=train.OnePersonFamily, split=True, ax=ax[1,1], palette="Purples");


# ### Insights
# 
# * In cases of TicketGroupSize - FamilySize > 0 we can see that a ticket group can be composed of multiple families.
# * In contrast all cases with TicketGroupSize - FamilySize = 0 a ticket belongs to only one family.
# * In the case TicketGroupSize - FamilySize < 0 a single family is splitted over several tickets. 

# ### Examples - A ticket hold by several families

# In[38]:


multi_example = combined[(combined.TicketOccupation=="MultipleFamilies") & (combined.Ticket=="PC 17569")]
multi_example


# It seems that they travelled as a group indeed. They have similar ages and cabins that are all on deck B with numbers close to each other. 

# In[39]:


multi_example = combined[(combined.TicketOccupation=="MultipleFamilies")]
multi_example[multi_example.Ticket=="3101295"]


# In this case it seems that Miss Riihivouri was something like a Nanny. 

# # Preprocessing <a class="anchor" id="preprocessing"></a>
# 
# 1. [Understanding the importance of preprocessing](#understanding_preproc)
# 
# 
# 

# # Understanding the importance of preprocessing
# 
# Why should we preprocess our data? Ok, there is one obvious reson - we need a numerical representation of our features as our algorithms won't work with objects. But besides that there are more important reasons like how the features can influence the learning process of a machine learning model. Let's try to understand it by looking at the loss/error function first. We have to solve a binary classification problem and one way to express the error is the binary cross-entropy loss:
# 
# 

# ## One-hot encoding <a class="anchor" id="onehot"></a>

# Let's take a look at our features so far:

# In[40]:


combined.info()


# We can see that there are a lot of object features that our models won't be able to work with. We need a numercial respresentation and one way to do so is to use one-hot encoding for ctaegorical features that only hold a few levels. 

# In[41]:


to_encode = ["Pclass", "Embarked"]

