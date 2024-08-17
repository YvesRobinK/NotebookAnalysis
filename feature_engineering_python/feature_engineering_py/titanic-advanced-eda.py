#!/usr/bin/env python
# coding: utf-8

# Please do NOT upvote this kernel!!
# 
# ***************************************************************
# Note: Oct 2021 - It has been slightly more than a year since I wrote this kernel. With some experience gained in real competitions (a modest 3 bronzes in last one year with weekend Kaggling), I went back to this code which was my first Kaggle program and hence special to me. There is one strong positive and one strong negative message I want to convey about this code. The negative message is about the readability & simplicity of the Python code. When moving from other conventional languages to Python, a lot of people often tend to get quite excited about the wonders 1-line of Python code can achieve. I too went overboard with the 1-liners trying to make the Python code as compact as possible and in the process the whole program has lost its simplicity and beauty. This is not how a program is meant to be written. If there is one thing I have learnt in the past year, it is that code should be kept as simple as possible and this kernel is definitely not at that standard. Simplicity has been compromised to make the code more compact. 
# 
# The GOOD news is that I still find the EDA very relevant and highly impactful. After 1 year year, if I had to go back and re-do the EDA and come up with observations about the dataset, I couldnt have done a better job than what is already done. Titanic competition is a decade old and yet the EDA has at least 5-6 interesting observations which havent been reported by anyone before in the last 10 years and is worthy of further investigation. Hence I will not be deleting this kernel, but would like to request users not to upvote the same unless the code is simplified. In simplicity lies elegance and just like the code your model too should reflect that simplicity. Guanshuo Xu is the Kaggle number 1 and his profile page carries a simple message - "Build a model is very simple, but build a simple model is the hardest thing there is". Very profound.
# 
# Wishing you all the very best on your Kaggle journey
# ***************************************************************
# 
# 
# # Captivating conversations with the Titanic Dataset using Python/Pandas
# ![Screen%20Shot%202020-06-08%20at%2010.12.08%20PM.jpg](attachment:Screen%20Shot%202020-06-08%20at%2010.12.08%20PM.jpg)
# 
# If you are a newcomer, this easy EDA notebook which has less than 70 lines of code takes you thru' the entire data exploration & visualization stage from beginners to advanced perspective. It is extremely easy to follow even if one is a beginner to both ML and Python.
# 
# Edit: 10 Sep: An even more simple version of Titanic exists at: https://www.kaggle.com/allohvk/titanic-simplest-tutorial-ever-code-as-a-story
# It is a fun tutorial meant for rank beginners and introduces the fascinating world of ML in a story-like fashion
# 
# Do check out my blog on towardsdatascience.com - A case for altruism on the high seas and other interesting stories - New inferences from a decade old Titanic dataset of Kaggle
# https://towardsdatascience.com/a-case-for-altruism-on-the-high-seas-and-other-interesting-stories-970a04444db2
# 
# Interesting vignettes tumbled out during the course of my recent discussions with the Titanic dataset. Some of them well documented in the past and some not. A few samples:
# - *Would you feel safer if you were traveling Second class or Third class?* Contrary to popular opinion made famous by Hollywood movies, you had **DOUBLE** the chance of surviving the Titanic as an adult male if you travelled **THIRD** class instead **SECOND** class. You many not see too many other kernels talking about this interesting fact
# - "*What's in a Name? That which we call a rose, By any other name would smell as sweet*" -  Act II, Scene II - Unfortunately the Titanic data set seems to violently disagree with Juliet (& the old Bard) for people with short names had extremely high mortality rate on the Titanic compared to people with long names
# - *A case for Altruism on the high seas* - Darwin was proved wrong that night, but does the data speak of any other acts of altruism that went undocumented?
# - *Oops Errata on the Titanic dataset tutorial blogs* - One original mistake, several copies
# - *Females on the Titanic may have been charged a little more* - But males paid with a lot more 
# - *Correlation is not Causation* - The fallacy of the 'Survival of the Groups' conclusion
# - *What is the single biggest indicator of survival on the Titanic data set* - No it is none of the 11 columns listed in the dataset
# - *No matter what the leaders say - going solo does not necessarily mean you had lower chances of survival*. If you were a female, you **WOULD** want to fly solo in order to survive the Titanic. Upon probing why, the data narrates a heart-wrenching tale involving the ultimate act of sacrifice that played out on that fateful night more than 100 years back...
# 
# The Titanic competition was hosted in late 2012. It involved predicting whether passengers on the Titanic would survive or not..given a set of characteristics like their age, Sex, class of travel, port Embarked  etc. The data set consists of a training data file and a test data file in CSV form. The files are very small. The aspiring data scientist is supposed to make his model 'learn' from the training data set (which contain an additional column on whether the passenger survived or not) and apply these 'learnings' to the test data set and predict which of the passengers in the test dataset survives. I thought of peppering my code with enough documentation so that it can act as a Tutorial for other new comers like me. 
# 
# These are just recorded transcripts of a series of fascinating conversations I had with a very old dataset (a centenarian) over the course of many Sundays. These conversations were in Python - a language I tried to learn just to talk to her. A bit of ML theory is good-to-have while going thru' this code, but even if you don't have that it is perfectly fine as you will be able to easily understand the code & the intent. It is not necessary for you to know either Python or Pandas to understand the 70 odd simple lines of code that follow. Interestingly enough, though I wanted this to be a beginners exercise, it ended up being an extremely inquisitive research into this dataset..leading to some very startling observations. A bit of programming experience in any language is all that is needed as a pre-requisite. Each time, I use a new Python or Pandas command, I explain what it does... So the whole code reads somewhat like an interesting story (at least that is my hope).
# 
# From 2012 to 2017, scores on the leaderboard for the Titanic competition moved upto a high of 81-82%. About couple of years back, there was a renewed interest in this data set and the needle was pushed by to 83-84% where it is currently stuck. The renewed interest was primarily due to Chris Deotte (currently Kaggle number 1 on the discussion board and number 2 on Notebooks board) who applied some innovative models to squeeze out more info from the dataset and inspired others to come up with similar such solutions. In fact, not only did Chris raise the bar higher but he also actively participated in many discussions around this dataset on Kaggle prompting, helping and motivating people to do better. Of course the Titanic survival outcomes are well-known and hence in recent times, one has seen multiple scores of '100%' without accompanying details. These along with some semi-complete models (with pre-tuned parameters) giving ridiculously high scores can be ignored as there were several 'unknown' factors at play that night on the Titanic (one simple example is - few people were 'doomed' to die but survived simply because when the ship sank, they happened to be near a lifeboat that had some space & some kind souls). Also as it is a beginners competition, many get excited on seeing scores of 90%+ on the training dataset and blog about it only to then get disappointed when they apply their model to the actual test data. But we will not worry about scores at all. The whole intent is to use this as a learning experience (though you will be pleasantly surprised to know the final score we end up with). We will SOLELY use the Kaggle dataset to come to our conclusions and will NOT refer to any external source of data. Some Kagglers do take into account all public information available other than survivorship (for e.g. which all cabins on the Titanic were located closer to the lifeboats) while designing their models but we will not do this. 
# 
# *"Talk is cheap. Show me the code"* - Linus Travolds
# 
# Couldn't agree more, Sir! Here we go.. 
# 
# but cannot resist adding one additional line to your immortal quote. Considering the plethora of modern day free software libraries available:
# 
# *"Talk is Cheap, Code has also become cheap. Show me the (engineered) Data"*

# In[1]:


import pandas as pd
##pandas is the library that contains lot of in-built functions that help us analyze the data

##read_csv(file) is one such in-built function. We load the data in 1 line. 
test_data = pd.read_csv ('/kaggle/input/titanic/test.csv')
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

##test and train_data CSV files are loaded into Pandas 'dataframe' objects. Dataframes are 
##nothing but a 2 dimensional representation of the data containing rows and cols. Let us 
##start by having a very high level look at the data. The dataframe.info() function helps us.
train_data.info()


# In[2]:


##Aah. You can see the ouput cell where the o/p of this function is printed. There are 12 
##columns for each of the 891 passengers. One of them is the 'Survived' flag. Our ML 
##algorithms are supposed to 'learn' from this 'train_data' and 
##then predict the 'Survived' col for the 'test_data' which does NOT contain this col.

##Notice that 3 of these 11 columns in train_data have missing data for - Age, Cabin & 
##Embarked. We obviously cannot drop these 3 columns entirely as they could contain 
##useful information which may help us determine if the passenger survives or not. 
##We also can't drop rows containing these missing columns as they could also contain 
##valuable info.

##Let us also look at the test data. Before that, we define a global var to print blanklines
global blankline
blankline='\n**************************************************************************\n'
print (blankline)
test_data.info()


# In[3]:


##So we see that test data has 418 rows. It does NOT contain the Survived column. We have to 
##predict this after we 'learn' from the train_data. It has the remaining 11 columns and notice 
##that here too Age, Cabin and Fare have missing data. We now set this test data aside. We dont 
##want to mix test & training data. We focus all attention to the training data!

##Let us look at the actual data now..few of the rows. Let's use the dataframe.head() function
train_data.head()


# Fare & Ticket number seem to be interesting columns and will need some exploration. If fare is low it may be possible (though not appropriate) that the chances of survival could be low. Low fare also could mean low Pclass. There are 3 Pclass indicating whether passenger is travelling 1st, 2nd or 3rd class. Common sense (partially influenced by Hollywood) dictates that  Pclass=3 has higher risk of mortality compared to Pclass=2 who have higher risk of mortality compared to Pclass=1 
# 
# Cabin seems to have lots of NaNs (missing data). Not sure if this is really going to be useful.
# 
# The NAME has the title which can be used to derive the Sex column and can also help in guessing missing ages. **Master** Title is used for male child though **Miss** is used for both girl child as well as grownups. Since there are so many missing ages, this column can definitely help in 'guessing' some of the missing ages. Of course we can 'feed' this data (with missing ages) as-is to the ML algorithm but it will not perform so good in that case. If we clean up the data a bit by filling in good estimates of missing values, it will perform so much better. Of course we could use one more ML model just to guess the missing ages as so many kernels on Kaggle have done and then feed it to the main ML which predicts the survival. At this stage I dont want to comment without checking the data but I feel Age (in terms of its numerical value) is not a very strong indicator of survival. However the fact whether the person is a child or a middle aged person or a senior citizen could play an important role. Since we dont want to use a electric hacksaw where a small chisel more than capably does the job, we will use simple logic to fill the missing age data and call upon our ML model friends only when perdicting the survival.
# 
# The surname could help us group the individuals into families. So the NAME column seems very promising. Initially I thought it could be completely discarded as NAME logically would not be linked to Survival. But it contains so much useful information and looks like it will prove to be very imp! Name also contains the maiden name of the female within brackets 
# 
# PassengerID seems to be of no special consequence as it seems to have been added by Kaggle (I initially thought it may help determine the cabin arrangement in some way)
# 
# Embarked indicates the port of embarkation..Ideally should not have any relevance to survival but who knows?
# 
# Sibsp  - Indicates the total number of Siblings and spouses on board for that passenger
# 
# Parch - Indicates the total number of parents and children on on board
# 
# Which all of these columns do you think would be useful for predicting Survival?
# Let us explore!!

# In[4]:


print(train_data.describe())
##Only numeric columns are shown in describe() function. You can see in the o/p cell that 
##min, max, std deviation etc get printed.
##Mean age is 30. One crude way to fill missing ages is to make them all the age=30. But 
##let us see if there is a better way.
##The mean of the SURVIVED col shows 38% survival rate. Roughly 1 in 3 survive the Titanic
##Fare=0 needs investigation


# In[5]:


##Let us get a bit more details on missing data
print(blankline)
print(round (train_data.isnull().sum().sort_values(ascending=False)/len(train_data)*100,1))
##The above command gives more details on the missing data. The command is explained in the
##next cell but let us analyze the o/p of this command first.
##The o/p shows what we already know but gives a better idea of the situation. With 77% 
##missing data the Cabin info seems pretty useless. Roughly 1 in 5 ages are missing and 
##we will have to guess the age..The Embarked needs to be fixed and we also have 1 fare 
##missing from the test data (as seen previously). 

##Apart from AGE and Cabin, rest of missing data is hardly couple of rows and we should not 
##pay too much time deducing how to fill them. Embarked can be the mode (remember mean, 
##medium, mode). Fare can be the mean fare.  
##Filling the missing Age values would be the key here. Should it be the mean, median or mode?
##Let us take a look at the data and decide.


# *print(round (train_data.isnull().sum().sort_values(ascending=False)/len(train_data)*100,1))
# ****
# Let us analyze this line step by step. It packs a punch and is typical of Pandas..Does a lot in 1 line of code
# 
# *train_data.isnull()* returns true or false for each row & each column of the dataframe telling whether it is null or not. Try to print this. The o/p of this command is also a ****dataframe**** as you can see.
# 
# *.sum()* acts on a ****dataframe**** object and returns 1-dimensional array called a '**Series**' object. As you can guess by the name, the '*sum*' function adds up all the values (across rows) for each column. Since True=1 and False=0, this will return us the count of all 'true's for each column. Print *train_data.isnull().sum()* and see for yourself. As we discussed, the o/p is a '**Series**' object and returns the count of all nulls for each column..  
# 
# Dividing that by the length will give you the %age of missing values. Note that each item of the series object gets divided by the length. You **dont** have to write a FOR loop for this.
# 
# Sort and round before printing this.
# 
# Print each command incrementally to break up this statement and see how it works. It is very important to understand what 'object' is returned at each intermediate stage. This is because the set of functions available for a '**Series**' object maybe different from the set of functions available for a '**dataframe**' object and you could hopelessely go around in circles trying to analyze why a function works in one case but throws some error in another case.

# In[6]:


print(train_data.groupby('Pclass').count())
##The groupby(col) literally implies grouping the rows of the dataset based 
##on unique values of the col. Here we have 3 unique values of Pclass so 
##there are 3 groups of rows created. Once the groups are created, we can 
##'apply' a function to it. That function acts on the groups. For e.g. we 
##can calculate the 'count' of each group or the mean etc. Let us use 'count


# In[7]:


##We see there are 216, 184 & 491 1st, 2nd and 3rd class passengers respectively.
##Note that some columns show lesser number because of missing data. The 
##'PassengerID' column does not have any missing data and we can assume whatever 
##count is mentioned there will be the actual counts. We can ignore the counts in
##other columns. So to summarize we have 216 passengers travelling in Pclass 1, 
##184 in Pclass2 and 491 in Pclass 3 or the economy class

##Now let us focus specifically on the 'Survived' column. If we do a 'mean' of 
##the survived column (after grouping), we get the percentage of how many survived 
##in each group. Remember if person survives, then Survived col=1 else it is 0.
print(blankline, train_data [['Pclass','Survived']].groupby('Pclass').mean())

##Notice above how we give only the columns we are interested in as input to the 
##groupby function. This avoids clutter. Also notice the double braces [[ & ]]. 
##It is imp to understand what is happening here. ['Pclass','Survived'] is a Python
##list. This is passed to the external square brackets [] which is the indexer 
##operation of the Pandas dataframe object which returns a dataframe based on 
##the list passed. So train_data [['Pclass','Survived']] therefore returns a 
##dataframe object with just 2 columns. Try printing it & see.

##Note: train_data [['Pclass']] returns a "dataframe" object with just 1 column
##train_data ['Pclass'] returns a "series" object with just 1 column. You cant groupby 
##a "series" object. Again, It is very important to understand the 'type' of objects 
##being returned by each function else you could spend hours debugging the code. 
##Unfortunately in few cases the error message is not very friendly. If you dont
##believe me try printing train_data ['Pclass','Survived'].groupby('Pclass').mean()


# In[8]:


##Let us put it in proper %age format
print(blankline, round(train_data [['Pclass','Survived']].groupby(['Pclass']).mean()*100,1))
##Do check the o/p cell
##So 2nd class passengers had twice the survival rate of 3rd class and 1st class passengers 
##had even better rates. Not a very pleasant discovery but it was on expected lines, I guess. 


# In[9]:


##Now let us turn to Sex. Does it make a difference?
print(blankline,round(train_data [['Sex','Survived']].groupby(['Sex']).mean()*100,1))
##74% of females survived and only 19% males survived. The overall survival rate was 38.3%
##as seen earlier. So it is hoplessely skewed in favour of females


# In[10]:


##Groupby is the one Panda command you need to know a little about because it is so useful
##Let us groupby with multiple columns
print(round(train_data [['Sex', 'Pclass','Survived']].\
                       groupby(['Pclass', 'Sex']).mean()*100,1))
##This is really useful data. Basically almost all Pclass 1 females survive and so do most 
##of Pclass 2. Pclass 3 female survival is 50% and almost all (86%) Pclass 3 males unfortunately
##do not survive. Notice how we passed only 3 cols to the groupby function - Sex, Pclass
##and Survived. We could also pass the whole dataframe as is, but this creates lot of clutter


# In[11]:


##Lastly let us see if Embarked has any real relevant significance to survival
print(blankline, round(train_data [['Embarked', 'Sex', 'Pclass','Survived']].\
                       groupby(['Embarked', 'Pclass', 'Sex']).mean()*100,1))


# In[12]:


##Ideally the percentages should be same as seen before but you can see some discrepancies begin
##to emerge. For e.g we see that for Embarked = Q, 100% of males in Pclass 1 and Pclass 2 expire 
##This goes against the data we just saw earlier. So let us see what is happening?

print(blankline, train_data[(train_data.Pclass==1) & (train_data.Sex=='male') & (train_data.Embarked=='Q')])
##Here we just apply a filter to the train_data dataframe. It is like saying SELECT ALL ROWS WITH
##CONDITION. Here the CONDITION we want to explore is: Pclass 1 males who embarked on port Q


# In[13]:


##Aah! We see just 1 male passenger embarked into Pclass 1 at port Q and he unfortunately did not
##survive. We based our entire analysis of 100% death based on this single passenger and we could 
##have gone completely off track. A good idea is to also print the counts along with the %ages
##so we are not misguided by statistically insignificant data and jump to conclusions. 
print(blankline, train_data [['Embarked', 'Sex', 'Pclass','Survived']].groupby \
                       (['Embarked', 'Pclass', 'Sex']).agg(['count','mean']))
##We use the aggregate function. We pass the mean and count as list to the agg function which 
##does the rest. Let us now turn to GRAPHS which gives us a better pictorial view 


# In[14]:


##Rather than writing a lengthy line of code, we try 'evolve' the graphs as we go 
##so it is easy to understand for the beginners
import seaborn as sns
import matplotlib.pyplot as plt
##We  need visuals for our next question - How is age linked to survival
##Let us use what is called a density plot & see the Age distribution
##We will need to import seaborn library and use its functions to plot the graph
##We also need Matplotlib which is a plotting lib for the Python programming language

sns.kdeplot(train_data['Age'], color="green")
##train_data['Age'] is just the AGE column of the Titanic data set. This is passed 
##as i/p. You can try printing train_data['Age'] independently and see for yourself.
##Now let us turn our attention to the kdeplot. Seems like while majority of folks 
##on the Titanic are in the 20-40 age range, there are lot of 10-20 range as well


# In[15]:


##I think a histogram would serve us better. Group into 10 bins
plt.figure()   ##Tell Python this is a new fig else it will overwrite the old one
print(train_data['Age'].plot(kind='hist',bins=10))
##Much clearer now. 20-30 has max folks. There is also a sizeable 0-20 & a 30-45.


# In[16]:


##The density plot has its uses though. For e.g. if I want to compare the age 
##distribution of male/female, here is what I would do
plt.figure()
sns.kdeplot(train_data[train_data.Sex=='female']['Age'], color="green")
sns.kdeplot(train_data[train_data.Sex=='male']['Age'], color="red", shade=True)
##We discussed what train_data[condition] returns. So train_data[train_data.Sex=='male']
##returns us a new dataframe containing all the males in the original dataframe. What is 
##this ['Age'] column next to that? Well, we are not interested in the remaining 11 columns
##We are just interested in AGE col. To retain only the columns we are interested in, we 
##just do a dataframe[col]. If we need more than one column, just pass the column names as 
##a list..e.g dataframe[[col1, col2]]. You can printing these individually to understand 
##in detail. For e.g. print train_data[train_data.Sex=='female'] first, then print 
##train_data[train_data.Sex=='female']['Age']
##Graph shows while the distribution is more or less equal, there are more females 
##around the 10 year age group


# In[17]:


##How does survival depend on age? Let us plot violin-plot which is best suited for this
sns.violinplot(x='Survived', y='Age', data=train_data, palette={0: "r", 1: "g"});
##There is an abundance of information here. The white dot you see is the median. The 
##thick black line at the center is the first and third quartile. Violin plots show the
##distribution across the range (of age in this case) & give a visual indication of
##Outliers. Quick comparision shows that between 0-10 age survival rate is higher (green
##is fatter than red in this age range). After age 15 or so, expiry (red) just fattens out
##rapidly peaking at around 22 years of age. Green too fattens but not as rapidly as Red. 
##In this age group far too many people expire (compared to survive). This trend pretty 
##much continues. After about 65 or so, almost all expire (green is hardly a line...wheras 
##Red is fatter). Note that the shape is mirrored across 2 sides of the plot. Using the 
##Split=true option we can retain only 1 side & save some real estate


# In[18]:


##We can squeeze in one more parameter using hue. hue needs to be a binary value parameter.
##Since Survived is a binary, let us plot it as well into the above graph
plt.figure() 
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=train_data, palette={0: "r", 1: "g"});
##Too messy. Let us use the 'split' option which gives 3 figs instead of 6 (without losing info)


# In[19]:


plt.figure() 
sns.violinplot(x='Pclass', y='Age', hue='Survived', split=True ,data=train_data, palette={0: "r", 1: "g"});
##Pclas 2 and 3 have a very sharp peak of expiry near the 20-35 age range
##Class 1 has far more survivors across all age levels except maybe after 50 or so. After 60 
##there is a sharp jump in expiry. This is not as sharp in Pclass 2 or 3. This is a curious 
##phenomenon. Let us investigate a bit


# In[20]:


print('Pclass 1 survivors above Age 60:', round(len(train_data[(train_data['Pclass']==1) & \
    (train_data['Age']>59) & (train_data['Survived']==True)])/len(train_data[(train_data\
    ['Pclass']==1) & (train_data['Age']>59)])*100,1), '%')
print('Pclass 2 survivors above Age 60:', round(len(train_data[(train_data['Pclass']==2) & \
    (train_data['Age']>59) & (train_data['Survived']==True)])/len(train_data[(train_data \
    ['Pclass']==2) & (train_data['Age']>59)])*100,1), '%')
print('Pclass 3 survivors above Age 60:', round(len(train_data[(train_data['Pclass']==3) & \
    (train_data['Age']>59) & (train_data['Survived']==True)])/len(train_data[(train_data \
    ['Pclass']==3) & (train_data['Age']>59)])*100,1), '%')

print('Pclass1 survivors between 20-30 Age:',round(len(train_data[(train_data['Pclass']==1) \
    &(train_data['Age']>19) & (train_data['Age']<31) & (train_data['Survived']==True)])/len( \
    train_data[(train_data['Pclass']==1) & (train_data['Age']>19) \
    & (train_data['Age']<31)])*100,1),'%')
print('Pclass2 survivors between 20-30 Age:',round(len(train_data[(train_data['Pclass']==2) \
    &(train_data['Age']>19) & (train_data['Age']<31) &(train_data['Survived']==True)])/len( \
    train_data[(train_data['Pclass']==2)&(train_data['Age']>19) \
    &(train_data['Age']<31)])*100,1),'%')
print('Pclass3 survivors between 20-30 Age:',round(len(train_data[(train_data['Pclass']==3) \
    &(train_data['Age']>19) & (train_data['Age']<31) &(train_data['Survived']==True)])/len( \
    train_data[(train_data['Pclass']==3) & (train_data['Age']>19) \
    &(train_data['Age']<31)])*100,1),'%')
##The syntax is straightforward. We are just putting a bunch of conditions and calculating 
##the length of the dataframe returned each time. The denominator condition does not have 
##survived col so we get the %age. Try printing the below line:
##train_data[(train_data['Pclass']==1)&(train_data['Age']>59)&(train_data['Survived']==True)]
##Then print len of above. It gives the count. The below kind of commands
##Dataframe[(condition 1) & (condition 2) & (condition 3)] will be used over and over again 
##in 90% of the code. So familiarize yourself on the same before stepping forward.


# Lo and behold, Indeed in Pclass=1, there is **2.5 times more survival chance** if you are between 20-30 as compared to 65. As the Pclass changes we can see how the survival gap narrows between the 2 age groups. Is this an act of altruism? Letting the younger males live? Why is this limited only to the first class? A quick analysis of data shows that Pclass 2 and Pclass 3 have statistically insignificant older males who survived. Perhapes this is skewing the data. Ok. But Pclass1 has 17 people greater than 60 which is not a insignificant number. Or is it that there were more males among the elderly and more females among the middleage group in Pclass 1..so that is the reason for the disparity. A quick series of commands reveal that only 2 males >60 out of 19 in pclass 1 survive as compared to 9 out of 19 for middle age group. Thus even after accounting for all parameters we see that this 14% survival versus a near 50% is a mystery. It is not that the middle age folks pushed around the older lot. In fact much of the evacuation was orderly except for the very last few unfortunate minutes. Most of the folks who exipred were able bodied middle aged males..which is why I talked about Darwin beng proved wrong that night as the first act of altruism. But looks like there is a twist to this theory as far as first class passengers are concerned. It does look like some of the older males voluntarily made way for the younger males..Could this be the 2nd act of altruism on the Titanic? The data is very small but not insignificant and it does point towards that direction.
# 
# So AGE definitely plays a factor in survival and this is not just limited to the kids getting a higher survival rate. There are other factors at play as well. It is imp to get the missing ages right and we should not be relying just on the mean to fill the same.
# 
# Another stunning data presents itself when we further add the sex condition to the filter and calculate the above formulas once again. We see that NONE of the 33 Pclass 2 males in the middle age group (20-30) age group survives (there is 1 survivor in the elderly group but that is stastically insignificant). Ideally, there should be around 16% survival chance for Pclass 2 Males as per our earlier observation. In fact it is so curious that I extend the age ranges and found that 4 out of 77 males between 20-60 age group survived. Thisis just 5%, the survival rate for Pclass 2 males should be 16%. This could mean either of 2 things: Eitherin Pclass 2 more kids survived than normal or there were more %age of kids in Pclass 2 males. This means that adult males in Pclass 2 had actually less survival chance compared to Pclass 3 and kind of turns popular theories around the head. Let us check this...
# 

# In[21]:


print('Pclass 2 adult male survivors:',round(len(train_data[(train_data['Pclass']==2) & \
        (train_data['Age']>19) & (train_data['Sex']=='male') & (train_data['Survived'] \
        == True)])/len(train_data[(train_data['Pclass']==2) & (train_data['Age']>19) & \
        (train_data['Sex']=='male')])*100,1),'%')

print('Pclass 3 adult male survivors:',round(len(train_data[(train_data['Pclass']==3) & \
        (train_data['Age']>19) & (train_data['Sex']=='male') & (train_data['Survived'] \
        == True)])/len(train_data[(train_data['Pclass']==3) & (train_data['Age']>19) & \
        (train_data['Sex']=='male')])*100,1),'%')


# Unbelievably fascinating and a wonder this has not been documented so far. Let me again paraphrase it - 'Adult males travelling in Pclass 3 had MORE THAN TWICE the chances of survival as compared to adult males travelling in Pclass 2' or to put it bluntly if you were an adult male, you had double the chances of survival if you got yourself booked in 3rd class on the Titanic rather than 2nd class! This does turn popular theories on the head.. The data was lying all out there waitingfor someone to unravel it. But more importantly how many more insights are still there - waiting to be unravelled. We have discovered 3 acts of altruism so far on the Titanic. The first one was well known. The second is not very conclusive due to lack of enough data but is not statistically insignificant. The third was a big surprise to me. A huge chunk of Pclass 2 males voluntarily gave up their lives so that more females and children (from all classes) survive. Are there any more acts left? Let us see.

# In[22]:


##Let us explore the Embarked column now. We will try a new kind of plot
fg = sns.FacetGrid(train_data, row='Embarked')
##Above statement creates one row of graphs for each unique value of 'Embarked'. We have 
##to specify what data we need. This is done below
fg.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')
##x-axis is the Pclass, y-axis is the survival %age. The hue (color) category is the Sex


# In[23]:


##Figs are small. Increase the ASPECT. Also add legend so we know what colors stand for what?
plt.figure()
fg = sns.FacetGrid(train_data, row='Embarked', aspect=2)
fg.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')
fg.add_legend()


# In[24]:


##Did you notice something ridiculously strange happening on Embarked=C port? Those passenges
##who embarked on port C have totally reversed the odds for survival as far as Sex is concerned. 
##This is FALSE info & can quickly be verified by a query to gather the %age of survivors grouped 
##by Sex for Port C embarkation. Obviously the colors for Male & Female have got mixed up in the 
##middle row. If your warnings are suppressed you may not even notice it.. Luckily the warning 
##is clear. Please use the 'order' parameter else you may get incorrect ordering
plt.figure()
fg = sns.FacetGrid(train_data, row='Embarked', aspect=2)
fg.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', \
       hue_order=['female', 'male'], order=[1,2,3], palette={'female': "r", 'male': "g"})
fg.add_legend()
##Better! One very popular blog post (& half-a-dozen other copied copies) make this erroneous 
##observation and goes on to other topics as if it were perfectly normal data behaviour. There 
##is no point in just visualizing the data and doing nothing about it. Please dig into it 
##especially if you see patterns or anamolies which need further investigation. Survival of 
##more Males compared to Females definitely needs an investigationand we have just unravelled 
##the mystery. This observation is erroneous!


# In[25]:


##Also Note some interesting variations between the port of embarkation. Ideally 
##one would expect the graphs to be similar to each other. In embark=S, females in pClass3 
##do extremely poorly against the other 2 classes. The difference is not so glaring in the
##other 2 ports. In Embarked=C, pClass=2 females have higher chance of survival than pclass=1.
##Embarked=Q has extremely high fatility rates for Males. For some strange reason pClass=3 
##have higher chances of survival in this port. This is so curious that we need to reverify it
print('Pclass 1 Male survivors % for Port Q embarkation :', round(len(train_data[(train_data \
    ['Pclass']==1) & (train_data['Embarked']=='Q') & (train_data['Sex']=='male')&(train_data \
    ['Survived']==True)])/len(train_data [(train_data['Pclass']==1) & (train_data['Sex']==\
    'male') & (train_data['Embarked']=='Q')])*100,1), '%')

print('Pclass 3 Male survivors % for Port Q embarkation :', round(len(train_data[(train_data \
    ['Pclass']==3) & (train_data['Embarked']=='Q') & (train_data['Sex']=='male')&(train_data \
    ['Survived']==True)])/len(train_data [(train_data['Pclass']==3) & (train_data['Sex']== \
    'male') & (train_data['Embarked']=='Q')])*100,1), '%')
##Aah - While it is true, looks like the data is statistically insignificant. Let us see...


# In[26]:


print('Pclass 1 Males for Port Q embarkation :',len(train_data[(train_data['Pclass']==1) \
    & (train_data ['Embarked']=='Q') & (train_data['Sex']=='male') & (train_data['Survived']\
    ==True)]))
print('Pclass 1 Male survivors for Port Q embarkation:',len(train_data[(train_data['Pclass']\
    ==1) & (train_data['Embarked']=='Q') & (train_data['Sex']=='male')]))
##There you go. 1 passenger for pClass1 got on at S Port and we made an entire hypothesis 
##based on the fate of this one passenger. This is a very important lesson & need to be 
##careful as far as %ages and graphs are concerned simply because they may be statistically 
##insignificant. However considering the major variations across the 3 rows it is possible
##that port of embarkation is corelated to the survival along with pClass, Sex, age
##(above 3 were obvious though). Maybe these folks were given cabins in certain areas which
##were closer to lifeboats? So we cant drop this column


# In[27]:


##let us check how the sex of the person could have made a difference to survival
sns.violinplot(x='Sex', y='Age', hue='Survived', split=True ,data=train_data, palette={0: "r", 1: "g"})
##If you are male your chance of survival is good upto 15 years of age else it is 
##not good. In case of female there is a decent chance of survival across all ages

##Let us merge Pclass and Sex to create a new col. I was trying to search for a graph 
##which could point 4 pieces of data when I thought of combining the sex & Pclass to 
##make it 1 col. We can now leverage Violinplots for 3 cols and get a nice view on what 
##we want to see.

train_data['PclassSex'] = train_data['Pclass'].astype(str) + train_data['Sex'] 
##In one line above, add a new column to the train_data dataframe with the data we want. 
##We need to convert Pclass which was Integer to String else it throws error. Let us 
##now plot this new col

plt.figure(figsize=(15,8))   ##Increase size of the figure
##Add cut=0 else it shows a -ve distribution. density is defined over -infinity 
##to +infinity. Just ignore the below 0 values. Also specify the order
sns.violinplot(x='PclassSex', y='Age', hue='Survived', split=True,data=train_data, cut=0, \
    palette={0: "r", 1: "g"}, order=['1male','2male','3male', '1female', '2female', '3female'])


# Aah now the curious observation we made earlier about Pclass 2 males can be clearly seen. See how narrow the green plot is from age 16 onwards? Survival rate is far more worse than Pclass 3 if you can compare. Let us jot what all we can see from this graph apart from the above.
# * All children upto age 18 survive in Pclass 1 male. All children up to age 16 survive in Pclass 2. 
# * Above age 45, almost all Pclass 3 males expire. Between 16-40 Pclass3Males have a decent survival chance, way above Pclass2Males and almost comparable to Pclass1Males in that age group
# * Pclass3female kids have lesser chance of survival than Pclass3Malekids
# * Pclass1female kids show higher mortality than all other class. As 97% Pclass1females survived, this may be a statistically insignificant data. 
# 
# We will revirify all this once again later during feature engg. Another observation is that during feature engg, we may want to combine Pclass and Sex..Together they give a lot more info rtather than separately as we just saw. The ML model may also appreciate this 'small' decision of yours vastly :)

# In[28]:


##Let us tackle Fare...This is the last column that needs some analysis before we move 
##onto the next stage
print(train_data['Fare'].describe())
##We need to investigate the 0's. Also let us do density plot

sns.kdeplot(train_data['Fare'][train_data.Survived == 1], color="green", shade=True)
sns.kdeplot(train_data['Fare'][train_data.Survived == 0], color="red")
plt.legend(['Survived', 'Not Survived'])
##Hope by now the command train_data['Fare'][train_data.Survived == 1] is clear. Try 
##printing individually if any confusion the below command:
##train_data, train_data['Fare'] and train_data['Fare'][condition]

##There is a heavy tail. limit x axis to zoom in only on relevant information
plt.xlim(-10,250)
plt.show()
##Somewhere between 0-20 range, there is a huge spike in expiry. Post that survival 
##chances improve significantly. Between 0-20, it looks like chances of expiry is 3 times
##chances of survival. Assuming all these are pclass=3. The point where green touches red line 
##is where I believe pclas changes from 3 to 2.


# In[29]:


##Let us get the fare=0 cases
print(train_data[(train_data['Fare']==0)])
##'Curiouser and curiouser' as Alice would say. None of them have siblings or children. 
##How can their fare be 0. All of them are middle aged males. All have embarked at one place. 
##Most likely this is the Cabin crew. 

print(train_data[(train_data['Fare']==0)].groupby('Pclass').agg('count'))
##This is split across pClasses - 5, 6 & 4 in each class

##Let us look at their ticketIDs to see if this provides a clue
print(train_data[(train_data['Fare']==0)].groupby(['Pclass', 'Ticket']).agg('count'))
##Ignoring the last number of the ticket, we can group them into 3 main categories. One for each 
##class. Looks like LINE are crew with pclass=3
##Others with Fare=0 are from pClass 2(23985x series) and pClass 1 (11205X series)
##unfortunately all of them go down. Also there are nearly 8 null values for Age. We have to fill 
##them all with the avg age of this group. This seems to be a neat find and will strengthen our 
##age prediction. Last but not the least we have to correct the fare...else this will lead the ML 
##algorithm on a diff path. So this fare=0 is to be treated as missing data! A few Kagglers have 
##done that but not too many.


# In[30]:


##Let us plot a histogram to see if we can deduce anything else
plt.hist([train_data[train_data['Survived'] == 1]['Fare'], train_data[train_data['Survived'] == 0]['Fare']],
stacked=True, color = ['g','r'], bins = 50, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend();
##As expected higher the fare the higher the survival chance. But this relationship is better 
##captured in the pClass. So this is like a duplicate info. Is it really required?

##Any other trends? Let us analyze the lowest fares and see their chances for survival
print(len(train_data[(train_data['Fare']<7.1) & (train_data['Fare']>0)]), blankline)
print(train_data[(train_data['Fare']<7.1) & (train_data['Fare']>0)].agg('mean'), blankline)
print(train_data[(train_data['Pclass']==3)&(train_data['Sex']=='male')].agg('mean'),blankline)
##These 23 folks seem to have abnormally abysmal chances of survival around 4%
##Even if they are all Male and Pclass=3 the survival percentage should be 13.5%. Why such a 
##glaring diff? There could be various reasons. Possibly the lowest fare tickets had cabins 
##furthest away from the lifeboats even amongst Pclass 3). This %age is statistically significant 
##and we can add a separate feature classifying these as High Risk.

##A lot of these low cost tickets seem to be single men
##When does first child make his presence..around 7.2 fare..
##Let us dig a bit more here
print(train_data[(train_data['Fare']<9)&(train_data['Age']<14)])
print(len(train_data[(train_data['Fare']<9)]))
print(len(train_data[(train_data['Age']<14)]))
##Out of 311 folks with <9 ticket, 2 are children...which means remaining 69 children on the 
##ship are in remaining 580. This is also statistically very very significant. This will be 
##very useful to fill the missing age column. Amongst the 311 folks, those wth missing ages are
##almost all adults

print(len(train_data[(train_data['Fare']<9)&(train_data['Age']>49)]))
##Not too many old people also. Bulk of the 311 folks are middle ages. How many nan's?

print(len(train_data[(train_data['Fare']<9)&(train_data['Age'].isnull())]))
##Wow. A whopping 99 of the 177 missing age group come in this range (fare<9). We have seen 
##above that there are minimal children and minimal old folks in fare<9. Based on Pclass
##(which should mostly be 3) and sex, we should calculate the avg age of this particular 
##group(fare<9) and fill in the missing 99 values for this group. This will be better fitting
##data than calculating the missing 177 age values based on any generic logic. Well, this fare 
##column is turning out to be the most interesting one and giving us a lot of information


# Let us also see if calculating the actual fare makes any sense. If Ticket num is same, then fare is the sum of all tickets having the same ticket number. So divide fare by the number of (same)tickets to get actual fare. Note that here we HAVE to dip into "test" data frame to get this data. Normally this is frowned upon as this results in a leak from test to train but for below purpose it is OK. Let us add a new column called PeopleInTicket which will list count of persons per each ticket. But before that let us combined test & train data and create a 'combined' dataframe. We use a 1 line Append command.
# 
# Next, we will use the value_counts() command which is like the swiss knife in the Pandas arsenal. value_counts() returns a "**series**" object containing the frequency of the column in question. You can print it and see. So try printing combined['Ticket'].value_counts()
# 
# The command we will use is:
# train_data['PeopleInTicket']=train_data['Ticket'].map(combined['Ticket'].value_counts())
# 
# We will pass the series object o/p returned by value_counts() to a map() command which acts on a "series" object X using a "series" object Y as input. Here X is Combined['Ticket'] and Y is combined['Ticket'].value_counts(). Print both invidivually and see if you have any confusion. map() maps the values from two series that have a common column. In this case Ticket is the common column. The net effect of above command is to add a new column in train_data which holds the number of people in each ticket. In other words, we can use map to perform a lookup on the key and return the corresponding 'mapped' value. Now calculating Fare per person is a simple job..

# In[31]:


combined=train_data.append(test_data)
train_data['PeopleInTicket']=train_data['Ticket'].map(combined['Ticket'].value_counts())
train_data['FarePerPerson']=train_data['Fare']/train_data['PeopleInTicket']

##For curiosity's sake also added a new column family count. Let us see if this tallies 
##with the PeopleInTicket
train_data['FamilyCount']=train_data['Parch']+train_data['SibSp']+1

pd.set_option("display.max_rows", None, "display.max_columns", None)
display(train_data.head())
##Wow. There we go. Let us spend a moment to analyze these new columns 


# In[32]:


print(len(train_data[train_data['FamilyCount'] != train_data['PeopleInTicket']]))
##There are 195 rows where FamilyCount does not match with PeopleInTicket. For 
##remaining 600 odd rows these match perfectly. We will have to resolve the 
##195 rows at some point of time or other

print(len(train_data[(train_data['FarePerPerson']<7.1) & (train_data['FarePerPerson']>0) \
    & (train_data['Survived']==0)]))
print(len(train_data[(train_data['FarePerPerson']<7.1) & (train_data['FarePerPerson']>0) \
    & (train_data['Survived']==1)]))
##What is happening here? Earlier we saw that the lowest 26 fares (<7.1) were associated with 
##nearly 95% expiry Why has that changed drastically now with same calculation for 
##"FarePerPerson"? Only one reason could be that somehow groups of travellers managed to survive
##better as opposed to solo travellers. This is a very very imp observation and we need to take
##this into account during feature engg. Groups of travellers have better chance of survival 
##than solo. Another reason could be that most solo travellers were men who anyway had higher 
##chances of expiry. Let us analyze more..but before that a small deviation..


# In[33]:


##Just for fun, I tried analyzing fares for Women to see if they differed from men. This was 
##merely a distraction and laying out (possible) proof of gender discrimination as far as fees 
##were concerned was never on my agenda but here were the observations:
print('Avg fare for solo man in Pclass 1: ', train_data[(train_data.Pclass==1) & (train_data\
    .FamilyCount==1) & (train_data.Sex=='male')]['FarePerPerson'].mean()) 
print('Avg fare for solo woman in Pclass 1: ', train_data[(train_data.Pclass==1) & (train_data\
    .FamilyCount==1) & (train_data.Sex=='female')]['FarePerPerson'].mean()) 
print('Avg fare for solo man in Pclass 2: ', train_data[(train_data.Pclass==2) & (train_data\
    .FamilyCount==1) & (train_data.Sex=='male')]['FarePerPerson'].mean()) 
print('Avg fare for solo woman in Pclass 2: ', train_data[(train_data.Pclass==2) & (train_data\
    .FamilyCount==1) & (train_data.Sex=='female')]['FarePerPerson'].mean()) 
print('Avg fare for solo man in Pclass 3: ', train_data[(train_data.Pclass==3) & (train_data\
    .FamilyCount==1) & (train_data.Sex=='male')]['FarePerPerson'].mean()) 
print('Avg fare for solo woman in Pclass 3: ', train_data[(train_data.Pclass==3) & (train_data\
    .FamilyCount==1) & (train_data.Sex=='female')]['FarePerPerson'].mean()) 

##On an avg they are higher by about 20% in Pclass 1, 8% in Pclass 2 and 4% in Pclass 3. I rest 
##my case. Maybe the men bargained more to get better deals? In any case, the fares are clearly
##more favorably disposed to males but they got more than what they bargained for at the end. 
##On a separate note, I sometimes feel we take the chivalry of these gentlemen for granted and 
##assume that the same would repeat under any circumstances but this was clearly not the case 
##in a few other parallel tragedies (woman and children had lower mortality than males in those
##incidents). So there was definitely something special about these gentlemen on the Titanic..
##or maybe it was the crew and the leadership team that made it happen. We can only speculate...


# In[34]:


##Let us take the max of FamilyCount & PeopleInTicket and create a new column called GroupSize
train_data['GroupSize'] = train_data[['FamilyCount','PeopleInTicket']].max(axis=1)
##We are taking the max of 2 columns but what is this axis=1 business? In Pandas, we have series
##object (1 dimensional) and we have dataframes (2D or a series of series objects). axis=0 
##represents rows and axis=1 represents columns. When we use the max function, we have to specify
##the axis else by default axis=0 is considered and you will get the max Familycount and 
##PeopleInTicket across all rows of the dataframe.Try printing:
##train_data[['FamilyCount','PeopleInTicket']].max() and see for yourself

##Now let us plot the new columns we have got. We will use a barplot this time and more 
##specifically a barplot that counts frequency of the column...called countplot
plt.figure(figsize=(16, 6))
sns.countplot(x='FamilyCount', hue='Survived', data=train_data)


# It appears as if solo travellers are at high risk. From 2-4 groups though the survival rate is high and then 5 onwards the survival rate dips completely. This observation is made by almost all Kaggle posts and indeed some of the very tall leaders have also made this observation in their analysis. Intuitively it may seem true - If you are travelling in a larger group, you would perhapes be made aware faster of the impeneding tragedy because of the sheer size of the group. On the other hand solo travellers, may have been fast asleep in their bunks and may have gotten the news late. As group size increases, perhapes it became more and more difficult to collect everyone & make a dash for safety and so maybe with larger groupsize the trend turns negative and survivals dip.
# 
# My initial conjecture was while this could be true, I felt that this could be a case of corelation instead of causation. Causation explicitly applies to cases where a factor X causes outcome B. On the other hand, correlation is simply a relationship. B happens when X happens but it does not necessarily mean that X causes B. In other words, correlation does not imply causation. What I mean by this is that while I felt that the above observation is true (the probability of survival is less if group sizes >4) but it does not mean that it is 'caused' by the same. In other words having larger groups did not 'cause' the survival rate to plunge. What I felt was actually happening is that large group sizes were more prevelant in Pclass 3 who anyway had a lower chance of survival based on the class of travel. In other words Pclass=3 was the causation of high mortality and groupsize>4 was just corelated to higher mortality. However I did feel that being solo DID increase mortality rate. Let us quickly see if all this is true.

# In[35]:


print('Between 2-4 familycount in Pclass 1,2: ', len(train_data[(train_data.FamilyCount \
        .between(2, 4)) & (train_data.Pclass.between(1,2))]))
print('Between 2-4 familycount in Pclass 3: ',len(train_data[(train_data.FamilyCount \
        .between(2, 4)) & (train_data.Pclass==3)]))
print('>4 familycount in Pclass 1,2: ',len(train_data[(train_data.FamilyCount>4) & \
        (train_data.Pclass.between(1,2))]))
print('>4 familycount in Pclass 3: ',len(train_data[(train_data.FamilyCount>4) & \
        (train_data.Pclass==3)]))
##Yipee! The data is as clear as it can be. There are 8 large groups in Pclass 1,2 
##VERSUS 54 large groups in Pclass 3. Obviously the chances of survival for former are high. 
##It is a corelation not a causation!!


# In[36]:


##Now armed with this knowledge, let us do an Apple to Apple comparison. Let us exclude 
##solos and compare based on class. Let us first re-print the old graph for comparing
plt.figure(figsize=(16, 6))
sns.countplot(x='FamilyCount', hue='Survived', data=train_data[(train_data.FamilyCount > 1)])

plt.figure(figsize=(16, 6))
sns.countplot(x='FamilyCount',hue='Survived',data=train_data[(train_data.Pclass==3) \
        & (train_data.FamilyCount >1)])


# In[37]:


##Situation looks gloomy now & quite the opposite from earlier graph. So groupsize does not 
##seem to have a very major role to play in survival. Being solo or not does but beyond that
##the size of group does not seem to matter. It could unmecessarily give some false positives
##and one must think whether this col is relevant enough to be fed to the ML model
##Even the 'solo' feature is worth a discussion. The general concession is that solo 
##travellers have higher mortality rate. Let us check this
print('Mortality rate overall: ', round(len(train_data[(train_data.FamilyCount==1) & \
    (train_data.Survived!=1)]) / len(train_data[train_data.FamilyCount==1])*100), '%')

print('Mortality rate Male: ', round(len(train_data[(train_data.FamilyCount==1) & \
    (train_data.Survived!=1) & (train_data.Sex=='male')]) / len(train_data[(train_data\
    .FamilyCount==1)& (train_data.Sex=='male')])*100), '%')

print('Mortality rate Female: ', round(len(train_data[(train_data.FamilyCount==1) & \
    (train_data.Survived!=1) & (train_data.Sex=='female')]) / len(train_data[(\
    train_data.FamilyCount==1)& (train_data.Sex=='female')])*100), '%')

##Solo females buck the trend. But you may say this is not a fair comparision as females 
##anyway have better survival rate. Let us do a final graph comparing mortality for Pclass 
##3 females between solo travellers & non-solo travellers before moving on
plt.figure(figsize=(16, 6))
sns.countplot(x='FamilyCount',hue='Survived',data=train_data[(train_data.Sex=='female') \
    & (train_data.Pclass==3)])


# See mortality rate percentage for solo females in Pclass 3 (or any Pclass for that matter) is much lower that mortality rates percentage for non-solo females. So basically if you were a solo woman traveller, you had much much better chance of survival compared to women with groups or families. Go figure :) and mind you the data is not statistically insignificant. There were 60 women solo travellers in Pclass 3 compared to 84 non-solo woman travellers and it is here that perhapes we see the 4th and the greatest act of altruism seen aboard the the Titanic. While there can be other reasons for this, in my view this happened because the women here travelling with their husbands and male children(>12 years of age) did not possibly want to leave them even though they had a chance to do so!! Some individual acts of chivalry and altruism have been recognized on the Titanic, but this perhapes is the greatest of them all and it is a pity that this has not been called out so far.

# In[38]:


##We are left with analysis of Embarked and Cabin..In my view both dont have a 
##major role to play in survival. Let us create a new col called Cabin letter
train_data['CabinLetter'] = train_data['Cabin'].str[0]

##Let us analyze this a bit
print(train_data.groupby(['CabinLetter', 'Pclass', 'Sex'])['Survived'].agg(['count', 'mean']))
##A, B, C are all Pclass 1. D contains very few 2's. E contains very few 3's. F is mix of 2, 3
##and G is just 3. A has very few females. G is all females. For the rest there is a equal 
##distribution of males/females. With this in mind, the anamolies are:
##Cabin C has a slightly higher mortality rate than can be expected. This is minimal and 
##can be ignored. Cabin D & E seem to be bit lucky for males. The data is also strong. Let 
##us make a note of it but there is not much we can do because 75% of folks dont have Cabin 
##letters. At best we could drill down at ticket level & see if there are a bunch of tickets 
##that have higher survival rate in males..Maybe group dynamics are at play


# In[39]:


##Let us now explore the 'Embarked' columne
print(train_data.groupby('Embarked')['Survived'].agg(['count','mean', 'size']))
##168 that onboarded at Port C have a disproportionate survival of 55%. Maybe there was 
##less Pclass=3 in these? Let us verify
print(train_data.groupby(['Embarked', 'Pclass'])['Survived'].agg(['count','mean']))
##There you go - There was a disproportionate amount of PClass=1 who onboarded at port C...
##almost 50%. To give a comparision PClass=1 was around 25% across all ports in the 
##train_data. Port C has twice the number of PClass=1. Naturally survival rate is disproprtionate
##when viewed across all classes. Overall 'Embarked' column does not seem to make much of a difference

##Let us do a more detailed check by bringing gender also into it
print(train_data.groupby(['Embarked', 'Pclass', 'Sex'])['Survived'].agg(['count','mean']))
##Anamolies - Port C Pclass 3 -  have higher chances of survival
##Anamolies - Port Q for Pclass 3  - females has high survival rate & low for Males
##Anamolies - Port S for Pclass 3 - females survival seems to be much lower


# All these observbations are statistically significant and cant be explained by any reason other than maybe there were more children in those groups? Unfortunately there are too many age=nulls here. The only other reason I could assume for high survival among Port Q Pclass 3 females is that there were more percentage of solos there. A quick check shows this to be true 25 out of 33 females embarked at Q were solos and 19 of them survived (out of total survived of 24). For port S - 19 out of 58 non-solo females in Pclass 3 survive wheras 14 out of 30 solos survive. So the low %age of survival among Port S Pclass 3 females and high % of survival among Port Q Pclass 3 females can be explained by the %afe of solos amongst them. There could be other explanations but this seems to be a logical one. 
# 
# For males the data is bit inconcusive
# * 1 in 10 Pclass 3 Port Q males with families survive. 2 out of 29 without families survive. Total 3 out of 39
# * 5 out of 33 bachelors survive in Pclass 3 C port. 5 of 10 with families survive. Total 10 out of 43
# * 9 out of 63 family men in Pclass 3 Port S survive. 25 out of 202 bachelors. Total 34 out of 265
# It does appear that fate dealt a poor hand to males @ Port Q and a better hand to males @ Port C. We may want to discard the Embarked column along with the Cabin col as that too does not seem to add much value, but we will keep it for now. Maybe folks on certain ports were allocated rooms closer to the lifeboats..There is a slim chance of this but we will keep this column for now
# 
# We havent still analyzed the Name and the Ticket columns. When checking the corelations between variables, I found something unbelievable..Those people with lengthy Names had clear high chances of survival! Then it stuck me that maiden names are part of the name for married females. Thus if we sort by the length of the name, we get mostly all the women first and then later the men. Since it is undeniable that Woman across all Pclasses have clearly higher mortality than men, this explained the corelation. Do create a new col containing the length of the person's name and plot it against survival rates using any of the above graphs we discussed.
# 
# The second aspect is that Names combined with Tickets allow us the ability to Group people. They will provide us valuable hints to who all are travelling together...which brings us to the last and most important point. I realized this while plotting the group size versus survival. While analyzing the data I realized that while there is no connection between large groups and survival, there was a definite connection between survival of members within a team. This - more than Sex, Pclass and Age determines if a person survives or not for a vast majority of the population (barring Pclass 1 and 2 females who almost always survive). More specifically within a group, if we do a sub-grouping based on Sex, we can get to a fine degree of prediction of survival. So we must now build families and groups on the Titanic which is our next job!
# 
# Next 3 kernels due are:
# - Imputing the missing values - https://www.kaggle.com/allohvk/titanic-missing-age-imputation-tutorial-advanced
# - Building families on the Titanic
# - Feature engineering and ML modelling
# 
# Your votes will motivate me to spend a lot of time docuemnting my code like I did this one.
