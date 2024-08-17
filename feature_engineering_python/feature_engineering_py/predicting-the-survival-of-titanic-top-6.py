#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------
# ------------------------------------------
# # <p style="background-color:gray; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 10px 100px; color:black; hight:max"> Upvote my work if you found it useful.🎯 </p>
# ------------------------------------------
# ------------------------------------------
# 
# # <p style="background-color:#C0392B; font-family:newtimeroman; font-size:175%; text-align:center; border-radius: 15px 50px;">Predicting whether passengers on the Titanic would survive or not 🚢</p>
# <img src="https://miro.medium.com/max/2400/1*ePbfZdw6sz397xLWlFZLCQ.jpeg" alt="Titanic" hight=50 width=800></img>

# <p style="background-color:skyblue; font-family:newtimeroman; font-size:200%; text-align:center; border-radius: 10px 100px;"><b>Introduction</b></p>
# <b>The objective of this project is to build a model to predict whether passengers on the Titanic would survive or not based on pattern extracted from analysing 14 descriptive features like their age, Sex, class of travel, port Embarked etc.<br></b>
# <b>This project consists of two phases:
# <ul>
#     <li>Phase I: Focuses on data preprocessing and exploration, as covered in this report.
#     <li>Phase II : The model building, validation and prediction.
# </ul>
# <p style="background-color:skyblue; font-family:newtimeroman; font-size:200%; text-align:center; border-radius: 10px 100px;">We have 12 descriptive features:</p>
# <ul>
#     <li>PassengerId : Passenger's Id
#     <li>Age : Age of the Passenger
#     <li>Sex : Sex of the Passenger
#     <li>Name : Name of the Passenger
#     <li>Embarked : 
#         <ul>
#             <li>Southampton
#             <li>Cherbourg
#             <li>Queenstown
#         </ul>
#     <li>Parch : Number of Parents/Children Aboard
#     <li>SibSp : Number of Siblings/Spouses Aboard
#     <li>Fare : Passenger Fare
#     <li>Ticket : Ticket Number
#     <li>Cabin : Cabin
#     <li>Pclass : 
#         <ul>
#             <li>1 = 1st
#             <li>2 = 2nd
#             <li>3 = 3rd
#         </ul>
#     <li>Survived :
#         <ul>
#             <li>1 for Survived 
#             <li>0 for Not-Survived
#         </ul>
#     </ul>
# <h2><span>&#8226;</span> Outline:</h2>
# <ul>
#     <li><a href="#Phase I"><b>Phase I</b><a/>
#         <ul>
#             <li><a href="#head-1">Data Pre-processing</a>
#             <li><a href="#head-2">Setup and Basic EDA</a>  
#                 <ul>
#                     <li><a href="#head-2-1">Univariate Visualisation</a>
#                         <ul>
#                             <li><a href="#head-2-1-1">Categorical Features</a>
#                                 <ul>
#                                     <li><a href="#sex">sex column</a>
#                                     <li><a href="#pclass">Pclass column</a>
#                                     <li><a href="#embarked">Embarked column</a>
#                                     <li><a href="#parch">parch column</a>
#                                     <li><a href="#sibsp">SibSp column</a>
#                                     <li><a href="#ticket">Ticket column</a>
#                                     <li><a href="#cabin">Cabin column</a>
#                                 </ul>
#                             <li><a href="#head-2-1-2">Numerical Features</a>
#                                 <ul>
#                                     <li><a href="#age">Age column</a>
#                                     <li><a href="#fare">Fare column</a>
#                                 </ul>
#                         </ul>
#                     <li><a href="#head-2-2">Multivariate Visualisation</a>
#                         <ul>
#                             <li><a href="#sct_mtx">Scatter Matrix for the data</a>
#                             <li><a href="#corr_mtx">Correlation Matrix for the data</a>
#                             <li><a href="#multi">Multi-Violin and Multi-Box plots for each column</a>
#                         </ul>
#                 </ul>
#         </ul>
#         <li><a href="#Phase II"><b>Phase II:</b></a>
#         <ul>
#             <li><a href="#prep_ml">Prepare the data for the machine learning model.</a>
#             <li><a href="#ml_models">Comparing different Machine learning models.</a>
#             </ul>

# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:100%; text-align:center; border-radius: 15px 50px;">Importing necessary modules and libraries📚</p>

# <img src="https://media1.tenor.com/images/047e6fd4e7169886e992a8899e62b90b/tenor.gif?itemid=12547153" height="200" style="margin: 0 ; max-width: 950px;" frameborder="0" scrolling="auto" title="House price prediction"></img>

# In[1]:


#main libraries
import os
import re
import pickle
import numpy as np
import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly 
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode
import cufflinks as cf

#machine learning libraries:
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score, train_test_split
from sklearn.preprocessing  import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")


# You can go offline on demand by using
cf.go_offline() 

# To connect java script to your notebook
init_notebook_mode(connected=False)

# set some display options:
pd.set_option("display.float", "{:.4f}".format)
plt.rcParams['figure.dpi'] = 100
colors = px.colors.qualitative.Prism
pio.templates.default = "plotly_white"

# see our files:
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a class="anchor" id="Phase I"></a>
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 100px;">Phase I</p>

# ## Data Pre-processing <a class="anchor" id="head-1"></a>

# ### Getting the Data

# In[2]:


#import the data
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

#join all the data together
full_df = pd.concat([train_df,test_df])

#make a copy of the original data
train_df_orig = train_df.copy()
test_df_orig = test_df.copy()

#show the head of the data
train_df.head()


# ### Data Exploration/Analysis

# In[3]:


#the shape of the data
print('This data contains {} rows and {} columns splited into train/test datasets with ratio {}'.\
      format(full_df.shape[0],full_df.shape[1],round((test_df.shape[0]/train_df.shape[0])*100,2)))


# In[4]:


cols = train_df.columns
print(f'We have {len(cols)} columns : \n{cols}')


# In[5]:


#see information about the data

#get the sum of all missing values in the dataset
missing_values = train_df.isnull().sum()

#sorting the missing values in a pandas Series
missing_values = missing_values.sort_values(ascending=False)

feature_dtypes = train_df.dtypes
feature_names = missing_values.index.values
missing_values = missing_values.values
rows, columns = train_df.shape

print("=" * 50)
print('====> This data contains {} rows and {} columns'.format(rows,columns))
print("=" * 50)
print()

print("{:15} {:15} {:35} {:15}".format('Feature Name'.upper(),
                                     'Data Format'.upper(),
                                     'Missing values (Num - Perc)'.upper(),
                                     'Three Samples'.upper()))

for feature_name, dtype, missing_value in zip(feature_names,feature_dtypes[feature_names],missing_values):
    print("{:17} {:17} {:25}".format(feature_name,
                                 str(dtype), 
                                 str(missing_value) + ' - ' + 
                                 str(round(100*missing_value/sum(missing_values),3))+' %'), end="")

    for i in np.random.randint(0,len(train_df),2):
        print(train_df[feature_name].iloc[i], end=",")
    print()


# In[6]:


#show the types of columns
train_df.dtypes.to_frame().rename(columns={0:'Column type'})


# In[7]:


#finding the unique values in each column
for col in train_df.columns:
    print('We have {} unique values in {} column'.format(len(train_df[col].unique()),col))
    print('__'*30)


# In[8]:


train_df['SibSp'].unique()


# In[9]:


train_df['Parch'].unique()


# In[10]:


train_df['Embarked'].unique()


# In[11]:


train_df['Pclass'].unique()


# In[12]:


train_df['Sex'].unique()


# In[13]:


print('Age columns vary from {} to {}'.format(train_df['Age'].min(),train_df['Age'].max()))


# In[14]:


#describe our data
train_df[train_df.select_dtypes(exclude='object').columns].drop('PassengerId',axis=1).describe().\
style.background_gradient(axis=1,cmap=sns.light_palette('skyblue', as_cmap=True))


# In[15]:


#find the null values in each column
train_df.isnull().sum().to_frame().rename(columns={0:'Null values'})


# In[16]:


#visuaize the null values in each column
plt.figure(figsize=(20,6));
sns.heatmap(train_df.isnull(), cmap='viridis');


# In[17]:


#lets see the correlation between columns and target column
corr = train_df.corr()
corr['Survived'].sort_values(ascending=False)[1:].to_frame()\
.style.background_gradient(axis=1,cmap=sns.light_palette('green', as_cmap=True))


# In[18]:


#lets take a look to the shape of columns
train_df.skew().to_frame().rename(columns={0:'Skewness'}).sort_values('Skewness')


# In[19]:


#Visualize columns have highest Skewness
fig, axes = plt.subplots(1,3, figsize=(20, 8));
fig.suptitle('Highest Skewness', fontsize=25);

for i,col in zip(range(3),['Fare','SibSp','Parch']):
    sns.kdeplot(train_df[col], ax=axes[i],hue=train_df['Survived'])
    axes[i].set_title(col+' Distribution')


# <h1>Conclusions</h1><br>
# <li>We have alot of null values in cabin and age columns
# <li>Survived column have a higher correlation with:
#     <ul>
#         <li>Pclass <b> -0.338481</b>
#         <li>Fare <b> 0.257307</b>
#         <li>Parch <b> 0.081629</b> 
#     </ul>
# <li>We have some Columns with a high Skewness:
#     <ul>
#         <li>Fare <b> 4.7873</b>
#         <li>SibSp <b> 3.6954</b>
#     </ul>

# # Setup and Basic EDA
# <a  id="head-2"></a>

# ### Basic plotting functions

# In[20]:


#use all data in visualization
df = pd.concat([train_df,test_df], axis=0)

#create a new column for the total number of family (Passenger )
df['family count']=df['Parch']+df['SibSp']+1 

#cpitalize sex column
df['Sex'] = df['Sex'].apply(lambda x:x.title())

#create a new column based on survived column (replace 1 with survived and 0 survived non-survived)
df['target'] = df['Survived'].map({1:'Survived',0:'Not Survived'})

#use columns with lowercases
df = df.rename(columns=lambda x:x.lower())


# In[21]:


# lets define a function to plot a bar plot easily

def bar_plot(df,x,x_title,y,title,colors=None,text=None):
    fig = px.bar(x=x,
                 y=y,
                 text=text,
                 labels={x: x_title.title()},          # replaces default labels by column name
                 data_frame=df,
                 color=colors,
                 barmode='group',
                 template="simple_white",
                 color_discrete_sequence=px.colors.qualitative.Prism)
    
    texts = [temp[col].values for col in y]
    for i, t in enumerate(texts):
        fig.data[i].text = t
        fig.data[i].textposition = 'inside'
        
    fig['layout'].title=title

    for trace in fig.data:
        trace.name = trace.name.replace('_',' ').title()

    fig.update_yaxes(tickprefix="", showgrid=True)

    fig.show()


# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 50px;">Data Exploration and Analysis 🔍</p>

# <a id='head-2-1'></a>
# <h1>Univariate Visualisation</h1>

# <a id='head-2-1-1'></a>
# <h2>Categorical Features</h2>

# <a id='sex'></a>
# ### Sex column 

# In[22]:


temp = pd.DataFrame()

for sex in pd.unique(df['sex']).tolist():
    temp[sex] = df[df['sex']==sex]['target'].value_counts()
    
temp = temp.rename(columns={0:'Female',1:'Male'}).T
temp['Total sum'] = temp.sum(axis=1)

bar_plot(temp.reset_index(),
         'index',
         'age',
         ['Total sum','Survived','Not Survived'],
         title='Survived and Not-survived grouped by sex')


# <h1>Conclusions</h1><br>
# <li>Most of passengers are males.
# <li>Females have a high probability of survival.
# <li>The male death rate is much higher than the female passenger’s death rate.

# <a id='pclass'></a>
# ### Pclass column 

# In[23]:


temp = pd.DataFrame()

for p in set(pd.unique(df['pclass'])):
    temp[p] = df[df['pclass']==p]['target'].value_counts()
    
temp = temp.rename(columns={1:'Class 1',2:'Class 2', 3:'Class 3'}).T
temp['Total sum'] = temp.sum(axis=1)

bar_plot(temp.reset_index(),
         'index',
         'Pclass',
         ['Total sum','Survived','Not Survived'],
         title='Survived and Not-survived grouped by Pclass')


# <h1>Conclusions</h1><br>
# <li>Most of passengers were in Class 3.
# <li>The survive rate in Class-3 is the worst, and followed by class-2 and lastly, class-1.(make sense because Class-3 ticket is cheaper) and class 1 is more expenssive.
# <li>passengers in Class 1 and Class 2 have a high probability of survival.<br>
# <p>So Pclass has a strong relation with the probability of survival

# <a id='family_count'></a>
# ### Family Count column 

# In[24]:


#before applying particular test we have to look for Contingency table
family_count = pd.crosstab(index=df['family count'],columns=df['target'])
family_count 


# In[25]:


temp = pd.crosstab(index=df['family count'],columns=df['target']).reset_index()

temp['Total sum'] = temp.sum(axis=1)

bar_plot(temp,
         'family count',
         'Family number',
         ['Total sum','Survived','Not Survived'],
         title='Survived and Not-survived grouped by Family Number')


# <h1>Conclusions</h1><br>
# <li>Solo travellers have low probability of survival.
# <li>Groupes (from 2 to 4) have a high survive rate.
# <li>As group size increases, the probability of survival decreases.

# <a id='embarked'></a>
# ### Embarked Count column 

# In[26]:


df['embarked'].value_counts().to_frame().rename(columns={'embarked':'Total Count'})


# In[27]:


#we are still using the whole data for visualizion 
#but only train_df is counted because test_df doesn't have Survived column
temp = pd.DataFrame()

for e in df['embarked'].unique().tolist():
    temp[e] = df[df['embarked']==e]['target'].value_counts()
    
temp = temp.T.rename(index={'S':'Southampton','C':'Cherbourg','Q':'Queenstown'})
temp['Total sum'] = temp.sum(axis=1)

bar_plot(temp.reset_index(),
         'index',
         'Embarked',
         ['Total sum','Survived','Not Survived'],
         title='Survived and Not-survived grouped by Embarked column')


# <h1>Conclusions</h1><br>
# <li>The graph shows that about 69.9% of the people boarded from Southampton (914/1309 = 0.698). 
# <li>Just over 20.6% boarded from Cherbourg (270/1309 = 0.206) and the rest boarded from Queenstown, which is about 9.39% (123/1309 = 0.206). 

# <a id="head-2-1-2"></a>
# <h2>Numerical Features</h2>

# <a id='age'></a>
# ### Age column 

# In[28]:


df['age_category'] = pd.cut(df['age'].fillna(df['age'].mean()).astype(int), bins=[-1,11,18,22,27,33,40,66,100],
                            labels=["<=11","11-18","19-22","23-27","28-33","34-40","41-66",">=67"])

temp = pd.DataFrame()
for age in df['age_category'].unique().tolist():
    temp[age] = df[df['age_category']==age]['target'].value_counts()

temp = temp.T.reset_index()
temp['Total sum'] = temp.sum(axis=1)

bar_plot(temp.reset_index(),
         'index',
         'Age Category',
         ['Total sum','Survived','Not Survived'],
         title='Survived and Not-survived grouped by Age column')


fig = make_subplots(rows=2, cols=2,
                    specs=[[{"colspan": 2}, None],
                           [{}, {}]],
                    subplot_titles=('Age distribution',
                                    'Survived',
                                    'Not Survived'))

fig.add_trace(go.Histogram(x=df['age']),
              row=1, col=1)

fig.add_trace(go.Histogram(x=df[df['target']=='Survived']['age']),
              row=2, col=1)

fig.add_trace(go.Histogram(x=df[df['target']=='Not Survived']['age']),
              row=2, col=2)

fig.update_layout(showlegend=False, title_text='Distribution for Age')
fig.show()


# <h1>Conclusions</h1><br>
# <li>Most of Passengers were between 28 and 33.
# <li>Age column is positive skewed, meaning that few Passengers were higher than 50.
# <li>The graph shows the relationship between Age and survival rate. It becomes apparent that age group between 15 and 25 has the worst survival rate.
# With this, we could conclude that. The attribute Age has a serious quality problem: some age values are negative and large number 177 values are missing. If it is to be used as a predictor in a prediction model, it needs a lot of work in the stage of preprocess.

# <a id='fare'></a>
# ### Fare column 

# In[29]:


fig = make_subplots(rows=2, cols=2,
                    specs=[[{"colspan": 2}, None],
                           [{}, {}]],
                    subplot_titles=('Fare distribution',
                                    'Survived',
                                    'Not Survived'))

fig.add_trace(go.Histogram(x=df['fare'][:len(train_df)]),
              row=1, col=1)

fig.add_trace(go.Histogram(x=df[df['target']=='Survived']['fare'][:len(train_df)]),
              row=2, col=1)

fig.add_trace(go.Histogram(x=df[df['target']=='Not Survived']['fare'][:len(train_df)]),
              row=2, col=2)

fig.update_layout(showlegend=False, title_text='Distribution for Fare')
fig.show()


# <a id='head-2-2'></a>
# <h1>Multivariate Visualisation</h1>

# <a id="sct_mtx"></a>
# ### Scatter Matrix

# In[30]:


#create a scatter plot for the columns that have a hih correlation with target column
train_df['target'] = df['target'][:len(train_df)]
plt.figure();
sns.set(style='whitegrid', context='talk', palette='viridis');
sns.pairplot(data=train_df,hue='target');


# <a id="corr_mtx"></a>
# ### Correlation Matrix 

# In[31]:


#Correlation Map
corr = df.corr()

corr.iplot(kind='heatmap',
           colorscale='Blues',
           hoverinfo='all',
           layout = go.Layout(title='Correlation Heatmap for the correlation between our columns',
                              titlefont=dict(size=20)))


# <a id="multi"></a>
# <h2>Multi-Violin and Multi-Box plots for each column</h2>

# In[32]:


#lets create a function to plot a multi-violin easily

def multi_violin(df,iter_col,dist_col,color_col='survived'):
    if len(df[color_col].unique())!= 2:
        return 'Maximun number of unique values in the color columns is 2'
    i = 0
    data = []
    for ite in df[iter_col]:
        data.append(go.Violin(x=df[df[iter_col]==ite][iter_col],
                              y=df[df[color_col]==df[color_col].unique().tolist()[0]][dist_col],
                              name=str(df[color_col].unique().tolist()[0]),
                              jitter=0,
                              meanline={'visible':True},
                              line={"color": '#F78181'},
                              side='negative',
                              marker=dict(color= '#81F781'),
                              showlegend=(i==0)))

        data.append(go.Violin(x=df[df[iter_col]==ite][iter_col],
                              y=df[df[color_col]==df[color_col].unique().tolist()[1]][dist_col],
                              name=str(df[color_col].unique().tolist()[1]),
                              jitter=0,
                              meanline={'visible':True},
                              line={"color": '#00FF40'},
                              side='positive',
                              marker=dict(color= '#81F781'),
                              showlegend=(i==0)))
        i+=1


    layout = dict(title='Distribution of {} column for each {} colored by {}'.format(dist_col.replace('_',' ').title(),
                                                                                     iter_col.replace('_',' ').title(),
                                                                                     color_col.replace('_',' ').title()),
                  width=1000,height=600,
                  yaxis=dict(title='Distribution',titlefont=dict(size=20)),
                  xaxis=dict(title=iter_col))

    iplot(dict(data=data,layout=layout))    


# In[33]:


#create a function to plot multi-box plots easily

def multi_box(df,cat_col,dist_col,color_col):
    
    y = []
    x = []
    
    if len(df[color_col].unique())!= 2:
        return 'Maximun number of unique values in the color columns is 2'
    
    for c in set(df[cat_col].unique().tolist()):
        for t in set(df[color_col].unique()):
            y.append(df[(df[cat_col]==c) & (df[color_col]==t)][dist_col].values)
            x.append(cat_col+'('+str(c)+')'+' ('+str(t)+')')        

    colors = ['rgba(251, 43, 43, 0.5)', 'rgba(125, 251, 137, 0.5)', 
              'rgba(251, 43, 43, 0.5)', 'rgba(125, 251, 137, 0.5)', 
              'rgba(251, 43, 43, 0.5)', 'rgba(125, 251, 137, 0.5)',
              'rgba(251, 43, 43, 0.5)', 'rgba(125, 251, 137, 0.5)', 
              'rgba(251, 43, 43, 0.5)', 'rgba(125, 251, 137, 0.5)', 
              'rgba(251, 43, 43, 0.5)', 'rgba(125, 251, 137, 0.5)']

    traces = []

    for xd, yd, cls in zip(x, y, colors[:2*len(df[cat_col].unique())]):
            traces.append(go.Box(y=yd,
                                 name=xd,
                                 boxpoints='all',
                                 jitter=0.5,
                                 whiskerwidth=0.2,
                                 fillcolor=cls,
                                 marker=dict(size=2),
                                 line=dict(width=1)))

    layout = go.Layout(title='{} distribution colord by {} grouped by {}'.format(dist_col.title(),
                                                                                 color_col.title(),
                                                                                 cat_col.title()),
        xaxis=dict(title=cat_col,
                   titlefont=dict(size=16)),
        
        yaxis=dict(title='Distribution',
                   autorange=True,
                   showgrid=True,
                   zeroline=True,
                   dtick=5,
                   gridcolor='rgb(255, 255, 255)',
                   gridwidth=1,
                   zerolinecolor='rgb(255, 255, 255)',
                   zerolinewidth=2,
                   titlefont=dict(
                   size=16)),
        
        margin=dict(l=40,
                    r=30,
                    b=80,
                    t=100),
        
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgb(255, 243, 192)',
        showlegend=False)

    fig = go.Figure(data=traces, layout=layout)
    iplot(fig)    


# ### Age distribution for Pclass column

# In[34]:


multi_box(df.dropna(),'pclass','age','target')


# In[35]:


multi_violin(df=df.dropna(),iter_col='pclass',dist_col='age',color_col='target')


# <h1>Conclusions</h1><br>
# <li>In any class the age distribution of survived passebgers is right skewed, meaning that most of survived passengers in each class were younger.

# In[36]:


multi_box(df.dropna(),'sex','age','target')


# In[37]:


multi_violin(df=df.dropna(),iter_col='sex',dist_col='age',color_col='target')


# <a id="Phase II"></a>
# <a class="anchor" id="head-1"></a>
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 100px;">Phase II</p>

# <a id="prep_ml"></a>
# <h2>Prepare the data for the machine learning model</h2>

# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 50px;">Data Cleaning🔧</p>

# In[38]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test_PassengerId = test['PassengerId'].values #for submission


# In[39]:


#extracting the unique titles
title_list = pd.concat([train,test])['Name'].apply(lambda x: re.findall(r'[, ]\w+[.]',x)[0][:-1]).unique()
                                           
# Using this iteratively I was able to get a full list of titles.
title_list = ['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms','Major', 
             'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess','Jonkheer', 'Dona']

 
# replacing all titles with mr, mrs, miss, master, and boy 
def replace_titles(x):
    
    title=x['Title'].strip()
    if (x['Age']<13):
        return 'Boy'
    if title in ['Don', 'Rev', 'Col','Capt','Sir','Major','Jonkheer']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms','Lady','Dona']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
    
#create a new columns containing the title for each name
train['Title'] = train['Name'].apply(lambda x: re.findall(r'[, ]\w+[.]',x)[0][:-1])
test['Title'] = test['Name'].apply(lambda x: re.findall(r'[, ]\w+[.]',x)[0][:-1])

# apply replacing title function to all titles
train['Title'] = train.apply(replace_titles, axis=1)
test['Title'] = test.apply(replace_titles, axis=1)

#delete name column,PassengerId,Ticket
del train['Name']
del test['Name']

# Since the Ticket attribute has 681 unique tickets, it will be a bit tricky to convert them into useful categories. 
# So we will drop it from the dataset.
del train['Ticket']
del test['Ticket']

#drop PassengerId column
del train['PassengerId']
del test['PassengerId']


# In[40]:


print(f'Train data has : {train["Title"].unique()}'),
print(f'Test data has : {test["Title"].unique()}')


# In[41]:


train.head()


# In[42]:


test.head()


# ### Null values
# <p>Lets impute the missing values using <a href="https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html">KNNImputer </a>from sklearn<br>
# <p>To impute Cabin and Embarked (Categorical) columns you have to encode the strings to numerical values. In the below code snippet I am using ordinal encoding method to encode the categorical variables in my training data and then imputing using KNN.
# <p>You can see <a href="https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/">Basic Feature Engineering with the Titanic Data</a> 

# In[43]:


train.isnull().sum()


# In[44]:


imputed_data = []                    #initialize a list for the imputed datasets

for df in [train,test]:
    
    imputed_df = df.copy()           #initialize the imputed Dataframe
    categorical = ['Pclass','Sex','SibSp','Parch','Cabin','Embarked','Title'] #define the categorical data
    numerical = ['Age','Fare']                                                #define the numerical data
    
    #encoding Cabin and Embarked columns to numeric values
    imputed_df[['Cabin','Embarked','Sex','Title']] = imputed_df[['Cabin','Embarked','Sex','Title']].\
    apply(lambda series: pd.Series(
        LabelEncoder().fit_transform(
            series[series.notnull()]),
        index=series[series.notnull()].index))

    #define an imputer for numerical columns
    imp_num = IterativeImputer(estimator=RandomForestRegressor(),
                               initial_strategy='median',
                               max_iter=10,
                               random_state=0)

    #define an imputer for categorical columns
    imp_cat = IterativeImputer(estimator=XGBClassifier(verbosity = 0),
                               max_iter=10,
                               initial_strategy='most_frequent',
                               random_state=0)
    
    #impute the numerical column(Age)
    imputed_df[numerical] = imp_num.fit_transform(imputed_df[numerical])
    
    #impute the categorical columns(Embarked,Cabin)
    imputed_df[categorical] = imp_cat.fit_transform(imputed_df[categorical])

    #return the imputed value to its string values (Decoding) 
    for col in ['Cabin','Embarked','Sex','Title']:
        imputed_df[col] = LabelEncoder().fit(df[col]).inverse_transform(imputed_df[col].astype(int))
    
    imputed_df['Age'] = imputed_df['Age'].apply(lambda x:int(round(x,0)))
    
    imputed_data.append(imputed_df)
    
train_df = imputed_data[0]  
test_df = imputed_data[1]


# ### Adding new features

# In[45]:


for dataset in [train_df,test_df]:
    
    dataset.columns = [x.lower() for x in dataset.columns]
    
    
    #lets create age category
    dataset['age_category'] = pd.cut(dataset['age'].astype(int), bins=[-1,11,18,22,27,33,40,66,100],
                                labels=[1,2,3,4,5,6,7,8]).to_frame()
    
    #create a new column for the total number of family (Passenger )
    dataset['family count'] = dataset['parch']+dataset['sibsp']+1 
    dataset['family count'] = dataset['family count'].astype(int)
    
    # Age times Class
    dataset['age_class'] = dataset['age_category']* dataset['pclass']
    dataset['age_class'] = dataset['age_class'].astype(int) 
    
    # Fare per Person
    dataset['fare_per_person'] = dataset['fare']/(dataset['family count'])
    dataset['fare_per_person'] = round(dataset['fare_per_person'].astype(float), 0).astype(int)
    
    # Is alone
    dataset['is_alone'] = 0
    dataset.loc[dataset['family count'] == 1, 'is_alone'] = 1
    
    #convert pclass, sibsp and parch columns to int
    dataset[['pclass','sibsp','parch']] = dataset[['pclass','sibsp','parch']].astype(int)


# In[46]:


train_df.head()


# In[47]:


test_df.head()


# #### Data Preprocessing

# In[48]:


train_set = train_df
X_set = train_set.drop('survived', axis=1)
y_set = train_set['survived']
test_set  = test_df


# In[49]:


# first choose columns to train the model with
scaled_data = []
num_features = ['age','fare']
cat_features = ['age_category','title','sex','embarked','is_alone','age_class','cabin','pclass','sibsp']

df = pd.concat([X_set,test_set])

# convert our categorical columns to dummies instead of LabelEncoding
for col in cat_features:
    dumm = pd.get_dummies(df[col], prefix = col,dtype=int)
    del df[col]
    df = pd.concat([df,dumm], axis=1)

X_set = df[:len(X_set)]
test_set = df[len(X_set):]

for data in [X_set,test_set]:
    # scaling  our numeric columns
    std = StandardScaler()
    for col in num_features:
        data[col] = pd.Series(std.fit_transform(data[col].values.reshape(-1,1)).reshape(-1))


# <a id="ml_models"></a>
# <h2>Comparing different Machine learning models</h2>

# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 15px 50px;">Initializing our models📝</p>

# In[50]:


#split the data
X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, 
                                                    test_size=0.2, random_state=42)


# In[51]:


# define models to test:
base_models = [("SVM",      SVC()),                                                               #Support Vector Machines
               ("kNN",      KNeighborsClassifier(n_neighbors = 3)),                               #KNeighborsClassifier
               ("LR_model", LogisticRegression(random_state=42,n_jobs=-1)),                       #Logistic Regression model
               ("DT_model", DecisionTreeClassifier(random_state=42)),                             #Decision tree model
               ("RF_model", RandomForestClassifier(random_state=42, n_jobs=-1)),                  #Random Forest model
               ("XGBoost", XGBClassifier()),                                                      #XGBoost model
               ("Bagging_model",BaggingClassifier(base_estimator=DecisionTreeClassifier(),        #Bagging model
                                                 max_samples=30,
                                                 n_estimators=500,
                                                 n_jobs=-1,
                                                 bootstrap=True,
                                                 oob_score=True)),
               ("Random_subspaces_model",BaggingClassifier(base_estimator=DecisionTreeClassifier(),#Random subspaces model
                                                           n_estimators=100,
                                                           bootstrap=False,
                                                           max_samples=1.0,
                                                           max_features=True,
                                                           bootstrap_features=True,
                                                           n_jobs=-1)),
                ("Random_Patches_model", BaggingClassifier(base_estimator=DecisionTreeClassifier(),#Random Patches model
                                                            n_estimators=100,
                                                            bootstrap=True,
                                                            max_samples=1.0,
                                                            max_features=True,
                                                            bootstrap_features=True,
                                                            n_jobs=-1)),
                ("AdaBoost_model",AdaBoostClassifier(DecisionTreeClassifier(),                      #AdaBoost model
                                                    n_estimators=100,
                                                    learning_rate=0.01)),
                ("GradientBoosting",GradientBoostingClassifier(max_depth=2,                        #GradientBoosting model
                                                              n_estimators=100))]


# <p>To evaluate the performance of any machine learning model we need to test it on some unseen data, based on the models 
# performance on unseen data we can say weather our model is :
#     <ul>
#         <li>Under-fitting.
#         <li>Over-fitting.
#         <li>Well generalized.
#     </ul>
# <b>Cross validation (CV)</b> is one of the technique used to test the effectiveness of a machine learning models, it is also a re-sampling procedure used to evaluate a model if we have a limited data.<br>
# To perform CV we need to keep aside a sample/portion of the data on which is not used to train the model, later use this sample for testing/validating.<br>
# So, k-fold cross validation is used for two main purposes:
# <ul>
#     <li>To tune hyper parameters.
#     <li>To better evaluate the performance of a model.

# <h3><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html">BaggingClassifier</a>:</h3><br>
# <li>Bagging and Pasting have the same algorithm, each model is trained using subsets but the way you chose the subsets changes:
# <ul>
#     <li>In Bagging you do sampling with replacement.
#     <li>In Pasting you do sampling without replacement.
# </ul>
# So two algorithms allow training instances to be sampled several times accroce multiple predictors, but only bagging allows training instances to be sampled several times for the same predictor.
# <li>In Bagging we control sampling by <b>(max_sample,Bootstrap).</b>
#     Random subspaces and Random Patches are extension to Bagging algorithm, but they do feature sampling instead of instances sampling, they are useful when we have a high dimensions inputs.<br>
# <ul>
#     <li>In Random subspaces you do sampling to both training and features.
#     <li>In Random Patches you keep all training instances <b>(Bootstrap=False, max_sample=1)</b>, but you do sampling feature by <b>(Bootstrap_feature=True)</b>.
# </ul>
# <h3><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html">AdaBoostClassifier</a>:</h3><br>
# <li>AdaBoost Combines a lot of week learners to make Classification <b>(week learners like stump in DecisionTree)</b>

# In[52]:


kfolds = 5   # it is better to be 1/(size of testing test)
split = StratifiedKFold(n_splits=kfolds,
                        shuffle=True, 
                        random_state=42)  # use shuffle to ensure random distribution of data

# Preprocessing, fitting, making predictions and scoring for every model:
models_data = {'min_score':{},'max_score':{},'mean_score':{},'std_dev':{}}
for name, model in base_models:
    # get cross validation score for each model:
    cv_results = cross_val_score(model, 
                                 X_set, y_set, 
                                 cv=split,
                                 scoring="accuracy",
                                 n_jobs=-1)
    
    # output:
    #To find the average of all the accuracies.
    min_score = round(min(cv_results)*100, 4)
    models_data['min_score'][name] = min_score
     
    #To find the max accuracy of all the accuracies.
    max_score = round(max(cv_results)*100, 4)
    models_data['max_score'][name] = max_score
    
    #To find the min accuracy of all the accuracies.
    mean_score = round(np.mean(cv_results)*100, 4)
    models_data['mean_score'][name] = mean_score
    
    # let's find the standard deviation of the data to see degree of variance in the results obtained by our model.
    std_dev = round(np.std(cv_results), 4)
    models_data['std_dev'][name] = std_dev
    
    print(f"{name} cross validation accuarcy score: {mean_score} +/- {std_dev} (std) ---> min: {min_score}, max: {max_score}")


# In[53]:


models_df = pd.DataFrame(models_data).sort_values(by='mean_score',ascending=False)
models_df


# In[54]:


title = 'Mininam, Maximam and Mean score for each model'
models_df.iplot(kind='bar',
               title=title)


# In[55]:


#lets train our models with all test sets 
accuracies = {}
models = {}
model = base_models
for name,model in base_models:
    model.fit(X_train, y_train)
    models[name]=model
    acc = model.score(X_test, y_test)*100
    accuracies[name] = acc
    print("{} Accuracy Score : {:.3f}%".format(name,acc))


# In[56]:


models_res = pd.DataFrame(data=accuracies.items())
models_res.columns = ['Model','Test score']
models_res.sort_values('Test score',ascending=False)


# In[57]:


new_model_df = models_df.join(models_res.set_index('Model'))
new_model_df['(Test Score - Cross_Validation Score)%'] = new_model_df['Test score'] - new_model_df['mean_score']
new_model_df.sort_values('(Test Score - Cross_Validation Score)%',ascending=True)


# <h1>Conclusions</h1><br>
# <li>We found that <b>GradientBoosting</b> with K-Fold Cross Validation reached a good mean accracy and a low value of standard deviation. This values of std is extremely low, which means that our model has a very low variance, which is actually very good since that means that the prediction that we obtained on one test set is not by chance. Also, the model performed good on all test sets. which mean that our model has no overfitting.

# In[58]:


Y_pred = models['GradientBoosting'].fit(X_set, y_set).predict(test_set)
submission = pd.DataFrame({
        "PassengerId": test_PassengerId,
        "Survived": Y_pred
    })
submission.to_csv('submission_GB.csv', index=False)


# ### <h1 align="center">Thanks for reading</h1>
# <h2 align="center" style='color:red' > If you like the notebook or learned something please upvote! </h2>
# <b><li>You can also see:</li></b>
# <ul>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/house-price-prediction-top-8'>House price prediction (Top 8%)</a>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/prediction-of-heart-disease-machine-learning'>Prediction of Heart Disease (Machine Learning)</a>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/data-exploration-and-visualization-uber-data'>Data exploration and visualization(Uber Data)</a><br>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/hotel-booking-eda-cufflinks-and-plotly'>Hotel booking EDA (Cufflinks and plotly)
# </a><br>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/suicide-rates-visualization-and-geographic-maps/edit/run/53135916'>Suicide Rates visualization and Geographic maps</a>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/superstore-data-analysis-with-plotly-clustering'>Superstore Data Analysis With Plotly(Clustering)</a>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/superstore-analysis-with-cufflinks-and-pandas'>Superstore Analysis With Cufflinks and pandas</a>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/learn-data-analysis-using-sql-and-pandas'>Learn Data Analysis using SQL and Pandas</a>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/european-soccer-database-with-sqlite3'>European soccer database with sqlite3</a>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/chinook-questions-with-sqlite'>Chinook data questions with sqlite3</a>
# 
# </ul>
