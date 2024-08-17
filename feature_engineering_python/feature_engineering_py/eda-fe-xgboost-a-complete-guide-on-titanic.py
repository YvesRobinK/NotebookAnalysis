#!/usr/bin/env python
# coding: utf-8

# <h1 style="background-color:lightgreen;font-family:newtimeroman;font-size:300%;text-align:center">Index</h1>
# <span style='font-family: "Times New Roman", Times; font-size: 28px;'>
# 
# * [1.Importing Libraries](#1)    
# * [2.Quick look using automated EDA](#2)
# * [3.EDA and Data Visualizations](#3)
# * [4.Data Cleaning and Feature Engineering](#4)
# * [5.Training and Ensemble](#5)
# * [6.Conclusion](#6)                    
# </span>

# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>Hello Everyone, <br>
#     This notebook showscases xgboost pipeline for titanic competition with EDA and FE, the conclusions drawn from this EDA are defined in section 3, our main focus will be to build a model capable of drawing inference from new features which we will be generating using feature engineering. Finally this notebook demonstrates how you can also do some stacking/ensembling. This notebook demonstrates common steps to be followed in Titanic competition. Further info can be found at this <a href = 'https://www.kaggle.com/c/titanic/discussion/218052'> post</a>.
# </span></p>

# <a id="1"></a>
# 
# <h1 style="background-color:lightgreen;font-family:newtimeroman;font-size:250%;text-align:center">Importing Libraires And Functions</h1>
# 

# In[1]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

#basics
import numpy as np 
import pandas as pd 


#methods n algo
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
import lightgbm as lgbm
import xgboost as xgb

#plots and other utilities
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
#pd.set_option('max_columns',100)
import plotly.express as ex

import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
pyo.init_notebook_mode()
from pandas.plotting import parallel_coordinates
import pandas_profiling
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
plt.rc('figure',figsize=(17,13))


# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>Importing Train and Test Dataset
# </span></p>

# In[2]:


train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.head()


# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>We will be installing sweetviz library for automated EDA, it can give us general overview about data, its distribution and null values.
# </span></p>

# In[3]:


#comment this after installation of sweetviz (running second time)
get_ipython().system('pip install sweetviz')


# <a id="2"></a>
# 
# <h1 style="background-color:lightgreen;font-family:newtimeroman;font-size:250%;text-align:center">Quick look using automated EDA</h1>

# In[4]:


#analyzing data using sweetviz (automated EDA)
import sweetviz as sv
data_report = sv.analyze(train_data)
data_report.show_html('Analysis.html')


# In[5]:


from IPython.display import IFrame
IFrame(src = 'Analysis.html',width=1000,height=600)


# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>We can notice some null values in Age, Cabin and Embarked! Now let's visualize the data distribution of individual features.
#     Feature selection seems trivial but it is of great importance when it comes to performance, we are going for feature engineering after this section, it will hopefully lead us to some conclusions with we can engineer new features</span></p>

# In[6]:


features = ["Pclass","Age","Fare", "Sex", "SibSp", "Parch","Embarked"]
target = "Survived"

train_df = train_data[features]
test_df = test_data[features]
target_data = train_data[target]
train_df.head()


# <a id="3"></a>
# 
# <h1 style="background-color:lightgreen;font-family:newtimeroman;font-size:250%;text-align:center">EDA and Data Visualizations</h1>

# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>Let's try our hands on some Exploratory Data Analysis, we will be using training data only for visualization as of now, as seen from the dataset, test and train are having almost same distribution, so it won't matter that much.
# </span></p>

# <a id="3"></a>
# 
# <h1 style="background-color:lightgreen;font-family:newtimeroman;font-size:200%;text-align:center">Numerical Features</h1>
# 

# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>Let's start with Age distribution among passengers
# </span></p>

# In[7]:


plt.figure(figsize=(14,8))
fig = sns.distplot(train_df['Age'], color='Blue')
fig.set_xlabel("Age of passengers",size=15)
fig.set_ylabel("Passengers Frequency",size=15)
plt.title('Age Distribution Among Passengers',size = 20)
plt.show()


# 
# <a id="2"></a>
# 
# <h2 style="background-color:lightgreen;font-family:newtimeroman;font-size:200%;text-align:center">Categorical vs Numerical </h2>
# 

# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>We have two numerical features, Age and Fare, which we will use to plot against categorical features, Pclass, Gender 
# </span></p>

# In[8]:


#colors = ['lightblue', 'mediumturquoise', 'red', 'lightgreen']*9
fig =   ex.histogram(train_df,x='Fare',title="Fare Distribution Among Passenger Classes",color='Pclass')
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.8)))
# Here we modify the layout
fig.update_layout(
                 title='Fare Distribution Among Passenger Classes',
                legend=dict(
                    x=1.0,
                    y=1.0,
                    bgcolor='rgba(50, 200, 100, 0.3)',
                    bordercolor='rgba(255, 255, 255, 0.6)'
                ),
                 )
fig.show()


# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>The Fare distribution looks skewed, to look closer we will only using data points with less than 150 fare value
# </span></p>

# In[9]:


#let's scale this distribution to get a clear picture
fig =   ex.histogram(train_df[train_df['Fare']<=150],x='Fare',title="Scaled Fare Distribution Among Passengers Classes",color='Pclass')
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.8)))
# Here we modify the layout
fig.update_layout(
                 title='Scaled Fare Distribution Among Passengers Classes',
                legend=dict(
                    x=1.0,
                    y=1.0,
                    bgcolor='rgba(50, 200, 100, 0.3)',
                    bordercolor='rgba(255, 255, 255, 0.6)'
                ),
                 )
fig.show()


# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>It looks pretty good, now we will be visualizing age distribution against class
# </span></p>

# In[10]:


fig =   ex.histogram(train_df,x='Age',title="Age Distribution Among Classes",color = 'Pclass')
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.8)))
# Here we modify the layout
fig.update_layout(
                 title='Age Distribution Among Classes',
                legend=dict(
                    x=1.0,
                    y=1.0,
                    bgcolor='rgba(50, 200, 100, 0.3)',
                    bordercolor='rgba(255, 255, 255, 0.6)'
                ),
                 )
fig.show()


# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>Let's have a look at Age against Gender
# </span></p>

# In[11]:


fig =   ex.histogram(train_df,x='Age',title="Age Distribution Among Gender",color = 'Sex')
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.8)))
# Here we modify the layout
fig.update_layout(
                 title='Age Distribution Among Gender',
                legend=dict(
                    x=1.0,
                    y=1.0,
                    bgcolor='rgba(50, 200, 100, 0.3)',
                    bordercolor='rgba(255, 255, 255, 0.6)'
                ),
                 )
fig.show()


# <a id="1"></a>
# 
# <h1 style="background-color:lightgreen;font-family:newtimeroman;font-size:200%;text-align:center">Categorical Features</h1>
# 

# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'> Different Categorical features can give us insight on how each value represents which fraction of overall dataset, this includes Gender, Pclass, and emabarked feature
# </span></p>

# In[12]:


colors = ['lightbrown', 'mediumturquoise', 'red', 'lightgreen']

labels = train_data.Embarked.value_counts().index
values = train_data.Embarked.value_counts().values

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.1)])

fig.update_traces(marker=dict(colors = colors, line=dict(color='#000000', width=3)))

# Here we modify the layout
fig.update_layout(
                 title='Passengers count by Embarked Feature',
                legend=dict(
                    x=1.0,
                    y=1.0,
                    bgcolor='rgba(50, 200, 100, 0.3)',
                    bordercolor='rgba(255, 255, 255, 0.6)'
                ),
                 )
fig.show()



# In[13]:


colors = ['mediumturquoise', 'lightred', 'lightgreen']

labels = train_data.Sex.value_counts().index
values = train_data.Sex.value_counts().values

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.1)])

fig.update_traces(marker=dict(colors = colors, line=dict(color='#000000', width=3)))
# Here we modify the layout
fig.update_layout(
                 title='Passengers count by Gender',
                legend=dict(
                    x=1.0,
                    y=1.0,
                    bgcolor='rgba(50, 200, 100, 0.3)',
                    bordercolor='rgba(255, 255, 255, 0.6)'
                ),
                 )

fig.show()


# In[14]:


colors = [' ', 'mediumturquoise', 'red', 'lightgreen']

labels = train_data.Pclass.value_counts().index
values = train_data.Pclass.value_counts().values

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.1)])

fig.update_traces(marker=dict(colors = colors, line=dict(color='#000000', width=3)))

# Here we modify the layout
fig.update_layout(
                 title='Passengers count by Class',
                legend=dict(
                    x=1.0,
                    y=1.0,
                    bgcolor='rgba(50, 200, 100, 0.3)',
                    bordercolor='rgba(255, 255, 255, 0.6)'
                ),
                 )

fig.show()


# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'> Next section describes how survived feature correlates with other categorical features, for example the next graph gives info about how many passengers from which class survived the most and which class suffered the most. Hopefully this will help better 
# </span></p>

# In[15]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=train_data['Survived'].unique(),
    y=train_data.query("Pclass == 1").groupby('Survived').count()['Pclass'],
    name='class 1',
    marker_color='mediumturquoise'
))
fig.add_trace(go.Bar(
    x=train_data['Survived'].unique(),
    y=train_data.query("Pclass == 2").groupby('Survived').count()['Pclass'],
    name='class 2',
    marker_color='lightsalmon'
))
fig.add_trace(go.Bar(
    x=train_data['Survived'].unique(),
    y=train_data.query("Pclass == 3").groupby('Survived').count()['Pclass'],
    name='class 3',
    marker_color='lightgreen'
))

# Here we modify the layout
fig.update_layout(barmode='group',
                 title='Survived Passengers Among Class',
                #xaxis_tickfont_size=14,
                yaxis=dict(
                    title='Passanger Count',
                    titlefont_size=16,
                    tickfont_size=14,
                ),
                  xaxis=dict(
                    title='Passanger Survived No/Yes',
                    titlefont_size=16,
                    tickfont_size=14,
                ),
                legend=dict(
                    x=1.0,
                    y=1.0,
                    bgcolor='rgba(50, 200, 100, 0.3)',
                    bordercolor='rgba(255, 255, 255, 0.6)'
                ),
                bargap=0.15, # gap between bars of adjacent location coordinates.
                bargroupgap=0. # gap between bars of the same location coordinate.
                 )
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.8)))
fig.show()


# In[16]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=train_data['Survived'].unique(),
    y=train_data.query("Sex=='male'").groupby('Survived').count()['Sex'],
    name='male',
    marker_color='mediumturquoise'
))
fig.add_trace(go.Bar(
    x=train_data['Survived'].unique(),
    y=train_data.query("Sex=='female'").groupby('Survived').count()['Sex'],
    name='female',
    marker_color='lightsalmon'
))

# Here we modify the layout
fig.update_layout(barmode='group',
                 title='Survived Passengers Among Gender',
                #xaxis_tickfont_size=14,
                yaxis=dict(
                    title='Passanger Count',
                    titlefont_size=16,
                    tickfont_size=14,
                ),
                  xaxis=dict(
                    title='Passanger Survived No/Yes',
                    titlefont_size=16,
                    tickfont_size=14,
                ),
                legend=dict(
                    x=1.0,
                    y=1.0,
                    bgcolor='rgba(50, 200, 100, 0.3)',
                    bordercolor='rgba(255, 255, 255, 0.6)'
                ),
                bargap=0.15, # gap between bars of adjacent location coordinates.
                bargroupgap=0. # gap between bars of the same location coordinate.
                 )
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.8)))
fig.show()


# In[17]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=train_data['Survived'].unique(),
    y=train_data.query("SibSp>=1").groupby('Survived').count()['SibSp'],
    name='Having Sib/Sp',
    marker_color='mediumturquoise'
))
fig.add_trace(go.Bar(
    x=train_data['Survived'].unique(),
    y=train_data.query("SibSp==0").groupby('Survived').count()['SibSp'],
    name='Not having Sib/Sp',
    marker_color='lightsalmon'
))

# Here we modify the layout
fig.update_layout(barmode='group',
                 title='Survived Passengers Having sibling/Spouse',
                #xaxis_tickfont_size=14,
                yaxis=dict(
                    title='Passanger Count',
                    titlefont_size=16,
                    tickfont_size=14,
                ),
                  xaxis=dict(
                    title='Passanger Survived No/Yes',
                    titlefont_size=16,
                    tickfont_size=14,
                ),
                legend=dict(
                    x=1.0,
                    y=1.0,
                    bgcolor='rgba(50, 200, 100, 0.3)',
                    bordercolor='rgba(255, 255, 255, 0.6)'
                ),
                bargap=0.15, # gap between bars of adjacent location coordinates.
                bargroupgap=0. # gap between bars of the same location coordinate.
                 )
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.8)))
fig.show()


# In[18]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=train_data['Survived'].unique(),
    y=train_data.query("Parch>=1").groupby('Survived').count()['Parch'],
    name='With Par/Ch',
    marker_color='mediumturquoise'
))
fig.add_trace(go.Bar(
    x=train_data['Survived'].unique(),
    y=train_data.query("Parch==0").groupby('Survived').count()['Parch'],
    name='Without Par/Ch',
    marker_color='lightsalmon'
))

# Here we modify the layout
fig.update_layout(barmode='group',
                 title='Survived Passengers With Parents/Children',
                #xaxis_tickfont_size=14,
                yaxis=dict(
                    title='Passanger Count',
                    titlefont_size=16,
                    tickfont_size=14,
                ),
                  xaxis=dict(
                    title='Passanger Survived No/Yes',
                    titlefont_size=16,
                    tickfont_size=14,
                ),
                legend=dict(
                    x=1.0,
                    y=1.0,
                    bgcolor='rgba(50, 200, 100, 0.3)',
                    bordercolor='rgba(255, 255, 255, 0.6)'
                ),
                bargap=0.15, # gap between bars of adjacent location coordinates.
                bargroupgap=0. # gap between bars of the same location coordinate.
                 )
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.8)))
fig.show()


# <p style="text-align: left;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>We have looked at data distribution and now we can draw some conclusions from our data analaysis. <br>
#    1. The passengers with parents/children and siblings/spouse had higher chances of survival (You better travel with your family next time)<br>
#  2. There was higher chance of death among male passengers comapred to female passengers<br>
#     3. The survival rate of class 3 is lower compared to other 2 classes<br>
#     4. Fare distribtuion is uneven showing the ticket prices may be split according to classes<br>
#     5. There are a lot of passengers in class 3 compared to class 1<br>
#   </span></p>

# <a id="4"></a>
# 
# <h1 style="background-color:lightgreen;font-family:newtimeroman;font-size:250%;text-align:center">Feature Engineering</h1>

# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>We have looked at data distribution and now it's time for feature engineering, precisely we will modify data based on conclusions drawn from above visualizations. Hopefully, these can help to see some patterns and boost our performance using our custom features</span></p>

# In[19]:


#describe train data
train_df.describe()


# In[20]:


train_df.isnull().sum(), test_df.isnull().sum()


# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>From the above data we can see that there are indeed some missing values in Age which we are going to fill using median values.</span></p>

# In[21]:


#remove null value from Age
print('median Age of the passengers: ',train_df['Age'].median())
print(train_df['Age'].isnull().sum(),test_df['Age'].isnull().sum(),"null values present in train, test")
#now we replace all missing values with that value
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)


test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
print(train_df['Age'].isnull().sum(),train_df['Age'].isnull().sum(),"null values remaining in train, test")


# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>Now it's time to generate some new features from our dataset, we have drawn some conclusions which will be helpful, for eg. merging Parch and SibSp features </span></p>

# In[22]:


def func(x):
    if x[0]>0 or x[1]>0:
        return True
    else:
        return False
    
#Apply new function to features, and generate with family feature
train_df['WithFamily']=train_df[['SibSp','Parch']].apply(func,axis=1)
test_df['WithFamily']=test_df[['SibSp','Parch']].apply(func,axis=1)


# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>The features Sibsp and Parch are redunadnt as its information is already stored inside withFamily feature, so we drop them</span></p>

# In[23]:


train_df.drop(columns=['SibSp','Parch'],inplace=True)
test_df.drop(columns=['SibSp','Parch'],inplace=True)


# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>We will replace the two rows in data where Embarked feature is missing with S since 72% of the Passengers  </span></p>

# In[24]:


train_df['Embarked'].fillna('S',inplace = True)
test_df['Embarked'].fillna('S',inplace = True)


# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>We have already done some of the feature processing in previous section and Now we will be focusing on converting numerical continuous features into binned features</span></p>

# In[25]:


#this part is adapted from this kernal
#checkout this great kernal on FE,https://www.kaggle.com/udit1907/titanic-disaster-beginner-python-documentation/notebook


#we will use binning for Age and Fare features, hopefully will produce better results
#func to process values
def encodeAgeFare(train):
    train.loc[train['Age'] <= 16, 'Age'] = 0
    train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
    train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
    train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
    train.loc[ (train['Age'] > 48) & (train['Age'] <= 80), 'Age'] = 4
    
    train.loc[train['Fare'] <= 7.91, 'Fare'] = 0
    train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
    train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31.0), 'Fare'] = 2
    train.loc[(train['Fare'] > 31.0) & (train['Fare'] <= 512.329), 'Fare'] = 3

encodeAgeFare(train_df)
encodeAgeFare(test_df)


# In[26]:


#no need to apply 
print(train_df.isnull().sum())
#train_df = train_df.dropna(axis = 0)


# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>We need to change encoding of our train-test data for categorical data, we will be using One-hot encoding for that. You can find further info <a href='https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding'>(here)</span></p>

# In[27]:


#To change categorical to numerical, we will be using above One-hot encoding 
train_df=pd.get_dummies(train_df,columns=['Sex','Embarked','WithFamily'],drop_first=True)
test_df=pd.get_dummies(test_df,columns=['Sex','Embarked','WithFamily'],drop_first=True)


# In[28]:


train_df


# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>We will be using parameters found by tuning using Grid Search, hopefully will produce better results than manual tuning</span></p>

# In[29]:


# param_grid_xgb = {'n_estimators':[400, 600],
#                   'learning_rate':[0.01, 0.03, 0.05],
#                   'max_depth':[3, 4],
#                   'subsample':[0.5, 0.7],
#                   'colsample_bylevel':[0.5, 0.7],
#                   'reg_lambda':[15, None],
#                  }

# clf = xgb.XGBClassifier()
# grid = GridSearchCV(clf, cv=5, param_grid = param_grid_xgb,scoring='accuracy', verbose=2, n_jobs=-1)
# grid.fit(train_df, target_data)
# grid.best_params_
# model = grid.best_estimator_


# <a id="5"></a>
# 
# <h1 style="background-color:lightgreen;font-family:newtimeroman;font-size:250%;text-align:center">Training and Ensemble</h1>

# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'> These parameters are tuned using grid search CV, it takes a lot of time to run through the search space, for optimal results, we will be directly using best parameters results, in case you want to try out yourself the HPO, expand the previous cell and have a look at it.
# </span></p>

# In[30]:


model = xgb.XGBClassifier(**{'colsample_bylevel': 0.7, 'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 1000,
                                 'reg_lambda': 15,'eval_metric': 'error','subsample': 0.5}).fit(train_df, target_data)


# In[31]:


#as a final step, we will be predicting with our best model

test_y = model.predict(test_df).astype(int)


# <p style="text-align: center;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>As the final step, we will be using ensembling from a <a href ='https://www.kaggle.com/mauricef/titanic'>previous notebook </a>, it contains information from Name feature which we have not taken into account
#     </span></p>

# In[32]:


test_z = pd.read_csv('/kaggle/input/k/mauricef/titanic/survived.csv')['Survived'].to_list()

final_preds = [z * 0.75 + y * 0.25 for z,y in zip(test_z,test_y)]

final_preds = (np.array(final_preds) > 0.5) * 1
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': final_preds})
output.to_csv('my_submission.csv', index=False)


# In[33]:


print('XGBoost score on train data:', round(model.score(train_df, target_data) * 100, 2))


# <a id="6"></a>
# 
# <h1 style="background-color:lightgreen;font-family:newtimeroman;font-size:250%;text-align:center">Conclusion </h1>

# <p style="text-align: left;"><span style='font-family: "Times New Roman", Times; font-size: 22px;'>We can draw some conclusions from above notebook and our process, notably : <br>
#     1.We found that EDA can be a powerful tool in some cases where you can get good insight into data<br>
#     2.Feature engineering can boost score significantly when features are used properly<br>
#     3. Hyper parameter optimization is important but don't waste too much time doing it<br>
#     4. Basic pipeline is a must to understanding problem statement and gettting started. <br>
#     5. Stacking with right model is important, but for that first we need to know the diversity of predictions and how different it is from our model<br>
#     At last we trained XGboost model and worked on our submission with ensembling with another predictions
#     </span></p>

# In[ ]:




