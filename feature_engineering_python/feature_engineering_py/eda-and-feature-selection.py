#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np 
import seaborn as sns
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split,GridSearchCV
from sklearn.decomposition import PCA
from collections import defaultdict
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import scale
from scipy import stats


# # So here are we with Forest Cover Analysis , that on various features we have to predict at least seven types of covers given to us ,
# # Feature engg , polynomial features with pca and xgboost apart from lgbm we can get best result since being a handling various dataset, this problem is merely on few featured ensembel models 
# # Decision tree , base line model
# # random forest , stackers, 
# # gbm , ada, xgboost and lgbm 
# # with taste of PCA too . 
# # evauation metrics will be accuracy , AUC , logloss and checking , variance and bias in final model

# In[ ]:


train = pd.read_csv("../input/train.csv")
test= pd.read_csv("../input/test.csv")


# In[ ]:


dftrain=train.copy()


# In[ ]:


dftrain.head()


# In[ ]:


dftrain.info()


# # As we can see that we are provided with non null dataset so we can more further by undertsanding outlier , feature scaling etc 
# 

# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(dftrain.corr(),fmt=".2f",cmap="YlGnBu")


# # from correlation analysis we can see that very few featues are depicting relation with one another a l like graph can been from top to bottom on LHS , VERY FEW soil type are showing some corrleation with another and other var as we can see some white lines has been drawn up this due tp heatmap scaling of points

# In[ ]:


dftrain.describe()


# # AS we can see that mean  vs max values lot of gap , outliers are here lot's of 
# # three types of elevation cover we can see with the help of quartiles 

# # now some interesting visualization by means of using correlation analysis

# # as we can see slope is increasing and elevation to we are getting various forest covers so this variable is nececssary to know which type of cover

# In[ ]:


dftrain.columns


# In[ ]:


for i in dftrain.columns:
    sns.FacetGrid(dftrain,hue="Cover_Type",size=8)\
        .map(sns.distplot ,i)
plt.legend()    


# # Let's apply Chi square test to find the relation of prediction with target variable 
# ## for those who don't who what is chi square here is description to understand what it is exactly
# Introduction
# Feature selection is an important part of building machine learning models. As the saying goes, garbage in garbage out. Training your algorithms with irrelevant features will affect the performance of your model. Also known as variable selection or attribute selection, choosing or engineering new features is often what separates the best performing models from the rest. 
# 
# Features selection can be both an art and science and it’s a very broad topic. In this blog we will focus on one of the methods you can use to identify the relevant features for your machine learning algorithm and implementing it in python using the scipy library. We will be using the chi square test of independence to identify the important features in the titanic dataset.
# 
# After reading this blog post you will be able to:
# 
# Gain an understanding of the chi-square test of independence
# Implement the chi-square test in python using scipy
# Utilize the chi-square test for feature selection
# 
# 
# 
# Chi Square Test
# The Chi-Square test of independence is a statistical test to determine if there is a significant relationship between 2 categorical variables.  In simple words, the Chi-Square statistic will test whether there is a significant difference in the observed vs the expected frequencies of both variables. 
# 
# 
# The Null hypothesis is that there is NO association between both variables. 
# 
# The Alternate hypothesis says there is evidence to suggest there is an association between the two variables. 
# 
# In our case, we will use the Chi-Square test to find which variables have an association with the Survived variable. If we reject the null hypothesis, it's an important variable to use in your model.
# 
# To reject the null hypothesis, the calculated P-Value needs to be below a defined threshold. Say, if we use an alpha of .05, if the p-value < 0.05 we reject the null hypothesis. If that’s the case, you should consider using the variable in your model.
# 
# Rules to use the Chi-Square Test:
# 
# 1. Variables are Categorical
# 
# 2. Frequency is at least 5
# 
# 3. Variables are sampled independently
# 
# Chi-Square Test in Python
# We will now be implementing this test in an easy to use python class we will call ChiSquare. Our class initialization requires a panda’s data frame which will contain the dataset to be used for testing. The Chi-Square test provides important variables such as the P-Value mentioned previously, the Chi-Square statistic and the degrees of freedom. Luckily you won’t have to implement the show functions as we will use the scipy implementation for this.

# <h1> Chi Square Formula  <h1>
# <br>
# <img src="https://latex.codecogs.com/gif.latex?\sum&space;=&space;(O-E)^2/E" title="\sum = (O-E)^2/E" />
# <br>
# <b> O = Observed Value and E= Expected Value<b>

# In[ ]:


import scipy.stats as stats
from scipy.stats import chi2_contingency

class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)

#Initialize ChiSquare Class
cT = ChiSquare(dftrain)

#Feature Selection
testColumns = ['Id', 'Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
       'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
for var in testColumns:
    cT.TestIndependence(colX=var,colY="Cover_Type" )  


# # id , Soil Type 7,8,15,25 has to be discarded from the final model

# ## So stay tuned we are with removed feature which were not necessary 
# ## what left boxcox tranform , depending upon what lamda will value will come , standard , log , sq , cubic , sqrt    transformation will be applied 
# ## 1.  After  that Some feature engg will be done to create more features within features like we polynomial but 
# ## before that we have move PCA , then poly , then KBEST after that DT , RF , GBT, ADA, XG
# 
# # So stay tuned more are coming some interesting visual and codes which DS family will like 
# # Pls upvote if u like 
# 

# # dftrain=dftrain.drop(['Soil_Type7','Soil_Type8','Soil_Type15','Soil_Type25'],axis=1)
# # we already dropping them  while not taking these featues when i was compressing the features , see code below 

# In[ ]:


dftrain.columns


# # As far of now we are dealing with these much columns since we saw that from chi-square predictions that some features are not required to predict cover type

# # Now removing outliers from the dftrain and duplicate outliers 

# In[ ]:


import itertools
#removing outliers
outlier_list=[]
for i in dftrain.columns:
    q1=np.percentile(dftrain.loc[:,i],25)
    q3=np.percentile(dftrain.loc[:,i],75)
    step=1.5*(q3-q1)
    print ("Data points considered outliers for the feature '{}':".format(i))

    outliers_rows = dftrain.loc[~((dftrain[i] >= q1 - step) & (dftrain[i] <= q3 + step)), :]
    outlier_list.append(list(outliers_rows.index))
    outliers = list(itertools.chain.from_iterable(outlier_list))

uniq_outlier=list(set(outliers))

#duplicate enteries removal

dup_outliers = list(set([x for x in outliers if outliers.count(x) > 1]))
print ('Outliers list:\n', uniq_outlier)
print( 'Length of outliers list:\n', len(uniq_outlier))

print ('Duplicate list:\n', dup_outliers)
print ('Length of duplicates list:\n', len(dup_outliers))

real_data=dftrain.drop(dftrain.index[dup_outliers]).reset_index(drop=True)

print (real_data.shape)


# # almost we are left with 58% percent of data very since we have to predict the lakh no records 
# # this no is very low for now ,  i will check my result in both cases as well 

# In[ ]:


soil_list = []
soil_not=[7,8,15,25]
for i in range(1, 41):
    if i not in soil_not:
       soil_list.append('Soil_Type' + str(i))

wilderness_area_list = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']

print(soil_list, "\n")
print(wilderness_area_list)


# # Removed some soil features which were not useful in prediction of forest cover 

# # Now I will compress the features so that i can do visualization and can drawout some inferences

# In[ ]:


def wilderness_compress(df):
    
    df[wilderness_area_list] = df[wilderness_area_list].multiply([1, 2, 3, 4], axis=1)
    df['Wilderness_Area'] = df[wilderness_area_list].sum(axis=1)
    return df


# In[ ]:


def soil_compress(df):
    
    df[soil_list] = df[soil_list].multiply([i for i in range(1, 37)], axis=1)
    df['Soil_Type'] = df[soil_list].sum(axis=1)
    return df


# In[ ]:


dftrain = wilderness_compress(dftrain)
dftrain = soil_compress(dftrain)

dftrain[['Wilderness_Area', 'Soil_Type']].head()


# In[ ]:


cols = dftrain.columns.tolist()
columns = cols[1:11] + cols[56:]

print("Useful columns: ", columns)

values = dftrain[columns]
labels = dftrain['Cover_Type']

print("Values: ", values.shape)
print("Labels: ", labels.shape)


# # AS if for now we have compressed the features and now we can go visualization using bokeh visualization , better than seaborn

# In[ ]:


import seaborn as sns
sns.set_style('whitegrid')
sns.set(rc={'figure.figsize':(11.7,8.27)})


# In[ ]:


ax = sns.countplot(labels, alpha=0.75)
ax.set(xlabel='Cover Type', ylabel='Number of labels')
plt.show()


# # No class imbalance issues 

# In[ ]:


ax = sns.distplot(dftrain['Elevation'], color='pink')
plt.show()


# # elevation is from 2260 to 3250 maximum values 

# In[ ]:


ax=sns.factorplot(x="Cover_Type",col="Wilderness_Area", data=dftrain , kind="count",size=6, aspect=.7,palette=['crimson','lightblue','yellow','green','orange','purple','black'])


# # So we can say that forest cover 1,2,5 and 7 are Area 1 
# # very less forest are in Areas 2
# # and Area 3 is more densely all types of cover except 4 
# # and Area 4 is more likely dense for 3,4 and 6 forest cover

# In[ ]:


ax = sns.violinplot(x="Cover_Type", y="Wilderness_Area", data=dftrain, inner=None)


# # analysis bring up like forest cover 1 is on 1 and 3 wilder areas , forest cover 2 so same
# # 1,2,5,7 forest cover are closey related to one another and 3 ,6 also to one another forest cover is on 4 areas , so we cans say that to predcit forest cover 4 areas 4 , for 1,2,5,7 we need 1,3 areas and for 3,6 we need 3, 4 areas 

# In[ ]:


ax = plt.scatter(x=dftrain['Horizontal_Distance_To_Hydrology'], y=dftrain['Vertical_Distance_To_Hydrology'], c=dftrain['Elevation'], cmap='jet')
plt.xlabel('Horizontal Distance')
plt.ylabel('Vertical Distance')
plt.title("Distance to Hydrology with Elevation")
plt.show()


# In[ ]:


ax = sns.countplot(dftrain['Soil_Type'], alpha=0.75)
ax.set(xlabel='Soil Type', ylabel='Count', title='Soil types - Count')
plt.show()


# In[ ]:


ax = sns.jointplot(x='Soil_Type', y='Cover_Type', data=dftrain, kind='kde', color='purple')
plt.show()


# # as we can see that most of the foret that is from 3 to 6 are more like to have soil type from 0 to 150 times 
# # forest cover 7 is with soil types of 1400 times , and 1 and 2 is soil types 250 to 800 times 

# In[ ]:


clean = dftrain[['Id', 'Cover_Type'] + columns]
clean.head()


# In[ ]:


y=clean.iloc[:,0]
clean=clean.drop(['Id','Cover_Type'],axis=1)
clean.info()


# # now we have clean data , now we will go for PCA And other Arrows annotations visualization too 

# In[ ]:


pca=PCA()
pca.fit_transform(clean)
ratio=pca.explained_variance_ratio_
plt.figure(figsize=(10,10))    
plt.plot(range(1,13),ratio,'-o',color='blue',alpha=0.75)
plt.ylim(0,1)
plt.xlim(0,10)
plt.grid(axis="both")
plt.xlabel('Principal Component')
plt.ylabel('percentages of variance in data set')
plt.show()


# In[ ]:


pca=PCA(n_components=2)

principal_comp=pd.DataFrame(pca.fit_transform(clean),index=clean.index,columns=['PC1','PC2'])
pca_load=pd.DataFrame(pca.fit(clean).components_.T,index=clean.columns,columns=['v1','v2'])
#plotting all states
for i in principal_comp.index:
    plt.annotate(i,(principal_comp.PC1.loc[i],principal_comp.PC2.loc[i]),ha='center')
#plotting reference lines
plt.figure(figsize=(15,15))
plt.hlines(0,-3.5,3.5,linestyle='dotted',color='green')
plt.vlines(0,-3.5,3.5,linestyle='dotted',color='green')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt1=plt.twinx().twiny()
#setting limits
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('eigen vectors',color='red')
#setting annotation
a=1.07
for i in pca_load.index:
    plt.annotate(i,(pca_load.v1.loc[i]*a,pca_load.v2.loc[i]*a),color='blue',ha='center')
    
#setting arrow    
for i,j in zip(range(1,13),['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']):    
    plt.arrow(0,0, pca_load.v1[i],pca_load.v2[i],color=j,shape='full',length_includes_head=True,head_width=.009,lw=3)

plt.show()


# # One more feature we can think of having euclidean distance of horizontal distance of water , fire and roadways since , they can bbe combined to form one by visualzing the PCA 

# In[ ]:


from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

def add_interactions(df):
    # Get feature names
    combos = list(combinations(list(df.columns), 2))
    colnames = list(df.columns) + ['_'.join(x) for x in combos]
    
    # Find interactions
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames
    
    # Remove interaction terms with all 0 values            
    noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
    df = df.drop(df.columns[noint_indicies], axis=1)
    
    return df


# In[ ]:


from sklearn.preprocessing import OneHotEncoder

A = add_interactions(clean)


# In[ ]:


from sklearn.model_selection  import train_test_split,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,accuracy_score
from scipy import stats


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(A,y,test_size=.30)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(x_train)
xtest = sc.transform(x_test)


# In[ ]:


pca=PCA(n_components=.95)
x_train=pca.fit_transform(xtrain)
x_test=pca.transform(xtest)
pca.explained_variance_


# # first baseline model is ready got accuracy of 83 % percent max

# In[ ]:





# In[ ]:





# In[ ]:




