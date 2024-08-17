#!/usr/bin/env python
# coding: utf-8

# <div align='center'><font size="7" color="#FF0000"><strong>AutoML</strong></font></div>
# <hr>
# <div align='left'><font size="3" color="#000000">  Machine learning is an application of artificial intelligence(AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. In machine learning supervised and unsupervised learning are the 2 major techniques.Supervised machine learning algorithms can apply what has been learned in the past to new data using labeled examples to predict future events where as unsupervised machine learning algorithms are used when the information used to train is neither classified nor labeled.</font></div>
# > <img style="float: centre;" src="https://analyticsindiamag.com/wp-content/uploads/2018/01/data-cleaning.png" width="450px"/>
# 
# <div align='left'><font size="3" color="#000000"> When applying machine learning models, we’d usually do data pre-processing,feature engineering,feature extraction and, feature selection. After this, we’d select the best algorithm and tune our parameters in order to obtain the best results.In order to reduce this burden AutoML Or Automatic machine learning tools are introduced. The main benefit of these tool is it reduce the burden of data cleaning, preprocessing. There are many AutoML packages(automl,AutoViML,PyCaret) available in python. Apart from this there are many automated visualization tools avaialble (sweetviz,AutoViz) which helps to understand the overview of the dataset. Below links are examples for automated visualization tools. </font></div>
# <hr>
# <div align='left'><font size="3"><a href="https://www.kaggle.com/nareshbhat/data-visualization-in-just-one-line-of-code" target="_blank">1.EDA in just two lines of code</a></div>
# <div align='left'><font size="3"><a href="https://www.kaggle.com/nareshbhat/eda-in-just-two-lines-of-code" target="_blank">2. Data visualization in just one line of code</a></div>
# <hr>   
# <div class="alert alert-info" ><font size="4"><strong>This kernel organized regression, classification, clustering and NLP at one platform with few lines of code using pycaret package</strong> </div>    
# <hr>
# <div align='left'><font size="6" color="#FF0000"><strong>About PyCaret</strong></font></div>
# <hr>
# > <img style="float: centre;" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOW3zspRu7Zyg3Xgbu34SZ4sSuS6tp9dvn8YLtsGG20QdW4tU&s" width="400px"/>
# 
# <div align='left'><font size="3" color="#000000"> PyCaret is an open source,customize, low-code Auto ML library in Python that aims to reduce the cycle time from hypothesis to insights and make data scientists more productive in their experiments and allows them to reach conclusions faster due to far less time spent in coding. PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks such as scikit-learn,XGBoost, Microsoft LightGBM, spaCy and many more.Through the use of PyCaret, the amount of time spent in coding experiments reduce drastically. </font></div>
# <hr>
# <div align='left'><font size="3" color="#000000"> PyCaret contains modules for some of the Machine learning task given below.</font></div>
# <div align='left'><font size="3" color="#000000"> 1. Classification</font></div>
# <div align='left'><font size="3" color="#000000"> 2. Regression</font></div>
# <div align='left'><font size="3" color="#000000"> 3. Clustering</font></div>
# <div align='left'><font size="3" color="#000000"> 4. Anomaly Detection</font></div>
# <div align='left'><font size="3" color="#000000"> 5. Natural Language Processing</font></div>
# <div align='left'><font size="3" color="#000000"> 6. Association Rule Mining</font></div>
# <hr>
# <div align='left'><font size="3" color="#000000"> Unique Features</font></div>
# 
# > <img style="float: centre;" src="https://miro.medium.com/max/875/1*jo9vPsQhQZmyXUhnrt9akQ.png" width="600px"/>
#   
# <hr>
# <div align='left'><font size="3" color="#000000"> PyCaret contains modules for some of the Machine learning task given below.</font></div>
# <div align='left'><font size="3" color="#000000"> 1. Imputing missing values.</font></div>
# <div align='left'><font size="3" color="#000000"> 2. Encode the categorical data.</font></div>
# <div align='left'><font size="3" color="#000000"> 3. Feature engineering.</font></div>
# <div align='left'><font size="3" color="#000000"> 4. Hyperparameter tuning.</font></div>
# <div align='left'><font size="3" color="#000000"> 5. Ensemble the models.</font></div>
# <div align='left'><font size="3" color="#000000"> 6. Explainability/Interpretation.</font></div>
# <div align='left'><font size="3" color="#000000"> 7. Perform end-to-end machine learning(Classification,Clustering,Regression).</font></div>
# <hr>
# 
# <div align='left'><font size="3"><a href="https://pycaret.org" target="_blank"> website and documentation of this Package</a></div>
# <hr>
# <div align='left'><font size="6" color="#FF0000">Installing PyCaret Package</font></div>
# <hr>
#   
#     
# 
#     

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install pycaret\n')


# <div align='left'><font size="6" color="#FF0000">This Notebook is organized as follows:</font></div>
# 
# * [<div align='left'><font size="4" color="#1E90FF">1. Regression</font></div>](#imports)
#     * <div align='left'><font size="3.5" color="#1E90FF">1.1 Setup the model</font></div>
#     * <div align='left'><font size="3.5" color="#1E90FF">1.2 Compare the model</font></div>
#     * <div align='left'><font size="3.5" color="#1E90FF">1.3 Create model</font></div>
#     * <div align='left'><font size="3.5" color="#1E90FF">1.4 Interpretation</font></div>
#     * <div align='left'><font size="3.5" color="#1E90FF">1.5 Ensembling</font></div> 
#         <div align='left'><font size="3" color="#1E90FF">* 1.5.1 stacking</font></div>
# - [<div align='left'><font size="4" color="#1E90FF">2. Classification</font></div>](#imports)
#     * <div align='left'><font size="3.5" color="#1E90FF">2.1 Setup the model</font></div>
#     * <div align='left'><font size="3.5" color="#1E90FF">2.2 Compare the model</font></div>
#     * <div align='left'><font size="3.5" color="#1E90FF">2.3 Create model</font></div>
#     * <div align='left'><font size="3.5" color="#1E90FF">2.4 Interpretation</font></div>
#     * <div align='left'><font size="3.5" color="#1E90FF">2.5 Ensembling</font></div> 
#         <div align='left'><font size="3" color="#1E90FF">* 2.5.1 blending</font></div>
#     * <div align='left'><font size="3.5" color="#1E90FF">2.6 Model Evaluation</font></div>            
# - [<div align='left'><font size="4" color="#1E90FF">3. Clustering</font></div>](#imports)
#     * <div align='left'><font size="3.5" color="#1E90FF">3.1 Setup the model</font></div>
#     * <div align='left'><font size="3.5" color="#1E90FF">3.2 Create model</font></div>
#         <div align='left'><font size="3" color="#1E90FF">* 3.2.1 K-mean Clustering</font></div>
#         <div align='left'><font size="3" color="#1E90FF">* 3.2.2 Hierarchical Clustering</font></div>   
#     * <div align='left'><font size="3.5" color="#1E90FF">3.3 Plot model</font></div>
#     * <div align='left'><font size="3.5" color="#1E90FF">3.4 Assign model</font></div>
# - [<div align='left'><font size="4" color="#1E90FF">4.Natural Language Processing</font></div>](#imports)
#     * <div align='left'><font size="3.5" color="#1E90FF">4.1 Converting the text data</font></div>
#     * <div align='left'><font size="3.5" color="#1E90FF">4.2 Setup the model</font></div>
#         <div align='left'><font size="3" color="#1E90FF">4.3 Compare the model</font></div>
# - [<div align='left'><font size="4" color="#1E90FF">5. Conclusion</font></div>](#imports)

# <div align='left'><font size="6" color="#FF0000"><strong>1.Regression</strong></font></div>
# <hr>
# <div align='left'><font size="3" color="#000000">  Here we use house-prices-advanced-regression-techniques dataset to perform regression. </font></div>
# <hr>
# > <img style="float: centre;" src="https://miro.medium.com/max/1024/1*Juv1bpp5--0Fl8cA4EmTPw.jpeg" width="450px"/>

# In[2]:


from pycaret.regression import *
import pandas as pd


# In[3]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test.head()


# <div align='left'><font size="5" color="#FF0000">Setup the model</font></div>
# <hr>

# In[4]:


reg = setup(data = train,target = 'SalePrice', numeric_imputation = 'mean',normalize = True,
             ignore_features = ['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','Utilities'],pca=True,
    pca_method='linear',pca_components=30,silent = True,session_id = 3650)


# <div align='left'><font size="5" color="#FF0000">Compare Model</font></div>
# <hr>

# In[5]:


compare_models(exclude = ['tr'] , turbo = True) 


# <div align='left'><font size="5" color="#FF0000">Create Model</font></div>
# <hr>

# In[6]:


cb = create_model('catboost')


# <div align='left'><font size="5" color="#FF0000">Interpretation</font></div>
# <hr>

# In[7]:


interpret_model(cb)


# In[8]:


prediction = predict_model(cb, data = test)


# In[9]:


output_reg = pd.DataFrame({'Id': test.Id, 'SalePrice': prediction.Label})
output_reg.to_csv('submission.csv', index=False)


# In[10]:


output_reg.head()


# <div align='left'><font size="3" color="#00000">Website and documentation of this Package:</font><a href="https://pycaret.org/regression" target="_blank"> https://pycaret.org/regression</a></div>

# <div align='left'><font size="6" color="#FF0000"><strong>2.Classification</strong></font></div>
# <hr>
# <div align='left'><font size="3" color="#000000">  Here we use Titanic dataset to perform classification. </font></div>
# <hr>
# > <img style="float: centre;" src="https://humansofdata.atlan.com/wp-content/uploads/2016/07/RMS_Titanic_3.jpg" width="450px"/>

# In[11]:


import pandas as pd 
from pycaret.classification import *


# In[12]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
test.head()


# <div align='left'><font size="5" color="#FF0000">Setup the model</font></div>
# <hr>

# In[13]:


classification_setup = setup(data= train, target='Survived',remove_outliers=True,normalize=True,normalize_method='robust',
                            ignore_features= ['Name'], silent = True,session_id = 6563)


# <div align='left'><font size="5" color="#FF0000">Compare Model</font></div>
# <hr>

# In[14]:


compare_models(exclude = ['lda'])


# In[15]:


dt = create_model('dt')


# <div align='left'><font size="5" color="#FF0000">Interpretation</font></div>
# <hr>

# In[16]:


plot_model(estimator = dt, plot = 'auc')


# In[17]:


plot_model(estimator = dt, plot = 'feature')


# <div align='left'><font size="5" color="#FF0000">Ensembling</font></div>
# <hr>

# In[18]:


rd = create_model('ridge');      
lgm  = create_model('lightgbm');            

#blending 3 models
blend = blend_models(estimator_list=[lgm,dt,rd])


# In[19]:


optimize_threshold(lgm, true_negative = 500, false_negative = -2000)


# <div align='left'><font size="5" color="#FF0000">Model Evaluation</font></div>
# <hr>

# In[20]:


plot_model(estimator = blend, plot = 'confusion_matrix')


# In[21]:


pred = predict_model(blend, data = test)


# In[22]:


output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred.Label})
output.to_csv('submission.csv', index=False)


# In[23]:


output.head()


# <div align='left'><font size="3" color="#00000">Website and documentation of this Package:</font><a href="https://pycaret.org/classification" target="_blank"> https://pycaret.org/classification</a></div>

# <div align='left'><font size="6" color="#FF0000"><strong>3.Clustering</strong></font></div>
# <hr>
# <div align='left'><font size="3" color="#000000">  Here we use mall customer dataset to perform Clustering.  </font></div>
# <hr>
# 
# > <img style="float: centre;" src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRPwRj1l9eIPJlLHFTQmVxOFdV8A-iRJpzHHw&usqp=CAU" width="450px"/>

# In[24]:


import pandas as pd
from pycaret.clustering import *
data = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
data.head()


# <div align='left'><font size="5" color="#FF0000">Setup the model</font></div>
# <hr>

# In[25]:


exp_clu = setup(data)


# <div align='left'><font size="5" color="#FF0000">K-means Clustering</font></div>
# <hr>

# In[26]:


kmeans = create_model('kmeans')
print(kmeans)


# <div align='left'><font size="5" color="#FF0000">Plot the model</font></div>
# <hr>

# In[27]:


plot_model(kmeans,plot = 'elbow')


# In[28]:


plot_model(kmeans,plot = 'tsne')


# In[29]:


plot_model(kmeans,plot = 'distribution')


# <div align='left'><font size="5" color="#FF0000">Assign model</font></div>
# <hr>

# In[30]:


kmeans_df = assign_model(kmeans)
kmeans_df.head()


# <div align='left'><font size="5" color="#FF0000">Hierarchical Clustering</font></div>
# <hr>

# In[31]:


hierarchical = create_model('hclust')


# <div align='left'><font size="5" color="#FF0000">Plot the model</font></div>
# <hr>

# In[32]:


plot_model(hierarchical,plot='cluster',label = True )


# <div align='left'><font size="5" color="#FF0000">Assign model</font></div>
# <hr>

# In[33]:


hierarchical_df = assign_model(hierarchical)
hierarchical_df.head()


# <div align='left'><font size="3" color="#00000">Website and documentation of this Package:</font><a href="https://pycaret.org/clustering" target="_blank"> https://pycaret.org/clustering</a></div>

# <div align='left'><font size="6" color="#FF0000"><strong>4.Natural Language Processing</strong></font></div>
# <hr>
# 
# <div align='left'><font size="3" color="#000000"> Here we use text message classification.  </font></div>
# <hr>
# 
# > <img style="float: centre;" src="https://miro.medium.com/max/1840/1*hsyCZOYoGrX6BJsj4Lgrhg.png" width="450px"/>

# In[34]:


from pycaret.nlp import *
import pandas as pd


# In[35]:


data = pd.read_csv('../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')
data.head()


# <div align='left'><font size="5" color="#FF0000">Converting Text Data</font></div>
# <hr>

# In[36]:


nlp = setup(data = data, target = 'Message', session_id = 1)


# In[37]:


lda = create_model('lda', multi_core = True)


# In[38]:


lda_data = assign_model(lda)
lda_data.head()


# In[39]:


evaluate_model(lda)


# In[40]:


from pycaret.classification import *


# <div align='left'><font size="5" color="#FF0000">Setup the model</font></div>
# <hr>

# In[41]:


model = setup(data = lda_data, target = 'Category',ignore_features=['Message','Dominant_Topic','Perc_Dominant_Topic'])


# <div align='left'><font size="5" color="#FF0000">Compare model</font></div>
# <hr>

# In[42]:


compare_models()


# <div align='left'><font size="3" color="#00000">Website and documentation of this Package:</font><a href="https://pycaret.org/nlp" target="_blank"> https://pycaret.org/nlp</a></div>

# <div align='left'><font size="6" color="#FF0000"><strong>5.Conclusion</strong></font></div>
# <hr>
# <div align='left'><font size="3" color="#000000">  PyCaret is simple and easy to use. It helps right from the start of data preparation to till the end of model analysis and deployment.It has over 70 ready-to-use open source algorithms,over 40 interactive visualizations to analyze machine learning models and over 25 pre-processing techniques that are fully orchestrated. Future releases will include Time Series Modeling, Recommender System and Deep Learning modules. I’ve personally found PyCaret to be quite useful for generating quick results. Apart from regression, clustering,classification and natural language processing, pycaret contain module for <strong>Anomly Detection</strong> and <strong>Association Rule Mining</strong></font></div>
# <hr>
# 
# <div align='left'><font size="6" color="#FF0000"><strong>References:</strong></font></div>
# <hr>
# <div align='left'><font size="3"><a href="https://pycaret.org" target="_blank">1. PyCaret</a></div>
# <div align='left'><font size="3"><a href="https://www.kdnuggets.com/2020/07/5-things-pycaret.html" target="_blank">2. 5 Things You Don’t Know About PyCaret</a></div>
# <div align='left'><font size="3"><a href="https://towardsdatascience.com/announcing-pycaret-an-open-source-low-code-machine-learning-library-in-python-4a1f1aad8d46" target="_blank">3. Announcing PyCaret 1.0.0</a></div>
# <div align='left'><font size="3"><a href="https://www.analyticsvidhya.com/blog/2020/05/pycaret-machine-learning-model-seconds" target="_blank">4. Build your Machine Learning Model in Seconds</a></div>
# <hr>
# 
#  <div class="alert alert-info" ><font size="4"><strong>If you found this helpful, <strong>Please upvote!</strong> Happy Learning😊</strong> </div> 

# In[ ]:




