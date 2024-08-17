#!/usr/bin/env python
# coding: utf-8

# ![](https://875478.smushcdn.com/1946825/wp-content/uploads/2019/03/quick_overview_autoML_1600x700_web.jpg?lossy=1&strip=1&webp=1)

# # Index: 
# ---
# 1. [Basics of AutoML](https://www.kaggle.com/soham1024/know-about-different-automl-frameworks?scriptVersionId=43751621#Firstly,-let's-know-about-AutoML)                 
#     a. [When to use AutoML](https://www.kaggle.com/soham1024/know-about-different-automl-frameworks?scriptVersionId=43751621#When-to-use-AutoML:-classify,-regression,-&-forecast)          
#     b. [How AutoML Works](https://www.kaggle.com/soham1024/know-about-different-automl-frameworks?scriptVersionId=43751621#How-AutoML-works)           
#     c. [Feature Engineering](https://www.kaggle.com/soham1024/know-about-different-automl-frameworks?scriptVersionId=43751621#Feature-engineering)             
#     d. [Ensemble Models](https://www.kaggle.com/soham1024/know-about-different-automl-frameworks?scriptVersionId=43751621#Ensemble-models)                    
#     e. [Pros and Cons](https://www.kaggle.com/soham1024/know-about-different-automl-frameworks?scriptVersionId=43751621#Pros-and-cons)        
# 2. [Types of AutoML](https://www.kaggle.com/soham1024/know-about-different-automl-frameworks?scriptVersionId=43751621#Types-of-AutoML)
# 3. [Examples of AutoML](https://www.kaggle.com/soham1024/know-about-different-automl-frameworks?scriptVersionId=43751621#Examples)    
#     a. [Google AI](https://www.kaggle.com/soham1024/know-about-different-automl-frameworks?scriptVersionId=43751621#Example-on-Google-Cloud-AutoML)    
#     b. [H2O.Ai](https://www.kaggle.com/soham1024/know-about-different-automl-frameworks?scriptVersionId=43751621#Example-on-H2O.Ai)     
#     c. [Auto-Sklearn](https://www.kaggle.com/soham1024/know-about-different-automl-frameworks?scriptVersionId=43751621#Auto-Sklearnhttps://www.kaggle.com/soham1024/know-about-different-automl-frameworks?scriptVersionId=43751621#Auto-Sklearn)    
#     d. [ML Box](https://www.kaggle.com/soham1024/know-about-different-automl-frameworks?scriptVersionId=43751621#ML-Box)       
#     e. [Auto-Keras](https://www.kaggle.com/soham1024/know-about-different-automl-frameworks?scriptVersionId=43751621#Auto-Keras)        
#     f. [TPOT](https://www.kaggle.com/soham1024/know-about-different-automl-frameworks?scriptVersionId=43751621#TPOT)         
#     g. Pycaret(Coming soon)
# 4. [Referances](https://www.kaggle.com/soham1024/know-about-different-automl-frameworks?scriptVersionId=43751621#Referances:)     

# ---
# 
# # Firstly, Let's Know About AutoML
# 
# ---

# Automated machine learning, also referred to as automated ML or AutoML, is the process of automating the time consuming, iterative tasks of machine learning model development. It allows data scientists, analysts, and developers to build ML models with high scale, efficiency, and productivity all while sustaining model quality. 
# 
# Traditional machine learning model development is resource-intensive, requiring significant domain knowledge and time to produce and compare dozens of models. With automated machine learning, you'll accelerate the time it takes to get production-ready ML models with great ease and efficiency.

# ![](https://miro.medium.com/max/2400/1*Vubgyj96KsskE6eRuxq58w.png)

# ---
# 
# # When to use AutoML: classify, regression, & forecast
# 
# ---

# Apply automated ML when you want Azure Machine Learning to train and tune a model for you using the target metric you specify. Automated ML democratizes the machine learning model development process, and empowers its users, no matter their data science expertise, to identify an end-to-end machine learning pipeline for any problem.
# 
# Data scientists, analysts, and developers across industries can use automated ML to:
# 
#     Implement ML solutions without extensive programming knowledge
#     Save time and resources
#     Leverage data science best practices
#     Provide agile problem-solving
# 

# ---
# 
# > ## Classification
# 
# ---
# 
# ![](https://miro.medium.com/max/1200/1*PM4dqcAe6N7kWRpXKwgWag.png)
# 
# ---
# 
# Classification is a common machine learning task. Classification is a type of supervised learning in which models learn using training data, and apply those learnings to new data. Azure Machine Learning offers featurizations specifically for these tasks, such as deep neural network text featurizers for classification
# 
# The main goal of classification models is to predict which categories new data will fall into based on learnings from its training data. Common classification examples include fraud detection, handwriting recognition, and object detection
# 
# ---

# ---
# 
# > ## Regression
# 
# ---
# 
# ![](https://miro.medium.com/max/4096/1*1sAafDO7V5HT5MzpknGw7w.png)
# 
# ---
# 
# Similar to classification, regression tasks are also a common supervised learning task
# 
# Different from classification where predicted output values are categorical, regression models predict numerical output values based on independent predictors. In regression, the objective is to help establish the relationship among those independent predictor variables by estimating how one variable impacts the others. For example, automobile price based on features like, gas mileage, safety rating, etc.
# 
# ---

# ---
# 
# > ## Time-series Forecasting
# 
# ---
# 
# ![](https://machinelearningblogcom.files.wordpress.com/2018/01/153775-636270895711868987-16x9.jpg?w=1400)
# 
# ---
# 
# Building forecasts is an integral part of any business, whether it's revenue, inventory, sales, or customer demand. You can use automated ML to combine techniques and approaches and get a recommended, high-quality time-series forecast.   
# 
# An automated time-series experiment is treated as a multivariate regression problem. Past time-series values are "pivoted" to become additional dimensions for the regressor together with other predictors. This approach, unlike classical time series methods, has an advantage of naturally incorporating multiple contextual variables and their relationship to one another during training. Automated ML learns a single, but often internally branched model for all items in the dataset and prediction horizons. More data is thus available to estimate model parameters and generalization to unseen series becomes possible.
# 
# Advanced forecasting configuration includes:
# 
#     holiday detection and featurization
#     time-series and DNN learners (Auto-ARIMA, Prophet, ForecastTCN)
#     many models support through grouping
#     rolling-origin cross validation
#     configurable lags
#     rolling window aggregate features
# 
# ---

# ---
# 
# # How AutoML Works
# 
# ---
# 
# ![](https://9to5google.com/wp-content/uploads/sites/4/2019/04/google-cloud-automl-updates.jpg?quality=82&strip=all&w=1600)
# 
# ---

# During training, Azure Machine Learning creates a number of pipelines in parallel that try different algorithms and parameters for you. The service iterates through ML algorithms paired with feature selections, where each iteration produces a model with a training score. The higher the score, the better the model is considered to "fit" your data. It will stop once it hits the exit criteria defined in the experiment.

# you can design and run your automated ML training experiments with these steps:
# 
#     Identify the ML problem to be solved: classification, forecasting, or regression
# 
#     Choose whether you want to use the Python SDK or the studio web experience
# 
#     Specify the source and format of the labeled training data: Numpy arrays or Pandas dataframe
# 
#     Configure the compute target for model training
# 
#     Configure the automated machine learning parameters that determine how many iterations over different models, hyperparameter settings, advanced preprocessing/featurization, and what metrics to look at when determining the best model.
# 
#     Submit the training run.
# 
#     Review the results
# 

# > ![](https://docs.microsoft.com/en-us/azure/machine-learning/media/concept-automated-ml/automl-concept-diagram2.png)

# You can also inspect the logged run information, which contains metrics gathered during the run

# ![](https://miro.medium.com/max/5060/1*SV9MUnhPSxQFt37wOr2iyw.png)

# ---
# 
# # Feature Engineering
# 
# ---
# 
# ![](https://miro.medium.com/max/1200/1*K6ctE0RZme0cqMtknrxq8A.png)
# 
# ---
# 
# Feature engineering is the process of using domain knowledge of the data to create features that help ML algorithms learn better. In Azure Machine Learning, scaling and normalization techniques are applied to facilitate feature engineering. Collectively, these techniques and feature engineering are referred to as featurization.
# 
# For automated machine learning experiments, featurization is applied automatically, but can also be customized based on your data
# 
# > Automated machine learning featurization steps (feature normalization, handling missing data, converting text to numeric, etc.) become part of the underlying model. When using the model for predictions, the same featurization steps applied during training are applied to your input data automatically.

# ---
# 
# # Ensemble Models
# 
# ---
# 
# ![](https://miro.medium.com/max/2556/1*P0ns6A56MtpGFMQ2g47IYA.png)
# 
# ---
# 
# Automated machine learning supports ensemble models, which are enabled by default. Ensemble learning improves machine learning results and predictive performance by combining multiple models as opposed to using single models. The ensemble iterations appear as the final iterations of your run. Automated machine learning uses both voting and stacking ensemble methods for combining models:
# 
#     Voting: predicts based on the weighted average of predicted class probabilities (for classification tasks) or predicted regression targets (for regression tasks).
#     Stacking: stacking combines heterogenous models and trains a meta-model based on the output from the individual models. The current default meta-models are LogisticRegression for classification tasks and ElasticNet for regression/forecasting tasks.
# 
# The [Caruana ensemble selection algorithm](http://www.niculescu-mizil.org/papers/shotgun.icml04.revised.rev2.pdf) with sorted ensemble initialization is used to decide which models to use within the ensemble. At a high level, this algorithm initializes the ensemble with up to five models with the best individual scores, and verifies that these models are within 5% threshold of the best score to avoid a poor initial ensemble. Then for each ensemble iteration, a new model is added to the existing ensemble and the resulting score is calculated. If a new model improved the existing ensemble score, the ensemble is updated to include the new model.

# ---
# 
# # Pros and Cons
# 
# ---
# 
# ![](https://miro.medium.com/max/2896/1*yznfOpXlLD_kqPW-5ggKsg.png)
# 
# ---

# ---
# 
# # Types of AutoML
# 
# ---

# ![](https://miro.medium.com/max/4180/1*d7n1wOYUE3e-fE6NVqTryg.png)

# ---
# ---
# 
# # Examples
# 
# ---

# ---
# 
# > # Example on Google Cloud AutoML
# 
# ---

# ![](https://helpdev.eu/wp-content/uploads/2018/11/GoogleAdaNet.jpg)
# 
# ---
# 
# Google launched the Google Auto ML framework which integrates the powers of neural network architecture. Its graphical user interface (GUI) is simple to use for model processing models which makes Google Cloud Auto ML useful for citizen developers and citizen data scientists who have limited ML knowledge to process ML-models development.
# 
# However, Google Cloud Auto ML is a paid platform, which makes it feasible to use if only for commercial projects. Besides this Auto ML toolkit is available free of charge for research purposes throughout the year.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from datetime import datetime

from sklearn.model_selection import train_test_split

from google.cloud import storage
from google.cloud import automl_v1beta1 as automl

# workaround to fix gapic_v1 error
from google.api_core.gapic_v1.client_info import ClientInfo

from automlwrapper import AutoMLWrapper


# This notebook utilizes a utility script that wraps much of the AutoML Python client library, to make the code in this notebook easier to read. Feel free to check out the utility for all the details on how we are calling the underlying AutoML Client Library!

# In[ ]:


# Set your own values for these. bucket_name should be the project_id + '-lcm'.
PROJECT_ID = 'cloudml-demo'
bucket_name = 'cloudml-demo-lcm'

region = 'us-central1' # Region must be us-central1
dataset_display_name = 'kaggle_tweets'
model_display_name = 'kaggle_starter_model1'

storage_client = storage.Client(project=PROJECT_ID)

# adding ClientInfo here to get the gapic_v1 call in place
client = automl.AutoMlClient(client_info=ClientInfo())

print(f'Starting AutoML notebook at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


nlp_train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
nlp_test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
def callback(operation_future):
    result = operation_future.result()


# In[ ]:


nlp_train_df.tail()


# ### Data spelunking
# How often does 'fire' come up in this dataset?

# In[ ]:


nlp_train_df.loc[nlp_train_df['text'].str.contains('fire', na=False, case=False)]


# Does the presence of the word 'fire' help determine whether the tweets here are real or false?

# In[ ]:


nlp_train_df.loc[nlp_train_df['text'].str.contains('fire', na=False, case=False)].target.value_counts()


# ### GCS upload/download utilities
# These functions make upload and download of files from the kernel to Google Cloud Storage easier. This is needed for AutoML

# In[ ]:


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket. https://cloud.google.com/storage/docs/ """
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print('File {} uploaded to {}'.format(
        source_file_name,
        'gs://' + bucket_name + '/' + destination_blob_name))
    
def download_to_kaggle(bucket_name,destination_directory,file_name,prefix=None):
    """Takes the data from your GCS Bucket and puts it into the working directory of your Kaggle notebook"""
    os.makedirs(destination_directory, exist_ok = True)
    full_file_path = os.path.join(destination_directory, file_name)
    blobs = storage_client.list_blobs(bucket_name,prefix=prefix)
    for blob in blobs:
        blob.download_to_filename(full_file_path)


# In[ ]:


bucket = storage.Bucket(storage_client, name=bucket_name)
if not bucket.exists():
    bucket.create(location=region)


# ### Export to CSV and upload to GCS

# In[ ]:


# Select the text body and the target value, for sending to AutoML NL
nlp_train_df[['text','target']].to_csv('train.csv', index=False, header=False) 


# In[ ]:


nlp_train_df[['id','text','target']].head()


# In[ ]:


training_gcs_path = 'uploads/kaggle_getstarted/full_train.csv'
upload_blob(bucket_name, 'train.csv', training_gcs_path)


# ## Create our class instance

# In[ ]:


amw = AutoMLWrapper(client=client, 
                    project_id=PROJECT_ID, 
                    bucket_name=bucket_name, 
                    region='us-central1', 
                    dataset_display_name=dataset_display_name, 
                    model_display_name=model_display_name)
       


# ## Create (or retreive) dataset
# Check to see if this dataset already exists. If not, create it

# In[ ]:


print(f'Getting dataset ready at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')
if not amw.get_dataset_by_display_name(dataset_display_name):
    print('dataset not found')
    amw.create_dataset()
    amw.import_gcs_data(training_gcs_path)

amw.dataset
print(f'Dataset ready at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')


# ## Kick off the training for the model
# And retrieve the training info after completion.      
# Start model deployment.

# In[ ]:


print(f'Getting model trained at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')

if not amw.get_model_by_display_name():
    print(f'Training model at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')
    amw.train_model()

print(f'Model trained. Ensuring model is deployed at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')
amw.deploy_model()
amw.model
print(f'Model trained and deployed at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')


# In[ ]:


amw.model_full_path


# ## Prediction
# Note that prediction will not run until deployment finishes, which takes a bit of time.
# However, once you have your model deployed, this notebook won't re-train the model, thanks to the various safeguards put in place. Instead, it will take the existing (trained) model and make predictions and generate the submission file.

# In[ ]:


nlp_test_df.head()


# In[ ]:


print(f'Begin getting predictions at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')

# Create client for prediction service.
prediction_client = automl.PredictionServiceClient()
amw.set_prediction_client(prediction_client)

predictions_df = amw.get_predictions(nlp_test_df, 
                                     input_col_name='text', 
#                                      ground_truth_col_name='target', # we don't have ground truth in our test set
                                     limit=None, 
                                     threshold=0.5,
                                     verbose=False)

print(f'Finished getting predictions at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')


# ## (optional) Undeploy model
# Undeploy the model to stop charges

# In[ ]:


amw.undeploy_model()


# ## Create submission output

# In[ ]:


predictions_df.head()


# In[ ]:


submission_df = pd.concat([nlp_test_df['id'], predictions_df['class']], axis=1)
submission_df.head()


# In[ ]:


# predictions_df['class'].iloc[:10]
# nlp_test_df['id']


# In[ ]:


submission_df = submission_df.rename(columns={'class':'target'})
submission_df.head()


# ---
# 
# > # Example on H2O.Ai
# 
# 
# ---
# 
# ![](https://analyticsindiamag.com/wp-content/uploads/2020/08/2020-08-20-2.jpg)
# 
# ---
# 
# This is an open-source, memory inclusive and distributed machine learning platform to build supervised and unsupervised machine learning models. It also includes a user-friendly UI platform called Flow where you can create these models. 
# 
# H2O AutoML framework is best suited to those who are searching for deep learning mechanisms. H2O AutoML can perform many tasks which requires many lines of code at the simultaneously.
# 
# H2O AutoML supports both traditional neural networks and machine learning models. It is especially suitable for developers who want to automate deep learning.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

test = pd.read_csv("../input/hackerearth-ml-challenge-pet-adoption/test.csv")
train = pd.read_csv("../input/hackerearth-ml-challenge-pet-adoption/train.csv")

#Train
train['issue_date'] = pd.to_datetime(train['issue_date'])
train['listing_date'] = pd.to_datetime(train['listing_date'])
train['duration'] = (train['listing_date'] - train['issue_date']).dt.days  
train = train.drop(['pet_id','issue_date','listing_date'],axis=1)


#Test
test['issue_date'] = pd.to_datetime(test['issue_date'])
test['listing_date'] = pd.to_datetime(test['listing_date'])
test['duration'] = (test['listing_date'] - test['issue_date']).dt.days  
test = test.drop(['pet_id','issue_date','listing_date'],axis=1)



#Train
from sklearn.preprocessing import LabelEncoder
train['color_number'] = LabelEncoder().fit_transform(train['color_type'])

#Test
test['color_number'] = LabelEncoder().fit_transform(test['color_type'])

#Train
train = train.fillna(-1)
#Test
test = test.fillna(-1)

#train
info_1 = pd.DataFrame()
info_1['length(m)'] = [np.percentile(train['length(m)'],25*i) for i in range(1,4)]
info_1['height(cm)'] = [np.percentile(train['height(cm)'],25*i) for i in range(1,4)]
info_1['duration'] = [np.percentile(train['duration'],25*i) for i in range(1,4)]
info_1

#test
info_2 = pd.DataFrame()
info_2['length(m)'] = [np.percentile(test['length(m)'],25*i) for i in range(1,4)]
info_2['height(cm)'] = [np.percentile(test['height(cm)'],25*i) for i in range(1,4)]
info_2['duration'] = [np.percentile(test['duration'],25*i) for i in range(1,4)]
info_2

#Train
info_1.loc[3] = [2.5*info_1.loc[0,column] - 1.5*info_1.loc[2,column] for column in info_1.columns]
info_1.loc[4] = [2.5*info_1.loc[2,column] - 1.5*info_1.loc[0,column] for column in info_1.columns]
info_1

#test
info_2.loc[3] = [2.5*info_2.loc[0,column] - 1.5*info_2.loc[2,column] for column in info_2.columns]
info_2.loc[4] = [2.5*info_2.loc[2,column] - 1.5*info_2.loc[0,column] for column in info_2.columns]
info_2

#train
def range_part_train(column,value):
    if value > info_1.loc[4,column]:
        return 5
    elif value > info_1.loc[2,column]:
        return 4
    elif value > info_1.loc[1,column]:
        return 3
    elif value > info_1.loc[0,column]:
        return 2
    elif value > info_1.loc[3,column]:
        return 1
    else:
        return 0

#test
def range_part_test(column,value):
    if value > info_2.loc[4,column]:
        return 5
    elif value > info_2.loc[2,column]:
        return 4
    elif value > info_2.loc[1,column]:
        return 3
    elif value > info_2.loc[0,column]:
        return 2
    elif value > info_2.loc[3,column]:
        return 1
    else:
        return 0

#train
from tqdm import tqdm
tqdm.pandas()
train['length_range'] = train['length(m)'].progress_apply(lambda x:range_part_train('length(m)',x))
train['height_range'] = train['height(cm)'].progress_apply(lambda x:range_part_train('height(cm)',x))
train['duration_range'] = train['duration'].progress_apply(lambda x:range_part_train('duration',x))

#test
tqdm.pandas()
test['length_range'] = test['length(m)'].progress_apply(lambda x:range_part_test('length(m)',x))
test['height_range'] = test['height(cm)'].progress_apply(lambda x:range_part_test('height(cm)',x))
test['duration_range'] = test['duration'].progress_apply(lambda x:range_part_test('duration',x))




X_train = train.drop(['breed_category','pet_category','color_type'], axis=1)
Y_train = train["breed_category"]
Z_train = train["pet_category"]

X_test=test.drop(['color_type'], axis=1)


# ### Now Preprocessing is Done. 

# In[ ]:


# # installing Automl
# !pip install automl


# In[ ]:


# importing 
import h2o
from h2o.automl import H2OAutoML

# Forming H2O Friendly Dataframe
h2o.init()
X_y_train_h = h2o.H2OFrame(pd.concat([X_train, Y_train], axis='columns'))
X_y_train_h['breed_category'] = X_y_train_h['breed_category'].asfactor()
#  the target column should have categorical type for classification tasks
#   (numerical type for regression tasks)

X_test_h = h2o.H2OFrame(X_test)

X_y_train_h.describe()


# In[ ]:


aml = H2OAutoML(
    max_runtime_secs=(3600*5),  # 5 hours
    max_models=None,  # no limit
    seed=42
)


# In[ ]:


# Defining Feature Columns
feature_cols = ['condition','length_range','height_range','duration_range','color_number','X1','X2']


# In[ ]:


get_ipython().run_cell_magic('time', '', '\naml.train(\n    x=feature_cols,\n    y=\'breed_category\',\n    training_frame=X_y_train_h\n)\n\nlb = aml.leaderboard\nmodel_ids = list(lb[\'model_id\'].as_data_frame().iloc[:,0])\nout_path = "."\n\nfor m_id in model_ids:\n    mdl = h2o.get_model(m_id)\n    h2o.save_model(model=mdl, path=out_path, force=True)\n\nh2o.export_file(lb, os.path.join(out_path, \'aml_leaderboard.h2o\'), force=True)\n')


# In[ ]:


# Checking Leaderboard for the Models
lb = h2o.automl.get_leaderboard(aml, extra_columns = 'ALL')
lb


# In[ ]:


# finding Prediction from the best fitted model
pred = aml.leader.predict(X_test_h)


# ---
# > # Example on Auto-Sklearn
# 
# ---
# 
# ![](https://www.automl.org/wp-content/uploads/2016/08/auto-sklearn-5.png)
# 
# ---
# 
# Auto Sklearn is an automated machine learning toolkit based on Bayesian optimization, meta-learning, ensemble construction. It frees a machine learning user from algorithm selection and hyperparameter tuning.
# 
# The Auto Sklearn, AutoML package includes 15 classification algorithms besides 14 for feature pre-processing which defines the right algorithm to optimise parameter accuracy at a precision level of more than 0.98. Auto Sklean translates well for small and medium datasets, however, developers face a hiccup with dealing with large datasets.

# In[ ]:


get_ipython().system('pip install auto-sklearn')


# In[ ]:


import autosklearn.classification

import sklearn.model_selection

import sklearn.datasets

import sklearn.metrics

X, y = sklearn.datasets.load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

automl = autosklearn.classification.AutoSklearnClassifier()

automl.fit(X_train, y_train)

y_hat = automl.predict(X_test)

print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))


# ---
# 
# > # Example on ML Box
# 
# ---
# 
# ![](https://mlbox.readthedocs.io/en/latest/_images/logo.png)
# 
# ---
# 
# ML Box is a data Python based library offering the features of read, pre-process, clean and format data with an option to choose specific features and detect a leak.ML Box Auto ML toolkit can classify and regress state-of-the-art models for predictions and model interpreting.
# 
# ML Box also offers developers with data preparation, model selection and hyper Parameter Search, however this AutoML toolkit is more suitable for the Linux operating systems. Windows and Mac users can experience some difficulties while installing ML Box

# In[ ]:


from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *


# In[ ]:


paths = ["../input/titanic/train.csv","../input/titanic/test.csv"]
target_name = "Survived"


# In[ ]:


rd = Reader(sep = ",")
df = rd.train_test_split(paths, target_name)   #reading and preprocessing (dates, ...)


# In[ ]:


dft = Drift_thresholder()
df = dft.fit_transform(df)   #removing non-stable features (like ID,...)


# In[ ]:


opt = Optimiser(scoring = "accuracy", n_folds = 5)


# In[ ]:


# LightGBM

space = {
    
        'est__strategy':{"search":"choice",
                                  "space":["LightGBM"]},    
        'est__n_estimators':{"search":"choice",
                                  "space":[150]},    
        'est__colsample_bytree':{"search":"uniform",
                                  "space":[0.8,0.95]},
        'est__subsample':{"search":"uniform",
                                  "space":[0.8,0.95]},
        'est__max_depth':{"search":"choice",
                                  "space":[5,6,7,8,9]},
        'est__learning_rate':{"search":"choice",
                                  "space":[0.07]} 
    
        }

params = opt.optimise(space, df,15)


# In[ ]:


prd = Predictor()
prd.fit_predict(params, df)


# 
# 
# But you can also tune the whole Pipeline ! Indeed, you can choose:
# 
#     different strategies to impute missing values
#     different strategies to encode categorical features (entity embeddings, ...)
#     different strategies and thresholds to select relevant features (random forest feature importance, l1 regularization, ...)
#     to add stacking meta-features !
#     different models and hyper-parameters (XGBoost, Random Forest, Linear, ...)
# 
# 

# In[ ]:


submit = pd.read_csv("../input/gendermodel.csv",sep=',')
preds = pd.read_csv("save/"+target_name+"_predictions.csv")

submit[target_name] =  preds[target_name+"_predicted"].values

submit.to_csv("mlbox.csv", index=False)


# ---
# 
# > # Example on Auto-Keras
# 
# ---
# 
# ![](https://pyimagesearch.com/wp-content/uploads/2019/01/autokeras_header.jpg)
# 
# ---
# 
# AutoKeras an open-source deep learning framework which is built on network morphism with an aim to boost Bayesian optimization. This AutoML framework can automatically search for hyperparameters and architecture for complex models. AutoKeras conducts searches with the help of Neural Architecture Search (NAS) algorithms to ultimately eliminate the need for deep learning engineers.
# 
# This Auto Machine Learning Toolkit follows the design of the classic scikit-learn API, however, it uses a powerful neural network search for model parameters using Keras.

# In[ ]:


get_ipython().system('pip install -q  keras torch torchvision graphviz autokeras  onnxmltools')
get_ipython().system('pip install -q    onnxmltools onnx-tf')
#!pip install --user autokeras 
get_ipython().system('pip install git+https://github.com/jhfjhfj1/autokeras.git')
import keras
import autokeras as ak
import onnxmltools


# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv', header=None)
test = pd.read_csv('../input/digit-recognizer/test.csv', header=None)
train.head()


# In[ ]:


train_data = train.iloc[:, 1:]
train_labels = train.iloc[:, 0]
test_data = test.iloc[:, 1:]
test_labels = test.iloc[:, 0]


# In[ ]:


clf = ak.ImageClassifier(verbose=True, augment=False)


# In[ ]:


time_limit=7*3600


# In[ ]:


clf.fit(x_train, y_train, time_limit=time_limit)


# In[ ]:


clf.final_fit(x_train, y_train, x_test, y_test)


# In[ ]:


clf.evaluate(x_test, y_test)


# In[ ]:


results = clf.predict(x_test)


# In[ ]:


model=clf.cnn.best_model #keras.models.load_model("model.h5")


# In[ ]:


import IPython
from graphviz import Digraph
dot = Digraph(comment='autokeras model')
graph=clf.cnn.best_model
for index, node in enumerate(graph.node_list):
    dot.node(str(index), str(node.shape))

for u in range(graph.n_nodes):
    for v, layer_id in graph.adj_list[u]:
      dot.edge(str(u), str(v), str(graph.layer_list[layer_id]))
dot.render(filename='model.png',format='png')
dot.render(filename='model.svg',format='svg')
IPython.display.Image(filename='model.png')


# In[ ]:


import IPython

keras_model=clf.cnn.best_model.produce_keras_model()

keras.utils.plot_model(keras_model, show_shapes=True, to_file='model_keras_mnist.png')
IPython.display.Image(filename='model_keras_mnist.png')


# In[ ]:


keras_model.summary()


# In[ ]:


keras_model.compile("adam","mse")
keras_model.save("model.h5")


# In[ ]:


#clf.export_keras_model("model.h5")


# In[ ]:


keras_model=keras.models.load_model("model.h5")
keras_model.compile("adam","mse")
onnx_model = onnxmltools.convert_keras(keras_model, target_opset=7) 


# In[ ]:


# Save as text
onnxmltools.utils.save_text(onnx_model, 'model.json')

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, 'model.onnx')


# In[ ]:


import keras2onnx
onnx_model = keras2onnx.convert_keras(keras_model,"autokeras nmist")
# Save as text
onnxmltools.utils.save_text(onnx_model, 'model_keras2onnx.json')

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, 'model_keras2onnx.onnx


# ---
# 
# > # Example on TPOT (Tree-basedPipeline Optimization Tool)
# 
# ---
# 
# ![](https://raw.githubusercontent.com/EpistasisLab/tpot/master/images/tpot-logo.jpg)
# 
# ---
# 
# It was 2018, that TPOT was put in the list of the most popular auto-machine learning frameworks on GitHub, and the popular AutoML framework has not looked back since then. The TPOT AutoML framework uses genetic programming to zero on a model for task implementation. TPOT AutoML framework can analyse thousands of pipelines to offer one with the best Python code.
# 
# TPOT comes with its own regression and classification algorithms. However, its disadvantages include the inability to interact with categorical lines and natural language.

# In[ ]:


# Importing libraries 
from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[ ]:


# Loading data
iris = load_iris()
iris.data[0:5], iris.target


# In[ ]:


# Splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    train_size=0.75, test_size=0.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


tpot = TPOTClassifier(generations=8, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print("Accuracy is {}%".format(tpot.score(X_test, y_test)*100))


# 
# You can also export the optimized model as output in a .py file. Check the output section to view the file and see the chosen model.
# Due to genetic programming, the resulting model can be different every time you run the model

# In[ ]:


tpot.export('tpot_iris_pipeline.py')


# ---
# 
# > # Example on Pycaret
# 
# ---

# ---
# ![](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/05/PyCaret-.jpg)
# 
# ---
# 
# PyCaret is an open source machine learning library in Python to train and deploy supervised and unsupervised machine learning models in a low-code environment. It is known for its ease of use and efficiency.
# 
# In comparison with the other open source machine learning libraries, PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with a few words only.

# In[ ]:


get_ipython().system('pip install pycaret')


# In[ ]:


from pycaret.datasets import get_data
dataset = get_data('credit')


# In[ ]:


data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# In[ ]:


from pycaret.classification import *


# In[ ]:


exp_clf101 = setup(data = data, target = 'default', session_id=123)


# In[ ]:


best_model = compare_models()


# In[ ]:


print(best_model)


# In[ ]:


models()


# In[ ]:


dt = create_model('dt')


# In[ ]:


print(dt)


# In[ ]:


knn = create_model('knn')


# In[ ]:


rf = create_model('rf')


# In[ ]:


tuned_dt = tune_model(dt)


# In[ ]:


print(tuned_dt)


# In[ ]:


import numpy as np
tuned_knn = tune_model(knn, custom_grid = {'n_neighbors' : np.arange(0,50,1)})


# In[ ]:


print(tuned_knn)


# In[ ]:


tuned_rf = tune_model(rf)


# In[ ]:


plot_model(tuned_rf, plot = 'auc')


# In[ ]:


plot_model(tuned_rf, plot = 'pr')


# In[ ]:


plot_model(tuned_rf, plot='feature')


# In[ ]:


plot_model(tuned_rf, plot = 'confusion_matrix')


# In[ ]:


evaluate_model(tuned_rf)


# In[ ]:


predict_model(tuned_rf);


# In[ ]:


final_rf = finalize_model(tuned_rf)


# In[ ]:


print(final_rf)


# In[ ]:


predict_model(final_rf);


# In[ ]:


unseen_predictions = predict_model(final_rf, data=data_unseen)
unseen_predictions.head()


# In[ ]:


from pycaret.utils import check_metric
check_metric(unseen_predictions.default, unseen_predictions.Label, 'Accuracy')


# In[ ]:


save_model(final_rf,'Final RF Model 08Feb2020')


# In[ ]:


saved_final_rf = load_model('Final RF Model 08Feb2020')


# In[ ]:


new_prediction = predict_model(saved_final_rf, data=data_unseen)


# In[ ]:


new_prediction.head()


# In[ ]:


from pycaret.utils import check_metric
check_metric(new_prediction.default, new_prediction.Label, 'Accuracy')


# > ### This one is example of Classification, To know about other methods, go to this [link](https://pycaret.org/tutorial/)

# In[ ]:





# ![](https://www.ibmx.website/wp-content/uploads/2019/10/evolution-chart-02.jpg)

# # References: 
# 1. https://www.kaggle.com/yufengg/automl-getting-started-notebook
# 2. https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml
# 3. https://automl.github.io/auto-sklearn/master/
# 4. https://www.kaggle.com/axelderomblay/running-mlbox-auto-ml-package-on-titanic
# 5. https://www.kaggle.com/cedriclacrambe/autokeras-emnist
# 6. https://www.kaggle.com/thebrownviking20/tpot-a-great-tool-to-automate-your-ml-workflow/data?
# 7. https://pycaret.org/

# # If you find this notebook useful leave an upvote to keep it in your favourites. 
# Also, If there is any error or any thing wrong written, please let me know! Thanks in Advance. 

# ![](https://www.charitydynamics.com/wp-content/uploads/2017/04/AdobeStock_66910958.jpeg)
