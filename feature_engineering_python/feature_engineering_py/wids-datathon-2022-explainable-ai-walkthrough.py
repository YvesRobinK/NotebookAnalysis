#!/usr/bin/env python
# coding: utf-8

#  <h1 style = "font-size:40px;font-family:verdana;text-align: center;background-color:#DCDCDC">WIDS DATATHON 2022</h1>
#  
# ![image.png](attachment:37446a0c-3f6f-4358-a6e7-4ff82b7e1842.png)
# 
# <h2 style = "font-size:30px;font-family:verdana;text-align: center">Problem Statement</h2>
# 
# <h3 style="font-size:18px;font-family:courier">The WiDS Datathon 2022 focuses on a prediction task involving roughly 100k observations of building energy usage records collected over 7 years and a number of states within the United States. The dataset consists of building characteristics (e.g. floor area, facility type etc), weather data for the location of the building (e.g. annual average temperature, annual total precipitation etc) as well as the energy usage for the building and the given year, measured as Site Energy Usage Intensity (Site EUI). Each row in the data corresponds to the a single building observed in a given year. Your task is to predict the Site EUI for each row, given the characteristics of the building and the weather data for the location of the building.</h3>
# 
# 
# 
# **This notebook is used for presentation in WIDS DATATHON 2022 Talk organized by WIDS Mysuru Community on the topic :A Walthrough Of Model Explainability and Interpretability Methods**
# 

# References:
# * https://www.kaggle.com/shrutisaxena/wids2022-starter-co
# * https://www.kaggle.com/usharengaraju/wids2022-lgbm-starter-w-b
# * https://www.kaggle.com/dansbecker/permutation-importance
# * https://medium.com/dataman-in-ai/explain-your-model-with-lime-5a1a5867b423
# * https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d

# In[1]:


import numpy as np
import pandas as pd
import datetime
import random
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
plt.style.use('ggplot')
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer
import shap
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=FutureWarning)


# # DATA

# In[2]:


#code copied from https://www.kaggle.com/usharengaraju/wids2022-lgbm-starter-w-b
train = pd.read_csv("../input/widsdatathon2022/train.csv")
test = pd.read_csv("../input/widsdatathon2022/test.csv")
print("Number of train samples are",train.shape)
print("Number of test samples are",test.shape)
categorical_features = ['State_Factor', 'building_class', 'facility_type']
numerical_features=train.select_dtypes('number').columns


# In[3]:


train.head()


# # MISSING VALUE
# 

# In[4]:


plt.figure(figsize = (25,11))
sns.heatmap(train.isna().values, xticklabels=train.columns)
plt.title("Missing values in training Data", size=20)


# # MISSING VALUE IMPUTATION

# In[5]:


#code copied from https://www.kaggle.com/shrutisaxena/wids2022-starter-code
missing_columns = [col for col in train.columns if train[col].isnull().any()]
missingvalues_count =train.isna().sum()
missingValues_df = pd.DataFrame(missingvalues_count.rename('Null Values Count')).loc[missingvalues_count.ne(0)]
missingValues_df .style.background_gradient(cmap="Pastel1")

train['year_built'] =train['year_built'].replace(np.nan, 2022)
test['year_built'] =test['year_built'].replace(np.nan, 2022)
null_col=['energy_star_rating','direction_max_wind_speed','direction_peak_wind_speed','max_wind_speed','days_with_fog']
imputer = SimpleImputer()
imputer.fit(train[null_col])
data_transformed = imputer.transform(train[null_col])
train[null_col] = pd.DataFrame(data_transformed)
test_data_transformed = imputer.transform(test[null_col])
test[null_col] = pd.DataFrame(test_data_transformed)


# In[6]:


plt.figure(figsize = (25,11))
sns.heatmap(train.isna().values, xticklabels=train.columns)
plt.title("Missing values in training Data", size=20)


# # LABEL ENCODING

# In[7]:


le = LabelEncoder()
for col in categorical_features:
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])


# In[8]:


train.head()


# # FEATURE SCALING

# In[9]:


train.describe().style.background_gradient()


# In[10]:


import copy
#code copied from https://www.kaggle.com/usharengaraju/wids2022-lgbm-starter-w-b
y = train["site_eui"]
train = train.drop(["site_eui","id"],axis =1)
test = test.drop(["id"],axis =1)
trainnames = copy.deepcopy(train)
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)


# In[11]:


train


# # MODEL:

# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.2, random_state = 50)


# In[13]:


X_test_ft=pd.DataFrame(X_test,columns=trainnames.columns)


# In[14]:


X_test_ft


# In[15]:


import xgboost
xgboost_model = xgboost.XGBRegressor(n_estimators=200, learning_rate=0.02, gamma=0, subsample=0.75,
                           colsample_bytree=0.4, max_depth=5)
xgboost_model.fit(X_train,y_train)


# 
# <h4 style = "font-size:40px;font-family:verdana;text-align: center">EXPLAINABLE AI</h4>
# 
# 
# 
# <h2 style = "font-size:30px;font-family:verdana;text-align: center">WHAT IS EXPLAINABLE AI?</h2>
# 
# Explainability in machine learning is the process of explaining to a human why and how a machine learning model made a decision. Model explainability means the algorithm and its decision or output can be understood by a human. It is the process of analysing machine learning model decisions and results to understand the reasoning behind the system’s decision. This is an important concept with ‘black box’ machine learning models, which develop and learn directly from the data without human supervision or guidance.  
# 
# ![image.png](attachment:03d23cf5-2338-43f9-85bf-7576631c7c47.png)
# 
# <h2 style = "font-size:30px;font-family:verdana;text-align: center">WHY IS MODEL EXPLAINABILITY IMPORTANT?</h2>
# 
# Many of the ML models, though achieving high-level of precision, are not easily understandable for how a recommendation is made. This is especially the case in a deep learning model. As humans, we must be able to fully understand how decisions are being made so that we can trust the decisions of AI systems. We need ML models to function as expected, to produce transparent explanations, and to be visible in how they work.
# 
# In this notebook,I will be walking you through the three important model explainability methods :**their internal working**,**the insights they provide** and **how we can interpret their results**:
# 
# 1. PERMUTATION IMPORTANCE
# 2. SHAP
# 3. LIME
# 
# In our problem statement,we have to predict the energy consumption of a building and the model we trained does so.It predicts ,for each sample in the test data,the energy consumption.But how can we get an answer to questions like why our model is predicting the value it is predicting?what are the variables positively correlated with the target?what are negatively correlated with the target?what variables have the highest importance in making predictions for the whole dataset/for a single example?
# With the help of these methods we can successfully answer all these WHYs and WHATs related to our model.Great, isn't it?
# 
# 

# # 1.PERMUTATION IMPORTANCE USING ELI5 LIBRARY:
# 
# Permutation Importance is a method of understanding which features are of highest importance to our model.It ranks the features according to the weights in descending order of their importance.
# 
# The algorithm that Permutation Importance uses to get the insights is very interesting and novel.It's easy to understand and it overcomes the shortcomings of other feature importance methods.
# 
# 1. First we train our model,here the xgboost model in our case on the train set.
# 2. Then we use the model as it is to get the predictions on our validation set.Then the rmse is calculated.
# 3. Further,the column whose feature importance has to be calculated is randomly shuffled.So,now our validation set is shuffled on 1 column.Then Step 2 is repeated on the shuffled validation set.The new rmse is compared to the old rmse and if there is an increase in the error,the feature importance is high,i.e. the shuffled feature was crucial in making predictions.It may also be possible that rmse decreases on shuffling or there is no change,then the shuffled feature is of nil importance to the model.The decrease in rmse after shuffling can be attributed to chance /luck or lower size of dataset that means higher probability of luck.
# 4. The validation set is then unshuffled to it's original state and the same steps are repeated for other features (one at a time) and then we obtain the feature importances.
# 
# 

# In[16]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(xgboost_model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = trainnames.columns.tolist(), top=63)


# We can infer that on the global level ,the top features in the descending order of their importance are:**Energy Star Rating>Facility Type> Floor area**

# # 2. SHAP 
# 
# 
# SHAP — which stands for SHapley Additive exPlanations — is an algorithm was first published in 2017 by Lundberg and Lee and it is a brilliant way to reverse-engineer the output of any predictive algorithm. 
# 
# 
# **WORKING**
# 
# We will see what the Shapley value is and how the SHAP (SHapley Additive exPlanations) value emerges from the Shapley concept.Understanding the idea behind the calculation of SHAP values is crucial to make sense of their outcome. We will go through the theoretical foundation of SHapley Additive exPlanations described in the article by Slundberg and Lee, and see why SHAP values are computed the way they are computed.
# 
# [This](https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30) article provides a very great explanation of SHAP values calculation.
# 
# 
# Important features of SHAP:
#  * Global interpretability 
#  * Local interpretability
# 

# In[17]:


explainer = shap.Explainer(xgboost_model)
shap_values = explainer(X_test_ft)


# #  GLOBAL INTERPRETABILITY
# 
# **Variable Importance Plot**

# In[18]:


shap.summary_plot(shap_values, X_test_ft,plot_type="bar")


# **INFERENCE:**
# 
# From this Variable Importance plot,we get similar results as the Permutation Importance.The features are arranged in descending order of their importance here.On the X-axis,we have the mean SHAP value for a feature.As the SHAP values are calculated per observation of the validation set ,to get a global interpretability across the dataset we need to have a mean of the calculated SHAP values.Here the positive or negative impact is not considered,just the magnitude of impact of a feature on the final predictions is taken into account.The below graph is also similar,just the impact is quantified by mentioning the mean(|SHAP|).

# In[19]:


shap.plots.bar(shap_values,max_display=12)


# **THE SUMMARY PLOT**:
# 
# The plot below shows the positive and negative relationship of a feature with the target variable prediction based on their SHAP values.Till now,we just saw the magnitude of the impact and not the nature of the impact.We'll see that now:
# 

# In[20]:


shap.summary_plot(shap_values, X_test_ft)


# **INFERENCE:**
# 
# This plot is made of all the dots in the train data. It demonstrates the following information:
# 
# * **Feature importance**: Variables are ranked in descending order.
# * **Impact**: The horizontal location shows whether the effect of that value is associated with a higher or lower prediction.
# * **Original value**: Color shows whether that variable is high (in red) or low (in blue) for that observation.
# * **Correlation**: A low value of the “energy star rating”  has a high and positive impact on the building energy consumption. The “low” comes from the blue color, and the “positive” impact is shown on the X-axis(the horizontal location). Similarly, "building class" and "facility type" are also  negatively correlated with the target variable(site_eui_id).

# **Interaction Plot**

# In[21]:


shap_interaction_values = explainer.shap_interaction_values(X_test_ft)
shap.summary_plot(shap_interaction_values, X_test_ft)


# # LOCAL INTERPRETABILITY
# **Waterfall Plot**

# In[22]:


shap.plots.waterfall(shap_values[0])


# The value at the bottom of the plot is E[f(X)] is the base value that is,“the value that would be predicted if we did not know any features for the current output.” In other words, it is the mean prediction, or mean(yhat),then we move upwards to see how much each feature is affecting the final prediction,f(X) for the given observation,either in the positive or negative direction.This gives a good local interpretation of the model's prediction.

# In[23]:


shap.plots.waterfall(shap_values[1])


# **Force plot**

# In[24]:


explainer = shap.TreeExplainer(xgboost_model)
shap_values = explainer.shap_values(X_test_ft)
shap.initjs()
def p(j):
    return(shap.force_plot(explainer.expected_value, shap_values[j,:], X_test_ft.iloc[j,:]))
p(0)


# The plot looks beautiful,but what do we infer from this?
# 
# * **The output value**:In the above plot the number in bold ,i.e. 58.78 is the value our model predicted for the first observation in the validation set.
# * **The base value**: The base value is “the value that would be predicted if we did not know any features for the current output.” In other words, it is the mean prediction, or mean(yhat).
# * **Red/blue**: Features that push the prediction higher (to the right) are shown in red, and those pushing the prediction lower are in blue.
# * The values corresponding to the features are mentioned for the observation.
# 
# Energy Star Rating: has a negative impact on the building energy consumption. The energy_star rating value is 1.255 for this observation,which is higher than the average value in the train set. So it pushes the prediction to the left.
# Similarly we can calculate for other features of the plot as well.
# 

# In[25]:


p(1)


# **Decision Plot**
# 
# Used as an alternative for waterfall plot when the number of predictors is high,to get a non-clumsy plot.

# In[26]:


expected_value = explainer.expected_value
shap_values = explainer.shap_values(X_test_ft)[0]
shap.decision_plot(expected_value, shap_values, X_test_ft)


# In[27]:


shap_values = explainer.shap_values(X_test_ft)[1]
shap.decision_plot(expected_value, shap_values, X_test_ft)


# # 3.LIME:
# 
# Local Interpretable Model-Agnostic Explanations (LIME) can explain the predictions of any classifier in “an interpretable and faithful manner, by learning an interpretable model locally around the prediction”. Their approach is to gain the trust of users for individual predictions and then to trust the model as a whole.
# 
# One of the most important and novel feature of **LIME** is:
# 
# **LOCAL FIDELITY**:
# The LIME model is all about interpreting local predictions in a faithful manner.The interpretation of a model's individual observation should correspond to atleast the points in it's vicinity.And,as you can see further how this is implemented in the LIME algorithm and how we can see this in the interpretation itself.
# A model can be trained on 100s of features but while making a prediction for an individual observation,only a subset of those features play a key role and LIME provides us that insight.That is, the local interpretability.LIME,as the name suggests is concerned with local interpretation only.
# 
# ![image.png](attachment:e804e349-5d12-4446-8dbf-bd0358683163.png)
# 
# The graph above was provided by the authors of LIME.The graph gives an intuition into the working of LIME model.The original model we want to explain(xgboost in our example) is represented by the blue/pink background.The prediction we need to explain is the bold red cross in the image above.So,LIME:
# 
# 1. Generates new samples in the vicinity of the red cross instance and then gets their predictions using the original model.
# 2. Weighting of the newly generated samples is done by the distance of the samples from the instance to be explained.
# 3. Finally,after getting the predictions for the newly generated samples including the red cross,a linear regression model is fitted,the black dashed line in the image above is the one used to explain our original model.This makes sure that the explanation is locally faithful as these explanations are generated from a linear regression model built taking the vicinity points +the original point into account.
# 

# In[28]:


import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(X_train,
                    feature_names=trainnames.columns, 
                    class_names=['site_eui'], 
                    categorical_features=categorical_features, 
                    verbose=True, mode='regression')


# # LOCAL EXPLANATION

# In[29]:


exp = explainer.explain_instance(X_test_ft.iloc[0], 
     xgboost_model.predict, num_features=10)
exp.as_pyplot_figure()


# * Green/Red color: features that have positive correlations with the target are shown in green, otherwise red.
# * Energy Star Rating>0.65:High energy star rating negatively correlate with high energy consumption.
# * Facility Type<=0.09:Lower Facility Type positively correlate with high energy consumption.
# 
# We can understand the other features by using the same logic.
# 

# **Coefficients of the LIME model by as_list():**

# In[30]:


pd.DataFrame(exp.as_list())


# # RESULTS IN NOTEBOOK FORMAT:

# In[31]:


exp.show_in_notebook(show_table=True, show_all=False)


# In[32]:


exp = explainer.explain_instance(X_test_ft.iloc[1], xgboost_model.predict)
exp.show_in_notebook(show_table=True, show_all=False)


# * The LIME model intercept: 79.40720440515193
# * The LIME model prediction: "Prediction_local [94.854192]"
# * The original XGBOOST model prediction: "Right: 74.22026"
# 
# The local lime prediction =intercept+sum of coefficients

# # ADVANTAGE OF LIME OVER SHAP:
# We know that SHAP can give us local +global interpretability and LIME is just giving local interpretability,then why use LIME?It is important to note that the two algorithms are very different.The SHAP calculates the SHAP values for each feature first and to accomplish that it has to train models across all the possible feature combinations globally to get local interpretability.
# But,the LIME algorithm is simpler and less time consuming when computing local interpretations.It does not go into global calculations in the process.So,the speed is an important factor.
# 
# 

# # VALIDATION LOSS

# In[33]:


from sklearn.metrics import explained_variance_score
predictions = xgboost_model.predict(X_test)
print(explained_variance_score(predictions,y_test))


# # SUBMISSISON

# In[34]:


res = xgboost_model.predict(test)
sub = pd.read_csv("../input/widsdatathon2022/sample_solution.csv")
sub["site_eui"] = res
sub.to_csv("submission.csv", index = False)

