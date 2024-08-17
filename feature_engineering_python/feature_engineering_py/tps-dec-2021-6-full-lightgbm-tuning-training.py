#!/usr/bin/env python
# coding: utf-8

# ## To  do: 
# 
# #### 1. Ensemble 
# - create a model with Cross Val (find best)
# - Create model with Neural Network (seperate kernel)
# - Add probabilities together and argmax
# 
# #### 2. DART Cross val
# - Run outside Kaggle as takes too long 
# 
# #### 3. Check Clipped values feature importance -- maybe remove
# 
# #### 4. Fold Additional Data into Cross Val 

# # Project Description and goals 
# In this months Tabular Playground we have a synthetic [forest cover dataset ](https://www.kaggle.com/c/forest-cover-type-prediction/data). 
# 
# ## Process:
# ### 1. Feature engineering 
# * feature extraction - as per notebooks and competition discussions 
# * feature processing - duplicates, nulls etc 
# 
# ### 2. Data Augmentation 
# * Sampling  - over / under sampling 
# * Additional data - external / synthetic(oversampling) \   
# * Feature Selection  - PCA 
# 
# ### 3. Model Build  / Transfer learning 
# * Model type  **Note** will only use LIGHTGBM
# * Loss / error type - multilogloss
# * Model build - classifier, train, cross-val
# 
# ### 4. Training Schedule and Optimization
# * Optimize LIGHTGBM  ---needed for multiple experiments 
# * Parameter Tuning - Optuna 
# 
# ### 5. Post processing 
# * Improve post predicted values - calibration
# 
# Scoring metric = accuracy

# ## Experiments 
# I ran multiple experiments using different parameters, the main experiments and submission scores are below:
# 
# <style type="text/css">
# .tg  {border-collapse:collapse;border-spacing:0;}
# .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
#   overflow:hidden;padding:10px 5px;word-break:normal;}
# .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
#   font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
# .tg .tg-vfn0{background-color:#efefef;border-color:#000000;text-align:left;vertical-align:top}
# .tg .tg-cjtp{background-color:#ecf4ff;border-color:inherit;text-align:left;vertical-align:top}
# .tg .tg-l0ny{background-color:#FFCE93;border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
# .tg .tg-pwrz{background-color:#FD6864;border-color:inherit;text-align:left;vertical-align:top}
# .tg .tg-jio7{background-color:#FD6864;border-color:#000000;text-align:left;vertical-align:top}
# .tg .tg-tcpe{background-color:#ecf4ff;border-color:inherit;color:#333333;font-weight:bold;text-align:left;vertical-align:top}
# .tg .tg-l6ss{background-color:#3531ff;border-color:#6200c9;color:#ffffff;font-weight:bold;text-align:left;vertical-align:top}
# .tg .tg-mylm{background-color:#ecf4ff;border-color:#000000;color:#333333;font-weight:bold;text-align:left;vertical-align:top}
# .tg .tg-zv36{background-color:#ffffff;border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
# .tg .tg-7od5{background-color:#9aff99;border-color:inherit;text-align:left;vertical-align:top}
# .tg .tg-jxgv{background-color:#FFF;border-color:inherit;text-align:left;vertical-align:top}
# .tg .tg-dvid{background-color:#efefef;border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
# .tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
# .tg .tg-pdeq{background-color:#FFF;border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
# .tg .tg-y698{background-color:#efefef;border-color:inherit;text-align:left;vertical-align:top}
# .tg .tg-pidv{background-color:#ffce93;border-color:inherit;text-align:left;vertical-align:top}
# .tg .tg-fymr{border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
# .tg .tg-73oq{border-color:#000000;text-align:left;vertical-align:top}
# .tg .tg-112g{background-color:#67FD9A;border-color:inherit;text-align:left;vertical-align:top}
# .tg .tg-q32f{background-color:#9AFF99;border-color:inherit;text-align:left;vertical-align:top}
# .tg .tg-smvl{background-color:#fd6864;border-color:inherit;text-align:left;vertical-align:top}
# .tg .tg-kbue{background-color:#9aff99;border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
# .tg .tg-c6of{background-color:#ffffff;border-color:inherit;text-align:left;vertical-align:top}
# .tg .tg-fgdu{background-color:#ecf4ff;border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
# </style>
# <table class="tg">
# <thead>
#   <tr>
#     <th class="tg-l6ss">No.</th>
#     <th class="tg-l6ss">Name</th>
#     <th class="tg-l6ss">Version</th>
#     <th class="tg-l6ss">Model Type</th>
#     <th class="tg-l6ss">Optuna</th>
#     <th class="tg-l6ss">Additional Data</th>
#     <th class="tg-l6ss">Boosting Type</th>
#     <th class="tg-l6ss">Scaler</th>
#     <th class="tg-l6ss">Score</th>
#     <th class="tg-l6ss">To Rerun</th>
#   </tr>
# </thead>
# <tbody>
#   <tr>
#     <td class="tg-0pky">1</td>
#     <td class="tg-tcpe" rowspan="4">Scaler</td>
#     <td class="tg-0pky">117</td>
#     <td class="tg-0pky">Cross Val</td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">GBDT</td>
#     <td class="tg-0pky">Robust</td>
#     <td class="tg-0pky">0.95562</td>
#     <td class="tg-pdeq"></td>
#   </tr>
#   <tr>
#     <td class="tg-0pky">2</td>
#     <td class="tg-0pky">212</td>
#     <td class="tg-0pky">Cross Val</td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">GBDT</td>
#     <td class="tg-0pky">MinMax </td>
#     <td class="tg-0pky">0.95570</td>
#     <td class="tg-pdeq"></td>
#   </tr>
#   <tr>
#     <td class="tg-0pky">3</td>
#     <td class="tg-0pky">213</td>
#     <td class="tg-0pky">Cross Val</td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">GBDT</td>
#     <td class="tg-0pky">Standard</td>
#     <td class="tg-0pky">0.95570</td>
#     <td class="tg-pdeq"></td>
#   </tr>
#   <tr>
#     <td class="tg-0pky">4</td>
#     <td class="tg-0pky">251</td>
#     <td class="tg-0pky">Cross Val</td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">GBDT</td>
#     <td class="tg-0pky">None</td>
#     <td class="tg-7od5">0.95571</td>
#     <td class="tg-zv36"></td>
#   </tr>
#   <tr>
#     <td class="tg-0pky">5</td>
#     <td class="tg-tcpe" rowspan="2">Optuna </td>
#     <td class="tg-y698">173</td>
#     <td class="tg-pidv">Train</td>
#     <td class="tg-pidv">Yes</td>
#     <td class="tg-pidv">PSEUDO 2<br>Original Data (Complete)</td>
#     <td class="tg-pidv">GBDT</td>
#     <td class="tg-pidv">None</td>
#     <td class="tg-pidv">0.95496</td>
#     <td class="tg-l0ny">Rerunning</td>
#   </tr>
#   <tr>
#     <td class="tg-0pky">6</td>
#     <td class="tg-y698">240</td>
#     <td class="tg-y698">Train</td>
#     <td class="tg-y698">Yes</td>
#     <td class="tg-y698">PSEUDO 2</td>
#     <td class="tg-y698">GBDT</td>
#     <td class="tg-dvid">MinMax</td>
#     <td class="tg-7od5">BASELINE</td>
#     <td class="tg-zv36"></td>
#   </tr>
#   <tr>
#     <td class="tg-0pky">9</td>
#     <td class="tg-tcpe" rowspan="3">Model Type</td>
#     <td class="tg-0pky">86</td>
#     <td class="tg-0pky">Train</td>
#     <td class="tg-0pky">Yes</td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">GBDT</td>
#     <td class="tg-0pky">Robust</td>
#     <td class="tg-7od5">0.95491</td>
#     <td class="tg-0pky"></td>
#   </tr>
#   <tr>
#     <td class="tg-0pky">10</td>
#     <td class="tg-0pky">112</td>
#     <td class="tg-0pky">Classifier</td>
#     <td class="tg-0pky">Yes</td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">GBDT</td>
#     <td class="tg-0pky">Robust</td>
#     <td class="tg-7od5">0.95510</td>
#     <td class="tg-0pky"></td>
#   </tr>
#   <tr>
#     <td class="tg-0pky">11</td>
#     <td class="tg-0pky">107</td>
#     <td class="tg-0pky">Cross Val</td>
#     <td class="tg-0pky">Yes</td>
#     <td class="tg-0pky">NO</td>
#     <td class="tg-0pky">GBDT</td>
#     <td class="tg-0pky">Robust</td>
#     <td class="tg-7od5">0.95556</td>
#     <td class="tg-fymr"></td>
#   </tr>
#   <tr>
#     <td class="tg-73oq">12</td>
#     <td class="tg-mylm">DART</td>
#     <td class="tg-vfn0">238</td>
#     <td class="tg-vfn0">Train</td>
#     <td class="tg-vfn0">No</td>
#     <td class="tg-vfn0">PSEUDO 2</td>
#     <td class="tg-vfn0">DART</td>
#     <td class="tg-vfn0">Robust</td>
#     <td class="tg-vfn0">0.95532</td>
#     <td class="tg-jio7">RE-RUNNING HOME</td>
#   </tr>
#   <tr>
#     <td class="tg-0pky"></td>
#     <td class="tg-cjtp" rowspan="11"><span style="font-weight:bold">CROSS VAL</span></td>
#     <td class="tg-fymr">259</td>
#     <td class="tg-fymr">Train </td>
#     <td class="tg-fymr">No</td>
#     <td class="tg-fymr">PSEUDO 2</td>
#     <td class="tg-fymr">GBDT</td>
#     <td class="tg-fymr">MinMax</td>
#     <td class="tg-112g">0.95692</td>
#     <td class="tg-jxgv"></td>
#   </tr>
#   <tr>
#     <td class="tg-0pky"></td>
#     <td class="tg-0pky">260</td>
#     <td class="tg-0pky">Train </td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">PSEUDO 2<br>Original Data (4,5,6)</td>
#     <td class="tg-0pky">GBDT</td>
#     <td class="tg-0pky">MinMax</td>
#     <td class="tg-7od5">0.95692</td>
#     <td class="tg-jxgv"></td>
#   </tr>
#   <tr>
#     <td class="tg-0pky"></td>
#     <td class="tg-0pky">267</td>
#     <td class="tg-0pky">Train </td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">PSEUDO 2<br>Original Data (Complete)</td>
#     <td class="tg-0pky">GBDT</td>
#     <td class="tg-0pky">MinMax</td>
#     <td class="tg-7od5">0.95692</td>
#     <td class="tg-jxgv"></td>
#   </tr>
#   <tr>
#     <td class="tg-0pky"></td>
#     <td class="tg-0pky">220</td>
#     <td class="tg-0pky">Train </td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">PSEUDO 2</td>
#     <td class="tg-0pky">GBDT</td>
#     <td class="tg-0pky">Robust</td>
#     <td class="tg-q32f">0.95688</td>
#     <td class="tg-smvl">RE-RUNNING 261</td>
#   </tr>
#   <tr>
#     <td class="tg-0pky"></td>
#     <td class="tg-0pky">228</td>
#     <td class="tg-0pky">Train </td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">PSEUDO 2<br>Original Data (4,5,6)</td>
#     <td class="tg-0pky">GBDT</td>
#     <td class="tg-0pky">Robust</td>
#     <td class="tg-q32f">0.95689</td>
#     <td class="tg-jxgv"></td>
#   </tr>
#   <tr>
#     <td class="tg-0pky"></td>
#     <td class="tg-pidv">234</td>
#     <td class="tg-pidv">Train </td>
#     <td class="tg-pidv">No</td>
#     <td class="tg-pidv">PSEUDO 2<br>Original Data (Complete)</td>
#     <td class="tg-pidv">GBDT</td>
#     <td class="tg-pidv">Robust</td>
#     <td class="tg-112g">0.95690</td>
#     <td class="tg-pwrz">RE-RUNNING 266</td>
#   </tr>
#   <tr>
#     <td class="tg-0pky"></td>
#     <td class="tg-0pky"></td>
#     <td class="tg-0pky">Train </td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">PSEUDO 2</td>
#     <td class="tg-0pky">GBDT</td>
#     <td class="tg-0pky">Standard</td>
#     <td class="tg-0pky"></td>
#     <td class="tg-pwrz">RUNNING 262</td>
#   </tr>
#   <tr>
#     <td class="tg-0pky"></td>
#     <td class="tg-0pky">264</td>
#     <td class="tg-0pky">Train </td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">PSEUDO 2<br>Original Data (Complete)</td>
#     <td class="tg-0pky">GBDT</td>
#     <td class="tg-0pky">Standard</td>
#     <td class="tg-7od5">0.95684</td>
#     <td class="tg-jxgv"></td>
#   </tr>
#   <tr>
#     <td class="tg-0pky"></td>
#     <td class="tg-0pky">258</td>
#     <td class="tg-0pky">Train </td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">PSEUDO 2</td>
#     <td class="tg-0pky">GBDT</td>
#     <td class="tg-0pky">None</td>
#     <td class="tg-7od5">0.95691</td>
#     <td class="tg-jxgv"></td>
#   </tr>
#   <tr>
#     <td class="tg-0pky"></td>
#     <td class="tg-fymr">252</td>
#     <td class="tg-fymr">Train </td>
#     <td class="tg-fymr">No</td>
#     <td class="tg-fymr">PSEUDO 2<br>Original Data (Complete)</td>
#     <td class="tg-fymr">GBDT</td>
#     <td class="tg-fymr">None</td>
#     <td class="tg-kbue">0.95701</td>
#     <td class="tg-jxgv"></td>
#   </tr>
#   <tr>
#     <td class="tg-0pky"></td>
#     <td class="tg-0pky">253</td>
#     <td class="tg-0pky">Train - unbalanced false</td>
#     <td class="tg-0pky">No</td>
#     <td class="tg-0pky">PSEUDO 2<br>Original Data (Complete)</td>
#     <td class="tg-0pky">GBDT</td>
#     <td class="tg-0pky">None</td>
#     <td class="tg-7od5">NO Change</td>
#     <td class="tg-c6of"></td>
#   </tr>
#   <tr>
#     <td class="tg-0pky"></td>
#     <td class="tg-fgdu">Ensemble</td>
#     <td class="tg-0pky"></td>
#     <td class="tg-0pky">Train, Class, Cross_val</td>
#     <td class="tg-0pky"></td>
#     <td class="tg-0pky"></td>
#     <td class="tg-0pky"></td>
#     <td class="tg-0pky"></td>
#     <td class="tg-0pky"></td>
#     <td class="tg-0pky"></td>
#   </tr>
# </tbody>
# </table>
# 
# **BEST Run** CROSS VAL v252 - No Scaling PSEUDO 2 +ADD DATA(Complete) **BEST**  ==0.95691 

# #### Over Sampling techniques 
# To check 
# 
# #### PCA
# To check
# 
# #### Summary 
# * Hyperparameters tuning              ---- using Optuna 
# * Oversampling                        ---- duplicated class5 increases accuracy -- checking SMOTEENN and SMOTEtomek
# * clustering                          ---- Kmeans reduces accuracy                  ---- CHECKING DBSCAN
# * Calibration                         ---- sigmoid reduces accuracy-- "isotonic" seems to improve accuracy
# * Cross Validation                    ---- Checking 
# * pseudo labelling                    ---- CHECKING 
# * Scaling                             ---- Unsure - need to confirm with standardised params

# # Libraries 

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

#scaling 
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from scipy.stats import mode

#model
import lightgbm as lgb

# parameter tuning
import optuna

# over and under sampling
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import CondensedNearestNeighbour

# analysis and files
from collections import Counter
import time
import os


# ## Experiments

# In[2]:


#Boosting
#dart has dropout and should stop overfitting , -- very long to run
#gdbt is default with good accuracy but tends to overfit
# goss is fast to converge 

#Scaling 
#Robust is good with outliers 
SCALER_NAME = "None"   #None
SCALER = MinMaxScaler() # RobustScaler() MinMaxScaler() StandardScaler()

## additional data and sampling
PSEUDOLABEL =2
ADD_DATA = False
CLUSTERING = False 
sample_technique = "class5" #'class5' "none" "SMOTEENN" "SMOTETomek"  "PCA"   ----SMOTEEN and SMOTETomek needs Add Data to run 

#Training Type and Tuning 
OPTUNA = False
NUM_TRIALS  = 200

#Training Params
MODEL_TYPE = "Cross_validation" #"train" "classifier" "classifier_cal" "Cross_validation"
Full_Train = False # retrain model on full dataset

FOLDS = 20
BOOSTING = 'gbdt'  # "goss" 'dart'  'gbdt'
EARLY_STOPPING_ROUNDS = 30 #20
EPOCHS = 1000 #1000

# Post processing 
CALIBRATION_METHOD = 'isotonic' #'sigmoid'


# In[3]:


#Params 
DOWNCASTING = True

DEVICE = "cpu" #"gpu" cpu

# multi classification method - weightings 
UNBALANCED = True

METRIC = "multi_logloss"


# In[4]:


title = "CROSS - Best Params -no scaling- PSEUDO 2+ADD DATA (Complete)"
version = 1
minor = 0

f= open(f"log_file_{version}_{minor}.txt","a")
f.write(f"########################### {title} ########################### ")
f.write(f"\n Boosting type: {BOOSTING}")
f.write(f"\n Sampling technique: {sample_technique}")
f.write(f"\n Scaler: {SCALER_NAME}")
f.write(f"\n Hyper Parameter Tuning: {OPTUNA}")
f.write(f"\n Model Type: {MODEL_TYPE}")
f.write(f"\n Pseudo labelling : {PSEUDOLABEL}")
f.close()


# ## Enable LightGBM GPU

# In[5]:


"""!pip uninstall -y lightgbm
!apt-get install -y libboost-all-dev
!git clone --recursive https://github.com/Microsoft/LightGBM"""


# In[6]:


"""%%bash
cd LightGBM
rm -r build
mkdir build
cd build
cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..
make -j$(nproc)"""


# In[7]:


"""!cd LightGBM/python-package/;python setup.py install --precompile
!mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
!rm -r LightGBM"""


# # EDA 

# In[8]:


train = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv", index_col=0)
test = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv", index_col=0)


# In[9]:


train.shape


# ## Load Additional & Pseudo data 
# * Load original [forest Cover Data ](https://www.kaggle.com/uciml/forest-cover-type-dataset) \
#    From previous experiments we noted classes 4,5, and 6 have the worst accuracy \
#    We will add these classes only
# 

# In[10]:


def additional_data():
    original_data = pd.read_csv("../input/forest-cover-type-dataset/covtype.csv")
    #only class 4, 5, 6
    #original_data = original_data [ (original_data["Cover_Type"] ==4) | (original_data["Cover_Type"] ==5)  | (original_data["Cover_Type"] ==6) ]
    original_data["Cover_Type"].value_counts()
    return original_data

original_data =additional_data() 
display(original_data.head())


# In[11]:


def pseudo_labelling_1():
    # obtain submissions
    sub_files = pd.DataFrame()
    for file in os.listdir("../input/submission-files-tps-dec-2021"):
        sub_files = pd.concat([sub_files,pd.read_csv("../input/submission-files-tps-dec-2021/"+file,index_col=0)],axis=1)    
        
    # get all rows where the values are the same for each column - num files =10
    filter_vals =  sub_files.sum(axis =1 )/ 10 == sub_files.iloc[:,0]
    
    # filter test and sub_files from above results & join 
    pseudo_df = test.copy(deep=True)[ filter_vals ] 
    pseudo_df["Cover_Type"] = sub_files[ filter_vals ].iloc[:,0]
    
    return pseudo_df


# In[12]:


if PSEUDOLABEL ==1:
    print("Pseudo 1 loaded")
    pseudo_df =  pseudo_labelling_1()
    print("Pseudo shape: ",pseudo_df.shape)
    
if PSEUDOLABEL ==2 or PSEUDOLABEL ==0:
    print("Pseudo 2 loaded")
    pseudo_df = pd.read_csv('../input/tps12-pseudolabels/tps12-pseudolabels_v2.csv', index_col=0, dtype=train.dtypes.to_dict())
    print("Pseudo shape: ",pseudo_df.shape)


# In[13]:


train["Cover_Type"].value_counts()


# In[14]:


plt.figure(figsize = (10,7))
sns.countplot(x = train["Cover_Type"])
plt.title("Count of Target (Cover_Type)")


# ### Correlation analaysis 
# We will exclude soil types as these are boolean values 

# In[15]:


cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
       'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Cover_Type']


# In[16]:


plt.figure(figsize = (15,10))
sns.heatmap(train[cols].corr(), vmin = -1, vmax=1, annot = True, cmap="Spectral") 


# **Note**: Elevation seems to have the highest negative correlation to Cover_type although relativelty weak correlation 

# # Downcasting 
# We note from train.info above that all columns are int64 \
# We should look at the min and max of the columns to check if they need full 64bit to store the integers 

# ![image.png](attachment:97f75336-b8a0-4e8b-a220-719f126bac38.png)

# In[17]:


def memory_usage_mb(df, *args, **kwargs):
    """Dataframe memory usage in MB. """
    return df.memory_usage(*args, **kwargs).sum() / 1024**2

def reduce_memory_usage(df, deep=True, verbose=True, categories=True):
    # All types that we want to change for "lighter" ones.
    # int8 and float16 are not include because we cannot reduce
    # those data types.
    # float32 is not include because float16 has too low precision.
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(df, deep=deep)

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if col_type == "object":
            df[col] = df[col].astype("category")
            best_type = "category"
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name
        # Log the conversion performed.
        #if verbose and best_type is not None and best_type != str(col_type):
         #   print(f"Column '{col}' converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(df, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(f"Memory usage decreased from"
              f" {start_mem:.2f}MB to {end_mem:.2f}MB"
              f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")


# In[18]:


if DOWNCASTING:
    reduce_memory_usage(train)
    reduce_memory_usage(test)
    reduce_memory_usage(original_data)
    reduce_memory_usage(pseudo_df)


# # Feature Engineering and splitting 
# All credit to their respecitve notebooks and discussion topics 

# In[19]:


all_df = pd.concat([train.assign(ds=0), test.assign(ds=1),original_data.assign(ds=2),pseudo_df.assign(ds=3)]).reset_index(drop=True).drop(columns=['Soil_Type7', 'Soil_Type15'] )

def start_at_eps(series, eps=1e-10): return series - series.min() + eps

pos_h_hydrology = start_at_eps(all_df.Horizontal_Distance_To_Hydrology)
pos_v_hydrology = start_at_eps(all_df.Vertical_Distance_To_Hydrology)

wilderness = all_df.columns[all_df.columns.str.startswith('Wilderness')]
soil_type = all_df.columns[all_df.columns.str.startswith('Soil_Type')]
hillshade = all_df.columns[all_df.columns.str.startswith('Hillshade')]

all_df = pd.concat([
    all_df,

    all_df[wilderness].sum(axis=1).rename('Wilderness_Sum').astype(np.float32),
    all_df[soil_type].sum(axis=1).rename('Soil_Type_Sum').astype(np.float32),

    (all_df.Aspect % 360).rename('Aspect_mod_360'),
    (all_df.Aspect * np.pi / 180).apply(np.sin).rename('Aspect_sin').astype(np.float32),
    (all_df.Aspect - 180).where(all_df.Aspect + 180 > 360, all_df.Aspect + 180).rename('Aspect2'),

    (all_df.Elevation - all_df.Vertical_Distance_To_Hydrology).rename('Hydrology_Elevation'),
    all_df.Vertical_Distance_To_Hydrology.apply(np.sign).rename('Water_Vertical_Direction'),

    (pos_h_hydrology + pos_v_hydrology).rename('Manhatten_positive_hydrology').astype(np.float32),
    (all_df.Horizontal_Distance_To_Hydrology.abs() + all_df.Vertical_Distance_To_Hydrology.abs()).rename('Manhattan_abs_hydrology'),
    (pos_h_hydrology ** 2 + pos_v_hydrology ** 2).apply(np.sqrt).rename('Euclidean_positive_hydrology').astype(np.float32),
    (all_df.Horizontal_Distance_To_Hydrology ** 2 + all_df.Vertical_Distance_To_Hydrology ** 2).apply(np.sqrt).rename('Euclidean_hydrology'),

    all_df[hillshade].clip(lower=0, upper=255).add_suffix('_clipped'),
    all_df[hillshade].sum(axis=1).rename('Hillshade_sum'),

    (all_df.Horizontal_Distance_To_Roadways * all_df.Elevation).rename('road_m_elev'),
    (all_df.Vertical_Distance_To_Hydrology * all_df.Elevation).rename('vhydro_elevation'),
    (all_df.Elevation - all_df.Horizontal_Distance_To_Hydrology * .2).rename('elev_sub_.2_h_hydro').astype(np.float32),

    (all_df.Horizontal_Distance_To_Hydrology + all_df.Horizontal_Distance_To_Fire_Points).rename('h_hydro_p_fire'),
    (start_at_eps(all_df.Horizontal_Distance_To_Hydrology) + start_at_eps(all_df.Horizontal_Distance_To_Fire_Points)).rename('h_hydro_eps_p_fire').astype(np.float32),
    (all_df.Horizontal_Distance_To_Hydrology - all_df.Horizontal_Distance_To_Fire_Points).rename('h_hydro_s_fire'),
    (all_df.Horizontal_Distance_To_Hydrology + all_df.Horizontal_Distance_To_Roadways).abs().rename('abs_h_hydro_road'),
    (start_at_eps(all_df.Horizontal_Distance_To_Hydrology) + start_at_eps(all_df.Horizontal_Distance_To_Roadways)).rename('h_hydro_eps_p_road').astype(np.float32),

    (all_df.Horizontal_Distance_To_Fire_Points + all_df.Horizontal_Distance_To_Roadways).abs().rename('abs_h_fire_p_road'),
    (all_df.Horizontal_Distance_To_Fire_Points - all_df.Horizontal_Distance_To_Roadways).abs().rename('abs_h_fire_s_road'),
], axis=1)

types = {'Cover_Type': np.int8}
train = all_df.loc[all_df.ds == 0].astype(types).drop(columns=['ds'])
test = all_df.loc[all_df.ds == 1].drop(columns=['Cover_Type', 'ds'])
original_data = all_df.loc[all_df.ds == 2].astype(types).drop(columns=['ds'])
pseudo_df = all_df.loc[all_df.ds == 3].astype(types).drop(columns=['ds'])

del all_df
del pos_h_hydrology
del pos_v_hydrology
del wilderness
del soil_type
del hillshade

if not ADD_DATA:
    del original_data
if PSEUDOLABEL==0:
    del pseudo_df


# In[20]:


print(train.shape)
print(test.shape)
display(train.head())


# ## Encode the Target
# Seems like a weird interaction in lighgbm.train \
# Need to specify the num_classes for the model and it only works if the target class starts from 0 (for classification) 

# In[21]:


#start from 0 --> 6
train["Cover_Type"] = train["Cover_Type"]-1

if ADD_DATA:
    original_data["Cover_Type"] = original_data["Cover_Type"]-1
    
if PSEUDOLABEL>0:
    pseudo_df["Cover_Type"] = pseudo_df["Cover_Type"]-1


# ## Sampling techniques
# 
# Sampling used:
# * none
# * Class 5 duplication 
# * SMOTEEEN   --- due to memory issues we have to run pca 
# * SMOTETomek --- due to memory issues we have to run pca 

# In[22]:


kfold = StratifiedKFold(n_splits= FOLDS, shuffle=True, random_state=42)


# In[23]:


class oversample_techniques():
    
    def __init__ (self, df ,num):
        self.df = df 
        self.num = num
        
    def sample_class5(self):
        
        #note that due to encoding class 5 = 4
        class5 = self.df[self.df["Cover_Type"]==4]
        #append class 5 for each fold -1 
        for i in range(self.num):
            self.df = self.df.append(class5, ignore_index=True)
            
        f= open(f"log_file_{version}_{minor}.txt","a")
        f.write("\n ##############################")
        f.write("\n Oversampling Technique: oversample class 5")
        f.write("\n ##############################")
        f.close()
        
        return self.df
    
    def no_change(self):
        f= open(f"log_file_{version}_{minor}.txt","a")
        f.write("\n ##############################")
        f.write("\n Oversampling Technique: no sampling")
        f.write("\n ##############################")
        f.close()
        
        
    def over_under_SMOTEENN(self):
        
        smote_enn = SMOTEENN(random_state=42, n_jobs = -1)
        X_resampled, y_resampled = smote_enn.fit_resample(self.df.drop("Cover_Type",axis =1),
                                                          self.df["Cover_Type"])
        X_resampled["Cover_Type"] = y_resampled
        
        f= open(f"log_file_{version}_{minor}.txt","a")
        f.write("\n ##############################")
        f.write("\n Oversampling Technique: SMOTEENN")
        f.write("\n ##############################")
        f.close()
        
        return X_resampled
    
    def over_under_SMOTETomek(self):

        smote_tomek = SMOTETomek(random_state=42, n_jobs = -1)
        X_resampled, y_resampled = smote_tomek.fit_resample(self.df.drop("Cover_Type",axis =1),
                                                            self.df["Cover_Type"])
        X_resampled["Cover_Type"] = y_resampled
        
        f= open(f"log_file_{version}_{minor}.txt","a")
        f.write("\n ##############################")
        f.write("\n Oversampling Technique: SMOTETomek")
        f.write("\n ##############################")
        f.close()
        
        return X_resampled


# In[24]:


# run class 5 sample here as we are duplicatign the training data not the split data 
if sample_technique == "class5":
    train = oversample_techniques(train,5).sample_class5()


# In[25]:


train["Cover_Type"].value_counts()


# # Data Augmentation 
# 
# 1. Adding original data 
# 1. adding Pseudo labels
# 1. Over/ under sample data 
# 
# **Note**: Im adverse to drop data so we shall keep class 5 and either append new data, over/undersample or duplicate class 5

# # Split 

# In[26]:


X = train.drop("Cover_Type",axis =1)
y = train["Cover_Type"]

print("X shape:",X.shape)

if MODEL_TYPE != "Cross_validation":
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("X_train shape:",X_train.shape)


# ## Additional Data 
# We will use the [original data](https://www.kaggle.com/uciml/forest-cover-type-dataset) and concatenate so our training data - to as to keep validation data clean

# In[27]:


if ADD_DATA:
    print("Adding Original Data")
    
    ## Cross val uses training data so add it to train
    if MODEL_TYPE =="Cross_validation":
        print("X shape prior: ",X.shape)
        train = pd.concat([X,original_data.drop("Cover_Type",axis =1)])
        print("X shape after: ",train.shape)
    else:
        print("X_train shape prior: ",X_train.shape)
        X_train = pd.concat([X_train,original_data.drop("Cover_Type",axis =1)])
        y_train = pd.concat([y_train,original_data["Cover_Type"]])
        print("X_Train shape after: ",X_train.shape)
del train


# ## Pseudo Labelling 
# Two ways we can do this: \
# **Process 1** \
# We can use historical submissions (10 submissions) and identify where all submissions predicted the same class \
# We can then append these rows to our training data 
#    
# **Process 2** \
# Run our a training model over the training data - where probability > threshold (i.e. 90%) then we add this data to training data and retrain model 

# In[28]:


# we concat here if not a cross validation session as cross validation concat will occur in fold
if PSEUDOLABEL >0 and MODEL_TYPE!="Cross_validation" :
    print("Pseudo shape: ",pseudo_df.shape)
    print("XTrain shape prior: ",X_train.shape)
    
    X_train = pd.concat([X_train,pseudo_df.drop("Cover_Type",axis =1)])
    y_train = pd.concat([y_train,pseudo_df["Cover_Type"]])
    
    print("XTrain shape after: ",X_train.shape)


# ### SPlit & Scale
# Split and scaling for **Train** and **Classifier** done here
# **Cross Val ** is done in fold 

# In[29]:


def scale_data(X_train, X_test, test):
    scaler= SCALER
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    test_scaled = scaler.transform(test)
    
    return X_train, X_test, test_scaled


# In[30]:


if SCALER_NAME != "None" and MODEL_TYPE !="Cross_validation":
    
    print(f"Scaling with {SCALER_NAME}")
    X_train, X_test, test_scaled = scale_data(X_train, X_test, test)
    
    display(test_scaled)


# ## PCA 
# * There are a certain features which may have low to no impact on the Target - PCA should have limit this impact and remove noise
# * Oversampling Techniques - Due to memory issues for sampling we need to run PCA prior to Sampling

# In[31]:


if sample_technique in ["SMOTETEENN", "SMOTETomek", "PCA"]:
    
    pca = PCA(n_components=10)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    test = pca.transform(test)

    pca_cols = []
    for i in range(10):
        pca_cols.append("pca_"+f"{i}")

    X_train = pd.DataFrame(X_train, columns = pca_cols)
    X_test = pd.DataFrame(X_test, columns = pca_cols)

    test = pd.DataFrame(test, columns = pca_cols)
    
    #Boxplot & df 
    plt.figure(figsize= (15,7))
    sns.boxplot(data = X_train)
    
    display(X_train.head())


# In[32]:


print(f"Applying Oversampling Technique: {sample_technique}")

if sample_technique == "none":
    oversample  = oversample_techniques(train,kfold.n_splits)
    train = oversample.no_change()
    
elif sample_technique =="SMOTEENN":
    X_train = oversample_techniques(X_train,kfold.n_splits).over_under_SMOTEENN()
    X_test = oversample_techniques(X_test,kfold.n_splits).over_under_SMOTEENN()
    
elif sample_technique =="SMOTETomek":
    X_train = oversample_techniques(X_train,kfold.n_splits).over_under_SMOTETomek()
    X_test = oversample_techniques(X_test,kfold.n_splits).over_under_SMOTETomek()


# # Cluster  - Check 

# In[33]:


from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage


# In[34]:


if CLUSTERING:
    cluster = KMeans(n_clusters= 7)
    y_cluster =cluster.fit_predict(train.drop("Cover_Type",axis =1))
    train["cluster"] = y_cluster

    y_cluster_test = cluster.predict(test)
    test["cluster"] = y_cluster_test
    test["cluster"].value_counts()


# # Optuna hyperparameter tuning 
# Optuna + cross val takes too long - will exclude any experiments with both enabled

# In[35]:


# 1. Define an objective function to be maximized.
def objective(trial):
    # 2. Suggest values of the hyperparameters using a trial object.
    lgb_params = {
        "is_unbalance": UNBALANCED,
        'objective': 'multiclass',
        "num_class": 7,
        'metric': "multi_logloss",
        'verbosity': -1,
        'num_iterations': EPOCHS,
        "num_threads": -1,
        #"force_col_wise": True
        "learning_rate": trial.suggest_float('learning_rate',0.01,0.2),
        'boosting_type': trial.suggest_categorical('boosting',[BOOSTING]),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        #'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1000, 10000),
        'max_depth': trial.suggest_int('max_depth', 2,15),
        #'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        #'bagging_freq': trial.suggest_int('bagging_freq', 1, 7)
    }
    
    if BOOSTING == "dart":
        lgb_params["drop_seed"]= 42
        
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "multi_logloss")
    
    model = lgb.train(params=lgb_params,train_set= train_data, 
                      valid_sets= [test_data], 
                      early_stopping_rounds = EARLY_STOPPING_ROUNDS,
                      callbacks=[pruning_callback]
                     )
    
    
    y_pred = model.predict(X_test)
    test_preds = [np.argmax(x) for x in y_pred]
    accuracy_s = accuracy_score(y_test,test_preds)
    print(f"Accuracy score of {accuracy_s}")
    print(classification_report(y_test,test_preds))
    
    return 1-accuracy_s


# ### Note 
# We are looking at the inverse accuracy (1-accuracy) which we want to minimize \
# we are doing this to keep the prunning callback in line with the objective metric

# In[36]:


get_ipython().run_cell_magic('time', '', 'if OPTUNA:\n            \n    train_data = lgb.Dataset(X_train, label=y_train)\n    test_data = lgb.Dataset(X_test, label=y_test)\n    \n    study = optuna.create_study(direction="minimize")\n    study.optimize(objective, n_trials=NUM_TRIALS)\n    trial = study.best_trial\n    \n    \n    #Print our results\n    print("Number of finished trials: {}".format(len(study.trials)))\n    print("Best trial:")\n    print(" Accuracy Value: {}".format(1- trial.value))\n    print("  Params: ")\n    for key, value in trial.params.items():\n        print("    {}: {}".format(key, value))\n\n    #write to log file\n    f= open(f"log_file_{version}_{minor}.txt","a")\n    f.write("\\n ##################  Hyper paramter tuning ###############")\n    f.write("\\n Number of finished trials: {}".format(len(study.trials)))\n    f.write(f"\\n Best trial accuracy score: {1-trial.value}")\n    f.write(f"\\n Best params: {trial.params}")\n    f.close()\n')


# In[37]:


if OPTUNA:
    lgb_params = trial.params
    lgb_params["is_unbalance"]= UNBALANCED
    lgb_params["objective"]= "multiclass"
    lgb_params["metric"]= "multi_logloss"
    lgb_params["num_class"]= 7
    lgb_params["device_type"]= DEVICE
    lgb_params["num_iterations"]= EPOCHS
    
else: #set to best params - see 'TEST DELETE notebook'
    lgb_params = {
    "is_unbalance": UNBALANCED,
    "objective" : "multiclass",
    "metric": "multi_logloss",
    "num_class": 7,
    #"num_threads": -1,
    #"force_col_wise": True
    "device_type": DEVICE,
    'boosting': BOOSTING,  
    'num_iterations': EPOCHS,
    "learning_rate": 0.16704206649880823,
    #"lambda_l1": 0.03469015403439412,
    "lambda_l2": 9.993162304351474,
    "num_leaves": 243,
    "max_depth": 12
                   }


# # Cross Validation 

# In[38]:


def cross_val(X,y, test):
    
    test_predictions = []
    lgb_scores = []

    for idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):

        print(10*"=", f"Fold={idx+1}", 10*"=")
        start_time = time.time()

        x_train, y_train = X.iloc[train_idx,:], y.iloc[train_idx,]
        x_valid, y_val = X.iloc[val_idx,:], y.iloc[val_idx,]

        if PSEUDOLABEL >0:
            #add pseduo data to each fold
            x_train = np.concatenate([x_train, pseudo_df.drop("Cover_Type", axis =1)], axis=0)
            y_train = np.concatenate([y_train, pseudo_df["Cover_Type"]], axis=0)
        
        if SCALER_NAME !="None":
            print(f"Scaling with {SCALER_NAME}")
            x_train,x_valid, test_scaled = scale_data(x_train,x_valid, test)
        else:
            test_scaled = test
        
        train_dat = lgb.Dataset(x_train, label=y_train)
        val_dat = lgb.Dataset(x_valid, label=y_val)
        
        del x_train
        del y_train

        model = lgb.train(params=lgb_params,
                          train_set= train_dat, 
                          valid_sets= [val_dat], 
                          early_stopping_rounds = EARLY_STOPPING_ROUNDS,
                          verbose_eval = -1
                     )

        preds_valid = model.predict(x_valid)
        preds_valid = [np.argmax(x) for x in preds_valid]
        accuracy_val = accuracy_score(y_val,  preds_valid)
        lgb_scores.append(accuracy_val)
        
        del x_valid
        del y_val
        
        run_time = time.time() - start_time
        print(f"Fold={idx+1}, accuracy: {accuracy_val}, Run Time: {run_time:.2f}")
        f.write(f"Fold={idx+1}, accuracy: {accuracy_val}, Run Time: {run_time:.2f}")

        test_preds = model.predict(test_scaled)
        test_preds = [np.argmax(x) for x in test_preds]
        test_predictions.append(test_preds)

    print("Mean Validation Accuracy :", np.mean(lgb_scores))
    return test_predictions


# In[39]:


if MODEL_TYPE == "Cross_validation":
    
    ##open log file
    f= open(f"log_file_{version}_{minor}.txt","a")
    f.write(f"########################### CROSSVAL SCORES ########################### ")
    
    test_predictions = cross_val(X,y, test)
    y_pred_test = np.squeeze(mode(np.column_stack(test_predictions),axis = 1)[0]).astype('int')
    
    #Close logs
    f.close()


# # Train Model - with best parms
# 
# If using Calibration - need to use LGBMClassifier as this is compatible with probability calibration - although much slower

# ## lgb.Train model 

# In[40]:


def scoring(model ):
    # train score 
    metric_score = model.best_score["valid_0"][METRIC]
    print(f"{METRIC} score of {metric_score}")
    
    train_probs = model.predict(X_train)
    train_preds = [np.argmax(x) for x in train_probs]
    accuracy_train = accuracy_score(y_train,train_preds)
    print(f"train accuracy score of {accuracy_train}")
    print("\n Train Classification Report:")
    print(classification_report(y_train,train_preds))
    
    #test score
    test_probs = model.predict(X_test)
    test_preds = [np.argmax(x) for x in test_probs]
    accuracy_s = accuracy_score(y_test,test_preds)
    print(f"Test accuracy score of {accuracy_s}")
    print("\n Test Classification Report:")
    print(classification_report(y_test,test_preds))
   
    #Write log
    f= open(f"log_file_{version}_{minor}.txt","a")
    f.write("\n ##################  SCORE ###############")
    f.write(f"\n {METRIC} score: {metric_score}")
    f.write(f"\n accuracy score: {accuracy_s}")
    f.close()
    return test_probs


# In[41]:


get_ipython().run_cell_magic('time', '', 'def train_model():   \n    \n    train_data = lgb.Dataset(X_train, label=y_train)\n    test_data = lgb.Dataset(X_test, label=y_test)\n    \n    model = lgb.train(params=lgb_params,\n                  train_set= train_data, \n                  valid_sets= [test_data], \n                  early_stopping_rounds = EARLY_STOPPING_ROUNDS,\n                  verbose_eval = -1\n                 )\n    \n    y_val_probs = scoring(model)\n    \n    #predict Test data\n    y_test_probs = model.predict(test_scaled)\n    \n    return model , y_test_probs,y_val_probs\n    \nif MODEL_TYPE == "train":\n    model, y_test_probs, y_val_probs = train_model()\n    y_pred_test = [np.argmax(x) for x in y_test_probs]\n')


# In[42]:


if MODEL_TYPE == "train":
    lgb.plot_importance(booster= model, figsize=(20,10))
    y_val = [np.argmax(x) for x in y_val_probs]
    
    #Confusion Matrix
    cm = confusion_matrix(y_test, y_val)
    ix = np.arange(cm.shape[0])
    cm[ix, ix] = 0
    col_names = [1,2,3,4,5,6,7]
    cm = pd.DataFrame(cm, columns=col_names, index=col_names)
    display(cm)


# ## lgb.LGBMClassifier model

# In[43]:


def scoring_model (model_i):
    y_pred_train = model_i.predict(X_train)
    accuracy_train = accuracy_score(y_train,y_pred_train)

    print(f"Train accuracy score of {accuracy_train}")
    print("\n Train classification report:")
    print(classification_report(y_train,y_pred_train))

    y_pred = model_i.predict(X_test)
    accuracy_test = accuracy_score(y_test,y_pred)

    print(f"Test accuracy score of {accuracy_test}")
    print("\n Test classification report:")
    print(classification_report(y_test,y_pred))
    
    f= open(f"log_file_{version}_{minor}.txt","a")
    f.write("\n ##################  Final Scoring ###############")
    f.write(f"\n Train accuracy score of {accuracy_train}")
    f.write(f"\n Test accuracy score of {accuracy_test}")
    f.write(f"\n Test Classification report: {classification_report(y_test,y_pred)}")
    f.close()
    
    return y_pred


# In[44]:


get_ipython().run_cell_magic('time', '', 'if MODEL_TYPE == "classifier" or MODEL_TYPE == "classifier_cal":\n        \n    lgb_clf = lgb.LGBMClassifier(**lgb_params)\n    lgb_clf.fit(\n        X_train,\n        y_train,\n        eval_set=(X_test,y_test),\n        early_stopping_rounds=EARLY_STOPPING_ROUNDS,\n        eval_metric= METRIC\n    )\n    y_pred = scoring_model(lgb_clf)\n    \n    y_pred_test = lgb_clf.predict(test_scaled)\n')


# In[45]:


if MODEL_TYPE == "classifier" or MODEL_TYPE == "classifier_cal":
    plt.figure(figsize=(20,7))
    plt.bar(X.columns, lgb_clf.feature_importances_)
    plt.xticks(rotation = 90)
    plt.title("Feature Importance")
    plt.show()


# # Post Processing - Probability Calibration 
# 
# sklearn calibration only works with LGBMClassifier

# In[46]:


if MODEL_TYPE == "classifier_cal":
    model = CalibratedClassifierCV(base_estimator = lgb_clf,  method = CALIBRATION_METHOD, cv="prefit") 
    model.fit(X_test, y_test) 

    y_pred = scoring_model(model)
    
    #predict Test data
    y_pred_test = model.predict(test_scaled)


# # Post Processing - Pseudo Labelling 3
# This creates pseudo data 2:
# Find the highest probabilities, set a threshold. If probabilities are > threshold \
# Keep rows as ground truth and retrain model (import data into Kaggle and rerun)

# In[47]:


if PSEUDOLABEL == 3:
    max_probabilities = []
    for val in y_test_probs:
        max_probabilities.append(max(val))
        
    new_train = test.copy(deep = True)
    new_train["Cover_Type"] = y_pred_test
    new_train["Cover_Type"] = new_train["Cover_Type"].astype("int32") +1
    new_train["Max_Proba"] = max_probabilities
    
    # Return only predictions > 0.99 
    new_train = new_train[new_train["Max_Proba"] >0.99]
    new_train.drop("Max_Proba", axis = 1)
    
    train = pd.concat([train,new_train])
    new_train.to_csv("psedo_labels_2.csv")


# # POST Full Training 
# As we are trying to get the most out of the model, we will do one final training on the full training data (no splitting)

# In[48]:


if Full_Train:
    
    X = np.concatenate((np.array(X),test),axis =0)
    y = np.concatenate((np.array(y),y_pred_test),axis =0)
    
    train_data = lgb.Dataset(X, label=y)
    
    model = lgb.train(params=lgb_params,
                  train_set= train_data, 
                  verbose_eval = -1
                 )
    
    y_probs = model.predict(X)
    y_pred = [np.argmax(x) for x in y_probs]
    print("accuracy check", accuracy_score(y,y_pred))
    print(classification_report(y,y_pred))
    
    #predict Test data
    y_test_probs = model.predict(test_scaled)
    y_pred_test = [np.argmax(x) for x in y_test_probs]


# # Submission 

# In[49]:


sub = pd.read_csv("../input/tabular-playground-series-dec-2021/sample_submission.csv", index_col=0)


# In[50]:


sub["Cover_Type"] = y_pred_test 

# Ensure interger column 
sub["Cover_Type"] = sub["Cover_Type"].astype("int32")

# reverse encoding on Target 
sub["Cover_Type"] = sub["Cover_Type"]+1


# In[51]:


sub.to_csv("submission.csv")


# In[52]:


sub["Cover_Type"].value_counts()


# In[53]:


sub.head(10)

