#!/usr/bin/env python
# coding: utf-8

# ### Baseline Tasks to Handle
#    - **Pipline**
#         1. **Data Exploration**
#         2. **Preprocessing**
#         3. **Model Selection**
#            1. Model Tuning
#            2. Model Evaluation
#         2. **Submission**
# <div class="alert alert-info">
#   <strong>The goal of competition</strong> , the goal of the competition is to predict MDS-UPDR scores, which measure the progression of Parkinson's disease in patients. The MDS-UPDRS is a widely used assessment tool for Parkinson's disease, which evaluates both motor and non-motor symptoms associated with the disease
# </div>

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install ydata_profiling\n!pip install fasteda\n')


# In[2]:


## Loading Packages 
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from fasteda import fast_eda
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')


# 
# 1. **Data Exploration**
#       1. Load the data: Start by loading the data provided for the competition, 
#       2. Check for missing data: Missing data can cause issues during modeling, so it's important to check for any missing data in the dataset. 
#       3. Analyze the distribution of features: It's important to understand the distribution of the features in the dataset. This will help you decide if you need to perform any scaling or normalization on the data before modeling. 
#       <div class="alert alert-warning">
#   <strong>Package Used fast_eda and ydata_profiling </strong> for not losing time during Full EDA from Sratch will explore some Built Packages has all nessecery functionalities to AutoEDA 
# </div>

# In[3]:


proteins = pd.read_csv("/kaggle/input/amp-parkinsons-disease-progression-prediction/train_proteins.csv")
peptides = pd.read_csv("/kaggle/input/amp-parkinsons-disease-progression-prediction/train_peptides.csv")
clinical = pd.read_csv("/kaggle/input/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv")


# #### Generate General Report of Clinical dataset 
#    * Using Fast Eda to Exploartion each dataset of training 
#       **proteins and peptides**

# In[4]:


profile = ProfileReport(clinical, title="Profiling Report")
profile.to_notebook_iframe()


# In[5]:


profile.to_file("Profiling Repor.html")


# In[6]:


fast_eda(proteins)


# In[7]:


fast_eda(peptides)


# ### **Preprocessing**
#    - Analyze the relationships between features: In addition to analyzing the distribution of individual features, it's important to understand how different features are related to each other. You can use scatter plots or correlation matrices to analyze the relationships between features. This will help you decide which features to include in the model and how to perform feature engineering.
# 
#  - Feature Engineering: Feature engineering is the process of creating new features or transforming existing features to improve the performance of the model. In this competition, you may need to perform feature engineering on the protein and peptide levels over time to extract useful information for predicting MDS-UPDR scores.
# 
#  - Split the data into train, validation, and test sets: Once you have preprocessed the data and performed feature engineering, you can split the data into a training set, a validation set, and a test set. The training set will be used to train the model, the validation set will be used to tune the hyperparameters of the model, and the test set will be used to evaluate the performance of the final model.

# In[8]:


df_0 = clinical[(clinical.visit_month == 0)][['visit_id','updrs_1']]
print('Train shape:', df_0.shape)
df_0.head()


# In[9]:


proteins_npx_ft = proteins.groupby('visit_id').agg(NPX_min=('NPX','min'), NPX_max=('NPX','max'), NPX_mean=('NPX','mean'), NPX_std=('NPX','std'))\
                .reset_index()
proteins_npx_ft.head()


# In[10]:


df_proteins = pd.merge(proteins, df_0, on = 'visit_id', how = 'inner').reset_index()
proteins_Uniprot_updrs = df_proteins.groupby('UniProt').agg(updrs_1_sum = ('updrs_1','mean')).reset_index()
proteins_Uniprot_updrs.head()


# In[11]:


df_proteins = pd.merge(proteins, proteins_Uniprot_updrs, on = 'UniProt', how = 'left')
proteins_UniProt_ft = df_proteins.groupby('visit_id').agg(proteins_updrs_1_min=('updrs_1_sum','min'), proteins_updrs_1_max=('updrs_1_sum','max'),\
                                                          proteins_updrs_1_mean=('updrs_1_sum','mean'), proteins_updrs_1_std=('updrs_1_sum','std'))\
                .reset_index()
proteins_UniProt_ft.head()


# In[12]:


peptides.head()


# In[13]:


peptides_PeptideAbundance_ft = peptides.groupby('visit_id').agg(Abe_min=('PeptideAbundance','min'), Abe_max=('PeptideAbundance','max'),\
                                                                Abe_mean=('PeptideAbundance','mean'), Abe_std=('PeptideAbundance','std'))\
                .reset_index()
peptides_PeptideAbundance_ft.head()


# In[14]:


df_peptides = pd.merge(peptides, df_0, on = 'visit_id', how = 'inner').reset_index()
peptides_PeptideAbundance_updrs = df_peptides.groupby('Peptide').agg(updrs_1_sum = ('updrs_1','mean')).reset_index()
peptides_PeptideAbundance_updrs.head()


# In[15]:


df_peptides = pd.merge(peptides, peptides_PeptideAbundance_updrs, on = 'Peptide', how = 'left')
peptides_ft = df_peptides.groupby('visit_id').agg(peptides_updrs_1_min=('updrs_1_sum','min'), peptides_updrs_1_max=('updrs_1_sum','max'),\
                                                          peptides_updrs_1_mean=('updrs_1_sum','mean'), peptides_updrs_1_std=('updrs_1_sum','std'))\
                .reset_index()
peptides_ft


# In[16]:


df_0_1 = clinical[(clinical.visit_month == 3)][['visit_id','updrs_1']]
df_0_2 = clinical[(clinical.visit_month == 3)][['visit_id','updrs_2']]
df_0_3 = clinical[(clinical.visit_month == 3)][['visit_id','updrs_3']]
df_0_4 = clinical[(clinical.visit_month == 3)][['visit_id','updrs_4']]

df_proteins = pd.merge(proteins, df_0_1, on = 'visit_id', how = 'inner').reset_index()
proteins_Uniprot_updrs1 = df_proteins.groupby('UniProt').agg(updrs_1_sum = ('updrs_1','mean')).reset_index()

df_proteins = pd.merge(proteins, df_0_2, on = 'visit_id', how = 'inner').reset_index()
proteins_Uniprot_updrs2 = df_proteins.groupby('UniProt').agg(updrs_1_sum = ('updrs_2','mean')).reset_index()

df_proteins = pd.merge(proteins, df_0_3, on = 'visit_id', how = 'inner').reset_index()
proteins_Uniprot_updrs3 = df_proteins.groupby('UniProt').agg(updrs_1_sum = ('updrs_3','mean')).reset_index()

df_proteins = pd.merge(proteins, df_0_4, on = 'visit_id', how = 'inner').reset_index()
proteins_Uniprot_updrs4 = df_proteins.groupby('UniProt').agg(updrs_1_sum = ('updrs_4','mean')).reset_index()

df_peptides = pd.merge(peptides, df_0_1, on = 'visit_id', how = 'inner').reset_index()
peptides_PeptideAbundance_updrs1 = df_peptides.groupby('Peptide').agg(updrs_1_sum = ('updrs_1','mean')).reset_index()

df_peptides = pd.merge(peptides, df_0_2, on = 'visit_id', how = 'inner').reset_index()
peptides_PeptideAbundance_updrs2 = df_peptides.groupby('Peptide').agg(updrs_1_sum = ('updrs_2','mean')).reset_index()

df_peptides = pd.merge(peptides, df_0_3, on = 'visit_id', how = 'inner').reset_index()
peptides_PeptideAbundance_updrs3 = df_peptides.groupby('Peptide').agg(updrs_1_sum = ('updrs_3','mean')).reset_index()

df_peptides = pd.merge(peptides, df_0_4, on = 'visit_id', how = 'inner').reset_index()
peptides_PeptideAbundance_updrs4 = df_peptides.groupby('Peptide').agg(updrs_1_sum = ('updrs_4','mean')).reset_index()

df_proteins_fts = [proteins_Uniprot_updrs1, proteins_Uniprot_updrs2, proteins_Uniprot_updrs3, proteins_Uniprot_updrs4]
df_peptides_fts = [peptides_PeptideAbundance_updrs1, peptides_PeptideAbundance_updrs2, peptides_PeptideAbundance_updrs3, peptides_PeptideAbundance_updrs4]
df_lst = [df_0_1, df_0_2, df_0_3, df_0_4]


# In[17]:


def features(df, proteins, peptides, classes):
    proteins_npx_ft = proteins.groupby('visit_id').agg(NPX_min=('NPX','min'), NPX_max=('NPX','max'), NPX_mean=('NPX','mean'), NPX_std=('NPX','std'))\
                    .reset_index()
    peptides_PeptideAbundance_ft = peptides.groupby('visit_id').agg(Abe_min=('PeptideAbundance','min'), Abe_max=('PeptideAbundance','max'),\
                                                                    Abe_mean=('PeptideAbundance','mean'), Abe_std=('PeptideAbundance','std'))\
                    .reset_index()

    df_proteins = pd.merge(proteins, df_proteins_fts[classes], on = 'UniProt', how = 'left')
    proteins_UniProt_ft = df_proteins.groupby('visit_id').agg(proteins_updrs_1_min=('updrs_1_sum','min'), proteins_updrs_1_max=('updrs_1_sum','max'),\
                                                              proteins_updrs_1_mean=('updrs_1_sum','mean'), proteins_updrs_1_std=('updrs_1_sum','std'))\
                    .reset_index()
    df_peptides = pd.merge(peptides, df_peptides_fts[classes], on = 'Peptide', how = 'left')
    peptides_ft = df_peptides.groupby('visit_id').agg(peptides_updrs_1_min=('updrs_1_sum','min'), peptides_updrs_1_max=('updrs_1_sum','max'),\
                                                              peptides_updrs_1_mean=('updrs_1_sum','mean'), peptides_updrs_1_std=('updrs_1_sum','std'))\
                    .reset_index()

    df = pd.merge(df, proteins_npx_ft, on = 'visit_id', how = 'left')
    df = pd.merge(df, peptides_PeptideAbundance_ft, on = 'visit_id', how = 'left')
    df = pd.merge(df, proteins_UniProt_ft, on = 'visit_id', how = 'left')
    df = pd.merge(df, peptides_ft, on = 'visit_id', how = 'left')
    df = df.fillna(df.mean())
    return df


# In[18]:


train_0 = features(df_0_1, proteins, peptides, 0)
train_0.head(3)


# ### Model Selection 
#    - Split the data into training, validation, and test sets: Before select model selection,
#     
#    - Tune hyperparameters: For each candidate model, you need to tune their hyperparameters to improve their performance. Hyperparameters are parameters that are set before training the model and cannot be learned from the data. in our case will use 
#       1. **GradientBoostingRegressor** 
#       2. **Neural Network Regression** 
# 
# 
#   - Select the best model: trained and evaluated all the candidate models, and tuned their hyperparameters, you can select the best model based on their performance on the validation set.retrain the best model on the training and validation sets combined and evaluate its performance on the test set.
# 
#   - Validate your results: Finally, it's important to validate our results by testing the performance of our best model on unseen data. If the performance on the test set is significantly worse than the performance on the validation set, it's possible that you overfit the hyperparameters to the validation set.

# In[19]:


#Evaluation metric
def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * np.mean(smap)


# In[20]:


# Preprocess data
mms = MinMaxScaler()
train_0 = features(df_lst[0], proteins, peptides, 0)
scale_col = ['NPX_min','NPX_max','NPX_mean','NPX_std', 'Abe_min', 'Abe_max', 'Abe_mean', 'Abe_std']
train_0[scale_col] = mms.fit_transform(train_0[scale_col])
X_train = train_0.drop(columns = ['visit_id','updrs_1'], axis = 1)
y_train = train_0['updrs_1']
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create PyTorch data loaders
train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
val_loader = DataLoader(val_dataset, batch_size=32)

# Define neural network architecture
class UPDRSNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(UPDRSNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 4)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize neural network
updrs_net = UPDRSNet(X_train.shape[1], 64)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(updrs_net.parameters(), lr=0.01)

# Train neural network
for epoch in range(500):
    train_loss = 0.0
    val_loss = 0.0

    # Train on training set
    updrs_net.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = updrs_net(inputs)
        loss = criterion(outputs.view(-1,1), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    # Evaluate on validation set
    updrs_net.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            outputs = updrs_net(inputs)
#             metric = smape(outputs.view(-1,1),labels.unsqueeze(1))
            loss = criterion(outputs.view(-1,1), labels)
            val_loss += loss.item() * inputs.size(0)
    if epoch % 30 == 0 : 
        print(f"Epoch {epoch+1} train loss: {train_loss/len(train_dataset):.2f}, val loss: {val_loss/len(val_dataset):.2f}")


# ### Submission and Evaluation
# 1. **Model Evaluation:** 
#     Once the model is trained and tuned, we will need to evaluate its performance on the test set. This involves making predictions on the test set and calculating the evaluation metric specified in the competition rules. It is important to ensure that the model generalizes well to new data and performs well on the test set.
# 
# 2. **Submission:**
#     Finally, we will need to make a submission to the competition by submitting your predictions for the test set. You should ensure that the submission file is in the correct format and that it meets the submission guidelines specified in the competition rules.

# In[21]:


updrs_3_pred = {}
up3 = clinical[['visit_month','updrs_3']].drop_duplicates(['visit_month','updrs_3'])
updrs_3_pred = dict(zip(up3.visit_month, up3.updrs_3))
updrs_3_pred


# In[22]:


import amp_pd_peptide
env = amp_pd_peptide.make_env()
iter_test = env.iter_test()


# In[23]:


def map_test(x):
    updrs = x.split('_')[2] + '_' + x.split('_')[3]
    month = int(x.split('_plus_')[1].split('_')[0])
    visit_id = x.split('_')[0] + '_' + x.split('_')[1]
    # set all predictions 0 where updrs equals 'updrs_4'
    if updrs=='updrs_3':
        rating = updrs_3_pred[month]
    elif updrs=='updrs_4':
        rating = 0
    elif updrs =='updrs_1':
        rating = df[df.visit_id == visit_id]['pred0'].values[0]
    else:
        rating = df[df.visit_id == visit_id]['pred1'].values[0]
    return rating

counter = 0
# The API will deliver four dataframes in this specific order:
for (test, test_peptides, test_proteins, sample_submission) in iter_test:
    df = test[['visit_id']].drop_duplicates('visit_id')
    pred_0 = features(df[['visit_id']], test_proteins, test_peptides, 0)
    scale_col = ['NPX_min','NPX_max','NPX_mean','NPX_std', 'Abe_min', 'Abe_max', 'Abe_mean', 'Abe_std']
    pred_0[scale_col] = mms.fit_transform(pred_0[scale_col])
    pred_0 = pred_0.drop(columns = ['visit_id'], axis = 1)
    pred_0_ = np.asarray(pred_0)
    pred_0_ = torch.from_numpy(pred_0_)
    updrs_net.eval()
    with torch.no_grad():
        pred_0 = updrs_net(pred_0_.float())
        df['pred0'] = torch.argmin(pred_0,dim=1)
    pred_1 = features(df[['visit_id']], test_proteins, test_peptides, 1)
    scale_col = ['NPX_min','NPX_max','NPX_mean','NPX_std', 'Abe_min', 'Abe_max', 'Abe_mean', 'Abe_std']
    pred_1[scale_col] = mms.fit_transform(pred_1[scale_col])
    pred_1 = pred_1.drop(columns = ['visit_id'], axis = 1)
    pred_1_ = np.asarray(pred_1).astype(np.float64)
    pred_1_ = torch.from_numpy(pred_1_)
    updrs_net.eval()
    with torch.no_grad():
        pred_1 = updrs_net(pred_1_.float())
        df['pred1'] = torch.argmin(pred_1,dim=1)

    sample_submission['rating'] = sample_submission['prediction_id'].apply(map_test)
    env.predict(sample_submission)
    
    if counter == 0:
        display(test)
        display(sample_submission)
        
    counter += 1     

