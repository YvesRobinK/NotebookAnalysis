#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-success">  
# </div>
# 
# <p style="border-bottom: 5px solid darkcyan"></p>
# 
# <div class="alert alert-success">  
#     <h1 align="center" style="color:darkcyan;">🧬Open Problems – Single-Cell Perturbations</h1> 
#     <h3 align="center" style="color:gray;">Predict how small molecules change gene expression in different cell types</h3> 
#     <h3 align="center" style="color:gray;">By: Somayyeh Gholami & Mehran Kazeminia</h3> 
# </div>
# 
# <p style="border-bottom: 5px solid darkcyan"></p>
# 
# # <div style="color:white;background-color:darkcyan;padding:1.5%;border-radius:15px 15px;font-size:1em;text-align:center">Feature Augmentation & Fragments of SMILES</div>
# 
# # <div style="color:yellow;display:inline-block;border-radius:5px;background-color:lightgray;font-block:Nexa;overflow:hidden"><p style="padding:15px;color:navy;overflow:hidden;font-size:70%;letter-spacing:0.5px;margin:0"><b> </b>Description for notebook number three :</p></div>
# 
# - This notebook is the continuation of [notebook number one](https://www.kaggle.com/code/mehrankazeminia/1-op2-eda-linearsvr-regressorchain).
# 
# - Two available features; are "cell_type" and "sm_name" and we want to add two new columns (two new features) to them.
# 
# - If we separate the cells based on 'cell_type' and assume that the drugs will usually have similar responses on each of these divisions, we can hope that by finding the average effects, we have obtained a new feature. For example, we will see that for y0 and the new feature of zero column, the correlation coefficient is 0.24. Of course, this amount is repeated for other columns as well.
# 
# - Also, if we separate the cells based on 'sm_name', we get a new feature by finding the average effects. In this case, for y0 and the new feature of column zero, the correlation coefficient is 0.62, and this value is almost repeated for other columns.
# 
# - Obviously, to add these two features, TrainData and TestData must be customized for each y column, and this may seem a bit complicated. For this reason, we first performed all the calculations only on column zero and then continued the main calculations in a loop with "range(y.shape[1])".
# 
# - By adding these two new features, the score of this notebook improved and probably the score of all notebooks that use the usual methods in machine learning (such as neural network, etc.) will be better.
# 
# - Of course, since the beginning of this challenge, many public notebooks have used averaging methods, but in these notebooks, the obtained values are directly considered as the answer.
# 
# - It should be noted that in the notebooks mentioned above, guesses are made to find the effect of averages or their combination, and these guesses will probably cause instability in the model as well as the risk of overfitting.
# 
# # <div style="color:yellow;display:inline-block;border-radius:5px;background-color:lightgray;font-block:Nexa;overflow:hidden"><p style="padding:15px;color:navy;overflow:hidden;font-size:70%;letter-spacing:0.5px;margin:0"><b> </b>Adding "fragments of SMILES" as a feature, has been done since version 8.</p></div>
# 
# We speculated that "fragments of SMILES" may represent a special feature to counteract the addition of drugs. So we did the same thing and added "fragments of SMILES" as a feature to the calculations. Because the results have improved, it has been included in our notebook from version eighth onwards. Please note that accuracy in splitting "SMILES" can probably improve the results. Also, because it seemed a little complicated, we tried to explain the matter a little more for the first line.
# 
# # <div style="color:yellow;display:inline-block;border-radius:5px;background-color:lightgray;font-block:Nexa;overflow:hidden"><p style="padding:15px;color:navy;overflow:hidden;font-size:70%;letter-spacing:0.5px;margin:0"><b> </b>Adding morgan fingerprint from SMILES, has been done since version 10.</p></div>
# 
# - Good luck.
# 
# ![](https://cdn-images-1.medium.com/max/1000/1*6lNZoZbkS_vBu54byPS3Yw.jpeg)
# 
# [Image Reference](https://www.a-star.edu.sg/gis/our-science/spatial-and-single-cell-systems)

# In[1]:


import warnings # suppress warnings
warnings.filterwarnings('ignore')
#:::::::::::::::::::::::::::::::::::
import os
import gc
import glob
import random
import numpy as np 
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from pathlib import Path
from itertools import groupby
#:::::::::::::::::::::::::::::::::::
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('ls ../input/*')


# <div class="alert alert-success">  
# </div>
# 
# <p style="border-bottom: 5px solid darkgray"></p>
# 
# <div>
#     <h1 align="center" style="color:gray;">Competition Data (Eight files)</h1>
# </div>

# # <div style="color:yellow;display:inline-block;border-radius:5px;background-color:pink;font-block:Nexa;overflow:hidden"><p style="padding:15px;color:darkred;overflow:hidden;font-size:70%;letter-spacing:0.5px;margin:0"><b> </b>(1) de_train.parquet</p></div>

# In[2]:


de_train = pd.read_parquet('../input/open-problems-single-cell-perturbations/de_train.parquet')
de_train.shape


# # <div style="color:yellow;display:inline-block;border-radius:5px;background-color:pink;font-block:Nexa;overflow:hidden"><p style="padding:15px;color:darkred;overflow:hidden;font-size:70%;letter-spacing:0.5px;margin:0"><b> </b>(7) id_map.csv</p></div>

# In[3]:


id_map = pd.read_csv('../input/open-problems-single-cell-perturbations/id_map.csv')
id_map.shape


# # <div style="color:yellow;display:inline-block;border-radius:5px;background-color:pink;font-block:Nexa;overflow:hidden"><p style="padding:15px;color:darkred;overflow:hidden;font-size:70%;letter-spacing:0.5px;margin:0"><b> </b>(8) sample_submission.csv</p></div>

# In[4]:


sample_submission = pd.read_csv('../input/open-problems-single-cell-perturbations/sample_submission.csv', index_col='id')
sample_submission.shape


# <div class="alert alert-success">  
# </div>
# 
# <p style="border-bottom: 5px solid darkgray"></p>
# 
# # <div style="color:yellow;display:inline-block;border-radius:5px;background-color:cyan;font-block:Nexa;overflow:hidden"><p style="padding:15px;color:navy;overflow:hidden;font-size:70%;letter-spacing:0.5px;margin:0"><b> </b>train | test | target</p></div>

# In[5]:


xlist  = ['cell_type','sm_name']
_ylist = ['cell_type','sm_name','sm_lincs_id','SMILES','control']

y = de_train.drop(columns=_ylist)
y.shape


# ### <span style="color:navy;">get_dummies (OneHotEncoder)</span>
# 
# <p style="border-bottom: 5px solid navy"></p>

# In[6]:


train = pd.get_dummies(de_train[xlist], columns=xlist)
train.shape


# In[7]:


test1 = pd.get_dummies(id_map[xlist], columns=xlist)
test1.shape


# ### <span style="color:navy;">Uncommon deleted</span>
# 
# <p style="border-bottom: 5px solid navy"></p>

# In[8]:


uncommon = [f for f in train if f not in test1]
len(uncommon)


# In[9]:


X1 = train.drop(columns=uncommon)
X1.shape[1], test1.shape[1]


# In[10]:


list(X1.columns) == list(test1.columns)


# <div class="alert alert-success">  
# </div>
# 
# <p style="border-bottom: 5px solid darkgray"></p>
# 
# # <div style="color:yellow;display:inline-block;border-radius:5px;background-color:cyan;font-block:Nexa;overflow:hidden"><p style="padding:15px;color:navy;overflow:hidden;font-size:70%;letter-spacing:0.5px;margin:0"><b> </b>Evaluation</p></div>
# 
# ### <span style="color:navy;">Mean Rowwise Root Mean Squared Error (MRRMSE)</span>

# In[11]:


def mrrmse_pd(y_pred: pd.DataFrame, y_true: pd.DataFrame):
    
    return ((y_pred - y_true)**2).mean(axis=1).apply(np.sqrt).mean()


# In[12]:


def mrrmse_np(y_pred, y_true):
    
    return np.sqrt(np.square(y_true - y_pred).mean(axis=1)).mean()


# <div class="alert alert-success">  
# </div>
# 
# <p style="border-bottom: 5px solid darkgray"></p>
# 
# # <div style="color:yellow;display:inline-block;border-radius:5px;background-color:navy;font-block:Nexa;overflow:hidden"><p style="padding:15px;color:lightgray;overflow:hidden;font-size:70%;letter-spacing:0.5px;margin:0"><b> </b>Feature Augmentation</p></div>

# In[13]:


de_cell_type = de_train.iloc[:, [0] + list(range(5, de_train.shape[1]))]
de_sm_name = de_train.iloc[:, [1] + list(range(5, de_train.shape[1]))]

de_cell_type.shape, de_sm_name.shape


# ### <span style="color:navy;">Calculate averages based on 'cell_type' and 'sm_name' for all columns</span>
# 
# <p style="border-bottom: 5px solid navy"></p>

# In[14]:


mean_cell_type = de_cell_type.groupby('cell_type').mean().reset_index()
mean_sm_name = de_sm_name.groupby('sm_name').mean().reset_index()

display(mean_cell_type)
display(mean_sm_name)


# ### <span style="color:navy;">Paste the results into the train file</span>
# 
# <p style="border-bottom: 5px solid navy"></p>

# In[15]:


rows = []
for name in de_cell_type['cell_type']:
    mean_rows = mean_cell_type[mean_cell_type['cell_type'] == name].copy()
    rows.append(mean_rows)

tr_cell_type = pd.concat(rows)
tr_cell_type = tr_cell_type.reset_index(drop=True)
tr_cell_type


# In[16]:


rows = []
for name in de_sm_name['sm_name']:
    mean_rows = mean_sm_name[mean_sm_name['sm_name'] == name].copy()
    rows.append(mean_rows)

tr_sm_name = pd.concat(rows)
tr_sm_name = tr_sm_name.reset_index(drop=True)
tr_sm_name


# ### <span style="color:navy;">Paste the results into the test file</span>
# 
# <p style="border-bottom: 5px solid navy"></p>

# In[17]:


rows = []
for name in id_map['cell_type']:
    mean_rows = mean_cell_type[mean_cell_type['cell_type'] == name].copy()
    rows.append(mean_rows)

te_cell_type = pd.concat(rows)
te_cell_type = te_cell_type.reset_index(drop=True)
te_cell_type


# In[18]:


rows = []
for name in id_map['sm_name']:
    mean_rows = mean_sm_name[mean_sm_name['sm_name'] == name].copy()
    rows.append(mean_rows)

te_sm_name = pd.concat(rows)
te_sm_name = te_sm_name.reset_index(drop=True)
te_sm_name


# <div class="alert alert-success">  
# </div>
# 
# <p style="border-bottom: 5px solid darkgray"></p>
# 
# # <div style="color:yellow;display:inline-block;border-radius:5px;background-color:pink;font-block:cyan;overflow:hidden"><p style="padding:15px;color:darkred;overflow:hidden;font-size:70%;letter-spacing:0.5px;margin:0"><b> </b>Sample: Column number zero - A1BG</p></div>

# In[19]:


y0 = y.iloc[:, 0].copy()
y0


# In[20]:


X0 = X1.join(tr_cell_type.iloc[:, 0+1]).copy()
X0 = X0.join(tr_sm_name.iloc[:, 0+1], lsuffix='_cell_type', rsuffix='_sm_name')
X0


# In[21]:


test0 = test1.join(te_cell_type.iloc[:, 0+1]).copy()
test0 = test0.join(te_sm_name.iloc[:, 0+1], lsuffix='_cell_type', rsuffix='_sm_name')
test0


# ### <span style="color:navy;">Correlation - Column #0</span>
# 
# <p style="border-bottom: 5px solid navy"></p>

# In[22]:


X0_corr = X0.copy()
X0_corr['y0'] = y0

corr = X0_corr.iloc[: , 131:].corr(numeric_only=True).round(3)
corr.style.background_gradient(cmap='Pastel1')


# In[23]:


cor_matrix = X0_corr.iloc[: , 131:].corr()
fig = plt.figure(figsize=(6,6));

cmap=sns.diverging_palette(240, 10, s=75, l=50, sep=1, n=6, center='light', as_cmap=False);
sns.heatmap(cor_matrix, center=0, annot=True, cmap=cmap, linewidths=5);
plt.suptitle('Train Set (Heatmap)', y=0.92, fontsize=16, c='darkred');
plt.show()


# ### <span style="color:navy;">LightGBM - Column #0</span>
# 
# <p style="border-bottom: 5px solid navy"></p>

# In[24]:


import lightgbm as lgb
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=0.20, random_state=421)


# In[25]:


model = lgb.LGBMRegressor()
model.fit(X_train, y_train)


# In[26]:


predict = model.predict(X_test) 
# mrrmse_pd(pd.DataFrame(predict), pd.DataFrame(y_test.values))


# In[27]:


N = 0
plt.style.use('seaborn-whitegrid') 
plt.figure(figsize=(8, 4), facecolor='lightyellow')
plt.title(f'Column:  #{N}', fontsize=12)
plt.gca().set_facecolor('lightgray')

sns.distplot(y_test.values-predict, bins=100, color='red')
plt.legend(['y_true','y_pred'], loc=1)
plt.show()


# <div class="alert alert-success">  
# </div>
# 
# <p style="border-bottom: 5px solid darkgray"></p>
# 
# # <div style="color:yellow;display:inline-block;border-radius:5px;background-color:cyan;font-block:Nexa;overflow:hidden"><p style="padding:15px;color:navy;overflow:hidden;font-size:70%;letter-spacing:0.5px;margin:0"><b> </b>Feature Augmentation - Based on : 'SMILES'</p></div>
# 
# # <div style="color:yellow;display:inline-block;border-radius:5px;background-color:pink;font-block:Nexa;overflow:hidden"><p style="padding:15px;color:darkred;overflow:hidden;font-size:70%;letter-spacing:0.5px;margin:0"><b> </b>Add column : id_map['SMILES']</p></div>

# In[28]:


sm_name_smiles_dict = de_train.set_index('sm_name')['SMILES'].to_dict()

id_map['SMILES'] = id_map['sm_name'].map(sm_name_smiles_dict)
id_map['SMILES']


# In[29]:


def split_sign(text):
    text = text.replace(')(', ' ')
    text = text.replace('(' , ' ')
    text = text.replace(')' , ' ')
    return text.split(" ")


# ### <span style="color:navy;">An example for the method of splitting each 'SMILES'</span>
# 
# <p style="border-bottom: 5px solid navy"></p>

# In[30]:


pip install rdkit


# In[31]:


from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem


# In[32]:


import re
# Thanks to: https://github.com/DocMinus

def element_count(data, N):
    smiles = data['SMILES'].iloc[N]

    pattern = "Si|Ti|Al|Zn|Pd|Pt|Br?|Cl?|N|O|S|P|F|I|B|b|c|n|o|s|p" 
    regex = re.compile(pattern)
    elements = [token for token in regex.findall(smiles)]
    ele_length = len(elements)
    lowercase_pattern = "b|c|n|o|s|p"
    regex_low = re.compile(lowercase_pattern)
    for i in range(ele_length):
        if regex_low.findall(elements[i]):
            elements[i] = elements[i].upper()
    element_count = [elements.count(ele) for ele in elements]
    formula = dict(zip(elements, element_count)) 

    print('==> Element_Count: ', str(formula))


# In[33]:


cols_sel = ['cell_type','sm_name','sm_lincs_id','SMILES']

smiles = de_train['SMILES'][0]
mol = Chem.MolFromSmiles(smiles)
img = Draw.MolToImage(mol, size=(700, 300), fitImage=True)    
display(pd.DataFrame(de_train[cols_sel].iloc[0]))

element_count(de_train, 0)
img


# In[34]:


smiles0 = de_train['SMILES'][0]
smiles0


# In[35]:


split0 = split_sign(smiles0)
split0


# In[36]:


mol = Chem.MolFromSmiles(split0[0])
img = Draw.MolToImage(mol, size=(700, 300), fitImage=True)   
img


# In[37]:


mol = Chem.MolFromSmiles(split0[1])
img = Draw.MolToImage(mol, size=(700, 300), fitImage=True)   
img


# ### <span style="color:navy;">Feature Augmentation - Based on : 'SMILES' - train file</span>
# 
# <p style="border-bottom: 5px solid navy"></p>

# In[38]:


de_train['_SMILES'] = [split_sign(text) for text in de_train['SMILES'].values]
de_train['_SMILES']


# In[39]:


sign = []
for row in de_train['_SMILES'].values:
    for ele in row:
        sign.append(ele)
        
de_train_sign_list = list(set(sign))
len(de_train_sign_list)


# In[40]:


data = np.zeros((len(de_train), len(de_train_sign_list)), dtype=int)
de_train_sign = pd.DataFrame(data=data, columns=de_train_sign_list)

for sign in de_train_sign_list:
    for i in range(len(de_train)):
        row = de_train['_SMILES'].values[i]
        
        if (sign in row):
            de_train_sign[sign].iloc[i] = 1
            
de_train_sign


# ### <span style="color:navy;">Feature Augmentation - Based on : 'SMILES' - test file</span>
# 
# <p style="border-bottom: 5px solid navy"></p>

# In[41]:


id_map['_SMILES'] = [split_sign(text) for text in id_map['SMILES'].values]
id_map['_SMILES']


# In[42]:


sign = []
for row in id_map['_SMILES'].values:
    for ele in row:
        sign.append(ele)
        
id_map_sign_list = list(set(sign))
len(id_map_sign_list)


# In[43]:


data = np.zeros((len(id_map), len(id_map_sign_list)), dtype=int)
id_map_sign = pd.DataFrame(data=data, columns=id_map_sign_list)

for sign in id_map_sign_list:
    for i in range(len(id_map)):
        row = id_map['_SMILES'].values[i]
        
        if (sign in row):
            id_map_sign[sign].iloc[i] = 1
            
id_map_sign


# ### <span style="color:navy;">Uncommon Deleted</span>
# 
# <p style="border-bottom: 5px solid navy"></p>

# In[44]:


uncommon = [f for f in de_train_sign if f not in id_map_sign]
len(uncommon)


# In[45]:


train_sign = de_train_sign.drop(columns=uncommon)
train_sign.shape, id_map_sign.shape


# In[46]:


list(train_sign.columns) == list(id_map_sign.columns)


# In[47]:


train_sign = train_sign.sort_index(axis = 1)
id_map_sign = id_map_sign.sort_index(axis = 1)

list(train_sign.columns) == list(id_map_sign.columns)


# In[48]:


X2 = X1.join(train_sign).copy()
X2.shape


# In[49]:


test2 = test1.join(id_map_sign).copy()
test2.shape


# <div class="alert alert-success">  
# </div>
# 
# <p style="border-bottom: 5px solid darkgray"></p>
# 
# # <div style="color:yellow;display:inline-block;border-radius:5px;background-color:cyan;font-block:Nexa;overflow:hidden"><p style="padding:15px;color:navy;overflow:hidden;font-size:70%;letter-spacing:0.5px;margin:0"><b> </b>Create a morgan fingerprint from SMILES</p></div>

# In[50]:


radius = 2
nBits = 2048

def get_frag(smiles_str):   
    mol = Chem.MolFromSmiles(smiles_str)
    frag = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return frag


# In[51]:


columns = np.arange(nBits).astype(str)

data1 = np.zeros((len(de_train), nBits), dtype=int)
data2 = np.zeros((len(id_map), nBits), dtype=int)

df_frag1 = pd.DataFrame(data=data1, columns=columns)
df_frag2 = pd.DataFrame(data=data2, columns=columns)

df_frag1.shape, df_frag2.shape


# In[52]:


for f in range(len(df_frag1)):
    df_frag1.iloc[f] = list(get_frag(de_train['SMILES'][f]))
    
df_frag1.shape


# In[53]:


for f in range(len(df_frag2)):
    df_frag2.iloc[f] = list(get_frag(id_map['SMILES'][f]))
    
df_frag2.shape


# In[54]:


X = X2.join(df_frag1).copy()
X.shape


# In[55]:


test = test2.join(df_frag2).copy()
test.shape


# <div class="alert alert-success">  
# </div>
# 
# <p style="border-bottom: 5px solid darkgray"></p>
# 
# # <div style="color:yellow;display:inline-block;border-radius:5px;background-color:navy;font-block:Nexa;overflow:hidden"><p style="padding:15px;color:lightgray;overflow:hidden;font-size:70%;letter-spacing:0.5px;margin:0"><b> </b>KNN & LinearSVR - Final model</p></div>

# In[56]:


model0 = lgb.LGBMRegressor()
model1 = KNeighborsRegressor(n_neighbors=13)
model2 = LinearSVR(max_iter= 2000, epsilon= 0.1)


# In[57]:


pred = []
for i in range(y.shape[1]):
    
    yi = y.iloc[:, i].copy()
       
    Xi = X.join(tr_cell_type.iloc[:, i+1], lsuffix='_', rsuffix='__').copy()
    Xi = Xi.join(tr_sm_name.iloc[:, i+1], lsuffix='_cell_type', rsuffix='_sm_name')
    
    testi = test.join(te_cell_type.iloc[:, i+1], lsuffix='_', rsuffix='__').copy()
    testi = testi.join(te_sm_name.iloc[:, i+1], lsuffix='_cell_type', rsuffix='_sm_name')
    
    model1.fit(Xi, yi)
    model2.fit(Xi, yi)
    
    pred1 = model1.predict(testi)
    pred2 = model2.predict(testi)    
    pred.append((pred1 *0.40) + (pred2 *0.60))
    
len(pred)


# In[58]:


de_train = pd.read_parquet('../input/open-problems-single-cell-perturbations/de_train.parquet')

prediction = pd.DataFrame(pred).T
prediction.columns = de_train.columns[5:]
prediction.index.name = 'id'
prediction


# In[59]:


prediction.to_csv('prediction.csv')
get_ipython().system('ls')


# <div class="alert alert-success">  
# </div>
# 
# <p style="border-bottom: 5px solid darkgray"></p>
# 
# # <div style="color:yellow;display:inline-block;border-radius:5px;background-color:cyan;font-block:Nexa;overflow:hidden"><p style="padding:15px;color:navy;overflow:hidden;font-size:70%;letter-spacing:0.5px;margin:0"><b> </b>Ensembling</p></div>

# Several public notebooks presented different methods based on averaging, different guesses, etc. The following file is actually the optimization and ensembling of these results.
# 
# Thanks to: **@alexandervc** 

# In[60]:


import1 = pd.read_csv('../input/op2-603/op2_603.csv', index_col='id')
import1.shape


# The file below is the result of a great notebook that uses the "Autoencoder" method.
# 
# Thanks to: **@vendekagonlabs**

# In[61]:


import2 = pd.read_csv('../input/op2-720/op2_720.csv', index_col='id')
import2.shape


# The file below is the result of a great notebook that uses the "Neural Network" method.
# 
# Thanks to: **@kishanvavdara** & **@liudacheldieva**

# In[62]:


import3 = pd.read_csv('../input/op-v3-neural-network-regression/submission_df.csv', index_col='id')
import3.shape


# The file below is the result of a great notebook that uses the "NLP" method.
# 
# Thanks to: **@kishanvavdara** & **@erotar**

# In[63]:


import4 = pd.read_csv('../input/op2-600-nlp/submission.csv', index_col='id')
import4.shape


# In[64]:


col = list(de_train.columns[5:])
submission = sample_submission.copy()

submission[col] = (import1[col] *0.30) + (import2[col] *0.10) + (import3[col] *0.25) + (import4[col] *0.25) + (prediction[col] *0.10)
submission.shape


# In[65]:


submission.to_csv('submission.csv')
get_ipython().system('ls')


# ### <span style="color:navy;">Ensembling histograms for a random column</span>
# 
# <p style="border-bottom: 5px solid navy"></p>

# In[66]:


N = random.randrange(y.shape[1])

print(':' *40)
print('Column number :', N)
print('Column name :', list(y.columns)[N])
print(':' *40)

hist_data = [submission.iloc[:, N], import1.iloc[:, N], import2.iloc[:, N], import3.iloc[:, N], import4.iloc[:, N], prediction.iloc[:, N]]
group_labels = ['Submission', 'Mean & Paste', 'Autoencoder', 'Neural Network', 'NLP', 'Prediction']
    
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_hist=False, show_rug=False)
fig.show()


# <div class="alert alert-success">  
# </div>
# 
# <p style="border-bottom: 5px solid darkgray"></p>
