#!/usr/bin/env python
# coding: utf-8

# # XGBoost - LB 0.40+ CV 0.40+ Train With Kaggle Data!
# # Download 5000 Single Point Mutations and 200 PDB Files!
# In notebook version 15 (and 14), we show you how to process Kaggle's train data and train a model using Kaggle's train data. This is an exciting accomplishment. Currently, as of today, all public notebooks better than LB 0.200 **do not use** Kaggle train data. Instead they use models and techniques from the internet to predict mutation stability. In this notebook we use only Kaggle train and achieve LB 0.35+
# 
# The notebook you are reading is the first notebook to use only Kaggle train data and score over LB 0.200! (Specifically, it achieves LB 0.359 single model, woohoo!). Many Kagglers have contributed to making this a success! Including Robert Hatch, GreySnow, Rope on Mars, Vladimir Slaykovskiy, Jinyuan Sun, Kaggleqrdl, Lucas Morin, Moth! It is quite tricky to utilize Kaggle's train data. The process is explained in detail [here][5] and summarized in the diagram below:
# 
# ![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/main/Oct-2022/k_train3.png)
# 
# In previous notebook versions, we built a pipeline to train XGB using features extracted from protein sequence and structure. Also in previous notebook versions, we demonstrated how to download and use external data. In notebook version 13 we added transformer embedding features. In notebook version 11, we download 5000 Single Point Mutations and 200 PDB files from Jinyuan Sun's GitHub [here][1] and discussion [here][4]. We then train a model to predict `dTm` from single point mutations with `dTm` targets and achieve 5-Fold CV Spearman correlation coefficient 0.26 on `dTm`. Next we use this model to predict a holdout dataset's `ddG` and achieve Spearman correlation coefficient 0.26 on `ddG`. Finally we predict Kaggle's test data using this model and achieve LB 0.192. In notebook version 12, we train with both dTm and ddG targets (mixed together) and achieve dTm CV 0.194, ddG CV 0.460, LB 0.209. Notebook version 13 achieves single model LB 0.3! In notebook version 13, we submit the ensemble of versions 8, 11, 12, and 13 and achieve LB 0.34!
# 
# An advantage of dowloading data is that we can train with PDB files and then utilize Kaggle's PDB test file during inference. But note we train with PDB files from Protein Data Bank and Kaggle provides PDB file from Alpha Fold for inference. These PDB files were **not** created in the same way. See discussion [here][2]. In order to have inference match training, we load our own test data wild type PDB file created from the internet as described [here][3]. This new PDB file has `b_factor` more similar to the real `b_factor` that we train on. However `atom_df` rows containing Hydrogen atoms are still different (between AlphaFold PDBs and Protein Data Bank PDBs), so be careful with feature engineering. If we submit this real `b_factor` (which isn't Alpha Fold's `pLDDT`) it scores LB 0.139 by itself. Our XGB model achieves a better LB, so it means that our model is learning something!
# 
# ![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/main/Oct-2022/t_model.png)
# 
# # Notes
# This notebook is a work in progress. The most helpful aspects of this notebook is that it shows (1) how to use Kaggle's train data, (2) how to extract features from ESM transformer, and (3) how to download 5000 more mutations from the internet and their corresponding 200 PDB files. Specifically 4000 of these mutations, have `ddG` targets and 1000 of these mutations have `dTm` targets. We must decide how to best use this external data and the two types of targets. Furthermore we must be careful training with Protein Data Bank PDB and inferring with AlphaFold PDB. 
# 
# * **Versions 1-6** use Kaggle's PDB with `b_factor` column different than Protein Data Bank's PDB `b_factor` column, so we can ignore those `submission.csv`.
# * **Version 7 LB 0.146** trains/KFold-validates on `ddG` and holdout-validates on `dTM` and uses an updated PDB for inference with a `b_factor` column that is similar to training data. However many features still do not transfer well from train PDB to infer PDB because the Protein Data Bank's PDB `atom_df` dataframe excludes many hydrogen atoms. Therefore feature engineering needs to be updated. We need to only use rows with `atom_name` isin `['N','H','CA','O']` when counting atoms and/or creating features from `x_coord`, `y_coord`, `z_coord`. (Discussion [here][2])
# * **Version 8 LB 0.173** trains/Kfold-validates on `dTm` and holdout-validates on `ddG`. Uses correct `b_factor` in both train and infer. Uses features that don't work well for train and infer (because of atom PDB differences). Furthermore Version 8 model scores LB 0.173 by itself, but Version 8 notebook submits a 50/50 ensemble of version 7 and version 8 which has benefits of train on both `ddG` and `ddT` and achieves LB 0.196.
# * **Version 9-11 LB LB 0.192**  trains/Kfold-validates on `dTm` and holdout-validates on `ddG`. Achieves CV 0.260 on both targets. Uses features that exclude hydrogen atoms in PDB since Alpha Fold PDB `atom_df` contains more hydrogen atom rows than corresponding Protein Data Bank PDBs. Also uses features that should be stable between train and test. 
# * **Version 12 LB 0.209** trains/Kfold-validates on all 5000 mutations! It trains with mixture of rank normalized `dTm` and `ddG` targets. Achieves CV 0.194 on `dTM` proteins and CV 0.460 on `ddG` proteins. Uses same features as versions 9-11 (but more train data and both targets). Furthermore Version 12 model scores LB 0.209 by itself, but Version 12 notebook submits a 33/33/33 ensemble of version 8, 11, 12 and achieves LB 0.298 woohoo!
# * **Version 13 LB 0.300** trains/Kfold-validates on all 5000 mutations! It trains with mixture of rank normalized `dTm` and `ddG` targets. Achieves CV 0.233 on `dTM` proteins and CV 0.619 on `ddG` proteins. Amazing `ddG` CV score! Uses same features as versions 12-13 and adds additional transformer embedding features. Furthermore Version 14 model scores LB 0.300 by itself! but Version 14 notebook submits a 50/50 ensemble of version 12 and 14 and achieves LB ???
# * **Version 14 LB 0.307** Woohoo! We finally trained a model and achieved a good CV LB using only Kaggle train data. This is a huge milestone which took collaboration from many Kagglers and 2 weeks of time. This single model achieves CV 0.34 LB 0.31. The LB score posted on version 14 is better than 0.31 because it is the ensemble of notebook version 14 with previous notebook versions 13, 12, 11, 8, and 7. Each previous version adds diversity by training with different data and/or features.
# * **Version 15 LB 0.359** In version 15 we add ESM transformer mutation probability and mutation entropy features. We also add protein surface area feature from Rope on Mars described [here][6]. This boosts single model train with only Kaggle train data to CV 0.35 LB 0.36! The LB score for notebook 15 is better than LB 0.36 because notebook version 15 submits an ensemble of notebook versions single models from 15, 14, 13, 12, 11, 8 and 7.
# * **Version 16 LB 0.397** Notebook version 16 achieves LB 0.397 single model training will all Jin external data (and no Kaggle data). It uses all our new features including surface area features, ESM mutation probabilities, and ESM mutation entropy. Notebook version 16 shows a better LB score than 0.397 because it submits an ensemble of single models from versions 16, 15, 14, 12, 12, 11, 8, and 7.
# * **Version 17 LB 0.415** Inspired by Juan Smith Perera's notebook [here][7], notebook version 17 adds substitution matrix features from four Blosum matrices and one DeMaSk matrix. This boosts single model performance to LB 0.415 woohoo!
# * **Version 18** Stay tuned for more versions...
# 
# [1]: https://github.com/JinyuanSun/mutation-stability-data
# [2]: https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/356920
# [3]: https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/356182#1968210
# [4]: https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/356182
# [5]: https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/358320
# [6]: https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/357899#1986149
# [7]: https://www.kaggle.com/code/jsmithperera/chris-deotte-s-5000x200-blosum100-lb-0-406

# In[1]:


# DEFINE WHAT TO TRAIN WITH (and KFOLD VALIDATE) VERSUS HOLDOUT VALIDATE WITH
# ADD WORDS "kaggle.csv", "jin_tm.csv", "jin_train.csv", "jin_test.csv" to lists below
# IF YOU ADD MORE DATASETS, ADD THOSE WORDS TOO

KFOLD_SOURCES = ['jin_tm.csv','jin_train.csv','jin_test.csv']
HOLDOUT_SOURCES = ['kaggle.csv']

# IF WILD TYPE GROUP HAS FEWER THAN THIS MANY MUTATION ROWS REMOVE THEM
EXCLUDE_CT_UNDER = 25

# IF WE TRAIN WITH ALPHA FOLD'S PDBS WE MUST INFER WITH "PLDDT = TRUE"
# KAGGLE.CSV USES ALPHA FOLD PDB, SO SET BELOW TO TRUE WHEN TRAIN WITH KAGGLE.CSV
# JIN.CSV EXTERNAL DATA USES PROTEIN DATA BANK, SO SET BELOW TO FALSE WITH JIN DATA
USE_PLDDT_INFER = False

# IF WE WISH TO TRAIN WITH MIXTURE OF ALPHA FOLD AND PROTEIN DATA BANK PDB FILES
# THEN WE CAN EXCLUDE B_COLUMN AND THEN THERE IS NO PROBLEM
USE_B_COLUMN = False

VER = 17


# # Download 3 External Mutation CSV
# We download three external datasets from Jinyuan sun GitHub [here][1] with discussion [here][2]
# 
# [1]: https://github.com/JinyuanSun/mutation-stability-data
# [2]: https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/356182

# In[2]:


import os, numpy as np, gc, seaborn as sns
from scipy.stats import spearmanr, pearsonr, rankdata
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', 500)


# In[3]:


os.system('wget https://raw.githubusercontent.com/JinyuanSun/mutation-stability-data/main/train.csv')
os.system('wget https://raw.githubusercontent.com/JinyuanSun/mutation-stability-data/main/test.csv')
os.system('wget https://raw.githubusercontent.com/JinyuanSun/mutation-stability-data/main/tm.csv')
os.system('mkdir downloaded_csv; mv *csv downloaded_csv')


# In[4]:


df = pd.read_csv('downloaded_csv/train.csv')
df = df.iloc[:,1:]
print('Downloaded train shape', df.shape )
df.head()


# In[5]:


df2 = pd.read_csv('downloaded_csv/test.csv')
df2 = df2.iloc[:,1:]
print('Downloaded test shape', df2.shape )
df2.head()


# In[6]:


df3 = pd.read_csv('downloaded_csv/tm.csv')
df3 = df3.iloc[:,1:]
print('Downloaded tm shape', df3.shape )
df3.head()


# # Transform Kaggle Train Data into Mutation CSV
# Kaggle's `train.csv` cannot be used as is. Each row has one protein. Instead we need to convert the dataframe where each row has one wild type protein and one single point mutation protein and target `dTm` (i.e. delta `Tm`). Then we can more readily train with the data to predict Kaggle's `test.csv`. This is explained in discussion [here][1].
# 
# Robert Hatch has processed the train data and found pairs of wild type and mutation in his notebook [here][2] with discussion [here][3]. We will load his output. GreySnow has made 73 of 78 train's wild type protein PDB's available in his dataset [here][7] with discussion [here][4] using code from Vladimir [here][5] and [here][6]. We will load his output.
# 
# Note that Robert finds 4195 pairs (in Kaggle train data) which include 78 wild types. We will filter this below. From each group involving one wild type, we will only keep the rows from the most common `data_source` and `pH` pair. Furthermore, we remove rows belonging to a single `data_source` `pH` pair where all `Tm` targets are the same. And we remove 4 out of 78 groups where GreySnow and Vladimir did not find the PDB file. We saved our processed mutation CSV as `kaggle_train.csv` so you can use it in other notebooks.
# 
# [1]: https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/358320
# [2]: https://www.kaggle.com/code/roberthatch/novo-train-data-contains-wildtype-groups/notebook
# [3]: https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/358156
# [4]: https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/359284
# [5]: https://www.kaggle.com/code/vslaykovsky/nesp-alphafold-v2-exact-match-data
# [6]: https://www.kaggle.com/code/vslaykovsky/nesp-alphafold2-all-close-matches
# [7]: https://www.kaggle.com/datasets/shlomoron/train-wildtypes-af

# In[7]:


# https://www.kaggle.com/code/roberthatch/novo-train-data-contains-wildtype-groups/notebook
kaggle = pd.read_csv('../input/novo-train-data-contains-wildtype-groups/train_wildtype_groups.csv')
print('Before processing Robert dataframe shape:', kaggle.shape )
kaggle.head()


# In[8]:


kaggle['id'] = kaggle.data_source.astype('str') + '_' + kaggle.pH.astype('str') + '_' + kaggle.group.astype('str')
kaggle['ct'] = kaggle.groupby('id').id.transform('count')
kaggle['n'] = kaggle.groupby('id').tm.transform('nunique')
kaggle = kaggle.loc[kaggle.n>1]
kaggle = kaggle.sort_values(['group','ct'],ascending=[True,False])
KEEP = kaggle.groupby('group').id.agg('first').values
kaggle = kaggle.loc[kaggle.id.isin(KEEP)]

def find_mut(row):
    mut = row.protein_sequence
    seq = row.wildtype
    same = True
    for i,(x,y) in enumerate(zip(seq,mut)):
        if x!=y: 
            same = False
            break
    if not same:
        row['WT'] = seq[i]
        row['position'] = i+1
        row['MUT'] = mut[i]
    else:
        row['WT'] = 'X'
        row['position'] = -1   
        row['MUT'] = 'X'     
    return row

grp = [f'GP{g:02d}' for g in kaggle.group.values]
kaggle['PDB'] = grp
kaggle = kaggle.apply(find_mut,axis=1)
kaggle = kaggle.loc[kaggle.position!=-1]
kaggle['base'] = kaggle.groupby('group').tm.transform('mean')
kaggle['dTm'] = kaggle.tm - kaggle.base
kaggle = kaggle.rename({'wildtype':'sequence','protein_sequence':'mutant_seq'},axis=1)
COLS = ['PDB','WT','position','MUT','dTm','sequence','mutant_seq']
kaggle = kaggle[COLS]

# https://www.kaggle.com/datasets/shlomoron/train-wildtypes-af
alphafold = pd.read_csv('../input/train-wildtypes-af/alpha_fold_df.csv')
dd = {}
for s in kaggle.sequence.unique():
    tmp = alphafold.loc[alphafold.af2_sequence==s,'af2id']
    if len(tmp)>0: c = tmp.values[0].split(':')[1]
    else: c = np.nan
    dd[s] = c
    
kaggle['CIF'] = kaggle.sequence.map(dd)
kaggle = kaggle.loc[kaggle.CIF.notnull()].reset_index(drop=True)
kaggle.to_csv('kaggle_train.csv',index=False)
print('After processing Robert dataframe shape:', kaggle.shape )
kaggle.head()


# # Combine 4 CSV Files

# In[9]:


df['source'] = 'jin_train.csv'
df['dTm'] = np.nan
df['CIF'] = None
df2['source'] = 'jin_test.csv'
df2['dTm'] = np.nan
df2['CIF'] = None

df3 = df3.loc[~df3.PDB.isin(['1RX4', '2LZM', '3MBP'])].copy()
df3['source'] = 'jin_tm.csv'
df3['ddG'] = np.nan
df3['CIF'] = None
df3 = df3.rename({'WT':'wildtype','MUT':'mutation'},axis=1)
df3 = df3[df.columns]

kaggle['source'] = 'kaggle.csv'
kaggle['ddG'] = np.nan
kaggle = kaggle.rename({'WT':'wildtype','MUT':'mutation'},axis=1)
kaggle = kaggle[df.columns]

df = pd.concat([df,df2,df3,kaggle],axis=0,ignore_index=True)
del df2, df3, kaggle
print('Combined data shape',df.shape)
df.to_csv(f'all_train_data_v{VER}.csv',index=False)
df = df.loc[df.source.isin(KFOLD_SOURCES+HOLDOUT_SOURCES)]
print('Kfold plus Holdout shape',df.shape)
df = df.sort_values(['PDB','position']).reset_index(drop=True)
df.head()


# # Download 200 PDB Files

# In[10]:


print('There are',df.PDB.nunique(),'PDB files to download')


# In[11]:


# THE FOLLOWING PROTEINS SEQUENCES CANNOT BE ALIGNED BETWEEN CSV AND PDB FILE (not sure why)
bad = [f for f in df.PDB.unique() if len(f)>4]
bad += ['1LVE', '2IMM', '2RPN', '1BKS', '1BLC', '1D5G', '1KDX', '1OTR', '3BN0', '3D3B', '3HHR', '3O39']
bad += ['3BDC','1AMQ','1X0J','1TPK','1GLM','1RHG','3DVI','1RN1','1QGV'] 
bad += ['1SVX','4E5K'] 
print(f'We will ignore mutations from {len(bad)} PDB files')


# In[12]:


os.system('mkdir downloaded_pdb')
for p in [f for f in df.PDB.unique() if f not in bad]:
    if p[:2]=='GP': continue # skip kaggle CIF
    os.system(f'cd downloaded_pdb; wget https://files.rcsb.org/download/{p}.pdb') 


# # Feature Engineer
# Below is 3D plot of Kaggle's test data protein structure from PDB file rendered by py3Dmol (with docs [here][1]). For each of our 5000 mutation downloaded train data, we also have protein structure since we downloaded 200 PDB files from Protein Data Bank. And when using Kaggle train data, we now have PDB files! Therefore when we feature engineer, we can create features from both protein sequence, protein structure, and amino acids. In this notebook, we use the following features:
# * Protein Structure features (from atom positions in PDB file)
# * Protein Sequence features
# * Amino Acid features 
# * Substitution matrix features
# * Transformer ESM embeddings
# 
# thanks @kaggleqrdl for his ESM starter notebook [here][2]
# 
# [1]: https://3dmol.csb.pitt.edu/doc/$3Dmol.GLViewer.html
# [2]: https://www.kaggle.com/code/kaggleqrdl/esm-quick-start-lb237

# ## Protein Structure Features
# We add handcrafted structure features such as amino acid distance to center of protein and the amino acid angle with its two neighbors. In notebook versions 15+ we add amino acid surface area features provided and explained by Rope on Mars [here][1].
# 
# [1]: https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/357899

# In[13]:


get_ipython().system('pip install py3Dmol -q')
import py3Dmol 
with open("../input/novozymes-enzyme-stability-prediction/wildtype_structure_prediction_af2.pdb") as ifile:
    protein = "".join([x for x in ifile])
#view = py3Dmol.view(query='pdb:1DIV', width=800, height=600)
view = py3Dmol.view(width=800, height=600) 
view.addModelsAsFrames(protein)
style = {'cartoon': {'color': 'spectrum'},'stick':{}}
view.setStyle({'model': -1},style) 
view.zoom(0.12)
view.rotate(235, {'x':0,'y':1,'z':1})
view.spin({'x':-0.2,'y':0.5,'z':1},1)
view.show()


# ## Amino Acid Features
# For amino acid features, we use the CSV provided by Moth @alejopaullier [here][1]. Lucas Morin shows their usefulness [here][2]
# 
# [1]: https://www.kaggle.com/datasets/alejopaullier/aminoacids-physical-and-chemical-properties
# [2]: https://www.kaggle.com/code/lucasmorin/nesp-changes-eda-and-baseline

# In[14]:


# BIOPANDAS PDB READER
get_ipython().system('pip install biopandas -q')
from biopandas.pdb import PandasPdb
from biopandas.mmcif import PandasMmcif
aa_map = {'VAL': 'V', 'PRO': 'P', 'ASN': 'N', 'GLU': 'E', 'ASP': 'D', 'ALA': 'A', 'THR': 'T', 'SER': 'S',
          'LEU': 'L', 'LYS': 'K', 'GLY': 'G', 'GLN': 'Q', 'ILE': 'I', 'PHE': 'F', 'CYS': 'C', 'TRP': 'W',
          'ARG': 'R', 'TYR': 'Y', 'HIS': 'H', 'MET': 'M'}
aa_map_2 = {x:y for x,y in zip(np.sort(list(aa_map.values())),np.arange(20))}
aa_map_2['X'] = 20

# https://www.kaggle.com/datasets/alejopaullier/aminoacids-physical-and-chemical-properties
aa_props = pd.read_csv('../input/aminoacids-physical-and-chemical-properties/aminoacids.csv').set_index('Letter')
PROPS = ['Molecular Weight', 'Residue Weight', 'pKa1', 'pKb2', 'pKx3', 'pl4', 
         'H', 'VSC', 'P1', 'P2', 'SASA', 'NCISC']
print('Amino Acid properties dataframe. Shape:', aa_props.shape )
aa_props.head(22)


# ## Substitution Matrix Features
# We will add features from Blosum, and DeMaSk substitution matrices. The DeMaSk matrix was downloaded from [here][1]
# 
# [1]: https://demask.princeton.edu/about/

# In[15]:


# BLOSUM SUBSTITUTION MATRICES
from Bio.SubsMat import MatrixInfo
def get_sub_matrix(matrix_name="blosum100"):
    sub_matrix = getattr(MatrixInfo, matrix_name)
    sub_matrix.update({(k[1], k[0]):v for k,v in sub_matrix.items() if (k[1], k[0]) not in list(sub_matrix.keys())})
    return sub_matrix
sub_mat_b100 = get_sub_matrix("blosum100")
sub_mat_b80 = get_sub_matrix("blosum80")
sub_mat_b60 = get_sub_matrix("blosum60")
sub_mat_b40 = get_sub_matrix("blosum40")

# DEMASK SUBSTITUTION MATRICES
dff = pd.read_csv('../input/nesp-test-wildtype-pdb/matrix.txt', sep='\t')
letters = list( dff.columns )
l_dict = {x:y for x,y in zip(letters,range(20))}
sub_mat_demask = {}
for x in letters:
    for y in letters:
        sub_mat_demask[(x,y)] = dff.iloc[l_dict[x],l_dict[y]]

# PLOT MATRICES
AA = np.sort(list(aa_map.values()))
blosum100 = np.zeros((20,20))
demask = np.zeros((20,20))
for (k1,k2),v in sub_mat_b100.items():
    if (k1!='Z')&(k2!='Z')&(k1!='B')&(k2!='B')&(k1!='X')&(k2!='X'):
        blosum100[ aa_map_2[k1], aa_map_2[k2] ] = v
for (k1,k2),v in sub_mat_demask.items():
    if (k1!='Z')&(k2!='Z')&(k1!='B')&(k2!='B')&(k1!='X')&(k2!='X'):
        demask[ aa_map_2[k1], aa_map_2[k2] ] = v
        
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.heatmap(blosum100, cmap='RdBu_r', annot=True, center=0.0)
plt.xticks(np.arange(20)+0.5,AA)
plt.yticks(np.arange(20)+0.5,AA)
plt.title('Blosum 100 Substitution Matrix',size=16)
plt.subplot(1,2,2)
sns.heatmap(demask, cmap='RdBu_r', annot=True, fmt='.1g') #, center=0.0)
plt.xticks(np.arange(20)+0.5,AA)
plt.yticks(np.arange(20)+0.5,AA)
plt.title('DeMaSk Substitution Matrix',size=16)
plt.show()


# ## Transformer ESM Features
# In order to convert amino acid sequences aka proteins into meaningful features, we will use embeddings from SOTA protein transformer. We use Facebook's pretrained protein transformer ESM (Evolutionary Scale Modeling) with research paper [here][1] and GitHub [here][2]. Kaggleqrdl provided a starter notebook [here][3]. In version 15+, we also extract mutation probabilties and mutation entropy from ESM!
# 
# [1]: https://www.biorxiv.org/content/10.1101/622803v4
# [2]: https://github.com/facebookresearch/esm
# [3]: https://www.kaggle.com/code/kaggleqrdl/esm-quick-start-lb237

# In[16]:


# https://github.com/facebookresearch/esm
get_ipython().system('pip install fair-esm -q')

# https://www.kaggle.com/code/kaggleqrdl/esm-quick-start-lb237
import torch, esm
token_map = {'L': 0, 'A': 1, 'G': 2, 'V': 3, 'S': 4, 'E': 5, 'R': 6, 'T': 7, 'I': 8, 'D': 9, 'P': 10, 
         'K': 11, 'Q': 12, 'N': 13, 'F': 14, 'Y': 15, 'M': 16, 'H': 17, 'W': 18, 'C': 19}
t_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
t_model.eval()  # disables dropout for deterministic results
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t_model.to(device)


# ### Embeddings
# We input each train and test wildtype into our transformer and extract the last hidden layers activations. For each protein, this has shape `(1, len_protein_seq, 1280)`. We will save the full embeddings and the pooled embeddings for use later. Additionally we will save the MLM pretrain task amino acid prediction which indicates mutation probability and mutation entropy. This has shape `(1, len_protein_seq, 33)` but we extract to `(len_protein_seq, 20)` where 20 is number of common amino acids.

# In[17]:


# TRAIN AND TEST WILDTYPES
PCA_CT = 16 # random sample size per protein to fit PCA with
all_pdb = [f for f in df.PDB.unique() if f not in bad]
base = 'VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK'
all_pdb_embed_pool = np.zeros((len(all_pdb)+1,1280))
all_pdb_embed_local = []
all_pdb_embed_tmp = []

from scipy.special import softmax 
from scipy.stats import entropy
all_pdb_prob = []

# EXTRACT TRANSFORMER EMBEDDINGS FOR TRAIN AND TEST WILDTYPES
print('Extracting embeddings from proteins...')
for i,p in enumerate(all_pdb+['TEST']):
    
    # WILDTYPE SEQUENCE
    print(p,', ',end='')
    if p=='TEST': seq = base
    else: seq = df.loc[df.PDB==p,'sequence'].iloc[0]
        
    # EXTRACT EMBEDDINGS, MUTATION PROBABILITIES, ENTROPY
    data = [("protein1", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = t_model(batch_tokens, repr_layers=[33])
    logits = (results['logits'].detach().cpu().numpy()[0,].T)[4:24,1:-1]
    all_pdb_prob.append(softmax(logits,axis=0))
    results = results["representations"][33].detach().cpu().numpy()
    
    # SAVE EMBEDDINGS
    all_pdb_embed_local.append(results)
    all_pdb_embed_pool[i,] = np.mean( results[0,:,:],axis=0 )
    
    # TEMPORARILY SAVE LOCAL MUTATION EMBEDDINGS
    tmp = df.loc[df.PDB==p,'position'].unique()
    if p=='TEST': tmp = np.random.choice(range(20,200),PCA_CT,replace=False)
    if len(tmp)>PCA_CT: tmp = np.random.choice(tmp,PCA_CT,replace=False)
    for j in tmp: all_pdb_embed_tmp.append( results[0,j,:] )
        
    del batch_tokens, results
    gc.collect(); torch.cuda.empty_cache()

all_pdb_embed_tmp = np.stack(all_pdb_embed_tmp)


# ### RAPIDS PCA
# The transformer embeddings have dimension 1280. Since we only have a few thousand rows of train data, that is too many features to include all of them in our XGB model. Furthermore, we want to use local, pooling, and delta embeddings. Which would be 3x1280. To prevent our model from overfitting as a result of the "curse of dimensionality", we reduce the dimension of embeddings using RAPIDS PCA. 

# In[18]:


# REDUCE EMBEDDING DIM FROM 1280 TO 32 OR 16 WITH PCA
from cuml import PCA
pca_pool = PCA(n_components=32)
pca_embeds = pca_pool.fit_transform(all_pdb_embed_pool.astype('float32'))
pca_local = PCA(n_components=16)
pca_local.fit(all_pdb_embed_tmp.astype('float32'))
pdb_map = {x:y for x,y in zip(all_pdb,range(len(all_pdb)))}
pdb_map['kaggle'] = len(all_pdb)
del all_pdb_embed_tmp
_ = gc.collect()


# ## Feature Engineering Function

# In[19]:


# FEATURE ENGINEER FUNCTION
def get_new_row(atom_df, j, row):
    ##################
    # ATOM_DF - IS PDB FILE'S ATOM_DF
    # J - IS RESIDUE NUMBER WHICH IS TRAIN CSV POSITION PLUS OFFSET
    # ROW - IS ROW FROM DOWNLOADED TRAIN CSV
    ##################
        
    dd = None
    tmp = atom_df.loc[(atom_df.residue_number==j)].reset_index(drop=True)
    prev = atom_df.loc[(atom_df.residue_number==j-1)].reset_index(drop=True)
    post = atom_df.loc[(atom_df.residue_number==j+1)].reset_index(drop=True)
    
    # FEATURE ENGINEER
    if len(tmp)>0:
        
        # GET MUTANT EMBEDDINGS
        data = [("protein1", row.mutant_seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = t_model(batch_tokens, repr_layers=[33]) 
        results = results["representations"][33].cpu().numpy()
        mutant_local = pca_local.transform(results[:1,row.position,:])[0,]
        mutant_pool = np.mean( results[:1,:,:],axis=1 )
        mutant_pool = pca_pool.transform(mutant_pool)[0,]
    
        # MUTATION AND POSITION
        dd = {}
        dd['WT'] = row.wildtype
        dd['WT2'] = tmp.residue_name.map(aa_map)[0]
        dd['MUT'] = row.mutation
        dd['position'] = row.position
        dd['relative_position'] = row.position / len(row.sequence)

        # B_FACTOR
        if USE_B_COLUMN: dd['b_factor'] = tmp.b_factor.mean()
        
        # ANIMO ACID PROPERTIES AND DELTAS
        for c in PROPS:
            dd[f'{c}_1'] = aa_props.loc[row.wildtype,c]
            dd[f'{c}_2'] = aa_props.loc[row.mutation,c]
            dd[f'{c}_delta'] = dd[f'{c}_2']-dd[f'{c}_1']
            
        # SUBSTITUTION MATRICES
        dd['blosum100'] = sub_mat_b100[(row.wildtype,row.mutation)]
        dd['blosum80'] = sub_mat_b80[(row.wildtype,row.mutation)]
        dd['blosum60'] = sub_mat_b60[(row.wildtype,row.mutation)]
        dd['blosum40'] = sub_mat_b40[(row.wildtype,row.mutation)]
        dd['demask'] = sub_mat_demask[(row.wildtype,row.mutation)]

        # PREVIOUS AND POST AMINO ACID INFO
        if (len(prev)>0):
            dd['prev'] = prev.residue_name.map(aa_map)[0]
            if USE_B_COLUMN: dd['b_factor_prev'] = prev.b_factor.mean()
        else:
            dd['prev'] = 'X'
            if USE_B_COLUMN: dd['b_factor_prev'] = -999            
            
        if (len(post)>0):
            dd['post'] = post.residue_name.map(aa_map)[0]
            if USE_B_COLUMN: dd['b_factor_post'] = post.b_factor.mean() 
        else:
            dd['post'] = 'X'
            if USE_B_COLUMN: dd['b_factor_post'] = -999 
            
        # ANGLE BETWEEN MUTATION AND NEIGHBORS
        if (len(prev)>0)&(len(post)>0):
            # BACKBONE ATOMS
            atm = ['N','H','CA','O']
            prev = prev.loc[prev.atom_name.isin(atm)]
            tmp = tmp.loc[tmp.atom_name.isin(atm)]
            post = post.loc[post.atom_name.isin(atm)]
            # VECTORS
            c_prev = np.array( [prev.x_coord.mean(),prev.y_coord.mean(),prev.z_coord.mean()] )
            c_tmp = np.array( [tmp.x_coord.mean(),tmp.y_coord.mean(),tmp.z_coord.mean()] )
            c_post = np.array( [post.x_coord.mean(),post.y_coord.mean(),post.z_coord.mean()] )
            vec_a = c_prev - c_tmp
            vec_b = c_post - c_tmp
            # COMPUTE ANGLE
            norm_a = np.sqrt(vec_a.dot(vec_a))
            norm_b = np.sqrt(vec_b.dot(vec_b))
            dd['cos_angle'] = vec_a.dot(vec_b)/norm_a/norm_b
        else:
            dd['cos_angle'] = -2
            
        # 3D LOCATION OF MUTATION
        atm = ['N','H','CA','O']
        atoms = atom_df.loc[atom_df.atom_name.isin(atm)]
        centroid1 = np.array( [atoms.x_coord.mean(),atoms.y_coord.mean(),atoms.z_coord.mean()] )
        tmp = tmp.loc[tmp.atom_name.isin(atm)]
        centroid2 = np.array( [tmp.x_coord.mean(),tmp.y_coord.mean(),tmp.z_coord.mean()] )
        dist = centroid2 - centroid1
        dd['location3d'] = dist.dot(dist)
        
        # TRANSFORMER ESM EMBEDDINGS
        wt_local = pca_local.transform(all_pdb_embed_local[pdb_map[row.PDB]][:1,row.position,:])[0,]
        wt_pool = pca_embeds[pdb_map[row.PDB],]
        for kk in range(32):
            dd[f'pca_pool_{kk}'] = mutant_pool[kk] - wt_pool[kk]
            if kk>=16: continue
            dd[f'pca_wt_{kk}'] = wt_local[kk]
            dd[f'pca_mutant_{kk}'] = mutant_local[kk]
            dd[f'pca_local_{kk}'] = mutant_local[kk] - wt_local[kk]
            
        # TRANSFORMER MUTATION PROBS AND ENTROPY
        dd['mut_prob'] = all_pdb_prob[pdb_map[row.PDB]][token_map[dd['MUT']],dd['position']-1]
        dd['mut_entropy'] = entropy( all_pdb_prob[pdb_map[row.PDB]][:,dd['position']-1] )
        
        # SURFACE AREA FEATURES
        PATH = '../input/nesp-kaggle-train-surface-area/'
        if row.CIF: 
            nm = f'{row.CIF}-model_v3.csv'
        elif row.PDB!='kaggle': 
            PATH = '../input/nesp-jin-external-surface-area/'
            nm = f'{row.PDB}.csv'
        else: 
            nm = 'wildtype_structure_prediction_af2_SASA.csv'
        try:    
            area = pd.read_csv(f'{PATH}{nm}')
            rw = area.loc[area.Residue_number==j].iloc[0]
            dd['sa_total'] = rw.Total
            dd['sa_apolar'] = rw.Apolar
            dd['sa_backbone'] = rw.Backbone
            dd['sa_sidechain'] = rw.Sidechain
            dd['sa_ratio'] = rw.Ratio
            dd['sa_in/out'] = -1
            if rw['In/Out']=='i': dd['sa_in/out'] = 1
            elif rw['In/Out']=='o': dd['sa_in/out'] = 0
        except:
            print('### NEED SURFACE AREA for PDB:',row.PDB,'residue_number:',j)
            return None
        
        # LABEL ENCODE AMINO ACIDS
        dd['AA1'] = aa_map_2[dd['WT']]
        dd['AA2'] = aa_map_2[dd['MUT']]
        dd['AA3'] = aa_map_2[dd['prev']]
        dd['AA4'] = aa_map_2[dd['post']]
        
        # TARGETS AND SOURCES
        dd['ddG'] = row.ddG
        dd['dTm'] = row.dTm
        dd['pdb'] = row.PDB
        dd['source'] = row.source
        
        del batch_tokens, results, mutant_local, mutant_pool, wt_local, wt_pool
        gc.collect(); torch.cuda.empty_cache()

    return dd


# # Transform Train Data

# In[20]:


pdb = None
rows = []
offsets = []

for index,row in df.iterrows():
    if row.PDB in bad: continue
        
    # READ PDB FILE WHICH CONTAINS MORE INFO ABOUT PROTEIN
    first = False
    if row.PDB != pdb:
        pdb = row.PDB
        if row.CIF:
            atom_df = PandasMmcif().read_mmcif(f'../input/train-wildtypes-af/cif/{row.CIF}-model_v3.cif')
            atom_df = atom_df.df['ATOM']
            atom_df = atom_df.rename({'label_seq_id':'residue_number','label_comp_id':'residue_name'},axis=1)
            atom_df = atom_df.rename({'Cartn_x':'x_coord','Cartn_y':'y_coord','Cartn_z':'z_coord'},axis=1)
            atom_df = atom_df.rename({'B_iso_or_equiv':'b_factor','label_atom_id':'atom_name'},axis=1)
        else:
            atom_df = PandasPdb().read_pdb(f'downloaded_pdb/{row.PDB}.pdb')
            atom_df = atom_df.df['ATOM']
        first = True

    # VERY IMPORTANT - ALIGN SEQUENCES
    # THE RESIDUE NUMBERS IN PDB FILES DONT MATCH THE POSTION NUMBERS IN CSV FILE!
    tmp = atom_df.drop_duplicates(['residue_name','residue_number']).sort_values('residue_number')
    tmp = tmp.iloc[20:36].reset_index(drop=True)
    d = (tmp.residue_number.diff()!=1.0).sum()
    if d>1: print(f'=> ERROR missing consecutive amino acids in PDB file {row.PDB}')
    tmp['letter'] = tmp.residue_name.map(aa_map)  
    pdb_seq = (''.join( tmp.letter.values ))
    csv_seq = df.loc[df.PDB==row.PDB,'sequence'].values[0]
    i = csv_seq.find(pdb_seq)
    if i==-1: print('=> ERROR cannot find PDB sequence in CSV sequence for {row.PDB}')
    x = tmp.loc[0,'residue_number']
    offset = (x-i)-1
    if first: 
        print(f'{row.PDB} PDB residue_number equals {offset} added to position in CSV')
        dd = {}
        dd['pdb'] = row.PDB
        dd['offset'] = offset
        offsets.append(dd)
    
    # FEATURE ENGINEER
    j = row.position + offset
    dd = get_new_row(atom_df, j, row)
    if dd is not None:
        rows.append(dd)


# In[21]:


# ADD THESE OFFSETS TO CSV'S POSITION TO GET PDB'S RESIDUE NUMBER
offsets = pd.DataFrame(offsets)
offsets = offsets.loc[offsets.pdb.str[:2]!='GP'] # drop kaggle CIF
if len(offsets)>0:
    offsets.to_csv('downloaded_csv/PDB_offset_from_CSV.csv',index=False)
    print('Add these offsets to CSV position to get PDB residue number')
    display( offsets.head() )


# # Create Train and Holdout
# We create train and holdout datasets based on variables in code cell #1. We will rank normalize the targets so that the model so the model can train with data from different sources and all targets will look similar to the model.

# In[22]:


# CREATE EXTERNAL TRAIN DATAFRAME
train = pd.DataFrame(rows)
train = train.loc[train.WT == train.WT2].reset_index(drop=True)
print('Train plus Holdout data shape', train.shape )
train['ct'] = train.groupby('pdb').WT.transform('count')
train = train.loc[train.ct>EXCLUDE_CT_UNDER].reset_index(drop=True)
train = train.drop(['WT2','ct'],axis=1)
print('Data shape after removing small mutation groups', train.shape )


# In[23]:


# RANK NORMALIZE ddG AND dTm TARGETS SO ALL CAN BE MIXED TOGETHER
train['target'] = 0.5
for g in train.pdb.unique():
    target = 'dTm'
    tmp = train.loc[train.pdb==g,'dTm']
    if tmp.isna().sum()>len(tmp)/2: target = 'ddG'
    train.loc[train.pdb==g,'target'] =\
        rankdata(train.loc[train.pdb==g,target])/len(train.loc[train.pdb==g,target])
train.head()


# In[24]:


# USE some sources TO TRAIN/VALIDATE AND other sources TO HOLDOUT VALIDATE
holdout = train.loc[train.source.isin(HOLDOUT_SOURCES)].reset_index(drop=True)
train = train.loc[train.source.isin(KFOLD_SOURCES)].reset_index(drop=True)

# LABEL ENCODE GROUPS FOR GROUP K FOLD
train['group'],_ = train.pdb.factorize()
holdout['group'],_ = holdout.pdb.factorize()


# In[25]:


EXCLUDE = ['WT','MUT','prev','post','ddG','dTm','pdb','source','target','group','oof']
FEATURES = [c for c in train.columns if c not in EXCLUDE]
print(f'We have {len(FEATURES)} features for our model:')
print( FEATURES )


# # XGBoost Model

# In[26]:


# LOAD XGB LIBRARY
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
print('XGB Version',xgb.__version__)

FOLDS = 11
SEED = 123

# XGB MODEL PARAMETERS
xgb_parms = { 
    'max_depth':4, 
    'learning_rate':0.001, 
    'subsample':0.6,
    'colsample_bytree':0.2, 
    'eval_metric':'rmse',
    'objective':'reg:squarederror',
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'random_state':SEED
}


# In[27]:


get_ipython().run_cell_magic('time', '', "importances = []\nimportances2 = []\noof = np.zeros(len(train))\npreds = np.zeros(len(holdout))\nos.system('mkdir xgb_models')\n\nskf = GroupKFold(n_splits=FOLDS)\nfor fold,(train_idx, valid_idx) in enumerate(skf.split(\n            train, train.target, train.group )):\n        \n    print('#'*25)\n    print('### Fold',fold+1)\n    print('### Train size',len(train_idx),'Valid size',len(valid_idx))\n    print('#'*25)\n    \n    # TRAIN, VALID, HOLDOUT FOR FOLD K\n    X_train = train.loc[train_idx, FEATURES]\n    y_train = train.loc[train_idx,'target']\n    X_valid = train.loc[valid_idx, FEATURES]\n    y_valid = train.loc[valid_idx, 'target']\n    X_holdout = holdout[FEATURES]\n    \n    dtrain = xgb.DMatrix(data=X_train, label=y_train)\n    dvalid = xgb.DMatrix(data=X_valid, label=y_valid)\n    dholdout = xgb.DMatrix(data=X_holdout)\n    \n    # TRAIN MODEL FOLD K\n    model = xgb.train(xgb_parms, \n                dtrain=dtrain,\n                evals=[(dtrain,'train'),(dvalid,'valid')],\n                num_boost_round=9999,\n                early_stopping_rounds=100,\n                verbose_eval=100) \n    model.save_model(f'xgb_models/XGB_fold{fold}.xgb')\n    \n    # GET FEATURE IMPORTANCE FOR FOLD K\n    dd = model.get_score(importance_type='weight')\n    df = pd.DataFrame({'feature':dd.keys(),f'importance_{fold}':dd.values()})\n    importances.append(df)\n    dd = model.get_score(importance_type='gain')\n    df = pd.DataFrame({'feature':dd.keys(),f'importance_{fold}':dd.values()})\n    importances2.append(df)\n            \n    # INFER OOF FOLD K\n    oof_preds = model.predict(dvalid)\n    rsme = mean_squared_error(y_valid.values, oof_preds, squared=False)\n    print('RSME =',rsme,'\\n')\n    oof[valid_idx] = oof_preds\n    \n    # HOLDOUT PREDS\n    if len(holdout)>0:\n        p = model.predict(dholdout)\n        preds += p/FOLDS\n    \n    del dtrain, X_train, y_train, dd, df\n    del X_valid, y_valid, dvalid, model\n    _ = gc.collect()\n    \nprint('#'*25)\nrsme = mean_squared_error(train.target.values, oof, squared=False)\nprint('OVERALL RSME =',rsme,'\\n')\n\ntrain['oof'] = oof\nif len(holdout)>0: holdout['preds'] = preds\n")


# # Feature Importance
# Below we display both XGB feature importance by `weight` and by `gain`.

# In[28]:


df = importances[0].copy()
for k in range(1,FOLDS): df = df.merge(importances[k], on='feature', how='left')
df['importance'] = df.iloc[:,1:].mean(axis=1)
df = df.sort_values('importance',ascending=False)
NUM_FEATURES = 50 #len(df)
plt.figure(figsize=(10,5*NUM_FEATURES//10))
plt.barh(np.arange(NUM_FEATURES,0,-1), df.importance.values[:NUM_FEATURES])
plt.yticks(np.arange(NUM_FEATURES,0,-1), df.feature.values[:NUM_FEATURES])
plt.title(f'XGB WEIGHT Feature Importance - Top {NUM_FEATURES}')
plt.show()


# In[29]:


df = importances2[0].copy()
for k in range(1,FOLDS): df = df.merge(importances2[k], on='feature', how='left')
df['importance'] = df.iloc[:,1:].mean(axis=1)
df = df.sort_values('importance',ascending=False)
NUM_FEATURES = 50 #len(df)
plt.figure(figsize=(10,5*NUM_FEATURES//10))
plt.barh(np.arange(NUM_FEATURES,0,-1), df.importance.values[:NUM_FEATURES])
plt.yticks(np.arange(NUM_FEATURES,0,-1), df.feature.values[:NUM_FEATURES])
plt.title(f'XGB GAIN Feature Importance - Top {NUM_FEATURES}')
plt.show()


# # Validate OOF (on either dTm or ddG)

# In[30]:


sp = []; sp_dtm = []; sp_ddg = []
for p in train.pdb.unique():

    tmp = train.loc[train.pdb==p].reset_index(drop=True)
    ttarget = 'dTm'
    if tmp['dTm'].isna().sum()>len(tmp)/2: ttarget = 'ddG'
    print('Protein',p,'has mutation count =',len(tmp),'and target =',ttarget)
    r = np.abs( spearmanr(tmp.oof.values, tmp[ttarget].values).correlation )
    print('Spearman Metric =',r)
    sp.append(r)
    if ttarget=='dTm': sp_dtm.append(r)
    else: sp_ddg.append(r)
    print()

print('#'*25)
if len(sp_dtm)>0:
    print(f'Overall Spearman Metric (predicting dTm) =',np.nanmean(sp_dtm))
if len(sp_ddg)>0:
    print(f'Overall Spearman Metric (predicting ddG) =',np.nanmean(sp_ddg))


# # Validate Holdout (on either dTm or ddG)

# In[31]:


sp = []; sp_dtm = []; sp_ddg = []
for p in holdout.pdb.unique():

    tmp = holdout.loc[holdout.pdb==p].reset_index(drop=True)
    ttarget = 'dTm'
    if tmp['dTm'].isna().sum()>len(tmp)/2: ttarget = 'ddG'
    print('Protein',p,'has mutation count =',len(tmp),'and target =',ttarget)
    r = np.abs( spearmanr(tmp.preds.values, tmp[ttarget].values).correlation )
    print('Spearman Metric =',r)
    sp.append(r)
    if ttarget=='dTm': sp_dtm.append(r)
    else: sp_ddg.append(r)
    print()

print('#'*25)
if len(sp_dtm)>0:
    print(f'Overall Spearman Metric (predicting dTm) =',np.nanmean(sp_dtm))
if len(sp_ddg)>0:
    print(f'Overall Spearman Metric (predicting ddG) =',np.nanmean(sp_ddg))


# # Transform Test Data
# The PDB file provided by Kaggle does not contain real `b_factor`. The column labeled `b_factor` is actually `pLDDT` predicted by Alpha Fold. We will load a `PDF` file below of Kaggle's test protein with estimated `b_factor` downloaded from internet and provided by @kaggleqrdl and @ropeonmars described [here][1]. If we train with Alpha Fold PDB then we will load Kaggle's test PDB created by Alpha Fold.
# 
# [1]: https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/356182#1968210

# In[32]:


# LOAD TEST WILDTYPE
base = 'VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK'
len(base)


# In[33]:


# LOAD TEST DATA
test = pd.read_csv('../input/novozymes-enzyme-stability-prediction/test.csv')
deletions = test.loc[test.protein_sequence.str.len()==220,'seq_id'].values
print('Test shape', test.shape )
test.head()


# In[34]:


# LOAD TEST DATA PDB FILE 
# NOTE KAGGLE'S PDB IS GENERATED BY ALPHA FOLD AND CONTAINS PLDDT IN B_FACTOR COLUMN
# WHEN TRAINING WITH KAGGLE.CSV WE NEED ALPHA FOLD PDB
# WHEN TRAINING WITH JIN DATA WE NEED PROTEIN DATA BANK PDB (with real b_factor)
if USE_PLDDT_INFER:
    atom_df = PandasPdb().read_pdb('../input/novozymes-enzyme-stability-prediction/wildtype_structure_prediction_af2.pdb')
else:
    atom_df = PandasPdb().read_pdb('../input/nesp-test-wildtype-pdb/model.pdb')
atom_df = atom_df.df['ATOM']
atom_df.head()


# In[35]:


def get_test_mutation(row):
    for i,(a,b) in enumerate(zip(row.protein_sequence,base)):
        if a!=b: break
    row['wildtype'] = base[i]
    row['mutation'] = row.protein_sequence[i]
    row['position'] = i+1
    return row

# TRANSFORM TEST DATAFRAME TO MATCH TRAIN DATAFRAME
test = test.apply(get_test_mutation,axis=1)
test['ddG'] = np.nan
test['dTm'] = np.nan
test['CIF'] = None
test['sequence'] = base
test = test.rename({'protein_sequence':'mutant_seq'},axis=1)
test['source'] = 'kaggle'
test['PDB'] = 'kaggle'

# FEATURE ENGINEER TEST DATA
rows = []
print(f'Extracting embeddings and feature engineering {len(test)} test rows...')
for index,row in test.iterrows():
    if index%10==0: print(index,', ',end='')
    j = row.position
    dd = get_new_row(atom_df, j, row)
    rows.append(dd)
test = pd.DataFrame(rows)
test.head()


# # Infer Test Data

# In[36]:


get_ipython().run_cell_magic('time', '', "# TEST DATA FOR XGB\nX_test = test[FEATURES]\ndtest = xgb.DMatrix(data=X_test)\n\n# INFER XGB MODELS ON TEST DATA\nmodel = xgb.Booster()\nmodel.load_model(f'xgb_models/XGB_fold0.xgb')\npreds = model.predict(dtest)\nfor f in range(1,FOLDS):\n    model.load_model(f'xgb_models/XGB_fold{f}.xgb')\n    preds += model.predict(dtest)\npreds /= FOLDS\n")


# In[37]:


plt.hist(preds,bins=100)
plt.title('Test preds histogram',size=16)
plt.show()


# # Create Submission CSV
# There are 2413 rows in test data. Among these, 2336 that are `edit mutations` and 77 are `delete mutations`. Our trained model can only predict `edit mutations`, so will set all `delete mutations` to the mean `edit mutation` prediction below.

# In[38]:


sub = pd.read_csv('../input/novozymes-enzyme-stability-prediction/sample_submission.csv')
sub.tm = preds
sub.loc[sub.seq_id.isin(deletions),'tm'] = sub.loc[~sub.seq_id.isin(deletions),'tm'].mean()
sub.to_csv(f'submission_ver{VER}.csv',index=False)
sub.head()

