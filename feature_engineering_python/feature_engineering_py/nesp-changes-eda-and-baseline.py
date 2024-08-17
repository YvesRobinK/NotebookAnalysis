#!/usr/bin/env python
# coding: utf-8

# # Data Exploration Notebook & Baseline
# 
# The goal of this competition is to predict impact of single mutations of an enzyme to its physical properties. We are provided with training data containing enzymes and a target. However all this enzyme appear to be pretty far of the base enzyme. Outside of the training data we have access to descriptive data of the base enzyme. Here is a small notebook exploring all the available data and data curated by different users. It generally provides an overview of the tools available.

# # Import libraries and Data
# 
# Importing general DS tools and some bioinformatic tools.

# In[1]:


get_ipython().system('pip install biopandas')


# In[2]:


get_ipython().system('pip install blosum')


# In[3]:


import numpy as np, pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px

import os, time
import urllib.request 

from biopandas.pdb import PandasPdb
import blosum as bl

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import Levenshtein
from Levenshtein import distance as levenshtein_distance


# # Exploring single mutations from train data
# 
# The train data present a very wide range of enzyme. It is significantly different of the test data that only ask for stability of single point mutations of a base enzyme.
# @cdeotte showed us that train data contains single point mutations (see: https://www.kaggle.com/code/cdeotte/train-data-contains-mutations-like-test-data). We explore associated data. There are some one significant changes: there appears to be significant discrepancies between data sources. We limit the data set to the pairs that were measured in same sources.

# In[4]:


train_df_group = pd.read_csv('../input/train-data-contains-mutations/train_with_groups.csv')

# remove group -1:
train_df = train_df_group[train_df_group.group!=-1]
count_by_group = train_df.groupby('group').seq_id.count()


# In[5]:


def build_change_list(group_df):
    
    list_output = []
    group_size = len(group_df)
    group_values = group_df.values
    
    col = ['pH','data_source','tm','group']

    for i in range(group_size):
        data1 = group_values[i]
        line1 = data1[1]
        values1  = data1[2:]
        for j in range(group_size):
            data2 = group_values[j]
            line2 = data2[1]
            values2  = data2[2:]
            if i!=j:
                edits = Levenshtein.editops(line1, line2)
                if len(edits)==1:
                    list_output.append(tuple([line1,line2])+edits[0]+tuple(line1[edits[0][1]])+tuple(line2[edits[0][1]])+ tuple(values1) + tuple(values2))
                else:
                    list_output.append(tuple([line1,line2])+('replace', 0, 0, 'A', 'A') + tuple(values1) + tuple(values2))

    changes = pd.DataFrame(list_output,columns=['seq1','seq2','operation','position1','position2','change1','change2']+[c+'1' for c in col] + [c+'2' for c in col])
    changes.change2 = np.where(changes.operation=='delete','',changes.change2)
    
    return changes


# In[6]:


get_ipython().run_cell_magic('time', '', "\ntop_n = 10 #408\nmain_groups = count_by_group.sort_values(ascending=False).index[:top_n]\n\ndf_list = []\n\nfor i in main_groups:\n    print(f'group {i}')\n    group_df = train_df[train_df.group==i]\n    df = build_change_list(group_df)\n    df_list.append(df)\n\ndf_sm = pd.concat(df_list)\n")


# In[7]:


# clean data a bit - same sources, change in protein, same pH and pH not too far of 7

df_clean = df_sm[(df_sm.data_source1 == df_sm.data_source2)&(df_sm.pH1 == df_sm.pH2)]
df_clean = df_clean[(df_clean.pH1>=6) & (df_clean.pH1<=8)]
df_clean = df_clean[df_clean.position1 != 0]

df_clean['target'] = df_clean['tm2'] - df_clean['tm1'] 

print(len(df_clean))
display(pd.crosstab(df_clean.change1,df_clean.change2).style.background_gradient(axis=None, cmap="YlGnBu"))


# In[8]:


# avg target by protein changes, independtly of emplacement

avg_target = df_clean.groupby(['change1','change2']).target.mean()
display(avg_target.unstack().fillna(0).style.background_gradient(axis=None, cmap="RdYlBu").format('{:.2f}'))


# # Exploring single mutations from additional train data
# 
# Using @jinyuansun additional data: https://github.com/JinyuanSun/mutation-stability-data

# In[9]:


add_data = pd.read_csv('../input/nesp-additional-data-from-jinyuansun/tm.csv')
add_data.head()


# In[10]:


pd.crosstab(add_data.WT,add_data.MUT).style.background_gradient(axis=None, cmap="YlGnBu")


# In[11]:


avg_target2 = add_data.groupby(['WT','MUT']).mean().dTm
display(avg_target2.unstack().fillna(0).style.background_gradient(axis=None, cmap="RdYlBu").format('{:.2f}'))


# # Exploring single mutations from test data
# 
# The goal here is to build and explore single changes mutations we need to predict for the test set.

# In[12]:


base = 'VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK'

test_df = pd.read_csv("/kaggle/input/novozymes-enzyme-stability-prediction/test.csv")


# In[13]:


def change_list(wild, mutation_list):

    list_output = []

    for mutation in mutation_list:
        edits = Levenshtein.editops(wild, mutation)
        if len(edits):
            list_output.append(edits[0]+tuple(mutation[edits[0][1]])+tuple(base[edits[0][1]]))
        else:
            list_output.append(('replace', 0, 0, 'A', 'A'))

    changes = pd.DataFrame(list_output,columns=['operation','position1','position2','change1','change2'])
    changes.change2 = np.where(changes.operation=='delete','',changes.change2)
    
    return changes


# In[14]:


changes = change_list(base, test_df.protein_sequence.to_list())
changes.operation.value_counts()


# We might want to treat replacement and deletions differently.

# In[15]:


changes.position1.hist();


# Sligthly less changes at the 'start' of the chain.

# In[16]:


pd.crosstab(changes.change1,changes.change2).style.background_gradient(axis=None, cmap="YlGnBu")


# Unequal repartition of modifications. We can notice that amino-acids C, H, M are not used as mutations. 

# In[17]:


test_df['past_tm'] = changes.merge(avg_target,left_on=['change1','change2'],right_index=True,how='left').target.fillna(0)
test_df['past_tm2'] = changes.merge(avg_target2,left_on=['change1','change2'],right_index=True,how='left').dTm.fillna(0)


# # b_factor as baseline
# 
# The b_factor is a metric of relative stability for each acid in the chain (see discussion [here](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/354476)). It seems strongly negatively correlated with tm. (It is not entirely clear that the negative correlation is a physical property or a non 'random' design of the test set).
# plot of the pdb file from: https://www.kaggle.com/code/alejopaullier/nesp-eda-xgboost-baseline-0-025#What-is-PDB?-%E2%86%91

# In[18]:


PDB_FILE = "/kaggle/input/novozymes-enzyme-stability-prediction/wildtype_structure_prediction_af2.pdb"
pdb_df =  PandasPdb().read_pdb(PDB_FILE)
pdb_df.df.keys()


# In[19]:


dict_enzyme = {
    'ALA':'A',
    'ARG':'R',
    'ASN':'N',
    'ASP':'D',
    'CYS':'C',
    'GLU':'E',
    'GLN':'Q',
    'GLY':'G',
    'HIS':'H',
    'ILE':'I',
    'LEU':'L',
    'LYS':'K',
    'MET':'M',
    'PHE':'F',
    'PRO':'P',
    'SER':'S',
    'THR':'T',
    'TRP':'W',
    'TYR':'Y',
    'VAL':'V'
}

atom_df = pdb_df.df['ATOM']
atom_df['residue_letter'] = atom_df.residue_name.map(dict_enzyme)


# In[20]:


fig = px.scatter_3d(atom_df, x = "x_coord",
                    y = "y_coord",
                    z = "z_coord",
                    color = "residue_letter")
fig.update_traces(marker = dict(size = 3))
fig.update_coloraxes(showscale = False)
fig.update_layout(template = "plotly_dark")
fig.show()


# In[21]:


test_df['modif'] = changes.position1

map_number_to_b = atom_df.groupby('residue_number').b_factor.mean()
test_df['b_factor'] = (test_df.modif+1).map(map_number_to_b).fillna(0)

atom_df['distance'] = np.sqrt(np.square(atom_df.x_coord) + np.square(atom_df.y_coord) + np.square(atom_df.z_coord))
map_number_to_distance = atom_df.groupby('residue_number').distance.mean()
test_df['distance'] = (test_df.modif+1).map(map_number_to_distance).fillna(0)

plt.scatter(map_number_to_b.index, map_number_to_b.values);


# In[22]:


fig = px.scatter_3d(atom_df, x = "x_coord",
                    y = "y_coord",
                    z = "z_coord",
                    color = "b_factor")
fig.update_traces(marker = dict(size = 3))
fig.update_coloraxes(showscale = False)
fig.update_layout(template = "plotly_dark")
fig.show()


# In[23]:


# try to improve b-factor prediction by composition of amino acid - not really working
#remove outliers ?

atom_remove_low = atom_df[atom_df.b_factor>80].copy()

X = atom_remove_low.groupby(['residue_number','element_symbol']).size().unstack(fill_value=0)
y = atom_remove_low.groupby(['residue_number']).b_factor.mean()

reg = LinearRegression().fit(X, y)

X.index = atom_remove_low.groupby('residue_number').residue_letter.first()

value_by_letter = pd.DataFrame({'letter':X.index, 'pred':reg.predict(X)})
map_b_by_letter = value_by_letter.groupby('letter').mean()

changes['Delta1'] = changes.change1.map(map_b_by_letter['pred'])
changes['Delta2'] = changes.change2.map(map_b_by_letter['pred'])
changes['Delta'] = (changes['Delta2'] - changes['Delta1']).fillna(0)


# # Get additional PDB
# 
# Additional data contains different wiltypes, for which we might want the associated PDB. It appears there is a very easy way to get common wildtype pdb files from rcsb.

# In[24]:


PDB_ex = add_data.PDB.iloc[0]
PDB_ex_df = urllib.request.urlretrieve(f'http://files.rcsb.org/download/{PDB_ex}.pdb', f'{PDB_ex}.pdb')


# In[25]:


PDB_FILE = f"./{PDB_ex}.pdb"
pdb_df =  PandasPdb().read_pdb(PDB_FILE)
pdb_df.df.keys()


# In[26]:


atom_df = pdb_df.df['ATOM']
atom_df['residue_letter'] = atom_df.residue_name.map(dict_enzyme)


# In[27]:


fig = px.scatter_3d(atom_df, x = "x_coord",
                    y = "y_coord",
                    z = "z_coord",
                    color = 'residue_letter')
fig.update_traces(marker = dict(size = 3))
fig.update_coloraxes(showscale = False)
fig.update_layout(template = "plotly_dark")
fig.show()


# # Enzyme properties
# 
# Each enzyme has different properties. Here we try to build feature as the change in property due to the individual mutation.

# In[28]:


properties = pd.read_csv('../input/aminoacids-physical-and-chemical-properties/aminoacids.csv')
properties = properties[['Letter', 'Residue Weight', 'pKa1', 'pKb2', 'pKx3', 'pl4', 'H', 'VSC', 'P1', 'P2', 'SASA', 'NCISC']]
properties = properties.set_index('Letter').fillna(0)

properties.hist();


# Using moth data we can explore amino-acids properties and try to see if their marginal change in property impact the target. 
# 
# 
# 

# I explore other correlations. I test the correlation between target and the change in property due to change of amino-acid.
# 
# Results so far:
# 
# | Feature  (Delta)       | Correlation (spearman)      |
# |--------------|-----------|
# | Residue Weight | -0.054      |
# | pKa1     | 0.05  |
# | pKb2     | 0.06  |
# | pKx3     | -0.029  |
# | pl4     | -0.016  |
# | H     |  0.05  |
# | VSC     | -0.05  |
# | P1     | -0.056  |
# | P2     | -0.043  |
# | SASA     | -0.056  |
# | NCISC     | 0.093  |

# In[29]:


changes = changes.merge(properties.add_suffix('1'),left_on='change1',right_index=True,how='left')
changes = changes.merge(properties.add_suffix('2'),left_on='change2',right_index=True,how='left')
changes = changes.fillna(0)

for c in ['Residue Weight', 'pKa1', 'pKb2', 'pKx3', 'pl4', 'H', 'VSC', 'P1', 'P2', 'SASA', 'NCISC']:
    changes['Delta '+c] = changes[c+'1']-changes[c+'2']


# # Blosum
# 
# There is an existing literrature about comparing properties of enzymes. The blosum score aggregate different properties to provide a general score.
# Similarity based measure we can use to predict change in tm; from https://www.kaggle.com/code/kvigly55/fork-of-nesp-b-factor-nad-subsitutions

# In[30]:


matrix = bl.BLOSUM(80) # 45, 50, 62, 80 and 90

changes_str = changes.change1 + changes.change2
test_df['blosum_score'] = changes_str.apply(lambda x: matrix[x])
test_df['blosum_score'] = np.nan_to_num(test_df['blosum_score'],nan=0.0, posinf=10, neginf=-10)

test_df['blosum_score_adj'] = (1 / (1+np.exp(-test_df['blosum_score'])))
test_df['b_factor_adj'] = test_df['b_factor'] * test_df['blosum_score_adj'] 

test_df['rank_blosum'] = (test_df['blosum_score_adj']).transform('rank')
test_df['rank_b_factor'] = (-test_df['b_factor']).transform('rank')

test_df['avg_rank'] = test_df['rank_blosum'] + 1.2*test_df['rank_b_factor']


# # Reverse Engineering ?
# 
# Mutations in the test set do not appear to be chose at random. That is the test building process give some indication of the goal (stabilisation) that novozymes want to achieve (stability). 
# 
# We get the following correlations:
# 
# | Feature  | Correlation (spearman)      |
# |--------------|-----------|
# | Count of change affecting that position | 0.051      |
# | Count of replaced protein     | -0.009  |
# | Count of new protein     | 0.116  |
# 
# That is Novozyme has targeted some position and favored some replacements.
# 

# In[31]:


position_change = changes.position1.value_counts()
old_prot_count = changes.change1.value_counts()
new_prot_count = changes.change2.value_counts()


# In[32]:


changes['count_pos'] = changes.position1.map(position_change)
changes['old_prot_count'] = changes.change2.map(old_prot_count)
changes['new_prot_count'] = changes.change2.map(new_prot_count)
changes = changes.fillna(0)


# # Correlation study

# In[33]:


all_feature = pd.concat([test_df[['pH', 'past_tm', 'b_factor', 'distance', 'blosum_score', 'past_tm2']],changes[['position1', 'Delta', 'Delta Residue Weight', 'Delta pKa1', 'Delta pKb2', 'Delta pKx3', 'Delta pl4', 'Delta H', 'Delta VSC', 'Delta P1', 'Delta P2', 'Delta SASA', 'Delta NCISC', 'count_pos', 'old_prot_count','new_prot_count']]],axis=1)
all_feature.corr('spearman').style.background_gradient(axis=None, cmap="YlGnBu").format('{:.2f}')


# In[34]:


pca = PCA()
scaler = StandardScaler()
all_feature = scaler.fit_transform(all_feature)
all_feature = pca.fit_transform(all_feature)

plt.plot(pca.explained_variance_ratio_.cumsum())


# In[35]:


pd.DataFrame(all_feature).corrwith(test_df['b_factor'], method='spearman')


# # Submission
# 
# We submit each feature to test correlation with the target.  Best feature so far is -test_df['b_factor'] with 0.25 spearman correlation.

# In[36]:


alpha = 0
variables = ['Residue Weight', 'pKa1', 'pKb2', 'pKx3', 'pl4', 'H', 'VSC', 'P1', 'P2', 'SASA', 'NCISC']

submission = pd.DataFrame()
submission["tm"] =  test_df['avg_rank']
submission["seq_id"] = test_df["seq_id"]
submission.to_csv("submission.csv", index=False)

