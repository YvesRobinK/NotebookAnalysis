#!/usr/bin/env python
# coding: utf-8

# ## Setup
# 

# In[1]:


get_ipython().system('nvidia-smi')


# In[2]:


get_ipython().system('pip install -q biopandas')
get_ipython().system('conda install -y -c bioconda msms')


# In[3]:


GLOBAL_SEED = 42

import os
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
import sys
import random as rnd
import gc
from time import time
import copy

import pandas as pd
from biopandas.pdb import PandasPdb
import numpy as np
from numpy import random as np_rnd
import pickle
from tqdm import tqdm

from matplotlib import pyplot as plt
import seaborn as sns

import sklearn as skl
from sklearn import linear_model as lm
from sklearn import model_selection
from sklearn.metrics.pairwise import cosine_similarity
from itertools import permutations, combinations
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn import metrics
from sklearn import model_selection
from sklearn.neural_network import MLPRegressor

import optuna
from optuna import Trial, create_study
from optuna.samplers import TPESampler

import lightgbm as lgb
import xgboost as xgb
import catboost as cat

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from scipy.special import softmax 
from scipy.stats import entropy
from scipy.special import expit
from scipy.spatial import distance

try:
    import cuml
    import cupy as cp
    from cuml import PCA, TruncatedSVD
except:
    from sklearn.decomposition import PCA, TruncatedSVD

from Bio.SubsMat import MatrixInfo
from scipy.stats import rankdata, spearmanr # smaller value, higher rank

import warnings
warnings.filterwarnings('ignore')


# In[4]:


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # python random
    rnd.seed(seed)
    # numpy random
    np_rnd.seed(seed)
    # RAPIDS random
    try:
        cp.random.seed(seed)
    except:
        pass
    # tf random
    try:
        tf_rnd.set_seed(seed)
    except:
        pass
    # pytorch random
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except:
        pass

def pickleIO(obj, src, op="w"):
    if op=="w":
        with open(src, op + "b") as f:
            pickle.dump(obj, f)
    elif op=="r":
        with open(src, op + "b") as f:
            tmp = pickle.load(f)
        return tmp
    else:
        print("unknown operation")
        return obj
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def get_scaled_dist(pos, seq_len=221):
    return 1.0 - (np.abs((seq_len / 2) - pos) / (seq_len / 2))

def findIdx(data_x, col_names):
    return [int(i) for i, j in enumerate(data_x) if j in col_names]


# # Set config & Loading dataset

# In[5]:


class CFG:
    debug = False
    colab = False
    target = "target_ensemble"
    TF = True
    only_neg = True
    target_ensemble = True

    TRAIN_FEATURES_DIR = "/content/14656-unique-mutations-voxel-features-pdbs/"
    kaggle_dataset_path = "./" if colab else "../input/"
    max_seq_len = 16
    max_seq_len = 512
    min_group_size = 5
    voxel_size = 8
    
    n_folds = 5
    batch_size = 8
    epochs = 20
    early_stopping_rounds = 10
    
if CFG.debug:
    CFG.epochs = 2


# In[6]:


seed_everything(GLOBAL_SEED)

# output : score
# type : "maximize" or "minimize"
output_dic = {
    "output": [],
    "type": "maximize"
}


# In[7]:


test_base_seq = 'VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK'
test_base_seq_id = 32559

aa_map = {'VAL': 'V', 'PRO': 'P', 'ASN': 'N', 'GLU': 'E', 'ASP': 'D', 'ALA': 'A', 'THR': 'T', 'SER': 'S',
          'LEU': 'L', 'LYS': 'K', 'GLY': 'G', 'GLN': 'Q', 'ILE': 'I', 'PHE': 'F', 'CYS': 'C', 'TRP': 'W',
          'ARG': 'R', 'TYR': 'Y', 'HIS': 'H', 'MET': 'M'}
aa_map["X"] = "X"
aa_map_2 = {x:y for x,y in zip(list(aa_map.values()),np.arange(21))}

# https://www.kaggle.com/datasets/alejopaullier/aminoacids-physical-and-chemical-properties
aa_props = pd.read_csv(CFG.kaggle_dataset_path + 'aminoacids-physical-and-chemical-properties/aminoacids.csv').set_index('Letter')
PROPS = ['Molecular Weight', 'Residue Weight', 'pKa1', 'pKb2', 'pKx3', 'pl4', 
         'H', 'VSC', 'P1', 'P2', 'SASA', 'NCISC', 'carbon', 'hydrogen', 'oxygen']
# remove pKx3 which includes na values
PROPS.remove("pKx3")
print('Amino Acid properties dataframe. Shape:', aa_props.shape )
for i in aa_props.index:
    if i not in aa_map.values():
        print(i)
        
aa_groups ={
    # Electrically Charged Side Chains - positive
    "AAG0": ["R", "H", "K"],
    # Electrically Charged Side Chains - negative
    "AAG1": ["D", "E"],
    # Polar Uncharged Side Chains
    "AAG2": ["S", "T", "N", "Q"],
    # Hydrophobic Side Chains
    "AAG3": ["A", "V", "I", "L", "M", "F", "Y", "W"],
}
tmp = []
for i in list(aa_groups.values()):
    tmp.extend(i)
aa_groups["AAG4"] = diff(list(aa_map.values()), tmp) + ["X"]
aa_groups_mlb = skl.preprocessing.MultiLabelBinarizer()
aa_groups_mlb.fit([[i] for i in aa_groups.keys()])

aa_interaction_indiv_mlb = skl.preprocessing.MultiLabelBinarizer()
aa_interaction_indiv = list(combinations(sorted(list(aa_map.values())), 2))
aa_interaction_indiv = [[i[0] + "_" + i[1]] for i in aa_interaction_indiv]
aa_interaction_indiv_mlb.fit(aa_interaction_indiv)

aa_interaction_groups_mlb = skl.preprocessing.MultiLabelBinarizer()
aa_interaction_groups = list(combinations(sorted(list(aa_groups.keys())), 2))
aa_interaction_groups = [[i[0] + "_" + i[1]] for i in aa_interaction_groups]
aa_interaction_groups_mlb.fit(aa_interaction_groups)

aa_props = aa_props.loc[[False if i in ["O", "U"] else True for i in aa_props.index]]
aa_props.loc["X"] = aa_props.iloc[:20].mean()
# aa_props.loc["X"] = 0
# aa_props.loc["X", ["Name", "Abbr", "Molecular Formula", "Molecular Formula"]] = "X"

aa_prop_scaler = skl.preprocessing.MinMaxScaler()
aa_props_scaled = pd.DataFrame(aa_prop_scaler.fit_transform(aa_props[PROPS]), index=aa_props.index, columns=PROPS)

# imputation 'X' values as mean of each features
aa_props_scaled.loc["X", PROPS] = aa_props_scaled[PROPS].iloc[:20].mean()
# aa_props_scaled.loc["X", ["Name", "Abbr", "Molecular Formula", "Molecular Formula"]] = "X"

# aa_props_scaled.head(21)


# In[8]:


def get_sub_matrix(matrix_name="blosum100"):
    if matrix_name == "demask":
        dff = pd.read_csv('../input/nesp-test-wildtype-pdb/matrix.txt', sep='\t')
        letters = list( dff.columns )
        l_dict = {x:y for x,y in zip(letters,range(20))}
        sub_matrix = {}
        for x in letters:
            for y in letters:
                sub_matrix[(x,y)] = dff.iloc[l_dict[x],l_dict[y]]
    else:
        sub_matrix = getattr(MatrixInfo, matrix_name)
        sub_matrix.update({(k[1], k[0]):v for k,v in sub_matrix.items() if (k[1], k[0]) not in list(sub_matrix.keys())})
    return sub_matrix

# subtitution matrix & default value when nan
CFG.sub_matrix_dic = {
    "blosum100": [get_sub_matrix("blosum100"), np.fromiter(get_sub_matrix("blosum100").values(), dtype="float32").min(), np.abs(np.fromiter(get_sub_matrix("blosum100").values(), dtype="float32")).max()],
    "blosum80": [get_sub_matrix("blosum80"), np.fromiter(get_sub_matrix("blosum80").values(), dtype="float32").min(), np.abs(np.fromiter(get_sub_matrix("blosum80").values(), dtype="float32")).max()],
    "blosum62": [get_sub_matrix("blosum62"), np.fromiter(get_sub_matrix("blosum62").values(), dtype="float32").min(), np.abs(np.fromiter(get_sub_matrix("blosum62").values(), dtype="float32")).max()],
    "blosum60": [get_sub_matrix("blosum60"), np.fromiter(get_sub_matrix("blosum60").values(), dtype="float32").min(), np.abs(np.fromiter(get_sub_matrix("blosum60").values(), dtype="float32")).max()],
    "blosum45": [get_sub_matrix("blosum45"), np.fromiter(get_sub_matrix("blosum45").values(), dtype="float32").min(), np.abs(np.fromiter(get_sub_matrix("blosum45").values(), dtype="float32")).max()],
    "blosum40": [get_sub_matrix("blosum40"), np.fromiter(get_sub_matrix("blosum40").values(), dtype="float32").min(), np.abs(np.fromiter(get_sub_matrix("blosum40").values(), dtype="float32")).max()],
    "demask": [get_sub_matrix("demask"), np.fromiter(get_sub_matrix("demask").values(), dtype="float32").min(), np.abs(np.fromiter(get_sub_matrix("demask").values(), dtype="float32")).max()],
}


# In[9]:


def transform_coord(data_pdb, voxel_size=CFG.voxel_size):
    lb_encoder = skl.preprocessing.KBinsDiscretizer(voxel_size, encode="ordinal", strategy="uniform")
    lb_encoder.fit(data_pdb[["x_coord", "y_coord", "z_coord"]].to_numpy().flatten().reshape(-1, 1))
    data_pdb[["x_coord"]] = lb_encoder.transform(data_pdb[["x_coord"]])
    data_pdb[["y_coord"]] = lb_encoder.transform(data_pdb[["y_coord"]])
    data_pdb[["z_coord"]] = lb_encoder.transform(data_pdb[["z_coord"]])
    data_pdb[["x_coord", "y_coord", "z_coord"]] = data_pdb[["x_coord", "y_coord", "z_coord"]].astype("int32")
    return data_pdb


# In[10]:


pdb_parser = PDBParser(QUIET=1)

def get_sasa(pdb_path, return_struct=False, pdb_identifier="ESM", sr_n_points=250):
    sr = ShrakeRupley(n_points=sr_n_points)
    struct = pdb_parser.get_structure(pdb_identifier, pdb_path)[0]
    sr.compute(struct, level="R")
    return [x.sasa for x in struct.get_residues()]

def get_rd_cd(pdb_path, return_struct=False, pdb_identifier="ESM", sr_n_points=250):
    struct = pdb_parser.get_structure(pdb_identifier, pdb_path)[0]
    output = ResidueDepth(struct)
    rd, cd = list(zip(*[(x[1][0], x[1][1]) for x in output]))
    return rd, cd


# In[11]:


ohe_aa_group = skl.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore")
ohe_aa_group.fit(pd.DataFrame(aa_groups.keys()))


# In[12]:


def get_features(df, inference=False):
    rows = []
    pbar = tqdm(total=len(df))

    for i in (df["wt_seq"].unique()):
        df_group = df[df["wt_seq"] == i].reset_index(drop=True)
        if inference:
            pdb_path = f"/kaggle/input/nesp-create-data-esm-pdb/pdbs/test/{df_group['seq_map'].iloc[0]}/wt.pdb"
        else:
            pdb_path = f"/kaggle/input/nesp-create-data-esm-pdb/pdbs/train/{df_group['seq_map'].iloc[0]}/wt.pdb"
        wt_pdb = PandasPdb().read_pdb(pdb_path).df["ATOM"]
        wt_plddt_mean = wt_pdb["b_factor"].mean()
        wt_plddt_std = wt_pdb["b_factor"].std()
        tmp_pdb = wt_pdb.copy()

        wt_pdb = transform_coord(wt_pdb)
        wt_pdb = wt_pdb.groupby(["residue_number", "residue_name"])[["x_coord", "y_coord", "z_coord", "occupancy", "b_factor"]].mean().reset_index()
        pdb_coord = wt_pdb[["x_coord", "y_coord", "z_coord"]]

        wt_pdb[["x_coord", "y_coord", "z_coord"]] = np.clip(wt_pdb[["x_coord", "y_coord", "z_coord"]].round(), 0, 16).astype("int32")
        wt_pdb["aa"] = wt_pdb["residue_name"].map(aa_map)
        wt_pdb[PROPS] = aa_props_scaled.loc[wt_pdb["aa"], PROPS].values
        wt_aa_group = []
        for j in wt_pdb["aa"]:
            output_group = "AAG4"
            for k, v in aa_groups.items():
                if j in v:
                    output_group = k
                    break
            wt_aa_group.append(output_group)
#         wt_pdb[["aa_group_" + i for i in ohe_aa_group.categories_[0]]] = ohe_aa_group.transform(pd.DataFrame(wt_aa_group))

        plddt = np.array(wt_pdb["b_factor"])
        sasa = np.array(get_sasa(pdb_path))
        rd, cd = get_rd_cd(pdb_path)
        rd = np.array(rd)
        cd = np.array(cd)

        for idx, mt_row in df_group.iterrows():
            row = {}
            if inference:
                pdb_path = f"/kaggle/input/nesp-create-data-esm-pdb/pdbs/test/{df_group['seq_map'].iloc[0]}/{mt_row['wt']}{mt_row['pos']}{mt_row['mt']}.pdb"
            else:
                pdb_path = f"/kaggle/input/nesp-create-data-esm-pdb/pdbs/train/{df_group['seq_map'].iloc[0]}/{mt_row['wt']}{mt_row['pos']}{mt_row['mt']}.pdb"

            mt_pdb = PandasPdb().read_pdb(pdb_path).df["ATOM"]
            mt_plddt_mean = mt_pdb["b_factor"].mean()
            mt_plddt_std = mt_pdb["b_factor"].std()

            wt_residue = tmp_pdb[tmp_pdb["residue_number"] == (mt_row["pos"]+1) + 0].reset_index(drop=True)
            mt_residue = mt_pdb[mt_pdb["residue_number"] == (mt_row["pos"]+1) + 0].reset_index(drop=True)
            prev = mt_pdb[mt_pdb["residue_number"] == (mt_row["pos"]+1) - 1].reset_index(drop=True)
            post = mt_pdb[mt_pdb["residue_number"] == (mt_row["pos"]+1) + 1].reset_index(drop=True)

            if ((len(prev) > 0) & (len(post) > 0)) & ((aa_map[wt_residue["residue_name"].iloc[0]] == mt_row["wt"]) & (aa_map[mt_residue["residue_name"].iloc[0]] == mt_row["mt"])):
                # BACKBONE ATOMS
                atm = ['N', 'H', 'CA', 'O']
                prev = prev.loc[prev.atom_name.isin(atm)]
                c_prev = np.array( [prev.x_coord.mean(),prev.y_coord.mean(),prev.z_coord.mean()] )
                post = post.loc[post.atom_name.isin(atm)]
                c_post = np.array( [post.x_coord.mean(),post.y_coord.mean(),post.z_coord.mean()] )

                # WT angle of VECTORS
                tmp = wt_residue.loc[wt_residue.atom_name.isin(atm)]
                c_tmp = np.array( [tmp.x_coord.mean(),tmp.y_coord.mean(),tmp.z_coord.mean()] )
                vec_a = c_prev - c_tmp
                vec_b = c_post - c_tmp
                norm_a = np.sqrt(vec_a.dot(vec_a))
                norm_b = np.sqrt(vec_b.dot(vec_b))
                wt_cos_angle = vec_a.dot(vec_b) / norm_a / norm_b

                # MT angle of VECTORS
                tmp = mt_residue.loc[mt_residue.atom_name.isin(atm)]
                c_tmp = np.array( [tmp.x_coord.mean(),tmp.y_coord.mean(),tmp.z_coord.mean()] )
                vec_a = c_prev - c_tmp
                vec_b = c_post - c_tmp
                norm_a = np.sqrt(vec_a.dot(vec_a))
                norm_b = np.sqrt(vec_b.dot(vec_b))
                mt_cos_angle = vec_a.dot(vec_b) / norm_a / norm_b
            else:
                wt_cos_angle = -2.0
                mt_cos_angle = -2.0
            row["dist_from_center_3d"] = distance.euclidean(((pdb_coord.max(axis=0) + pdb_coord.min(axis=0)) / 2).values, pdb_coord.iloc[mt_row["pos"]].values)
            row["cos_angle_mt"] = mt_cos_angle
            row["cos_angle_diff"] = wt_cos_angle - mt_cos_angle

            mt_pdb = mt_pdb.groupby(["residue_number", "residue_name"])[["x_coord", "y_coord", "z_coord", "occupancy", "b_factor"]].mean().reset_index()

            mt_pdb[["x_coord", "y_coord", "z_coord"]] = wt_pdb[["x_coord", "y_coord", "z_coord"]].values
            mt_pdb["aa"] = mt_pdb["residue_name"].map(aa_map)
            mt_pdb[PROPS] = aa_props_scaled.loc[mt_pdb["aa"], PROPS].values
            mt_aa_group = []
            for j in mt_pdb["aa"]:
                output_group = "AAG4"
                for k, v in aa_groups.items():
                    if j in v:
                        output_group = k
                        break
                mt_aa_group.append(output_group)
#             mt_pdb[["aa_group_" + j for j in ohe_aa_group.categories_[0]]] = ohe_aa_group.transform(pd.DataFrame(mt_aa_group))
            mt_pdb["sasa"] = sasa
            mt_pdb["rd"] = rd
            mt_pdb["cd"] = cd
            mt_pdb["pLDDT"] = plddt

    #         feauture_3d = np.zeros((mt_pdb.drop(["residue_number", "residue_name", "occupancy", "aa"], axis=1).shape[1] - 3, CFG.voxel_size, CFG.voxel_size, CFG.voxel_size))
    #         for idx2, j in enumerate(mt_pdb[["x_coord", "y_coord", "z_coord"]].values):
    #             feauture_3d[:, j[0], j[1], j[2]] = mt_pdb.drop(["residue_number", "residue_name", "occupancy", "aa"], axis=1).iloc[idx2, 3:].values

            row["wt_aa_group"] = wt_aa_group[mt_row["pos"]]
            row["mt_aa_group"] = mt_aa_group[mt_row["pos"]]

            row["blosum100"] = CFG.sub_matrix_dic["blosum100"][0].get((mt_row["wt"], mt_row["mt"]), CFG.sub_matrix_dic["blosum100"][1])
            row["blosum62"] = CFG.sub_matrix_dic["blosum62"][0].get((mt_row["wt"], mt_row["mt"]), CFG.sub_matrix_dic["blosum100"][1])
            row["blosum45"] = CFG.sub_matrix_dic["blosum45"][0].get((mt_row["wt"], mt_row["mt"]), CFG.sub_matrix_dic["blosum100"][1])
            row["demask"] = CFG.sub_matrix_dic["demask"][0].get((mt_row["wt"], mt_row["mt"]), CFG.sub_matrix_dic["demask"][1])

            # estimate plddt, sasa, residue depth, ca depth with values of wt & demask score
            mt_pdb["pLDDT"].iloc[mt_row["pos"]] = (-1) * (mt_pdb["pLDDT"].iloc[mt_row["pos"]] * (1 - (row["demask"] / 1e+1)))
            mt_pdb["sasa"].iloc[mt_row["pos"]] = mt_pdb["sasa"].iloc[mt_row["pos"]] * (1 + row["demask"])
            mt_pdb["rd"].iloc[mt_row["pos"]] = (-1) * (mt_pdb["rd"].iloc[mt_row["pos"]] * (1 - row["demask"]))
            mt_pdb["cd"].iloc[mt_row["pos"]] = (-1) * (mt_pdb["cd"].iloc[mt_row["pos"]] * (1 - row["demask"]))

            row["pLDDT"] = mt_pdb["pLDDT"].iloc[mt_row["pos"]]
            row["sasa"] = mt_pdb["sasa"].iloc[mt_row["pos"]]
            row["rd"] = mt_pdb["rd"].iloc[mt_row["pos"]]
            row["cd"] = mt_pdb["cd"].iloc[mt_row["pos"]]
            row.update((aa_props.loc[mt_row["wt"], PROPS] - aa_props.loc[mt_row["mt"], PROPS]).to_dict())

            seq_obj = ProteinAnalysis(mt_row["mt_seq"])
            row["pt_molecular_weight"] = seq_obj.molecular_weight()
            row["pt_aromaticity"] = seq_obj.aromaticity()
            row["pt_instability_index"] = seq_obj.instability_index()
            row["pt_flexibility_std"] = np.array(seq_obj.flexibility()).std()
            row["pt_gravy"] = seq_obj.gravy()
            row["pt_isoelectric_point"] = seq_obj.isoelectric_point()
            row["pt_molar_extinction_coefficient"] = seq_obj.molar_extinction_coefficient()[0]
            row["pLDDT_global_mean"] = wt_plddt_mean - mt_plddt_mean
            row["pLDDT_global_std"] = wt_plddt_std - mt_plddt_std
            row["pLDDT_local_mean"] = wt_residue["b_factor"].mean() - mt_residue["b_factor"].mean()
            row["pLDDT_local_std"] = wt_residue["b_factor"].std() - mt_residue["b_factor"].std()
            row.update(seq_obj.amino_acids_percent)
            if "X" not in row.keys():
                row["X"] = 0.0

            row["dist_from_center"] = get_scaled_dist(mt_row["pos"], len(mt_row["wt_seq"]))
            row["mt_p"] = 1.0 if mt_row["mt"] == "P" else 0.0
#             row["blosum100_neg"] = 1.0 if row["blosum100"] < 0.0 else 0.0
#             row["blosum62_neg"] = 1.0 if row["blosum62"] < 0.0 else 0.0
#             row["blosum45_neg"] = 1.0 if row["blosum45"] < 0.0 else 0.0
            row["blosum_neg_ratio"] = ((1.0 if row["blosum100"] < 0.0 else 0.0) + (1.0 if row["blosum62"] < 0.0 else 0.0) + (1.0 if row["blosum45"] < 0.0 else 0.0)) / 3

#             try:
#                 prev_aa = mt_row["mt_seq"][mt_row["pos"]-1]
#             except:
#                 prev_aa = ""

#             try:
#                 post_aa = mt_row["mt_seq"][mt_row["pos"]+1]
#             except:
#                 post_aa = ""

    #             prev_aa_group = ""
    #             post_aa_group = ""
    #             for k, v in aa_groups.items():
    #                 if prev_aa in v:
    #                     prev_aa_group = k
    #                 if post_aa in v:
    #                     post_aa_group = k
    #             row.update({"side_aa_group_" + j: k for j, k in zip(aa_groups_mlb.classes_, aa_groups_mlb.transform([[prev_aa_group], [post_aa_group]]).mean(axis=0))})

            if "ddG" in mt_row.index:
                row["ddG"] = mt_row["ddG"]

            rows.append(row)
            pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


# ## Create train data

# In[13]:


df_full = pd.read_csv("/kaggle/input/nesp-create-data-esm-pdb/df_train.csv")
df_full = get_features(df_full.iloc[:100] if CFG.debug else df_full)


# In[14]:


df_full[["wt_aa_group_" + i for i in ohe_aa_group.categories_[0]]] = ohe_aa_group.transform(df_full[["wt_aa_group"]])
df_full[["mt_aa_group_" + i for i in ohe_aa_group.categories_[0]]] = ohe_aa_group.transform(df_full[["mt_aa_group"]])
df_full= df_full.drop(["wt_aa_group", "mt_aa_group"], axis=1)


# In[15]:


df_full = df_full.sample(frac=1, random_state=42).reset_index(drop=True)


# In[16]:


df_full.head()


# In[17]:


df_full.info()


# In[18]:


df_full.iloc[:, :36].columns


# In[19]:


scaled_vars = df_full.iloc[:, :36].columns


# # Define helper functions

# In[20]:


# optuna function
def optuna_objective_function(trial: Trial, fold, train_x, train_y, train_groups, val_x, val_y, val_groups, categoIdx,
                              model_name, output_container, ntrees=1000, eta=1e-2):

    if model_name == "ElasticNet":
        # for regression
        tuner_params = {
            "alpha": trial.suggest_categorical("C", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0, step=0.05)
        }
        # for RAPIDS module
        try:
            model = cuml.ElasticNet(output_type="numpy", **tuner_params)
            model.fit(train_x, train_y)
        except:
            model = lm.ElasticNet(**tuner_params)
            model.fit(train_x, train_y)
    elif model_name == "LGB_RF":
        # objective
        # regession : "mae", "mse"
        # classification - binary : "binary"
        # classification - binary : "multiclass" (num_class=n)
        # ranking : "xe_ndcg_mart"

        # metric
        # regession : "mae", "mse", "rmse"
        # classification - binary : "binary_logloss", "binary_error", "auc"
        # classification - muticlass : "multi_logloss", "multi_error"
        # ranking : "ndcg", "map"

        tuning_params = {
            # "learning_rate": trial.suggest_categorical("learning_rate", [1e-2, 5e-3, 1e-3]),
            "num_leaves": trial.suggest_categorical("num_leaves", [pow(2, i) - 1 for i in [4, 5, 6, 7, 8]]),
            # goss sampling hyper-parameter replacing the "sumample"
            "subsample": trial.suggest_float("subsample", 0.5, 0.8, step=0.1),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 51, step=2),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8, step=0.1),
            "reg_alpha": trial.suggest_categorical("reg_alpha", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
            "reg_lambda": trial.suggest_categorical("reg_lambda", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
            "min_child_weight": trial.suggest_categorical("min_child_weight", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 51, step=2),
            "min_gain_to_split": trial.suggest_categorical("min_gain_to_split", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
            # # for binary
            # "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
        }
        model = lgb.LGBMRegressor(boosting_type="rf", objective="mae",
                                   n_estimators=int(ntrees / 10), device_type="gpu",
                                   random_state=fold, verbose=-1, **tuning_params)
        cb_list = [
            lgb.early_stopping(stopping_rounds=int(ntrees / 10 * 0.2), first_metric_only=True, verbose=False),
        ]
        model.fit(train_x, train_y, categorical_feature=categoIdx,
            eval_set=(val_x,val_y), eval_metric="mae", callbacks=cb_list)
        output_container["ntrees"] = model.best_iteration_
    elif model_name == "LGB_GOSS":
        # objective
        # regession : "mae", "mse"
        # classification - binary : "binary"
        # classification - binary : "multiclass" (num_class=n)
        # ranking : "xe_ndcg_mart"

        # metric
        # regession : "mae", "mse", "rmse"
        # classification - binary : "binary_logloss", "binary_error", "auc"
        # classification - muticlass : "multi_logloss", "multi_error"
        # ranking : "ndcg", "map"

        tuning_params = {
            "num_leaves": trial.suggest_categorical("num_leaves", [pow(2, i) - 1 for i in [4, 6, 8]]),
            # goss sampling hyper-parameter replacing the "sumample"
            "top_rate": trial.suggest_float("top_rate", 0.1, 0.4, step=0.1),
            "other_rate": trial.suggest_float("other_rate", 0.1, 0.4, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8, step=0.1),
            "reg_alpha": trial.suggest_categorical("reg_alpha", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
            "reg_lambda": trial.suggest_categorical("reg_lambda", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
            "min_child_weight": trial.suggest_categorical("min_child_weight", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 51, step=2),
            "min_gain_to_split": trial.suggest_categorical("min_gain_to_split", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
            # # for binary
            # "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
        }
        
        model = lgb.LGBMRegressor(boosting_type="goss", objective="mse", device_type="gpu",
                                   n_estimators=ntrees, learning_rate=eta,
                                   random_state=fold, verbose=-1, **tuning_params)
        cb_list = [
            lgb.early_stopping(stopping_rounds=int(ntrees * 0.2), first_metric_only=True, verbose=False),
        ]
        model.fit(train_x, train_y, categorical_feature=categoIdx,
                  eval_set=(val_x,val_y), eval_metric="rmse", callbacks=cb_list)
        output_container["ntrees"] = model.best_iteration_
    elif model_name == "XGB_GBT":
        # objective
        # regession : "reg:squarederror"
        # classification - binary : "binary:logistic"
        # classification - multicalss :"multi:softmax" (num_class=n)
        # ranking : "rank:ndcg"

        # metric
        # regession : "mae", "rmse"
        # classification - binary : "logloss", "error@t" (t=threshold), "auc"
        # classification - multicalss : "mlogloss", "merror"
        # ranking : "ndcg", "map"

        tuning_params = {
            "max_depth": trial.suggest_categorical("max_depth", [4, 5, 6, 7, 8]),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8, step=0.1),
            "reg_lambda": trial.suggest_categorical("reg_lambda", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
            "min_child_weight": trial.suggest_categorical("min_child_weight", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
            "gamma": trial.suggest_categorical("gamma", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
            # # for binary
            # "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
        }

        model = xgb.XGBRegressor(booster="gbtree", objective="reg:squarederror",
                            n_estimators=ntrees, learning_rate=eta, tree_method="gpu_hist",
                            random_state=fold, verbosity=0, **tuning_params)
        
        model.fit(train_x, train_y,
              eval_set=[(val_x, val_y)], eval_metric="rmse",
              early_stopping_rounds=int(ntrees * 0.2), verbose=False)
        output_container["ntrees"] = model.best_iteration
    elif model_name == "CAT_GBM":
        # objective
        # regession : "MAE", "RMSE", "MAPE"
        # classification - binary : "Logloss"
        # classification - multicalss :"MultiClass"
        # ranking : "PairLogit", "YetiRank"

        # metric
        # regession : "MAE", "RMSE", "R2"
        # classification - binary : "Logloss", "Accuracy", "AUC", "F1"
        # classification - multicalss : "MultiClass", "Accuracy", "TotalF1" (average=Weighted,Macro,Micro)
        # ranking : "PairLogit", "YetiRank", "NDCG", "MAP"

        tuning_params = {
            "max_depth": trial.suggest_categorical("max_depth", [4, 6, 8]),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 1e-1, 1.0, step=1e-1),
            # rsm = colsample_bylevel (not supported for GPU)
            # "rsm": trial.suggest_float("rsm", 0.5, 0.8, step=0.1)
            "random_strength": trial.suggest_categorical("random_strength", [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
            "reg_lambda": trial.suggest_categorical("reg_lambda", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
            "min_child_samples": trial.suggest_float("min_child_samples", 1, 51, step=2),
            # # for binary
            # "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", list(np.linspace(1e-3, 1.0, num=150, endpoint=False)) + list(np.linspace(1.0, 1e+2, num=50, endpoint=True))),
        }

        model = cat.CatBoostRegressor(boosting_type="Plain", loss_function="RMSE", eval_metric="RMSE", task_type="GPU",
                            n_estimators=ntrees, learning_rate=eta, bootstrap_type="Bayesian",
                            verbose=False, random_state=fold, **tuning_params)

        model.fit(train_x, train_y, cat_features=categoIdx,
                  eval_set=[(val_x, val_y)], early_stopping_rounds=int(ntrees * 0.2),
                  use_best_model=True, verbose=False)
        output_container["ntrees"] = model.best_iteration_

    elif model_name == "KNN":
        tuner_params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, max(49, int(train_x.shape[0] * 0.01)), step=2),
        }
        
        try:
            # for RAPIDS module
            model = cuml.neighbors.KNeighborsRegressor(output_type="numpy", **tuner_params)
            model.fit(train_x, train_y)
        except:
            model = KNeighborsRegressor(**tuner_params)
            model.fit(train_x, train_y)
    elif model_name == "MLP":
        max_iter = trial.suggest_int("max_iter", 5, 50, step=5)
        hidden_layer_units = trial.suggest_categorical("hidden_layer_units", [8, 16, 32, 64, 128])
        hidden_layer_depth = trial.suggest_categorical("hidden_layer_depth", [2, 3, 4, 5, 6])
        hidden_layer_sizes = tuple([hidden_layer_units] * hidden_layer_depth)

        tuner_params = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "max_iter": max_iter
        }
        
        model = MLPRegressor(activation="relu", solver="adam", learning_rate_init=5e-4, learning_rate="constant",
                             batch_size=CFG.batch_size, early_stopping=False, shuffle=True, random_state=fold,
                             **tuner_params)
        model.fit(train_x, train_y)
    else:
        print("unknown")
        return -1
    
    pred = model.predict(val_x)
    
    y_true = (-1) * np.expm1(val_y) if CFG.TF else val_y
    y_pred = (-1) * np.expm1(pred) if CFG.TF else pred
    
    output_score_dic = {
        "rmse": skl.metrics.mean_squared_error(y_true, y_pred, squared=False),
        "spearman": spearmanr(y_true, y_pred)[0],
    }
    optuna_score = output_score_dic["rmse"]
    
    if optuna_score < output_container["optuna_score"]:
        if output_container["ntrees"] is not None:
            print("number of trees :", output_container["ntrees"])
        output_container["model"] = model
        output_container["pred"] = pred
        output_container["optuna_score"] = optuna_score
        output_container["output_score_dic"] = output_score_dic

    return optuna_score

import operator
class Optuna_EarlyStoppingCallback(object):
    """Early stopping callback for Optuna."""

    def __init__(self, early_stopping_rounds: int, direction: str = "minimize") -> None:
        self.early_stopping_rounds = early_stopping_rounds

        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Do early stopping."""
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()


# In[21]:


# fold training
def do_fold_training(fold, train_idx, valid_idx):
    print("\n===== Fold", fold, "=====\n")
    
    df_train = df_full_x.iloc[train_idx].reset_index(drop=True)
    df_valid = df_full_x.iloc[valid_idx].reset_index(drop=True)
    
    scaler = MinMaxScaler()
    pca = None
    categoIdx = None
    
    scaler.fit(df_train[scaled_vars])
    df_train[scaled_vars] = scaler.transform(df_train[scaled_vars])
    df_valid[scaled_vars] = scaler.transform(df_valid[scaled_vars])
        
    output_container = {"model": None, "pred": None, "optuna_score": np.inf, "ntrees": None, "scaler": None}
    optuna_direction = "minimize"
    optuna_trials = 3 if CFG.debug else 300
    optuna_earlyStopping = Optuna_EarlyStoppingCallback(max(1, int(optuna_trials * 0.2)), direction=optuna_direction)
    # consider only boosting algorithm when calcaulating timeout
    optuna_timout = int(8 * 3600 / CFG.n_folds * model_weight[model_name])
    optuna_study = create_study(direction=optuna_direction, sampler=TPESampler())
    
    optuna_study.optimize(
        lambda trial: optuna_objective_function(
            trial, fold,
            df_train, df_full_y.iloc[train_idx], None,
            df_valid, df_full_y.iloc[valid_idx], None,
            categoIdx=categoIdx, model_name=model_name, output_container=output_container,
            ntrees=ntrees, eta=eta,
        ),
        n_jobs=1, n_trials=optuna_trials, timeout=optuna_timout, callbacks=[optuna_earlyStopping]
    )
    
    model_ouptut_dic[model_name].append(output_container["model"])
    feature_transformer_dic[model_name].append((scaler, pca))
    score_ouptut_dic[model_name].append(output_container["output_score_dic"])
    cv_pred[model_name][valid_idx] = output_container["pred"]


# # Training

# In[22]:


df_full_x = df_full
df_full_y = np.log1p((-1) * df_full_x["ddG"].astype("float32"))
df_full_x = df_full_x.drop("ddG", axis=1).astype("float32")


# In[23]:


# model_name_list = ["ElasticNet", "LGB_GOSS", "XGB_GBT", "CAT_GBM", "KNN"]
model_name_list = ["ElasticNet", "LGB_GOSS", "CAT_GBM"]
model_weight = {
    "ElasticNet": 0.15,
    "LGB_GOSS": 0.35,
    "CAT_GBM": 0.5,
}

model_ouptut_dic = {i: [] for i in model_name_list}
feature_transformer_dic = {i: [] for i in model_name_list}
score_ouptut_dic = {i: [] for i in model_name_list}
cv_pred = {i: np.zeros(len(df_full_x)) for i in model_name_list}

ntrees = 100 if CFG.debug else 5000
eta = 5e-3

startVec = skl.preprocessing.KBinsDiscretizer(CFG.n_folds, encode="ordinal").fit_transform(df_full_y.to_frame()).flatten().astype("int32")
start_time_training = time()
for model_name in model_ouptut_dic.keys():
    kfolds_spliter = skl.model_selection.StratifiedKFold(CFG.n_folds, shuffle=True, random_state=42)
    
    seed_everything()
    for fold, (train_idx, valid_idx) in enumerate(kfolds_spliter.split(df_full_x, startVec)):
        
        do_fold_training(fold, train_idx, valid_idx)
        gc.collect()
        
        if CFG.debug:
            if fold >= 1:
                break
    
end_time_training = time()
print("total training time :", end_time_training)


# In[24]:


score_table = pd.DataFrame([pd.DataFrame(v).mean(axis=0) for k, v in score_ouptut_dic.items()], index=score_ouptut_dic.keys())
display(score_table)


# In[25]:


# ensemble_weight = (1 / score_table["rmse"]) / (1 / score_table["rmse"]).sum()
# display(ensemble_weight)
ensemble_weight = pd.Series(model_weight.values(), index=model_name_list)


# In[26]:


y_true = df_full_y.values
y_pred = np.stack([v * ensemble_weight[k] for k, v in cv_pred.items()]).sum(axis=0)

y_true = (-1) * (np.expm1(y_true) if CFG.TF else y_true)
y_pred = (-1) * (np.expm1(y_pred) if CFG.TF else y_pred)


# In[27]:


fig, ax = plt.subplots(figsize=(12, 6))
graph = sns.histplot(y_true, bins=50, color="orange")
plt.title("Distribution on original target value", fontsize=15, fontweight="bold", pad=15)


# In[28]:


fig, ax = plt.subplots(figsize=(12, 6))
graph = sns.histplot(y_pred, bins=50, color="green")
plt.title("Distribution on validation target value", fontsize=15, fontweight="bold", pad=15)


# ## Visualization on normalized feature importances

# In[29]:


lgb_fi = np.stack([i.feature_importances_ for i in model_ouptut_dic["LGB_GOSS"]]).mean(axis=0)
lgb_fi /= lgb_fi.max()
cat_fi = np.stack([i.feature_importances_ for i in model_ouptut_dic["CAT_GBM"]]).mean(axis=0)
cat_fi /= cat_fi.max()

fi = (lgb_fi + cat_fi) / 2
fi = pd.Series(fi, index=df_full_x.columns).sort_values(ascending=False)
fi = fi.iloc[:20]


# In[30]:


fig, ax = plt.subplots(figsize=(10, 10))
graph = sns.barplot(x=fi.values, y=fi.index)
plt.title("Feature Importances", fontsize=15, fontweight="bold", pad=15)
plt.yticks(fontsize=12, fontweight="bold")
plt.show()


# # Inference

# In[31]:


df_test_x = pd.read_csv("/kaggle/input/nesp-create-data-esm-pdb/df_test.csv")
df_test_x = df_test_x[~(df_test_x["mt"] == "X")].reset_index(drop=True)
df_test_x = get_features(df_test_x, inference=True)


# In[32]:


df_test_x[["wt_aa_group_" + i for i in ohe_aa_group.categories_[0]]] = ohe_aa_group.transform(df_test_x[["wt_aa_group"]])
df_test_x[["mt_aa_group_" + i for i in ohe_aa_group.categories_[0]]] = ohe_aa_group.transform(df_test_x[["mt_aa_group"]])
df_test_x= df_test_x.drop(["wt_aa_group", "mt_aa_group"], axis=1)


# In[33]:


df_test_x = df_test_x.astype("float32")


# In[34]:


test_pred = {i: np.zeros(len(df_test_x)) for i in model_name_list}

seed_everything()
for model_name in model_ouptut_dic.keys():
    for fold in range(CFG.n_folds):
        df_test_x_ft = df_test_x.copy()
        df_test_x_ft[scaled_vars] = feature_transformer_dic[model_name][fold][0].transform(df_test_x_ft[scaled_vars])
                
        test_pred[model_name] += model_ouptut_dic[model_name][fold].predict(df_test_x_ft) / CFG.n_folds
        if CFG.debug:
            if fold >= 1:
                break


# In[35]:


test_pred = np.stack([v * ensemble_weight[k] for k, v in test_pred.items()]).sum(axis=0)
test_pred = ((-1) * np.expm1(test_pred)) if CFG.TF else test_pred


# In[36]:


fig, ax = plt.subplots(figsize=(12, 6))
graph = sns.histplot(test_pred, bins=50, color="grey")
plt.title("Distribution on inferenced target value", fontsize=15, fontweight="bold", pad=15)


# In[37]:


test_pred[:5]


# # Save raw output

# In[38]:


df_test = pd.read_csv(CFG.kaggle_dataset_path + "nesp-create-dataset/df_test.csv")


# In[39]:


# Q1 value for deletion by distance scaling - not std, minimum distance from value 
df_test.loc[~(df_test["mutation"].isna() | (df_test["seq_id"] == test_base_seq_id)), "target"] = test_pred
df_test.loc[(df_test["mutation"].isna()), "target"] = pd.Series(test_pred).describe()["25%"] - (np.abs(test_pred - test_pred.mean()).min() * get_scaled_dist(df_test.loc[(df_test["mutation"].isna()), "position"]))
df_test.loc[(df_test["seq_id"] == test_base_seq_id), "target"] = np.nanmax(test_pred) + (pd.Series(test_pred).std())


# In[40]:


output_dic["output"] = df_test["target"].values
pickleIO(output_dic, "./raw_output.pkl")


# # Submission

# In[41]:


submission = pd.read_csv(CFG.kaggle_dataset_path + "novozymes-enzyme-stability-prediction/sample_submission.csv")


# In[42]:


submission["tm"] = df_test["target"].values

# ranking & normalizing
submission["tm"] = rankdata(submission["tm"]) if output_dic["type"] == "maximize" else rankdata(-1 * submission["tm"])
submission["tm"] /= len(submission["tm"])

submission.to_csv("./submission.csv", index=False)


# In[43]:


submission.head()


# In[ ]:




