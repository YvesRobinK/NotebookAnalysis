#!/usr/bin/env python
# coding: utf-8

# # Tutorial: Preprocessing Deep Learning Input from RNA Strings
# 
# # 1. Why bother with RNA properties prediction?
# 
# RNA Technology is now a new promising hope in Health Science. In addtion to the global-scaled use of COVID19 mRNA-vaccines from BioNTech/Pfizer and Moderna, RNA drugs also provides [a new hint on how to cure important disease such as cancers](https://jhoonline.biomedcentral.com/articles/10.1186/s13045-020-00951-w).
# 
# Serious learners about RNA should take this [amazing course](https://www.edx.org/course/introduction-to-biology-the-secret-of-life-3), or just look at this [cute video](https://youtu.be/JQByjprj_mA) (and the others in this channel) if you don't have much time. 
# 
# Leading labs including [BioNTech](https://biontech.de/how-we-translate/mrna-therapeutics), [Moderna](https://www.modernatx.com/mrna-technology/mrna-platform-enabling-drug-discovery-development) and [others](https://www.accenttx.com/our-scientific-focus/rmps-and-cancer/) state that RNA-related technology has an important advantage over DNA-technology such that it will not provide a **permanent effects** on a patient, i.e. if we modify our genes in DNA, the modification will permanently embed in the DNA, including any modification mistakes.
# 
# In contrast, injected RNA will do its job i.e. making protiens, and then completely disappeared, without leaving any permanent effects. Proteins made by RNA could prevent some
# important diseases such as in the case of mRNA vaccines. Intuitively, Moderna compares DNA as a life "hardware" while RNA as a "software" which can be safely injected to human body.
# 
# To design RNA drug or vaccine, some properties of RNA must be known. In the past, these properties have to be tested in bio labs which can be time and resource consuming.
# Therefore, it is difficult to conduct experiments in a large scale e.g. millions of RNAs. 
# 
# Deep learning technology then can help predicting specific properties of millions of RNAs in a flash. Even though, its predictions are not entirely 100% accurate, it can serve as a first filter
# for biologists to select most promising samples to further make a small-scale intensive case in the next phase.
# 
# 
# ## 1.1 About this Tutorial
# This interactive tutorial notebook demonstrates how to preprocess RNA data into an input ready to feed into Deep Learning models.
# The processed data will be similar to the ones provided by [Kaggle's mRNA OpenVaccine competition](https://www.kaggle.com/c/stanford-covid-vaccine/overview).
# So that once you finish processing your own RNA data, you can use any of the [great public codes](https://www.kaggle.com/c/stanford-covid-vaccine/code?competitionId=22111&sortBy=voteCount) 
# to make a prediction. We will provide a simplified SOTA top-solution, which will be easily reproducible and adapt to your own work in the near future, so stay tune!!
# 
# Figure 1 shows the overall big picture of this notebook. 
# 
# Here, we assume that the user has a list of "RNA strings" as input where an RNA string is a string consisting of only 4 characters 'ACGU'.
# The true RNA structure of a given string is unknown as can be illustrated in Figure 1 (bottom-left).
# The user wants to extract this unknown structure and represents it in the format which is ready-to-use on a Deep Learning model, e.g. a neural network.
# 
# 
# In this tutorial, this unknown structure will be estimated using various python and bioinformatics libraries. 
# To feed into a deep learning model, the estimated structure will be best represented by "Graph data" $G = (X, E)$ where
# each node in the graph corresponding to each character in a string.
# 
# Note that this **Graph Representation** generalizes **Sequential Data Representation** so while you can still use well-known
# sequential layers/models such as '1D Convolution', 'Transformers' or 'Recurrent Neural Networks', you can also use a new frontier
# like 'Graph Neural Networks' which is actually shown effective in this RNA data.
# 
# $X$ is a sequence of **node feature vectors**,
# and $E$ is an adjacency matrix specifying linkages for each pair of nodes whether the pair is connected in the true structure.
# 
# In general, each element in an adjacency matrix can be a real number (e.g. specifying a "probability" whether a pair is connected).
# Moreover, there can be more than one adjacency matrices (e.g. differently estimated from various libraries). 
# 
# In this general setting of having a list of several adjacency matrices, we will shortly call $E$ as a list of **edge features**.
# 
# **Acknowledgement** this tutorial is built on amazing works of many kagglers especially [@its7171](https://www.kaggle.com/its7171/how-to-generate-augmentation-data) and [@group16](https://www.kaggle.com/group16/generating-bpps-with-arnie). However, the URLs and packages used in their notebooks are already outdated. This tutorial fix the problems, ensure complete reproducibility using static Kaggle's dataset, as well as providing full detailed explanation with minimum requirements on the input data.
# 

# <!-- ![Data preprocessing from an RNA string](https://i.ibb.co/8mkQ1vh/RNA-data-preprocessing.png) -->
# <img src=https://i.ibb.co/8mkQ1vh/RNA-data-preprocessing.png width="1024">
# Figure 1. Data preprocessing from an RNA string

# 
# ## 1.1 Overview of this tutorial
# 
# We will assume that users have already list of RNA strings to extract the graph features explained above. Here, we will use Kaggle's OpenVaccine dataset containing 6,000 RNA strings. Note that in OpenVaccine, the dataset already contained some of node and edge features. We will not use them in this tutorial but will show how to extract features even beyond ones in OpenVaccine.
# 
# **Node Features.** Each character in an RNA string is represented by its character "A, C, G or U" and its loop-type as shown in Figure 1 "Exterior, Bulge, Stem, Multibranch or Hairpin". We will represent each character by 1-hot vector $R^4$ and will represent the loop-type vector in $R^6$ since there are 6 possible loop types. 
# 
# Note that RNA structure can be very complex and have a wide variety as shown in Figure 2.
# 
# ![](https://i.ibb.co/swD1DBF/rna-folding-examples.png)
# Figure 2. RNA can have a variety of structure due to folding.
# 
# Therefore, due to complexity of the RNA folding structure, exact loop-type property is unknown (already mentioned in Figure 1), the `CapR` library we used can estimate loop-type probability. Therefore, it will return a probability vector in $R^6$.
# 
# Therefore, by concatenating the two vectors, our node features will lies in $R^{10}$. Note that it's certainly possible that to also employ other libraries to extract more node features so that our node-feature vector lies in a higher dimensional than $R^{10}$.
# 
# 
# **Edge Features.** Similar to loop-type features, exact link between a pair of nodes is unknown. Therefore, **"base-pair probabilities matrix (bpps)"** is extracted instead. Suppose that an RNA string is length of $N$, bpps then is in $R^{N \times N}$.
# 
# This tutorial show how to extract edge features using bpps libraries such as `vienna`, `contrafold` and `rnastructure`. From bpps of these libraries, we can also get "most probable" structure, which can also be further extracted as either node or edge feature.
# 
# After the node and edge features are extracted, they can be input directly to deep learning models as shown in Figure 3. See The [RNA Deep Learning notebook tutorial](https://www.kaggle.com/ratthachat/tutorial-pretrained-sota-deeprna-model-made-easy) for how to apply pretrained SOTA RNA model into general RNA prediction problem.
# 
# <img src=https://i.ibb.co/r3WB24R/RNA-feature-and-model.png width="750">
# Figure 3. Extracted feature can be used directly to models such as LSTM, GRU, Transformers and Graph Neural Networks (GNNs)
# 

# ## 1.2 Install necessary external libraries
# 
# As mention above, we will install `CapR`, `vienna`, `contrafold` and `rnastructure` packages to extract node and edge features.
# For convenient, We will also install the package called `arnie` which provides a unified interface on all bpps libraries mentioned above.

# 
# ### Important Note on Kaggle's Docker Version
# This kaggle notebook's environment is frozen, not to update to the latest docker version.
# The libraries used in this notebook are also stored as Kaggle's dataset and work perfectly in the current frozen environment.

# In[1]:


WORKING_DIR = '/kaggle/working/'
TOOLKITS_DIR = '/kaggle/input/rna-data-preprocessing-toolkits'
get_ipython().system('ls {TOOLKITS_DIR}')
get_ipython().system('/opt/conda/bin/python3.7 -m pip install --upgrade pip')


# The benefit of using kaggle notebook over alternative notebook such as Colab is the availability of datasets for everybody.
# Here, all external libraries are stored in a Kaggle dataset on a path specified by`TOOLKITS_DIR`. Firstly, we install `vienna` package (ref: https://www.tbi.univie.ac.at/RNA/)
# 

# In[2]:


get_ipython().system('apt-get install {TOOLKITS_DIR}/viennarna_2.4.15-1_amd64.deb -y')


# The `CapR` package is already compiled and executable. It is retrieved from https://github.com/fukunagatsu/CapR

# In[3]:


get_ipython().system('cp -rf {TOOLKITS_DIR}/CapR/CapR/ .')
get_ipython().system('chmod 755 ./CapR/CapR')


# Next, the `arnie` package:   https://github.com/DasLab/arnie

# In[4]:


get_ipython().system('cp -rf {TOOLKITS_DIR}/arnie/arnie .')
get_ipython().system('cp -rf {TOOLKITS_DIR}/draw_rna_pkg/draw_rna_pkg .')
get_ipython().system('cd draw_rna_pkg && python setup.py install')


# Next is Stanford's `contrafold` package (ref: http://contra.stanford.edu/contrafold/). Note that it requires `g++4.8` which is not available in the latest Kaggle environment.

# In[5]:


get_ipython().system('cp -rf {TOOLKITS_DIR}/contrafold-se/contrafold-se .')
get_ipython().system('apt-get install -y g++-4.8')
get_ipython().system('sed -i.bak "1 s/^.*$/CXX = g++-4.8/" contrafold-se/src/Makefile')
get_ipython().system('cd contrafold-se/src; make')


# Lastly, the`rnastructure` package from https://rna.urmc.rochester.edu/RNAstructure.html

# In[6]:


get_ipython().system('tar zxvf {TOOLKITS_DIR}/RNAstructureLinuxTextInterfaces64bit.tgz --directory .')


# # 2. Feature Extraction
# 
# In order to use `arnie` as an interface to all bpps libraries, we need to configure the paths to `arnie.conf` as shown below. After that, we import all necessary packages that will be used in this tutorial.

# In[7]:


import os
import glob
import sys

get_ipython().system('echo "vienna_2: /usr/bin" > arnie.conf')
get_ipython().system('echo "contrafold_2: /kaggle/working/contrafold-se/src" >> arnie.conf')
get_ipython().system('echo "rnastructure: /kaggle/working/RNAstructure/exe" >> arnie.conf')
get_ipython().system('echo "TMP: /kaggle/working/tmp" >> arnie.conf')
get_ipython().system('mkdir -p /kaggle/working/tmp')
os.environ["ARNIEFILE"] = f"/kaggle/working/arnie.conf"
os.environ["DATAPATH"] = f"/kaggle/working/RNAstructure/data_tables"
sys.path.append('/kaggle/working/draw_rna_pkg/')
sys.path.append('/kaggle/working/draw_rna_pkg/ipynb/')

get_ipython().system('cat arnie.conf')


# In[8]:


import numpy as np
import pandas as pd
from tqdm.notebook import tqdm as tqdm
import subprocess
from multiprocessing import Pool

# from arnie.pfunc import pfunc
from arnie.mea.mea import MEA
# from arnie.free_energy import free_energy
from arnie.bpps import bpps
# from arnie.mfe import mfe
import arnie.utils as utils

from IPython.display import display


# ## 2.1 Getting Edge Features
# 
# Equpping with packages like `vienna`, `contrafold` and `rnastructure`,`arnie` can be used to extract both bpps matrix and most-probable structure.
# Here, we use RNA strings from the OpenVaccine dataset without using any already-extracted-features. 
# 
# The readers can easily substitute their own RNA strings to the ones in OpenVaccine by creating a pandas `DataFrame` with two columns of `['id','sequence']`. Note that `id` for each string can be arbitrary -- just make sure to be unique for each string.
# 

# In[9]:


debug = True # To extract features of all 6,000 strings, set debug=False


# In[10]:


# extract features from all packages and save in corresponding dir
# you can also select only one package
packages = ['contrafold_2', 'vienna_2', 'rnastructure']

train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)

if debug:
    train = train[:10]
    test = test[:10]
target_df = train.append(test)[['id','sequence']]

print(target_df.shape)
target_df.head(3)


# As Kaggle notebook provides 4-cores CPU, we will take advantage of the multiprocessing when extracting edge/node features. Here, we creat directory for each selected package separately.

# In[11]:


get_ipython().system('grep processor /proc/cpuinfo | wc -l')
MAX_THREADS = 4
BPPS_DIR = WORKING_DIR+'bpps/'
os.mkdir(BPPS_DIR) # we will store the extracted bpps here


# In[12]:


for package in packages:
    os.mkdir(BPPS_DIR+package)
get_ipython().system('ls {BPPS_DIR}')


# The parallel computation can be done easily by applying `imap` from `multiprocessing` package to the specified function, which in our case is the `extract_edge_features` function. `arnie` provides `bpps` and `MEA` function to extract the bpps matrix and most-probable structure, respectively.
# The extracted bpps matrix will be saved to Numpy's `.npy` format.

# In[13]:


def extract_edge_features(arg):
    sequence, seq_id, package = arg
    
    bp_matrix = bpps(sequence, package=package)
        
    mea_mdl = MEA(bp_matrix)
    np.save(BPPS_DIR+package+f'/{seq_id}.npy', bp_matrix)
    
    return seq_id, sequence, mea_mdl.structure, mea_mdl.score_expected()[2], package


# In[14]:


# prepare input for extract_edge_features
arg_list = []
for i, (seq, seq_id) in enumerate(target_df[['sequence','id']].values):
    for pack in packages:
        arg_list.append([seq, seq_id, pack])

# apply multiprocessing to extract_edge_features
p = Pool(processes=MAX_THREADS)
results = []
for ret in tqdm(p.imap(extract_edge_features, arg_list),total=len(arg_list)):
    results.append(ret)
    
structure_df = pd.DataFrame(results, columns=['id', 'sequence', 'structure', 'score', 'package'])


# In[15]:


structure_df.head(10)


#  While the most-probable structures of all strings for each package will be stored separately in a `DataFrame`, namely, `structure_df_list` for easy usage. Each element of `structure_df_list` corresponds to one package as shown below.

# In[16]:


structure_df_list = []
for package in packages:
    structure_df_list.append(structure_df[structure_df['package']==package])
    print(structure_df_list[-1].shape)


# Now, we can investigate the resulted most-probable structure for each string here! The structure is encoded in [a **Dot-Bracket** format](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/rna_structure_notations.html). The Dot-Bracket notation as introduced in the early times of the ViennaRNA Package denotes base pairs by matching pairs of pair () and unpaired nucleotides by dots. Note that most probable structures of the same id but different packages may not be the same.

# In[17]:


for i, package in enumerate(packages):
    print(structure_df_list[i].head(3))


# If the above print() command is not so intutive, we can also visualize the most-probable structure easily  🤩, even though we won't use the visualization in our feature extraction procedure.

# In[18]:


from draw import draw_struct
sequence, structure = structure_df.iloc[0][['sequence', 'structure']]
print(sequence)
print(structure)
draw_struct(sequence, structure,  cmap='plasma')


# In[19]:


# Finally, let us investigate the saved bpps files
for i, package in enumerate(packages):
    print(package)
    get_ipython().system('ls {BPPS_DIR}{package} | wc')
    get_ipython().system('ls {BPPS_DIR}{package} | head')


# ## 2.2 Getting Node Features
# 
# Now we will extract one-hot ACGU vector as well as the probabilisttic loop-type vector using `CapR` package. The`CapR` is a shell script command which could be run one string at a time with specified input and output files, and we automate the run in the `run_CapR` function and store the probabilistic output with a pandas' dataframe.
# 

# In[20]:


NODE_DIR = WORKING_DIR+'node_features/'
os.mkdir(NODE_DIR)


# In[21]:


def run_CapR(rna_id, rna_string, max_seq_len=1024):
    in_file = '%s.fa' % rna_id
    out_file = '%s.out' % rna_id
             
    fp = open(in_file, "w")
    fp.write('>%s\n' % rna_id)
    fp.write(rna_string)
    fp.close()
    
    subprocess.run('/kaggle/working/CapR/CapR %s %s %d' % (in_file, out_file, max_seq_len),
                   shell=True,capture_output=False)
             
    df = pd.read_csv(out_file, skiprows=1,
                     header=None, delim_whitespace=True,
            )
    df2 = df.T[1:]
    df2.columns = df.T.iloc[0].values
    
    return df2


# We can test run the function just to have fun :)

# In[22]:


test_string = 'AGGGUUUUCCCC'
df = run_CapR('test_id_1234',test_string )
df.head(len(test_string ))


# Next, `extract_rna_node_features` will combine the above `run_CapR` with one-hot 'AGUC' vector. We also can have the most-probable structure in DotBracket notation as one-hot vector as an additional option to node features. This `extract_rna_node_features` will save a `.csv` file for each string in the `NODE_DIR` directory

# In[23]:


def extract_rna_node_features(rna_id, rna_string, mfe_structure=None):
    # looptype features
    df = run_CapR(rna_id, rna_string)
    
    # onehot AGCU features
    def onehot_np(length, i):
        vect = [0] * length
        vect[i] = 1
        return np.array(vect)
    
    base_vocab = 'ACGU'
    token2onehot = {x:onehot_np(len(base_vocab), i) for i, x in enumerate(base_vocab)}
    out = list(map(lambda y : token2onehot[y], rna_string))
    df[list(base_vocab)] = np.array(out)
    
    # onehot mfe_structure features (optional)
    if mfe_structure is not None:
        structure_vocab = '(.)'
        token2onehot = {x:onehot_np(len(structure_vocab), i) for i, x in enumerate(structure_vocab)}
        out = list(map(lambda y : token2onehot[y], mfe_structure))
        df[list(structure_vocab)] = np.array(out)
    
    return df


# Run for another fun to see the complete sequence of node features for each character 🚀 💥 🚀!! 

# In[24]:


rna_id = 'testmol'
rna_string = 'AGGGGCCUUUUAAGGAAUUUC'
struct_string = '(.................)()'

df = extract_rna_node_features(rna_id, rna_string, struct_string)
print(df.shape)
df.head(10)


# Now we make a simple function to allow parallelization and save the resulted node features into an individual dataframe. Note that here, use only 1 most-probable structure from the first package to make a feature, but the reader can extend to include structures from all packages easily

# In[25]:


def extract_rna_node_features_and_save(arg):
    rna_id = arg[0]
    rna_string = arg[1]
    struct_string = arg[2]

    df = extract_rna_node_features(rna_id, rna_string, struct_string)
    df.to_csv(NODE_DIR+'%s_node_features.csv' % rna_id, index=False)
    
    return 0


# In[26]:


arg_list = []

for i, (seq_id, sequence, structure) in enumerate(structure_df_list[0][['id','sequence','structure']].values):
#     for pack in packages:
    arg_list.append([seq_id, sequence, structure])
            
p = Pool(processes=MAX_THREADS)

for ret in tqdm(p.imap(extract_rna_node_features_and_save, arg_list),total=len(arg_list)):
    pass # we save the output to files


# Node-feature extraction is done ☄️ 💥 🔥 !!

# In[27]:


# investigate the save bpps files
get_ipython().system('ls {NODE_DIR} | wc')
get_ipython().system('ls {NODE_DIR} | head')


# In[28]:


rna_id = "id_0049f53ba"
file = rna_id + "_node_features.csv" # 'id_00073f8be_node_features.csv'
df = pd.read_csv(NODE_DIR+file)

print(df.shape)
df.head()

df2 = train[train.id==rna_id]
print(df2.shape)


# # (Optional) 3. Advanced Node-Feature Extraction
# 
# We are done, but what else? We can actually go beyond to explore some feature engineering like SOTA [models](https://www.kaggle.com/group16/covid-19-mrna-4th-place-solution), which also was inspried by [this](https://www.kaggle.com/its7171/gru-lstm-with-feature-engineering-and-augmentation) and this [notebooks](https://www.kaggle.com/its7171/dangerous-features). Yes, SOTA models were those who got gold medals in the very-intense OpenVaccine competition!
# 
# Advanced node features that we are going to extract further in this section are
# 
# * The ratio of Dot-Bracket prediction `'(', '.',` and `')'` predicted by 3 packages (`vienna`, `contrafold` and `rnastructure`) -- instead of depending only on one package like we did in previous section
# * Some of bpps for each base
# * Maximum probability with respect to bpps for each base
# * Number of non-zeros on bpps for each base
# * Codon's (triplet) position for each base which is simply 012012012012... for every RNA string.
# * (Optional: Problem-specific, but conceptually applicatble) Sample-weight which is standard in model fitting. In OpenVaccine problem, we could use "signal-to-noise" ratio as sample weight whereas in other contexts, you can consult expert for appropriate specification.
# * (Optional: Problem-specific) Error-bar of each labels (problem specific and is provided in OpenVaccine). We could use this feature for advanced "sample_weight" where the weights are different for each base/label (whereas the standard sample_weight, one weight is used for all bases/labels).  This feature set may not obtainable from other problems, so we can switch on or off
# 
# The last-two optional features can be further boosted the prediction performance of a deep learning model. For example, [RNA Deep Learning notebook tutorial](https://www.kaggle.com/ratthachat/tutorial-pretrained-sota-deeprna-model-made-easy) will employ the error-bar features.

# In[29]:


# we will save these advanced node features separated from basic node features, so that users can choose either of them
ADV_NODE_DIR = '/kaggle/working/advanced_node_features/'
os.mkdir(ADV_NODE_DIR) 


# In[30]:


rna_id_list = structure_df.id.unique()
structure_df.head(2)


# In[31]:


# Choose whether to extract "sample_weight" and/or "error_bar" features here
SAMPLE_WEIGHT_FEAT = False # you can switch on/off sample-weight features by setting this variable
ERROR_BAR_FEAT = False # you can switch on/off error-bar features by setting this variable
err_cols = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_Mg_50C', 'deg_error_pH10', 'deg_error_50C']

def pandas_list_to_array(df):
    """
    Simple toolkit for extracting error_bar features
    Input: dataframe of shape (num_examples, num_cols), each cell contains a list of length "seq_len"
    Return: np.array of shape (num_examples, seq_len, num_cols)
    """
    
    return np.transpose(
        np.array(df.values.tolist()),
        (0, 2, 1)
    )


# In[32]:


def str_to_charflag(string, char):
    bin_str = [c==char for c in string]
    return np.array(bin_str).astype(np.int32)

# this function assumes the existent of structure_df built in last section
def make_advanced_features(rna_id,debug=False):
    # first, let us copy the basic features    
    df = pd.read_csv(NODE_DIR+rna_id+'_node_features.csv')
    
    ####
    # Ratio of the open-pair, close-pair and non-pair
    ####
    struct_string_list = structure_df.query('id == @rna_id')['structure'].values
    
    open_pair_list = np.array([str_to_charflag(s, '(') for s in struct_string_list])
    open_pair_ratio = np.mean(open_pair_list, axis=0)
    df['(-ratio'] = open_pair_ratio
    
    close_pair_list = np.array([str_to_charflag(s, ')') for s in struct_string_list])
    close_pair_ratio = np.mean(close_pair_list, axis=0)
    df[')-ratio'] = close_pair_ratio
    
    non_pair_list = np.array([str_to_charflag(s, '.') for s in struct_string_list])
    non_pair_ratio = np.mean(non_pair_list, axis=0)
    df['.-ratio'] = non_pair_ratio
    
    ####
    # Codon positioning feature (trinary)
    ####
    rna_len = len(struct_string_list[0])
    codon_pos0 = (np.arange(rna_len) % 3 == 0).astype(np.int32)
    codon_pos1 = (np.arange(rna_len) % 3 == 1).astype(np.int32)
    codon_pos2 = (np.arange(rna_len) % 3 == 2).astype(np.int32)
    df['codon-pos0'] = codon_pos0
    df['codon-pos1'] = codon_pos1
    df['codon-pos2'] = codon_pos2
    
    ####
    # bpps feat. engineering
    ####
    bpps_list = []
    for i, package in enumerate(packages):
        bpps_list.append(np.load(BPPS_DIR+package+f'/{rna_id}.npy'))
    bpps = np.mean(bpps_list, axis=0)
    
    df['bpps-max'] = np.max(bpps,axis=0)
    df['bpps-sum'] = np.sum(bpps,axis=0)
    df['bpps-num-ratio'] = np.sum(bpps==0, axis=0)/bpps.shape[0]
    
    ####
    # error-bar
    ####
    if ERROR_BAR_FEAT:
        err_template = np.ones([rna_len, 5]) # maybe longer than recorded err_bar
        rna_df = train[train.id==rna_id]
        if rna_df.shape[0] > 0: # training data
            err_bar = np.abs(pandas_list_to_array(rna_df[err_cols]))
            
            # USE this version if you want to also use SN_filter and use err_bar as features
#             err_bar = np.clip(err_bar,0.5,10) # set min-max not too extreme
            
            # use this version if you will not SN_filter and use err_bar in loss directly
            err_bar = np.where(err_bar<0.5, 0.5, err_bar) # prevent extreme minimum only
            
            err_bar = err_bar.squeeze() # (68, 5)
            len_err_bar = len(err_bar)
            err_template[:len_err_bar, :] = err_bar
            err_template[len_err_bar:, :] *= 20 # -> according to host's info, err_bar at the end are bad
        else: # test data -> no record of err_bar
            pass # just use err_template (default err_bar = 1) 
        
        for i, err_name in enumerate(err_cols):
            df[err_name] = err_template[:,i]
    
    if SAMPLE_WEIGHT_FEAT:
        rna_df = train[train.id==rna_id]
        if rna_df.shape[0] > 0: # training data
            df["sample_weight"] = float(train[train.id==rna_id].signal_to_noise.values)
        else: # test data, use weight = 5.0 for pseudo-training (mean SNR in training = 4.53)
            df["sample_weight"] = 5.0
    
    df.to_csv(ADV_NODE_DIR+'%s_node_features.csv' % rna_id, index=False)
    
    if debug:
        return df
    return 0

df = make_advanced_features(rna_id_list[19], debug=True)
print(df.shape)
display(df.head(25))
display(df.tail(5))


# In[33]:


p = Pool(processes=MAX_THREADS)

for ret in tqdm(p.imap(make_advanced_features, rna_id_list),total=len(rna_id_list)):
    pass # we save the output to files


# In[34]:


# investigate the save bpps files
get_ipython().system('ls {ADV_NODE_DIR} | wc')
get_ipython().system('ls {ADV_NODE_DIR} | head')


# # Epilogue: Clean up before finish!!
# To be compact, we will zip all the extract features and remove all the packages before leaving.
# By this clean up, it will easy to export into another Kaggle's dataset to be used in another Deep-Modeling notebook
# 
# This concludes the tutorial; hope it will be useful 🥰 😘!!

# In[35]:


structure_df.to_csv('most_probable_structure.csv',index=False)
get_ipython().system('zip -r bpps.zip {BPPS_DIR}')
get_ipython().system('zip -r node_features.zip {NODE_DIR}')
get_ipython().system('zip -r advanced_node_features.zip {ADV_NODE_DIR}')


# In[36]:


get_ipython().system('rm -rf CapR/')
get_ipython().system('rm -r tmp/')
get_ipython().system('rm -r RNAstructure/')
get_ipython().system('rm -r arnie/')
get_ipython().system('rm -r contrafold-se/')
get_ipython().system('rm -r draw_rna_pkg/')
get_ipython().system('rm -rf {BPPS_DIR}')
get_ipython().system('rm -rf {NODE_DIR}')
get_ipython().system('rm -rf {ADV_NODE_DIR}')
get_ipython().system('rm arnie.conf')


# In[37]:


# delete temp files relating CapR
for f in glob.glob("*.fa"):
    os.remove(f)
for f in glob.glob("*.out"):
    os.remove(f)

get_ipython().system('ls -h')


# In[ ]:




