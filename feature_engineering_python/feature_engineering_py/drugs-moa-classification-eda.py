#!/usr/bin/env python
# coding: utf-8

# # Mechanisms of Action (MoA) Prediction: EDA
# ***
# 
# 
# ### BLOG POST: [DRUG DISCOVERY WITH NEURAL NETWORKS](https://medium.com/swlh/drug-discovery-with-neural-networks-a6a68c76bb53?sk=dc332f724905461ac5a9b5060c62141d)
# 
# 
# 
# ![image1](https://miro.medium.com/max/700/1*AYJLRDxyO6MWGVSaP6Fm2w.png)<div align="center">Image source: Blog Post</div>
# 
# In this competition, we are suposed to develop algorithms and train models **to determine the mechanism of action of a new drug based on the gene expression and cell viability information.** In this EDA, we will try to find patterns in the data, interactions between the targets in both scored and nonscored datasets and the relationship between targets and their target genes.
# 
# 
# 
# <div class="list-group" id="list-tab" role="tablist">
#   <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">PROJECT CONTENT</h3>
#     
# > ####  1- FEATURES OVERVIEW
# > ####  2- CELL VIABILITY FEATURES
# > ####  3. GENE EXPRESSION FEATURES
# > ####  4. TARGETS *(MoA)*
#     >> ##### 4.1 Scored targets
#     >> ##### 4.2 Non-Scored targets
#     >> ##### 4.3 Drug_ID
# 
# > ####  5. TEST FEATURES

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
############################################
### DATASETS:

a= pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
b= pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
c=pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')
d=pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
e=pd.read_csv('../input/lish-moa/train_drug.csv')
f=a.merge(e, how='left', on='sig_id')

merged=pd.concat([a,b])

#Datasets for treated and control experiments
treated= a[a['cp_type']=='trt_cp']
control= a[a['cp_type']=='ctl_vehicle']

#Datasets for treated and control: TEST SET
treated_t= b[b['cp_type']=='trt_cp']
control_t= b[b['cp_type']=='ctl_vehicle']

#Treatment time datasets
cp24= a[a['cp_time']== 24]
cp48= a[a['cp_time']== 48]
cp72= a[a['cp_time']== 72]

#Merge scored and nonscored labels
all_drugs= pd.merge(d, c, on='sig_id', how='inner')

#Treated drugs without control
treated_list = treated['sig_id'].to_list()
drugs_tr= d[d['sig_id'].isin(treated_list)]

#Select the columns c-
c_cols3 = [col for col in b.columns if 'c-' in col]
#Filter the TEST set
cells3=treated[c_cols3]

#Treated drugs:
nonscored= c[c['sig_id'].isin(treated_list)]
scored= d[d['sig_id'].isin(treated_list)]

#adt= All Drugs Treated
adt= all_drugs[all_drugs['sig_id'].isin(treated_list)]

#Select the columns c-
c_cols = [col for col in a.columns if 'c-' in col]
#Filter the columns c-
cells=treated[c_cols]

#Select the columns g-
g_cols = [col for col in a.columns if 'g-' in col]
#Filter the columns g-
genes=treated[g_cols]



#####################################################
#### HELPER FUNCTIONS

def plotd(f1):
    plt.style.use('seaborn')
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(15,5))
    #1 rows 2 cols
    #first row, first col
    ax1 = plt.subplot2grid((1,2),(0,0))
    plt.hist(control[f1], bins=4, color='mediumpurple',alpha=0.5)
    plt.title(f'control: {f1}',weight='bold', fontsize=18)
    #first row sec col
    ax1 = plt.subplot2grid((1,2),(0,1))
    plt.hist(treated[f1], bins=4, color='darkcyan',alpha=0.5)
    plt.title(f'Treated with drugs: {f1}',weight='bold', fontsize=18)
    plt.show()
    
def plott(f1):
    plt.style.use('seaborn')
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(15,5))
    #1 rows 2 cols
    #first row, first col
    ax1 = plt.subplot2grid((1,3),(0,0))
    plt.hist(cp24[f1], bins=3, color='deepskyblue',alpha=0.5)
    plt.title(f'Treatment duration 24h: {f1}',weight='bold', fontsize=14)
    #first row sec col
    ax1 = plt.subplot2grid((1,3),(0,1))
    plt.hist(cp48[f1], bins=3, color='lightgreen',alpha=0.5)
    plt.title(f'Treatment duration 48h: {f1}',weight='bold', fontsize=14)
    #first row 3rd column
    ax1 = plt.subplot2grid((1,3),(0,2))
    plt.hist(cp72[f1], bins=3, color='gold',alpha=0.5)
    plt.title(f'Treatment duration 72h: {f1}',weight='bold', fontsize=14)
    plt.show()

def plotf(f1, f2, f3, f4):
    plt.style.use('seaborn')
    sns.set_style('whitegrid')

    fig= plt.figure(figsize=(15,10))
    #2 rows 2 cols
    #first row, first col
    ax1 = plt.subplot2grid((2,2),(0,0))
    sns.distplot(a[f1], color='crimson')
    plt.title(f1,weight='bold', fontsize=18)
    plt.yticks(weight='bold')
    plt.xticks(weight='bold')
    #first row sec col
    ax1 = plt.subplot2grid((2,2), (0, 1))
    sns.distplot(a[f2], color='gainsboro')
    plt.title(f2,weight='bold', fontsize=18)
    plt.yticks(weight='bold')
    plt.xticks(weight='bold')
    #Second row first column
    ax1 = plt.subplot2grid((2,2), (1, 0))
    sns.distplot(a[f3], color='deepskyblue')
    plt.title(f3,weight='bold', fontsize=18)
    plt.yticks(weight='bold')
    plt.xticks(weight='bold')
    #second row second column
    ax1 = plt.subplot2grid((2,2), (1, 1))
    sns.distplot(a[f4], color='black')
    plt.title(f4,weight='bold', fontsize=18)
    plt.yticks(weight='bold')
    plt.xticks(weight='bold')

    return plt.show()

def ploth(data, w=15, h=9):
    plt.figure(figsize=(w,h))
    sns.heatmap(data.corr(), cmap='hot')
    plt.title('Correlation between targets', fontsize=25, weight='bold')
    return plt.show()

# corrs function: Show dataframe of high correlation between features
def corrs(data, col1='Gene 1', col2='Gene 2',rows=5,thresh=0.8, pos=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53]):
        #Correlation between genes
        corre= data.corr()
         #Unstack the dataframe
        s = corre.unstack()
        so = s.sort_values(kind="quicksort", ascending=False)
        #Create new dataframe
        so2= pd.DataFrame(so).reset_index()
        so2= so2.rename(columns={0: 'correlation', 'level_0':col1, 'level_1': col2})
        #Filter out the coef 1 correlation between the same drugs
        so2= so2[so2['correlation'] != 1]
        #Drop pair duplicates
        so2= so2.reset_index()
        pos = pos
        so3= so2.drop(so2.index[pos])
        so3= so3.drop('index', axis=1)
        #Show the first 10 high correlations
        cm = sns.light_palette("Red", as_cmap=True)
        s = so3.head(rows).style.background_gradient(cmap=cm)
        print(f"{len(so2[so2['correlation']>thresh])/2} {col1} pairs have +{thresh} correlation.")
        return s

def plotgene(data):
    sns.set_style('whitegrid')    
    data.plot.bar(color=sns.color_palette('Reds',885), edgecolor='black')
    set_size(13,5)
    #plt.xticks(rotation=90)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.ylabel('Gene expression values', weight='bold')
    plt.title('Mean gene expression of the 772 genes', fontsize=15)
    return plt.show()

def mean(row):
    return row.mean()

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    
def unidrug(data):
    #Filter out just the treated samples
    scored= data[data['sig_id'].isin(treated_list)]

    #Count unique values per column
    cols = data.columns.to_list() # specify the columns whose unique values you want here
    uniques = {col: data[col].nunique() for col in cols}
    uniques=pd.DataFrame(uniques, index=[0]).T
    uniques=uniques.rename(columns={0:'count'})
    uniques= uniques.drop('sig_id', axis=0)
    return uniques

def avgdrug(data):
    
    uniques=unidrug(data)

     #Calculate the mean values
    scored= data[data['sig_id'].isin(treated_list)]
    average=scored.mean()
    average=pd.DataFrame(average)
    average=average.rename(columns={ 0: 'mean'})
    average['percentage']= average['mean']*100
    
    return average

def avgfiltered(data):
    
    average= avgdrug(data)
    #Filter just the drugs with mean >0.01
    average_filtered= average[average['mean'] > 0.01]
    average_filtered= average_filtered.reset_index()
    average_filtered= average_filtered.rename(columns={'index': 'drug'})
    return average_filtered

def plotc(data, column, width=10, height=6, color=('silver', 'gold','lightgreen','skyblue','lightpink'), edgecolor='black'):
    
        fig, ax = plt.subplots(figsize=(width,height))
        title_cnt=data[column].value_counts()[:15].sort_values(ascending=True).reset_index()
        mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color=sns.color_palette('Reds',len(title_cnt)))

        tightout= 0.008*max(title_cnt[column])
        ax.set_title(f'Count of {column}', fontsize=15, weight='bold' )
        ax.set_ylabel(f"{column}", weight='bold', fontsize=12)
        ax.set_xlabel('Count', weight='bold')
        if len(data[column].unique()) < 17:
            plt.xticks(rotation=65)
        else:
            plt.xticks(rotation=90)
        for i in ax.patches:
            ax.text(i.get_width()+ tightout, i.get_y()+0.1, str(round((i.get_width()), 2)),
             fontsize=10, fontweight='bold', color='grey')
        return
    
def plot_drugid(drug_id):
    g=d[f['drug_id']==drug_id]
    average_filtered2=avgfiltered(g)

    plt.figure(figsize=(5,2))
    average_filtered2.sort_values('percentage', inplace=True) 
    plt.scatter(average_filtered2['percentage'], average_filtered2['drug'], color=sns.color_palette('Reds',len(average_filtered2)))
    plt.title(f'Targets with higher presence in the drug: {drug_id} ', weight='bold', fontsize=15)
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    plt.xlabel('Percentage', fontsize=13)
    return plt.show()


# # 1-Overview: Features
# > ### 1.1 Categorical features:
# ***
# **First glimpse: 876 features with:**
# * Features **g-** signify gene expression data.
# * Features **c-** signify cell viability data.
# * **cp_type** indicates samples treated with a compound, **trt_cp** samples treated with the compounds. 
# * **cp_vehicle** or with a control perturbation (ctrl_vehicle); control perturbations have no MoAs.
# * **cp_time** and **cp_dose** indicate treatment duration (24, 48, 72 hours) and dose (high or low).

# In[2]:


a.head()


# In[3]:


plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
#1 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((1,2),(0,0))
sns.countplot(x='cp_type', data=a, palette='rainbow', alpha=0.75)
plt.title('Train: Control and treated samples', fontsize=15, weight='bold')
#first row sec col
ax1 = plt.subplot2grid((1,2),(0,1))
sns.countplot(x='cp_dose', data=a, palette='Purples', alpha=0.75)
plt.title('Train: Treatment Doses: Low and High',weight='bold', fontsize=18)
plt.show()


# In[4]:


plt.figure(figsize=(15,5))
sns.distplot( a['cp_time'], color='red', bins=5)
plt.title("Train: Treatment duration ", fontsize=15, weight='bold')
plt.show()


# * **Few control samples.**
# * **The low and high doses were applied equally.**
# * **3 treatment durations: 24h, 48h and 72h.**

# # 2- `c-` Features are related to cell viability. What is cell viability? 
# 
# A viability assay is an assay that is created to determine the ability of organs, cells or tissues to maintain or recover a state of survival. Viability can be distinguished from the all-or-nothing states of life and death by the use of a quantifiable index that ranges between the integers of 0 and 1 or, if more easily understood, the range of 0% and 100%. Viability can be observed through the physical properties of cells, tissues, and organs. Some of these include mechanical activity, motility, such as with spermatozoa and granulocytes, the contraction of muscle tissue or cells, mitotic activity in cellular functions, and more. Viability assays provide a more precise basis for measurement of an organism's level of vitality.[1]
# 
# > ### 2.1 c-xxx features

# In[5]:


plotf('c-10', 'c-50', 'c-70', 'c-90')


# 
# ***First observation in this EDA:***
# 
# As mentioned in the definition, cell viability should range between the integers 0 and 1. Here, we have values in the range -10 and 6 because the data were z-scored and then normalized using a procedure called [quantile normalization](https://clue.io/connectopedia/glossary#Q).
# 
# *A high negative cell viability measure reflects a high fraction of killing [@by the host](https://www.kaggle.com/c/lish-moa/discussion/191487).* In other words:
# * High negative values = High number of dead cells
# * High positive values = High number of living cells.
# 
# > ### 2.2 Cell viability:
# 
# Let's see the difference between the cell viability in a control and treated sample.

# In[6]:


plotd("c-30")


# This is just a first impression after checking multiple cell lines *(We need to check all the cell lines to draw conclusions).*
# 
# > ### 2.3 Treatment time:
# 
# **Next, let's see the impact of the treatment time on the cell viability.**

# In[7]:


plott('c-30')


# > ### 2.4 Cells correlation
# 
# **Let's see the correlation between cell viability features (in the treated samples, no control).**

# In[8]:


#Select the columns c-
c_cols = [col for col in a.columns if 'c-' in col]
#Filter the columns c-
cells=treated[c_cols]
#Plot heatmap
plt.figure(figsize=(15,6))
sns.heatmap(cells.corr(), cmap='coolwarm', alpha=0.9)
plt.title('Correlation: Cell viability', fontsize=15, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()


# In[9]:


corrs(cells, 'Cell', 'Cell 2', rows=7)


# * **Many high correlations between c- features. This is something to be taken into consideration in feature engineering.**
# ***
# 
# # 3- `g-` Features are related to gene expression. What is gene expression?
# 
# Gene expression is the process by which information from a gene is used in the synthesis of a functional gene product. These products are often proteins. You can refer to my notebook [COVID_19: Viral proteins identification](https://www.kaggle.com/amiiiney/covid-19-proteins-identification-with-biopython) to understand more how gene expression works.
# 
# In short, the mechanism of action of the 207 targets in this study will activate some genes, gene expression will take place and byproducts (proteins) will be synthesized.
# > ### 3.1 g-xxx features

# In[10]:


plotf('g-10','g-100','g-200','g-400')


# > ### 3.2 Gene expression:
# 
# ***How to interpret those values?***
# 
# Gene expression levels are calculated by the ratio between the expression of the target gene (i.e., the gene of interest) and the expression of one or more reference genes (often household genes). [3]
# 
# > To understand this better, let's compare the samples treated with the drugs and the control samples.

# In[11]:


plotd('g-510')


# > ### 3.3 Treatment time:
# 
# **let's check the impact of the treatment time on the gene expression.**

# In[12]:


plott('g-510')


# > ### 3.4 Genes correlation:
# 
# **let's see the correlation between gene expression features. (in the treated samples, no control)**

# In[13]:


#Select the columns g-
g_cols = [col for col in a.columns if 'g-' in col]
#Filter the columns g-
genes=treated[g_cols]
#Plot heatmap
plt.figure(figsize=(15,7))
sns.heatmap(genes.corr(), cmap='coolwarm', alpha=0.9)
plt.title('Gene expression: Correlation', fontsize=15, weight='bold')
plt.show()


# * **We have both negative and positive correlations here between genes. Interesting!** *(The control samples were not included, so the negative correlation between some genes is not related to the control/treated samples).*
# 
# Let's have a closer look at the high correlation genes.

# In[14]:


corrs(genes, 'Gene', 'Gene 2')


# **Strong negative correlation genes**

# In[15]:


#Correlation between drugs
corre= genes.corr()
#Unstack the dataframe
s = corre.unstack()
so = s.sort_values(kind="quicksort", ascending=False)
#Create new dataframe
so2= pd.DataFrame(so).reset_index()
so2= so2.rename(columns={0: 'correlation', 'level_0':'Drug 1', 'level_1': 'Drug2'})
#Filter out the coef 1 correlation between the same drugs
so2= so2[so2['correlation'] != 1]
#Drop pair duplicates
so2= so2.reset_index()
so2= so2.sort_values(by=['correlation'])
pos = [1,3,5,7,9,11,13,15,17,19,21]
so2= so2.drop(so2.index[pos])
so2= so2.round(decimals=4)
so2=so2.drop('index', axis=1)
so3=so2.head(4)
#Show the first 10 high correlations
cm = sns.light_palette("Red", as_cmap=True)
s = so2.head().style.background_gradient(cmap=cm)
s


# We have 34 gene pairs with **+0.8** correlation. This information will be useful, we will come back to it in the following sections.
# > ### 3.5 Mean gene expression of g-xxx features

# In[16]:


#Transpose the dataframe
genesT=genes.T
#Calculate the mean of each g_xxx feature
genesT['mean'] = genesT.apply (lambda row: mean(row), axis=1)
#Plot the mean values
genesTm=genesT.reset_index()
genesTm=genesTm[['index', 'mean']]
plotgene(genesTm)


# The mean gene expression of the 772 genes show some strong negative and positive gene expression values. This can be due to many factors:
# * Some drugs **upregulate** and others **downregulate** some genes: For example, drug-A could reduce gene-X expression level while drug-B could elevate gene-Y expression level.
# 
# * Some genes have high **negative correlation**, so gene-X has a high positive gene expression value which means gene-Y will have a high negative gene expression value.
# 
# * Other factors are related to the drug types and the genes, the information about drugs, genes and cells is not provided!
# 
# 
# 
# # 4-Targets *(MoA)*:
# ***
# > ## 4-1 Scored targets:
# 
# This is a multi-label classification, we have 207 MoA and we have to find out the mechanism of action of the 5000 drugs that were treated in the `sig_id` samples. A single sample treated with a drug can have many active targets, in other words, one drug can have more than 1 mechanism of action, so we have to predict the mechanisms of action of each drug.
# 
# *We will filter the **train_targets_scored** dataset and keep just the treated rows (we discard the control rows because they are not treated with the drugs).*
# 

# In[17]:


average= avgdrug(d)
uniques=unidrug(d)
average_filtered=avgfiltered(d)

plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
#1 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((1,2),(0,0))
sns.countplot(uniques['count'], color='deepskyblue', alpha=0.75)
plt.title('Unique elements per target [0,1]', fontsize=15, weight='bold')
#first row sec col
ax1 = plt.subplot2grid((1,2),(0,1))
sns.distplot(average['percentage'], color='orange', bins=20)
plt.title("The targets mean distribution", fontsize=15, weight='bold')
plt.show()


# * All the targets are present in at least one sample.
# * The presence of the targets is very low in the samples (Mostly less than 0.75%).
# * Some targets *(outliers)* have a higher presence in comparison with the rest of targets with a percentage in the range (3%, 4%).
# 
# > ### 4-1-1 **Most frequent targets:**

# In[18]:


plt.figure(figsize=(7,7))
average_filtered.sort_values('percentage', inplace=True) 
plt.scatter(average_filtered['percentage'], average_filtered['drug'], color=sns.color_palette('Reds',len(average_filtered)))
plt.title('Targets with higher presence in train samples', weight='bold', fontsize=15)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.xlabel('Percentage', fontsize=13)
plt.show()


# * It seems we have 2 outliers here: **nfkb_inhibitor** and **proteasome_inhibitor.** *(We will come back to this later)*
# 
# * We can see many target labels in the plot: inhibitor, agonist, antagonist.
# > ### 4-1-2 Targets: Drug types:

# In[19]:


inhibitors = [col for col in d.columns if 'inhibitor' in col]
activators = [col for col in d.columns if 'activator' in col]
antagonists = [col for col in d.columns if 'antagonist' in col]
agonists = [col for col in d.columns if 'agonist' in col]
modulators = [col for col in d.columns if 'modulator' in col]
receptors = [col for col in d.columns if 'receptor' in col]
receptors_ago = [col for col in d.columns if 'receptor_agonist' in col]
receptors_anta = [col for col in d.columns if 'receptor_antagonist' in col]


labelss= {'Drugs': ['inhibitors', 'activators', 'antagonists', 'agonists', 'receptors', 'receptors_ago', 'receptors_anta'],
          'Count':[112,5,32,60, 53, 24, 26]}


labels= pd.DataFrame(labelss)
labels=labels.sort_values(by=['Count'])
plt.figure(figsize=(15,5))
plt.bar(labels['Drugs'], labels['Count'], color=sns.color_palette('Reds',len(labels)))
plt.xticks(weight='bold')
plt.title('Target types', weight='bold', fontsize=15)
plt.show()


# * **Inhibitor targets** are dominating with 121 targets (out of 206).
# * Agonists, receptors and antagonists come at the second plan.
# ***
# > ### 4-1-3 Correlation between targets:

# In[20]:


ploth(drugs_tr)


# Most of the targets have 0 correlation. It is worth recalling that the presence of active targets in the samples in very low (mainly 1 or 2 targets per sample).
# 
# However, we notice some yellow dots *(high correlation)* between some targets. Let's have a closer look over these targets.
# 
# > ### 4-1-4 Targets with the highest MoA correlation

# In[21]:


#Correlation between drugs
corre= drugs_tr.corr()
#Unstack the dataframe
s = corre.unstack()
so = s.sort_values(kind="quicksort", ascending=False)
#Create new dataframe
so2= pd.DataFrame(so).reset_index()
so2= so2.rename(columns={0: 'correlation', 'level_0':'Target 1', 'level_1': 'Target 2'})
#Filter out the coef 1 correlation between the same drugs
so2= so2[so2['correlation'] != 1]
#Drop pair duplicates
so2= so2.reset_index()
pos = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35]
so2= so2.drop(so2.index[pos])
so2= so2.round(decimals=4)
so2=so2.drop('index', axis=1)
so3=so2.head(4)
#Show the first 10 high correlations
cm = sns.light_palette("Red", as_cmap=True)
s = so2.head().style.background_gradient(cmap=cm)
s


# * **2 Target-pairs have +0.9 correlation:** 
# 
# Those target pairs must be in the few samples that have more than two active targets. The functionality of these targets and their distribution is something to be taken into consideration because in this case, we have **a multi-label classification problem**, where the correlation between the labels is also important and the model selection should be based on the labels correlation. Select a model that finds patterns not just in the train data but also in the **multi-label target data.**
# 
# Below, we will try to connect the dots and try to match the high correlated targets with the targets with the most presence in the samples.

# In[22]:


plt.figure(figsize=(8,10))
the_table =plt.table(cellText=so3.values,colWidths = [0.35]*len(so3.columns),
          rowLabels=so3.index,
          colLabels=so3.columns
          ,cellLoc = 'center', rowLoc = 'center',
          loc='left', edges='closed', bbox=(1,0, 1, 1)
         ,rowColours=sns.color_palette('Reds',10))
the_table.auto_set_font_size(False)
the_table.set_fontsize(10.5)
the_table.scale(2, 2)
average_filtered.sort_values('percentage', inplace=True) 
plt.scatter(average_filtered['percentage'], average_filtered['drug'], color=sns.color_palette('Reds',len(average_filtered)))
plt.title('Targets with higher presence in train samples', weight='bold', fontsize=15)
plt.xlabel('Percentage', weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()


# * **Observations:**
# 
# >1. **nfkb_inhibitor and proteasome_inhibitor** have +0.9 correlation and are highly presented in the samples.
# 2.  **Kit_inhibtor** is highly correlated with 2 targets: **pdgfr_inhibitor and flt3_inhibitor**.
# 
# The samples having +2 active targets most probably include those 4 target pairs.
# ***
# 
# > ## 4-2 Nonscored targets:
# 
# In this section, we will have a look over the dataset provided that will not be used in the score. This dataset has 402 MoAs *(more than the 206 MoAs in the targets_scored dataset that will be used in the score).* 
# 
# This dataset can be used for transfer learning!

# In[23]:


#Extract unique elements per column
cols2 = nonscored.columns.to_list() # specify the columns whose unique values you want here
uniques2 = {col: nonscored[col].nunique() for col in cols2}
uniques2=pd.DataFrame(uniques2, index=[0]).T
uniques2=uniques2.rename(columns={0:'count'})
uniques2= uniques2.drop('sig_id', axis=0)

#############################
### PLOT
plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
#1 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((1,2),(0,0))
sns.countplot(uniques2['count'], palette='Blues', alpha=0.75)
plt.title('Nonscored: Unique elements per target [0,1]', fontsize=13, weight='bold')
#first row sec col
ax1 = plt.subplot2grid((1,2),(0,1))
sns.countplot(uniques['count'], color='cyan', alpha=0.75)
plt.title('Scored: Unique elements per target [0,1]', fontsize=13, weight='bold')
plt.show()


# In[24]:


print(f"{len(uniques2[uniques2['count']==1])} targets without ANY mechanism of action in the nonscored dataset")


# * We have seen in the previous section that the scored targets had [0,1] in all the samples, in other words, **ALL THE TARGETS HAD A MoA IN AT LEAST ONE SAMPLE.**
# 
# * Here, we see that the extra nonscored dataset contains **71 targets without ANY mechanism or action MoA.**
# 
# Now that we know that 71 targets don't have any mechanism of action, let's compare the targets with MoA in the nonscore dataset with the score one.
# > ### 4-2-1 Percentage of non-scored MoA in the samples:

# In[25]:


#Filter out just the treated samples
#Calculate the mean values
average2=nonscored.mean()
average2=pd.DataFrame(average2)
average2=average2.rename(columns={ 0: 'mean'})
average2['percentage']= average2['mean']*100
#Filter just the drugs with mean >0.01
average_filtered2= average2[average2['mean'] > 0.01]
average_filtered2= average_filtered2.reset_index()
average_filtered2= average_filtered2.rename(columns={'index': 'drug'})

#####################
#Plot the percentage of MoAs
plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
#1 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((1,2),(0,0))
sns.distplot(average2['percentage'], color='blue', bins=20)
plt.title('Percentage of the nonscored MoAs in the samples',weight='bold', fontsize=13)
#first row sec col
ax1 = plt.subplot2grid((1,2),(0,1))
sns.distplot(average['percentage'], color='gold', bins=20)
plt.title('Percentage of the scored MoAs in the samples',weight='bold', fontsize=13)
plt.show()


# * The presence of the nonscored targets in the samples is 10 times lower than the scored ones, which will be translated basically to 1 MoA per target in the samples.
# 
# We can deduct from the latest findings that the nonscored targets have less MoA in comparison to the scored ones. However, this doesn't mean that the information in the nonscored dataset can't be useful because if the nonscored targets have just 1 MoA and it happens that this 1 single MoA coincides with the scored targets in the same samples, then we might have interesting correlation between both targets.
# 
# To understand this better, let's first merge both scored and nonscored targets datasets and try to find patterns and relationships between the targets in both datasets. 
# > ### 4-2-2 Non-scored targets correlation:

# In[26]:


corrs(adt, 'target', 'target 2', 15, thresh= 0.7)


# **Great!** This nonscored dataset seems promising. 
# 
# If in the `scored_target` dataset we had just 4 target pairs with +0.7 correlation, by merging both datasets we have more than 15 target pairs highly correlated.
# 
# > ### 4-2-3 Heatmap of the 31 target with high correlation:

# In[27]:


#Correlation between drugs
corre= adt.corr()
#Unstack the dataframe
s = corre.unstack()
so = s.sort_values(kind="quicksort", ascending=False)
#Create new dataframe
so2= pd.DataFrame(so).reset_index()
so2= so2.rename(columns={0: 'correlation', 'level_0':'Drug 1', 'level_1': 'Drug2'})
#Filter out the coef 1 correlation between the same drugs
so2= so2[so2['correlation'] != 1]
#Drop pair duplicates
so2= so2.reset_index()
pos = [1,3,5,7,9, 11,13,15,17,19,21,23, 25, 27, 29, 31,33,35, 37, 39, 41, 43, 45]
so2= so2.drop(so2.index[pos])
#so2= so2.round(decimals=4)
so3=so2.head()
#Show the first 10 high correlations
cm = sns.light_palette("Red", as_cmap=True)
s = so2.head(16).style.background_gradient(cmap=cm)
s
#High correlation adt 22 pairs
adt15= so2.head(22)
#Filter the drug names
adt_1=adt15['Drug 1'].values.tolist()
adt_2=adt15['Drug2'].values.tolist()
#Join the 2 lists
adt3= adt_1 + adt_2
#Keep unique elements and drop duplicates
adt4= list(dict.fromkeys(adt3))
#Filter out the selected drugs from the "all drugs treated" adt dataset
adt5= adt[adt4]


# In[28]:


ploth(adt5)


# **Interesting!** Even though this is just a visual representation of the table above, here we can clearly see that many targets from the `scored_target` dataset have high correlation with several targets from the `nonscored_targets`.
# 
# > ### 4.3 Drug IDs: *(UPDATE)*
# 
# We are now provided the drug_id just for the train set, it will be valuable for a more stable CV/LB, however, I don't think it will be valuable for feature engineering since we don't have the drug_id for the test set, let's dig into this new feature:
# > ### 4-3-1 Most frequent drugs in the train set:

# In[29]:


plotc(f, 'drug_id')


# In[30]:


print('First observation:')
print(f"Number of rows of the Control vehicle is {len(a[a['cp_type']=='ctl_vehicle'])}")
print(f"Number of rows of the Drug cacb2b860 is {f.drug_id.value_counts()[0]}")


# So, Drug cacb2b860 is the control vehicle, this explains it's high presence in the train set. The second most frequent drugs is 87d714366 with 718 rows!
# > ### 4-3-2 Examples:

# In[31]:


plot_drugid('87d714366')


# In section 4.1, figure 2, we have seen that we had 2 outliers: **proteasome_inhibitor and nfkb_inhibitor**, I first thought that that there were many drugs with those MoAs in the train set, but apparently most of them belong to one single drug: **87d714366** that was profiled 718 times.
# 
# Another drug: **d50f18348** was profiled 178 times and has 3 MoAs:

# In[32]:


plot_drugid('d50f18348')


# > The other most frequent drugs have cdk_inhibitor, egfr_inhibitor and tubulin_inhibitor as MoAs.

# In[33]:


plt.figure(figsize=(7,7))
average_filtered.sort_values('percentage', inplace=True) 
plt.scatter(average_filtered['percentage'], average_filtered['drug'], color=sns.color_palette('Reds',len(average_filtered)))
plt.title('Targets with higher presence in train samples', weight='bold', fontsize=15)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.xlabel('Percentage', fontsize=13)

plt.axhline(y=21.5, color='deepskyblue', linestyle='-.')
plt.axhline(y=23.5, color='deepskyblue', linestyle='-.')
plt.text(1.7,23, 'Drug: 87d714366 ', fontsize=12, color='black', ha='left' ,va='top')
plt.text(1.7,15.7, 'Drug: 8b87a7a83 ', fontsize=12, color='black', ha='left' ,va='top')
plt.text(1.7,14.7, 'Drug: 5628cb3ee ', fontsize=12, color='black', ha='left' ,va='top')
plt.text(1.7,13.7, 'Drug: d08af5d4b ', fontsize=12, color='black', ha='left' ,va='top')
plt.show()


# Those are the outlier drugs, the drugs that were profiled more than 18 times! Next, let's check the other drugs, normally 1 drug was profiled 6 times *(2 doses x 3 treatment times)*, if a drug was profiled twice it will have 12 samples per drug_id.

# In[34]:


drug_count=f[['drug_id']].value_counts().to_frame()
drug_count=drug_count.rename(columns={0:'drug_count'})
drug_count2=drug_count['drug_count'].value_counts().to_frame().reset_index()
drug_count2=drug_count2.rename(columns={'index': 'Samples per drug', 'drug_count':'Number of Drugs'})
drug_count2[:12]


# **Observations:**
# * As expected, 2774 drugs out of 3700 drugs have 6 rows that correspond to 2 doses and 3 treatment times. 
# * Only 64 drugs have 12 samples, I was expecting more drugs to be profiled twice.
# * Only 3 drugs have 18 sample, the drugs were profiled 3 times.
# * 196 drugs were profiled 7 times, so 1 additional sample with respect to most of the drugs, perhaps a dosage or treatment time experiment was repeated for those drugs!
# 
# **My Drug_id conclusion:**
# 
# I am not sure why the competition host chose some drugs to be profiled hundreds of times but the data shows that most of the drugs,exactly 2774 drug, were profiled just once (6 samples), my intuition is that those samples will not be present in the test, however, the drugs with + 18 samples were intentionally profiled 100s of times because they have some mechanism of action targets that the test set drugs have.
# 
# For more info, you can check this discussion topic: [Extracting insights from Drug_ID](https://www.kaggle.com/c/lish-moa/discussion/195217).
# 
# ***
# # 5- Test features:
# After understanding the relationship between the features and the labels, we move on to the test set to understand the features and their relationship with the train features.
# > ## 5-1 Features:

# In[35]:


plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
#1 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((1,2),(0,0))
sns.countplot(x='cp_type', data=b, palette='rainbow', alpha=0.75)
plt.title('Test: Control and treated samples', fontsize=15, weight='bold')
#first row sec col
ax1 = plt.subplot2grid((1,2),(0,1))
sns.countplot(x='cp_dose', data=b, palette='rainbow', alpha=0.75)
plt.title('Test: Treatment Doses: Low and High',weight='bold', fontsize=18)
plt.show()


# In[36]:


plt.figure(figsize=(13,3))
sns.distplot( b['cp_time'], color='gold', bins=5)
plt.title("Test: Treatment duration ", fontsize=15, weight='bold')
plt.show()


# Everything seems similar to the train set:
# * The doses are equally applied.
# * Very few control samples.
# * Same treatment duration 24h, 48h and 72h.
# 
# **Good news!** It seems that both train and test datasets are similar in terms of *experimental conditions.* The variation would be in the gene expression and cell viability since the samples used in the test set are different than the train set.
# 
# Let's see how different are those samples!
# 
# > ## 5-2 Gene expression:

# In[37]:


#Filter out just the treated samples
treated2= b[b['cp_type']=='trt_cp']
treated_list2 = treated2['sig_id'].to_list()
full_tr= b[b['sig_id'].isin(treated_list2)]

#Select the columns c-
c_cols2 = [col for col in full_tr.columns if 'g-' in col]
#Filter the columns c-
cells2=treated2[c_cols2]

plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
#1 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((1,2),(0,0))
sns.heatmap(cells2.corr(), cmap='coolwarm', alpha=0.9)
plt.title('Test: Gene expression correlation', fontsize=15, weight='bold')
#first row sec col
ax1 = plt.subplot2grid((1,2),(0,1))
sns.heatmap(genes.corr(), cmap='coolwarm', alpha=0.9)
plt.title('Train: Gene expression: Correlation', fontsize=15, weight='bold')
plt.show()


# We can see several high positive and negative correlations between some genes, same as in the train set. However more investigation is needed to find some patterns and differences in gene expression between the train and test sets.

# > ### Top 10 gene pairs in the test set:

# In[38]:


#Correlation between drugs
corre= cells2.corr()
#Unstack the dataframe
s = corre.unstack()
so = s.sort_values(kind="quicksort", ascending=False)
#Create new dataframe
so2= pd.DataFrame(so).reset_index()
so2= so2.rename(columns={0: 'correlation', 'level_0':'Gene 1', 'level_1': 'Gene 2'})
#Filter out the coef 1 correlation between the same drugs
so2= so2[so2['correlation'] != 1]
#Drop pair duplicates
so2= so2.reset_index()
#so2= so2.sort_values(by=['correlation'])
pos = [1,3,5,7,9,11,13,15,17,19,21]
so2= so2.drop(so2.index[pos])
so2= so2.round(decimals=4)
so2=so2.drop('index', axis=1)
so4=so2.head(10)
cm = sns.light_palette("Red", as_cmap=True)
s = so2.head(10).style.background_gradient(cmap=cm)
s


# **Observations:**
# * **5/10** high correlated genes are the same as in the train set *(see section 3.2).*
# * **5** new gene pairs seem to be highly correlated in the test set then in the train set.
# * This is just a quick look over the TOP10 genes. I will update this section in the future with deeper analysis.
# 
# Understanding the difference between the genes correlation in the train and test sets will be crucial to determine and prevent the shake-up.
# 
# > ## 5-3 Cell viability:

# In[39]:


#Select the columns c-
c_cols3 = [col for col in b.columns if 'c-' in col]
#Filter the columns c-
cells3=treated_t[c_cols3]

plt.style.use('seaborn')
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15,5))
#1 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((1,2),(0,0))
sns.heatmap(cells3.corr(), cmap='coolwarm', alpha=0.9)
plt.title('Test: CVB correlation', fontsize=15, weight='bold')
#first row sec col
ax1 = plt.subplot2grid((1,2),(0,1))
sns.heatmap(cells.corr(), cmap='coolwarm', alpha=0.9)
plt.title('Train: CVB correlation', fontsize=15, weight='bold')
plt.show()


# Now, we check the high correlation cells and comapre them to the train test.

# In[40]:


corrs(cells3, 'Cell', 'Cell 2', rows=10)


# Findings based on the top 10 cell pairs:
# * 4377 cell pairs have +0.8 correlation in comparison with the 4187 in the train set.
# * 3/10 cell pairs in the test set are not present in the train set.
# * The order of the cells correlation is different in the test set!

# ***
# # References:
# 
# [1] Viability assay https://en.wikipedia.org/wiki/Viability_assay
# 
# [2] COVID-19 viral proteins identification https://www.kaggle.com/amiiiney/covid-19-proteins-identification-with-biopython
# 
# [3] Gene expression level https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/gene-expression-level
# 
# [4] Image credit: https://www.labiotech.eu/cancer/forx-therapeutics-cancer-treatment/
