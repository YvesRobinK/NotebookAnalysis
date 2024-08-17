#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Installing modin for pandas query acceleration:-
get_ipython().system('pip install modin;')

# General imports:-
import numpy as np;
from scipy.stats import iqr, mode, kurtosis;
from termcolor import colored;
from warnings import filterwarnings;
filterwarnings('ignore');
from gc import collect;

import matplotlib.pyplot as plt;
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns;

# Using pandas modin extension:-
from pandas import DataFrame, merge as pdmerge;
import modin.pandas as pd;
from ray import init;
init(ignore_reinit_error=True);

pd.set_option("precision", 4);


# In[2]:


# Model specifics:-
from sklearnex import patch_sklearn;
patch_sklearn()

from sklearn_pandas import gen_features, DataFrameMapper;
from sklearn.base import TransformerMixin, BaseEstimator;
from sklearn.pipeline import Pipeline, make_pipeline;
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler;
from sklearn.feature_selection import SequentialFeatureSelector as SFS;
from sklearn.model_selection import GroupKFold, cross_val_score;
from sklearn.metrics import roc_auc_score, roc_curve;

from sklearn.ensemble import HistGradientBoostingClassifier as HGBMC;


# # Tabular Playground Series- April 2022:- 
# 
# This is a time series classification involving 13 sensor readings along nearly 26,000 training sequences on 671 training subjects to elicit the biological state as a classification/ prediction. Each sequence contains 60 steps (1 min totally and 1 step per second) to elicit 1 state over the 60 steps.
# 
# I have prepared this notebook from the work done by AmbroseM using the modin extension for pandas and intel sklearn extensions as a learning experience.

# In[3]:


# Loading and visualizing the data:-
xtrain = pd.read_csv('../input/tabular-playground-series-apr-2022/train.csv', 
                           encoding = 'utf8');
xtest = pd.read_csv('../input/tabular-playground-series-apr-2022/test.csv', 
                          encoding = 'utf8');
ytrain = pd.read_csv('../input/tabular-playground-series-apr-2022/train_labels.csv', 
                    encoding = 'utf8');
SubFl = pd.read_csv('../input/tabular-playground-series-apr-2022/sample_submission.csv', 
                    encoding = 'utf8');

print(colored(f"\nTrain data\n", color= 'blue', attrs= ['dark', 'bold']));
display(xtrain.head(5));

print(colored(f"\nTarget data\n", color= 'blue', attrs= ['dark', 'bold']));
display(ytrain.head(5));

print(colored(f"\nTest data\n", color= 'blue', attrs= ['dark', 'bold']));
display(xtest.head(5));

print(colored(f"\nSubmission data\n", color= 'blue', attrs= ['dark', 'bold']));
display(SubFl.head(5));

sensor_col_lst = list(xtrain.iloc[0:2, :].\
columns[xtrain.iloc[0:2,:].columns.str.startswith('sensor_')]);
print(colored(f"\nSensor columns\n", color= 'blue', attrs= ['dark', 'bold']));
print(colored(f"{sensor_col_lst}", color = 'blue'));

collect();


# # 1. Data preprocessing and visualization
# 
# In this section, we reduce the train-test set memory usage, develop basic distribution plots and study feature interaction metrics to elicit better development endeavors.

# In[4]:


# Performing data preprocessing:-
print(colored(f"\nTrain data information\n", color= 'blue', attrs= ['dark', 'bold']));
print(f"{xtrain.info()}");

print(colored(f"\nTest data information\n", color= 'blue', attrs= ['dark', 'bold']));
print(f"{xtest.info()}");


# In[5]:


# Checking if the target class is balanced/ imbalanced:-
fig, ax= plt.subplots(1,1, figsize= (4,6));
sns.barplot(y= ytrain.state.value_counts(normalize= False).values, 
            x= ytrain.state.unique(), palette= 'Blues',saturation= 0.90, ax=ax);
ax.set_title(f"Target column distribution for (im)balanced classes\n", color= 'tab:blue', fontsize= 12);
ax.set_yticks(range(0,14001,1000));
ax.set_xlabel('Target Class');
plt.show();


# ### Model Development Plan:-
# 
# These tables are structured as below-
# 1. xtrain:- This encapsulates the training features only. Each sequence number is associated with 60 time steps with numerical readings for that sequence number from 13 sensors on a unique subject number for the sequence.
# 2. ytrain:- This is the target table, with sequence numbers and class labels for the sequence. This data is balanced, as seen in the previous cell target distribution, hence, no over-sampling is needed
# 3. xtest:- This continues the sequences from the train-set but we are unaware of the classification state
# 
# Model development requires feature engineering with new features encapsulating the sensor readings' descriptive statistics over the sequence number. 
# New columns including the mean, std, skewness, kurtosis, IQR, median, etc. can be created across all 13 sensors and their efficacy in the model development may be assessed for inclusion. 
# Classifier models like an LSTM/ ML models could be used after eliciting relevant features and pre-processing

# In[6]:


# Reducing memory usage for train-test sets:-
def ReduceMemory(df:pd.DataFrame):
    """
    This function assigns new dtypes to the relevant dataset attributes and reduces memory usage.
    The relevant data-type is determined from the description seen earlier in the kernel.
    
    Input:- df (dataframe):- Analysis dataframe
    Returns:- df (dataframe):- Modified dataframe
    """; 
    
    df[['subject']] = df[['subject']].astype(np.int16);
    df[['sequence']] = df[['sequence']].astype(np.int32);
    df[['step']] = df[['step']].astype(np.int8);
    
    #  selecting all sensor float columns and reassigning data-types:-   
    global sensor_col_lst;
    df[sensor_col_lst] = df[sensor_col_lst].astype(np.float32);
    
    return df; 


# In[7]:


# Developing the single step pipeline for memory reduction:-
MemReducer = Pipeline(steps= [('ReduceMemory', FunctionTransformer(ReduceMemory))]);
xtrain = MemReducer.fit_transform(xtrain, ytrain.state.values);
xtest = MemReducer.transform(xtest);

# Performing memory reduction in the target table:-
ytrain['sequence'] = ytrain['sequence'].astype(np.int16);
ytrain['state'] = ytrain['state'].astype(np.int8);

# Performing data preprocessing after memory reduction:-
print(colored(f"\nTrain data information after memory reduction\n", 
              color= 'blue', attrs= ['dark', 'bold']));
print(f"{xtrain.info()}");

print(colored(f"\nTest data information after memory reduction\n", 
              color= 'blue', attrs= ['dark', 'bold']));
print(f"{xtest.info()}");

# Merging the target with the features to form a single train dataframe:-
xytrain = xtrain.merge(ytrain, how= 'left', left_on= 'sequence', right_on= 'sequence',
                       suffixes = ('',''));
del xtrain, ytrain, MemReducer;

collect();


# In[8]:


# Plotting correlation heatmap for the train data sensor readings:-
fig, ax= plt.subplots(1,1, figsize= (18,10));
sns.heatmap(data= xytrain.loc[:, sensor_col_lst].corr(), 
            vmin= 0.0, vmax=1.00, annot= True, fmt= '.1%',
            cmap= sns.color_palette('Spectral_r'), linecolor='black', linewidth= 1.00,
            ax=ax);

ax.set_title('Correlation heatmap for the train-set sensor data\n', fontsize=12, color= 'black');
plt.yticks(rotation= 0, fontsize= 9);
plt.xticks(rotation= 0, fontsize= 9);
plt.show();


# In[9]:


# Plotting sensor readings with the target to elicit mutual information and importance:-
fig, ax= plt.subplots(1,1, figsize= (12,6));
xytrain.drop(['sequence', 'subject', 'step'], axis=1).corr()[['state']].drop('state').plot.bar(ax= ax);

ax.set_title("Correlation analysis for all sensor columns\n", color= 'tab:blue', fontsize= 12);
ax.grid(visible= True, which= 'both', color= 'grey', linestyle= '--', linewidth= 0.50);
ax.set_xlabel('\nColumns', color= 'black');
ax.set_ylabel('Correlation', color= 'black');
plt.show();


# In[10]:


# Plotting boxplots to study the column distributions:-

fig, ax= plt.subplots(1,1, figsize= (18,10));
xytrain.loc[:, sensor_col_lst].plot.box(ax=ax, color = 'blue');
ax.set_title(f"Distribution analysis for sensor data in train-set\n", color= 'tab:blue', fontsize= 12);
ax.set_yticks(range(-700,701,100));
ax.grid(visible=True, which='both', color= 'lightgrey', linestyle= '--');
ax.set_xlabel('\nSensor Columns\n', fontsize= 12, color= 'tab:blue');
ax.set_ylabel(f'Sensor Readings\n', fontsize= 12, color= 'tab:blue');

plt.xticks(rotation= 90);
plt.show()


# ### Subject analysis:- 
# 
# This sub-section elicits key insights derived from the train-test set subjects as below-
# 1. We plan to study the common subject characteristics for state
# 2. We also plan to develop descriptive statistics of sensor readings based on subjects 
# 3. We will check if data leakage exists between the train-test subjects for any manual adjustments over the model results at the end of the assignment

# In[11]:


# Analyzing subject characteristics with an interim subject profile object:-

sub_prf_train= \
xytrain[['subject', 'sequence', 'state']].drop_duplicates().set_index('sequence').\
pivot_table(index= 'subject', values= 'state', aggfunc= [np.size, np.sum]);
sub_prf_train.columns= ['Nb_Min', 'Nb_S1'];
sub_prf_train['Nb_S0'] = sub_prf_train['Nb_Min'] - sub_prf_train['Nb_S1'];
sub_prf_train['S1_Rate'] = sub_prf_train['Nb_S1']/ sub_prf_train['Nb_Min'];

sub_prf_train.sort_values(['S1_Rate'], ascending= False);


# In[12]:


# Analyzing subject details from xytrain and subject profile objects:-
print(colored(f"\nTrain set subject inferences:-", color= 'red', attrs= ['bold', 'dark']));
print(colored(f"Number of train-set subjects = {len(sub_prf_train)}", color = 'blue'));
print(colored(f"Number of train-set subjects never going to state 1 = {len(sub_prf_train.query('S1_Rate == 0.0'))}", 
              color = 'blue'));
print(colored(f"Number of train-set subjects never going to state 0 = {len(sub_prf_train.query('S1_Rate == 1.0'))}", 
              color = 'blue'));

print(colored(f"\nDescriptive summary statistics for the training subjects\n", color = 'red', attrs= ['bold']));
display(sub_prf_train.iloc[:,:-1].describe().transpose().style.format('{:.1f}'));

print(colored(f"\nDescriptive summary statistics for the test-set subjects\n", color = 'red', attrs= ['bold']));
display(xtest[['sequence', 'subject']].drop_duplicates().groupby(['subject']).\
agg(Nb_Min = pd.NamedAgg('sequence', np.size)).describe().transpose().style.format('{:.1f}'));

print(colored(f"\nDescriptive summary statistics for the training subjects never in state 1\n", 
              color = 'red', attrs= ['bold']));
display(sub_prf_train.loc[sub_prf_train.S1_Rate== 0.0].describe().transpose().style.format('{:.1f}'));

print(colored(f"\nSensor summary statistics for the training subjects never in state 1\n", 
              color = 'red', attrs= ['bold']));
display(xytrain.loc[xytrain.subject.isin(sub_prf_train.loc[sub_prf_train.S1_Rate== 0.0].index), sensor_col_lst].\
        describe().transpose().style.format('{:,.1f}'));

print(colored(f"\nSensor summary statistics for the training subjects in state 1 and 0\n", 
              color = 'red', attrs= ['bold']));
display(xytrain.loc[xytrain.subject.isin(sub_prf_train.loc[sub_prf_train.S1_Rate > 0.0].index),sensor_col_lst].\
        describe().transpose().style.format('{:,.1f}'));

print(colored(f"\nSensor summary statistics for all training subjects\n", color = 'red', attrs= ['bold']));
display(xytrain.loc[:,sensor_col_lst].describe().transpose().style.format('{:,.1f}'));


# In[13]:


# Plotting unique sequences per subject:-

_ = xytrain[['sequence', 'subject', 'state']].drop_duplicates().\
             groupby(['subject','state'])['sequence'].nunique().reset_index().\
             pivot_table(index= 'subject', columns= 'state', values= 'sequence', aggfunc= [np.sum]);
_.columns = ['Nb_Unq_Seq0', 'Nb_Unq_Seq1'];

fig, ax= plt.subplots(2,1, figsize= (18,15));

sns.lineplot(data= _.values , palette= 'rainbow', ax= ax[0], linestyle= '-');
ax[0].set_title(f"Number of unique sequences per subject in the training set\n", color= 'black', fontsize= 12);
ax[0].legend(loc= 'upper right', fontsize= 8);
ax[0].set_xlabel('Subjects\n', color= 'black', fontsize= 10);
ax[0].set_ylabel('Sequences', color= 'black', fontsize= 10);
ax[0].grid(visible= True, which= 'both', linestyle= '-', color= 'lightgrey');
ax[0].set_xticks(range(0, 680, 25));
ax[0].set_yticks(range(0, 181, 15));

sns.lineplot(data=_.loc[sub_prf_train.loc[sub_prf_train.S1_Rate== 0.0].index][['Nb_Unq_Seq0']].values, 
             palette= 'Dark2',ax= ax[1]);
ax[1].set_title(f"\nNumber of unique sequences per subject in the training set never in state1\n",
                color= 'black', fontsize= 12);
ax[1].grid(visible= True, which= 'both', linestyle= '-', color= 'lightgrey');
ax[1].set_xticks(range(0, 65, 5));
plt.show();

del _;


# In[14]:


# Analyzing the distributions of the sensor readings across state:-
for col in sensor_col_lst:
    fig, ax= plt.subplots(1,1, figsize = (12,3.5));
    sns.kdeplot(data=xytrain[[col, 'state']], x= col, hue="state", 
                multiple="stack", palette = 'rainbow', ax= ax);
    ax.grid(visible= True, which= 'both', color= 'lightgrey', linestyle= '--');
    ax.set_xlabel('');
    ax.set_title(f'\n{col}\n');
    plt.show();


# In[15]:


# Analyzing the univariate characteristics for all sensors across the states:-

print(colored(f"\nState 0 descriptions\n", color= 'blue', attrs= ['bold', 'dark'])); 
display(xytrain.loc[xytrain.state == 0, sensor_col_lst].describe().transpose()\
        .style.format('{:,.2f}'));
print();

print(colored(f"\nState 1 descriptions\n", color= 'blue', attrs= ['bold', 'dark'])); 
display(xytrain.loc[xytrain.state == 1, sensor_col_lst].describe().transpose().\
        style.format('{:,.2f}'));


# In[16]:


# Deleting interim tables after usage:-
del sub_prf_train;
collect();


# # 2. Feature Engineering
# 
# In this section, we create features based on descriptive statistics, remove outliers and shortlist important features for further model development. We will create custom functions and classes and assemble a pipeline for the same.

# In[17]:


# Creating new features based on descriptive statistics and sequences per subject:-
def MakeFeatures(df: pd.DataFrame):
    """
    This function creates summary stats based features for the model development grouped by sensor using-
    1. mean
    2. std
    3. iqr
    4. kurtosis
    5. std/ absolute mean
    6. up-down passes for sensor 2- special sensor from EDA
    7. number of unique sequences per subject
    """;
    
    global sensor_col_lst; 
    
    # Creating grouper object with all sensor features:-
    grouper = df.groupby(['sequence'])[sensor_col_lst];
    
    # Creating model dataframe with the summary features:-
    mdl_df = \
    pd.concat((
        grouper.mean().add_prefix('mean_'),
        grouper.std().add_prefix('std_'),
        np.clip(grouper.std()/ np.abs(grouper.mean()), a_min= -1e30, a_max= 1e30).add_prefix('stdvsmean_')
    ),axis=1);
    del grouper;

    # Adding kurtosis and IQR of each sensor to the model dataframe:-
    for col in sensor_col_lst:
        mdl_df = \
        pd.concat((mdl_df,
                   df[['sequence',col]].groupby(['sequence']).agg({col: kurtosis}).add_prefix('kutosis_'),
                   df[['sequence',col]].groupby(['sequence']).agg({col: iqr}).add_prefix('iqr_')),
                  axis=1);
    
    # Creating transpose for sensor 2 readings:-    
    _ = df[['sequence', 'step','sensor_02']].pivot(index= 'sequence', columns = 'step', values= 'sensor_02');

    # Creating additional features for sensor 2:-   
    mdl_df['up_sensor_02'] = (_.diff(axis=1) > 0.0).sum(axis=1);
    mdl_df['down_sensor_02'] = (_.diff(axis=1) < 0.0).sum(axis=1);
    mdl_df['upsum_sensor_02'] = _.diff(axis=1).clip(0.0, None).sum(axis=1);
    mdl_df['downsum_sensor_02'] = _.diff(axis=1).clip(None, 0.0).sum(axis=1);
    mdl_df['upmax_sensor_02'] = _.diff(axis=1).max(axis=1);
    mdl_df['downmax_sensor_02'] = _.diff(axis=1).min(axis=1);
    mdl_df['upmean_sensor_02'] = np.nan_to_num(mdl_df['upsum_sensor_02'] / mdl_df['up_sensor_02'], 
                                               posinf=40);
    mdl_df['downmean_sensor_02'] = np.nan_to_num(mdl_df['downsum_sensor_02'] / mdl_df['down_sensor_02'], 
                                                 neginf=-40)

    del _;
    
    # Creating sequences per subject:-
    mdl_df = mdl_df.merge(df[['sequence', 'subject']].drop_duplicates(),
                          how= 'left', left_index= True, right_on = 'sequence', suffixes= ('',''));
    mdl_df = mdl_df.merge(df[['subject', 'sequence']].drop_duplicates().groupby('subject').\
                          agg(nb_seq_per_sub = pd.NamedAgg('sequence', np.size)),
                          how = 'left', on = 'subject', suffixes= ('',''));
    mdl_df.set_index(['sequence', 'subject'], inplace= True);
    
    print(colored(f"\nColumns in the dataset= {len(mdl_df.columns):.0f}\n", color= 'blue'));
    global Ftre_Lst;
    Ftre_Lst = list(mdl_df.columns);
    return mdl_df;


# In[18]:


# Removing outliers from the feature engineered dataset:-
class OutlierRemover(BaseEstimator, TransformerMixin):
    "This class removes outliers based on IQR multiplier (usually 1.5*IQR)";
    
    def __init__(self, iqr_mult:float = 1.50):
        "This function initializes the IQR multiplier for the outlier removal";
        self.iqr_mult_ = iqr_mult;
        
    def fit(self, X, y=None, **fit_params):
        "This function calculates the cutoff for outlier removal on the train-data";
        X_iqr = iqr(X, axis=0);
        self.OtlrLB_ = np.percentile(X.values, 25, axis=0) - self.iqr_mult_* X_iqr;
        self.OtlrUB_ = np.percentile(X.values, 75, axis=0) + self.iqr_mult_* X_iqr;
        del X_iqr;
        return self;
    
    def transform(self, X, y= None, **transform_params):
        "This function clips the outliers off the data-set and returns a pandas dataframe";
        return DataFrame(data= np.clip(X.values, a_min= self.OtlrLB_, a_max= self.OtlrUB_), 
                         index= X.index, columns= X.columns);


# In[19]:


# Shortlisting features based on univariate dependency:-
class FeatureSelector(BaseEstimator, TransformerMixin):
    "This class reduces features from the engineered data based on the dependency with the target";
    
    def __init__(self,iqr_cutoff:np.float16 = 0.0, train_obs_pct:np.float16= 0.05):
        """
        This function initializes the parameters for the class.
        This including iqr cutoff and train percent for rolling window
        """;
        self.iqr_cutoff = iqr_cutoff;
        self.train_obs_pct = train_obs_pct;
    
    def fit(self, X:pd.DataFrame, y= None, **fit_params):
        """
        This function calculates the dependency as below-
        1. Development of target dataframe and 1 column from the engineered data
        2. Shuffle the interim table with a random seed for reproducability
        3. Sort the data with the column value
        4. Develop a rolling mean across 5% training set length for the target column
        5. Plot the rolling mean vs index and store the describe() output in a global dataframe
        6. Shortlist columns with an IQR of rolling mean >= cutoff (self.iqr_cutoff)
        """;
        
        #  Creating function parameters and output univariate profile to store the dependency results:-      
        len_window= np.int32(self.train_obs_pct*len(y));
        ftre_lst = list(X.columns);
        ncols= 5;
        nrows= np.int16(np.ceil(len(ftre_lst)/ncols));
        Unv_Prf = pd.DataFrame();
        
        # Creating global feature selection storage object:-
        global Ftre_Sel_Lst;
        Ftre_Sel_Lst = [];
        
        X = pd.DataFrame(data= X.values, index= X.index, columns= X.columns);
        
        fig, ax= plt.subplots(nrows=nrows, ncols=ncols, figsize= (ncols*4, nrows*4)); 
        for i, col in enumerate(ftre_lst):
            df= y.merge(X[[col]].droplevel('subject'),
                        how= 'left',left_on= 'sequence',right_on= 'sequence',suffixes= ('',''));
            df= df.sample(frac=1.0, random_state= 10).sort_values([col], ascending= True); 
            df.reset_index(inplace= True);
            df['RollMean'] = df.state.rolling(len_window).mean();
            
            Unv_Prf = pd.concat((Unv_Prf, 
                                 df[['RollMean']].describe().\
                                 rename({'RollMean': col}, axis=1).transpose()[['25%', '75%', 'std']]),
                               axis=0, ignore_index= False);
            
            # Developing feature dependency plot:-                                    
            plt.subplot(nrows, ncols, i+1);
            sns.scatterplot(y=df.RollMean.values, x= df.index, palette= 'Blues', alpha= 0.60,
                            ax= ax[i//ncols,i%ncols]);
            ax[i//ncols, i%ncols].set_title(f'\n{col}\n', color= 'black', fontsize=9);
            ax[i//ncols, i%ncols].grid(visible= True,which= 'both',linestyle= '--',color='lightgrey');
            ax[i//ncols, i%ncols].set_xticks(range(0, 25001,5000));
        
        plt.xticks(fontsize=7.5);
        plt.tight_layout();
        plt.show();
        
        # Developing feature shortlist:-
        self.sel_ftre = list(Unv_Prf.loc[Unv_Prf['75%']-Unv_Prf['25%'] >=self.iqr_cutoff].index);
        Ftre_Sel_Lst = self.sel_ftre;
        
        return self;
    
    def transform(self, X:pd.DataFrame, y=None, **transform_params):
        "This function returns a truncated dataset with the shortlisted features";
        X1= X.copy();
        return X1[self.sel_ftre];
            


# In[20]:


# Recreating target training set from the combined xytrain data:-
ytrain = xytrain[['sequence','state']].drop_duplicates().set_index('sequence');

# Initializing global parameters for the data transformer pipeline:-
iqr_cutoff= 0.10;
train_obs_pct = 0.05;
Ftre_Lst= [];

# Initializing the data processor pipeline:-
Data_Processor=\
Pipeline(verbose= True, steps= 
        [('MakeFeatures', FunctionTransformer(MakeFeatures)),
         ('RemoveOutliers', OutlierRemover(iqr_mult=1.5)),
         ('Standardize', DataFrameMapper(input_df= True, df_out= True, default=None, drop_cols=None,
                                        features= gen_features(columns= np.expand_dims(Ftre_Lst, axis=1),
                                                              classes= [RobustScaler])
                                        )
         ),
         ('SelectFeatures', FeatureSelector(iqr_cutoff=iqr_cutoff, train_obs_pct=train_obs_pct))
        ]);

# Implementing the transformer pipeline on the training data:-
Xtrain = Data_Processor.fit_transform(xytrain.drop('state', axis=1), ytrain);
Xtest = Data_Processor.transform(xtest);

print(colored(f"Train-set size after feature processing is {Xtrain.shape}", 
              color= 'blue', attrs= ['bold', 'dark']));
print(colored(f"Test-set size after feature processing is {Xtest.shape}", 
              color= 'blue', attrs= ['bold', 'dark']));
print(colored(f"\nData-type of pipeline output = {type(Xtrain)}", 
              color = 'blue', attrs= ['bold', 'dark']));

collect();


# # 3. Model training:-
# 
# In this section, we aim to create n-classifiers using standard ML algorithms on a specific (large) sample of the train set (say 90%) with varying random seeds. This closely follows bootstrapping sampling with replacement from the training data. We finally calculate the central tendency of the trained classifiers and prepare the submission file for the competition
# 

# In[21]:


# Initializing other global parameters for the model training:-
nb_mdl = 100;
train_frac = 0.95;

# Initializing output dataframe to store the test set predictions:-
Mdl_Pred_Prf = pd.DataFrame(data= None, index= SubFl.sequence, 
                            columns = ['State'+str(i) for i in range(nb_mdl)],dtype= np.float32);

# Fitting the classifiers and collating test predictions:-
print(colored(f"Model training and prediction collation", color= 'red', attrs= ['bold', 'dark']));
for seed in range(nb_mdl):
    print(colored(f"Current seed = {seed}", color= 'blue', attrs= ['bold']));
    
    # Initializing the model instance:-
    model = HGBMC(learning_rate=0.10, max_leaf_nodes= 25, max_iter=1000, 
                  min_samples_leaf= 500, validation_fraction= (1-train_frac),
                  l2_regularization=1, max_bins=63, random_state= seed, verbose= 0,
                  early_stopping = True, scoring= 'roc_auc', n_iter_no_change= 50);
    # Fitting the model on the train-set:-    
    model.fit(Xtrain, ytrain.values);
    # Collating the test predictions:-   
    Mdl_Pred_Prf[f"State{seed}"] = model.predict_proba(Xtest)[:,1];


# In[22]:


# Preparing the submission file:-
pd.DataFrame(data=Mdl_Pred_Prf.mean(axis=1), columns= ['state']).\
reset_index().rename({'index': 'sequence'}, axis=1).to_csv('submission.csv', index= False)

