#!/usr/bin/env python
# coding: utf-8

# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > TABLE OF CONTENTS<br><div>  
# 
# * [INTRODUCTION](#2)
# * [PREPROCESSING](#3)
# * [EDA](#4)    
# * [DATA TRANSFORMS](#5)
# * [MODEL TRAINING](#6)
# * [ENSEMBLE AND SUBMISSION](#7)
# * [OUTRO](#8)

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfrom IPython.display import clear_output;\n!pip install -q category_encoders;\n!pip install -q sklego;\nclear_output();\n\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# General library imports:-\nimport requests;\n\nfrom copy import deepcopy;\nimport pandas as pd;\nimport numpy as np;\nfrom functools import partial;\nfrom collections import Counter;\nfrom itertools import product;\nfrom colorama import Fore, Style, init;\nfrom warnings import filterwarnings;\nfilterwarnings('ignore');\n\nfrom tqdm.notebook import tqdm;\nimport seaborn as sns;\nimport matplotlib.pyplot as plt;\n%matplotlib inline\nfrom pprint import pprint;\n\nfrom datetime import date;\nfrom itertools import product;\nfrom holidays import CountryHoliday;\nimport dateutil.easter as easter;\n\nfrom IPython.display import display_html, clear_output;\nfrom gc import collect;\n\nclear_output();\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Data analysis and model specifics:-\nfrom sklearn.preprocessing import LabelEncoder, SplineTransformer;\nfrom statsmodels.tsa.stattools import adfuller, kpss;\nfrom category_encoders import OneHotEncoder as OHE, OrdinalEncoder as OE;\n\nfrom sklearn.pipeline import Pipeline, make_pipeline;\nfrom sklearn.base import BaseEstimator, TransformerMixin;\nfrom sklearn.model_selection import (GroupKFold as GKF, \n                                     StratifiedKFold as SKF, \n                                     RepeatedStratifiedKFold as RSKF,\n                                     TimeSeriesSplit as TSKF,\n                                     KFold, \n                                     RepeatedKFold as RKF,\n                                     GroupKFold as GKF\n                                    );\nfrom sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler;\n\nfrom sklearn.linear_model import Ridge;\nfrom sklearn.ensemble import (HistGradientBoostingRegressor as HGBR,\n                              RandomForestRegressor as RFR,\n                              ExtraTreesRegressor as ETR);\n\nfrom lightgbm import LGBMRegressor as LGBMR, log_evaluation, early_stopping;\nfrom catboost import CatBoostRegressor as CBR;\nfrom xgboost import XGBRegressor as XGBR;\nfrom sklego.linear_model import LADRegression;\nfrom sklearn.metrics import r2_score;\n\n# Ensemble and tuning specifics:-\nimport optuna;\nfrom optuna import Trial, trial, create_study;\nfrom optuna.samplers import TPESampler;\noptuna.logging.set_verbosity = optuna.logging.ERROR;\n\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Setting global options:-\nsns.set({"axes.facecolor"       : "#ffffff",\n         "figure.facecolor"     : "#ffffff",\n         "axes.edgecolor"       : "#000000",\n         "grid.color"           : "#ffffff",\n         "font.family"          : [\'Cambria\'],\n         "axes.labelcolor"      : "#000000",\n         "xtick.color"          : "#000000",\n         "ytick.color"          : "#000000",\n         "grid.linewidth"       : 0.85,  \n         "grid.linestyle"       : "--",\n         "axes.titlecolor"      : \'tab:blue\',\n         \'axes.titlesize\'       : 9.5,\n         \'axes.labelweight\'     : "bold",\n         \'legend.fontsize\'      : 7.0,\n         \'legend.title_fontsize\': 7.0,\n         \'font.size\'            : 8.0,\n         \'xtick.labelsize\'      : 7.5,\n         \'ytick.labelsize\'      : 7.5,        \n        });\n\n# Color printing    \ndef PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):\n    "Prints color outputs using colorama using a text F-string";\n    print(style + color + text + Style.RESET_ALL); \n    \ndef DisplayAdjTbl(*args):\n    """\n    This function displays pandas tables in an adjacent manner, sourced from the below link-\n    https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side\n    """;\n    \n    html_str=\'\'\n    for df in args:\n        html_str+=df.render()\n    display_html(html_str.replace(\'table\',\'table style="display:inline"\'),raw=True);\n    \n    \n# Setting global options for dataframe display and pipeline outputs:-    \npd.set_option(\'display.max_columns\', 50);\npd.set_option(\'display.max_rows\', 100);\nfrom sklearn import set_config; \nset_config(transform_output = "pandas");\n\nclear_output();\nprint();\n')


# <a id="2"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #00802b" > INTRODUCTION<br><div> 

# **This is a time series forecasting challenge, emphasizing on forecasting book-sales for 5 countries and 3 stores across the calendar years 2017-2021.**
# 
# The performance metric is Symmetric MAPE, as seen in the link below- 
# https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
# 
# In this section, we import the data and engender a complete analysis including-
# 1. Basic display
# 2. Feature information check and holidays analysis across countries
# 3. Column plotting and visualization
# 4. Seasonality checks
# 5. Analysis of trends and most importantly, trend shifts in the data
# 6. Impact of holidays, covid19 events and vaccination rollouts
# 7. Feature interactions for potential relations and model development 

# <a id="2.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CONFIGURATION<br><div> 

# | Version<br>Number | Version Details | Best CV score| Single/ Ensemble|LB score|
# | :-: | --- | :-: | :-: |:-:|
# | **V1** |* Detailed EDA, plots and analysis<br>* Date features<br>* Holiday analysis<br>* Retained covid months in the train data<br>* ML baseline models and optuna ensemble|10.64122|Ensemble<br>Optuna||
# | **V2** |* Similar to V1<br>* Removed ETR in ML suite|10.59738|Ensemble<br>Optuna||
# | **V3** |* Similar to V2<br>* Post-processing with multiplier|10.59738|Ensemble<br>Optuna|32.8924|
# | **V4** |* Similar to V3<br>* Post-processing with multiplier|10.59738|Ensemble<br>Optuna||

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Configuration class:-\nclass CFG:\n    "Configuration class for parameters and CV strategy for tuning and training";\n    \n    # Data preparation:-   \n    version_nb         = 4;\n    test_req           = "N";\n    test_frac          = 1.00;\n    gpu_switch         = "OFF"; \n    state              = 42;\n    target             = \'num_sold\';\n    episode            = 19;\n    path               = f"/kaggle/input/playground-series-s3e{episode}/";\n    dtl_preproc_req    = "N";\n        \n    # Feature processing and adversarial CV:-   \n    ftre_plots_req     = "N";\n    \n    # Data transforms and scaling:-    \n    sec_ftre_req       = "Y";\n    scl_method         = "Robust";\n    scale_req          = "N";\n    remove_covid_mth   = "N";\n\n    # Model Training:- \n    ML                 = "Y";\n    n_splits           = 5;\n    n_repeats          = 1;\n    score_train_req    = "Y";\n    nbrnd_erly_stp     = 200 ;\n    mdlcv_mthd         = \'GKF\';\n    methods            = ["LGBMR", "XGBR", "CBR", "RFR", "HGBR"];\n    drop_ftre          = [\'date\'];\n    \n    # Ensemble:-\n    ensemble_req       = "Y";\n    direction          = "minimize";\n    n_ens_trials       = 500 if test_req == "N" else 5;\n    pst_prcess_mthd    = "mult";\n    pst_prcess_fct     = 1.81;\n    \n    # Global variables for plotting:-\n    grid_specs = {\'visible\': True, \'which\': \'both\', \'linestyle\': \'--\', \n                           \'color\': \'lightgrey\', \'linewidth\': 0.75};\n    title_specs = {\'fontsize\': 9, \'fontweight\': \'bold\', \'color\': \'tab:blue\'};\n \nPrintColor(f"\\n--> Configuration done!\\n");\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Commonly used CV strategies for later usage:-\nall_cv= {\'KF\'    : KFold(n_splits= CFG.n_splits, shuffle   = True,            random_state= CFG.state),\n         \'RKF\'   : RKF(n_splits  = CFG.n_splits, n_repeats = CFG.n_repeats,   random_state= CFG.state),\n         \'RSKF\'  : RSKF(n_splits = CFG.n_splits, n_repeats = CFG.n_repeats,   random_state= CFG.state),\n         \'SKF\'   : SKF(n_splits  = CFG.n_splits, shuffle   = True,            random_state= CFG.state),\n         "TSKF"  : TSKF(n_splits = CFG.n_splits, gap = 0, test_size = 27375),\n         "GKF"   : GKF(n_splits  = CFG.n_splits),\n        };\n\n# Scaler to be used for continuous columns:- \nall_scalers = {\'Robust\': RobustScaler(), \n               \'Z\'     : StandardScaler(), \n               \'MinMax\': MinMaxScaler()\n              };\nscaler      = all_scalers.get(CFG.scl_method);\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Defining the competition metric:-\ndef ScoreMetric(ytrue, ypred):\n    """\n    This function calculates the metric for the competition. \n    ytrue- ground truth array\n    ypred- predictions\n    returns - metric value (float)\n    """;\n    \n    SMAPE = abs(ytrue - ypred) / (abs(ytrue) + abs(ypred));\n    SMAPE = SMAPE.mean() * 200;\n    return SMAPE;\n\ncollect();\nprint();\n')


# <a id="3"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > PREPROCESSING<br><div> 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nclass GDPRequestor():\n    """\n    This class obtains the data for GDP across the countries in the train-test data\n    Reference - https://www.kaggle.com/competitions/playground-series-s3e19/discussion/423725\n    """;\n    \n    def __init__(self, country):\n        self.country = country;\n    \n    def _ObtainGDP(self, country, year):\n        """\n        This method scraps the website for World Bank to obtain the GDP values\n        """;\n        \n        alpha3 = {\'Argentina\':\'ARG\',\'Canada\':\'CAN\',\'Estonia\':\'EST\',\'Japan\':\'JPN\',\'Spain\':\'ESP\'}\n        url="https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json".format(alpha3[country],year)\n        self.response = requests.get(url).json()\n        return self.response[1][0][\'value\'];\n    \n    def ScrapGDP(self):\n        """\n        This method scraps for the GDP values and converts the data to a dataframe\n        """;\n\n        self.gdp = [];\n        for country in self.country:\n            row = [];\n            for year in range(2017,2023):\n                row.append(self._ObtainGDP(country,year));\n            self.gdp.append(row);\n\n        self.gdp = np.array(self.gdp);\n        self.gdp /= np.sum(self.gdp, axis=0);\n\n        return pd.DataFrame(self.gdp,index= self.country, columns=range(2017,2023));\n    \ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_line_magic('time', '')

class Preprocessor():
    """
    This class aims to do the below-
    1. Read the datasets
    2. Check information
    3. Collate starting features 
    """;
    
    def __init__(self):
        self.train    = pd.read_csv(CFG.path + f"train.csv", index_col = 'id', parse_dates= ['date']);
        self.test     = pd.read_csv(CFG.path + f"test.csv", index_col = 'id', parse_dates= ['date']);
        self.target   = CFG.target;
        self.dtl_preproc_req = CFG.dtl_preproc_req;
        
        self.sub_fl   = pd.read_csv(CFG.path + f"sample_submission.csv");
        
        PrintColor(f"Data shape - train-test = {self.train.shape} {self.test.shape}");
        
        PrintColor(f"\nTrain set head", color = Fore.GREEN);
        display(self.train.head(5).style.format(precision = 3));
        PrintColor(f"\nTest set head", color = Fore.GREEN);
        display(self.test.head(5).style.format(precision = 3));
        
        self.strt_ftre = self.test.columns.tolist();
  
    def _CollateInfoDesc(self):
        if self.dtl_preproc_req == "Y":
            PrintColor(f"\n{'-'*20} Information and description {'-'*20}\n", color = Fore.MAGENTA);

            # Creating dataset information and description:
            for lbl, df in {'Train': self.train, 'Test': self.test}.items():
                PrintColor(f"\n{lbl} information\n");
                display(df.info());
                collect();
        return self;
    
    @staticmethod
    def _MakeFmtColor(df):
        "This function highlights the associated text using different background colors based on the store values";

        colors = ['#EEF6F7', '#CCD0D1'];
        x = df.copy();
        factors = list(x['store'].unique());
        i = 0;
        for factor in factors:
            style = f'background-color: {colors[i]}'
            x.loc[x['store'] == factor, :] = style
            i = not i
        return x;

    def DisplayContents(self):
        "This method displays the train-test grouped contents";
        
        PrintColor(f"\nTrain-test set contents\n");
        display(pd.concat([self.train[['country', 'store', 'product']].\
                           groupby(['country', 'store', 'product']).size().reset_index().rename({0: 'Train'}, axis=1),
                           self.test[['country', 'store', 'product']].\
                           groupby(['country', 'store', 'product']).size().reset_index().rename({0: 'Test'}, axis=1)[['Test']]], 
                          axis=1).style.apply(self._MakeFmtColor, axis=None)
               );
        return self;
    
    def Analyzeholidays(self):
        "This method enlists monthly holidays per country and common holidays across all countries";
        
        # Analyzing country specific holidays:-
        self.Holidays = [];
        self.Cntry_Lst = self.train.country.unique();

        for country in tqdm(self.Cntry_Lst):
            for h in CountryHoliday(country, years = np.arange(2017,2022,1)).items():
                self.Holidays.append({'date' : h[0], 'holiday' : h[1], 'country': country});

        self.Holidays = pd.DataFrame(self.Holidays);
        self.Holidays['IsHoliday'] = 1;
        self.Holidays['IsHoliday'] = self.Holidays['IsHoliday'].astype(np.int8);

        self.Holidays = self.Holidays.pivot(index= 'date',columns= 'country', values= 'IsHoliday');
        self.Holidays.fillna(0.00, inplace= True);
        self.Holidays = self.Holidays.astype(np.int8);

        # Encoding country names in consonance with the rest of the code:-
        self.Holidays.columns = [f'IsHoliday{i}' for i in range(len(self.Cntry_Lst))];

        self.Holidays['IsCommonHoliday'] = np.where(np.sum(self.Holidays, axis=1) == len(self.Cntry_Lst),1,0).astype(np.int8);
        self.Holidays.index = pd.DatetimeIndex(self.Holidays.index);

        PrintColor(f"\nHoliday distribution by year and month and country across the model period\n");
        display(self.Holidays.groupby([self.Holidays.index.year, self.Holidays.index.month]).agg(np.sum));

        PrintColor(f"\nCommon holiday distribution by year and month and country across the model period\n");
        display(self.Holidays.loc[self.Holidays.IsCommonHoliday == 1].index);
        
        return self;
       
    def DoPreprocessing(self):
        
        PrintColor(f"{'-'*30} Data preprocessing {'-'*30}", color = Fore.MAGENTA);
        
        if self.dtl_preproc_req == "Y": self._CollateInfoDesc();
        
        self.Analyzeholidays();
        self.DisplayContents();
        return self; 
    
pp = Preprocessor();
pp.DoPreprocessing()
                 
collect();
print();


# **Key inferences:-**
# 
# 1. We have 5 countries- Argentina, Canada, Estonia, Spain, Japan<br>
# 2. The data contains 3 stores- Kagglazon, Kaggle Learn and Kaggle Store<br>
# 3. We have 5 products in the data- Using LLMs to Improve Your Coding, Using LLMs to Train More LLMs, Using LLMs to Win Friends and Influence People and Using LLMs to Win More Kaggle Competitions and Using LLMs to Write Better<br>
# 
# The train data runs from Jan17- Dec21 with a daily frequency and the test data is across the year 2022. There are no nulls in the data.
# We have common holidays across the countries on Jan1 (New Year). Country specific holidays are also analysed for potential sales impact.<br>
# 
# **The data appears to be a small set without too many features, but we actually have numerous time series in the table, each series corresponds to a unique combination of country, store and product. We shall encode the data for ease-of-use and then elicit this pivot in the next section**

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Extracting the GDP for countries:-\ngdp = GDPRequestor(country = pp.train.country.unique());\nGDP_Snp = gdp.ScrapGDP();\nGDP_Snp = \\\nGDP_Snp.reset_index().\\\nmelt(id_vars = ['index']).\\\nrename({'index': 'country', 'variable': 'year', 'value': 'GDP'}, axis=1);\n\nprint();\ncollect();\n")


# <a id="4"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > EDA<br><div> 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nxytrain = pp.train.copy();\n# Encoding the object features:-\nfor col in tqdm([\'country\', \'store\', \'product\']):\n    xytrain[col] = LabelEncoder().fit_transform(xytrain[[col]]) + 1;\n    xytrain[col] = xytrain[col].astype(np.int8);\n    \n# Combining the object features into a single column:-\nxytrain[\'Ftre_Comb_Lbl\'] = xytrain[\'country\'].astype(str) + xytrain[\'store\'].astype(str) + xytrain[\'product\'].astype(str);\n\n# Developing a link between the encoded labels and the original column values:-\nPrintColor(f"\\nFeature combinations between encoded values and original feature labels\\n");\ndisplay(xytrain[[\'country\', \'store\', \'product\', \'Ftre_Comb_Lbl\']].drop_duplicates().T);\n\ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Converting the data into a pivot for further analysis:-\nSales_Prf = xytrain.pivot(index= \'date\', columns= \'Ftre_Comb_Lbl\', values = CFG.target);\n\nSales_Prf.insert(0, \'Year\', Sales_Prf.index.year);\nSales_Prf[\'Year\'] = Sales_Prf[\'Year\'].astype(np.uint16);\n\nSales_Prf.insert(1, \'Month\', Sales_Prf.index.month);\nSales_Prf[\'Month\'] = Sales_Prf[\'Month\'].astype(np.int8);\n\nSales_Prf.insert(2, \'Day\', Sales_Prf.index.day);\nSales_Prf[\'Day\'] = Sales_Prf[\'Day\'].astype(np.int8);\n\nSales_Prf.insert(3, \'WeekNb\', Sales_Prf.index.week);\nSales_Prf[\'WeekNb\'] = Sales_Prf[\'WeekNb\'].astype(np.int8);\n\nSales_Prf.insert(4, \'DayNb\',Sales_Prf.index.weekday);\nSales_Prf[\'DayNb\'] = Sales_Prf[\'DayNb\'].astype(np.int8);\n\nSales_Prf.insert(5, \'IsWeekend\', np.where(Sales_Prf.DayNb >=5,1,0));\nSales_Prf[\'IsWeekend\'] = Sales_Prf[\'IsWeekend\'].astype(np.int8);\n\n# Merging the sales profile table with public holidays:-\nSales_Prf = pp.Holidays.merge(Sales_Prf, how = \'right\', left_index= True, right_index= True);\nSales_Prf[pp.Holidays.columns] = Sales_Prf[pp.Holidays.columns].fillna(0.0).astype(np.int8);\n\n# Displaying the pivot information:-\nPrintColor(f"\\nComplete pivot table columns for sales across all combinations\\n");\ndisplay(Sales_Prf.columns);\n\nPrintColor(f"\\nSales Profile information\\n");\ndisplay(Sales_Prf.info());\n\nPrintColor(f"\\nComplete combinations\\n");\nCols = Sales_Prf.iloc[0:2, -75:].columns;\ndisplay(Cols);\n')


# <a id="4.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > KDE PLOT- YEAR AND COVID FLAG<br><div> 
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Creating xticks for end of quarters with labels:-\nDate_Labels             = pd.DataFrame(data= Sales_Prf.index[Sales_Prf.index.is_quarter_end]);\nDate_Labels['date_lbl'] = Date_Labels['date'].dt.year*100 + Date_Labels['date'].dt.month;\n\ncollect();\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    fig, ax = plt.subplots(25, 3, figsize= (21,80));\n\n    for i,col in tqdm(enumerate(Cols)):\n        sns.kdeplot(data= Sales_Prf[[\'Year\', col]], x= col, \n                    hue=\'Year\',\n                    palette = \'Spectral\',\n                    fill= True, \n                    ax= ax[i//3,i%3]\n                   );\n        ax[i//3,i%3].set(xlabel= \'\', ylabel = \'\');\n        ax[i//3,i%3].set_title(f"Combination = {col}",**CFG.title_specs);\n        ax[i//3,i%3].grid(**CFG.grid_specs);\n\n    plt.suptitle(f"Sales Units grouped by Year", **CFG.title_specs, y = 1.00);\n    plt.tight_layout();\n    plt.show();\n    \ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    fig, ax = plt.subplots(25,3, figsize= (21,80));\n\n    _ = Sales_Prf.iloc[:, -75:];\n    _.loc[_.index.date < date(2020,1,1), \'Is_Covid\'] = 0;\n    _.loc[_.index.date >=date(2020,1,1), \'Is_Covid\'] = 1;\n    _[\'Is_Covid\'] = _[\'Is_Covid\'].astype(np.int8);\n\n    for i,col in tqdm(enumerate(_.columns[0:-1])):\n        sns.kdeplot(data= _[[\'Is_Covid\',col]], \n                    x= col, \n                    hue=\'Is_Covid\',\n                    palette= [\'#F6C2BB\', \'#3E3E3C\'], \n                    fill= True, ax= ax[i//3,i%3]\n                   );\n        ax[i//3,i%3].set(xlabel= \'\', ylabel = \'\');\n        ax[i//3,i%3].set_title(f"\\nCombination = {col}\\n",**CFG.title_specs);\n        ax[i//3,i%3].grid(**CFG.grid_specs);\n\n    plt.suptitle(f"Sales Units grouped by pre-covid and covid periods", **CFG.title_specs, y= 1.00);\n    plt.tight_layout();\n    plt.show();\n    del _;\n    \ncollect();\n')


# <div style="color:'#050a14';
#            display:fill;
#            border-radius:5px;
#            background-color:#d6e0f5;
#            font-size:110%;
#            font-family:Calibri;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;color:'#050a14'; font-weight: bold">
#               Key inference:-<br> Covid -19 induced stress period elicits high impact on the sales unit volumes, possibly inducing some regime shifts in the data.<br> Weekly and monthly sales plots and growth rate plots will showcase these regime shifts more appropriately.<br>
# </p>
# </div>

# <a id="4.2"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > PERIODIC SALES PLOT<br><div> 
#     
# 1. We calculate the total weekly/ monthly sales by country, store and product
# 2. We plot them on a panel to elicit the evidence of time trends, regime shifts and prospective cyclical and seasonal patterns

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef MakeGrpLinePlot(dtpart:str, covid_id:str, figsize= (20,96)):\n    \n    global Sales_Prf;\n    \n    _ = Sales_Prf[[\'Year\', dtpart] + list(Cols)];\n    _.insert(0, \'Id\', _[\'Year\']*100.0 + _[dtpart]);\n    _[\'Id\'] = _[\'Id\'].astype(np.int32).astype(str);\n    _ = _.drop([\'Year\', dtpart], axis=1).groupby(\'Id\').agg([np.sum]);\n    _.columns = [j+\'_\'+i for i, j in _.columns];\n\n    combs=[str(i)+str(j) for i,j in list(product(range(1,6,1), [1,2,3]))];\n\n    fig, ax = plt.subplots(15,1,figsize = figsize, sharex= True);\n    for i, comb in enumerate(combs):    \n        sns.lineplot(data = _[_.columns[_.columns.str[4:6] == comb]],\n                     palette= [\'black\',\'#014F1C\', \'#890E03\',\'#034AD0\', "cyan"],\n                     ax= ax[i],\n                     **{"linewidth": 2.0}\n                    );\n        ax[i].axvline(x = covid_id, color= \'red\', linestyle= \'-\', linewidth= 2.5);\n        ax[i].grid(**CFG.grid_specs);\n        ax[i].set_ylabel(\'Units sold\');\n        ax[i].set_title(f"\\nTotal units sold for combination = {comb}\\n",**CFG.title_specs);\n        ax[i].set_xlabel(\'\');\n    \n    plt.suptitle(f"Total Sales by {dtpart}", **CFG.title_specs, y = 1.0)\n    plt.xticks(rotation= 90);\n    plt.tight_layout();\n    plt.show();\n\n    del combs;\n    collect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    MakeGrpLinePlot(dtpart=\'Month\', covid_id= \'202001\', figsize= (30,108));\n    \nprint();\ncollect();\n')


# <div style="color:'#050a14';
#            display:fill;
#            border-radius:5px;
#            background-color:#d6e0f5;
#            font-size:110%;
#            font-family:Calibri;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;color:'#050a14'; font-weight: bold">
# Key inferences:-<br>
#     
# 1. We can clearly look into the regime shifts in the data in early 2020, primarily caused by covid-19 stress.  
# 2. Spikes and anomalies exist in several combinations, especially in second half of the year  
# 3. Sales growth across H2 2020 seems to be an interesting pattern to observe and model.
# </p>
# </div>

# <a id="4.3"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > TOTAL SALES<br><div> 
# 
# We aim to do the below in the section- 
# 1. Calculation of total sales by product and store by weekday number to elicit higher sales during weekend
# 2. Calculation of weekly total sales to indicate periods of high and low sales
# 3. Calculation of weekly total sales to indicate periods of seasonality

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef PltDtPrtSales(df:pd.DataFrame, dtpart:str):\n    "This function plots the line plots for the given series to elicit seasonality and cyclicality";\n    \n    fig, ax = plt.subplots(2,3, figsize= (30,11), sharex= True, \n                           gridspec_kw = {\'hspace\': 0.25, "wspace": 0.2}\n                          );\n    for i in range(1,7,1):\n        try:\n            a = ax[(i-1)//3, (i-1)%3];\n            df[df.columns[df.columns.str.startswith(str(i))]].plot.line(ax = a, marker = \'o\');\n            a.set_title(f"\\nTotal sales per product and store across country {i}\\n", **CFG.title_specs);\n            a.grid(**CFG.grid_specs);\n            a.legend(loc = \'upper left\', fontsize= 6);\n            a.set(xlabel = \'\', ylabel = \'\');\n            a.legend(bbox_to_anchor = (1,1));\n        except:\n            pass\n    \n    plt.suptitle(f"Total daily sales for year = {yy} across {dtpart.upper()}",\n                 color = \'blue\', fontsize = 12, fontweight = \'bold\', \n                 y = 0.99);\n    plt.tight_layout();\n    plt.show();\n    collect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    for yy in tqdm(range(2017,2022,1)):\n        PltDtPrtSales(df= Sales_Prf.loc[Sales_Prf.Year == yy, [\'DayNb\'] + list(Cols)].groupby(\'DayNb\').sum()/1000,\n                      dtpart = "DayNb"\n                     );\n    \ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    for yy in tqdm(range(2017,2022,1)):\n        PltDtPrtSales(df= Sales_Prf.loc[Sales_Prf.Year == yy, [\'WeekNb\'] + list(Cols)].groupby(\'WeekNb\').sum()/1000,\n                      dtpart = "WeekNb"    \n                     );\n        \nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    for yy in tqdm(range(2017,2022,1)):\n        PltDtPrtSales(df= Sales_Prf.loc[Sales_Prf.Year == yy, [\'Month\'] + list(Cols)].groupby(\'Month\').sum()/1000,\n                      dtpart = "Month"\n                     );\n        \ncollect();\nprint();\n')


# <div style="color:'#050a14';display:fill;border-radius:5px;background-color:#d6e0f5;font-size:110%;font-family:Calibri;letter-spacing:0.5px">
# 
# <p style="padding: 10px; color:'#050a14'; font-weight: bold">
# Key inferences- 
# 
# 1. Sales are higher for all combinations across weekends, increasing from Wednesday- Sunday<br>
# 2. Monthly and weekly plots indicate December sales are high almost throughout<br>
# 3. Sales decline in the last week of the year is due to holiday period<br>
# </p>
# </div>

# <a id="4.4"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > TOTAL SALES CONTRIBUTION BY STORE<br><div> 
# 
# In this section, we compare the total periodic sales across individual stores to understand their contribution to the total.<br>
# Please refer the discussion post - https://www.kaggle.com/competitions/playground-series-s3e19/discussion/423654 for the cyclical daily sales graph.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef CalcStoreComb(store1, store2):\n    "This function calculates the monthly sales ratio across provided store ids";\n    \n    _ = \\\n    pd.DataFrame(np.sum(Sales_Prf[[\'Month\', \'Year\'] + list(Cols[Cols.str[1] == store1])].\\\n                        groupby([\'Year\',\'Month\']).sum(), axis=1)/np.sum(Sales_Prf[[\'Month\', \'Year\'] + list(Cols[Cols.str[1] == store2])].\\\n                                                                        groupby([\'Year\',\'Month\']).sum(), axis=1)).\\\n    reset_index();\n    \n    _ = \\\n    _.pivot(index= \'Month\', columns= \'Year\').\\\n    style.highlight_max(props= "color:red;fontweight:bold;background:lightgrey").\\\n    format(precision= 4).set_caption(f"Sales Ratio {store1}{store2}").\\\n    set_table_attributes("style=\'display:inline\'");\n\n    return _;\n    \ncollect();\n\n# Calculating store contribution ratios:-\nif CFG.ftre_plots_req == "Y":\n    DisplayAdjTbl(*[CalcStoreComb(store1 = "1", store2 = "2"),\n                    CalcStoreComb(store1 = "1", store2 = "3"),\n                    CalcStoreComb(store1 = "2", store2 = "3") \n                   ]);\n\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Calculating and displaying total daily sales rate:-\ndf = \\\npp.train.groupby([\'date\',\'product\'])[[CFG.target]].sum().reset_index().\\\njoin(pp.train.groupby(\'date\')[[CFG.target]].sum(), \n     on = \'date\',\n     rsuffix = \'_daily\'\n    );\n\ndf[\'SalesRate\'] = df[CFG.target]/df[f\'{CFG.target}_daily\'];\n\nfig, ax = plt.subplots(1,1, figsize = (16, 6));\nfor product in df[\'product\'].unique():\n    X = df[df[\'product\'] == product];\n    plt.plot(X[\'date\'], X[\'SalesRate\'], label = product);\n\nplt.title(f"\\nProduct level daily sales rate across model period\\n", **CFG.title_specs);\nplt.legend(bbox_to_anchor = (0.6, -0.10));\nplt.show();\n\ncollect();\nprint();\n')


# <a id="4.5"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > TOTAL PERIODIC SALES<br><div> 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfig, axes = plt.subplots(3,1, figsize=(15, 15), gridspec_kw = {"hspace": 0.2}, sharex = True);\n\nfor i, dtpart in enumerate([\'d\', "W", "MS"]):\n    _ = pp.train.groupby([pd.Grouper(key="date", freq= dtpart)])[CFG.target].sum().reset_index();\n    ax = axes[i];\n    sns.lineplot(data = _, x = "date", y= CFG.target, ax = ax);\n    ax.set_xticks(Date_Labels.date.values, labels = Date_Labels.date_lbl.values, rotation = 45);\n    ax.set(xlabel = \'\', ylabel = \'\');\n    ax.set_title(f"Grouped sales by {dtpart.upper()}", **CFG.title_specs);\n\nplt.suptitle(f"Total sales by date-part across all stores during training period", **CFG.title_specs, y = 0.925)\nplt.tight_layout();\nplt.show();\n\ncollect();\nprint();\n')


# <div style="color:'#050a14';display:fill;border-radius:5px;background-color:#b3ecff;font-size:110%;font-family:Calibri;letter-spacing:0.5px">
# 
# <p style="padding: 10px; color:'#ADD8E6'; font-weight: bold">
# Key inferences-<br>
# 1. Regime shifts in Dec2019 need to be observed.<br>
# 2. Covid impact in Q1-Q2 2020 are pronounced and significantly impact the model data - trend reversals and deviations are seen in this period compared to the pre-covid period<br>
# 3. December months inherently show spikes in sales<br>
# 4. Non-stationary behaviour in long run sales growth series needs to be factored in model development<br>
# 5. Feature interactions across stores, products and even countries may be worth noting for model development (Vector models)<br>
# 6. We have lots of synergy between sales of stores. In fact, sales of store 1 are almost always a fixed multiple of store 2 sales and store 3 sales<br>
# 7. Seasonality is seen in monthly sales and growth rates as evinced in the associated plots<br>
#     
# </p>
# </div>

# <a id="5"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > DATA TRANSFORMS<br><div> 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nclass Xformer(BaseEstimator, TransformerMixin):\n    def __init__(self, GDP_Snp = GDP_Snp):\n        self.GDP_Snp = GDP_Snp;\n\n    def fit(self, X, y= None, **params):\n        return self;\n\n    def transform(self, df: pd.DataFrame, y = None, **params):\n        "This method develops new features from the date column";\n\n        X = df.copy();\n        X["month"]     = X["date"].dt.month;\n        X["month_sin"] = np.sin(X[\'month\'] * (2 * np.pi / 12));\n        X["day"]       = df["date"].dt.day;\n        X["day_sin"]   = np.sin(X[\'day\'] * (2 * np.pi / 12));\n\n        X["day_of_week"] = df["date"].dt.dayofweek;\n        X["day_of_week"] = \\\n        X["day_of_week"].apply(lambda x: 0 if x<=3 else(1 if x==4 else (2 if x==5 else (3))));\n\n        X[\'is_friday\']   = X.date.dt.weekday.eq(4).astype(np.uint8);\n        X[\'is_saturday\'] = X.date.dt.weekday.eq(5).astype(np.uint8);\n        X[\'is_sunday\']   = X.date.dt.weekday.eq(6).astype(np.uint8);\n        X["day_of_year"] = df["date"].dt.dayofyear;\n\n        X["day_of_year"] = X.apply(lambda x: x["day_of_year"]-1 \n                                             if (x["date"] > pd.Timestamp("2020-02-29") \n                                                 and x["date"] < pd.Timestamp("2021-01-01"))  \n                                             else x["day_of_year"], axis=1\n                                            );\n        X["year"]        = df["date"].dt.year;\n\n        for day in range(24, 32):\n            X[f\'Dec{day}\'] = (X.date.dt.day.eq(day) & X.date.dt.month.eq(12)).astype(np.int8);\n\n        X         = pd.get_dummies(X, columns = ["day_of_week"], drop_first=True); \n        X         = X.merge(self.GDP_Snp, how = \'left\', on = [\'year\', \'country\'], suffixes = (\'\', \'\'));\n        X.columns = X.columns.str.replace(r"-","M", regex = True);\n\n        return X;\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nclass HolidayMapper(BaseEstimator, TransformerMixin):\n    def __init__(self):\n        pass;\n\n    def fit(self, X, y= None, **params):\n        self.years_list = pp.train.date.dt.year.unique().tolist() + pp.test.date.dt.year.unique().tolist();\n        self.holidays = {};\n        \n        for country in set(X.country):\n            self.holidays[country] = CountryHoliday(country, years = self.years_list);\n        return self;\n    \n    def transform(self, X, y = None):\n        df = X.copy();\n        df['holiday_name'] = df['date'].map(self.holidays)\n        df['is_holiday']   = np.where(df['holiday_name'].notnull(), 1, 0)\n        df['holiday_name'] = df['holiday_name'].fillna('NotHoliday')\n        return df;\n    \ncollect();\nprint();\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Performing spline transformation:-\n# Reference:- https://www.kaggle.com/code/chingiznurzhanov/timeseriessplit-catboost-trick\n\ndef DoSplineXform(period, n_splines = None, degree=3):\n    """\n    This function performs spline transform preprocessing on the provided data-frame\n    """;\n    \n    if n_splines is None: n_splines = period;\n    n_knots = n_splines + 1;\n    return SplineTransformer(degree=degree, n_knots=n_knots, \n                             knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),\n                             extrapolation="periodic",\n                             include_bias=True\n                            );\n\ndef MakeSplineFtre(hours = np.arange(1,32)):\n    """\n    This function makes spline features from the data\n    """;\n    \n    hour_df    = pd.DataFrame(np.linspace(1, 32, 32).reshape(-1, 1),columns= ["day"]);\n    splines    = DoSplineXform(32, n_splines=4).fit_transform(hour_df);\n    splines_df = pd.DataFrame(splines.values,columns=[f"spline{i}" for i in range(splines.shape[1])]);\n    splines_df = pd.concat([pd.Series(hours, name=\'day\'), splines_df], axis="columns");\n    \n    return splines_df;\n\ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nPrintColor(f"\\n {\'-\'*30} Data transforms {\'-\'*30}\\n", color = Fore.MAGENTA);\n\n# Implementing the data pipeline:-\nenc            = OE(cols= [\'holiday_name\', \'country\', \'store\', \'product\']);\nxform          = make_pipeline(Xformer(), HolidayMapper(), enc);\nXtrain, ytrain = pp.train.drop(columns = [CFG.target]), pp.train[CFG.target];\nxform.fit(Xtrain, ytrain);\nXtrain, Xtest  = xform.transform(Xtrain), xform.transform(pp.test);\n\nPrintColor(f"---> Training columns after the pipeline");\npprint(Xtrain.columns);\n\nPrintColor(f"---> Data shape after the pipeline = {Xtrain.shape} {Xtest.shape}");\n\n# Implementing spline transforms:-\ndf = pd.concat([Xtrain.assign(Source = "Train"), \n                Xtest.assign(Source = "Test")], \n               axis= 0, ignore_index = False\n              );\n\nPrintColor(f"---> Data shape for spline transforms = {df.shape}");\nSpline_Prf = MakeSplineFtre();\ndf = df.merge(Spline_Prf, on = \'day\', how = \'left\');\nXtrain, Xtest = (df.loc[df.Source == "Train"].drop(columns = [\'Source\']), \n                 df.loc[df.Source == "Test"].drop(columns = [\'Source\']));\n\nPrintColor(f"---> Data shape after the spline transforms = {Xtrain.shape} {Xtest.shape}");\nPrintColor(f"---> Training columns after spline addition");\npprint(Xtrain.columns);\n\ndel df;\ncollect();\n')


# <a id="6"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > MODEL TRAINING<br><div> 
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Initializing model I-O:-\nif CFG.ML == "Y":\n    Mdl_Master = \\\n    {\'CBR\': CBR(**{\'task_type\'           : "GPU" if CFG.gpu_switch == "ON" else "CPU",\n                   \'loss_function\'       : \'MAPE\',\n                   \'eval_metric\'         : \'MAPE\',\n                   \'objective\'           : \'MAPE\',\n                   \'random_state\'        : CFG.state,\n                   \'bagging_temperature\' : 1.0,\n                   \'colsample_bylevel\'   : 0.3,\n                   \'iterations\'          : 3000,\n                   \'learning_rate\'       : 0.055,\n                   \'od_wait\'             : 25,\n                   \'max_depth\'           : 7,\n                   \'l2_leaf_reg\'         : 1.20,\n                   \'min_data_in_leaf\'    : 20,\n                   \'random_strength\'     : 0.45, \n                   \'max_bin\'             : 400,\n                   \'use_best_model\'      : True, \n                  }\n               ),\n\n     \'LGBMR\': LGBMR(**{\'device\'            : "gpu" if CFG.gpu_switch == "ON" else "cpu",\n                       \'objective\'         : \'regression\',\n                       \'metric\'            : \'mape\',\n                       \'boosting_type\'     : \'gbdt\',\n                       \'random_state\'      : CFG.state,\n                       \'feature_fraction\'  : 0.65,\n                       \'learning_rate\'     : 0.075,\n                       \'max_depth\'         : 7,\n                       \'n_estimators\'      : 2000,\n                       \'num_leaves\'        : 120,                    \n                       \'reg_alpha\'         : 0.00001,\n                       \'reg_lambda\'        : 1.25,\n                       \'verbose\'           : -1,\n                      }\n                   ),\n\n     \'XGBR\': XGBR(**{\'objective\'          : \'reg:squarederror\',\n                     \'eval_metric\'        : \'mape\',\n                     \'random_state\'       : CFG.state,\n                     \'tree_method\'        : "gpu_hist" if CFG.gpu_switch == "ON" else "hist",\n                     \'colsample_bytree\'   : 0.95,\n                     \'subsample\'          : 0.65,\n                     \'learning_rate\'      : 0.05,\n                     \'max_depth\'          : 6,\n                     \'n_estimators\'       : 2000,                         \n                     \'reg_alpha\'          : 0.000001,\n                     \'reg_lambda\'         : 3.75,\n                     \'min_child_weight\'   : 30,\n                    }\n                 ),\n\n     \'HGBR\': HGBR(loss              = \'squared_error\',\n                  learning_rate     = 0.05215,\n                  max_iter          = 800,\n                  max_depth         = 7,\n                  min_samples_leaf  = 20,\n                  l2_regularization = 1.25,\n                  max_bins          = 255,\n                  n_iter_no_change  = 75,\n                  tol               = 1e-04,\n                  verbose           = 0,\n                  random_state      = CFG.state\n                 ),\n     \n     "RFR" : RFR(n_estimators     = 250,\n                 criterion        =\'squared_error\',\n                 max_depth        = 7,\n                 min_samples_leaf = 50,\n                 max_features     = 1.0,\n                 bootstrap        = True,\n                 oob_score        = True,\n                 n_jobs           = -1,\n                 random_state     = CFG.state,\n                 verbose          = 0,\n                ),\n    };\n\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n    # Selecting relevant columns for the train and test sets:-\n    \n    if CFG.remove_covid_mth == "Y":\n        PrintColor(f"--> Removing covid period from the training data", color = Fore.RED);\n        df = \\\n        pd.concat([Xtrain, ytrain], axis=1).\\\n        loc[~(Xtrain.date.between(pd.to_datetime("01-03-2020"), pd.to_datetime("05-31-2020")))].\\\n        drop(columns = CFG.drop_ftre, errors = \'ignore\');\n        Xtrain = df.drop(columns = [CFG.target]);\n        ytrain = df[CFG.target];\n        del df;\n        \n    else:\n        PrintColor(f"--> Retaining covid period in the training data", color = Fore.RED);\n        Xtrain = Xtrain.drop(columns = CFG.drop_ftre, errors = \'ignore\');\n        \n    Xtest = Xtest.drop(columns = CFG.drop_ftre, errors = \'ignore\');\n    Xtrain.index = range(len(Xtrain));\n    Xtest.index  = range(len(Xtest));\n    \n    PrintColor(f"\\n---> Training columns for the model");\n    pprint(Xtrain.columns);\n        \n    # Initializing output tables for the models:-\n    methods   = deepcopy(CFG.methods);\n    OOF_Preds = pd.DataFrame(columns = methods);\n    Mdl_Preds = pd.DataFrame(index = Xtest.index, columns = methods);\n    FtreImp   = pd.DataFrame(index = Xtrain.drop(columns = [\'id\', CFG.target, \'Source\', \'Label\', \'date\', \'year\'],\n                                                 errors = \'ignore\').columns, \n                             columns = methods\n                            );\n    Scores       = pd.DataFrame(columns = methods);\n    Scores_Train = pd.DataFrame(columns = methods);\n\n\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef TrainMdl(method:str):\n    "This function trains the regression models and collates the scores and predictions";\n    \n    global Mdl_Master, Mdl_Preds, OOF_Preds, all_cv, FtreImp, Xtrain, ytrain, Scores_Train, Scores; \n    \n    model     = Mdl_Master.get(method); \n    cols_drop = [\'id\', \'Source\', \'Label\', \'year\', \'date\'];\n    scl_cols  = [col for col in Xtrain.columns if col not in cols_drop];\n    cv        = all_cv[CFG.mdlcv_mthd];\n    Xt        = Xtest.copy(deep = True);\n    \n    if CFG.scale_req == "N" and method.upper() in [\'RIDGER\', \'LASSOR\', \'SVR\', \'KNNR\']:\n        X, y        = Xtrain.copy(deep = True), ytrain.copy(deep = True);\n        scaler      = all_scalers[CFG.scl_method];\n        X[scl_cols] = scaler.fit_transform(X[scl_cols].values);\n        Xt[scl_cols]= scaler.transform(Xt[scl_cols].values);\n        PrintColor(f"--> Scaling the data for {method} model");\n \n    X,y = Xtrain.copy(deep = True), ytrain.copy(deep = True);\n                \n    # Initializing I-O for the given seed:-        \n    test_preds = 0;\n    oof_preds  = pd.DataFrame(); \n    scores     = [];\n    scores_tr  = [];\n    ftreimp    = 0;\n           \n    for fold_nb, yr in enumerate(X.year.unique()): \n        Xtr  = X.loc[X.year != yr].drop(columns = cols_drop, errors = \'ignore\');   \n        Xdev = X.loc[X.year == yr].drop(columns = cols_drop, errors = \'ignore\'); \n        ytr  = y.loc[y.index.isin(Xtr.index)];\n        ydev = y.loc[y.index.isin(Xdev.index)];\n        \n        # Fitting the model:- \n        if method.upper() in [\'CBR\', \'LGBMR\', \'XGBR\']:     \n            model.fit(Xtr, ytr, \n                      eval_set = [(Xdev, ydev)], \n                      verbose = 0,\n                      early_stopping_rounds = CFG.nbrnd_erly_stp\n                     );   \n        else:\n            model.fit(Xtr, ytr);\n            \n        # Collecting predictions and scores and post-processing OOF:-\n        # Collating OOF period scores:- \n        dev_preds = model.predict(Xdev);\n        score     = ScoreMetric(ydev, dev_preds);\n        scores.append(score); \n        Scores.loc[fold_nb, method] = np.round(score, decimals= 6);\n        oof_preds = pd.concat([oof_preds,\n                               pd.DataFrame(index   = Xdev.index, \n                                            data    = dev_preds,\n                                            columns = [method])\n                              ],axis=0, ignore_index= False\n                             );  \n    \n        oof_preds = pd.DataFrame(oof_preds.groupby(level = 0)[method].mean());\n        oof_preds.columns = [method];\n        \n        # Collating train period scores:-      \n        train_preds = model.predict(Xtr);\n        score       = ScoreMetric(ytr, train_preds);\n        scores_tr.append(score); \n        Scores_Train.loc[fold_nb, method] = np.round(score, decimals= 6);\n        \n        # Collating test period predictions:- \n        test_preds =\\\n        test_preds + model.predict(Xt.drop(columns = cols_drop, errors = \'ignore\')); \n        \n        # Collating feature importances:-\n        try: \n            ftreimp += model.feature_importances_;\n        except:\n            ftreimp = 0;\n            \n    num_space = 20 - len(method);\n    PrintColor(f"--> {method} {\'-\' * num_space} OOF-> {np.mean(scores):.6f} Train-> {np.mean(scores_tr):.6f}\\n", \n               color = Fore.BLUE);  \n    del num_space;\n    \n    OOF_Preds[f\'{method}\'] = oof_preds.values.flatten();\n    Mdl_Preds[f\'{method}\'] = test_preds.flatten()/ CFG.n_splits; \n    FtreImp[method]        = ftreimp / CFG.n_splits;\n    collect(); \n      \ncollect();\nprint();\n\n# Implementing the ML models:-\nif CFG.ML == "Y": \n    for method in tqdm(methods, f"---- {CFG.mdlcv_mthd} CV ----"): \n        TrainMdl(method);\n    \n    clear_output();  \n    PrintColor(f"\\n ----- ML models - {CFG.methods} -----\\n");\n    \n    DisplayAdjTbl(*[Scores.style.highlight_max(props= "color:red;fontweight:bold;background:lightgrey").\\\n                    format(precision= 5).\\\n                    set_caption(f"OOF score").\\\n                    set_table_attributes("style=\'display:inline\'"),\n                    \n                    Scores_Train.style.\\\n                    highlight_max(props= "color:blue; fontweight: bold; background: #f0f5f5").\\\n                    format(precision= 5).\\\n                    set_caption(f"Training score").\\\n                    set_table_attributes("style=\'display:inline\'")\n                   ]\n                 );\n    \n    OOF_Preds.clip(lower = 0, upper = np.inf, inplace = True,);\n    Mdl_Preds.clip(lower = 0, upper = np.inf, inplace = True,);\n    \nelse:\n    PrintColor(f"\\nML models are not needed\\n", color = Fore.RED);\n    \ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Analysing feature importances and regression plots:-\nif CFG.ML == "Y":\n    try: del col, method;\n    except: pass;\n    \n    df = pd.concat([OOF_Preds, ytrain], axis=1);\n\n    fig, axes = plt.subplots(len(methods), 2, figsize = (25, len(methods) * 7.5),\n                             gridspec_kw = {\'hspace\': 0.2, \'wspace\': 0.2},\n                             width_ratios= [0.75, 0.25],\n                            );\n\n    for i, method in enumerate(methods):\n        ax = axes[i, 0];  \n        FtreImp[f"{method}"].plot.barh(ax = ax, color = \'#0073e6\');\n        ax.set_title(f"{method} Importances", **CFG.title_specs);\n        ax.set(xlabel = \'\', ylabel = \'\');\n\n        ax = axes[i,1];\n        rsq = r2_score(df[CFG.target].values, df[method].values);\n\n        sns.regplot(data= df, y = method, x = f"{CFG.target}", \n                    seed= CFG.state, color = \'#dce6f5\', marker = \'o\',\n                    line_kws= {\'linewidth\': 2.25, \'linestyle\': \'--\', \'color\': \'black\'},\n                    label = f"{method}",\n                    ax = ax,\n                   );\n        ax.set_title(f"{method} RSquare = {rsq:.2%}", **CFG.title_specs);\n        ax.set(ylabel = \'Predictions\', xlabel = \'Actual\')\n        del rsq;           \n\n    plt.suptitle(f"ML model analysis after training", \n                 color = \'blue\', fontweight = \'bold\', fontsize = 14, \n                 y = 0.91,\n                );\n    plt.tight_layout();\n    plt.show();\n    \ncollect();\nprint();\n')


# <a id="7"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > ENSEMBLE AND SUBMISSION<br><div> 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef Objective(trial):\n    "This function defines the objective for the optuna ensemble using variable models";\n    \n    global OOF_Preds, all_cv, ScoreMetric, Xtrain;\n    \n    # Define the weights for the predictions from each model:-\n    opt_ens_mdl = list(OOF_Preds.drop(columns = [CFG.target], errors = \'ignore\').columns);\n    weights  = [trial.suggest_float(f"M{n}", 0.005, 0.995, step = 0.001) \\\n                for n in range(len(opt_ens_mdl))];\n    weights  = np.array(weights)/ sum(weights);\n    \n    # Calculating the CV-score for the weighted predictions on the competition data:-\n    scores = [];  \n    cv     = all_cv[CFG.mdlcv_mthd];\n    X,y    = pd.concat([OOF_Preds[opt_ens_mdl], Xtrain[[\'year\']]], axis=1), ytrain.copy(deep = True);\n    \n    for fold_nb, yr in enumerate(Xtrain.year.unique()):\n        Xtr, Xdev = X.loc[X.year != yr].drop(columns = [\'year\']), X.loc[X.year == yr].drop(columns = [\'year\']);\n        ytr, ydev = y.loc[Xtr.index], y.loc[Xdev.index];\n        scores.append(ScoreMetric(ydev, np.average(Xdev, weights = weights, axis=1)));\n        \n    collect();\n    clear_output();\n    return np.mean(scores);\n\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ensemble_req == "Y":\n    PrintColor(f"{\'-\'* 30} Ensemble with optuna {\'-\'* 30}", color = Fore.MAGENTA);\n    \n    opt_ens_mdl = list(OOF_Preds.drop(columns = [CFG.target], errors = \'ignore\').columns);\n    study = optuna.create_study(direction  = CFG.direction, \n                                study_name = "OptunaEnsemble", \n                                sampler    = TPESampler(seed = CFG.state)\n                               );\n    study.optimize(Objective, \n                   n_trials          = CFG.n_ens_trials, \n                   gc_after_trial    = True,\n                   show_progress_bar = True);\n    \n    weights       = study.best_params;\n    clear_output();\n    \n    PrintColor(f"\\n--> Post ensemble weights\\n");\n    pprint(weights, indent = 5, width = 10, depth = 1);\n    \n    PrintColor(f"\\n--> Best ensemble CV score = {study.best_value :.5f}\\n");\n  \n    # Making weighted predictions on the test set:-\n    sub_fl = pp.sub_fl.copy(deep = True);\n    sub_fl[sub_fl.columns[-1]] = np.average(Mdl_Preds[opt_ens_mdl], \n                                            weights = list(weights.values()), \n                                            axis=1\n                                           );\n    \n    # Post-processing predictions with a multiplier/ additive factor:-\n    if CFG.pst_prcess_mthd == "mult":\n        sub_fl[sub_fl.columns[-1]] = sub_fl[sub_fl.columns[-1]].values * CFG.pst_prcess_fct;\n    elif CFG.pst_prcess_mthd == "add":\n        sub_fl[sub_fl.columns[-1]] = sub_fl[sub_fl.columns[-1]].values + CFG.pst_prcess_fct;\n    else:\n        pass;\n        \n    PrintColor(f"\\n--> Post ensemble test-set predictions\\n");\n    display(sub_fl.head(5).style.format(precision = 4));  \n    \n    sub_fl.to_csv(f"submission_V{CFG.version_nb}.csv", index = None);\n         \ncollect();\nprint();    \n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":  \n    OOF_Preds.add_prefix(f"V{CFG.version_nb}_").to_csv(f"OOF_Preds_V{CFG.version_nb}.csv");\n    Mdl_Preds.index = pp.sub_fl.id;\n    Mdl_Preds.add_prefix(f"V{CFG.version_nb}_").to_csv(f"Mdl_Preds_V{CFG.version_nb}.csv");\n    Scores.merge(Scores_Train, how = \'left\', \n                 left_index = True, \n                 right_index = True,\n                 suffixes = (\'_OOF\', "_Train")\n                ).add_prefix(f"V{CFG.version_nb}_").\\\n    to_csv(f"Scores_V{CFG.version_nb}.csv");\n    \ncollect();\nprint();\n')


# <a id="8"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > OUTRO<br><div> 

# **References**<br>
# 1. https://www.kaggle.com/code/chingiznurzhanov/timeseriessplit-catboost-trick
# 2. https://www.kaggle.com/competitions/playground-series-s3e19/discussion/423654 
# 
# **Next steps**<br>
# 
# 1. Improve feature engineering<br>
# 2. Try and improve the CV strategy<br>
# 3. Use more models in the CV ensemble<br>
# 4. Use a better ensemble strategy<br>
# 
