#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------
# ------------------------------------------
# 
# # <p style="background-color:#CCE3F2; font-family:newtimeroman; font-size:120%; text-align:center; border-radius: 15px 50px;">Object Oriented Programming (OOP) approach for <br>Data Science problems.</p>
# 
# 
# # <p style="background-color:gray; font-family:newtimeroman; font-size:150%; text-align:center; border-radius: 10px 100px; color:black; hight:max"> Upvote my work if you found it useful.🎯 </p>
# 
# ------------------------------------------
# ------------------------------------------
# 

# <p style="background-color:#CCE3F2; font-family:newtimeroman; font-size:200%; text-align:center; border-radius: 10px 500px;"><b>Introduction
#     </b>
# </p>
# <b>The objective of this project is to apply Object Oriented Programming(OOP) approach for Data Science problems.<br>
# Object oriented programming is largely based on personal experience and is open to development. For this reason, code design can be improved according to the comments to be made for the kernel. Comments and criticisms will provide a better code design.<br>
# This project consists of five classes:</b>
# <ul>
#     <li>Class I   : <a href="#hpopp_class"><b>HouseObjectOriented Class.</b></a></li>
#     <ul>
#         <li>1) Adds the data to the object.
#         <li>2) Concats the data in one DataFrame.
#         <li>3) Shows information about the data.
#         <li>4) Preprocess the data before Ml part.
#         <li>5) Adds the data after Preprocessing for the ML part.</li>
#     </ul><br>
#     <li>Class II  : <a href="#info_class"><b>Information Class.</b></a></li>
#     <ul>
#         <li>1) Calculates the missing values.
#         <li>2) Gets feature dtypes.
#         <li>3) Gets feature names.
#         <li>4) Gets shape of the data.
#         <li>5) Prints all of these information.</li>
#     </ul><br>
#     <li>Class III : <a href="#pre_process_class"><b>Pre-processing Class.</b></a></li>
#     <ul>
#         <li>1) Drops the unwanted columns and rows.
#         <li>2) Fills the null values with (mean, meadian, zero,....etc).
#         <li>3) Applys feature engineering to the data like adding new columns, transforming columns,...etc.
#         <li>4) Encodes the data using label encoder to be able to apply ML algorithms.
#         <li>5) Converts your data to dummies values.
#         <li>6) Normalizes the data before ML.</li>
#     </ul><br>
#     <li>Class IV  : <a href="#processor_class"><b>Preprocessor Class</b></a></li>
#     <ul>
#         <li>1) Applys the Pre_processing techniques and returns the new data.</li>
#     </ul><br>
#     <li>Class V   : <a href="#ml_class"><b>ML Class</b></a></li>
#     <ul>
#         <li>1) Initializes the ML algorithms.
#         <li>2) Show the available ML algorithms.
#         <li>3) Applys Train-Test evaluation and shows the results.
#         <li>4) Applys Cross-Validation evaluation and shows the results.
#         <li>5) Visualizes the results of Train-Test and Cross-Validation evaluations.
#         <li>6) Find the best model and then fits it to the data.
#         <li>7) Show the Predictions in a DataFrame.
#         <li>8) Save the predictions to a csv file.</li>
#     </ul>
# </ul>
# <br>
# <p style="background-color:#CCE3F2; font-family:newtimeroman; font-size:200%; text-align:center; border-radius: 10px 100px;"><a id="outlines">Outlines : </a></p>
# <ul>
#     <li><a href="#1.0"><b>Create HouseObjectOriented object</b></a>
#     <li><a href="#1.1"><b>Adding our data</b></a>
#     <li><a href="#2.0"><b>Display Information about the data</b></a>
#     <li><a href="#3.0"><b>Pre-Process the data</b></a>
#     <li><a href="#2.1"><b>Display Information about the data after Pre-Processing</b></a>
#     <li><a href="#4.0"><b>Create a Machine Learning object</b></a>
#     <li><a href="#4.1"><b>Show the available algorithms</b></a>
#     <li><a href="#4.2"><b>Initialize the ML Regressors</b></a>
#     <li><a href="#4.3"><b>Train-Test Validation</b></a>
#     <li><a href="#4.4"><b>Visualize the results of train-test validation</b></a>
#     <li><a href="#4.5"><b>Applying Cross-Validation</b></a>
#     <li><a href="#4.6"><b>Visualize the results of Cross-Validation</b></a>
#     <li><a href="#4.7"><b>Find the best model and fit it to the data</b></a>
#     <li><a href="#4.8"><b>Predict and show the prediction</b></a>
#     <li><a href="#4.9"><b>Save the predictions to a csv file</b></a></li>
# </ul>

# <b><a href='https://www.kaggle.com/serkanpeldek/object-oriented-titanics'>Recommended Notebook</a>   

# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:100%; text-align:center; border-radius: 15px 50px;">Importing necessary modules and libraries📚</p>

# In[1]:


#main libraries
import os
import numpy as np
import pandas as pd
import warnings

#visualization libraries
import plotly 
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode
import cufflinks as cf

#machine learning libraries:
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, train_test_split, KFold, cross_val_score
from sklearn.preprocessing  import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# You can go offline on demand by using
cf.go_offline() 

# initiate notebook for offline plot
init_notebook_mode(connected=False)         

# set some display options:
colors = px.colors.qualitative.Prism
pio.templates.default = "plotly_white"

# see our files:
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# In[2]:


warnings.filterwarnings('ignore')
print("Warnings were ignored")


# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:100%; text-align:center; border-radius: 15px 50px;">Creating our Classes 🐍</p>

# <a id="info_class"></a>
# <h1>Information Class</h1>

# In[3]:


class Information:
    """
    This class shows some information about the dataset
    """
    def __init__(self):
        
        print()
        print('Information object is created')
        print()
        
    def get_missing_values(self, data):
        """
        This function finds the missing values in the dataset
        ...
        Attributes
        ----------
        data : Pandas DataFrame
        The data you want to see information about
        
        Returns
        ----------
        A Pandas Series contains the missing values in descending order
        """
        #get the sum of all missing values in the dataset
        missing_values = data.isnull().sum()
        #sorting the missing values in a pandas Series
        missing_values = missing_values.sort_values(ascending=False)
        
        #returning the missing values Series
        return missing_values
    
    def _info_(self, data):
        """
        This function shows some information about the data like 
        Feature names,data type, number of missing values for each feature 
        and ten samples of each feature
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            The data you want to see information about
        
        Returns
        ----------
        Information about the DataFrame
        """
        self.data=data
        feature_dtypes=self.data.dtypes
        self.missing_values=self.get_missing_values(self.data)
        feature_names=self.missing_values.index.values
        missing_values=self.missing_values.values
        rows, columns=data.shape

        print("=" * 50)
        print('====> This data contains {} rows and {} columns'.format(rows,columns))
        print("=" * 50)
        print()
        
        print("{:13} {:13} {:30} {:15}".format('Feature Name'.upper(),
                                               'Data Format'.upper(),
                                               'Null values(Num-Perc)'.upper(),
                                               'Seven Samples'.upper()))
        for feature_name, dtype, missing_value in zip(feature_names,feature_dtypes[feature_names],missing_values):
            print("{:15} {:14} {:20}".format(feature_name,
                                             str(dtype), 
                                             str(missing_value) + ' - ' + 
                                             str(round(100*missing_value/sum(self.missing_values),3))+' %'), end="")

            for i in np.random.randint(0,len(data),7):
                print(data[feature_name].iloc[i], end=",")
            print()

        print("="*50)


# <a href="#outlines"><b>Up to outlines</b>

# <a id="pre_process_class"></a>
# <h1>Data pre-processing Class</h1>

# In[4]:


class Pre_processing:
    """
    This class prepares the data berfore applying ML
    """
    def __init__(self):
        
        print()
        print('pre-processing object is created')
        print()        
        
    def drop(self, data, drop_strategies):
        """
        This function is used to drop a column or row from the dataset.
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            The data you want to drop data from.
        drop_strategies : A list of tuples, each tuple has the data to drop,
        and the axis(0 or 1)
        
        Returns
        ----------
        A new dataset after dropping the unwanted data.
        """
        
        self.data=data
        
        for columns, ax in drop_strategies:
            if len(columns)==1:
                self.data=self.data.drop(labels=column, axis=ax)
            else:
                for column in columns:
                    self.data=self.data.drop(labels=column, axis=ax)
        return self.data

    def fillna(self, ntrain, fill_strategies):       
        """
        This function fills NA/NaN values in a specific column using a specified method(zero,mean,...)
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            The data you want to impute its missing values
        fill_strategies : A dictionary, its keys represent the columns, 
        and the values represent the value to use to fill the Nulls.
        
        Returns
        ----------
        A new dataset without null values.
        """
        def fill(column, fill_with):
            
                if str(fill_with).lower() in ['zero', 0]:
                    self.data[column].fillna(0, inplace=True)
                elif str(fill_with).lower()=='mode':
                    self.data[column].fillna(self.data[column].mode()[0], inplace=True)
                elif str(fill_with).lower()=='mean':
                    self.data[column].fillna(self.data[column].mean(), inplace=True)
                elif str(fill_with).lower()=='median':
                    self.data[column].fillna(self.data[column].median(), inplace=True)
                else:
                    self.data[column].fillna(fill_with, inplace=True)

                return self.data
            
        #LotFrontage: Linear feet of street connected to property
        self.data['LotFrontage'] = self.data.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median())).values

        # Meaning that NO Masonry veneer
        self.data['MSZoning'] = self.data['MSZoning'].transform(lambda x: x.fillna(x.mode().values[0]))

        #imputing columns according to its strategy
        for columns, strategy in fill_strategies:
            if len(columns)==1:
                fill(columns[0], strategy)
            else:
                for column in columns:
                    fill(column, strategy)

        return self.data
    
    def feature_engineering(self):
        """
        This function is used to apply some feature engineering on the data.
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            The data you want to apply feature engineering on.
        
        Returns
        ----------
        A new dataset with new columns and some additions.
        """
        # creating new columns
        self.data['TotalSF'] = self.data['TotalBsmtSF'] + self.data['1stFlrSF'] + self.data['2ndFlrSF']
                
        # Convert some columns from numeric to string
        self.data[['YrSold','MSSubClass','MoSold','OverallCond']] = self.data[['YrSold','MSSubClass','MoSold','OverallCond']].astype(str)
        
        # Convert some columns from numeric to int
        self.data[['BsmtHalfBath','BsmtFinSF1', 'BsmtFinSF2','BsmtFullBath','BsmtUnfSF','GarageCars','GarageArea']]\
        =self.data[['BsmtHalfBath','BsmtFinSF1', 'BsmtFinSF2','BsmtFullBath','BsmtUnfSF','GarageCars','GarageArea']].astype(int)

        return self.data    
   
    def label_encoder(self, columns):
        """
        This function is used to encode the data to categorical values to benefit from increasing or 
        decreasing to build the model    
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            The data you want to encode.
        columns : columns to convert.
        
        Returns
        ----------
        A dataset without categorical data.
        """

        # Convert all categorical collumns to numeric values
        lbl = LabelEncoder() 
        
        self.data[columns] = self.data[columns].apply(lambda x:lbl.fit_transform(x.astype(str)).astype(int))
        
        return self.data 
    
    def get_dummies(self, columns):
        """
        This function is used to convert the data to dummies values.
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            The data you want to convert.
        
        Returns
        ----------
        A dataset with dummies.
        """
        
        # convert our categorical columns to dummies
        for col in columns:
            dumm = pd.get_dummies(self.data[col], prefix = col, dtype=int)
            self.data = pd.concat([self.data, dumm], axis=1)

        self.data.drop(columns, axis=1, inplace=True)
        
        return self.data
        
    def norm_data(self, columns):
        """
        This function is used to normalize the data.   
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            The data you want to normalize.
        
        Returns
        ----------
        A new normalized dataset.
        """
        
        # Normalize our numeric data
        self.data[columns] = self.data[columns].apply(lambda x:np.log1p(x)) #Normalize the data with Logarithms
        
        return self.data      


# <a href="#outlines"><b>Up to outlines</b>

# # <h1>Processor Class</h1>
# <a id="processor_class"></a>

# In[5]:


class Preprocessor:
    
    def __init__(self):
        self.data=None
        self._preprocessor=Pre_processing()

    def _process(self, data, ntrain):

        self.data=data
        
        self.ntrain=ntrain
        
        cols_drop=['Utilities', 'OverallQual','TotRmsAbvGrd']
        
        # Numeric columns
        num_cols = ['LotFrontage','LotArea','YearBuilt','YearRemodAdd','MasVnrArea',
                    'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
                    'LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath', 
                    'BedroomAbvGr','KitchenAbvGr','Fireplaces','GarageYrBlt','GarageCars','GarageArea', 
                    'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']

        # Categorical columns
        cat_cols = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','LotConfig','LandSlope', 
                    'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl', 
                    'Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual', 
                    'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir', 
                    'Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish', 
                    'GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','MoSold','SaleType', 
                    'SaleCondition','OverallCond', 'YrSold']
        
        drop_strategies=[(cols_drop,1)]

        fill_strategies=[(['BsmtFinType2','BsmtQual','BsmtCond','ExterQual','ExterCond','MasVnrArea',
                          'TotalBsmtSF','HeatingQC','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2','GarageYrBlt',
                          'BsmtFullBath','BsmtUnfSF','GarageCars','GarageArea','MasVnrArea'],0),
                         (['FireplaceQu','GarageQual','GarageCond','BsmtFinType1','MasVnrType',
                          'BsmtExposure','GarageFinish','PoolQC','Fence','LandSlope','GarageType',
                          'LotShape','PavedDrive','Street','Alley','CentralAir','MiscFeature',
                          'MSSubClass','OverallCond','YrSold','MoSold'],'NA'),
                         (['Functional'],'Typ'), # Typical Functionality
                         (['KitchenQual'],'TA'),
                         (['LotFrontage'],'median'),
                         (['MSZoning'],'mode'),
                         (['SaleType','Exterior1st','Exterior2nd','SaleType'],'Oth'), #other
                         (['Electrical'],'SBrkr')]  # Standard Circuit Breakers & Romex

        #drop
        self.data = self._preprocessor.drop(self.data, drop_strategies)
        
        #fill nulls
        self.data = self._preprocessor.fillna(self.ntrain, fill_strategies)
        
        #feature engineering
        self.data = self._preprocessor.feature_engineering()
        
        #label encoder
        self.data = self._preprocessor.label_encoder(cat_cols)
        
        #normalizing
#         self.data = self._preprocessor.norm_data(self.data, num_cols)
        
        #get dummies
        self.data = self._preprocessor.get_dummies(cat_cols)
        return self.data


# <a href="#outlines"><b>Up to outlines</b>

# <h1>Machine Learning Class</h1>
# <a id="ml_class"></a>

# In[6]:


class ML: 
    def __init__(self, data, ytrain, testID, test_size, ntrain):
         
        print()
        print('Machine Learning object is created')
        print()

        self.data=data
        self.ntrain=ntrain
        self.test_size=test_size
        self.train=self.data[:self.ntrain]
        self.test=self.data[self.ntrain:]
        self.testID=testID
        self.ytrain=ytrain
        
        self.reg_models={}

        # define models to test:
        self.base_models = {
            "Elastic Net":make_pipeline(RobustScaler(),                   #Elastic Net model(Regularized model)
                                        ElasticNet(alpha=0.0005,
                                                   l1_ratio=0.9)),
            "Kernel Ridge" : KernelRidge(),                               #Kernel Ridge model(Regularized model)
            "Bayesian Ridge" : BayesianRidge(compute_score=True,          #Bayesian Ridge model
                                            fit_intercept=True,
                                            n_iter=200,
                                            normalize=False),                             
            "Lasso" : make_pipeline(RobustScaler(), Lasso(alpha =0.0005,   #Lasso model(Regularized model)
                                                          random_state=2021)),
            "Lasso Lars Ic" : LassoLarsIC(criterion='aic',                  #LassoLars IC model 
                                        fit_intercept=True,
                                        max_iter=200,
                                        normalize=True,
                                        precompute='auto',
                                        verbose=False), 
            "Random Forest": RandomForestRegressor(n_estimators=300),      #Random Forest model
            "Svm": SVR(),                                                  #Support Vector Machines
            "Xgboost": XGBRegressor(),                                     #XGBoost model                                             
            "Gradient Boosting":make_pipeline(StandardScaler(),
                                             GradientBoostingRegressor(n_estimators=3000, #GradientBoosting model
                                                                       learning_rate=0.005,     
                                                                       max_depth=4, max_features='sqrt',
                                                                       min_samples_leaf=15, min_samples_split=10, 
                                                                       loss='huber', random_state = 2021))}
        
    def init_ml_regressors(self, algorithms):
        
        if algorithms.lower()=='all':
            for model in self.base_models.keys():
                self.reg_models[model.title()]=self.base_models[model.title()]
                print(model.title(),(20-len(str(model)))*'=','>','Initialized')
            
        else:
            for model in algorithms:
                if model.lower() in [x.lower() for x in self.base_models.keys()]:
                    print(self.base_models[model])
                    print(model.title(),(20-len(str(model)))*'=','>','Initialized')

                else:
                    print(model.title(),(20-len(str(model)))*'=','>','Not Initialized')
                    print('# Only (Elastic Net,Kernel Ridge,Lasso,Random Forest,SVM,XGBoost,LGBM,Gradient Boosting,Linear Regression)')    
    

    def show_available(self):
        print(50*'=')
        print('You can fit your data with the following models')
        print(50*'=','\n')
        for model in [m.title() for m in self.base_models.keys()]:
            print(model)
        print('\n',50*'=','\n')
        
    def train_test_eval_show_results(self, show=True):
        
        if not self.reg_models:
            raise TypeError('Add models first before fitting')
      
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.train, self.ytrain, 
                                                                                test_size=self.test_size, random_state=2021)

        #Preprocessing, fitting, making predictions and scoring for every model:
        self.result_data = {'R^2':{'Training':{},'Testing':{}},
                            'Adjusted R^2':{'Training':{},'Testing':{}},
                            'MAE':{'Training':{},'Testing':{}},
                            'MSE':{'Training':{},'Testing':{}},
                            'RMSE':{'Training':{},'Testing':{}}}
        
        self.p = train.shape[1]
        self.train_n = self.X_train.shape[0]
        self.test_n = self.X_test.shape[0]
        
        for name in self.reg_models: 
            #fitting the model
            model = self.reg_models[name].fit(self.X_train, self.y_train)
            
            #make predictions with train and test datasets
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)

            #calculate the R-Squared for training and testing
            r2_train,r2_test = model.score(self.X_train, self.y_train),\
                               model.score(self.X_test, self.y_test)
            self.result_data['R^2']['Training'][name],\
            self.result_data['R^2']['Testing'][name] = r2_train, r2_test

            #calculate the Adjusted R-Squared for training and testing
            adj_train, adj_test = (1-(1-r2_train)*(self.train_n-1)/(self.train_n-self.p-1)) ,\
                                  (1-(1-r2_test)*(self.train_n-1)/(self.train_n-self.p-1))
            self.result_data['Adjusted R^2']['Training'][name],\
            self.result_data['Adjusted R^2']['Testing'][name] = adj_train, adj_test

            #calculate the Mean absolute error for training and testing
            mae_train, mae_test = mean_absolute_error(self.y_train, y_pred_train),\
                                  mean_squared_error(self.y_test, y_pred_test)         
            self.result_data['MAE']['Training'][name],\
            self.result_data['MAE']['Testing'][name] = mae_train, mae_test

            #calculate Mean square error for training and testing
            mse_train, mse_test = mean_squared_error(self.y_train, y_pred_train),\
                                  mean_squared_error(self.y_test, y_pred_test)
            self.result_data['MSE']['Training'][name],\
            self.result_data['MSE']['Testing'][name] = mse_train, mse_test

            #calculate Root mean error for training and testing    
            rmse_train, rmse_test = np.sqrt(mse_train), np.sqrt(mse_test)
            self.result_data['RMSE']['Training'][name],\
            self.result_data['RMSE']['Testing'][name] = rmse_train, rmse_test
            
            if show:
                print('\n',25*'=','{}'.format(name),25*'=')
                print(10*'*','Training',23*'*','Testing',10*'*')
                print('R^2    : ',r2_train,' '*(25-len(str(r2_train))),r2_test) 
                print('Adj R^2: ',adj_train,' '*(25-len(str(adj_train))),adj_test) 
                print('MAE    : ',mae_train,' '*(25-len(str(mae_train))),mae_test) 
                print('MSE    : ',mse_train,' '*(25-len(str(mse_train))),mse_test) 
                print('RMSE   : ',rmse_train,' '*(25-len(str(rmse_train))),rmse_test)
 
    def cv_eval_show_results(self, num_models=4, n_folds=5, show=False):
        
        # prepare configuration for cross validation test
        #Create two dictionaries to store the results of R-Squared and RMSE 
        self.r_2_results = {'R-Squared':{},'Mean':{},'std':{}}   
        self.rmse_results = {'RMSE':{},'Mean':{},'std':{}}
        
        #create a dictionary contains best Adjusted R-Squared results, then sort it
        adj=self.result_data['Adjusted R^2']['Testing']
        adj_R_sq_sort=dict(sorted(adj.items(), key=lambda x:x[1], reverse=True))
        
        #check the number of models to visualize results
        if str(num_models).lower()=='all':
            models_name={i:adj_R_sq_sort[i] for i in list(adj_R_sq_sort.keys())}
            print()
            print('Apply Cross-Validation for {} models'.format(num_models))
            print()
            
        else:
            print()
            print('Apply Cross-Validation for {} models have highest Adjusted R-Squared value on Testing'.format(num_models))
            print()
            
            num_models=min(num_models,len(self.base_models.keys()))
            models_name={i:adj_R_sq_sort[i] for i in list(adj_R_sq_sort.keys())[:num_models]}
        
        models_name=dict(sorted(models_name.items(), key=lambda x:x[1], reverse=True))
        
        #create Kfold for the cross-validation
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=2021).get_n_splits(self.train)
        
        
        for name,_ in models_name.items():
            model = self.base_models[name]
            r_2 = cross_val_score(model, self.train, self.ytrain,    #R-Squared 
                                  scoring='r2', cv=kfold)          
            rms = np.sqrt(-cross_val_score(model, self.train, self.ytrain, #RMSE
                                           cv=kfold, scoring='neg_mean_squared_error'))

            #save the R-Squared reults
            self.r_2_results['R-Squared'][name] = r_2
            self.r_2_results['Mean'][name] = r_2.mean()
            self.r_2_results['std'][name] = r_2.std()

            #save the RMSE reults
            self.rmse_results['RMSE'][name] = rms
            self.rmse_results['Mean'][name] = rms.mean()
            self.rmse_results['std'][name] = rms.std()
            
            print(name,(30-len(name))*'=','>','is Done!')
            
        if show : return self.r_2_results, self.rmse_results
        
    def visualize_results(self, 
                          cv_train_test,
                          metrics=['r_squared','adjusted r_squared','mae','mse','rmse'],
                          metrics_cv=['r_squared','rmse']):
        
        if cv_train_test.lower()=='cv':
            
            #visualize the results of R-Squared CV for each model
            self.r_2_cv_results = pd.DataFrame(index=self.r_2_results['R-Squared'].keys())
            #append the max R-Squared for each model to the dataframe
            self.r_2_cv_results['Max'] = [self.r_2_results['R-Squared'][m].max() for m in self.r_2_results['R-Squared'].keys()]
            #append the mean of all R-Squared for each model to the dataframe
            self.r_2_cv_results['Mean'] = [self.r_2_results['Mean'][m] for m in self.r_2_results['Mean'].keys()]
            #append the min R-Squared for each model to the dataframe
            self.r_2_cv_results['Min'] = [self.r_2_results['R-Squared'][m].min() for m in self.r_2_results['R-Squared'].keys()]
            #append the std of all R-Squared for each model to the dataframe
            self.r_2_cv_results['std'] = [self.r_2_results['std'][m] for m in self.r_2_results['std'].keys()]

            #visualize the results of RMSE CV for each model
            self.rmse_cv_results = pd.DataFrame(index=self.rmse_results['RMSE'].keys())
            #append the max R-Squared for each model to the dataframe
            self.rmse_cv_results['Max'] = [self.rmse_results['RMSE'][m].max() for m in self.rmse_results['RMSE'].keys()]
            #append the mean of all R-Squared for each model to the dataframe
            self.rmse_cv_results['Mean'] = [self.rmse_results['Mean'][m] for m in self.rmse_results['Mean'].keys()]
            #append the min R-Squared for each model to the dataframe
            self.rmse_cv_results['Min'] = [self.rmse_results['RMSE'][m].min() for m in self.rmse_results['RMSE'].keys()]
            #append the std of all R-Squared for each model to the dataframe
            self.rmse_cv_results['std'] = [self.rmse_results['std'][m] for m in self.rmse_results['std'].keys()]

            for parm in metrics_cv:
                if parm.lower() in ['rmse','root mean squared']:
                    self.rmse_cv_results = self.rmse_cv_results.sort_values(by='Mean',ascending=True)
                    self.rmse_cv_results.iplot(kind='bar',
                                               title='Maximum, Minimun, Mean values and standard deviation <br>For RMSE values for each model')
                    self.scores = pd.DataFrame(self.rmse_results['RMSE'])
                    self.scores.iplot(kind='box',
                                      title='Box plot for the variation of RMSE values for each model')

                elif parm.lower() in ['r_squared','rsquared','r squared']:
                    self.r_2_cv_results = self.r_2_cv_results.sort_values(by='Mean',ascending=False)
                    self.r_2_cv_results.iplot(kind='bar',
                                              title='Max, Min, Mean, and standard deviation <br>For R-Squared values for each model')
                    self.scores = pd.DataFrame(self.r_2_results['R-Squared'])
                    self.scores.iplot(kind='box',
                                 title='Box plot for the variation of R-Squared for each model')
                else:
                    print('Not avilable')
                    
        elif cv_train_test.lower()=='train test':
            R_2 = pd.DataFrame(self.result_data['R^2']).sort_values(by='Testing',ascending=False)
            Adjusted_R_2 = pd.DataFrame(self.result_data['Adjusted R^2']).sort_values(by='Testing',ascending=False)
            MAE = pd.DataFrame(self.result_data['MAE']).sort_values(by='Testing',ascending=True)
            MSE = pd.DataFrame(self.result_data['MSE']).sort_values(by='Testing',ascending=True)
            RMSE = pd.DataFrame(self.result_data['RMSE']).sort_values(by='Testing',ascending=True)

            for parm in metrics:
                if parm.lower()=='r_squared':
                    #order the results by testing values
                    fig=px.line(data_frame=R_2.reset_index(),
                                x='index',y=['Training','Testing'],
                                title='R-Squared for training and testing')
                    fig.show()

                elif parm.lower()=='adjusted r_squared':
                    #order the results by testing values
                    fig=px.line(data_frame=Adjusted_R_2.reset_index(),
                                x='index',y=['Training','Testing'],
                                title='Adjusted R-Squared for training and testing')
                    fig.show()

                elif parm.lower()=='mae':
                    #order the results by testing values
                    fig=px.line(data_frame=MAE.reset_index(),
                                x='index',y=['Training','Testing'],
                                title='Mean absolute error for training and testing')
                    fig.show()

                elif parm.lower()=='mse':
                    #order the results by testing values
                    fig=px.line(data_frame=MSE.reset_index(),
                                x='index',y=['Training','Testing'],
                                title='Mean square error for training and testing')
                    fig.show()

                elif parm.lower()=='rmse':
                    #order the results by testing values
                    fig=px.line(data_frame=RMSE.reset_index(),
                                x='index',y=['Training','Testing'],
                                title='Root mean square error for training and testing')
                    fig.show()

                else:
                    print('Only (R_Squared, Adjusted R_Squared, MAE, MSE, RMSE)')

        else:
            raise TypeError('Only (CV , Train Test)')
            
    def fit_best_model(self):
        self.models=list(self.r_2_results['Mean'].keys())
        self.r_2_results_vals=np.array([r for _,r in self.r_2_results['Mean'].items()])
        self.rmse_results_vals=np.array([r for _,r in self.rmse_results['Mean'].items()])
        self.best_model_name=self.models[np.argmax(self.r_2_results_vals-self.rmse_results_vals)]
        print()
        print(30*'=')
        print('The best model is ====> ',self.best_model_name)
        print('It has the highest (R-Squared) and the lowest (Root Mean Square Erorr)')
        print(30*'=')
        print()
        self.best_model=self.base_models[self.best_model_name]
        self.best_model.fit(self.train, self.ytrain)
        print(self.best_model_name,' is fitted to the data!')
        print()
        print(30*'=')
        self.y_pred=self.best_model.predict(self.test)
        self.y_pred=np.expm1(self.y_pred)              #using expm1 (The inverse of log1p)
        self.temp=pd.DataFrame({"Id": self.testID,
                                "SalePrice": self.y_pred })
    
    def show_predictions(self):
        return self.temp
    
    def save_predictions(self, file_name):
        self.temp.to_csv('{}.csv'.format(file_name))


# <a href="#outlines"><b>Up to outlines</b>

# <a id="hpopp_class"></a>
# <h1>House Price OOP class</h1>

# In[7]:


class HouseObjectOriented:
    """
    param train: train data will be used for modelling
    param test:  test data will be used for model evaluation
    """
    def __init__(self):
        #properties
        self.ntrain=None
        self.testID=None
        self.y_train=None
        self.train=None
        self.test=None
        self._info=Information()
        self._Preprocessor = Preprocessor()
        
        print()
        print('HouseObjectOriented object is created')
        print()
        
    def add_data(self, train, test):
        #properties
        self.ntrain=train.shape[0]
        self.testID=test.reset_index().drop('index',axis=1)['Id']
        self.y_train=train['SalePrice'].apply(lambda x:np.log1p(x))
        self.train=train.drop('SalePrice', axis=1)
        self.test=test
        
        # concatinating the whole data
        self.data=self.concat_data(self.train, self.test)
        self.orig_data=self.data.copy()
        print()
        print('Your data has been added')
        print()

    def concat_data(self, train, test):
        
        data = pd.concat([self.train.set_index('Id'), self.test.set_index('Id')]).reset_index(drop=True)

        return data
    
    #using the objects
    def information(self):
        """
        using _info object gives summary about dataset
        :return:
        """
        print(self._info._info_(self.data))


    def preprocessing(self):
        
        """
        preprocess the data before applying Ml algorithms
        """
        self.data=self._Preprocessor._process(self.data, self.ntrain)

        print()
        print('Data has been Pre-Processed')
        print()
        
    class visualizer:
        
        def __init__(self, House_Price_OOP):
            
            self.hp=House_Price_OOP
            self.data=self.hp.data
            self.ytrain=self.hp.y_train  
            self.ntrain=self.hp.ntrain
            self.testID=self.hp.testID
            self.data_vis=data_visualization  
            
            
        def box_plot(self, columns):
            
            self.data_vis.box_plot(columns)
            
            
        def box_plot(self, columns):
            
            self.data_vis.box_plot(columns)
        
        
        def bar_plot(self, columns):
            
            self.data_vis.bar_plot(columns)
            
        
    class ml:
        
        def __init__(self, House_Price_OOP):
            
            self.hp=House_Price_OOP
            self.data=self.hp.data
            self.ytrain=self.hp.y_train  
            self.ntrain=self.hp.ntrain
            self.testID=self.hp.testID
            self._ML_=ML(data=self.data, ytrain=self.ytrain,
                         testID=self.testID, test_size=0.2, ntrain=self.ntrain)
        
        def show_available_algorithms(self):
            
            self._ML_.show_available()
        
        
        def init_regressors(self, num_models='all'):
        
            self._ML_.init_ml_regressors(num_models)
        
        
        def train_test_validation(self, show_results=True):
        
            self._ML_.train_test_eval_show_results(show=show_results)

            
        def cross_validation(self, num_models=4, n_folds=5, show_results=False):
            
            self._ML_.cv_eval_show_results(num_models=num_models, n_folds=n_folds, show=show_results)

            
        def visualize_trai_test(self, metrics=['r_squared','adjusted r_squared','mae','mse','rmse']):
            
            self._ML_.visualize_results(cv_train_test='train test', metrics=metrics)

            
        def visualize_cv(self, metrics=['r_squared','rmse']):
            
            self._ML_.visualize_results(cv_train_test='cv', metrics_cv=metrics)   
            
            
        def fit_best_model(self):
            
            self._ML_.fit_best_model()

            
        def show_predictions(self):
            
            return self._ML_.show_predictions()

        def save_predictions(self, file_name):
            
            self._ML_.save_predictions(file_name)
            print('The prediction is saved')


# <a href="#outlines"><b>Up to outlines</b>

# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:100%; text-align:center; border-radius: 15px 50px;">Creating Objects of our Classes</p>

# In[8]:


#collect the data
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# <a id="1.0"></a>
# <h1>Create HouseObjectOriented object</h1>

# In[9]:


#create HouseObjectOriented object
HOOP = HouseObjectOriented()


# <a id="1.1"></a>
# <h1>Adding our data</h1>

# In[10]:


#adding the data 
HOOP.add_data(train, test)


# 
# 
# <a id="2.0"></a>
# <h1>Display Information about the data</h1>

# In[11]:


HOOP.information()


# <a href="#outlines"><b>Up to outlines</b>

# <a id="3.0"></a>
# <h1>Pre-Process the data</h1>

# In[12]:


HOOP.preprocessing()


# <a id="2.1"></a>
# <h1>Display Information about the data after Pre-Processing</h1>

# In[13]:


HOOP.information()


# <a href="#outlines"><b>Up to outlines</b>

# <a id="4.0"></a>
# <h1>Create a Machine Learning object</h1>

# In[14]:


ML = HOOP.ml(HOOP)


# <a id="4.1"></a>
# <h1>Show the available algorithms</h1>

# In[15]:


ML.show_available_algorithms()


# <a id="4.2"></a>
# <h1>Initialize the ML Regressors</h1>

# In[16]:


ML.init_regressors('all')


# <a href="#outlines"><b>Up to outlines</b>

# <a id="4.3"></a>
# <h1>Train-Test Validation</h1>

# In[17]:


ML.train_test_validation()


# <a id="4.4"></a>
# <h1>Visualize the results of train-test validation</h1>

# In[18]:


ML.visualize_trai_test()


# <a href="#outlines"><b>Up to outlines</b>

# <a id="4.5"></a>
# <h1>Applying Cross-Validation</h1>

# In[19]:


ML.cross_validation('all')


# <a id="4.6"></a>
# <h1>Visualize the results of Cross-Validation</h1>

# In[20]:


ML.visualize_cv()


# <a href="#outlines"><b>Up to outlines</b>

# <a id="4.7"></a>
# <h1>Find the best model and fit it to the data</h1>

# In[21]:


ML.fit_best_model()


# <a id="4.8"></a>
# <h1>Predict and show the prediction</h1>

# In[22]:


ML.show_predictions()


# <a id="4.9"></a>
# <h1>Save the predictions to a csv file</h1>

# In[23]:


# ML.save_predictions('Results')


# <a href="#outlines"><b>Up to outlines</b>

# <h2 align="center" style='color:red' > If you liked the notebook or learned something please <b>Upvote</b>! </h2>
# <p style="background-color:skyblue; font-family:newtimeroman; font-size:200%; text-align:center; border-radius: 10px 100px;">You can also see:</p>
# <ul>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/predicting-the-survival-of-titanic-top-6'>Predicting the Survival of Titanic (Top 6%)</a>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/house-price-prediction-top-8'>
# ✔ House Price prediction(Top 8%)</a>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/prediction-of-heart-disease-machine-learning'>Prediction of Heart Disease (Machine Learning)</a>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/data-exploration-and-visualization-uber-data'>Data exploration and visualization(Uber Data)</a><br>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/hotel-booking-eda-cufflinks-and-plotly'>Hotel booking EDA (Cufflinks and plotly)
# </a><br>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/suicide-rates-visualization-and-geographic-maps/edit/run/53135916'>Suicide Rates visualization and Geographic maps</a>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/superstore-data-analysis-with-plotly-clustering'>Superstore Data Analysis With Plotly(Clustering)</a>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/superstore-analysis-with-cufflinks-and-pandas'>Superstore Analysis With Cufflinks and pandas</a>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/learn-data-analysis-using-sql-and-pandas'>Learn Data Analysis using SQL and Pandas</a>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/european-soccer-database-with-sqlite3'>European soccer database with sqlite3</a>
# <li><b><a href='https://www.kaggle.com/alaasedeeq/chinook-questions-with-sqlite'>Chinook data questions with sqlite3</a>

# ### <h1 align="center" style="color:red ">Thanks for reading</h1>
