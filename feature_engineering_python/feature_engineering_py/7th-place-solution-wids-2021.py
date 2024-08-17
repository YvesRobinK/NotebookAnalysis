#!/usr/bin/env python
# coding: utf-8

# # Data Engineering Phase

# In[1]:


from sklearn.linear_model import LinearRegression, LassoCV
from tqdm import tqdm
import pickle
from scipy import stats, special
import pandas as pd
import numpy as np
import os
import warnings
warnings.simplefilter("ignore")


# In[2]:


#remove columns with large number of nan values
def remove_nan_cols (df, threshold=0.5):
    nan_cols = []
    for col in df.columns:
        nan_ratio = df[col].isnull().sum() / df.shape[0]
        if nan_ratio >= threshold:
            nan_cols.append(col)
    df = df.drop(nan_cols, axis=1)
    return df


# In[3]:


#filling missing values based on linear regression and the most correlated variables
def fillna_using_linear_model(df, feature_cols):

    correl = df[feature_cols].corr()

    for col in tqdm(feature_cols):
        nan_ratio = df[col].isnull().sum() / df.shape[0]
        if nan_ratio > 0:
            best_nan_ratio = nan_ratio
            best_col = None
            for id in correl.loc[(correl[col] > 0.7) | (correl[col] < -0.7), col].index:
                nan_temp_ratio = df[id].isnull().sum() / df.shape[0]
                if best_nan_ratio > nan_temp_ratio:
                    best_nan_ratio = nan_temp_ratio
                    best_col = id
            if best_col != None:
                sub = df[[col, best_col]].copy()
                sub = sub.dropna()
                reg = LinearRegression(fit_intercept=True).fit(np.expand_dims(sub[best_col], axis=1), sub[col])
                print(reg.score(np.expand_dims(sub[best_col], axis=1), sub[col]))
                if reg.score(np.expand_dims(sub[best_col], axis=1), sub[col])>0.7:
                    if df.loc[(~df[best_col].isnull()) & (df[col].isnull()), col].shape[0] > 0:
                        df.loc[(~df[best_col].isnull()) & (df[col].isnull()), col] = \
                            reg.predict(np.expand_dims(
                                df.loc[(~df[best_col].isnull()) & (df[col].isnull()), best_col], axis=1))

    return df


# The creation of additional variables played an important role in improving the competition score. This notebook https://www.kaggle.com/siavrez/2020fatures was used.Thank you for helping us improve our score. 

# In[4]:


def feature_generation(df):

    agg = df['icu_id'].value_counts().to_dict()
    df['icu_id_counts'] = np.log1p(df['icu_id'].map(agg))
    agg = df['age'].value_counts().to_dict()
    df['age_counts'] = np.log1p(df['age'].map(agg))

    df['nan_counts'] = df.isnull().sum(axis=1)

    df['sq_age'] = df['age'].values ** 2
    df['sq_bmi'] = df['bmi'].values ** 2
    df['bmi_age'] = df['bmi'].values / df['age'].values
    df['weight_age'] = df['age'].values / df['weight'].values

    df['comorbidity_score'] = df['aids'].values * 23 + df['cirrhosis'].values * 4 + df['diabetes_mellitus'].values * 1 + \
                                 df['hepatic_failure'].values * 16 + df['immunosuppression'].values * 10 + \
                              df['leukemia'].values * 10 + df['lymphoma'].values * 13 + df['solid_tumor_with_metastasis'].values * 11
    
    
    #source https://www.omnicalculator.com/health/risk-dm
    df['diabete_risk'] = 100 / (1 + np.exp(-1*(0.028*df['age'].values + 0.661*np.where(df['gender'].values=="M", 1, 0) +
                                               0.412 * np.where(df['ethnicity'].values=="Native American", 0, 1) +
                                               0.079 * df['glucose_apache'].values + 0.018 * df['d1_diasbp_max'].values +
                                               0.07 * df['bmi'].values + 0.481 * df['cirrhosis'].values - 13.415)))

    df['gcs_sum'] = df['gcs_eyes_apache'].values + df['gcs_motor_apache'].values + df['gcs_verbal_apache'].values
    df['apache_2_diagnosis_type'] = df.apache_2_diagnosis.round(-1).fillna(-100).astype('int32')
    df['apache_3j_diagnosis_type'] = df.apache_3j_diagnosis.round(-2).fillna(-100).astype('int32')
    df['bmi_type'] = df.bmi.fillna(0).apply(lambda x: 5 * (round(int(x) / 5)))
    df['height_type'] = df.height.fillna(0).apply(lambda x: 5 * (round(int(x) / 5)))
    df['weight_type'] = df.weight.fillna(0).apply(lambda x: 5 * (round(int(x) / 5)))
    df['age_type'] = df.age.fillna(0).apply(lambda x: 10 * (round(int(x) / 10)))
    df['gcs_sum_type'] = df.gcs_sum.fillna(0).apply(lambda x: 2.5 * (round(int(x) / 2.5))).divide(2.5)
    df['apache_3j_diagnosis_x'] = df['apache_3j_diagnosis'].astype('str').str.split('.', n=1, expand=True)[0]
    df['apache_2_diagnosis_x'] = df['apache_2_diagnosis'].astype('str').str.split('.', n=1, expand=True)[0]
    df['apache_3j_diagnosis_split1'] = np.where(df['apache_3j_diagnosis'].isna(), np.nan,
                                                   df['apache_3j_diagnosis'].astype('str').str.split('.', n=1,
                                                                                                        expand=True)[1])
    df['apache_2_diagnosis_split1'] = np.where(df['apache_2_diagnosis'].isna(), np.nan,
                                                  df['apache_2_diagnosis'].apply(lambda x: x % 10))

    IDENTIFYING_COLS = ['age_type', 'gcs_sum_type', 'ethnicity', 'gender', 'bmi_type']
    df['profile'] = df[IDENTIFYING_COLS].apply(lambda x: hash(tuple(x)), axis=1)

    df["diff_bmi"] = df['bmi'].copy()
    df['bmi'] = df['weight'].values / ((df['height'].values / 100) ** 2)
    df["diff_bmi"] = df["diff_bmi"].values - df['bmi'].values

    df['pre_icu_los_days'] = df['pre_icu_los_days'].apply(lambda x: special.expit(x))

    d_cols = [c for c in df.columns if (c.startswith("d1"))]
    h_cols = [c for c in df.columns if (c.startswith("h1"))]
    df["dailyLabs_row_nan_count"] = df[d_cols].isna().sum(axis=1)
    df["hourlyLabs_row_nan_count"] = df[h_cols].isna().sum(axis=1)
    df["diff_labTestsRun_daily_hourly"] = df["dailyLabs_row_nan_count"].values - df["hourlyLabs_row_nan_count"].values

    lab_col = [c for c in df.columns if ((c.startswith("h1")) | (c.startswith("d1")))]
    lab_col_names = list(set(list(map(lambda i: i[3: -4], lab_col))))

    first_h = []
    for v in tqdm(lab_col_names):
        colsx = [x for x in df.columns if v in x]
        df[v + "_nans"] = df.loc[:, colsx].isna().sum(axis=1)
        df[v + "_d1_h1_max_eq"] = (df[f"d1_{v}_max"] == df[f"h1_{v}_max"]).astype(np.int8)
        df[v + "_d1_h1_min_eq"] = (df[f"d1_{v}_min"] == df[f"h1_{v}_min"]).astype(np.int8)

        for freq in ['h1', 'd1']:
            df[v + f"_{freq}_value_range"] = df[f"{freq}_{v}_max"].subtract(df[f"{freq}_{v}_min"])
            df[v + f"_{freq}_zero_range"] = (df[v + f"_{freq}_value_range"] == 0).astype(np.int8)
            df[v + f"_{freq}_mean"] =np.nanmean(df[[f"{freq}_{v}_max", f"{freq}_{v}_min"]].values, axis=1)
            df[v + f"_{freq}_std"] = np.nanstd(df[[f"{freq}_{v}_max", f"{freq}_{v}_min"]].values, axis=1)

            for g in ['apache_3j_diagnosis', 'profile', 'icu_id']:
                for m in ['max', 'min']:
                    temp = df[[g, f"{freq}_{v}_{m}"]].groupby(g)
                    df[v + f"_{freq}_{m}_{g}_mean"] = temp.transform('mean')
                    df[v + f"_{freq}_{m}_{g}_diff"] = df[v + f"_{freq}_{m}_{g}_mean"].subtract(df[f"{freq}_{v}_{m}"])
                    df[v + f"_{freq}_{m}_{g}_std"] = temp.transform('std')
                    df[v + f"_{freq}_{m}_{g}_norm_std"] = df[v + f"_{freq}_{m}_{g}_std"].div(df[f"{freq}_{v}_{m}"])
                    df[v + f"_{freq}_{m}_{g}_rank"] =temp.transform('rank')
                    df[v + f"_{freq}_{m}_{g}_count"] = temp.transform('count')
                    df[v + f"_{freq}_{m}_{g}_norm_rank"] = df[v + f"_{freq}_{m}_{g}_rank"].div(df[v + f"_{freq}_{m}_{g}_count"])
                    df[v + f"_{freq}_{m}_{g}_skew"] = temp.transform('skew')

            if v + "_apache" in colsx:
                for m in ['max', 'min']:
                    df[v + f"_apache_{freq}_{m}_ratio"] = df[f"{freq}_{v}_{m}"].div(df[v + "_apache"])

            for m in ['max', 'min']:
                df[f"{freq}_{v}_{m}_bmi_ratio"] = df[f"{freq}_{v}_{m}"].div(df['bmi'])
                df[f"{freq}_{v}_{m}_weight_ratio"] = df[f"{freq}_{v}_{m}"].div(df['weight'])

        df[v + "_range_between_d_h"] = df[v + "_d1_mean"].values - df[v + "_h1_mean"].values
        df[v + "_d1_h1_mean"] = np.nanmean(df[[f"d1_{v}_max", f"d1_{v}_min", f"h1_{v}_max", f"h1_{v}_min"]].values, axis=1)
        df[v + "_d1_h1_std"] = np.nanstd(df[[f"d1_{v}_max", f"d1_{v}_min", f"h1_{v}_max", f"h1_{v}_min"]].values, axis=1)

        df[v + "_tot_change_value_range_normed"] = abs((df[v + "_d1_value_range"].div(df[v + "_h1_value_range"])))
        df[v + "_started_after_firstHour"] = ((df[f"h1_{v}_max"].isna()) & (df[f"h1_{v}_min"].isna())) & (~df[f"d1_{v}_max"].isna())
        first_h.append(v + "_started_after_firstHour")
        df[v + "_day_more_extreme"] = ((df[f"d1_{v}_max"] > df[f"h1_{v}_max"]) | (df[f"d1_{v}_min"] < df[f"h1_{v}_min"]))
        df[v + "_day_more_extreme"].fillna(False)

    df["total_Tests_started_After_firstHour"] = df[first_h].sum(axis=1)

    df['diasbp_indicator'] = (
            (df['d1_diasbp_invasive_max'] == df['d1_diasbp_max']) & (
            df['d1_diasbp_noninvasive_max'] == df['d1_diasbp_invasive_max']) |
            (df['d1_diasbp_invasive_min'] == df['d1_diasbp_min']) & (
                    df['d1_diasbp_noninvasive_min'] == df['d1_diasbp_invasive_min']) |
            (df['h1_diasbp_invasive_max'] == df['h1_diasbp_max']) & (
                    df['h1_diasbp_noninvasive_max'] == df['h1_diasbp_invasive_max']) |
            (df['h1_diasbp_invasive_min'] == df['h1_diasbp_min']) & (
                    df['h1_diasbp_noninvasive_min'] == df['h1_diasbp_invasive_min'])).astype(np.int8)

    df['mbp_indicator'] = (
            (df['d1_mbp_invasive_max'] == df['d1_mbp_max']) & (
            df['d1_mbp_noninvasive_max'] == df['d1_mbp_invasive_max']) |
            (df['d1_mbp_invasive_min'] == df['d1_mbp_min']) & (
                    df['d1_mbp_noninvasive_min'] == df['d1_mbp_invasive_min']) |
            (df['h1_mbp_invasive_max'] == df['h1_mbp_max']) & (
                    df['h1_mbp_noninvasive_max'] == df['h1_mbp_invasive_max']) |
            (df['h1_mbp_invasive_min'] == df['h1_mbp_min']) & (
                    df['h1_mbp_noninvasive_min'] == df['h1_mbp_invasive_min'])
    ).astype(np.int8)

    df['sysbp_indicator'] = (
            (df['d1_sysbp_invasive_max'] == df['d1_sysbp_max']) & (
            df['d1_sysbp_noninvasive_max'] == df['d1_sysbp_invasive_max']) |
            (df['d1_sysbp_invasive_min'] == df['d1_sysbp_min']) & (
                    df['d1_sysbp_noninvasive_min'] == df['d1_sysbp_invasive_min']) |
            (df['h1_sysbp_invasive_max'] == df['h1_sysbp_max']) & (
                    df['h1_sysbp_noninvasive_max'] == df['h1_sysbp_invasive_max']) |
            (df['h1_sysbp_invasive_min'] == df['h1_sysbp_min']) & (
                    df['h1_sysbp_noninvasive_min'] == df['h1_sysbp_invasive_min'])
    ).astype(np.int8)

    df['d1_mbp_invnoninv_max_diff'] = df['d1_mbp_invasive_max'].div(df['d1_mbp_noninvasive_max'])
    df['h1_mbp_invnoninv_max_diff'] = df['h1_mbp_invasive_max'].div(df['h1_mbp_noninvasive_max'])
    df['d1_mbp_invnoninv_min_diff'] = df['d1_mbp_invasive_min'].div(df['d1_mbp_noninvasive_min'])
    df['h1_mbp_invnoninv_min_diff'] = df['h1_mbp_invasive_min'].div(df['h1_mbp_noninvasive_min'])
    df['d1_diasbp_invnoninv_max_diff'] = df['d1_diasbp_invasive_max'].div(df['d1_diasbp_noninvasive_max'])
    df['h1_diasbp_invnoninv_max_diff'] = df['h1_diasbp_invasive_max'].div(df['h1_diasbp_noninvasive_max'])
    df['d1_diasbp_invnoninv_min_diff'] = df['d1_diasbp_invasive_min'].div(df['d1_diasbp_noninvasive_min'])
    df['h1_diasbp_invnoninv_min_diff'] = df['h1_diasbp_invasive_min'].div(df['h1_diasbp_noninvasive_min'])
    df['d1_sysbp_invnoninv_max_diff'] = df['d1_sysbp_invasive_max'].div(df['d1_sysbp_noninvasive_max'])
    df['h1_sysbp_invnoninv_max_diff'] = df['h1_sysbp_invasive_max'].div(df['h1_sysbp_noninvasive_max'])
    df['d1_sysbp_invnoninv_min_diff'] = df['d1_sysbp_invasive_min'].div(df['d1_sysbp_noninvasive_min'])
    df['h1_sysbp_invnoninv_min_diff'] = df['h1_sysbp_invasive_min'].div(df['h1_sysbp_noninvasive_min'])

    more_extreme_cols = [c for c in df.columns if (c.endswith("_day_more_extreme"))]
    df["total_day_more_extreme"] = df[more_extreme_cols].sum(axis=1)
    df["d1_resprate_div_mbp_min"] = df["d1_resprate_min"].div(df["d1_mbp_min"])
    df["d1_resprate_div_sysbp_min"] = df["d1_resprate_min"].div(df["d1_sysbp_min"])
    df["d1_lactate_min_div_diasbp_min"] = df["d1_lactate_min"].div(df["d1_diasbp_min"])
    df["d1_heartrate_min_div_d1_sysbp_min"] = df["d1_heartrate_min"].div(df["d1_sysbp_min"])
    df["total_chronic"] = df[["aids", "cirrhosis", 'hepatic_failure']].sum(axis=1)
    df["total_cancer_immuno"] = df[['immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']].sum(axis=1)
    df["has_complicator"] = df[["aids", "cirrhosis", 'hepatic_failure','immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']].max(axis=1)

    df['apache_3j'] = np.where(df['apache_3j_diagnosis_type'] < 0, np.nan,
                               np.where(df['apache_3j_diagnosis_type'] < 200, 'Cardiovascular',
                                        np.where(df['apache_3j_diagnosis_type'] < 400, 'Respiratory',
                                                 np.where(df['apache_3j_diagnosis_type'] < 500, 'Neurological',
                                                          np.where(df['apache_3j_diagnosis_type'] < 600, 'Sepsis',
                                                                   np.where(df['apache_3j_diagnosis_type'] < 800,
                                                                            'Trauma',
                                                                            np.where(
                                                                                df['apache_3j_diagnosis_type'] < 900,
                                                                                'Haematological',
                                                                                np.where(df[
                                                                                             'apache_3j_diagnosis_type'] < 1000,
                                                                                         'Renal/Genitourinary',
                                                                                         np.where(df[
                                                                                                      'apache_3j_diagnosis_type'] < 1200,
                                                                                                  'Musculoskeletal/Skin disease',
                                                                                                  'Operative Sub-Diagnosis Codes')))))))))


    return df


# In[5]:


#remove columns with low variation
def remove_feature_with_low_var(df, threshold=0.1):
    for col in df.columns:
        if df[col].std() < threshold:
            df = df.drop([col], axis=1)
    return df


# Data preprocess running phase. On kaggle notebook, out of memory error will be occurred. We shared our preprocessed dataset here https://www.kaggle.com/lhagiimn/wids-2021-preprocessed-data and feature importance https://www.kaggle.com/lhagiimn/feature-importance
# 
# "feature_importance_v2.csv" file doesn't include the features of "Additional Feature Engineering" and "Lasso Based Additional Feature Engineering". 
# "feature_importance_v1.csv" file doesn't include the features of "Lasso Based Additional Feature Engineering". 
# "feature_importance.csv" file includes all features. 

# In[6]:


if os.path.isfile('../input/wids-2021-preprocessed-data/train_df.pkl'):
    with open('../input/wids-2021-preprocessed-data/train_df.pkl', 'rb') as handle:
        train_df = pickle.load(handle)

    with open('../input/wids-2021-preprocessed-data/test_df.pkl', 'rb') as handle:
        test_df = pickle.load(handle)

else:
    
    train_df = pd.read_csv ( "../input/widsdatathon2021/TrainingWiDS2021.csv").drop(columns=['Unnamed: 0'], axis=1)
    test_df = pd.read_csv("../input/widsdatathon2021/UnlabeledWiDS2021.csv").drop(columns=['Unnamed: 0'], axis=1)
    data_dict = pd.read_csv ("../input/widsdatathon2021/DataDictionaryWiDS2021.csv")

    train_df['which_data'] = 'train'
    test_df['which_data'] = 'test'
    test_df['diabetes_mellitus'] = 0

    df = pd.concat([train_df, test_df[train_df.columns]], axis=0)

    del train_df, test_df

    df['age'] = np.where(df['age'].values==0, np.nan, df['age'].values)
    df = df.replace([np.inf, -np.inf], np.nan)

    min_max_feats=[f[:-4] for f in df.columns if f[-4:]=='_min']
    for col in min_max_feats:
        df.loc[df[f'{col}_min'] > df[f'{col}_max'], [f'{col}_min', f'{col}_max']] = \
            df.loc[df[f'{col}_min'] > df[f'{col}_max'], [f'{col}_max', f'{col}_min']].values

    nan_col = False
    if nan_col ==True:
        df = remove_nan_cols(df, threshold=0.75)

    cont_cols = []
    for col in df.columns:
        if df[col].dtype=='float64':
            cont_cols.append(col)

    fillna=True
    if fillna==True:
        df = fillna_using_linear_model(df, cont_cols)

    df = feature_generation(df)
    df = df.replace([np.inf, -np.inf], np.nan)
        
    cats = ['elective_surgery', 'icu_id', 'arf_apache', 'intubated_apache', 'ventilated_apache', 'cirrhosis',
        'hepatic_failure', 'immunosuppression', 'leukemia', 'solid_tumor_with_metastasis', 'apache_3j_diagnosis_x',
        'apache_2_diagnosis_x', 'apache_3j', 'apache_3j_diagnosis_split1', 'apache_2_diagnosis_split1', 'gcs_sum_type',
        'hospital_admit_source', 'glucose_rate', 'glucose_wb', 'gcs_eyes_apache', 'glucose_normal', 'total_cancer_immuno',
        'gender', 'total_chronic', 'icu_stay_type', 'apache_2_diagnosis_type', 'apache_3j_diagnosis_type']


    df['hospital_admit_source'] = df['hospital_admit_source'].replace({'Other ICU': 'ICU','ICU to SDU':'SDU',
                                                                             'Step-Down Unit (SDU)': 'SDU',
                                                                             'Other Hospital':'Other','Observation': 'Recovery Room',
                                                                             'Acute Care/Floor': 'Acute Care'})

    drop_cols = []
    features = df.columns
    features = [f for f in features if f not in cats]
    for col in features:
        if df[col].dtype!='object' and col!='diabetes_mellitus':
            if df[col].std()==0 or df[col].std()==np.nan:
                drop_cols.append(col)

    df = df.drop(drop_cols, axis=1)

    cats = [f for f in cats if f in df.columns]
    train_df = df.loc[df['which_data']=='train']
    test_df = df.loc[df['which_data']=='test']
    train_df = train_df.drop(['which_data'], axis=1)
    test_df = test_df.drop(['which_data'], axis=1)

    for col in train_df.select_dtypes(exclude = np.number).columns.tolist():
        train_only = list(set(train_df[col].unique()) - set(test_df[col].unique()))
        test_only = list(set(test_df[col].unique()) - set(train_df[col].unique()))
        both = list(set(test_df[col].unique()).union(set(train_df[col].unique())))
        train_df.loc[train_df[col].isin(train_only), col] = np.nan
        test_df.loc[test_df[col].isin(test_only), col] = np.nan

        test_df[col] = test_df[col].astype('str')
        train_df[col] = train_df[col].astype('str')
        le = LabelEncoder()
        le.fit(pd.concat([train_df[col], test_df[col]]))

        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])


    for col in tqdm(cats):
        train_only = list(set(train_df[col].unique()) - set(test_df[col].unique()))
        test_only = list(set(test_df[col].unique()) - set(train_df[col].unique()))
        both = list(set(test_df[col].unique()).union(set(train_df[col].unique())))
        train_df.loc[train_df[col].isin(train_only), col] = np.nan
        test_df.loc[test_df[col].isin(test_only), col] = np.nan
        try:
            le = LabelEncoder()
            le.fit(pd.concat([train_df[col], test_df[col]]))
            train_df[col] = le.transform(train_df[col])
            test_df[col] = le.transform(test_df[col])
        except:
            test_df[col] = test_df[col].astype('str').fillna('-1')
            train_df[col] = train_df[col].astype('str').fillna('-1')
            le = LabelEncoder()
            le.fit(pd.concat([train_df[col], test_df[col]]))
            train_df[col] = le.transform(train_df[col])
            test_df[col] = le.transform(test_df[col])

        temp = pd.concat([train_df[[col]], test_df[[col]]], axis=0)
        temp_mapping = temp.groupby(col).size()/len(temp)
        temp['enc'] = temp[col].map(temp_mapping)
        temp['enc'] = stats.rankdata(temp['enc'])
        temp = temp.reset_index(drop=True)
        train_df[f'rank_frqenc_{col}'] = temp[['enc']].values[:train_df.shape[0]]
        test_df[f'rank_frqenc_{col}'] = temp[['enc']].values[train_df.shape[0]:]
        test_df[col] = test_df[col].astype('category')
        train_df[col] = train_df[col].astype('category')


    feature_cols = list(train_df)
    feature_cols.remove('diabetes_mellitus')
    feature_cols.remove('encounter_id')
    feature_cols.remove('hospital_id')

    cont_cols = feature_cols
    cont_cols = [f for f in cont_cols if f not in cats]

    for col in cont_cols:
        r = stats.ks_2samp(train_df[col].dropna(), test_df[col].dropna())
        print(col, r)
        if r[0] >= 0.05:
            cont_cols.remove(col)

    '''
    temp = pd.concat([train_df[cont_cols], test_df[cont_cols]], axis=0)
    corr = temp[cont_cols].corr()

    drop_columns=[]
    # Drop highly correlated features
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >=0.999:
                if columns[j]:
                    columns[j] = False
                    print('FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(temp[cont_cols].columns[i] , temp[cont_cols].columns[j], corr.iloc[i,j]))
            elif corr.iloc[i,j] <= -0.999:
                if columns[j]:
                    columns[j] = False
                    print('FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(temp[cont_cols].columns[i], temp[cont_cols].columns[j], corr.iloc[i, j]))

    drop_columns = temp[cont_cols].columns[columns == False].values

    print('drop_columns',len(drop_columns),drop_columns)

    cont_cols = [col for col in cont_cols if col not in drop_columns]
    '''
    
    print("Number of input variables:", len(cont_cols)+len(cats))

    train_df = train_df[cont_cols+cats+['encounter_id', 'diabetes_mellitus']]
    test_df = test_df[cont_cols+cats+['encounter_id']]

    with open('data/train_df.pkl', 'wb') as fout:
        pickle.dump(train_df, fout)

    with open('data/test_df.pkl', 'wb') as fout:
        pickle.dump(test_df, fout)


# # Additional Feature Engineering

# We also perform additional features. Our team member @Aristotelis Charalampous performed it. 

# In[7]:


def get_apache_meta_features(df):

    df['d1_glucose_min_flag'] = np.where(df['d1_glucose_min'] > (120*.0555), 1, 0)

    df['SerumPotassium_apache'] = (df['d1_potassium_min'].values + df['d1_potassium_max'].values)/2
    df['SerumBicarb_apache'] = (df['d1_hco3_min'].values + df['d1_hco3_max'].values)/2

    df['temperature'] = 0
    df.loc[(df['temp_apache']< 36) | (df['temp_apache'] > 38.5), "temperature"] = 4

    df['arterial_pressure'] = 0
    df.loc[df['map_apache'] < 50, 'arterial_pressure'] = 4
    df.loc[(df['map_apache'] >= 50) &
           (df['map_apache'] < 70), 'arterial_pressure'] = 2
    df.loc[(df['map_apache'] >= 70) &
           (df['map_apache'] < 110), 'arterial_pressure'] = 0
    df.loc[(df['map_apache'] >= 110) &
           (df['map_apache'] < 130), 'arterial_pressure'] = 2
    df.loc[(df['map_apache'] >= 130) &
           (df['map_apache'] < 160), 'arterial_pressure'] = 3
    df.loc[df['map_apache'] >= 160, 'arterial_pressure'] = 4

    df['heart_rate_pulse'] = 0
    df.loc[df['heart_rate_apache'] < 40, 'heart_rate_pulse'] = 4
    df.loc[(df['heart_rate_apache'] >= 40) &
           (df['heart_rate_apache'] < 55), 'heart_rate_pulse'] = 3
    df.loc[(df['heart_rate_apache'] >= 55) &
           (df['heart_rate_apache'] < 70), 'heart_rate_pulse'] = 2
    df.loc[(df['heart_rate_apache'] >= 70) &
           (df['heart_rate_apache'] < 110), 'heart_rate_pulse'] = 0
    df.loc[(df['heart_rate_apache'] >= 110) &
           (df['heart_rate_apache'] < 140), 'heart_rate_pulse'] = 2
    df.loc[(df['heart_rate_apache'] >= 140) &
           (df['heart_rate_apache'] < 180), 'heart_rate_pulse'] = 3
    df.loc[df['heart_rate_apache'] >= 180, 'heart_rate_pulse'] = 4

    df['respiration_rate'] = 0
    df.loc[df['resprate_apache'] < 6, 'respiration_rate'] = 4
    df.loc[(df['resprate_apache'] >= 6) &
           (df['resprate_apache'] < 10), 'respiration_rate'] = 3
    df.loc[(df['resprate_apache'] >= 10) &
           (df['resprate_apache'] < 12), 'respiration_rate'] = 2
    df.loc[(df['resprate_apache'] >= 12) &
           (df['resprate_apache'] < 25), 'respiration_rate'] = 0
    df.loc[(df['resprate_apache'] >= 25) &
           (df['resprate_apache'] < 35), 'respiration_rate'] = 2
    df.loc[(df['resprate_apache'] >= 35) &
           (df['resprate_apache'] < 50), 'respiration_rate'] = 3
    df.loc[df['resprate_apache'] >= 50, 'respiration_rate'] = 4

    df['oxygenation_rate'] = 0
    df['pao2_apache'] = df['pao2_apache'].fillna(0)

    df.loc[df['pao2_apache'] < 200, 'oxygenation_rate'] = 0
    df.loc[(df['pao2_apache'] >= 200) &
           (df['pao2_apache'] < 350), 'oxygenation_rate'] = 2
    df.loc[(df['pao2_apache'] >= 350) &
           (df['pao2_apache'] < 500), 'oxygenation_rate'] = 3
    df.loc[df['pao2_apache'] >= 500, 'oxygenation_rate'] = 4

    df.drop(columns=['pao2_apache'], inplace=True)

    df['serum_bicarb'] = 0
    df.loc[df['SerumBicarb_apache'] < 15, 'serum_bicarb'] = 4
    df.loc[(df['SerumBicarb_apache'] >= 14) &
           (df['SerumBicarb_apache'] < 18), 'serum_bicarb'] = 3
    df.loc[(df['SerumBicarb_apache'] >= 18) &
           (df['SerumBicarb_apache'] < 22), 'serum_bicarb'] = 2
    df.loc[(df['SerumBicarb_apache'] >= 22) &
           (df['SerumBicarb_apache'] < 32), 'serum_bicarb'] = 0
    df.loc[(df['SerumBicarb_apache'] >= 32) &
           (df['SerumBicarb_apache'] < 41), 'serum_bicarb'] = 2
    df.loc[(df['SerumBicarb_apache'] >= 41) &
           (df['SerumBicarb_apache'] < 52), 'serum_bicarb'] = 3
    df.loc[df['SerumBicarb_apache'] >= 52, 'serum_bicarb'] = 4

    df['arterial_ph'] = 0

    df.loc[df['ph_apache'] < 7.15, 'arterial_ph'] = 4
    df.loc[(df['ph_apache'] >= 7.15) &
           (df['ph_apache'] < 7.25), 'arterial_ph'] = 3
    df.loc[(df['ph_apache'] >= 7.25) &
           (df['ph_apache'] < 7.33), 'arterial_ph'] = 2
    df.loc[(df['ph_apache'] >= 7.33) &
           (df['ph_apache'] < 7.5), 'arterial_ph'] = 0
    df.loc[(df['ph_apache'] >= 7.5) &
           (df['ph_apache'] < 7.6), 'arterial_ph'] = 1
    df.loc[(df['ph_apache'] >= 7.6) &
           (df['ph_apache'] < 7.7), 'arterial_ph'] = 3
    df.loc[df['ph_apache'] >= 7.7, 'arterial_ph'] = 4

    df.drop(columns=['ph_apache'], inplace=True)

    df['serum_sodium'] = 0
    df.loc[df['sodium_apache'] < 111, 'serum_sodium'] = 4
    df.loc[(df['sodium_apache'] >= 111) &
           (df['sodium_apache'] < 120), 'serum_sodium'] = 3
    df.loc[(df['sodium_apache'] >= 120) &
           (df['sodium_apache'] < 130), 'serum_sodium'] = 2
    df.loc[(df['sodium_apache'] >= 130) &
           (df['sodium_apache'] < 150), 'serum_sodium'] = 0
    df.loc[(df['sodium_apache'] >= 150) &
           (df['sodium_apache'] < 155), 'serum_sodium'] = 1
    df.loc[(df['sodium_apache'] >= 155) &
           (df['sodium_apache'] < 160), 'serum_sodium'] = 2
    df.loc[(df['sodium_apache'] >= 160) &
           (df['sodium_apache'] < 180), 'serum_sodium'] = 3
    df.loc[df['sodium_apache'] >= 180, 'serum_sodium'] = 4

    df['serum_potassium'] = 0
    df.loc[df['SerumPotassium_apache'] < 2.5, 'serum_potassium'] = 4
    df.loc[(df['SerumPotassium_apache'] >= 2.5) &
           (df['SerumPotassium_apache'] < 3), 'serum_potassium'] = 3
    df.loc[(df['SerumPotassium_apache'] >= 3) &
           (df['SerumPotassium_apache'] < 3.5), 'serum_potassium'] = 2
    df.loc[(df['SerumPotassium_apache'] >= 3.5) &
           (df['SerumPotassium_apache'] < 5.5), 'serum_potassium'] = 0
    df.loc[(df['SerumPotassium_apache'] >= 5.5) &
           (df['SerumPotassium_apache'] < 6), 'serum_potassium'] = 2
    df.loc[(df['SerumPotassium_apache'] >= 6) &
           (df['SerumPotassium_apache'] < 7), 'serum_potassium'] = 3
    df.loc[df['SerumPotassium_apache'] >= 7, 'serum_potassium'] = 4

    df['creatinine'] = 0
    df.loc[df['creatinine_apache'] < .62, 'creatinine'] = 4
    df.loc[(df['creatinine_apache'] >= .62) &
           (df['creatinine_apache'] < 1.47), 'creatinine'] = 0
    df.loc[(df['creatinine_apache'] >= 1.47) &
           (df['creatinine_apache'] < 1.98), 'creatinine'] = 4
    df.loc[(df['creatinine_apache'] >= 1.98) &
           (df['creatinine_apache'] < 3.39), 'creatinine'] = 2
    df.loc[df['creatinine_apache'] >= 3.39, 'creatinine'] = 8

    df['acute_renal_failure'] = 0
    df.loc[df['arf_apache'] == 1, 'acute_renal_failure'] = 4

    df['hematocrits'] = 0
    df.loc[df['creatinine_apache'] < 20, 'hematocrits'] = 4
    df.loc[(df['creatinine_apache'] >= 20) &
           (df['creatinine_apache'] < 30), 'hematocrits'] = 2
    df.loc[(df['creatinine_apache'] >= 30) &
           (df['creatinine_apache'] < 46), 'hematocrits'] = 0
    df.loc[(df['creatinine_apache'] >= 46) &
           (df['creatinine_apache'] < 50), 'hematocrits'] = 1
    df.loc[(df['creatinine_apache'] >= 50) &
           (df['creatinine_apache'] < 60), 'hematocrits'] = 2
    df.loc[df['creatinine_apache'] >= 60, 'hematocrits'] = 4

    df['white_blood_cells'] = 0
    df.loc[df['wbc_apache'] < 1, 'white_blood_cells'] = 4
    df.loc[(df['wbc_apache'] >= 1) &
           (df['wbc_apache'] < 3), 'white_blood_cells'] = 2
    df.loc[(df['wbc_apache'] >= 3) &
           (df['wbc_apache'] < 15), 'white_blood_cells'] = 0
    df.loc[(df['wbc_apache'] >= 15) &
           (df['wbc_apache'] < 20), 'white_blood_cells'] = 1
    df.loc[(df['wbc_apache'] >= 20) &
           (df['wbc_apache'] < 40), 'white_blood_cells'] = 2
    df.loc[df['wbc_apache'] >= 40, 'white_blood_cells'] = 4

    df['glasgow_comma_score_gcs'] = 0
    df.loc[(
                   df['gcs_eyes_apache'] + df['gcs_motor_apache'] +
                   df['gcs_verbal_apache']) <=3,
           'glasgow_comma_score_gcs'] = 12

    df.loc[(df['gcs_eyes_apache'] +
            df['gcs_motor_apache'] +
            df['gcs_verbal_apache']) > 3,
           'glasgow_comma_score_gcs'] = 15 - df.loc[
        (df['gcs_eyes_apache'] + df['gcs_motor_apache'] +
         df['gcs_verbal_apache']) > 3, 'gcs_eyes_apache'] + \
                                        df.loc[(df['gcs_eyes_apache'] + df['gcs_motor_apache'] +
                                                df['gcs_verbal_apache']) > 3, 'gcs_motor_apache'] + \
                                        df.loc[(df['gcs_eyes_apache'] + df['gcs_motor_apache'] +
                                                df['gcs_verbal_apache']) > 3, 'gcs_verbal_apache']

    df['age_death_prob'] = 0
    df.loc[df['wbc_apache'] < 45, 'age_death_prob'] = 0
    df.loc[(df['wbc_apache'] >= 45) &
           (df['wbc_apache'] < 55), 'age_death_prob'] = 2
    df.loc[(df['wbc_apache'] >= 55) &
           (df['wbc_apache'] < 65), 'age_death_prob'] = 3
    df.loc[(df['wbc_apache'] >= 65) &
           (df['wbc_apache'] < 75), 'age_death_prob'] = 5
    df.loc[df['wbc_apache'] >= 75, 'age_death_prob'] = 6

    df['inmunodeficiencia'] = 0
    df.loc[(df['aids'] +
            df['cirrhosis'] +
            df['hepatic_failure'] +
            df['immunosuppression'] +
            df['leukemia'] +
            df['lymphoma'] +
            df['solid_tumor_with_metastasis']) >= 1, 'inmunodeficiencia'] = 5

    df['post_operative'] = 0
    df.loc[df['apache_post_operative'] == 1, "post_operative"] = 3

    df['danger_level'] = df['temperature'] + \
                         df['arterial_pressure'] + df['heart_rate_pulse'] + \
                         df['respiration_rate'] + df['oxygenation_rate'] + \
                         df['serum_bicarb'] + df['arterial_ph'] + \
                         df['serum_sodium'] + df['serum_potassium'] + \
                         df['creatinine'] + df['acute_renal_failure'] + \
                         df['hematocrits'] + df['white_blood_cells'] + \
                         df['glasgow_comma_score_gcs'] + df['age_death_prob'] + \
                         df['inmunodeficiencia'] + df['post_operative']

    df.drop(columns=['temperature', 'arterial_pressure',
                     'heart_rate_pulse', 'respiration_rate',
                     'oxygenation_rate', 'serum_bicarb',
                     'arterial_ph', 'serum_sodium', 'serum_potassium',
                     'creatinine', 'acute_renal_failure', 'hematocrits',
                     'white_blood_cells', 'glasgow_comma_score_gcs',
                     'age_death_prob', 'inmunodeficiencia', 'post_operative'
                     ], inplace=True)

    return df['danger_level']


# In[8]:


if os.path.isfile('../input/wids-2021-preprocessed-data/train_gfe.pkl'):
    with open('../input/wids-2021-preprocessed-data/train_gfe.pkl', 'rb') as handle:
        train_gfe = pickle.load(handle)

    with open('../input/wids-2021-preprocessed-data/test_gfe.pkl', 'rb') as handle:
        test_gfe = pickle.load(handle)

else:

    train = pd.read_csv ( "../input/widsdatathon2021/TrainingWiDS2021.csv").drop(columns=['Unnamed: 0'], axis=1)
    test = pd.read_csv("../input/widsdatathon2021/UnlabeledWiDS2021.csv").drop(columns=['Unnamed: 0'], axis=1)

    train_danger_level = get_apache_meta_features(train.copy())
    test_danger_level = get_apache_meta_features(test.copy())

    train = train.rename(columns={'pao2_apache':'pao2fio2ratio_apache','ph_apache':'arterial_ph_apache'})
    test = test.rename(columns={'pao2_apache':'pao2fio2ratio_apache','ph_apache':'arterial_ph_apache'})

    additional_cols=[]
    for v in ['albumin','bilirubin','bun','glucose','hematocrit','pao2fio2ratio','arterial_ph','resprate','sodium','temp','wbc','creatinine']:
        additional_cols.append(f'{v}_indicator')
        train[f'{v}_indicator'] = (((train[f'{v}_apache']==train[f'd1_{v}_max']) & (train[f'd1_{v}_max']==train[f'h1_{v}_max'])) |
                     ((train[f'{v}_apache']==train[f'd1_{v}_max']) & (train[f'd1_{v}_max']==train[f'd1_{v}_min'])) |
                     ((train[f'{v}_apache']==train[f'd1_{v}_max']) & (train[f'd1_{v}_max']==train[f'h1_{v}_min'])) |
                     ((train[f'{v}_apache']==train[f'h1_{v}_max']) & (train[f'h1_{v}_max']==train[f'd1_{v}_max'])) |
                     ((train[f'{v}_apache']==train[f'h1_{v}_max']) & (train[f'h1_{v}_max']==train[f'h1_{v}_min'])) |
                     ((train[f'{v}_apache']==train[f'h1_{v}_max']) & (train[f'h1_{v}_max']==train[f'd1_{v}_min'])) |
                     ((train[f'{v}_apache']==train[f'd1_{v}_min']) & (train[f'd1_{v}_min']==train[f'd1_{v}_max'])) |
                     ((train[f'{v}_apache']==train[f'd1_{v}_min']) & (train[f'd1_{v}_min']==train[f'h1_{v}_min'])) |
                     ((train[f'{v}_apache']==train[f'd1_{v}_min']) & (train[f'd1_{v}_min']==train[f'h1_{v}_max'])) |
                     ((train[f'{v}_apache']==train[f'h1_{v}_min']) & (train[f'h1_{v}_min']==train[f'h1_{v}_max'])) |
                     ((train[f'{v}_apache']==train[f'h1_{v}_min']) & (train[f'h1_{v}_min']==train[f'd1_{v}_min'])) |
                     ((train[f'{v}_apache']==train[f'h1_{v}_min']) & (train[f'h1_{v}_min']==train[f'd1_{v}_max']))
                    ).astype(np.int8)

        test[f'{v}_indicator'] = (((test[f'{v}_apache']==test[f'd1_{v}_max']) & (test[f'd1_{v}_max']==test[f'h1_{v}_max'])) |
                     ((test[f'{v}_apache']==test[f'd1_{v}_max']) & (test[f'd1_{v}_max']==test[f'd1_{v}_min'])) |
                     ((test[f'{v}_apache']==test[f'd1_{v}_max']) & (test[f'd1_{v}_max']==test[f'h1_{v}_min'])) |
                     ((test[f'{v}_apache']==test[f'h1_{v}_max']) & (test[f'h1_{v}_max']==test[f'd1_{v}_max'])) |
                     ((test[f'{v}_apache']==test[f'h1_{v}_max']) & (test[f'h1_{v}_max']==test[f'h1_{v}_min'])) |
                     ((test[f'{v}_apache']==test[f'h1_{v}_max']) & (test[f'h1_{v}_max']==test[f'd1_{v}_min'])) |
                     ((test[f'{v}_apache']==test[f'd1_{v}_min']) & (test[f'd1_{v}_min']==test[f'd1_{v}_max'])) |
                     ((test[f'{v}_apache']==test[f'd1_{v}_min']) & (test[f'd1_{v}_min']==test[f'h1_{v}_min'])) |
                     ((test[f'{v}_apache']==test[f'd1_{v}_min']) & (test[f'd1_{v}_min']==test[f'h1_{v}_max'])) |
                     ((test[f'{v}_apache']==test[f'h1_{v}_min']) & (test[f'h1_{v}_min']==test[f'h1_{v}_max'])) |
                     ((test[f'{v}_apache']==test[f'h1_{v}_min']) & (test[f'h1_{v}_min']==test[f'd1_{v}_min'])) |
                     ((test[f'{v}_apache']==test[f'h1_{v}_min']) & (test[f'h1_{v}_min']==test[f'd1_{v}_max']))
                    ).astype(np.int8)


    rank_features = ['age', 'bmi', 'd1_heartrate_min', 'weight']
    df = pd.concat([train[rank_features], test[rank_features]])
    for f in rank_features:
        if f in list(train.columns):
            additional_cols.append(f + '_rank')
            train[f + '_rank'] = np.log1p(df[f].rank()).iloc[0:train.shape[0]].values
            test[f + '_rank'] = np.log1p(df[f].rank()).iloc[train.shape[0]:].values

    train['danger_level'] = train_danger_level
    test['danger_level'] = test_danger_level

    additional_cols.append('danger_level')
    additional_cols.append('encounter_id')

    with open('train_gfe.pkl', 'wb') as fout:
        pickle.dump(train[additional_cols], fout)

    with open('test_gfe.pkl', 'wb') as fout:
        pickle.dump(test[additional_cols], fout)


# # Lasso Based Additional Feature Engineering

# We calculate diabetes risk as a variable. It improved our score. Therefore, we randomly produced additional score features based on Lasso regression and important features. 

# In[9]:


if os.path.isfile('../input/wids-2021-preprocessed-data/train_lasso.pkl'):
    with open('../input/wids-2021-preprocessed-data/train_lasso.pkl', 'rb') as handle:
        train_lasso = pickle.load(handle)

    with open('../input/wids-2021-preprocessed-data/test_lasso.pkl', 'rb') as handle:
        test_lasso = pickle.load(handle)

else:
    
    additional_fe = ['albumin_indicator', 'bilirubin_indicator', 'bun_indicator',
                 'glucose_indicator', 'hematocrit_indicator', 'pao2fio2ratio_indicator',
                 'arterial_ph_indicator', 'resprate_indicator', 'sodium_indicator',
                 'temp_indicator', 'wbc_indicator', 'creatinine_indicator', 'age_rank',
                 'bmi_rank', 'd1_heartrate_min_rank', 'weight_rank', 'danger_level']

    for col in additional_fe:
        train_df[col] = train_gfe[col].values
        test_df[col] = test_gfe[col].values
        
    feature_imp = pd.read_csv('../input/feature-importance/feature_importance_v1.csv')
    feature_cols = list(feature_imp.loc[feature_imp['Value']>250, 'Feature'])

    cat_cols = [f for f in cat_cols if f in feature_cols]
    cont_cols = feature_cols
    cont_cols = [f for f in cont_cols if f not in cat_cols]

    print("Number of input variables:", len(cont_cols+cat_cols))

    for col in cont_cols:
        df = pd.concat([train_df[col], test_df[col]])
        train_df[col] = train_df[col].fillna(df.median())
        test_df[col] = test_df[col].fillna(df.median())

    cont_cols = [f for f in cont_cols if f not in cat_cols]

    for col in cat_cols:
        train_df[col] = train_df[col].astype('int')
        test_df[col] = test_df[col].astype('int')

    cat_cols = [f for f in cat_cols if f not in cont_cols]
    print("Number of input variables:", len(cont_cols+cat_cols))

    y = train_df['diabetes_mellitus'].values

    number_features = np.random.randint(10, 30, 100)
    list_features = []
    generate_feats = []

    for i, num in enumerate(number_features):
        list_feat = np.random.randint(0, len(cont_cols), num)
        cols = []
        for c in list_feat:
            cols.append(cont_cols[c])

        list_features.append(cols)

        X = train_df[cols].values
        clf = LassoCV(fit_intercept=True).fit(X, y)
        y_pred = clf.predict(X)
        y_pred_test = clf.predict(test_df[cols].values)

        generate_feats.append(f"col_{i}")
        train_df[f"col_{i}"] = y_pred
        test_df[f"col_{i}"] = y_pred_test

    with open('data/train_lasso.pkl', 'wb') as fout:
        pickle.dump(train_df[generate_feats], fout)

    with open('data/test_lasso.pkl', 'wb') as fout:
        pickle.dump(test_df[generate_feats], fout)

    with open('data/list_features.pkl', 'wb') as fout:
        pickle.dump(list_features, fout)
    


# # The best LightGBM model

# In[10]:


from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import gc


# In[11]:


def model_train(X_train, y_train, X_test, y_test, num_iter, params_lgb):

    dtrain = lgb.Dataset(X_train, y_train)
    dvalid = lgb.Dataset(X_test, y_test, reference=dtrain)
    model_lgb = lgb.train(params_lgb, dtrain, num_iter,
                          valid_sets=(dtrain, dvalid),
                          valid_names=('train', 'valid'),
                          early_stopping_rounds=150,
                          verbose_eval=20)

    return model_lgb

def CV(X, y, params_lgb, eval_method, shuffle=True, NFolds=5, num_iter=50):

    models = []
    y_folds = np.zeros((X.shape[0], 1))
    y_preds = np.zeros((X.shape[0], 1))

    I=0

    if eval_method=="kf":
        skf = KFold(n_splits=NFolds, random_state=42, shuffle=shuffle)
        for train_index, test_index in tqdm(skf.split(X, y)):
            print('[Fold %d/%d]' % (I + 1, NFolds))
            print('=' * 60)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model_lgb = model_train(X_train, y_train, X_test, y_test, num_iter, params_lgb)

            preds = model_lgb.predict(X_test)

            y_folds[test_index, 0] = y_test
            y_preds[test_index, 0] = preds
            models.append(model_lgb)

            I += 1

    return models, y_preds

def Kaggle_submission(file_name, models, test_data, ids_list, feature_imp, num):

    submit = pd.DataFrame()
    submit['encounter_id'] = ids_list
    submit['diabetes_mellitus'] = 0

    cols = feature_imp.loc[feature_imp['Value']>num, 'Feature']
    nfolds = len(models)
    for model in models:
        test_pred = model.predict(test_data[cols], num_iteration=model.best_iteration)
        submit['diabetes_mellitus'] += (test_pred / nfolds)

    submit.to_csv(file_name, index=False)

    return submit


# In[12]:


## merge preprocessed datasets
cats = ['elective_surgery', 'icu_id', 'arf_apache', 'intubated_apache', 'ventilated_apache', 'cirrhosis',
        'hepatic_failure', 'immunosuppression', 'leukemia', 'solid_tumor_with_metastasis', 'apache_3j_diagnosis_x',
        'apache_2_diagnosis_x', 'apache_3j', 'apache_3j_diagnosis_split1', 'apache_2_diagnosis_split1', 'gcs_sum_type',
        'hospital_admit_source', 'glucose_rate', 'glucose_wb', 'gcs_eyes_apache', 'glucose_normal', 'total_cancer_immuno',
        'gender', 'total_chronic', 'icu_stay_type', 'apache_2_diagnosis_type', 'apache_3j_diagnosis_type']

additional_fe = ['albumin_indicator', 'bilirubin_indicator', 'bun_indicator',
                 'glucose_indicator', 'hematocrit_indicator', 'pao2fio2ratio_indicator',
                 'arterial_ph_indicator', 'resprate_indicator', 'sodium_indicator',
                 'temp_indicator', 'wbc_indicator', 'creatinine_indicator', 'age_rank',
                 'bmi_rank', 'd1_heartrate_min_rank', 'weight_rank', 'danger_level']

for col in additional_fe:
    train_df[col] = train_gfe[col].values
    test_df[col] = test_gfe[col].values

del train_gfe, test_gfe
gc.collect()

print(train_lasso.columns)

for col in train_lasso.columns:
    train_df[col] = train_lasso[col].values
    test_df[col] = test_lasso[col].values

del train_lasso, test_lasso
gc.collect()


# In[13]:


##feature importance
#based on feature importance, decrease memory usage
feature_imp = pd.read_csv('../input/feature-importance/feature_importance.csv')
cols = feature_imp.loc[feature_imp['Value']>50, 'Feature']

train_df = train_df[list(cols)+['diabetes_mellitus', 'encounter_id']]
test_df = test_df[list(cols)+['encounter_id']]
gc.collect()

feature_cols = list(train_df)
feature_cols.remove('diabetes_mellitus')
feature_cols.remove('encounter_id')

cats = [f for f in cats if f in feature_cols]
cont_cols = feature_cols
cont_cols = [f for f in cont_cols if f not in cats]

print("Number of input variables:", len(cont_cols+cats))

X=train_df[feature_cols]
y=train_df['diabetes_mellitus']

params_lgb ={
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.015,
        'subsample': 1,
        'colsample_bytree': 0.2,
        'reg_alpha': 3,
        'reg_lambda': 1,
        'verbose': 1,
        'max_depth': -1,
        'seed':100,
        }


cat_cols = ['icu_id']
if os.path.isfile('../input/feature-importance/feature_importance.csv'):
    feature_imp = pd.read_csv('../input/feature-importance/feature_importance.csv')
    plt.figure(figsize=(16, 10))
    sns.set(font_scale=1.1)
    sns.barplot(
        color="#3498db",
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False)[0:50],
    )
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
else:
    dtrain = lgb.Dataset(X, y)
    model_lgb = lgb.train(params_lgb, dtrain, 500, valid_sets=(dtrain),valid_names=('train'),
                          verbose_eval=20, early_stopping_rounds=100)

    feature_imp = pd.DataFrame(
            {
                "Value": model_lgb.feature_importance(importance_type="gain"),
                "Feature": X.columns,
            })

    feature_imp=feature_imp.sort_values(by='Value', ascending=False)
    feature_imp.to_csv('feature_importance.csv', index=False)
    print(feature_imp.head())

    plt.figure(figsize=(16, 10))
    sns.set(font_scale=1.1)
    sns.barplot(
        color="#3498db",
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False)[0:50],
    )
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()


# In[14]:


### training model
models_list = []
preds = []
y_folds = []
feature_importance = [160]
eval_methods = ['kf']
params = [params_lgb]

if os.path.isfile('models_lgb.pkl'):
    with open('models_lgb.pkl', 'rb') as handle:
        models_list = pickle.load(handle)

    i = 0
    for param in range(len(params)):
        for num in feature_importance:
            for method in eval_methods:
                submit_lgb = Kaggle_submission('submission_lgb_gfe_lasso_%s_%s_%s.csv' % (method, num, param), models_list[i], test_df,
                                               test_df['encounter_id'].tolist(), feature_imp, num)
                i += 1

else:
    for p, param in enumerate(params):
        for eval in eval_methods:
            for num_feat in feature_importance:
                cols = feature_imp.loc[feature_imp['Value']>num_feat, 'Feature']
                cat_cols = [f for f in cat_cols if f in cols]
                models, y_preds = CV(X[cols], y, param, eval_method=eval, NFolds=10, num_iter=20000)

                print('AUC of %s eval method with %s features type params %s' % (eval, num_feat, p), roc_auc_score(train_df['diabetes_mellitus'].values, y_preds))
                submit_lgb = Kaggle_submission('submission_lgb_gfe_lasso_%s_%s_%s.csv' % (eval, num_feat, p), models, test_df,
                                               test_df['encounter_id'].tolist(), feature_imp, num_feat)

                train_df['pred'] = y_preds
                train_df[['diabetes_mellitus', 'pred']].to_csv('oof_lgb_gfe_lasso_%s_%s_%s.csv' % (eval, num_feat, p), index=False)


                models_list.append(models)
        
del models_list, y_preds, submit_lgb, X, y
gc.collect()


# # The best catboost training 

# In[15]:


from catboost import Pool
from catboost import CatBoostClassifier


# In[16]:


def model_train(X_train, y_train, X_test, y_test,  params_cat):
    dtrain = Pool(X_train, y_train, cat_features=cat_cols)
    dvalid = Pool(X_test, y_test, cat_features=cat_cols)
    model_cat = CatBoostClassifier(**params_cat)
    model_cat.fit(dtrain, eval_set=dvalid, use_best_model=True, verbose=20)
    return model_cat

def CV(X, y, params_cat, eval_method, shuffle=True, NFolds=3):

    models = []
    y_folds = np.zeros((X.shape[0], 1))
    y_preds = np.zeros((X.shape[0], 1))

    I=0

    if eval_method=="kf":
        skf = KFold(n_splits=NFolds, random_state=42, shuffle=shuffle)
        for train_index, test_index in tqdm(skf.split(X, y)):
            print('[Fold %d/%d]' % (I + 1, NFolds))
            print('=' * 60)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model_cat = model_train(X_train, y_train, X_test, y_test, params_cat)

            preds = model_cat.predict(X_test, prediction_type='Probability')[:,1]

            y_folds[test_index, 0] = y_test
            y_preds[test_index, 0] = preds
            models.append(model_cat)

            I += 1

    return models, y_preds


def Kaggle_submission(file_name, models, test_data,
                      ids_list, feature_imp, num):

    submit = pd.DataFrame()
    submit['encounter_id'] = ids_list
    submit['diabetes_mellitus'] = 0

    cols = feature_imp.loc[feature_imp['Value']>num, 'Feature']
    nfolds = len(models)
    for model in models:
        test_pred = model.predict(test_data[cols], prediction_type='Probability')[:,1]
        submit['diabetes_mellitus'] += (test_pred / nfolds)

    submit.to_csv(file_name, index=False)

    return submit


# In[17]:


for col in cats:
    train_df[col] = train_df[col].astype('int')
    test_df[col] = test_df[col].astype('int')

for col in cont_cols:
    df = pd.concat([train_df[col], test_df[col]])
    train_df[col] = train_df[col].fillna(df.median())
    test_df[col] = test_df[col].fillna(df.median())

del df
X=train_df[feature_cols]
y=train_df['diabetes_mellitus']


# In[18]:


params_cat =  {
             'n_estimators' : 10000,
            'learning_rate': 0.02,
             'eval_metric' : 'AUC',
             'early_stopping_rounds': 100,
            'loss_function': 'Logloss',
            'depth': 8,
            'bootstrap_type': 'Bernoulli',
            'class_weights': [0.25, 0.75],
             'task_type':'GPU'
            }


# In[19]:


cat_cols = cats
models_list = []
preds = []
y_folds = []
num_feats = [150]
methods = ['kf']

if os.path.isfile('models_cat.pkl'):
    with open('models_cat.pkl', 'rb') as handle:
        models_list = pickle.load(handle)

    i = 0
    for num in num_feats:
        for method in methods:
            submit_cat = Kaggle_submission('submission_cat_gfe_lasso_%s_%s.csv' % (method, num), models_list[i], test_df,
                                           test_df['encounter_id'].tolist(), feature_imp, num)
            i += 1

else:
    i = 0
    for num_feat in num_feats:
        cols = feature_imp.loc[feature_imp['Value']>num_feat, 'Feature']
        cat_cols = [f for f in cat_cols if f in cols]
        models, y_preds = CV(X[cols], y, params_cat, eval_method='kf', shuffle=True, NFolds=10)

        print('AUC with %s features' % (num_feat), roc_auc_score(train_df['diabetes_mellitus'].values, y_preds))

        train_df['pred'] = y_preds
        train_df[['diabetes_mellitus', 'pred']].to_csv('oof_cat_gfe_lasso_%s.csv' % (num_feat), index=False)

        models_list.append(models)

        for method in methods:
            submit_cat = Kaggle_submission('submission_cat_gfe_lasso_%s_%s.csv' % (method, num_feat), models_list[i], test_df,
                                           test_df['encounter_id'].tolist(), feature_imp, num_feat)
            i += 1

del models_list, submit_cat, X, y, y_preds
gc.collect()


# # The best Neural Network Model

# In[20]:


from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
from torch.autograd import Variable
import random

device = ('cuda' if torch.cuda.is_available() else 'cpu')


# In[21]:


def normalization(df):
    if len(df.shape)==2:
        scaler = MinMaxScaler().fit(df.values)
        X = scaler.transform(df.values)
    else:
        scaler = MinMaxScaler().fit(np.expand_dims(df.values, axis=1))
        X = scaler.transform(np.expand_dims(df.values, axis=1))

    return X, scaler

def make_2Dinput(dt, cont_cols, cat_cols):
    input = {"cont": dt[cont_cols].to_numpy()}
    for i, v in enumerate(cat_cols):
        input[v] = dt[[v]].to_numpy()
    return input

class Loader:
    def __init__(self, X, y, shuffle=True, batch_size=1000, cat_cols=[]):
        self.X_cont = X["cont"]
        try:
            self.X_cat = np.concatenate([X[k] for k in cat_cols], axis=1)
        except:
            self.X_cat = np.concatenate([np.expand_dims(X[k], axis=1) for k in cat_cols], axis=1)
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_conts = self.X_cont.shape[1]
        self.len = self.X_cont.shape[0]
        n_batches, remainder = divmod(self.len, self.batch_size)

        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        self.remainder = remainder  # for debugging
        self.idxes = np.array([i for i in range(self.len)])

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            ridxes = self.idxes
            np.random.shuffle(ridxes)
            self.X_cat = self.X_cat[[ridxes]]
            self.X_cont = self.X_cont[[ridxes]]
            if self.y is not None:
                self.y = self.y[[ridxes]]

        return self

    def __next__(self):
        if self.i >= self.len:
            raise StopIteration

        if self.y is not None:
            y = torch.FloatTensor(self.y[self.i:self.i + self.batch_size].astype(np.float32))

        else:
            y = None

        xcont = torch.FloatTensor(self.X_cont[self.i:self.i + self.batch_size])
        xcat = torch.LongTensor(self.X_cat[self.i:self.i + self.batch_size])

        batch = (xcont, xcat, y)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def ensemble_structure(number_models, num_features, MODEL_ROOT):

    if number_models == 1:

        hidden_dims = [256]
        number_of_dims = [num_features]
        input_dims = [np.arange(num_features)]

    elif os.path.isfile('hidden_dims.pkl'):
        with open('hidden_dims.pkl', 'rb') as handle:
            hidden_dims = pickle.load(handle)

        with open('number_of_dims.pkl', 'rb') as handle:
            number_of_dims = pickle.load(handle)

        with open('input_dims.pkl', 'rb') as handle:
            input_dims = pickle.load(handle)

    else:

        hidden_dims = np.random.randint(128, 256, number_models)
        number_of_dims = np.random.randint(int(num_features*0.75), num_features+1, number_models)
        input_dims = []
        for i in range(number_models):
            input_dims.append(np.random.randint(0, num_features, number_of_dims[i]))

        with open('hidden_dims.pkl', 'wb') as handle:
            pickle.dump(hidden_dims, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('number_of_dims.pkl', 'wb') as handle:
            pickle.dump(number_of_dims, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('input_dims.pkl', 'wb') as handle:
            pickle.dump(input_dims, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return hidden_dims, number_of_dims, input_dims

class WeightedFocalLoss(nn.Module):
    "Weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


# ****Neural Network Model

# In[22]:


############## MLP ####################################
class simple_linear_layer(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(simple_linear_layer, self).__init__()
        self.dense = nn.Linear(input_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.dropout =nn.Dropout(0.20)


    def forward(self, x):

        x = self.dense(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        return x

class MLP_Model(nn.Module):

    def __init__(self, input_dim, hidden_dim, out_dim=1):
        super(MLP_Model, self).__init__()

        self.block1 = simple_linear_layer(input_dim, hidden_dim)
        self.block2 = simple_linear_layer(hidden_dim, hidden_dim)
        self.block3 = simple_linear_layer(hidden_dim, hidden_dim)
        self.block4 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):

        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        out = self.block4(h)

        return out
    
####################### Final Ensemble Model ######################################
class Ensemble_Model(nn.Module):

    def __init__(self, input_dims, emb_dims, number_of_dims, hidden_dims,
                 model_name, out_dim=1):
        super(Ensemble_Model, self).__init__()

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        n_embs = sum([y for x, y in emb_dims])

        self.models = torch.nn.ModuleList()
        self.input_dims = input_dims
        self.model_name = model_name

        for i in range(len(hidden_dims)):
            if self.model_name=='Simple_MLP':
                self.models.append(MLP_Model(input_dim=number_of_dims[i]+n_embs,
                                                    hidden_dim=hidden_dims[i], out_dim=out_dim))

                print("Please check model name. There is no this model!!!")

        # train embedding layers and concat cat and cont variables

    def encode_and_combine_data(self, cat_data):
        xcat = [el(cat_data[:, k]) for k, el in enumerate(self.emb_layers)]
        xcat = torch.cat(xcat, 1)

        return xcat

    def forward(self, cont_data, cat_data):

        xcat = self.encode_and_combine_data(cat_data)

        out = []
        for i in range(len(self.input_dims)):
            temp = self.models[i](torch.cat([cont_data[:, self.input_dims[i]], xcat], dim=1))
            out.append(temp.unsqueeze(0))

        out = torch.cat(out, dim=0)
        out = out.permute(1, 0, 2)

        return out


# In[23]:


def training_nn(df, test, target, cont_cols, cat_cols, MLP_model,
                Nfolds, epoch=50, patience=5, MODEL_ROOT='models',
                hidden_dim=1024):

    uniques = {}
    dims = {}
    dt = pd.concat([df[cat_cols], test[cat_cols]], axis=0)
    for i, v in enumerate(cat_cols):
        uniques[v] = len(dt[v].unique())
        dims[v] = min(int(len(dt[v].unique())/2), 16)

    if not os.path.exists(MODEL_ROOT):
        os.makedirs(MODEL_ROOT)

    kfolds = KFold(n_splits=Nfolds, random_state=42, shuffle=True)

    final_test = np.zeros((test.shape[0],))
    oof_train = np.zeros((df.shape[0], 1))

    for i, (trn_ind, val_ind) in tqdm(enumerate(kfolds.split(X=df,
                                                             y=df[target]))):

        print('[Fold %d/%d]' % (i + 1, Nfolds))
        print('=' * 60)
        # Split Train/Valid
        train = df.loc[trn_ind]
        valid = df.loc[val_ind]

        model_path = MODEL_ROOT + '/model_nn_%s_%s.pt' % (hidden_dim, i)

        # Make input for pytorch loader because we have categorical and continues features
        X_train, y_train = make_2Dinput(train[cont_cols + cat_cols], cont_cols=cont_cols, cat_cols=cat_cols), train[target]
        validx, validy = make_2Dinput(valid[cont_cols + cat_cols], cont_cols=cont_cols, cat_cols=cat_cols), valid[target]

        # Make loader for pytorch
        train_loader = Loader(X_train, y_train.values, cat_cols=cat_cols, batch_size=1024, shuffle=True)
        val_loader = Loader(validx, validy.values, cat_cols=cat_cols, batch_size=2000, shuffle=False)

        if os.path.isfile(model_path):
            model_final = torch.load(model_path)
            y_pred = []
            with torch.no_grad():
                model_final.eval()
                for i, (X_cont, X_cat, y) in enumerate(val_loader):
                    out = model_final(X_cont.to(device), X_cat.to(device))
                    y_pred += list(np.mean(out.sigmoid().detach().cpu().numpy(), axis=1).flatten())

            oof_train[val_ind, 0] = y_pred

        else:
            ## make embedding dimensions
            emb_dims = [(uniques[col], dims[col]) for col in cat_cols]

            # number of continues variables
            n_cont = train_loader.n_conts

            # # neural network model
            hidden_dims, number_of_dims, input_dims = ensemble_structure(1,
                                                                         n_cont,
                                                                         MODEL_ROOT)

            model = Ensemble_Model(input_dims=input_dims, emb_dims=emb_dims,
                                   number_of_dims=number_of_dims,
                                   hidden_dims=hidden_dims,
                                   model_name='Simple_MLP', out_dim=1).to(device)

            # loss function
            criterion = WeightedFocalLoss()#nn.BCEWithLogitsLoss()

            # adam optimizer has been used for training
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
            # learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.1, patience=3,
                                                            min_lr=0.00001, verbose=True)



            best_auc = 0
            counter = 0
            for ep in range(epoch):
                train_loss, val_loss = 0, 0
                # training phase for single epoch
                model.train()
                for i, (X_cont, X_cat, y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    out = model(X_cont.to(device), X_cat.to(device))

                    loss = 0 # criterion(out, y.unsqueeze(1).to(device)) #+ F.mse_loss(X_cont.to(device), x_hat)
                    for i in range(out.shape[1]):
                        loss = loss + criterion(out[:, i, :],
                                              y.unsqueeze(1).to(device))

                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        train_loss += loss.item() / len(train_loader)

                # Validation phase for single epoch
                phase = 'Valid'
                with torch.no_grad():
                    model.eval()
                    y_true = []
                    y_pred = []
                    rloss = 0
                    for i, (X_cont, X_cat, y) in enumerate(val_loader):
                        out = model(X_cont.to(device), X_cat.to(device))

                        loss = 0 #criterion(out, y.unsqueeze(1).to(device)) #+ F.mse_loss(X_cont.to(device), x_hat)
                        for i in range(out.shape[1]):
                            loss = loss + criterion(out[:, i, :],
                                                    y.unsqueeze(1).to(device))

                        rloss += loss.item() / len(val_loader)
                        y_pred += list(np.mean(out.sigmoid().detach().cpu().numpy(), axis=1).flatten())
                        y_true += list(y.cpu().numpy())

                    AUC= roc_auc_score(y_true, y_pred)
                    scheduler.step(AUC)
                    print(
                        f"[{phase}] Epoch: {ep} | Tain loss: {train_loss:.4f} | Val Loss: {rloss:.4f} | AUC: {AUC:.4f} ")

                    if best_auc < AUC:
                        best_auc = AUC
                        best_model = model
                        torch.save(best_model, model_path)
                        counter = 0
                        oof_train[val_ind, 0] = y_pred
                    else:
                        counter = counter + 1

                # early stopping
                if counter >= patience:
                    print("Early stopping")
                    break

        # call the best model for each fold
        model_final = torch.load(model_path)

        # make prediction for test set
        testx = make_2Dinput(test[cont_cols + cat_cols], cont_cols=cont_cols, cat_cols=cat_cols)
        test_loader = Loader(testx, None, cat_cols=cat_cols, batch_size=len(testx), shuffle=False)
        y_pred = []

        with torch.no_grad():
            model_final.eval()
            for i, (X_cont, X_cat, y) in enumerate(test_loader):
                out = model_final(X_cont.to(device), X_cat.to(device))
                y_pred += list(np.mean(out.sigmoid().detach().cpu().numpy(), axis=1).flatten())

        final_test += np.asarray(y_pred) / Nfolds

    return final_test, oof_train


# In[24]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(seed=42)

feature_imp = pd.read_csv('../input/feature-importance/feature_importance.csv')
feature_cols = list(feature_imp.loc[feature_imp['Value']>150, 'Feature'])

### data import

cat_cols = ['elective_surgery', 'icu_id', 'arf_apache', 'intubated_apache', 'ventilated_apache', 'cirrhosis',
        'hepatic_failure', 'immunosuppression', 'leukemia', 'solid_tumor_with_metastasis', 'apache_3j_diagnosis_x',
        'apache_2_diagnosis_x', 'apache_3j', 'apache_3j_diagnosis_split1', 'apache_2_diagnosis_split1', 'gcs_sum_type',
        'hospital_admit_source', 'glucose_rate', 'glucose_wb', 'gcs_eyes_apache', 'glucose_normal', 'total_cancer_immuno',
        'gender', 'total_chronic', 'icu_stay_type', 'apache_2_diagnosis_type', 'apache_3j_diagnosis_type']

cat_cols = [f for f in cat_cols if f in feature_cols]
cont_cols = feature_cols
cont_cols = [f for f in cont_cols if f not in cat_cols]

print("Number of input variables:", len(cont_cols+cat_cols))

train_df[cont_cols], scalerX = normalization(train_df[cont_cols])
test_df[cont_cols] = scalerX.transform(test_df[cont_cols].values)

#### Model Training
nn_pred, nn_oof = training_nn(df=train_df, test=test_df,  target='diabetes_mellitus', MLP_model=Ensemble_Model,
                                cont_cols=cont_cols, cat_cols=cat_cols, Nfolds=10,
                                epoch=500, patience=10, MODEL_ROOT='models',
                                hidden_dim=512)

print(roc_auc_score(train_df['diabetes_mellitus'].values, nn_oof))
test_df['diabetes_mellitus']=nn_pred
test_df[['encounter_id', 'diabetes_mellitus']].to_csv('submission_nn_gfe_220.csv', index=False)

train_df['pred'] = nn_oof
train_df[['diabetes_mellitus', 'pred']].to_csv('oof_nn_gfe_220.csv', index=False)


# # Ensemble

# In[25]:


sub_lgb = pd.read_csv('./submission_lgb_gfe_lasso_kf_160_0.csv')
sub_cat = pd.read_csv('./submission_cat_gfe_lasso_kf_150.csv')
sub_nn = pd.read_csv('./submission_nn_gfe_220.csv')

sub = sub_lgb.copy()
sub['diabetes_mellitus']=0
sub['diabetes_mellitus'] = 0.4*sub_lgb['diabetes_mellitus'].values + 0.3*sub_cat['diabetes_mellitus'].values + 0.3*sub_nn['diabetes_mellitus'].values

sub.to_csv('submission_ensemble.csv', index=False)

