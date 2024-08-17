#!/usr/bin/env python
# coding: utf-8

# # Random Forest rank prediction
# 
# In this kernel I'll check one interesting idea - can we improve AUC score, when we predict ranks, not probabilities from every tree?

# In[1]:


import numpy as np
import pandas as pd
from tqdm import tqdm
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold


# In[2]:


# df = pd.read_csv("/kaggle/input/playground-series-s3e7/train.csv")
# df_test = pd.read_csv("/kaggle/input/playground-series-s3e7/test.csv")

# # remove leak data
# res_leak = df.merge(df_test, how="inner", on=df.drop(["id", "booking_status"], axis=1).columns.tolist())
# df = df[~df["id"].isin(res_leak["id_x"].tolist())]

# # remove full duplicates from train data
# df = df.drop_duplicates(subset=df.drop(["id", "booking_status"], axis=1).columns.tolist())

# cat_cols = ["market_segment_type", "type_of_meal_plan", "type_of_meal_plan", "type_of_meal_plan", "repeated_guest"]


# In[3]:


df = pd.read_csv("/kaggle/input/playground-series-s3e12/train.csv")
df = df.drop("id", axis=1)
df_tst = pd.read_csv('/kaggle/input/playground-series-s3e12/test.csv')


# In[4]:


train_index, test_index = (np.array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
        52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
        65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
        78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
        91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
       104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
       117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
       130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
       143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
       156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
       169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
       182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
       195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
       208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
       221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
       234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
       247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259,
       260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272,
       273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
       286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
       299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
       312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324,
       325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337,
       338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350,
       351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363,
       364, 365, 366, 367, 368, 369, 370, 371, 372]), np.array([373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385,
       386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398,
       399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411,
       412, 413]))


# # Feature engineering

# In[5]:


# def feature_preprocess(X: pd.DataFrame) -> pd.DataFrame:
#     """Feature preprocessing process
#     """
#     cat_cols = ["market_segment_type"]
#     X["all_no"] = X["no_of_adults"] + X["no_of_children"]
#     X["all_nights"] = X["no_of_weekend_nights"] + X["no_of_week_nights"]
#     X["avg_price_per_night"] = X.apply(lambda x: x["avg_price_per_room"]/x["all_nights"]
#                                        if x["all_nights"]!=0 else -1, axis=1)
#     for col in cat_cols:
#         X[col] = X[col].astype("category")
#     return X

# df = df.drop(["id", "arrival_year", "arrival_month", "arrival_date"], axis=1)
# df_test = df_test.drop(["arrival_year", "arrival_month", "arrival_date"], axis=1)

# for dataset in [df, df_test]:
#     dataset = feature_preprocess(dataset)


# # Standard Random Forest

# In[6]:


class MyRandomForestClassifier:
    """Simple Random Forest class implementation
    """
    def __init__(
        self,
        n_estimators=150,
        max_depth=6,
        max_features="auto",
        random_state=813
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.tree_list = []
        self.classes_ = np.array([0,1])

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.tree_list = []

        for k in range(self.n_estimators):
            bagging_subs = np.random.choice(X.shape[0], X.shape[0], replace=True)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=self.random_state
            )
            tree.fit(X.iloc[bagging_subs], y.iloc[bagging_subs])

            self.tree_list.append(tree)

    def predict_proba(self, X):
        """Predict proba standard process
        """
        tree_count = len(self.tree_list)
        prediction = np.zeros(X.shape[0])

        for tree in self.tree_list:
            raw_prediction = tree.predict_proba(X)[:,1]
            prediction += raw_prediction
        return prediction / tree_count
    
    def predict_rank(self, X):
        """Predict normalize rank values
        """
        tree_count = len(self.tree_list)
        prediction = np.zeros(X.shape[0])

        for tree in self.tree_list:
            raw_prediction = tree.predict_proba(X)[:,1]
            # create rank normalize prediction
            raw_prediction = pd.DataFrame(raw_prediction)
            raw_prediction["rank"] = raw_prediction.rank(method="min")
            # raw_prediction["new_pred_rank"] = raw_prediction["rank"] / raw_prediction.shape[0]
            prediction += raw_prediction["rank"].values
                
        return prediction / tree_count


# ### Kfold scoring

# In[7]:


# kf = KFold(n_splits=10)
# kf_scores = []
# for i, (train_index, test_index) in tqdm(enumerate(kf.split(df))):
    
#     fold_df = df.iloc[train_index]
#     fold_test = df.iloc[test_index]
    
#     fold_df_train = fold_df.drop("booking_status", axis=1)
#     fold_df_target = fold_df["booking_status"]
    
#     encoder = ce.cat_boost.CatBoostEncoder(cat_cols)
#     encoder.fit(fold_df_train, fold_df_target)
    
#     clf = MyRandomForestClassifier(n_estimators=300, max_features="sqrt")
#     clf.fit(encoder.transform(fold_df_train), fold_df_target)
#     kf_scores.append(
#         roc_auc_score(
#             fold_test["booking_status"],
#             clf.predict_proba(
#                 encoder.transform(fold_test.drop("booking_status", axis=1)))
#     ))


# ### AUC score mean +- std

# In[8]:


# print(f"AUC score: {np.mean(kf_scores):.3f} +- {np.std(kf_scores):.3f}")


# # Random Forest with rank prediction

# In[9]:


# kf = KFold(n_splits=10)
# kf_my_scores = []
# for i, (train_index, test_index) in tqdm(enumerate(kf.split(df))):
    
#     fold_df = df.iloc[train_index]
#     fold_test = df.iloc[test_index]
    
#     fold_df_train = fold_df.drop("target", axis=1)
#     fold_df_target = fold_df["target"]
    
#     # encoder = ce.cat_boost.CatBoostEncoder(cat_cols)
#     # encoder.fit(fold_df_train, fold_df_target)
    
#     clf = MyRandomForestClassifier(n_estimators=100, max_depth=4)
#     clf.fit(fold_df_train, fold_df_target)
#     # clf.fit(encoder.transform(fold_df_train), fold_df_target)
#     kf_my_scores.append(
#         roc_auc_score(
#             fold_test["target"],
#             clf.predict_rank(fold_test.drop("target", axis=1))
#     #        clf.predict_rank(
#     #            encoder.transform(fold_test.drop("booking_status", axis=1)))
#     ))


# ### AUC score mean +- std
# 

# In[10]:


# print(f"AUC score: {np.mean(kf_my_scores):.3f} +- {np.std(kf_my_scores):.3f}")


# In[11]:


fold_df = df.iloc[train_index]
fold_test = df.iloc[test_index]
fold_df_train = fold_df.drop("target", axis=1)
fold_df_target = fold_df["target"]

clf = MyRandomForestClassifier()
clf.fit(fold_df_train, fold_df_target)


# In[12]:


df_subs = pd.DataFrame()
df_subs["id"] = df_tst["id"]
df_subs["target"] = clf.predict_rank(df_tst.drop("id", axis=1))
df_subs.to_csv("submission_baseline_v2.csv", index=False)

