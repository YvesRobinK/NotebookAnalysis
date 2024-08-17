#!/usr/bin/env python
# coding: utf-8

# # Dimensionality Reduction w. UMAP, Hyperparameter Tuning w. Optuna, XAI with SHAP
# ![](https://images.pexels.com/photos/2859169/pexels-photo-2859169.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940)
# <figcaption style="text-align: center;">
#     <strong>
#         Photo by 
#         <a href='https://www.pexels.com/@andrew?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels'>Andrew Neel</a>
#         on 
#         <a href='https://www.pexels.com/photo/assorted-map-pieces-2859169/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels'>Pexels.</a>
#     </strong>
# </figcaption>

# ## Setup

# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import warnings
import xgboost as xgb
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import QuantileTransformer
import shap
import umap
import umap.plot
import joblib

train = pd.read_csv("../input/tabular-playground-series-aug-2021/train.csv").drop("id", axis=1).fillna(0)
test = pd.read_csv("../input/tabular-playground-series-aug-2021/test.csv").drop("id", axis=1).fillna(0)
submission = pd.read_csv("../input/tabular-playground-series-aug-2021/sample_submission.csv")

X, y = train.drop("loss", axis=1), train[['loss']]

plt.style.use("ggplot")
warnings.filterwarnings("ignore")


# I guess we can all agree that this month's TPS is a bit boring. I think that is mainly because of the low prospects of doing effective feature engineering to imrpove the score. We are mostly left with hyperparameter tuning, which honestly makes this whole competition a matter of who has got the most time and compute. 
# 
# I have been doodling around recently with UMAP and I wanted to share with you my insights in this notebook. Though my experiments didn't spark substantial ideas for me, maybe they can aid you in coming up with some ideas.

# # Reduction with UMAP

# To make computations less time consuming, we will take 30k samples and try to project and plot them in 2D with UMAP:

# In[2]:


import umap, umap.plot

sample = train.sample(30000, replace=False)


# Let's try UMAP with 1000 neighbors (this is the number I ended up with after playing around a bit). Essentially, `n_neigbors` in UMAP controls the zoom level of the projection. 

# In[3]:


# Project to 2d
sample_X, sample_y = sample.iloc[:, :-1], sample.iloc[:, -1]

mapper_2d = umap.UMAP(n_neighbors=1000).fit(sample_X, sample_y)

# Plot
umap.plot.points(mapper_2d, labels=sample_y, theme='fire');


# Even though a bit beautiful, this plot does not tell much. The plot is a long thread of the data points dominated by the small values, especially 0s. If you you pay a little more attention, you can see some discontinuity in the thread on the left side. 
# 
# In simple terms, UMAP uses linear distance between the points which makes it highly sensitive to feature scales. Let's try the same operation by scaling all features with `StandardScaler`:

# In[4]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.preprocessing import StandardScaler\n\n# Scale\nsample_X.iloc[:, :] = StandardScaler().fit_transform(sample_X)\n\n# Create a new embedding\nmapper_2d = umap.UMAP(n_neighbors=1000).fit(sample_X, sample_y)\n\n# Plot\numap.plot.points(mapper_2d, labels=sample_y, theme='fire');\n")


# Well, this did the trick. We can see that the biggest clusters correspond to the low vlaues in the target. In fact, from 0 to about 20 `loss` values, the clusters are pretty distinct. The higher the `loss`, the less grouped the points are.
# 
# To see if this new-found insight from the visualization translated to a score improvement, I tried projecting the whole data. Unfortunately, the kernel ran out of RAM every time I tried, so I leave this experiment for those with a machine with a larger RAM. 

# # Hyperparameter tuning with Optuna

# In this section, we will peprform XGBoost hyperparameter tuning with Optuna and plot the search history to explore hyperparameter importances. 
# 
# I will be using `QuantileTransformer` as mentioned in this [notebook](https://www.kaggle.com/oxzplvifi/tabular-denoising-residual-network). It is found to work best with those bimodal/trimodal and skewed features observed in my previous [EDA notebook](https://www.kaggle.com/bextuychiev/relevant-eda-xgboost):

# In[50]:


from sklearn.preprocessing import QuantileTransformer

# Define the objective function
def objective(trial, X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.3, random_state=1121218)
    qt = QuantileTransformer(random_state=1121218)
    X_train.iloc[:, :] = qt.fit_transform(X_train)
    X_valid.iloc[:, :] = qt.transform(X_valid)
    param = {
        "tree_method": "gpu_hist",
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000, 100),
        "booster": 'gbtree',
        "reg_lambda": trial.suggest_int("reg_lambda", 1, 100),
        "reg_alpha": trial.suggest_int("reg_alpha", 1, 100),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0, step=0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        "learning_rate": 0.01,
        "gamma": trial.suggest_float("gamma", 0, 20)
    }
    # Set up the CV
    eval_set = [(X_valid, y_valid)]
    fit_params = dict(eval_set=eval_set, eval_metric='rmse', 
                      early_stopping_rounds=100, verbose=False)
    xgb_reg = xgb.XGBRegressor(**param)
    # Fit/predict
    _ = xgb_reg.fit(X_train, y_train)
    preds = xgb_reg.predict(X_valid)
    # Compute rmse
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    
    return rmse


# In[51]:


# Callback function to print log messages when the best trial is updated
def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            "Trial {} finished with best value: {}. ".format(
            frozen_trial.number,
            frozen_trial.value
            )
        )


# In[52]:


get_ipython().run_cell_magic('time', '', "\nfrom optuna.samplers import TPESampler\nfrom sklearn.model_selection import KFold\noptuna.logging.set_verbosity(optuna.logging.WARNING)\n\nstudy = optuna.create_study(sampler=TPESampler(seed=1121218), direction='minimize', study_name='xgb')\nfunc = lambda trial: objective(trial, X, y)\n\nstudy.optimize(func, timeout=60*30, callbacks=[logging_callback])\n")


# In[57]:


print("Best trial: 17")
print(f"\twith value: {study.best_value:.5f}")
print(f"\tBest params:")
for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")


# In[58]:


# Save the study
import joblib

joblib.dump(study, "xgb_study.pkl")


# We have retrieved the best set of parameters with a best score of 7.8395. These parameters will the base for what will be doing in model explainability section with SHAP and ELI5.

# # Submitting predictions with the found parameters

# In[3]:


study = joblib.load("../input/new-insights-from-umap-optuna-shap/xgb_study.pkl")

# New regressor with the optimal parameters
final_xgb = xgb.XGBRegressor(
    **study.best_params, tree_method='gpu_hist', learning_rate=0.01, booster='gbtree'
)

# Extract 5% of the training data for early stopping
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.05, random_state=1121218)

# Apply the scaler to the sets
qt = QuantileTransformer()
X_train_scaled = qt.fit_transform(X_train)
X_valid_scaled = qt.transform(X_valid)
test_scaled = qt.transform(test)

eval_set = [(X_valid_scaled, y_valid)]

# Train / predict
_ = final_xgb.fit(
    X_train_scaled, 
    y_train, eval_metric='rmse', 
    eval_set=eval_set, 
    early_stopping_rounds=100, 
    verbose=False
)

preds = final_xgb.predict(test_scaled)

# Submit
final_sub = pd.DataFrame({"id": submission.id, "loss": preds})
final_sub.to_csv("submission.csv", index=False)


# # Exploring the Optuna study for more insight into hyperparameters

# Let's plot the optimization history first:

# In[71]:


from optuna.visualization.matplotlib import plot_optimization_history

plot_optimization_history(study)
fig = plt.gcf()
fig.set_size_inches(10, 6)


# On the second thought, a more useful plot would be the hyperpamater importances:

# In[77]:


from matplotlib import rcParams
from optuna.visualization.matplotlib import plot_param_importances

rcParams['figure.figsize'] = 10, 6
plot_param_importances(study);


# According to this plot, `colsample_bytree` and `min_child_weight` had little influence over the objective function of the study. For future reference and tuning, you may get faster and better results by excluding those hyperparameters from the search and giving a larger serach interval to more important ones.

# # Model Explainability with ELI5 and SHAP

# Since we have 100 features, it will be useful to see which features influence the predictions the most.
# 
# For that, we will be using a permutation importance (PI) plot, because it gives a more robust information than simple feature importances or coefficients that the model came up with:

# In[10]:


get_ipython().run_cell_magic('time', '', '\nimport eli5\nfrom eli5.sklearn import PermutationImportance\n\nperm = PermutationImportance(final_xgb, random_state=1).fit(X_valid_scaled, y_valid)\neli5.show_weights(perm, feature_names = X.columns.tolist())\n')


# Here is a simple explanation of how PI works:
# 1. One feature is chosen and its values are shuffled while others are left fixed
# 2. This new set of features is given to the already fitted model
# 3. Model makes new predictions on these new features
# 4. The new predictions are compared to predictions made with the original set of features. For more info, check out the Kaggle course [here](https://www.kaggle.com/dansbecker/permutation-importance).
# 5. If a shuffled feature is important, then it will have a significant impact on models predictions because shuffling it makes it a useless feature for the model, decreasing its predictive power.
# 
# From the above PI plot, we can see that there aren't specific features that have higher influence than others. If I had to pick though, I guess, the top 5 - f81, f52, f69, f77, f25 would be my choices. Shuffling these features would hurt the accuracy of the predictions  slightly more than other features. 
# 
# We can confirm this by computing the Shapley values and plotting them:

# In[10]:


import shap

# Choose a smaller subset of the validation data
small_valid = X_valid_scaled[:200]

# Create the explainer
explainer = shap.TreeExplainer(final_xgb)
shap_values = explainer.shap_values(small_valid)

shap.summary_plot(shap_values, small_valid);


# Here is how to interpret this plot:
# - Each dot represents a single row from the data
# - The Y axis determines which feature the dot belongs to
# - The X axis determines the Shapley value for that point and how much it influenced the prediction
# - The color represents actual value of the point, as it appears in the dataset
# 
# This summary plot reveals an interesting global trend for each feature. When some features like the bottom 4, f13, f28, f39 increase, they increasingly have more positive impact on the model output. In contrast, an increase in others like 52, 3, 58 works against the model output. In other words, an increase/decrease in the values of the feature either postively or negatively influences the model's decision. There are no overlaps.
# 
# The only incosistent feature with this trend is feature 81 because it has some outliers (red dots farthest away from the rest).
# 
# Now, for the sake of completeness, let's select a random prediction and look at the features that most influenced the model's decision to produce that particular output:

# In[14]:


# Select a random row like 17
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[17, :], small_valid[17, :], feature_names=X.columns.tolist())


# For the prediction of row 17, feature 25, 9, 3 had the most positive impact while 84, 51, 46 had the most negative. 

# # Summary

# Honestly, I went into this section expecting more. I wanted to generate a few potential ideas for feature engineering but I didn't get any.
# 
# Maybe, I misinterpreted the plots and missed something. Other than this, I currently don't see any other way we could go around the dataset and come up with something that does not involve pure model and tuning. Maybe I would have gotten somewhere with UMAP if not for the hardware limitations.
# 
# Let me know what you guys think and if you did experiments with UMAP or SHAP, or you know, anyhting new.
