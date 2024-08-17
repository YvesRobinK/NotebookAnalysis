#!/usr/bin/env python
# coding: utf-8

# # Learnings From the Typical Tabular Modelling Pipeline
# 
# # Abstract
# Quick, accurate and expert handling of tabular data is key in machine learning. This essay examines the recent tabular data modelling pipelines of successful Kagglers by exploring individual discussion posts and aggregated Kaggle metadata. Our goal is to extract key learnings from these recent high performing tabular data solutions and contrast it to latest developments in literature. 
# 
# 
# # Table of Contents
# 1. [Introduction](#introduction)
# 2. [Key Learnings From the Typical Tabular Modelling Pipeline in Kaggle](#overview)
# 3. [How Do the Insights of Kagglers Compare to Recent Literature](#compare)
# 4. [Observations and Future Directions for Learning](#future)
# 5. [Conclusion](#conclusion)
# 6. [References](#references)
# 7. [Code and Acknowledgements](#code)
# 
# <a id='introduction'></a>
# # 1. Introduction
# Tabular data refers to structured data organized in tables with rows and columns, much like a spreadsheet, which represents distinct entities and their attributes. Time series data is similar however there is a temporal ordering to the points, i.e. the points can be ordered by time such as daily stock prices or monthly rainfall. Both generic tabular data and time series data are frequently used in Kaggle competitions. In this essay we seek to understand the current state of handling such data in both the Kaggle community and in literature. We will do this in three key sections.
# 
# We first break the modelling journey into individual components and identify trends and themes through either qualitative analysis of individual discussion posts or by leveraging Kaggle metadata. We do this paying particular attention to high achieving solutions of tabular data competitions. From this review of Kaggle competitions we identify key learnings in recent Kaggle competitions. 
# 
# Armed with an understanding of a typical high performing modelling pipeline on Kaggle and the learnings this gives us, we are then in a position to compare the learnings from Kaggle to relevant recent literature.
# 
# Lastly, we take the learnings from the previous two sections to recommend areas of future improvement for Kagglers, and possible areas of research for literature. 
# 
# <a id='overview'></a>
# # 2. Key Learnings From the Typical Tabular Modelling Pipeline in Kaggle
# 
# Although every pipeline is different, there are a few key areas we consistently see discussed in write ups. These are as follows.
# 
# - **Feature Engineering**:  Includes understanding and visualizing data, cleaning and preprocessing, normalizing or scaling features, selecting features, and creating new ones.
# - **Model Selection and Tuning**: Involves choosing an algorithm, tuning its hyperparameters, and evaluating performance metrics.
# - **Ensembling**: Pertains to the use of multiple machine learning models and the combination of their predictions to enhance accuracy.
# - **Cross-Validation Scheme**: Refers to the implementation of techniques like K-Fold, Stratified K-Fold, or Time-Series Cross-Validation to ensure model reliability and generalization.
# - **Write Ups**: Unique to Kaggle, competitions typically end with top performing teams sharing their solutions and ideas to the wider community. 
# 
# Let's dive into each of these areas:
# 
# #### Feature Engineering ####
# Feature engineering is the process of creating, selecting, and transforming variables to maximize their potential in predictive models. Here are two write-ups of high performing solutions that provide an outline of their feature engineering process: 
# - **[AMEX 2nd place solution [1]](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347637)**: As the author states this is a good example of a brute-force approach to feature engineering with the process involving aggregations, differences, ratios, and lags of the features. After generating so many features the selection of the most impactful features was then based based on hierarchical permutation importance, stepped permutation importance and forward feature selection resulting in a final subset of approximately 2500 features.
# 
# - **[AMP Parkinson's Disease Progression Prediction 4th place solution [2]](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/discussion/411398)**: In this competition, the most potent source of signal was identified in the patient visit dates rather than the provided protein and peptide data that was the focus of the competition organiser. After learning this key fact the author then engineered hundreds of features based on the visit date data.
# 
# These two submissions were chosen because they represent two important aspects of feature generation - in [1] we see a broad approach to generating thousands of features, while [2] concentrates in taking a single feature and expanding it in a number of ways. These two very different approaches show the importance of adapting to the dataset at hand while conducting feature engineering.
# 
# > Learning 1: Kaggler's understand feature engineering is an important yet dynamic process requiring creativity and flexibility in approach. 
# 
# #### Model Selection ####
# Model selection and tuning are crucial steps in any machine learning pipeline, involving the choice of an algorithm suitable for the task and the tuning of its hyperparameters for optimal results. Current trends in model selection can be gleaned from the winners' methods in Kaggle's tabular data competitions.
# 
# Below is a table of the first-place solutions from the first 13 tabular data playground competitions in the recent Season 3. We see the use of gradient boosting machine (GBM) libraries like LightGBM, XGBoost, and CatBoost remain popular. We also see a small amount of neural network solutions, but note these generally accompany the more popular tree based approaches. Note that some competitions have been excluded due to lack of write-ups or insufficient details on the models used.
# 
# | Competition | Winner's Discussion Post | LightGBM | XGBoost | CatBoost | Neural Network | Random Forest | Ridge | Ensemble |
# |-------------|:----------:|:--------:|:-------:|:--------:|:--------------:|:-------------:|:-----:|:--------:|
# |[Playground Series S3E13](https://www.kaggle.com/competitions/playground-series-s3e13/overview) | [Discussion [3]](https://www.kaggle.com/competitions/playground-series-s3e13/discussion/406433) | ✔️ | | | ✔️ | | | ✔️ |
# |[Playground Series S3E11](https://www.kaggle.com/competitions/playground-series-s3e11/overview) | [Discussion [4]](https://www.kaggle.com/competitions/playground-series-s3e11/discussion/399401) | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |
# |[Playground Series S3E10](https://www.kaggle.com/competitions/playground-series-s3e10/overview) | [Discussion [5]](https://www.kaggle.com/code/seascape/how-to-detect-pulsars-with-gam-1st-place) | ✔️ | ✔️ | | | | | ✔️ |
# |[Playground Series S3E9](https://www.kaggle.com/competitions/playground-series-s3e9/overview) | [Discussion [6]](https://www.kaggle.com/competitions/playground-series-s3e9/discussion/394592) | ✔️ | | | | ✔️ | ✔️ | ✔️ |
# |[Playground Series S3E8](https://www.kaggle.com/competitions/playground-series-s3e8/overview) | [Discussion [7]](https://www.kaggle.com/competitions/playground-series-s3e8/discussion/392926) | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | | ✔️ |
# |[Playground Series S3E7](https://www.kaggle.com/competitions/playground-series-s3e7/overview) | [Discussion [8]](https://www.kaggle.com/competitions/playground-series-s3e7/discussion/390976) | ✔️ | ✔️ | | | | | ✔️ |
# |[Playground Series S3E5](https://www.kaggle.com/competitions/playground-series-s3e5/overview) | [Discussion [9]](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/387882) | | ✔️ | | | | | |
# |[Playground Series S3E3](https://www.kaggle.com/competitions/playground-series-s3e3/overview/) | [Discussion [10]](https://www.kaggle.com/competitions/playground-series-s3e3/discussion/380920) | ✔️ | ✔️ | ✔️ | | | | ✔️ |
# |[Playground Series S3E2](https://www.kaggle.com/competitions/playground-series-s3e2/overview) | [Discussion [11]](https://www.kaggle.com/competitions/playground-series-s3e2/discussion/378795) | | ✔️ | | | | | ✔️ |
# |[Playground Series S3E1](https://www.kaggle.com/competitions/playground-series-s3e1/overview) | [Discussion [12]](https://www.kaggle.com/competitions/playground-series-s3e1/discussion/377137) | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |  | ✔️ |
# 
# Table 1: The algorithms used in the first-place solutions of the first 13 tabular data competitions in Kaggle's Season 3. Each row represents a competition, and each tick represents the use of a particular algorithm in the first-place solution.
# 
# > Learning 2: Machine learning methods, such as GBMs and Neural Networks are used for tabular contests, rather than more simpler methods such as linear regression. 
# 
# > Learning 3: Deep learning approaches are still less popular than Gradient Boosting tree based methods such as LightGBM and XGBoost. 
# 
# #### Ensembling ####
# As evidenced in Table 1 above, 9 out of the 10 winning solutions employed some form of ensembling. This ranged from simple averaging of models, to more complex blending and stacking approaches. 
# 
# Ensemble methods work on the principle of combining the predictions from multiple machine learning algorithms to improve the overall prediction accuracy. This approach can be particularly effective when the individual models make uncorrelated errors, as these can offset each other. For more context, consider this insightful essay: **[The Unreasonable Effectiveness of Ensemble Learning [13]](https://www.kaggle.com/code/yeemeitsang/unreasonably-effective-ensemble-learning)**.
# 
# > Learning 4: Kagglers love an ensemble, and it appears necessary for winning solutions
# 
# #### Cross Validation ####
# 
# Cross validation refers to the process of building a local validation process. This is important because it enables kagglers to understand whether they are reliably improving model accuracy, rather than just overfitting the public leaderboard. The below chart shows the distribution of submission attempts by the top 10 solutions to the first 13 tabular playground competitions in season 3. Note that many top 3 teams are able to obtain high rankings despite only a small number of competition submissions. The reason for this must be strong local cross validation techniques that are more reliable than the leaderboard itself. 
# 
# ![Submission_counts](https://raw.githubusercontent.com/rhyscook92/tabular_data_esssay/main/submission_counts.png)
# *Figure 1: Submission attempts distribution for top solutions of the first 13 tabular playground competitions in season 3. Each dot represents a team, with color indicating the team's final rank (Gold for 1st place, Silver for 2nd, Bronze for 3rd).*
# 
# Pay particular attention to the second last competition. The gold dot all the way to the left is a submission that took first place, despite the author only making three submissions to the public leaderboard. The [source code [14]](https://www.kaggle.com/code/ambrosm/pss3e11-zoo-of-models#Final-evaluation) shows such a small number of submissions was possible due to simple reliance on local 5 fold cross validation. 
# 
# > Learning 5: A local cross validation set up is of vital importance, and in Kaggle competitions it is much more important than making multiple submissions to the public leaderboard. 
# 
# #### Write Ups ####
# A unique and crucial part of the modelling pipeline on Kaggle is the post-competition sharing of results. These write-ups provide a learning opportunity for the community, enabling everyone to understand the top-performing strategies, methodologies, and techniques employed in the competition.
# 
# Reviewing the competitions above, we observe that at least one of the top 10 teams provided a write-up in each competition. In some cases, as many as six of the top 10 teams shared their solutions. 
# 
# ![Writeups by competition](https://raw.githubusercontent.com/rhyscook92/tabular_data_esssay/main/submission_writeups.png)
# *Figure 2: The proportion of top 10 teams sharing write-ups after each of the first 13 tabular playground competitions in season 3.*
# 
# The sharing of results is not merely a post-competition formality but an opportunity for mutual learning and growth within the Kaggle community.
# 
# > Bonus Learning: Kagglers are very generous in sharing what they know!
# 
# <a id='compare'></a>
# # 3. How Do the Insights of Kagglers Compare to Recent Literature?
# We have now laid out the typical pipeline for a Kaggle competition and importantly we extracted 5 key learnings. We now stress test these learnings by comparing them to current literature. 
# 
# - **The Significance of Feature Engineering:** Our exploration of feature engineering demonstrated that creating a wealth of new features can boost performance, as can honing in on the most impactful features, as evidenced by the AMEX competition [1] and Parkinson's competition [2]. Unfortunately there seems to be little recent literature on feature engineering, but we can take a simple example from the 2019 practical guide [Feature Engineering and Selection: A Practical Approach for Predictive Models [15]](http://www.feat.engineering/). In the below we have two charts of identical variables, except in the right chart the variables have undergone a 1/x transformation. Note how after this transformation is applied, the data is more clearly separable into the two classes. This is based on a real world biological dataset, and illustrates how feature engineering can create greater separation between features which will lead to higher performing models. Importantly, the authors of [15] reflect our previous observation that there is rarely a single approach to feature engineering stating in feature engineering and modelling a "campaign of trial and error to achieve the best results" is necessary. 
# 
# ![Combined Feature Engineering](https://raw.githubusercontent.com/rhyscook92/tabular_data_esssay/main/feat_engineering_combined.png)
# 
# *Figure 3: These two images from [15] show how a simple reciprocal transformation increase the seperability of a dataset. This simple transformation increased the area under the ROC curve produced by a logstic regression from 0.794 to 0.848 - a substantial improvement.*
# 
# - **The Superiority of ML Models:** We noted that machine based models such as GBMs and Neural Networks dominated those used by winning solutions in table 1. The paper [M5 accuracy competition: Results, findings, and conclusions [16]](https://www.sciencedirect.com/science/article/pii/S0169207021001874) exploring results to the [M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) on Kaggle had the same finding, noting "advanced ML models have consistently outperformed more classical approaches."
# 
# - **The Popularity of Tree Based Methods versus Deep Learning:**
# Quite interestingly when reviewing Table 1 we also saw a few deep learning (neural network) approaches, but not nearly as often as we saw tree based methods. This suggest Kagglers are not yet convinced on the use of deep learning for tabular data contests. Again we can see this supported in literature, the 2022 paper [Why do tree-based models still outperform deep learning on typical tabular data? [17]](https://arxiv.org/abs/2207.08815) performed a detailed comparison of the accuracy of tree based methods and deep learning solutions on both classification and regression tasks, and noted the out performance of the tree based methods.
# 
# ![Tree versus Deep Learning](https://raw.githubusercontent.com/rhyscook92/tabular_data_esssay/main/updated_tree_vs_deep_learning.PNG)
# 
# *Figure 3: In [17] the authors compare three popular tree based methods (XGBoost, RandomForest, GradientBoosting Tree) to four recent deep learning methods (MLP, a resnet variant, FT transformer and SAINT). Note even after a number of iterations of tuning the tree based methods outperform the deep learning based approaches.*
# 
# - **The Importance of Cross-Validation:** We previously saw strong local cross-validation techniques can be highly effective. Returning to the M5 paper [16] it is pointed out that a large number of participants in the Kaggle M5 forecsasting competition selected final submissions that actually performed worse than their earlier entries, suggesting that a reliable local validation strategy can be more trustworthy than the leaderboard.
# 
# - **The Importance of Ensembling:** In Table 1 we saw 9 out of the 10 winning solutions employed some form of ensembling. This was also observed in [16], which states "The M5 'Accuracy' competition confirmed the findings of the previous four M competitions as well as those of numerous other studies by demonstrating that accuracy can be improved by combining forecasts obtained with different methods". More so, the 2021 paper [Tabular Data: Deep Learning is Not All You Need [18]](https://arxiv.org/abs/2106.03253) showed that a simple ensemble of XGBoost, CatBoost and an SVM outperformed XGBoost as well as many deep learning models. It also showed an ensemble made of both XGBoost and a deep learning model outperformed everything else tested, reaffirming that ensembles work best when diverse models are combined. 
# 
# ![Ensemble Methods](https://raw.githubusercontent.com/rhyscook92/tabular_data_esssay/main/deep_learning_not_all_you_need_result.png)
# 
# *Figure 4: In [18], the authors show the power of ensembling, with an ensemble model having the best relative performance score (a lower score is better) across 11 tabular datasets. We have accented the ensemble methods with an arrow to show they are amoungst the best performing models.*
# 
# <a id='future'></a>
# # 4. Observations and Future Directions for Learning
# Having contrasted the practical findings of high performing Kagglers to the recent developments in literature we make the following observations and suggestions:
# 
# - **Where Kagglers Agree With Literature**: Kagglers and academics seem to agree on two key points. Firstly Gradient Boosting Machines such as XGBoost and LightGBM outperform deep learning methods when performance is averaged out across multiple datasets. Then additionally, Kagglers and modern literature agree that ensembling models provides a strong improvement, especially when the models being combined are diverse. 
# 
# - **The Omission of Feature Engineering in Literature:** The two high performing solutions [1] and [2] clearly placed a key priority on feature engineering. This is true of many Kaggle competitions, and in the authors experience very true in industry. Yet, when researching literature it was quite difficult to find papers exploring feature engineering in any detail. This is likely because it is a practical, messy affair requiring domain knowledge and patience. That said - it is still absolutely necessary. Thus, it would be beneficial to see more research into feature engineering. 
# 
# - **A Lack of Simplicity:** Both Kagglers and recent literature are driven to complexity. As competitive individuals, Kagglers often find themselves driven by the leaderboard, piling on state-of-the-art approaches to eke out those last few decimals of accuracy. Similarly in research it is important to always have fresh results to maintian a cadence of publication. While undeniably exciting, this complexity may not always be necessary in practical applications - the M5 competition paper observes that only 7.5% of teams could outperform a relatively simple benchmark based on exponential smoothing! Further, this complexity often comes at the cost of explainability. It would be great to see Kaggle support the community in finding simple solutions, that are easily explainable. This could potentially be done by competitions that reward both accuracy and simplicity. 
# 
# - **Communication:** As highlighted in the aims of this competition, hundreds of AI papers are published daily, and effective communication of models and results is a crucial skill for data scientists in the industry. Kagglers excel at sharing their work, as demonstrated earlier, but there's always room for improvement. Enhancing communication could involve incentivizing top teams to produce even more write-ups, or perhaps reserving a portion of prize money to reward high-quality write-ups. Both methods could spur further development of communication skills within the Kaggle community.
# 
# <a id='conclusion'></a>
# # 5. Conclusion
# 
# This essay presented a detailed exploration of the typical tabular data Kaggle modelling pipeline, which encompasses crucial aspects such as feature engineering, model selection, ensembling, cross validation, and the sharing of insights through community write-ups. We used the write ups of high performing solutions to obtain a number of key learnings in these areas. 
# 
# We contrasted these learnings to recent literature, and found there is clear agreement between Kagglers and academics across a number of areas: both agree that Gradient Boosting Methods and Ensembles are the path to accuracy. We found there is less literature on feature engineering yet Kagglers know this to be very important in practice. We further noted both parties tend to concentrate on complex methods, which could leave room to explore simple yet explainable methods more. 
# 
# Lastly we noted a strength of Kagglers - our communication with each other, and highlight this strength as an area Kagglers should continue to grow. 
# 
# <a id='references'></a>
# # 6. References
# [1] AMEX 2nd place solution. Available at: https://www.kaggle.com/competitions/amex-default-prediction/discussion/347637
# 
# [2] AMP Parkinson's Disease Progression Prediction 4th place solution. Available at: https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/discussion/411398
# 
# [3] Playground Series S3E13 Winner's Discussion Post. Available at: https://www.kaggle.com/competitions/playground-series-s3e13/discussion/406433
# 
# [4] Playground Series S3E11 Winner's Discussion Post. Available at: https://www.kaggle.com/competitions/playground-series-s3e11/discussion/399401
# 
# [5] Playground Series S3E10 Winner's Discussion Post. Available at: https://www.kaggle.com/code/seascape/how-to-detect-pulsars-with-gam-1st-place
# 
# [6] Playground Series S3E9 Winner's Discussion Post. Available at: https://www.kaggle.com/competitions/playground-series-s3e9/discussion/394592
# 
# [7] Playground Series S3E8 Winner's Discussion Post. Available at: https://www.kaggle.com/competitions/playground-series-s3e8/discussion/392926
# 
# [8] Playground Series S3E7 Winner's Discussion Post. Available at: https://www.kaggle.com/competitions/playground-series-s3e7/discussion/390976
# 
# [9] Playground Series S3E5 Winner's Discussion Post. Available at: https://www.kaggle.com/competitions/playground-series-s3e5/discussion/387882
# 
# [10] Playground Series S3E3 Winner's Discussion Post. Available at: https://www.kaggle.com/competitions/playground-series-s3e3/discussion/380920
# 
# [11] Playground Series S3E2 Winner's Discussion Post. Available at: https://www.kaggle.com/competitions/playground-series-s3e2/discussion/378795
# 
# [12] Playground Series S3E1 Winner's Discussion Post. Available at: https://www.kaggle.com/competitions/playground-series-s3e1/discussion/377137
# 
# [13] The Unreasonable Effectiveness of Ensemble Learning. Available at: https://www.kaggle.com/code/yeemeitsang/unreasonably-effective-ensemble-learning
# 
# [14] Source code for Playground Series S3E11 solution. Available at: https://www.kaggle.com/code/ambrosm/pss3e11-zoo-of-models#Final-evaluation
# 
# [15] Max Kuhn and Kjell Johnson, Feature Engineering and Selection: A Practical Approach for Predictive Models, Avaliable at: http://www.feat.engineering/
# 
# [16] Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2021). The M5 accuracy competition: Results, findings and conclusions. International Journal of Forecasting, 37(3), 1268-1279. Available at: https://www.sciencedirect.com/science/article/pii/S0169207021001874
# 
# [17] Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux (2022). Why do tree-based models still outperform deep learning on tabular data? Avaliable at: https://arxiv.org/abs/2207.08815
# 
# [18] Ravid Shwartz-Ziv, Amitai Armon (2021). Tabular Data: Deep Learning is Not All You Need. Avaliable at https://arxiv.org/abs/2106.03253
# 
# <a id='code'></a>
# # 7. Code and Acknowledgements
# The code used to produce Figures 1, 2 and 4, as well as the figures themselves are avaliable at my github: https://github.com/rhyscook92/tabular_data_esssay . I wanted to make the code avaliable but found including it in the essay would disrupt the flow. Figure 3 is taken directly from images in [15].  
# 
# Chat GPT support: I used GPT4 as a partner throughout writing this essay. I had an idea of the essay I wanted to write and built each section individually then would share my writing with chatGPT for feedback and edits. Sometimes they were valuable, othertimes they were less so! Personally I believe this way of working strikes a good compromise given chatGPT excels in some areas (it has much better spelling and grammar than me, and knows much more about markdown!), but inferior in others (it always trying to use the biggest words and longest sentences!). 

# In[1]:


import pandas as pd
df = pd.read_csv('/kaggle/input/2023-kaggle-ai-report/sample_submission.csv')
df.iloc[0,1] = 'Tabular and/or time series data'
df.iloc[1,1] = 'https://www.kaggle.com/code/rhysie/learnings-from-the-typical-tabular-pipeline'
df.iloc[2,1] = 'https://www.kaggle.com/code/coder98/state-of-content-recommendation-engines-in-2023/comments#2343338'
df.iloc[3,1] = 'https://www.kaggle.com/code/aliwisterr/transfomers-for-time-series-forecasting/comments#2343371'
df.iloc[4,1] = 'https://www.kaggle.com/code/pinkythomas/the-state-of-time-series-in-kaggle-competitions/comments#2343384'
df.to_csv('submission.csv', index=False)

