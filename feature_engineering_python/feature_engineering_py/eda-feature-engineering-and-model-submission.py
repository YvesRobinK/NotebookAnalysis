#!/usr/bin/env python
# coding: utf-8

# ## <p style="font-family:Arial; font-weight:normal; letter-spacing: 2px; color:#A52A2A; font-size:180%; text-align:center;padding: 0px; ">Comprehensive EDA of Vector Borne Disease Dataset</p>
# #### Vector-borne diseases are infections that are transmitted by arthropod vectors, such as mosquitoes, ticks, flies, and fleas. These diseases cause a significant health burden globally, especially in low- and middle-income countries. 
# > In this Kaggle notebook, we will explore a dataset that contains information on 11 vector-borne diseases, including Chikungunya, Dengue, Zika, Yellow Fever, Raft Valley Fever, West Nile Fever, Malaria, Tungiasis, Japanese Encephalitis, Plague, and Lyme Disease. 

# In[1]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer, label_ranking_average_precision_score
from sklearn.model_selection import cross_val_score, KFold,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, LabelEncoder


# #### We're going to use both the train and test data provided by the competition and also the original Vector Borne Disease Dataset.

# In[2]:


train = pd.read_csv("/kaggle/input/playground-series-s3e13/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s3e13/test.csv")
original = pd.read_csv("/kaggle/input/vector-borne-disease-prediction/trainn.csv")


# #### <p style="font-family:Arial; font-weight:normal; letter-spacing: 2px; color:#A52A2A; font-size:180%; text-align:center;padding: 0px; border-bottom: 1px solid #000000;">Kolmogorov-Smirnov test</p>
# #### Before concatenating the train and original data, let's check if the train and original data have the same distributions using the Kolmogorov-Smirnov test. If the p-value is less than 0.05, the distributions are significantly different. Otherwise, there is no significant difference.

# In[3]:


# Drop the id and prognosis columns from the train dataframe
train_features = train.drop(['id', 'prognosis'], axis=1).astype('int64')
original_features = original.drop('prognosis', axis=1)

# Perform the Kolmogorov-Smirnov test for each feature
for column in train_features.columns:
    ks_stat, p_value = stats.ks_2samp(train_features[column], original_features[column])
    if p_value <0.05:
        print(f"Feature: {column}")
        print('Kolmogorov-Smirnov statistic:', ks_stat)
        print('p-value:', p_value)
        print()


# #### Looking at our results, most of the features have p-values greater than 0.05, which suggests that the distributions of these features in the train_features and original datasets are not significantly different. However, a few features have p-values less than 0.05 (e.g., rigor, bitter_tongue, convulsion, prostraction, stiff_neck, etc.). This indicates that there is a significant difference in the distributions of these features between the train_features and original datasets.
# #### Because the number of features with a p-value less than 0.05 is small, and the train data is small too we will proceed with concatenating them but we'll have to carefully monitor the performance of our model to ensure we don't overfit.

# In[4]:


df = pd.concat([train.drop("id", axis = 1), original])
df = df.drop_duplicates()
df = df.reset_index(drop=True)
df = shuffle(df)
df_drop = df.drop("prognosis",axis = 1)


# #### We concatenated the datasets making sure to remove duplicates and shuffle it. Let's make sure the prognosis column has the same names:

# In[5]:


train["prognosis"].value_counts()


# In[6]:


original["prognosis"].value_counts()


# #### As we can see, the diseases made up of two or three names are not identical, the ones in the train data have an underscore and in the original they don't. Let's fix this:

# In[7]:


replace_dict = {
    "Japanese encephalitis": "Japanese_encephalitis",
    "Rift Valley fever": "Rift_Valley_fever",
    "Yellow Fever": "Yellow_Fever",
    "West Nile fever": "West_Nile_fever",
    "Lyme disease": "Lyme_disease"
}

df["prognosis"] = df["prognosis"].replace(replace_dict)


# #### Problem solved!

# ## <p style="font-family:Arial; font-weight:normal; letter-spacing: 2px; color:#A52A2A; font-size:180%; text-align:center;padding: 0px; border-bottom: 1px solid #000000;">Feature Engineering</p>
# #### To better understand the patterns and relationships between symptoms and vector-borne diseases, we have grouped the symptoms based on the affected region of the body. This grouping can help us to identify the common symptoms that occur in specific regions and provide insights into the diseases that affect those regions. The following are some possible clusters of symptoms based on the affected region:
# #### **1. Head and neck symptoms:** sudden fever, headache, mouth bleed, nose bleed, orbital pain, neck pain, red eyes, light sensitivity, facial distortion, microcephaly, stiff neck, speech problem.
# 
# #### **2. Musculoskeletal symptoms:** muscle pain, joint pain, back pain, toe inflammation, finger inflammation.
# 
# #### **3. Gastrointestinal symptoms:** vomiting, diarrhea, stomach pain, digestion trouble, abdominal pain, nausea, gum bleed, ulcers.
# 
# #### **4. Cardiovascular symptoms:** hypotension, slow heart rate, hyperpyrexia.
# 
# #### **5. Respiratory symptoms:** pleural effusion, breathing restriction.
# 
# #### **6. Skin symptoms:** rash, skin lesions, swelling, yellow skin, yellow eyes, itchiness, bullseye rash.
# 
# #### **7. Hematological symptoms:** anemia, cocacola urine.
# 
# #### **8. Neurological symptoms:** coma, diziness, convulsion, irritability, confusion, tremor, paralysis.
# 
# #### **9. General symptoms:** chills, myalgia, fatigue, weakness, weight loss, loss of appetite, lymph node swelling, prostration, bitter tongue, hypoglycemia.
# 
# 

# In[8]:


head_and_neck = ['sudden_fever','headache','mouth_bleed','nose_bleed','orbital_pain','neck_pain','red_eyes','light_sensitivity','facial_distortion','microcephaly','stiff_neck','speech_problem']
musculoskeletal = ['muscle_pain', 'joint_pain', 'back_pain', 'toe_inflammation', 'finger_inflammation']
gastrointestinal = ['vomiting', 'diarrhea', 'stomach_pain', 'digestion_trouble', 'abdominal_pain', 'nausea', 'gum_bleed', 'ulcers']
cardiovascular = ['hypotension', 'slow_heart_rate', 'hyperpyrexia']
respiratory = ['pleural_effusion', 'breathing_restriction']
skin = ['rash', 'skin_lesions', 'swelling', 'yellow_skin', 'yellow_eyes', 'itchiness', 'bullseye_rash']
hematological = ['anemia', 'cocacola_urine']
neurological = ['coma', 'diziness', 'convulsion', 'irritability', 'confusion', 'tremor', 'paralysis']
general = ['chills', 'myalgia', 'fatigue', 'weakness', 'weight_loss', 'loss_of_appetite', 'lymph_swells', 'prostraction', 'bitter_tongue', 'hypoglycemia']


# #### We'll create a new dataset with columns for each disease and symptom category. We'll iterate through each unique disease in the prognosis column of the original dataset (df). For each disease, we'll calculate the **proportion** of symptoms present in each category and append the results as a new row in the 'dataset' DataFrame. We are doing this in order to transform the original dataset into a summarized version, making it easier to analyze and compare diseases based on their symptom profiles.
# #### For example, if the first patient in the df has `sudden_fever` and `headache` but no other symptoms in the `head_and_neck` region, then the value in the column `head_and_neck` will be 2/12. Because the patient has 2 out of 12 symptoms present in that category.

# In[9]:


prognosis = df["prognosis"].unique()
categories = [head_and_neck, musculoskeletal, gastrointestinal, cardiovascular, respiratory, skin, hematological, neurological, general]
categories = (
    ('head_and_neck', head_and_neck),
    ('musculoskeletal', musculoskeletal),
    ('gastrointestinal', gastrointestinal),
    ('cardiovascular', cardiovascular),
    ('respiratory', respiratory),
    ('skin', skin),
    ('hematological', hematological),
    ('neurological', neurological),
    ('general', general)
)

dataset = pd.DataFrame(columns=['disease'] + [category[0] for category in categories])

for disease in prognosis:
    disease_subset = df[df["prognosis"] == disease]
    for i in range(len(disease_subset)):
        row = {'disease': disease}
        for category in categories:
            count = disease_subset.iloc[i][category[1]].sum()/len(category[1])
            row[category[0]] = count
        dataset = dataset.append(row, ignore_index=True)
dataset = shuffle(dataset)


# In[10]:


dataset


# #### We create `test_dataset` by following a similar process. The test dataset is built using the 'test' data, and the same symptom categories are used to calculate the proportions of symptoms present in each category. 

# In[11]:


test_dataset = pd.DataFrame(columns=[category[0] for category in categories])
for i in range(len(test)):
    row = {}
    for category in categories:
        count = test.loc[i][category[1]].sum()/len(category[1])
        row[category[0]] = count
    test_dataset = test_dataset.append(row, ignore_index=True)


# In[12]:


test_dataset


# ## <p style="font-family:Arial; font-weight:normal; letter-spacing: 2px; color:#A52A2A; font-size:180%; text-align:center;padding: 0px; border-bottom: 1px solid #000000;">Logistic regression</p>
# #### Let's evaluate a Logistic Regression model using Mean Average Precision at 3 (MAP@3). This evaluation metric is helpful because we want to assess how well our model performs in recommending the top 3 most relevant diseases.
# #### Firstly, let's define the MAP@3 scoring function:

# In[13]:


def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


# ### And now compute the cross-validation scores

# In[14]:


X = dataset.drop(columns=["disease"], axis=1)
y = dataset["disease"]
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#label encoder
le = LabelEncoder()
y = le.fit_transform(y)
map3_scores = []
for train_index, test_index in cv.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
 
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)
 
    # Get the sorted indices of predictions and take the top 3
    sorted_prediction_ids = np.argsort(-predictions, axis=1)
    top_3_prediction_ids = sorted_prediction_ids[:,:3]
 
    map3score = mapk(y_test.reshape(-1, 1), top_3_prediction_ids, k=3)
    map3_scores.append(map3score)
 
map3_scores = np.array(map3_scores)
print("Cross-validation MAP@3 scores:", map3_scores)
print("MAP@K: %0.2f (+/- %0.2f)" % (map3_scores.mean(), map3_scores.std() * 2))


# #### Not too bad. Our model's performance was relatively consistent, with a moderate level of accuracy in returning relevant results. Is time to make some predictions on the test data.

# In[15]:


model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X, y)

# Step 6: Make predictions and take top 3 predictions
probabilities = model.predict_proba(test_dataset)
top_3_predictions = []
for i in range(len(test_dataset)):
    sorted_probabilities = probabilities[i].argsort()[::-1][:3]  # get indices of top 3 classes in descending order
    top_3_classes = [model.classes_[j] for j in sorted_probabilities]  # get class labels for top 3 classes
    decoded_top_3_classes = le.inverse_transform(top_3_classes)  # decode class labels back into original form
    top_3_predictions.append(decoded_top_3_classes.tolist())  # convert array to list


# ## <p style="font-family:Arial; font-weight:normal; letter-spacing: 2px; color:#A52A2A; font-size:180%; text-align:center;padding: 0px; border-bottom: 1px solid #000000;">Submit prediction</p>

# In[16]:


submission = pd.DataFrame({
    "id": test["id"],  # use the first n ids
    "prognosis": top_3_predictions  # convert to a list for DataFrame creation
})


# In[17]:


submission


# #### Submitting the predictions as they are now would result in a score of 0 on the leaderboard. This is due to the prognosis column containing lists of diseases separated by commas. We need to convert these lists into strings where diseases are separated by spaces instead.

# In[18]:


def list_to_string(disease_list):
    return ' '.join(disease_list)
# Apply the function to the 'prognosis' column
submission['prognosis'] = submission['prognosis'].apply(list_to_string)
print(submission)


# In[19]:


# Save the submission DataFrame to a CSV file
submission.to_csv("submission.csv", index=False)

