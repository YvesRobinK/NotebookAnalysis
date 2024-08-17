#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info" style="font-size:20px; font-family:verdana;">
#     <b>Pytorch 🆚 Disaster | EDA & neural network ⚡️</b>
#     <br>Hello dear friends 👋<br>
#     This is my first work so do not judge strictly 🫣
#     <br>I decided to edit it to make it easier for you to extract new knowledge ⚡️<br>
#     I also want to make sure that here I solve the problem exclusively with the help of a neural network
# </div>

# # Have you set your goal? Then let's get started ⚡️
# ![](https://cdn.dribbble.com/users/995553/screenshots/2589741/4.gif)

# # Import libraries

# In[1]:


import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import missingno as msno

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings


# # Configuring matplotlib & seaborn 

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore') 

little = 11
medium = 16
large = 21
huge = 25

params = {"axes.titlesize": large,
          "figure.titlesize": huge,
          "legend.fontsize": medium,
          "figure.figsize": (16, 10),
          "axes.labelsize": medium,
          "axes.titlesize": medium,
          "xtick.labelsize": medium,
          "ytick.labelsize": medium,
          "axes.grid": True }

plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

pd.set_option("display.float_format", lambda x: "%0.4f" % x)
np.set_printoptions(suppress=True)



print(mpl.__version__)  
print(sns.__version__) 
print(pd.__version__)  
print(np.__version__) 


# ## Loading data

# In[3]:


train_df = pd.read_csv('/kaggle/input/titanic/train.csv') 
test_df = pd.read_csv('/kaggle/input/titanic/test.csv') 
submission_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
result_df = test_df[["PassengerId"]]


# # EDA

# In[4]:


print(train_df.shape)
print(test_df.shape)


# In[5]:


msno.matrix(train_df)


# In[6]:


msno.matrix(test_df)


# In[7]:


train_df.head()


# In[8]:


train_df.info()


# In[9]:


test_df[test_df.Fare.isna()]
test_df.Fare.fillna(test_df.Fare[(test_df.Age) > 55 & (test_df.Age < 65)].mean(), inplace=True)


# In[10]:


test_df.info()


# In[11]:


submission_df.head()


# In[12]:


sns.pairplot(data=train_df)


# In[13]:


sns.barplot(x="Sex", y= "Survived", data=train_df)


# In[14]:


sns.barplot(y="Age", x="Survived", data=train_df)


# In[15]:


sns.barplot(x="Pclass", y= "Survived", data=train_df)


# In[16]:


sns.barplot(
    x="Embarked",
    y="Survived",
    palette='hls',
    data=train_df)


# In[17]:


sns.barplot(x="Pclass", y="Survived", hue="Embarked", data=train_df)


# In[18]:


sns.barplot(y="SibSp", x="Survived", data=train_df)


# In[19]:


sns.barplot(y="Parch", x="Sex", hue="Survived" ,data=train_df)


# In[20]:


sns.barplot(y="Parch", x="Survived", data=train_df)


# In[21]:


sns.heatmap(train_df.corr(), cmap="BrBG", vmin=-1, vmax=1, annot=True)


# In[22]:


heatmap = sns.heatmap(train_df.corr()[['Survived']].sort_values(by='Survived', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Survived', fontdict={'fontsize':18}, pad=16);


# # Data integrity check

# In[23]:


column_targets = ["Survived"]
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
all_columns = column_targets + features


# In[24]:


def nan_checking():
    null_columns = {}
    for feature in all_columns:
        nulls = train_df[feature].isnull().sum()
        if nulls > 0:
            null_columns[feature] = nulls
    print(null_columns)


# In[25]:


train_df.drop(labels=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True, axis=1)
test_df.drop(labels=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True, axis=1)
train_df.head()


# In[26]:


targets_df = train_df[column_targets]
targets_df


# In[27]:


nan_checking()


# In[28]:


def one_hot_encoding(df):
    df['Sex'].replace(['male','female'],[0,1],inplace=True)
    df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
 

def fillna(df):
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
  


# In[29]:


fillna(train_df)
one_hot_encoding(train_df)
fillna(test_df)
one_hot_encoding(test_df)
test_df


# In[30]:


train_df


# In[31]:


nan_checking()
train_df.drop(labels=["Survived"], inplace=True, axis=1)


# # Feature engineering 🔩

# # What is feature engineering?
# ![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*SJmN6FeHtmyd87JqADWmhw.png)

# <div style="background-color:#d4f1f4; padding: 20px;">
# <p style="font-size:20px; font-family:verdana; line-height: 1.7em">Exploratory Data Analysis (EDA) is the very first step before you can perform any changes to the dataset or develop a statistical model to answer business problems. In other words, the process of EDA contains summarizing, visualizing and getting deeply acquainted with the important traits of a data set. You should have noticed that the phrase “Exploratory” contained in the name itself should have the reason, right?</p>
#     <b><br>You can read more on the link ⬇️⬇️⬇️<br></b>
#     <a href=https://medium.com/@ndleah/eda-data-preprocessing-feature-engineering-we-are-different-d2a5fa09f527>EDA, Data Preprocessing, Feature Engineering: We are different!</a>
#     <br><a href=https://towardsdatascience.com/what-is-feature-engineering-importance-tools-and-techniques-for-machine-learning-2080b0269f10>What is feature engineering?</a><br>
# </div>

# In[32]:


train_df # All (7 features)
train_df_2 = train_df.copy()
train_df_2["Family"] = train_df_2.loc[:, "SibSp":"Parch"].sum(axis=1) # Sum SibSp and Parch (6 features)
train_df_2.drop(labels=["SibSp", "Parch"], inplace=True, axis=1)
train_df_3 = train_df.drop(columns="SibSp", inplace=False) # Without SibSp (6 features) 


# In[33]:


test_df # All (7 features)
test_df_2 = test_df.copy()
test_df_2["Family"] = test_df_2.loc[:, "SibSp":"Parch"].sum(axis=1) # Sum SibSp and Parch (6 features)
test_df_2.drop(labels=["SibSp", "Parch"], inplace=True, axis=1)
test_df_3 = test_df.drop(columns="SibSp", inplace=False) # Without SibSp (6 features)


# In[34]:


train_df.head()


# In[35]:


train_df_2.head()


# In[36]:


train_df_3.head()


# In[37]:


test_df.head()


# In[38]:


test_df_2.head()


# In[39]:


test_df_3.head()


# In[40]:


datadict = {
    "all": [train_df, test_df],
    "family": [train_df_2, test_df_2],
    "sibsp": [train_df_3, test_df_3]
}
datadict


# # Control data verification

# In[41]:


msno.matrix(train_df)


# In[42]:


msno.matrix(test_df)


# # Data Status Control Panel

# In[43]:


#   0        1        2
data_state = [ "all", "family", "sibsp"]
STATE = 1
train_df, test_df = datadict.get(data_state[STATE])


# In[44]:


train_df = train_df.to_numpy()
targets_df = targets_df.to_numpy()
test_df = test_df.to_numpy()


# In[45]:


# death life
double_targets = [ [j-1, j] 
   if j == 1
   else [j+1, j]
   for j in 
   [targets_df[i,0] for i in range(targets_df.shape[0])]]

double_targets = np.array(double_targets)


# In[46]:


TEST_SIZE = 0.33
RANDOM_STATE = 356 # 60
NORMALIZATION = True


# In[47]:


if NORMALIZATION:
    normal = MinMaxScaler().fit(train_df)
    train_df = normal.transform(train_df)
    test_df = normal.transform(test_df)

else:
    scaler = StandardScaler().fit(train_df)
    train_df = scaler.transform(train_df)
    test_df = scaler.transform(test_df)


# In[48]:


x_train, x_test, y_train, y_test = train_test_split(train_df, double_targets, shuffle=True, random_state=RANDOM_STATE, test_size=TEST_SIZE)


# # Models

# # 1. NeuralNetwork with PyTorch

# In[49]:


get_ipython().system('nvidia-smi')


# **CUDA**

# In[50]:


CUDA_NOT = True
device = torch.device("cuda")

if not torch.cuda.is_available() or cuda_not:
    device = torch.device("cpu")


# **Dataset**

# In[51]:


class TitanicDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = torch.from_numpy(x_train.astype(np.float32)).to(device=device)
        self.y_train = torch.from_numpy(y_train.astype(np.float32)).to(device=device)
        self.n_samples = x_train.shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]


# **Settings of dataset**

# In[52]:


dataset = TitanicDataset(x_train, y_train)

EPOCH = 22 # 37
BATCH_SIZE = 60
ITERS = len(dataset) // BATCH_SIZE 

dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=BATCH_SIZE)
x_test_1 = torch.from_numpy(x_test.astype(np.float32)).to(device=device)
y_test_1 = torch.from_numpy(y_test.astype(np.float32)).to(device=device)


# **Seetings of NeuralNetwork**

# In[53]:


INPUT = x_train.shape[1]
HIDDEN_1 = 16
HIDDEN_2 = 16
HIDDEN_3 = 16
OUTPUT = 2
ALPHA = 0.01


# ![](https://machinelearningknowledge.ai/wp-content/uploads/2019/10/Feed-Forward-Neural-Network.gif)

# In[54]:


class TitanicNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_1, hidden_2, hidden_3, output_size, device):
        super(TitanicNeuralNetwork, self).__init__()
        self.input_hidden = nn.Linear(in_features=input_size, out_features=hidden_1, device=device)
        self.hidden_hidden = nn.Linear(in_features=hidden_1, out_features=hidden_2, device=device)
        self.hidden_hidden_2 = nn.Linear(in_features=hidden_2, out_features=hidden_3, device=device)
        self.hidden_output = nn.Linear(in_features=hidden_3, out_features=output_size, device=device)
    
    def forward(self, x_data):
        x_data = F.leaky_relu(self.input_hidden(x_data))
        x_data = F.leaky_relu(self.hidden_hidden(x_data))
        x_data = F.leaky_relu(self.hidden_hidden_2(x_data))
        x_data = F.dropout(x_data, p = 0.28)
        y_predict = self.hidden_output(x_data)
        return y_predict


# In[55]:


def calc_accuracy(x_test_1, y_test_1):
    with torch.no_grad():
        x_predict = model(x_test_1).argmax(axis=1)
    y_test_max = y_test_1.argmax(axis=1)
    accuracy = torch.eq(x_predict, y_test_max).to(torch.int8).sum() / len(y_test_max)
    return accuracy


# **Model & Criterion & Optimizer**

# In[56]:


model = TitanicNeuralNetwork(INPUT, HIDDEN_1, HIDDEN_2, HIDDEN_3, OUTPUT, device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=ALPHA)


# # Trainig

# In[57]:


accuracy_list = []


for epoch in range(EPOCH):
    for iters, (x, y) in enumerate(dataloader):
        
        y_predict = model(x)
        loss = criterion(y_predict, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        accuracy = calc_accuracy(x_test_1, y_test_1)
        accuracy_list.append(accuracy)
        
    if epoch % 2 == 0:
        print(f">> Epoch ~ [ {epoch+1} ] || >> Loss ~ [ {loss.item():.4f} ] || >> Accuracy ~ [ {accuracy:.2f} ]")


# In[58]:


plt.plot([i.to(device=torch.device("cpu")) for i in accuracy_list])


# # Submit our result

# In[59]:


data = torch.from_numpy(test_df.astype(np.float32)).to(device=device)
target_column = model(data).detach().argmax(axis=1).to(dtype=torch.int64)
result_df["Survived"] = target_column
result_df.to_csv("result_sub_2.csv", index=False)


# In[60]:


result_df


# <div class="alert alert-block alert-info" style="font-size:20px; font-family:verdana;">
#     <b>At this stage, we can say goodbye to you 👋</b>
#     <br>I do not consider myself a professional, but I really hope that you were able to learn something new 🙃<br>
#     I will be glad if you rate my other works
# </div>

# ![](https://media1.giphy.com/media/3ohs7JG6cq7EWesFcQ/giphy.gif?cid=6c09b95210a814c52be2ed364f4973dfe1357d43211d2e85&rid=giphy.gif&ct=g)
