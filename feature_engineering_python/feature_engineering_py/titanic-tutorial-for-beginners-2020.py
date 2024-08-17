#!/usr/bin/env python
# coding: utf-8

# # Titanic Challenge

# ![](https://i.imgur.com/rRFchA8.png)

# 이 튜토리얼은 캐글에 처음 도전하는 분들을 위한 것입니다. 커널부문 세계1위인 안드레이 룩야넨코의 노트북에 한글로설명을 추가로 붙인 것입니다. 
# 이 노트의 결과는 작성 현재 Top 4% 결과를 보입니다. (랜덤이 되는 부분이 있어서 결과가 약간씩 다르게 나오긴 합니다만 여러번 하시면 4% 선까지 올라갑니다.)
# Kaggle에 처음으로 로그인하는 것은 쉬운 결정이 아닙니다. 내가 무슨 머신러닝을? 이런 생각을 하실 수 도 있겠습니다. 
# 캐글은 대회로서도 의미가 있겠으나 배움의 터입니다. 이런 데이터를 초보자가 어디서 구하며 어떻게 다른 사람들이 일한 것을 통해 이리 쉽게 배우겠습니까? 
# 이 튜토리얼이 끝날 때 즈음에는 Kaggle의 온라인 코딩 환경을 사용하는 방법에 대한 이해를 얻게되며 그 와중에 머신러닝 학습 모델을 학습하게됩니다.
# 
# 우측 상단에 Copy & Edit을 눌러서 복사.

# <a id = "table_of_contents"></a>
# ## Table of contents
# 
# [Part 1: 데이터 준비 및 모듈 임포트](#part1)			
# [Part 2: 파일 병합](#part2)			
# [Part 3: 파일 탐색](#part3)			
# [Part 4: 데이터 탐구 (Exploratory Data Analysis)](#part4)			
# [Part 5: Feature Engineering](#part5)	
# 
# [Part 6: 마지막 항목 결정](#part6)			
# [Part 7: 머신러닝 모델 만들기](#part7)			
# [Part 8: 중요도에 따라 모델 재 설정 ](#part8)			
# [Part 9: 하이퍼 파라미터 튜닝](#part9)			
# [Part 10: 모델 재 트레이닝](#part10)			
# 
# [Part 11: 마지막 보팅](#part11)	
# [Part 12: 마지막 모델 예측](#part12)   
# [Part 13: 제출](#part13)

# <a id = "part1"></a>
# ## Part 1: 데이터 준비 및 모듈 임포트
# [Go to the Table of Contents](#table_of_contents)

# In[ ]:


# 기본 데이터 정리 및 처리
import numpy as np
import pandas as pd

# 시각화
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('seaborn-whitegrid')
import missingno

# 전처리 및 머신 러닝 알고리즘
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

# 모델 튜닝 및 평가
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn import model_selection

# 경고 제거 (판다스가 에러 메세지를 자주 만들어 내기 때문에 이를 일단 무시하도록 설정합니다.)
import sys
import warnings

import warnings
warnings.filterwarnings('ignore')


# ### 중요 링크
# 
# [구글 콜랩](https://colab.research.google.com/) 
# 
# [깃헙브](https://github.com)
# 
# ### 초보자이실 경우는 아래 링크 중 본인에게 맞는 것을 미리 공부해 보는 것도 도움이 됩니다.
# 
# [edx의 파이썬 기초수업](https://learn.edx.org/topic-python/?g_acctid=926-195-8061&g_campaign=gs-nonbrand-topic-python&g_campaignid=1535528542&g_adgroupid=58645176415&g_adid=405031374941&g_keyword=learn%20python%20online%20tutorial&g_keywordid=kwd-357260176984&g_network=g?utm_source=adwords&utm_campaign=1535528542&utm_medium=58645176415&utm_term=learn%20python%20online%20tutorial&gclid=CjwKCAjwwYP2BRBGEiwAkoBpAu6FTNidm90CEM1AJQGDn_mvW_bDMJSkWuWk6DnUJCrCXnG0-vW_aBoC8GwQAvD_BwE)
# 
# [파이썬 org에 나오는 파이썬 기초수업](https://docs.python.org/3/tutorial/)
# 
# [w3schools 파이썬 기초수업](https://www.w3schools.com/python/)
# 
# [Joshua Choi님의 파이썬 기초 문법 연습](https://joshua-mobile-choi-1756.trinket.io/python-3-4#/tasks/task-1-print-statement)
# 
# [Joshua Choi님의 타이타닉 판다스 101 사용 예](https://www.kaggle.com/joshuajhchoi/101-pandas-tips-for-beginners-titanic-en-kr)
# 
# [Joshua Choi님의 타이타닉 시각화 101 사용 예](https://www.kaggle.com/joshuajhchoi/101-data-visualization-tips-for-titanic-beginners)
# 
# [Seaborn연습](https://seaborn.pydata.org/tutorial/aesthetics.html)

# ### CSV to DF
# 
# * csv를 임포트하여 데이터셋이 판다스 데이터프레임이 되도록 합니다.

# In[ ]:


# 이 것이 처음하는 사람에게 예상보다 어려울 수 있는데 복사한 것에서 데이터가 전달이 잘 안 되었다면 "+Add Data" 누르시고 'Competition Data'에서 "Titanic Data" 불러온 후 파일을 찍어서 경로 주소 확인해야 함 
test = pd.read_csv('../input/titanic/test.csv')
train = pd.read_csv('../input/titanic/train.csv')

# 이제 csv file들 (test & train)은 데이터 프레임이 되었습니다.


# 구글 콜랩에서 사용하실 때는 컴퓨터에 파일을 다운로드 한 후 아래 코드를 입력하면 불러올 수 있게 됩니다.
# 
#     from google.colab import files
#     uploaded = files.upload()
# 
# 그런 다음 아래 코드를 통해서 csv를 데이터프레임으로 바꿀 수 있게 됩니다.
# 
#     import io
#     test = pd.read_csv(io.BytesIO(uploaded['test.csv']))
#     train = pd.read_csv(io.BytesIO(uploaded['train.csv']))

# ### 데이터프레임을 보는 다양한 방법

# `head()`첫 5행을 볼 수 있습니다.

# In[ ]:


train.head(n=3)


# `tail()` 마지막 5행을 볼 수 있습니다.

# In[ ]:


train.tail()


# `describe()` 각 열의 통계적인 면을 보여 줍니다. 
# 
# 기본은 연속된 값을 가진 열만 보여주나 `include='all'로 세팅하면 모두 볼 수 있습니다.

# In[ ]:


train.describe(include='all')


# `dtypes` 모든 열의 데이터 종류를 보여 줍니다.

# In[ ]:


train.dtypes


# `info()` 는 `dtypes` 의 좀  더 발전된 개념으로 데이터 타입뿐만 아니라 빈칸이 아닌 갯수까지 보여 줍니다.
# 
# [describe 학습](https://www.w3resource.com/pandas/dataframe/dataframe-describe.php)
# 

# In[ ]:


train.info()


# `columns`은 데이터 프레임의 열의 제목들을 보여 줍니다.

# In[ ]:


train.columns


# 단순히 연습으로, 한 번 원하는 컬럼만 인덱싱해 보겠습니다.
# 
# 인덱싱에 대해 더 공부하고 싶으시다면 아래 링크를 클릭하여 공부해 보세요.
# 
# [w3schools 넘파이 어레이 인덱싱](https://www.w3schools.com/python/numpy_array_indexing.asp)
# 
# [Joshua CHoi님의 파이썬 인덱싱](https://joshua-mobile-choi-1756.trinket.io/python-3-4#/tasks/task-4-string-indexing)
# 
# [Joshua Choi님의 파이썬 리스트 인덱싱](https://joshua-mobile-choi-1756.trinket.io/python-3-4#/tasks/task-15-list-methods)

# In[ ]:


train.columns[3], train.columns[3:5]


# 한 행 뿐만 아니라 여러행을 인덱싱 할 수도 있습니다.

# In[ ]:


train[5:20]


# `shape` 은 행의 갯수와 열의 갯수를 보여 줍니다

# In[ ]:


train.shape


# <a id = "part2"></a>
# ## Part 2:  파일 병합
# [Go to the Table of Contents](#table_of_contents)
# 
# * ntrain과 ntest의 shape을 확보해놓습니다. (병합 한 것을 나중에 다시 갈라 놓기 위한 준비)
# * y_train은 알려진 결과 값이니 따로 모셔 놓고
# * 테스트의 승객 아이디는 나중에 최종 결과에 넣을 것이기 때문에 따로 떼어 놓습니다.
# * train과 test를 병합하여 data 란 파일을 만듭니다. 문자로 된 것을 숫자로 바꾼다든가. 숫자를 인터발 별로 그룹화 한다든가 할 때 한꺼번에 하기 위해 합해 놓습니다.

# In[ ]:


# 병합 준비
ntrain = train.shape[0]
ntest = test.shape[0]

# 아래는 따로 잘 모셔 둡니다.
y_train = train['Survived'].values
passId = test['PassengerId']

# 병함 파일 만들기
data = pd.concat((train, test))

# 데이터 행과 열의 크기는
print("data size is: {}".format(data.shape))


# #### 파이썬의 프린트문과 포맷에 대해 보시려면 아래 링크를 클릭하세요.
# 
# [프린트문](https://joshua-mobile-choi-1756.trinket.io/python-3-4#/tasks/task-1-print-statement)
# 
# [포맷팅](https://joshua-mobile-choi-1756.trinket.io/python-3-4#/tasks/task-11-string-format)

# In[ ]:


train['Survived'].value_counts()


# 트레인 데이터에 있는 인원 중 342명이 살아남고 549명이 사망했다는 것을 볼 수 있습니다.

# <a id = "part3"></a>
# ## Part 3: 파일 탐색
# [Go to the Table of Contents](#table_of_contents)

# In[ ]:


missingno.matrix(data, figsize = (15,8))


# 위 도표는 빈 값을 직관적으로 볼 수 있게 합니다.
# 아래 쪽 `Survived`가 비어 있는데 이 것은 테스트에 우리가 답으로 예측해야 하는 부분이라서 그렇 습니다.

# In[ ]:


data.isnull().sum() #비어 있는 값들을 체크해 본다.


# In[ ]:


data.Age.isnull().any()


# 열 이름을 보겠습니다.

# [Kaggle 페이지에서 타이타닉 열 이름 보기 (해당 페이지 중간에 Dictionary 보시면 됨)](https://www.kaggle.com/c/titanic/data)

# In[ ]:


data.columns


# ### Features 항목
# 
# #### 항목의 종류 
# 
# * 범주형 항목 (Categorical Features)
# 
# 범주형 항목은 법주형 변수로 된 항목으로 범주형 변수는 둘 이상의 결과 요소가 있는 변수이며 해당 기능의 각 값을 범주별로 분류 할 수 있습니다. 
# 
# 예를 들어 성별은 두 가지 범주 (남성과 여성)의 범주 형 변수입니다. 
# 
# 이산형 변수(discrete variable) = 범주형 변수 (categorical variable) 의 하나로 명목 변수 norminal variable 라고도합니다.
# 
# 데이터 셋에서 명목 항목 : Sex, Embark 이며 우리는 Name, Ticket 등을 이로 변환해야 할 것 같습니다. 
# 
# 
# * Ordinal Variable :
# 
# 순위 변수는 범주 형의 하나지만 그 차이점은 값 사이의 상대 순서(=서열) 또는 정렬이 가능하다는 것입니다.
# 
# 데이터 셋에서 순위 항목 : PClass 이며 우리는 Cabin을 이 범주로 변환해서 사용해야 할 것 같습니다.
# 
# 
# * 연속형 항목 (Continuous Features):
# 
# 서로 연속된 값을 가진 변수를 가진 항목이며 여기에서 우리는 연령을 대표적인 것으로 볼 수 있습니다.
# 
# Age, SipSp, Parch, Fare는 interval variable로 만들어 이에 적용해야 할 것 같습니다.
# 
#  
# 
# * 아래의 항목에서 열의 이름을 볼 수 있습니다.
# 
#           Variable          정의                Key
# 
#           survival          생존 여부            0 = No, 1 = Yes
# 
#           pclass            선실 등급            1 = 1st, 2 = 2nd, 3 = 3rd
# 
#           sex               성별    
# 
#           Age               나이  
# 
#           sibsp             형재 자매의 수/ 배우자 등이 승선한 경우 수    
# 
#           parch             부모나 자식과 같이 탄 경우 수   
# 
#           ticket            표 번호    
# 
#           fare              요금
# 
#           cabin             선실 번호   
# 
#           embarked          승선한 항구         C = Cherbourg, Q = Queenstown, S = Southampton

# <a id = "part4"></a>
# ## Part 4: 데이터 탐구 (Exploratory Data Analysis)
# [Go to the Table of Contents](#table_of_contents)
# 
# * train파일 순서대로 데이터 파일의 열들을 봅니다.

# In[ ]:


train.head()


# * 파일 각 열의 상관 관계를 보겠습니다.
# 
# Co-relation 매트릭스는 seaborn에서 변수 간 상관 계수를 보여주는 표입니다. 표의 각 셀은 두 변수 간의 상관 관계를 보여줍니다. 상관 매트릭스는 고급 분석에 대한 입력 및 고급 분석에 대한 진단으로 데이터를 요약하는 데 사용됩니다. 
# 
# 참고: https://seaborn.pydata.org/examples/many_pairwise_correlations.html
# 
# 아래 마스크 셋업은 0로 행렬을 상관 행렬과 같은 모양으로 만든 후 여기에 불리안 값을 넣고 이를 다시 True만 만듭니다.
# 
# triu 는 우측 상단 삼각행렬을 의미
# 
# annot= True는 각 셀에 숫자를 표시하라는 것이고, False는 하지 말라는 것이구요
# 
# 이어서 이를 heatmap으로 런칭합니다.

# In[ ]:


# Co-relation 매트릭스
corr = data.corr()
# 마스크 셋업
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# 그래프 셋업
plt.figure(figsize=(14, 8))
# 그래프 타이틀
plt.title('Overall Correlation of Titanic Features', fontsize=18)
#  Co-relation 매트릭스 런칭
sns.heatmap(corr, mask=mask, annot=False,cmap='RdYlGn', linewidths=0.2, annot_kws={'size':20})
plt.show()


# * [히트맵](https://seaborn.pydata.org/generated/seaborn.heatmap.html) 
# 
# * [판다스 코릴레이션 매트릭스](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html)  
# 
# * [씨본 콜릴레이션 히트맵](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
# 

# * "Surived" 분석
# 
# 한 열씩 검토해 보겠습니다.
# 
# Survived - Key: (0 - Not Survived, 1- Survived)
# 
# Survived는 숫자로 값을 주지만 Categorical Variable인 셈입니다.
# 
# 죽던지 살던지 둘 중 하나의 값을 줍니다.
# 
# countplot을 그려 봅니다.
# 
# 사이즈는 가로 10인치 세로 2인치
# 
# 생존 여부 0과 1의 숫자를 세어 본 후 그림을 그리도록 명령을 하는 것입니다.
# 
# pyplot(plt)의 figure라는 메소드를 써서 그림판의 크기를 정하고, seaborn의 카운트플롯을 그리라는 것입니다.

# In[ ]:


fig = plt.figure(figsize=(10,2))
sns.countplot(y='Survived', data=train)
print(train.Survived.value_counts())


# * 불행히도 사망자가 훨씬 많아 보입니다.
# * 전체 사망자 비율을 좀 보겠습니다.
# 
# * 파이그래프랑 카운트 플롯을 서브플롯으로 그립니다.
# * 행은 하나 열은 2개의 서브 플롯입니다. 사이즈는 가로 15인치 세로 6인치
# * 'Survived'의 값을 카운트해서 파이플롯을 만듭니다.
# * explode는 폭발하는 것이니까 1이면 튀어 나가는 것인데 0을 주면 분리만 되고 돌출은 되지 않습니다. 이어서 0, 1인 것은 첫 번째 것은 아니고 두번 째 것은 분리된다는 의미로 생각하시면 됩니다.
# * autopercent는 1.1이 표현하는 부분은 소수점 한 자리까지 보여 주라는 의미입니다. 뒤에 점 이하가 4면 둘 다 소수점 4자리수 까지 보여 줍니다.
# * ax[0]은 첫번째 칸입니다.
# * set_title 메소드는 서브 플롯의 제목을 보여 줍니다.

# In[ ]:


f,ax=plt.subplots(1, 2, figsize=(15, 6))
train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=train, ax=ax[1])
ax[1].set_title('Survived')
plt.show()


# * 위의 것을 아래와 같이 함수로 만들겠습니다. (물론 자주 쓰이지는 않겠지만 연습이니)

# In[ ]:


def piecount(col):
    f, ax = plt.subplots(1, 2, figsize=(15, 6))
    train[col].value_counts().plot.pie(explode=[0.1 for i in range(train[col].nunique())], autopct='%1.1f%%', ax=ax[0], shadow=True)
    ax[0].set_title(col)
    ax[0].set_ylabel('')
    sns.countplot(col, data=train, ax=ax[1])
    ax[1].set_title(col)
    plt.show()

piecount('Survived')


# [함수 만들기 공부](https://trinket.io/joshua-mobile-choi-1756/courses/python-3-4#/tasks/task-20-making-functions)
# 
# [파라미터와 아규먼트 공부](https://trinket.io/joshua-mobile-choi-1756/courses/python-3-4#/tasks/task-21-parameters-and-arguments)
# 
# * "Pclass" 분석
# 
# * Pclass는 값이 숫자이나 서열이 정해진 Ordinal Feature이다.
# * Key:1 = 1st, 2= 2nd, 3 = 3rd
# * 각 클래스 당 생존자를 보겠습니다.

# In[ ]:


train.groupby(['Pclass','Survived'])['Survived'].count()


# In[ ]:


pd.crosstab(train.Pclass, train.Survived, margins=True).style.background_gradient(cmap='summer_r')


# * 1등급 객실의 사람들은 생존자가 더 많고, 2등급은 생존자에 비해 사망자가 조금 더 많으나, 3등급은 사망자가 3배 이상 많다는 것을 알 수 있습니다.

# In[ ]:


f, ax = plt.subplots(1, 2, figsize=(12, 6))
train[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived per Pcalss')
sns.countplot('Pclass', hue='Survived', data=train, ax=ax[1])
ax[1].set_title('Pcalss Survived vs Not Survived')
plt.show()


# * 위에 만든 함수를 한 번 써 먹어 볼까요?

# In[ ]:


piecount("Pclass")


# * %는 3등칸이 반이 넘으나 위의 그래프에서 생존자는 1등석이 가장 많다는 것을 알 수 있습니다.
# * 각 클래스 당 생존률을 볼까요?

# ### "Name" 분석
# * 이름은 거의 모두 다를 가능성이 큽니다. Family Name, First Name, Middle Name and even Dr. Capt, master and so on 모두 감안하면...
# * 분류를 한 번 해 봅니다.
# * 리스트를 한 번 주 욱 보겠습니다.

# In[ ]:


data.Name.value_counts()


# * 이름은 언뜻 보아서 감이 안 옵니다. 중간에 있는 Mr. 같은 호칭을 볼까요.
# * ['Initial']이란 열을 새로 만들어서 여기에 Name에서 추출한 Regular Expression을 넣습니다.
# * 아래에서 str.extract('([A-Za-z]+).')부분은 str에서 대문자 A~Z, 소문자 a~z 중에 . 명령을 통해 .으로 끝나는 부분을 추출해 내는 것입니다.
# * ('^([A-Za-z]+)')으로 하면 처음에 나오는 문자 덩어리가 될 것이고 +를 빼면 첫 스펠링 한캐릭터만 추출합니다.

# [Regex 공부](https://en.wikipedia.org/wiki/Regular_expression)
# 
# * 안전을 위해 카피를 하나 만들어서 새로운 항목을 만들어 봅니다.

# In[ ]:


temp = data.copy()
temp['Initial'] = 0
temp['Initial'] = data.Name.str.extract('([A-Za-z]+)\.')


# Miss나 Mr등은 많으나 익숙하지 않은 몇 개가 보입니다.

# In[ ]:


temp['Initial'].value_counts()


# 이를 성별로 봅니다.

# In[ ]:


pd.crosstab(temp.Initial, temp.Sex).T.style.background_gradient(cmap='summer_r')


# * 생존률로 봅니다.
# 
# * 생존율 함수를 만들어 보겠습니다.

# In[ ]:


def survpct(col):
    return temp.groupby(col)['Survived'].mean()

survpct('Initial')


# * 생존 숫자로 봅니다.

# * test 에 있는 Dona의 나이를 보고 어디에 넣을지 보겠습니다.
# * Ms. 는 현대처럼 Miss + Mrs를 합친 말이 아니라 당시에는 귀족미망인을 의미하는 것이 었습니다. Mlle나 Mme등도 마드모아젤과 마담의 줄인말일 경우일 것입니다. 귀족 여성들로 보아야겠죠.

# In[ ]:


temp['LastName'] = data.Name.str.extract('([A-Za-z]+)')


# In[ ]:


pd.crosstab(temp.LastName, temp.Survived).T.style.background_gradient(cmap='summer_r')


# * 이제 우리는 Initial에서 Mr.등의 호칭을 뽑아내었고, 성을 뽑아내었습니다.
# 
# * 머신이 알파벳보다는 숫자를 좋아 하므로 숫자로 바꿉니다.
# 
# * 아, 그러기 전에 Dona를 처리해야지요.

# In[ ]:


temp.loc[temp['Initial'] == 'Dona']


# * 나이로 추측해서 Mrs.로 넣습니다.
# * 결측치를 처리하는 방법은 많으나 이렇게 하나일 경우에는 가장 적절한 추측을 사용하여 넣는 것도 괜찮습니다.
# 
# locate method 학습 : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html
# 

# In[ ]:


temp.loc[temp['Initial'] == 'Dona', 'Initial'] = 'Mrs'


# In[ ]:


pd.crosstab(temp.Initial, temp.Survived).T.style.background_gradient(cmap='summer_r')


# * Last name 은 전부 숫자로 바꿉니다.

# In[ ]:


temp['NumName'] = temp['LastName'].factorize()[0]


# In[ ]:


pd.crosstab(temp.NumName, temp.Survived).T.style.background_gradient(cmap='summer_r')


# In[ ]:


temp.loc[temp['LastName'] == 'Ali']


# * 보시다시피 같은 Last name에 같은 번호가 쓰여졌다.
# 
# * 끝에 [0]은 라벨만 보고 번호를 붙이는 것으로 정말 unique한 것이란 것은 안 본다는 것입니다.
#  
# * 자 이제 이름을 의미하는 중요한 요소 두 개를 숫자로 바꾸었으니 다음으로 갑니다.

# ### "Sex" 분석
# 
# * 함수를 만들어서 train파일을 보지요

# In[ ]:


train[['Sex','Survived']].groupby(['Sex']).mean()


# In[ ]:


def bag(col, target, title, title1):
    f,ax=plt.subplots(1,2,figsize=(12,5))
    train.groupby([col])[target].mean().plot(kind='bar', ax=ax[0])
    ax[0].set_title(title)
    sns.countplot(col, hue=target, data=train, ax=ax[1])
    ax[1].set_title(title1)
    plt.show()

bag('Sex','Survived','Survived per Sex','Sex Survived vs Not Survived')


# * 배에 있던 남자의 수는 여자의 수보다 훨씬 많습니다. 여전히 생존 여성 수는 남성 수의 거의 두 배입니다. 선박 여성의 생존율은 약 75 % 인 반면 남성의 생존율은 약 18-19 %입니다.
#  
# * 이 것은 남성/여성을 1,2로 나누면 될 것 같은 뻔해 보이는 것이지만 좀 더 새분화하면 좋아 보입니다.
# * 예를 들어 아기들은 아기이지, 남자인지 여자인지 구명보트 태울 때 안 물어 볼 것이기 때문입니다.
# * 오히려 (불행하게도) 귀족 아기인지 서민의 아기인지는 행과불행을 가를 수 있습니다 ㅠㅠ
# * 생존 Pclass별로 성별을 봅니다.

# In[ ]:


pd.crosstab([train.Sex, train.Survived],train.Pclass,margins=True).style.background_gradient(cmap='summer_r')


# * 사회는 불공평 했으나 최소한 남자들의 신사도는 있었다고 할 수 있을 것 같습니다.

# ### Age 분석
#  
# * Age는 Continuous한 값입니다.
# * 빈칸이 많아서 빈칸처리가 결정적인 역할을 할 것 같습니다.
#  
# * Age의 최대, 최소, 중간을 보겠습니다.

# In[ ]:


print('Oldest Passenger was', data['Age'].max(), 'Years')
print('Youngest Passenger was', data['Age'].min(), 'Years')
print('Average Age on the ship was', int(data['Age'].mean()), 'Years')


# In[ ]:


sns.swarmplot(x=train['Survived'], y=train['Age'])
plt.xlabel("Survived")
plt.ylabel("Age")
plt.show()


# In[ ]:


f, ax = plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass", "Age", hue="Survived", data=train, split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))
sns.violinplot("Sex","Age", hue="Survived", data=train, split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))
plt.show()


# #### 관찰 :
# 
# 1) Pclass에 따라 어린이 수가 증가하고 10 세 미만의 어린이 (즉, 어린이)의 생존율은 Pclass에 상관없이 양호해 보입니다.
# 
# 2) Pclass1에서 20-50세의 Passeneger의 생존 가능성은 높고 여성에게는 더 좋습니다.
# 
# 3) 남성의 경우 생존 확률은 나이가 증가함에 따라 감소합니다.
# 
# 
# 우선 age의 빈칸 부터 해결 합니다.
# 
# 앞에서 살펴본 것처럼 Age 항목에는 177 null 값이 있습니다. 이러한 NaN 값을 대체하기 위해 데이터 집합의 평균 수명을 지정할 수 있습니다.
# 
# 그러나 문제는 평균 연령이 29 세를 4세 아이에게 할당 할 수 없습니다. 승객이 어떤 연령대에 있는지 알 수있는 방법이 있을까요? 이름에서 힌트를 찾아 봅니다.

# * 그리고 Initial 별 평균 연령을 보고 Age에 적용 시키는 것이 좋을 것 같습니다.

# In[ ]:


temp.groupby('Initial').agg({'Age': ['mean', 'count']}) #이니셜 별 평균 연령 체크


# In[ ]:


# 이니셜 별 평균 연령을 빈값에 넣어 봅니다.

temp = temp.reset_index(drop=True)

temp['Age'] = temp.groupby('Initial')['Age'].apply(lambda x: x.fillna(x.mean()))

temp[31:50]


# [람다함수 공부](https://trinket.io/joshua-mobile-choi-1756/courses/python-3-4#/tasks/task-34-lambda)
# 
# [람다함수 메소드](https://trinket.io/joshua-mobile-choi-1756/courses/python-3-4#/tasks/task-35-lambda-and-functions)
# 
# * 이제 Initial을 좀 정리합니다.

# In[ ]:


temp['Initial'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona' , 'Dr', 'Jonkheer', 'Lady', 'Major', 'Master',  'Miss'  ,'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir'], ['Sacrificed', 'Respected', 'Nobles', 'Mr', 'Mrs', 'Respected', 'Mr', 'Nobles', 'Respected', 'Kids', 'Miss', 'Nobles', 'Nobles', 'Mr', 'Mrs', 'Nobles', 'Sacrificed', 'Nobles'],inplace=True)
temp['Initial'].replace(['Kids', 'Miss', 'Mr', 'Mrs', 'Nobles', 'Respected', 'Sacrificed'], [4, 4, 2, 5, 6, 3, 1], inplace=True)


# In[ ]:


temp['Age_Range'] = pd.qcut(temp['Age'], 10)


# In[ ]:


survpct('Age_Range')


# In[ ]:


temp['Agroup'] = 0

temp.loc[temp['Age'] < 1.0, 'Agroup'] = 1
temp.loc[(temp['Age'] >=1.0) & (temp['Age'] <= 3.0), 'Agroup'] = 2
temp.loc[(temp['Age'] > 3.0) & (temp['Age'] < 11.0), 'Agroup'] = 7
temp.loc[(temp['Age'] >= 11.0) & (temp['Age'] < 15.0), 'Agroup'] = 13
temp.loc[(temp['Age'] >= 15.0) & (temp['Age'] < 18.0), 'Agroup'] = 16
temp.loc[(temp['Age'] >= 18.0) & (temp['Age'] <=  20.0), 'Agroup'] = 18
temp.loc[(temp['Age'] > 20.0) & (temp['Age'] <= 22.0), 'Agroup'] = 21
temp.loc[(temp['Age'] > 22.0) & (temp['Age'] <= 26.0), 'Agroup'] = 24
temp.loc[(temp['Age'] > 26.0) & (temp['Age'] <= 30.0), 'Agroup'] = 28
temp.loc[(temp['Age'] > 30.0) & (temp['Age'] <= 32.0), 'Agroup'] = 31
temp.loc[(temp['Age'] > 32.0) & (temp['Age'] <= 34.0), 'Agroup'] = 33
temp.loc[(temp['Age'] > 34.0) & (temp['Age'] <= 38.0), 'Agroup'] = 36
temp.loc[(temp['Age'] > 38.0) & (temp['Age'] <= 52.0), 'Agroup'] = 45
temp.loc[(temp['Age'] > 52.0) & (temp['Age'] <= 75.0), 'Agroup'] = 60
temp.loc[temp['Age'] > 75.0, 'Agroup'] = 78


# In[ ]:


temp.head()


# * Age는 그룹화 시키면 좋으나 학습을 위해서 그냥 놓아두고, 그룹화 연습은 Fare로 하겠습니다.

# * 위를 보고 sex를 남,녀, 1세 이하 Baby로 나누겠습니다. 1,2,3번을 주지오

# In[ ]:


temp.loc[(temp['Sex'] == 'male'), 'Sex'] = 1
temp.loc[(temp['Sex'] == 'female'), 'Sex'] = 2
temp.loc[(temp['Age'] < 1), 'Sex'] = 3


# In[ ]:


survpct('Sex')


# #### Family or Alone?
# * "SibSp" + "Parch" 분석
#  
# * SibSp - 이 항목은 탑승자가 혼자인지 또는 가족과 함께 있는지를 나타냅니다.
# * *Sibling = 형제, 자매, 의붓 형제, 이복 누이
#  
# * Spouse = 남편, 아내
#  
# * Parch는 부모와 함께 탔는지를 봅니다.
#  
# * 이 그룹 둘을 'Alone"그룹과 "Family'그룹으로 나눕니다.

# In[ ]:


temp.loc[(temp['SibSp'] == 0) & (temp['Parch'] == 0), 'Alone'] = 1


# In[ ]:


temp['Family'] = temp['Parch'] + temp['SibSp'] + 1


# In[ ]:


temp.head(n=10)


# In[ ]:


survpct('Family')


# * 크로스 탭은 다시 식구 많은 쪽은 Pclass3에 있음을 보여줍니다.
#  
# * 여기에서도 결과는 매우 비슷합니다. 부모와 함께 탑승 한 승객은 생존 가능성이 더 높습니다. 그러나 숫자가 올라 갈수록 줄어 듭니다.
#  
# * 생존 가능성은 배에 1-3 명의 부모가있는 누군가에게 좋습니다. 혼자 또한 생존 가능성이 낮은 것으로 판명되고 가족이 4 명이상 있으면 생존 가능성이 줄어 듭니다. 이는 소수의 가족들이 있는 귀족층이 생존하고, 혼자가 많은 젊은 이들은 양보를 할 수 밖에 없고, 가족이 많은 사람들(특히 귀족이 아닌 3등칸 사람들)은 전원이 타지 못 하면 어느 누구도 탈 수가 없는 비극적인 당시 상황을 보여 줍니다.

# In[ ]:


bag('Parch', 'Survived', 'Survived per Parch', 'Parch Survived vs Not Survived')


# In[ ]:


pd.crosstab([temp.Family, temp.Survived], temp.Pclass, margins=True).style.background_gradient(cmap='summer_r')


# #### "Ticket"분석
# * Ticket의 형태를 보겠습니다.

# In[ ]:


temp.Ticket.head()


# * 도무지 감이 안 잡히는 배열입니다.
# * 빈칸이 없는지 보겠습니다.

# In[ ]:


temp.Ticket.isnull().any()


# * 티켓에서 영문있는 것과 숫자만 있는 것을 따봅니다.

# In[ ]:


temp['Initick'] = temp.Ticket.str.extract('^([A-Za-z0-9]+)')

temp = temp.reset_index(drop=True)  # 복사한 항목들을 사용하다보면 'ValueError: cannot reindex from a duplicate axis` 요런 에러가 나오는 경우가 많은데 이런 것은 요 코드로 리셋을 한 번 해주면 됩니다.

temp.head()


# In[ ]:


temp['Initick'] = temp.Ticket.str.extract('^([A-Za-z]+)')


# In[ ]:


temp.head()


# In[ ]:


temp['NumTicket'] = temp['Initick'].factorize()[0]


# In[ ]:


temp.head(n=15)


# In[ ]:


temp.groupby('NumTicket')['Survived'].mean().to_frame().plot(kind='hist')
plt.title('Distribution of survival rate for different tickets');


# ### "Fare" 분석

# In[ ]:


print('Highest Fare was:', temp['Fare'].max())
print('Lowest Fare was:', temp['Fare'].min())
print('Average Fare was:', temp['Fare'].mean())


# In[ ]:


f,ax=plt.subplots(1, 3, figsize=(20, 6))
sns.distplot(train[train['Pclass'] == 1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(train[train['Pclass'] == 2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(train[train['Pclass'] == 3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show()


# * Pclass1의 승객 요금에는 큰 분포가있는 것으로 보이며 불연속 값으로 변환 할 수 있습니다.
# * Fare를 그룹으로 나누어 놓겠습니다.
# * qcut을 활용하면 원하는 조각으로 데이터를 나누어 줍니다.
# * cut와 qcut의 차이

# In[ ]:


def groupmean(a,b):
    return temp.groupby([a])[b].mean().to_frame().style.background_gradient(cmap='summer_r')

temp['Fare_Range'] = pd.qcut(train['Fare'], 10)
groupmean('Fare_Range', 'Fare')


# * Fare를 그룹화 시킵니다. Fgroup이라고 이름 짓겠습니다.

# 0 and below -> 0
# 
# 7.125 and below-> 5.0
# 
# 7.9 and below-> 7.5
# 
# 8.03 or less-> 8.0
# 
# Less than 10.5-> 9.5
# 
# Less than 23-> 16.0
# 
# 27.8 and below-> 25.5
# 
# 51 and below-> 38
# 
# 73.5 and below-> 62
# 
# Over 73.5-> 100

# In[ ]:


temp['Fgroup'] = 0

temp.loc[temp['Fare'] <= 0,'Fgroup'] = 0
temp.loc[(temp['Fare'] > 0) & (temp['Fare'] <= 7.125), 'Fgroup'] = 1
temp.loc[(temp['Fare'] > 7.125) & (temp['Fare'] <= 7.9), 'Fgroup'] = 2
temp.loc[(temp['Fare'] > 7.9) & (temp['Fare'] <= 8.03), 'Fgroup'] = 3
temp.loc[(temp['Fare'] > 8.03) & (temp['Fare'] < 10.5), 'Fgroup'] = 4
temp.loc[(temp['Fare'] >= 10.5) & (temp['Fare'] < 23.0), 'Fgroup'] = 5
temp.loc[(temp['Fare'] >= 23.0) & (temp['Fare'] <= 27.8), 'Fgroup'] = 6
temp.loc[(temp['Fare'] > 27.8) & (temp['Fare'] <= 51.0), 'Fgroup'] = 7
temp.loc[(temp['Fare'] > 51.0) & (temp['Fare'] <= 73.5), 'Fgroup'] = 8
temp.loc[temp['Fare'] > 73.5, 'Fgroup'] = 9

temp.head()


# ### "Cabin" 분석
# * cabin 의 위치에 따라 달라지는 것이 있는지 보겠습니다.

# In[ ]:


temp.Cabin.value_counts().head(10)


# In[ ]:


temp.Cabin.isnull().sum()


# * 빈칸이 무척 많습니다.
# * Cabin에 비어 있는 것이 많아 이를 다른 분류로 일단 잡고 기존 것은 이니셜로 분류합니다.
# * 빈 것은 X로 구분하려는데 이 또한 1,2,3 Pclass와 연동될 것 같으니 비어있고 1등급은 X, 2등급은 Y, 3등급은 Z로 하겠습니다.

# In[ ]:


temp['Inicab'] = 0
temp['Inicab'] = temp['Cabin'].str.extract('^([A-Za-z]+)')
temp.loc[((temp['Cabin'].isnull()) & (temp['Pclass'].values == 1)), 'Inicab'] = 'X'
temp.loc[((temp['Cabin'].isnull()) & (temp['Pclass'].values == 2)), 'Inicab'] = 'Y'
temp.loc[((temp['Cabin'].isnull()) & (temp['Pclass'].values == 3)), 'Inicab'] = 'Z'
    
temp.head()


# In[ ]:


temp['Inicab'] = temp['Inicab'].factorize()[0]
    
temp[11:20]


# #### "Embarked" 분석

# In[ ]:


pd.crosstab([temp.Embarked, temp.Pclass], [temp.Sex, temp.Survived], margins=True).style.background_gradient(cmap='summer_r')


# * 승선 장소 별로 생존 확률

# In[ ]:


sns.factorplot('Embarked', 'Survived', data=temp)
fig = plt.gcf()
fig.set_size_inches(5, 3)
plt.show()


# In[ ]:


f,ax=plt.subplots(2,2,figsize=(20,15))
sns.countplot('Embarked', data=temp, ax=ax[0,0])
ax[0,0].set_title('No. Of Passengers Boarded')
sns.countplot('Embarked', hue='Sex', data=temp, ax=ax[0,1])
ax[0,1].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked', hue='Survived', data=temp, ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked', hue='Pclass', data=temp, ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()


# 1) 포트 C의 생존 가능성은 0.55 정도이며 S는 가장 낮습니다.S에서 탑승 최대. 대다수는 Pclass3
# 
# 2) C의 승객들은 많은 비율이 살아남았습니다. 그 이유는 Pclass1 및 Pclass2 승객이 많아서 일 것입니다
# 
# 3) Embark S는 대부분의 부자들이 탑승한 항구지만 생존 가능성은 낮습니다. Pclass3의 승객도 많았습니다.
# 
# 4) 포트 Q는 승객의 거의 95 %가 Pclass3
# 
# * 빈칸이 두개 있는데 보겠습니다.

# In[ ]:


temp.loc[(temp.Embarked.isnull())]


# * 두 사람의 티켓 번호가 같습니다.
# * 혹시 같은 티켓 번호가 있는 다른 사람이 있는지 봅니다.

# In[ ]:


temp.loc[(temp.Ticket == '113572')]


# * 가장 비슷한 번호를 찾아 보겠습니다.

# In[ ]:


temp.sort_values(['Ticket'], ascending = True)[55:70]


# * 앞 뒤로 모두 S이고 Pclass도 모두 1인 것으로 봐서 S일 가능성이 큽니다.

# In[ ]:


temp.loc[(temp.Embarked.isnull()), 'Embarked'] = 'S'


# In[ ]:


temp.loc[(temp.Embarked.isnull())]


# In[ ]:


temp['Embarked'] = temp['Embarked'].factorize()[0]
    
temp[11:20]


# <a id = "part5"></a>
# ## Part 5: Feature Engineering
# [Go to the Table of Contents](#table_of_contents)

# * 문자를 숫자로 바꾸는 것도 Feature Engineering의 일부이나 위에서 대부분 다 했습니다.

# ### 항목 추가하기
# 
# * 위에 추가 항목을 몇 개 만들어 보았습니다.
# 
# * 그래도 몇 개 더 만들어 볼까요? 
# 
# * 5개 정도 만들어 봅니다.
# 
# * Priority - Nobles, Women in Pclass 1 & 2, Babies under 1, Kids under 17 in Pclass 1 & 2, higher fare, Women in Pclass 3 and so on
# * FH - Female Higher Survival Group
# * MH - Male Higher Survival Group
# * FL - Female Lower Surival Group
# * ML - Male Lower Survival Group

# * Priority - (1) Nobles (2) Women in Pclass 1  (3) Babies under 1 (4) Kids under 17 in Pclass 1 & 2  (5) Women in Pclass 2 (6) Higher Fare

# In[ ]:


survpct('Initial')


# In[ ]:


survpct('Pclass')


# In[ ]:


survpct('Sex')


# In[ ]:


survpct('Age').head()


# In[ ]:


survpct('Fgroup')


# In[ ]:


temp['Priority'] = 0
temp.loc[(temp['Initial'] == 6), 'Priority'] = 1
temp.loc[(temp['Pclass'] == 1) & (temp['Sex'] == 2), 'Priority'] = 2
temp.loc[(temp['Age'] < 1), 'Priority'] = 3
temp.loc[(temp['Pclass'] == 1) & (temp['Age'] <= 17), 'Priority'] = 4
temp.loc[(temp['Pclass'] == 2) & (temp['Age'] <= 17), 'Priority'] = 5
temp.loc[(temp['Pclass'] == 2) & (temp['Sex'] == 2), 'Priority'] = 6
temp.loc[(temp['Fgroup'] == 9), 'Priority'] = 7


# In[ ]:


survpct('Priority')


# In[ ]:


temp.Priority.value_counts()


# In[ ]:


survpct('Family')


# In[ ]:


survpct('Fgroup')


# In[ ]:


temp['FH'] = 0
temp.loc[(temp['Sex'] == 1), 'FH'] = 0
temp.loc[(temp['Sex'] == 2), 'FH'] = 1
temp.loc[(temp['Sex'] == 2) & (temp['Family'] == 2), 'FH'] = 2
temp.loc[(temp['Sex'] == 2) & (temp['Family'] == 3), 'FH'] = 3
temp.loc[(temp['Sex'] == 2) & (temp['Family'] == 4), 'FH'] = 4
temp.loc[(temp['Sex'] == 2) & (temp['Family'] == 1) & (temp['Pclass'] == 1), 'FH'] = 5
temp.loc[(temp['Sex'] == 2) & (temp['Family'] == 1) & (temp['Pclass'] == 2), 'FH'] = 6
temp.loc[(temp['Sex'] == 2) & (temp['Fgroup'] == 3), 'FH'] = 7
temp.loc[(temp['Sex'] == 2) & (temp['Fgroup'] >= 5), 'FH'] = 8


# In[ ]:


survpct('FH')


# In[ ]:


temp.FH.value_counts()


# In[ ]:


temp['MH'] = 0
temp.loc[(temp['Sex'] == 2), 'MH'] = 0
temp.loc[(temp['Sex'] == 1), 'MH'] = 1
temp.loc[(temp['Sex'] == 1) & (temp['Family'] == 2), 'MH'] = 2
temp.loc[(temp['Sex'] == 1) & (temp['Family'] == 3), 'MH'] = 3
temp.loc[(temp['Sex'] == 1) & (temp['Family'] == 4), 'MH'] = 4
temp.loc[(temp['Sex'] == 1) & (temp['Family'] == 1) & (temp['Pclass'] == 1), 'MH'] = 5
temp.loc[(temp['Sex'] == 1) & (temp['Family'] == 1) & (temp['Pclass'] == 2), 'MH'] = 6
temp.loc[(temp['Sex'] == 1) & (temp['Fgroup'] == 3), 'MH'] = 7
temp.loc[(temp['Sex'] == 1) & (temp['Fgroup'] >= 5), 'MH'] = 8


# In[ ]:


survpct('MH')


# In[ ]:


temp.MH.value_counts()


# In[ ]:


temp['FL'] = 0
temp.loc[(temp['Sex'] == 1), 'FL'] = 0
temp.loc[(temp['Sex'] == 2) & (temp['Fgroup'] < 5), 'FL'] = 1
temp.loc[(temp['Sex'] == 2) & (temp['Fgroup'] != 3), 'FL'] = 2
temp.loc[(temp['Sex'] == 2) & (temp['FH'] == 1), 'FL'] = 3
temp.loc[(temp['Sex'] == 2) & (temp['Family'] < 2), 'FL'] = 4
temp.loc[(temp['Sex'] == 2) & (temp['Family'] > 4), 'FL'] = 5
temp.loc[(temp['Sex'] == 2) & (temp['Family'] == 1) & (temp['Pclass'] == 3), 'FL'] = 6


# In[ ]:


survpct('FL')


# In[ ]:


temp.FL.value_counts()


# In[ ]:


temp['ML'] = 0
temp.loc[(temp['Sex'] == 2), 'ML'] = 0
temp.loc[(temp['Sex'] == 1) & (temp['Fgroup'] < 5), 'ML'] = 1
temp.loc[(temp['Sex'] == 1) & (temp['Fgroup'] != 3), 'ML'] = 2
temp.loc[(temp['Sex'] == 1) & (temp['MH'] <7), 'ML'] = 3
temp.loc[(temp['Sex'] == 1) & (temp['Family'] < 2), 'ML'] = 4
temp.loc[(temp['Sex'] == 1) & (temp['Family'] > 4), 'ML'] = 5
temp.loc[(temp['Sex'] == 1) & (temp['Family'] == 1) & (temp['Pclass'] == 3), 'ML'] = 6


# In[ ]:


survpct('ML')


# In[ ]:


temp.ML.value_counts()


# <a id = "part6"></a>
# ## Part 6: 마지막 항목 결정
# [Go to the Table of Contents](#table_of_contents)

# In[ ]:


temp['F1'] = temp['Priority']
temp['F2'] = temp['FH']
temp['F3'] = temp['MH']
temp['F4'] = temp['FL']
temp['F5'] = temp['ML']
temp['F6'] = temp['Initial']
temp['F7'] = temp['Fgroup']
temp['F8'] = temp['NumName']
temp['F9'] = temp['NumTicket']
temp['F10'] = temp['Family']
temp['F11'] = temp['Embarked']
temp['F12'] = temp['Sex']
temp['F13'] = temp['Pclass']


# * 이제 다음 단계로 갑니다.

# * 두개의 새로운 데이터 프레임을 만듭니다. 하나는 레이블 인코딩 다른 하나는 원핫 인코딩 (둘이 꼭 필요한 것이 아니라 연습이나 두 가지 방 법 모두 사용해봄)

# * 대표적인 인코딩에 Label Encoding이 있는데 이는 각 항목의 값을 서열화 시켜 주~욱 줄세운 것이라 생각하시면 됩니다.
# * 그 외에 자주쓰는 One Hot Encoding 같은 경우 열 내에서의 항목을 나누어서 (열이 주~욱 늘어나며) 이를 0이냐 1이냐로 구분해 놓은 것입니다.

# * 다시 말씀 드려서 레이블 인코딩은 줄을 세워서 번호를 부여하는 것이고, 원핫인코딩은 긴가 아닌가 두 가지입니다.
# 
# * 예를들어 나이별로 줄을 세워 너는 5번, 너는 6번이런식이 레이블 인코딩이고
# 
# * 16살이야? 1, 16살 아냐 0 ..그 다음 17살이야? 1 17살 아냐 0 ..이런 식이라 열의 수가 무지하게 늘어납니다.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# 새로운 Data Frame을 만듭니다.

# In[ ]:


dfl = pd.DataFrame() # for label encoding


# In[ ]:


good_columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13']
dfl[good_columns] = temp[good_columns]


# In[ ]:


dfl.head()


# In[ ]:


dfh = dfl.copy()


# In[ ]:


dfl_enc = dfl.apply(LabelEncoder().fit_transform)
                          
dfl_enc.head()


# In[ ]:


one_hot_cols = dfh.columns.tolist()
dfh_enc = pd.get_dummies(dfh, columns=one_hot_cols)

dfh_enc.head()


# <a id = "part7"></a>
# ## Part 7:  머신러닝 모델 만들기
# [Go to the Table of Contents](#table_of_contents)

# * 자, 이제 머신 러닝 모델을 만들어 보지요.
# * 우선 인코딩한 파일을 train과 test로 아까 구분해 놓은 행으로 쪼갭니다

# In[ ]:


train = dfh_enc[:ntrain]
test = dfh_enc[ntrain:]


# In[ ]:


X_test = test
X_train = train


# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


ran = RandomForestClassifier(random_state=1)
knn = KNeighborsClassifier()
log = LogisticRegression()
xgb = XGBClassifier()
gbc = GradientBoostingClassifier()
svc = SVC(probability=True)
ext = ExtraTreesClassifier()
ada = AdaBoostClassifier()
gnb = GaussianNB()
gpc = GaussianProcessClassifier()
bag = BaggingClassifier()

# 리스트 준비
models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         
model_names = ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier']
scores = {}

# 이어서 연속적으로 모델을 학습 시키고 교차 검증합니다.
for ind, mod in enumerate(models):
    mod.fit(X_train, y_train)
    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)
    scores[model_names[ind]] = acc


# * [리스트 학습](https://trinket.io/joshua-mobile-choi-1756/courses/python-3-4#/tasks/task-14-list)
# * [딕쇼너리 학습](https://trinket.io/joshua-mobile-choi-1756/courses/python-3-4#/tasks/task-16-dictionary)

# In[ ]:


# 결과 테이블을 만듭니다.
results = pd.DataFrame(scores).T
results['mean'] = results.mean(1)

result_df = results.sort_values(by='mean', ascending=False)#.reset_index()
result_df.head(11)


# In[ ]:


result_df = result_df.drop(['mean'], axis=1)
sns.boxplot(data=result_df.T, orient='h')
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)');


# In[ ]:


# 중요도를 보는 함수를 만듭니다.
def importance_plotting(data, xlabel, ylabel, title, n=20):
    sns.set(style="whitegrid")
    ax = data.tail(n).plot(kind='barh')
    
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    plt.show()


# In[ ]:


# 데이터 프레임에 항목 중요도를 넣습니다.
fi = {'Features':train.columns.tolist(), 'Importance':xgb.feature_importances_}
importance = pd.DataFrame(fi, index=fi['Features']).sort_values('Importance', ascending=True)


# In[ ]:


# 그래프 제목
title = 'Top 20 most important features in predicting survival on the Titanic: XGB'

# 그래프 그리기
importance_plotting(importance, 'Importance', 'Features', title, 20)


# In[ ]:


# 중요도를 데이터프레임에 넣습니다. Logistic regression에서는 중요도보다 coefficients를 사용합니다. 
# 아래는 Features라는 열에 트레인의 열들의 이름을 리스트로 만들어서 넣고 Importance에는 Logistic regression에는 coefficient를 바꾸어 넣어라는 넘파이 명령입니다.(즉 가로를 세로로)
fi = {'Features':train.columns.tolist(), 'Importance':np.transpose(log.coef_[0])}
importance = pd.DataFrame(fi, index=fi['Features']).sort_values('Importance', ascending=True)
# 그래프 타이틀
title = 'Top 20 important features in predicting survival on the Titanic: Logistic Regression'

# 그래프 그리기
importance_plotting(importance, 'Importance', 'Features', title, 20)


# In[ ]:


# 5가지 모델에 대한 항목 중요도 얻기
gbc_imp = pd.DataFrame({'Feature':train.columns, 'gbc importance':gbc.feature_importances_})
xgb_imp = pd.DataFrame({'Feature':train.columns, 'xgb importance':xgb.feature_importances_})
ran_imp = pd.DataFrame({'Feature':train.columns, 'ran importance':ran.feature_importances_})
ext_imp = pd.DataFrame({'Feature':train.columns, 'ext importance':ext.feature_importances_})
ada_imp = pd.DataFrame({'Feature':train.columns, 'ada importance':ada.feature_importances_})

# 이를 하나의 데이터프레임으로
importances = gbc_imp.merge(xgb_imp, on='Feature').merge(ran_imp, on='Feature').merge(ext_imp, on='Feature').merge(ada_imp, on='Feature')

# 항목당 평균 중요도
importances['Average'] = importances.mean(axis=1)

# 랭킹 정하기
importances = importances.sort_values(by='Average', ascending=False).reset_index(drop=True)


# <a id = "part8"></a>
# ## Part 8: 중요도에 따라 모델 재 설정
# [Go to the Table of Contents](#table_of_contents)

# In[ ]:


# 중요도를 다시 데이터 프레임에 넣기
fi = {'Features':importances['Feature'], 'Importance':importances['Average']}
importance = pd.DataFrame(fi).set_index('Features').sort_values('Importance', ascending=True)

# 그래프 타이틀
title = 'Top 20 important features in predicting survival on the Titanic: 5 model average'

# 그래프 보기
importance_plotting(importance, 'Importance', 'Features', title, 20)


# In[ ]:


importance1 = importance[-381:]

importance1[371:381]


# In[ ]:


# 영양가 있는 380개만 넣기
mylist = list(importance1.index)


# In[ ]:


train1 = pd.DataFrame()
test1 = pd.DataFrame()

for i in mylist:
    train1[i] = train[i]
    test1[i]= test[i]
    
train1.head()


# In[ ]:


train = train1
test = test1

# 모델의 변수를 다시 정의하고
X_train = train
X_test = test

# 바꿉니다.
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


ran = RandomForestClassifier(random_state=1)
knn = KNeighborsClassifier()
log = LogisticRegression()
xgb = XGBClassifier(random_state=1)
gbc = GradientBoostingClassifier(random_state=1)
svc = SVC(probability=True)
ext = ExtraTreesClassifier(random_state=1)
ada = AdaBoostClassifier(random_state=1)
gnb = GaussianNB()
gpc = GaussianProcessClassifier()
bag = BaggingClassifier(random_state=1)

# 리스트 준비
models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         
model_names = ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier']
scores2 = {}

# 학습 및 교차 검증
for ind, mod in enumerate(models):
    mod.fit(X_train, y_train)
    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)
    scores2[model_names[ind]] = acc


# In[ ]:


# 결과 테이블을 만듭니다.
results = pd.DataFrame(scores2).T
results['mean'] = results.mean(1)

result_df = results.sort_values(by='mean', ascending=False)#.reset_index()
result_df.head(11)
result_df = result_df.drop(['mean'], axis=1)
sns.boxplot(data=result_df.T, orient='h')
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)');


# <a id = "part9"></a>
# ## Part 9: 하이퍼 파라미터 튜닝
# [Go to the Table of Contents](#table_of_contents)
# 
# ### SVC
# * Scikit-Learn에서는 3가지 모형 최적화 도구를 지원하는데 validation_curve/ GridSearchCV/ ParameterGrid이다
# * fit 메소드를 호출하면 grid search가 자동으로 여러개의 내부 모형을 생성하고 이를 모두 실행시켜서 최적 파라미터를 찾는다.
# 
# * bestscore는 최고 점수이고 best estimator는 최고 점수를 낸 파라미터를 가진 모형
# * c값과 gamma값은 10의 배수로 일반적으로 한다.
# * 감마 매개 변수는 단일 학습 예제의 영향이 도달하는 정도를 정의하며 낮은 값은 'far'를, 높은 값은 'close'를 나타냅니다. 감마 매개 변수는 서포트 벡터로 모델에 의해 선택된 샘플의 영향 반경의 역으로 볼 수 있습니다.
# * C 매개 변수는 의사 결정 표면의 단순성에 대한 훈련 예제의 오 분류를 제거합니다. C가 낮을수록 결정 표면이 매끄럽고 높은 C는 모델이 더 많은 샘플을 서포트 벡터로 자유롭게 선택할 수 있도록하여 모든 학습 예제를 올바르게 분류하는 것을 목표로합니다.
# * Verbose는 불리안 값으로 True로 넣으면 꼬치 꼬치 다 알려주는데, 대신 시간이 좀 더 오래 걸립니다.
# * cv =5는 5 fold로 교차 검증한다는 뜻입니다.

# In[ ]:


# 파라미터 서치
Cs = [0.01, 0.1, 1, 5, 10, 15, 20, 50]
gammas = [0.001, 0.01, 0.1]

# 파라미터 그리드 셋팅
hyperparams = {'C': Cs, 'gamma' : gammas}

# 교차검증
gd=GridSearchCV(estimator = SVC(probability=True), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy", n_jobs=-1)

# 모델 fiting 및 결과
gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_params_)


# ### Gradient Boosting Classifier
# * learning_rate는 각 트리의 기여를 줄이는 역할을 합니다.
# * n_estimator는 각 경우의 트리 숫자입니다.

# In[ ]:


learning_rate = [0.01, 0.05, 0.1, 0.2, 0.5]
n_estimators = [100, 1000, 2000]
max_depth = [3, 5, 10, 15]

hyperparams = {'learning_rate': learning_rate, 'n_estimators': n_estimators}

gd=GridSearchCV(estimator = GradientBoostingClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy", n_jobs=-1)

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_params_)


# ### Logistic Regression
# * Penalty - L1 을 사용하는 회귀 모델을 Lasso Regression이라고하고 L2를 사용하는 모델을 Ridge Regression이라고합니다. 이 둘의 주요 차이점은 페널티입니다. 릿지 회귀는 손실 함수에 페널티 항으로 계수의 "제곱 크기"를 추가합니다. L2-norm이 오차를 제곱하기 때문에 (오류> 1 인 경우 로트가 증가 함) 모델은 L1-norm보다 훨씬 큰 오차 (e vs e ^ 2)를 보게되므로 훨씬 더 민감합니다. 따라서 오류를 최소화하기 위해 모델을 조정해줍니다.
# * C는 estimator 입니다. logspace 1차원 10개 배열로 0에서 4까지를 estimator로 놓은 것입니다.

# In[ ]:


penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)

hyperparams = {'penalty': penalty, 'C': C}

gd=GridSearchCV(estimator = LogisticRegression(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy", n_jobs=-1)

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_params_)


# ### XGBoost Step 1.

# In[ ]:


learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
n_estimators = [10, 50, 100, 250, 500, 1000]

hyperparams = {'learning_rate': learning_rate, 'n_estimators': n_estimators}

gd=GridSearchCV(estimator = XGBClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy", n_jobs=-1)

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_params_)


# ### XGB Step 2.

# In[ ]:


max_depth = [3, 4, 5, 6, 7, 8, 9, 10]
min_child_weight = [1, 2, 3, 4, 5, 6]

hyperparams = {'max_depth': max_depth, 'min_child_weight': min_child_weight}

gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.2, n_estimators=10), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy", n_jobs=-1)

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_params_)


# ### XGB Step 3.

# In[ ]:


gamma = [i*0.1 for i in range(0,5)]

hyperparams = {'gamma': gamma}

gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.2, n_estimators=10, max_depth=6, 
                                          min_child_weight=1), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy", n_jobs=-1)

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_params_)


# ### XGB Step 4

# In[ ]:


subsample = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
colsample_bytree = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    
hyperparams = {'subsample': subsample, 'colsample_bytree': colsample_bytree}

gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.2, n_estimators=10, max_depth=6, 
                                          min_child_weight=1, gamma=0), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy", n_jobs=-1)

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_params_)


# ### XGB Step 5

# In[ ]:


reg_alpha = [1e-5, 1e-2, 0.1, 1, 100]
    
hyperparams = {'reg_alpha': reg_alpha}

gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.2, n_estimators=10, max_depth=6, 
                                          min_child_weight=1, gamma=0, subsample=1, colsample_bytree=1),
                                         param_grid = hyperparams, verbose=True, cv=5, scoring = "accuracy", n_jobs=-1)

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_params_)


# ### Gaussian Process

# In[ ]:


n_restarts_optimizer = [0, 1, 2, 3]
max_iter_predict = [1, 2, 5, 10, 20, 35, 50, 100]
warm_start = [True, False]

hyperparams = {'n_restarts_optimizer': n_restarts_optimizer, 'max_iter_predict': max_iter_predict, 'warm_start': warm_start}

gd=GridSearchCV(estimator = GaussianProcessClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy", n_jobs=-1)

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_params_)


# ### Adaboost.

# In[ ]:


n_estimators = [10, 100, 200, 500]
learning_rate = [0.001, 0.01, 0.1, 0.5, 1, 1.5, 2]

hyperparams = {'n_estimators': n_estimators, 'learning_rate': learning_rate}

gd=GridSearchCV(estimator = AdaBoostClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy", n_jobs=-1)

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_params_)


# ### KNN

# In[ ]:


n_neighbors = [1, 2, 3, 4, 5]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = [1, 2, 3, 4, 5, 10]

hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 
               'n_neighbors': n_neighbors}

gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy", n_jobs=-1)

# Fitting model and return results
gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_params_)


# ### Random Forest.

# In[ ]:


n_estimators = [10, 50, 100, 200]
max_depth = [3, None]
max_features = [0.1, 0.2, 0.5, 0.8]
min_samples_split = [2, 6]
min_samples_leaf = [2, 6]

hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features,
               'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}

gd=GridSearchCV(estimator = RandomForestClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy", n_jobs=-1)

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_params_)


# ### Extra Trees

# In[ ]:


n_estimators = [10, 25, 50, 75, 100]
max_depth = [3, None]
max_features = [0.1, 0.2, 0.5, 0.8]
min_samples_split = [2, 10]
min_samples_leaf = [2, 10]

hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features,
               'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}

gd=GridSearchCV(estimator = ExtraTreesClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy", n_jobs=-1)

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_params_)


# ### Bagging Classifier

# In[ ]:


n_estimators = [10, 50, 75, 100, 200]
max_samples = [0.1, 0.2, 0.5, 0.8, 1.0]
max_features = [0.1, 0.2, 0.5, 0.8, 1.0]

hyperparams = {'n_estimators': n_estimators, 'max_samples': max_samples, 'max_features': max_features}

gd=GridSearchCV(estimator = BaggingClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy", n_jobs=-1)

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_params_)


# <a id = "part10"></a>
# ## Part 10: 모델 재 트레이닝
# [Go to the Table of Contents](#table_of_contents)

# In[ ]:


# 튜닝 모델 시작
# sample을 split하는 것은 전체데이터 80%를 트레인셋에 20%는 테스트셋에 줌  
ran = RandomForestClassifier(max_depth=None, max_features=0.1, min_samples_leaf=2, min_samples_split=2, n_estimators=50, random_state=1)

knn = KNeighborsClassifier(leaf_size=1, n_neighbors=4, weights='distance')

log = LogisticRegression(C=2.7825594022071245, penalty='l2')

xgb = XGBClassifier(learning_rate=0.1, n_estimators=10, max_depth=7, 
                                          min_child_weight=5, gamma=0, subsample=1, colsample_bytree=1, reg_alpha=1e-05)

gbc = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, n_estimators=1000)

svc = SVC(probability=True, gamma=0.001, C=5)

ext = ExtraTreesClassifier(max_depth=None, max_features=0.2, min_samples_leaf=10, min_samples_split=2, n_estimators=100, random_state=1)

ada = AdaBoostClassifier(learning_rate=0.5, n_estimators=500, random_state=1)

gpc = GaussianProcessClassifier(max_iter_predict=1, n_restarts_optimizer=0, warm_start=True)

bag = BaggingClassifier(max_features=1.0, max_samples=1.0, n_estimators=75, random_state=1)

# 리스트
models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         
model_names = ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier']
scores3 = {}

# Sequentially fit and cross validate all models
for ind, mod in enumerate(models):
    mod.fit(X_train, y_train)
    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)
    scores3[model_names[ind]] = acc


# In[ ]:


results = pd.DataFrame(scores).T
results['mean'] = results.mean(1)
result_df = results.sort_values(by='mean', ascending=False)
result_df.head(11)


result_df = result_df.drop(['mean'], axis=1)
sns.boxplot(data=result_df.T, orient='h')
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)');


# <a id = "part11"></a>
# ## Part 11: 마지막 보팅
# [Go to the Table of Contents](#table_of_contents)

# In[ ]:


#튜닝한 파라미터로 하드보팅
grid_hard = VotingClassifier(estimators = [('Random Forest', ran), 
                                           ('Logistic Regression', log),
                                           ('XGBoost', xgb),
                                           ('Gradient Boosting', gbc),
                                           ('Extra Trees', ext),
                                           ('AdaBoost', ada),
                                           ('Gaussian Process', gpc),
                                           ('SVC', svc),
                                           ('K Nearest Neighbour', knn),
                                           ('Bagging Classifier', bag)], voting = 'hard')

grid_hard_cv = model_selection.cross_validate(grid_hard, X_train, y_train, cv=10)
grid_hard.fit(X_train, y_train)

print("Hard voting on test set score mean: {:.2f}". format(grid_hard_cv['test_score'].mean() * 100))


# In[ ]:


grid_soft = VotingClassifier(estimators = [('Random Forest', ran), 
                                           ('Logistic Regression', log),
                                           ('XGBoost', xgb),
                                           ('Gradient Boosting', gbc),
                                           ('Extra Trees', ext),
                                           ('AdaBoost', ada),
                                           ('Gaussian Process', gpc),
                                           ('SVC', svc),
                                           ('K Nearest Neighbour', knn),
                                           ('Bagging Classifier', bag)], voting = 'soft')

grid_soft_cv = model_selection.cross_validate(grid_soft, X_train, y_train, cv=10)
grid_soft.fit(X_train, y_train)

print("Soft voting on test set score mean: {:.2f}". format(grid_soft_cv['test_score'].mean() * 100))


# <a id = "part12"></a>
# ## Part 12: 마지막 모델 예측
# [Go to the Table of Contents](#table_of_contents)

# In[ ]:


# Final predictions2
predictions = grid_hard.predict(X_test)

submission = pd.concat([pd.DataFrame(passId), pd.DataFrame(predictions)], axis = 'columns')

submission.columns = ["PassengerId", "Survived"]
submission.to_csv('titanic_submission1.csv', header = True, index = False)


# In[ ]:


# Final predictions
predictions = grid_soft.predict(X_test)

submission = pd.concat([pd.DataFrame(passId), pd.DataFrame(predictions)], axis = 'columns')

submission.columns = ["PassengerId", "Survived"]
submission.to_csv('titanic_submission2.csv', header = True, index = False)


# <a id = "part13"></a>
# ## Part 13: 제출
# [Go to the Table of Contents](#table_of_contents)

# In[ ]:


# And we finally make a submission 그리고 제출 합니다.
# Please make sure you "commit" (It take a few minutes) / commit버턴을 누르시는 것을 잊지 마세요 (몇 분 걸립니다)
# And then you will see the submission file on the top right hand side at Data>Output>Kaggle/working / 그럼 우측 상단 데이터 아웃풋에서 제출용 결과물이 나올 것입니다.

