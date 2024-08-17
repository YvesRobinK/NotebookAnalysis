#!/usr/bin/env python
# coding: utf-8

# ## Step 1 : Define The Problem
# * We want to predict which passengers survived in titanic
# * This is a classification problem

# ## STEP 2 :  Collect the Data
# * we take the date from Kaggle.

# ## STEP 3 : Preapare Data for Consumption

# ### >> 3.1 : IMPORT THE LIBRARIES

# In[ ]:


#load packages
import sys 
print("Python version: {}". format(sys.version))

import pandas as pd 
print("pandas version: {}". format(pd.__version__))

import matplotlib # visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))


import random
import time


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ### >> 3.11 :Load Data Modelling Libraries

# In[ ]:


#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
#from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


# ### >> 3.2 Meet and Greet Data (like First Date)

# In[ ]:


# Verileri tanı ve bilgi edin.
# Veri neye benziyor --> datatype and values
# Veriyi oluşturanlar neeler (independent/feature variables(s)),
# Verinin gerçek hayattaki hedefleri nelerdir --> (dependent/target variable(s)). 
#  info() sample() function ile veriye hızlı bir bakalım -> Değişken türleri (i.e. qualitative vs quantitative). 
# Survived  -->  outcome or dependent variable(bağımlı değişken) , binary veri tipidir (1 ya da 0 değeri alır)
#           --> 1 = survived ; 0 --> not survived
# "Survived" dışındakilerin diğer tüm değişkenler, potansiyel belirleyici veya bağımsız değişkenlerdir.
# "PassengerID" -->  random unique identifier diyebiliriz yani rastgele unqiue tanımlayıcılar --> bağımlı değişkene etkisi yoktur --> analiz dışı bırakılır
# "Ticket " --> random unique identifier diyebiliriz yani rastgele unqiue tanımlayıcılar --> bağımlı değişkene etkisi yoktur --> analiz dışı bırakılır
# "Pclass " --> ticket için ordinal veri tipidir ; sosyo ekonomik durum için önbilgi verebilir ; yolcuların oturduğu class bilgisi
            # 1 = upper class, 2 = middle class, and 3 = lower class.
# "Name " --> yolcuların isimleri  ; nominal veri türüdür 
#         --> Nominal veriler, birbiriyle örtüşmeyen çeşitli gruplara ayrılabilen “etiketli” veya “adlandırılmış” verilerdir (aralarında sınıf farkı yok) 
# "Sex " --> cinsiyeti temsil eder ; nominal veri tipi -> Matematiksel hesaplamalar için dummy değişkenlere dönüştürüleceklerdir.
# "Embarked " --> Yolcuların bindiği limanı temsil eder ; nominal veri tipi -> Matematiksel hesaplamalar için dummy değişkenlere dönüştürüleceklerdir.
# "Age" --> Yolcuların yaşını temsil eder ;sürekli(continious) kantitatif veri tipidir
# "Fare " --> Bilet ücretini temsil ede --> sürekli quantitative  veri tipidir(quantitative = nicel)
# "SibSp " --> gemideki ilgili kardeşlerin/eşlerin sayısını temsil eder. --> discrete quantitative veri tipidir
# "Parch " --> gemideki ilgili ebeveyn/çocuk sayısını temsil eder. --> discrete quantitative veri tipidir
    # Bu iki değişkeni(SibSp ve Parch ) "feature engineering" yaparak tek bir "Family" değişekine dönüştürebiliriz
# "Cabin " -->    nominal veri tipidir ; Fazla sayıda null değer oldugu için bunu sileriz


# In[ ]:


# verilerimi  dahil ettim
data_train = pd.read_csv('/kaggle/input/titanic/train.csv')
data_test  = pd.read_csv('/kaggle/input/titanic/test.csv')

# verimi kopyalıyorum
data1 = data_train.copy()

# Test setimi ve train veri setimi tek bir veri setine  getirip clean işlemleri yapıcam
data_cleaner = [data1 , data_test]


# In[ ]:


# veri setim hakkında ön bilgi ediniyorum
data1.info()


# In[ ]:


# veri setimin ilk verilerine bakıyorum
data1.head()


# In[ ]:


# veri setimin içindeki rastgele 10 gözlem birimine bakıyorum
data1.sample(10)


# In[ ]:


# veri setimdeki veri tiplerim
data1.dtypes


# In[ ]:


data1.shape
# 891 gözlem ve 12 değişken var


# In[ ]:


# değişken isimleri
data1.columns


# In[ ]:


# train veri setimdeki eksik değerleri check ediyorum 
data1.isnull().sum()


# In[ ]:


# test veri setimdeki eksik değerleri check ediyorum
data_test.isnull().sum()


# In[ ]:


# print fonksiyonue ile bakıyorum
print('Train columns with null values:\n', data1.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', data_test.isnull().sum())
print("-"*10)


# ### >> 3.21  The 4 C's of Data Cleaning: Correcting, Completing, Creating, and Converting

# In[ ]:


# Bu kısımda verilerimizi temizleyeceğiz ---> 4C
# 1-) Correcting (Düzeltme) --> Aykırı(outliers) ve anormal değerleri(aberrant values ) düzeltmek
# - Bu veri setinde aykırı ya da kabul-edilemez veriler görünmüyor ancak age ve fare değişkeninde 
#.. potansiyel aykırı gözlemler(outliers) olabilir ancak onlar su an makul veriler gibi duruyor..
#..onları keşifçi veri analizi kısmından sonra veri setiminden atıp atmayacagımıza karar vericez.
#Ancak eğer buradaki aykırı değerler örneğin yaş değişkeninde 80 değişde 800 olsaydı bu kısımda düzeltirdim

# 2-) Completing(Tamamlama) --> Eksik(missing) Verileri tamamlama
# Age ,Cabin  and Embarked kısımlarında null values ya da missing data var.
# -Missing data'lar problemlere yol açabilir çünkü bazı algortimalar null değerler ile baş edemiyorlar..
#..ve başarısız oluyorlar ancak öte yandan "decision trees" null değerler ile baş edebiliyor.Bu nedenle
#.. modellemeye başlamdan evvel bunları düzeltek önemlidir çünkü kurdugumuz modelleri karılaaştırıcağız.
#..Eksik veriyi tamamlama ile ilgili 2 tane genel method vardır.Ya sileriz ya da eksik veri ile doldururuz..
#..ancak veriyi silmek çok fazla önerilmez,gerçekten büyük bir eksiklik yoksa silinmemelidir.Bunun yerine..
#.. verileri doldurmak en iyi çözümdür.Qualitative data için n cok kullanılan çözüm ise eksiklikleri ..
#.. mean, median, ya da mean + randomized standard deviation ya da mode ile doldurmaktır..
#..Specific bir kritere göre doldurulabilir mesela yaş kriteri doldurulacak ise sınıflara göre yaşlar analiz edilip..
#.. o sınfılardaki yaş ortalamasına göre o sınıflardaki eksik yaş veriileri doldurulabilir..
#..Daha komplex doldurma yöntemleri de vardır.Bu complexity nin  gerçekten değer katıp katmadığını belirlemek için temel modelle karşılaştırılmalıdır.
#..Biz burada bu veri seti için ; age i median ile doldurucaz ; "cabin" attribute u sileceğiz ; "embark" median ile doldurulacak
#Bunları yapıcaz ancak bu karar modelin accuracy scorunu beğenmezsek değiştirebiliriz

# 3-) Creating (Feature engineering kısmı)
#Feature engineering ; outcome değişkenimizi tahmin etmek için ielimizdeki mevcut değişkenler ile yeni değişkenler üretmektir
#..Yani yeni bir değişken oluşturucaz ve  "survived" bağımlı değişkenimiz etkileyip etkilemediğine karar vericez.

# 4-) Converting (Dönüştürme)
# Our categorical datamız bize "object" olarak verilmiş ki bunu math islemlerini zorlaştırır
#..Bu veri setinde object veri tiplerini  categorical dummy değişkenlerine dönüştürücem.


# In[ ]:


data1.describe().T


# In[ ]:


data1.describe(include ="all").T


# ### >> 3.22 Clean Data

# In[ ]:


print("The median age for train set: ",data1['Age'].median())
print("The median age for test set: ",data_test['Age'].median())
print("-"*10)
print("The mode of embarked for train set: ",data1['Embarked'].mode()[0])
print("The mode of embarked for test set",data_test['Embarked'].mode()[0])
print("-"*10)
print("The median Fare for train set: ",data1['Fare'].median())
print("The median Fare for test set: ",data_test['Fare'].median())


# In[ ]:


###COMPLETING: complete or delete missing values in train and test/validation dataset --> dataclenar" a yani
for dataset in data_cleaner: # data_cleaner = [data1 , data_test] demiştir for ile bu iki veri setine sırayla ulasıyoruz
                            # Bunun amacı hem test hem de train veri setine aynı anda işlemler yapmaktır.
        
    #median ile doldurma 
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True) 
    # yani test setinin içindeki age değişkenin medyanı ile test setindeki age i doldurdum..
    # ve train setinin içindeki age değişkenin medyanı ile train setindeki age i doldurdum

    #mod ile doldurma
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #median ile doldurma 
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)


# In[ ]:


#delete the cabin feature/column and others previously stated to exclude in train dataset
#  "PassengerId" ve  "Ticket" ile bağımlı değişken ilişkisi yok bunlar sadece o gözlemi temsil eden değerler
# "Cabin" değişkeni ise içinde bir çok null değeri taşıdıgı için silinmesinde problem yoktur.
drop_column = ['PassengerId','Cabin', 'Ticket'] # silecenek olanları bir liste haline getirdim.
data1.drop(drop_column, axis=1, inplace = True) # drop ile train veri setinden sildim


# In[ ]:


# Neden sadece train veri setinde sildik bunu bul !


# In[ ]:


print(data1.isnull().sum())
print("-"*10)
print(data_test.isnull().sum())


# In[ ]:


###CREATE: Feature Engineering for train and test/validation dataset
for dataset in data_cleaner:    
    #Discrete variables
    
    # Ailesi ile gemide olanları belirtmek için bir değişken olusturdum
    # burada 2 değişkeni birleştirip tek değişken olusturdum --> "SibSp" + "Parch" =  "FamilySize"
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1 # +1 amacı eğer diğerleri 0 ise..
                                                                     # yani kişi sayısı 1 ise "FamilySize" = 1 olmalı
    
    # Yanlız kişiler için yeni bir değişken olusturdum
    dataset['IsAlone'] = 1 #  ve hepsine ilk olarak hepsine 1 koydum
    
    # Burada ise tüm sütünları 1 olan IsAlone değişkenimi düzeltiyorum
    # Diyorum ki ;
    # dataset['IsAlone'] değişkeninden (dikkat !!!) 
    # dataset['FamilySize'] > 1 olanları yani tek başıan olmayanları "loc" ile al ve bunlara 0 değerini koy !
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

    #quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    # ilk olarak veri setimdeki "Name" değişkenini "str" fonksiyonu string e çevirdim --> dataset['Name'].str
    # ardından bu "Name" i split ile virgülden itibaren ayırdım --> dataset['Name'].str.split(", ", expand=True) (
    #  Mr.James kısımlarına eriştim --> dataset['Name'].str.split(", ", expand=True)[1] --> burada dönen değer object dir
    # objecti str yapıp tekrar split ettim ve Mr James olarak bölmüş oldum -->  dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)
    # Mr kısmını aldım --> dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    # ... ve veri setimde yeni "Title" değişkeni olusturup bunun içine koydum -->  dataset['Title']

    #detaylı görmek için alt kısımlara bak)


    #Continuous variable   bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
    #Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    
    #Burada Fare değişkenimi yani Yolcu ücretini temsil eden değişkenimi qcut fonksiyonu ile 4 intervale ayırdım
    #  [(-0.001, 7.896] < (7.896, 14.454] < (14.454, 31.472] < (31.472, 512.329]]
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    

    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
    #Burada "Age" değişkenini ilk olarak integer a çevirdim --> dataset['Age'].astype(int) 
    # Burada int e çevirme sadece bu kısımda geçerli olur "Age" değişkenini asıl tipi değişmez
                
    # Ardından cut fonksiyonu ile 5 aralığa böldüm
    # cut fonksiyonu -->
    #"Age" sürekli değişkenini 5 sınıflı kategorik değişkene çevirecem
    # Değerleri  discrete interval lara ayırdım

    # Age          714 non-null    float64
    # Age değişken i float idi bunu integer yaptım

    #Neden birinde qcut birinde cut kullandık
    # qcut --> belli sıklığa göre ayırır
    # cut --> tüm age değerlerin bakar ve ilk ile son değer arasında verdiğin aralığa ayırır
    # Cevap !
    # I choose qcut or "frequency bins" for fare because..
    #...I wanted to break into even quartiles representing low, mid, high, and upper class. 
    # and
    #  I choose cut or "value bins" for age..
    #...because I wanted to break it into nice age ranges for child, young adult, middle-age adult, and elderly adult


    
#cleanup rare title names (Nadir olan ünvanları temizliyorum, bu kısım zaruri değil biraz keyfii)
#print(data1['Title'].value_counts()) ve  print(dataset['Title'].value_counts()), hangi ünvandan kaç tane olduguna bakabilirim.
stat_min = 10 
title_names = (data1['Title'].value_counts() < stat_min) #this will create a true false series with title name as index
# title_names içinde bunlar var ;
#Mrs             False
#Master          False
#Dr               True

# burada veri setimdeki "Title" değişkenini yukarıdaki değerlere göre tekrar tanımlayacağım.
# veri setime apply fonksiyonu ile içine uygulamak istediğim fonksiyonu yazıyorum 
# title_names değişkenimdeki False değerlere "Misc" yazıcam
# loc --> bir veri kümesinden veri değerlerini kolayca almamıza yardımcı olur
# Bunun esas amacı dummy değişkeni kullanınca sadece 1 2 tane olan title lar için yeni değişken oluşmamasını sağlamak
# Yani 10 dan az olan title ları bir değişkende belirticem ve dummy olusturdugumda fazladan değişken olmamıs olacak !
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(data1['Title'].value_counts())
print("-"*10)


#preview data again
data1.sample(10)


# In[ ]:


data1["IsAlone"]


# In[ ]:


dataset['Name'].str.split(", ")


# In[ ]:


#  expand=True ne işe yarar dersen
dataset['Name'].str.split(", ", expand=True)


# In[ ]:


dataset['Name'].str.split(", ", expand=True)[1]
# bu kısma eriştim ve bunun dtypei object dikkat !!


# In[ ]:


dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)


# In[ ]:


dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


# In[ ]:


data1.head()


# In[ ]:


dataset.head()


# In[ ]:


dataset['FareBin'] 


# In[ ]:


data1['AgeBin'] 


# In[ ]:


dataset['FareBin'].value_counts()


# In[ ]:


dataset['AgeBin'].value_counts()


# In[ ]:


data1.head()


# In[ ]:


dataset.dtypes


# In[ ]:


print(dataset['Title'].value_counts())


# In[ ]:


title_names


# In[ ]:


title_names.count()


# ### >> 3.23 Convert Formats

# In[ ]:


# Our categorical datamız bize "object" olarak verilmiş ki bunu math islemlerini zorlaştırır
#..Bu veri setinde object veri tiplerini Label Encoder kullanarak categorical dummy değişkenlerine dönüştürücem.
# Hem train seti hem test seti için yapıyorum
# Anladıgım kadarı ile 3 farklı sekilde dönüştürücem ve bunları deneyeceğim.


# In[ ]:


#code categorical data
label = LabelEncoder() # bir dönüştürücü nesnesi oluşturdum

# label dönüştürcümü hem train hem test setine uyguluyorum

for dataset in data_cleaner: # data_cleaner = [data1 , data_test]
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])


# In[ ]:


#bağımlı y değişkenimi aldım
Target = ['Survived']

# bağımsız x değişkenlerini aldım

# bunlar orjinal veri setimdeki bağımsız değişkenlerimi listeye koydum 
# get_dummies fonksiyonuna veri setimi ve içine bu listeyi koyucam ve  o çevirmesi gereklenleri dummye çeviricek
data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']  

#coded for algorithm calculation
data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] 


# In[ ]:


data1_xy =  Target + data1_x
print('Original X Y: ', data1_xy, '\n')
# veri setim bu


# In[ ]:


#define x variables for original w/bin features to remove continuous variables
data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')


# In[ ]:


#define x and y variables for dummy features original
# pandas.get_dummies() is used for data manipulation. It converts categorical data into dummy or indicator variables.

data1_dummy = pd.get_dummies(data1[data1_x])
#burası dummy kısmından bağımsız keyfi ypaılan iş 
# Burada dummy değişkenindne sonra değişen değişkenlerin(misal "Sex") hala veri setimde olup olmadıgını checck ettim
data1_x_dummy = data1_dummy.columns.tolist() #yukardaki veri setinin columns isimlerini aldım ve listeye çevirdim
data1_xy_dummy = Target + data1_x_dummy # dummy tanımladıktan sonraki değişkenlerim bunlar
print('Dummy X Y: ', data1_xy_dummy, '\n')


# In[ ]:


data1_x


# In[ ]:


data1[data1_x] # bunları dummy değişkenine döndürücem (bir alttakine bak)


# In[ ]:


data1_dummy.head()


# In[ ]:


data1_x_dummy


# ### >> 3.24 Da-Double Check Cleaned Data

# In[ ]:


print('Train columns with null values: \n', data1.isnull().sum())
print("-"*10)
print (data1.info())
print("-"*10)

print('Test/Validation columns with null values: \n', data_test.isnull().sum())
print("-"*10)
print (data_test.info())
print("-"*10)

data_train.describe(include = 'all')


# ### >>3.25 Split Training and Testing Data

# In[ ]:


#random_state -> seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
# data1[data1_x_calc],data1[data1_x_bin] ve data1_dummy[data1_x_dummy] kısımlarına altta bak
# train_test_split (X,y,random_state = 0) konur (bağımsız değişkenler,bağımlı değişkenler)
#Neden 3 kere yaptık 
# Cünkü değişkenleri 3 farklı sekilde convert ettim (dönüştürdüm).Bu 3 farklı dönüştürme için deneme yapıcağım (?)

train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target] , random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)


print("Data1 Shape: {}".format(data1.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))
train1_x_dummy.head() # örnek ıcın bir tanesine baktım


# In[ ]:


data1


# In[ ]:


data1[data1_x_calc] # categorik olanlar label + sayısal olanlar normal


# In[ ]:


print(data1_x_calc)


# In[ ]:


data1[data1_x_bin] # hem kategorik olanlar label code hem sayısal olanlar label (sayısal değişkenler için bin olanı kullandım)


# In[ ]:


print(data1_x_bin)


# In[ ]:


data1_dummy[data1_x_dummy] #  dummy


# In[ ]:


data1_x_bin


# In[ ]:


print(data1_x_dummy)


# ## Step 4: Perform Exploratory Analysis with Statistics

# In[ ]:


# İstatistikler ile Keşifçi Veri Analizi


# In[ ]:


# Artık verimiz temiz artık bu veriyi keşefedelim
# Değişkenlerimizi tanımak ve özetlemek için  grafiksel istatiksiksel ile verimizi keşfedicez.
# Değişkenlerin ; hedef y değişkeni ile olan ilişkilerini ve birbirleri ile olan ilişkilerini belirleyeceğiz.


# In[ ]:


data1_x #aşağıda for ile bunun ıcınde gezicem


# In[ ]:


# Aşağıdaki for döngüsü içindekinin neyi temsil ettiğini anlamak için;
print("Age:",data1["Age"].dtype)
print("Sex:",data1["Sex"].dtype)
print("Embarked:",data1["Embarked"].dtype)
print("Title:",data1["Title"].dtype)
print("IsAlone:",data1["IsAlone"].dtype)
print("Fare:",data1["Fare"].dtype)


# In[ ]:


data1[["Sex", Target[0]]]
# "İlk olarak veri setimden "bağımsız değişkenimi "Sex" ve bağımlı değişkenimi seçtim 


# In[ ]:


data1[["Sex", Target[0]]].groupby("Sex", as_index=False).mean()
# "İlk olarak veri setimden "bağımsız değişkenimi "Sex" ve bağımlı değişkenimi seçtim 
# ardından "groupby" ile "Sex" değişkeine göre grupladım
# ve grupladıklarımın ortalamasını aldım .


# In[ ]:


#Discrete Variable Correlation by Survival using group by aka pivot table
# Group by kullnarak discrete değişkenlerin ; bağımlı y değişkeni "Survived" ile olan ilişkileri ;
# For döngüsü ile bağımısz değişkenimi içeren değişken içinde geziniorum

for x in data1_x: # bütün bağımsız değişkenler burada -> data1_x 
    if data1[x].dtype != 'float64' : #yukarıdak bu kısmın neyi temsil ettiğine bakabilirsin.
        print('Survival Correlation by:', x)
        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')


# In[ ]:


# Burada bir çaprazlama tablosu oluşturuyorum.
# Burada frekans sayıyor diyebiliriz.
# Pandas ın içinde crosstab fonksiyonunu aldım --> pd.crosstab()
# data1[Target[0]] diyerek direkt bağımlı değişkenin değerini alıyoruz --> 0-1 değerini yani
#pd.crosstab(bağımsız değişken , bağımlı değişken) diyebiliriz.

print(pd.crosstab(data1['Title'],data1[Target[0]]))


# In[ ]:


data1[Target[0]]


# In[ ]:


#graph distribution of quantitative data
# Sayısal verilerin grafikleri;

plt.figure(figsize=[16,12]) #aşağıdaki tabloların büyüklüğü ile ilgili ayarlama


# subplot(x,y,z) : Grafiklerin düzlemini ve kaçıncı grafik olduğunu belirtir. 
#İlk sayı satırı, ikinci sayı sütunu, üçüncü sayı ise kaçıncı grafik olduğunu ifade eder
# buradaki subplot aşağıdaki çıktının nasıl görüneceğine karar verir 
        # (231) --> 2 satır olsun 3 sütün olsun ve bu tablo 1. sırada olsun gibi
        # (322) --> 3 satır olsun 2 sütün olsun ve bu tablo 2. sırada olsun gibi


# x ekseninze veri setimizin içinden istediğim bağımsız değişkeni seçiyorum
plt.subplot(231) #  plt.subplot(2,3,1) ile aynı sonucu verir.
plt.boxplot(x=data1['Fare'], showmeans = True, meanline = True)
                                # "meanline" ile kutunun içindeki tırtıklı çiziyi temsil eder.
plt.title('Fare Boxplot') # boxplot adını koydum
plt.ylabel('Fare ($)') # y eksenini adı (üst kısım)

plt.subplot(232)
plt.boxplot(data1['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

plt.subplot(233)
plt.boxplot(data1['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')



# Histogramlar ;
# Histogram ve yoğunluk grafikleri sayısal değişkenlerin dağılımını ifade etmek için kullanılan veri görselleştirme teknikleridir
# Elimizdeki sayısal değişkenin değerlerini belirli aralıklara bölerek o aralıklardaki gözlem ve frekanslarını yansıtır

# label = ['Survived','Dead'] --> Hisrogramın sağ üst kısmındaki isimleridr
#  color = ['g','r'] --> Renkleri belirttim
# stacked=True, --> Bunun sayesinde 2 renk de üst üste durur ; eğer bunu koymazsan 2 renk yan yana olur

# Bu kısmı daha iyi anlamak için 3 basamak aşağı bak

# "Fare" değişkeni için;
plt.subplot(234)
plt.hist(x = [data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
            # buradaki x e 2 tane değişken verdik. --> kurtulanların fare'i ve kurtulmayanların fare'i
            # x = [] dedim ;
            # data1[data1['Survived']==1]['Fare'] --> Kurulanların fare'i --> data1[]['Fare']
            # data1[data1['Survived']==0]['Fare'] --> kurtulmayanların fare'i
        
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()


# "Age" değişkeni için

plt.subplot(235)
plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()


# "'Family Size" değişkeni için

plt.subplot(236) # plt.subplot(2,3,1) ile aynı sonuc verir.
plt.hist(x = [data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend();


# In[ ]:


# yukarıdakinin aynısı sadece "subplot" kısmının ne işe yaradıgını daha net görmek için
plt.figure(figsize=[16,12])


plt.subplot(321) #  plt.subplot(2,3,1) ile aynı sonucu verir.
plt.boxplot(x=data1['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(322)
plt.boxplot(data1['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

plt.subplot(323)
plt.boxplot(data1['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')

plt.subplot(324)
plt.hist(x = [data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(325)
plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(326) # plt.subplot(2,3,1) ile aynı sonuc verir.
plt.hist(x = [data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend();


# In[ ]:


plt.subplot(235)
plt.hist(x = [data1[data1['Survived']==1]['Age']])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()


# In[ ]:


# Çok değişkenli karşılaştırma seaborn kullanarak yapıcam.

# "survived" bağımlı değişkenine göre bireysel değişkenleri yazıcam

# subplots ile aşağıdaki grafiğin nasıl görüneceğini ayarladım
# 2 tane satır ; 3 sütüna bunları sığdır ; boyutunu ayarladım -> (figsize=(16,12))
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

# "barplot" çiziyorum 

# x ekseninde bağımsız değişkenler
# y ekseninde bağımlı değişkenler
# "data" argümanına barplot çizeceğim verimi koyuyorum
# "ax" argümanına yukarıda tanımladıgım "saxis" değişkenini koydum ve o grafiğin konumunu belirledim
# ax = saxis[0,0] --> x için 0. idex ; y için 0. index
# "order" argümanına plotların nasıl sıralacagını veriyorum. --> order=[2,1,3]

sns.barplot(x = 'Embarked', y = 'Survived', data=data1, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[2,1,3], data=data1, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data1, ax = saxis[0,2]);


# pointplot çiziyorum

sns.pointplot(x = 'FareBin', y = 'Survived',  data=data1, ax = saxis[1,0])
sns.pointplot(x = 'AgeBin', y = 'Survived',  data=data1, ax = saxis[1,1])
sns.pointplot(x = 'FamilySize', y = 'Survived', data=data1, ax = saxis[1,2]);


# In[ ]:


# Pclass değişkeni için grafik içiyorum
# Pclass değişkeninin "survived" üzeride etkili oldugu biliyoruz 
# Burada Pclass özelliği + Survived özelliğinin yanına 2.bir özellik ekleyerek grafiğe döküyorum
# "hue" argümanı ile boyut ekliyorum

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))

sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = data1, ax = axis1)
axis1.set_title('Pclass vs Fare Survival Comparison') # grafiğe isim koyduk


# Hangi sınıfta hangi yaş grubundan insanlar kurtulmuş
sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data1, split = True, ax = axis2)
                        # "split" ile tek bir violin plot olmasını sağlarız.
axis2.set_title('Pclass vs Age Survival Comparison') # grafiğe isim koyduk


sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = data1, ax = axis3)
axis3.set_title('Pclass vs Family Size Survival Comparison') # grafiğe isim koyduk


# In[ ]:


# split argümanı ne işe yarar ?
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))


# Hangi sınıfta hangi yaş grubundan insanlar kurtulmuş
sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data1, split = True, ax = axis2)
axis2.set_title('Pclass vs Age Survival Comparison') # grafiğe isim koyduk

sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data1 ,ax = axis3)
axis3.set_title('Pclass vs Age Survival Comparison') # grafiğe isim koyduk


# In[ ]:


# "Sex" değişkeni için grafik içiyorum
# "Sex" değişkeninin "survived" üzeride etkili oldugu biliyoruz 
# Burada Sex özelliği + Survived özelliğinin yanına 2.bir özellik ekleyerek grafiğe döküyorum
# "hue" argümanı ile boyut ekliyorum

fig, qaxis = plt.subplots(1,3,figsize=(14,12)) # 1 satır ; 3 sütün


#grafiğin aşağıdaki cıktıda duracağı konumu ayarlıyor --> , ax = qaxis[0]

# set_title ile grafiğie baslık koyabiliriz.

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=data1, ax = qaxis[0]).set_title('Sex vs Embarked Survival Comparison')
#axis1.set_title('Sex vs Embarked Survival Comparison') --< böyle de olmalı aslında (?)

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=data1, ax  = qaxis[1]).set_title('Sex vs Pclass Survival Comparison')
#axis1.set_title('Sex vs Pclass Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=data1, ax  = qaxis[2]).set_title('Sex vs IsAlone Survival Comparison')
#axis1.set_title('Sex vs IsAlone Survival Comparison')


# In[ ]:


# Karılaştırmalara devam ediyorum.
fig, (maxis1, maxis2) = plt.subplots(1, 2,figsize=(14,12))

#how does family size factor with sex & survival compare
# Family size değişlenini sex  ve survived ile karsılastırıyorum
# palette = {} içine hangi cinsiyet hangi renk olsun diyorum
# bunun içine sanırım hue argümaındaki değişkene göre yazıyorum(male ve famele için)
# markers --> male ve famele için farklı iki işaretleme 
# linestyles --> çizgileri karıstırmamak için 2 farklı line tipi veriyorum.
sns.pointplot(x="FamilySize", y="Survived", hue="Sex", data=data1,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis1)

#how does class factor with sex & survival compare
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data1,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)


# In[ ]:


# FacetGrid. Grafik üzerine eklenen boyutları bölerek göstermek için kullanılır
e = sns.FacetGrid(data1, col = 'Embarked') # diyerek 3 tane grafik olusturdu


# In[ ]:


# "Embarked" değişkeni ; pclass ve sex değişkeni ile beraber y bağımlı değişkenine yaptığı etkiye bakıyorum
# facetgrid in kullanılışı diğerlerinden farklıdır.
# FacetGrid. Grafik üzerine eklenen boyutları bölerek göstermek için kullanılır

e = sns.FacetGrid(data1, col = 'Embarked')
# map fonksiyonu ile içine koydugumuz değeri içindekine uygular ye uygular.
# map(fun, iter)
# map() içine sns.poinplot koydum -> map fosnkiyonu buna uygulayacak yani ilk olarak
    # 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep' --> bunları "sns.pointplot" a uygulayacak
    # ortaya cıkanları da "e" değişkenine uygulacak

e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep')
    
e.add_legend()


# In[ ]:


# Bağımlı y "Survived" değişkenin yaşlara göre dağılımı
# yani hayatta kalan veya hayatta kalmayan yolcuların yaş dağılımlarını
a = sns.FacetGrid( data1, hue = 'Survived', aspect=4 ,xlim=(0 , data1['Age'].max()) )
    # ilk argüman olarak veri setimizi ; bağımlı değişkeni koyarız
    # aspect argümanı grafiğin x ekseninde ne kadar yayılacağını belirliyor..
        #...aralıklar aynı amaa görüntü olarak değiştiriyor
    # Olusturdugumuz grafiğin x ve y eksenini ayarlayabiliriz    
        # xlim --> x eksenini ayarladım..
            #.. 0 ile yaş değişkeninin en büyük değeri arasında olsun dedim.
            
#map fonksiyonu ile ilk olarak fonksiyon koyarsın ve ardından elimizdeki bir datanın elemanlarını koyarız..
#..ve bize bir obje döndürür
#Burada biz seaborn un kdeplot fonksiyonu kullanarak Age e göre kdeplot grafiği oluşturucaz            
#..ve bunu "Survived" bağımlı değişkenini sınıflarını (1 ve 0 ) hue arügmanı ile boyut olarak ekleyecez...
#...ve ile "map" ile eşleyecez
#yani "Survived" ile "Age" ın kdeplot unu eşleyecez bunu da map fonksiypnu ile yapıyoruz
a.map(sns.kdeplot, 'Age', shade= True )
        # shade=True diyerek de her bir grafiğin altındaki alanı boyuyacak
a.add_legend()


# In[ ]:


# Yukarıdakinin aynısı sadece "set" kısmı farklı 
# alternatif olarak böyle de kullanılır

a = sns.FacetGrid( data1, hue = 'Survived', aspect=4 )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0 , data1['Age'].max()))
a.add_legend()


# In[ ]:


# "Survived" bağımlı değişkenine göre "sex", "pclass" ve "Age" in histogram karşılaştırması
h = sns.FacetGrid(data1, row = 'Sex', col = 'Pclass', hue = 'Survived')
h.add_legend()


# In[ ]:


h = sns.FacetGrid(data1, row = 'Sex', col = 'Pclass', hue = 'Survived')
    # row = "Sex" dersen 2 tane satır oluşur cünkü 2 cinsiyet var
    # col = "Pclass" dersen 3 tane column olusur cunkü 3 tane class var
    # row ve col kısmını x ve y ekseni olarak DÜŞÜNMEMELİSİN !
    
#map fonksiyonu ile ilk olarak fonksiyon koyarsın ve ardından elimizdeki bir datanın elemanlarını koyarız..    
h.map(plt.hist, 'Age', alpha = .75)
# alpha değişkeni saydamlıga yarar (alt kısımda denedim bak)
h.add_legend()

# alttaki cıktıya dikkat
# üstteki grafiklirin altında da "age" yazmalı karıstırma !
# Grafik şunu diyor
# Birini için bakalım
# Cinsiyeti "male" olan ve "Pclass" =1 olanlar için;
#Yani 1.sınıftaki yolculuk eden erkeklerin "Survived" bağımlı değişkenine göre yaş cağılımı
#Yani 1.sınıftaki yolculuk eden erkeklerin hangi yaştan kaç tanesi kurtulmuş bunu gösteriyor.


# In[ ]:


# "alpha" argümanını deniyorum.
h = sns.FacetGrid(data1, row = 'Sex', col = 'Pclass', hue = 'Survived')
h.map(plt.hist, 'Age', alpha = 0.1)
h.add_legend()


# In[ ]:


# tüm veri setinin pairplot u
# daha professiyonel gösterimi
pp = sns.pairplot(data1, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
pp.set(xticklabels=[])


# In[ ]:


# böyle de aynı cıktıyı alabilirsin.
sns.pairplot(data1, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )


# In[ ]:


#correlation heatmap of dataset
# Veri setinin korelasyon ısı haritası
#Heat map ;Elimizdeki değişkenleri biraz daha yapısal anlamda...
#...daha geniş persfektiften görmek istedğimizde kullanabileceğimiz grafik görselleştirme teknikleriden birisidir
# Sadece zamansal bağlamda değil,...
#eğer elimizde cok sınıflı kategorik değişken ve bunu belirli bi sayısal değişken acısından görselleştirme ihtiyacımız varsa da kullanabiliriz

def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, #sağ kısımdaki renk scalası
        ax=ax,
        annot=True, #ısı hartiasının her hücresini sayılar ile doldurabilir
        linewidths=0.1,vmax=1.0, linecolor='white', # kutular arasında çizgi çeker
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(data1)


# ## STEP 5 : Model Data

# In[ ]:


# Bu kısımda ne yaptıgımızdan ziyade neden yaptığımız önemlidir.
# Machiene Learing in genel amacı bizim problemlerimizi(human problems) çözmektir.
# Machiene Learning 3 kısıma ayrılır; supervised learning, unsupervised learning, and reinforced learning.
# Supervised learning is where you train the model by presenting it a training dataset that includes the correct answer
#(Gözetimli makine öğrenmesinde ; içinde doğru cevaplar olan train setimizi modele gösteririrz ve modeli eğitiriz)
#(Modele elmaların elma oldugunu söyleyerek modele veririz ve bu onun elma oldugunu öğrenir;yeni elma gelince bu elma der)
#-
#Unsupervised learning is where you train the model using a training dataset that does not include the correct answer. 
# (Gözetimsiz makine öğrenmesinde; içinde doğru cevaplar olmadan train setimizi modele gösterir ve eğitiriz)
#(modele bu elma demeden elmalar veririz ve o bunun bir elma oldugunu anlar; yeni bir elma gelince bu elma der)
#-
#reinforced learning is a hybrid of the previous two, where the model is not given the correct answer immediately, but later after a sequence of events to reinforce learning.
# Modele doğru cevap hemen verilmez, birkaç işlem sonrasında modele doğru cevap verilir ;  Supervised learning ve Unsupervised learning karışımıdır
#-------------------
#Biz burada ise Supervised learning algoritmaları kullanmalıyız.Çünkü elimizde featrues ve bunlara bağlı y bağımlı değişkenimiz var.Bunları modele sunarak eğitmeye calıyoruz.
#Ardından aynı features lara sahip olan test setimizi algoritmaya verip tahmin yapacağız.
# Burada hangisini kullanıcaz ?
# Supervised learning algoritmaları kullanıcaz --> Sınıflandırma(classification) ve Regression --> Bu  kısımdan birini kullanıcaz
# Bizim problemimiz bir yolcunun kurtulup kurtulmadığını tahmin etmektir yani bağımlı hedef değişkenimiz discrete  variable.
# Bundan dolayı biz classification algoritms kullanıcaz --> sklearn library
# Machine Learning Classification Algorithms: 
#-Ensemble Methods
#-Generalized Linear Models (GLM)
#-Naive Bayes
#-Nearest Neighbors
#-Support Vector Machines (SVM)
#-Decision Trees
#-Discriminant Analysis
#Bunları kullanıcaz.
# KISA NOT : (Logistic Regresyon her ne kadar regresyon desede aslında bir classification işlemi yapar yani  classification algorithm  diyebiliriz.)


# In[ ]:


#Machine Learning Algorithm (MLA) Selection and Initialization

# Bir liste oluşturdum ve model nesnelerimi oluşturuyorum
MLA = [
    #Ensemble Methods (Topluluk Öğrenmesi)
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost
    XGBClassifier()    
    ]
#MLA listesinde modellerim var


#note: ShuffleSplit is an alternative to train_test_split

#(alternatif)
#cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) 
    # run model 10x with 60/30 split intentionally leaving out 10%

# test-train ayrımı yapıyorum
X_train, X_test, y_train, y_test = model_selection.train_test_split(data1[data1_x_bin], data1[Target], test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%


# Makine öğrenmesi modellerimi karışlaştırıcam bunun için  yeni bir tane veri seti olusturucam

# ilk olarak Kolon isimleri oluşturuyorum
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']

# İkinci olarak bir veri seti oluşturuyorum
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = data1[Target]

#index through MLA and save performance to table
row_index = 0
for alg in MLA: #yukarıda tanımladıgım MLA listesi içinde for ile geziniyorum.

    
    #set name and parameters
    MLA_name = alg.__class__.__name__ #bütün modellerin isimlerini alıcam XGBClassifier. __class__ . __name__ --> XGBClassifier gibi hepsi gelecek
    
    # Yukarıda "MLA_name" değişkenine koydugum isimleri modelleri karılaştırmak için oluşturdugum "MLA_compare" veri setime koyuyorum
    
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name # veri setime loc ile işlem yapıcam --> [ekleyeceğim index , ekleceğim sütün adı]
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params()) # o algortimanın parametrelerini koyyurum
    
    
    
    # Cross validation part
    # cross validation yaparken sırasyıla bu argümanlar verilir (modeli koy ; x_train ; y _train,cv = )
  
    
    cv_results = model_selection.cross_validate(alg, data1[data1_x_bin], data1[Target], cv = 10, return_train_score=True)
                                 # return_train_score=True koymazsan hata alırsın
        
    # cross validation işleminden cıkan sonucları veri setime loc ile koyuyorum
    
    # fit_time --> her bir cross validation için estimator u train setine yerleştirme süresi (10 katlı cv için düşün.)
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    
    
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    #yani istatistikte std +-3 sağ soldan vardı ya o kısımı burası
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    

    #save MLA predictions - see section 6 for usage
    alg.fit(data1[data1_x_bin], data1[Target]) # modeli fit ediyorum --> modelim.fit (x_train,y_train) gibi düşün
    MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin]) # modelim ile tahmin yapıyorum fit(x_train)
    
    
    row_index+=1 # indexi bir arttırıp diğerine geciyorum
    
MLA_compare    


# In[ ]:


# Modelerimi en iyiden en kötüye srıalıyorum
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare


# In[ ]:


# Modelimin test doğruluk ortalamalarını grafikte gösterdim
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')

plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')


# In[ ]:


last_model = MLA[14]


# In[ ]:


last_model


# In[ ]:


#Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]

svr_params ={ #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [1,2,3,4,5], #default=1.0
                'gamma': grid_ratio, #edfault: auto
                'decision_function_shape': ['ovo', 'ovr'], #default:ovr
                'probability': [True],
                'random_state': grid_seed }


# In[ ]:


best_search = model_selection.GridSearchCV(estimator = last_model, param_grid = svr_params, cv = 10, scoring = 'roc_auc')


# In[ ]:


best_search.fit(data1[data1_x_bin], data1[Target])


# In[ ]:


best_search.best_params_


# In[ ]:


pd.Series(best_search.best_params_)[1]


# In[ ]:


last_model_tune =  svm.SVC(kernel = 'linear',
                        C = pd.Series(best_search.best_params_)[0] , #default=1.0
                        decision_function_shape = pd.Series(best_search.best_params_)[1] , #default:ovr
                        gamma = pd.Series(best_search.best_params_)[2] , #edfault: auto
                        probability = pd.Series(best_search.best_params_)[3] ,
                        random_state = pd.Series(best_search.best_params_)[4]  )


# In[ ]:


last_model_tune.fit(data1[data1_x_bin], data1[Target])


# In[ ]:


data_test["Survived"] = last_model_tune.predict(data_test[data1_x_bin])
# test testi için tahmin değerleri data_test["Survived"] oluyor


# In[ ]:


data_test["Survived"]


# In[ ]:


submit = data_test[['PassengerId','Survived']]
submit.to_csv("/kaggle/working/submission.csv", index=False)

print('Validation Data Distribution: \n', data_test['Survived'].value_counts(normalize = True))
submit.sample(10)


# In[ ]:


data_test.head()


# In[ ]:





# In[ ]:




