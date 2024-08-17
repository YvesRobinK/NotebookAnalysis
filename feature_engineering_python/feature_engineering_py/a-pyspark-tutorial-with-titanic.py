#!/usr/bin/env python
# coding: utf-8

# <img src="http://data.freehdw.com/ships-titanic-vehicles-best.jpg"  Width="800">
# 
# 
# 
# 
# This kernel will give a tutorial for starting out with PySpark using Titanic dataset. Let's get started. 
# 
# 
# ### Kernel Goals
# <a id="aboutthiskernel"></a>
# ***
# There are three primary goals of this kernel.
# - <b>Provide a tutorial for someone who is starting out with pyspark.
# - <b>Do an exploratory data analysis(EDA)</b> of titanic with visualizations and storytelling.  
# - <b>Predict</b>: Use machine learning classification models to predict the chances of passengers survival.
# 
# ### What is Spark, anyway?
# Spark is a platform for cluster computing. Spark lets us spread data and computations over clusters with multiple nodes (think of each node as a separate computer). Splitting up data makes it easier to work with very large datasets because each node only works with a small amount of data.
# As each node works on its own subset of the total data, it also carries out a part of the total calculations required, so that both data processing and computation are performed in parallel over the nodes in the cluster. It is a fact that parallel computation can make certain types of programming tasks much faster.
# 
# Deciding whether or not Spark is the best solution for your problem takes some experience, but you can consider questions like:
# * Is my data too big to work with on a single machine?
# * Can my calculations be easily parallelized?
# 
# 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


## installing pyspark
get_ipython().system('pip install pyspark')


# In[3]:


## installing pyarrow
get_ipython().system('pip install pyarrow')


# The first step in using Spark is connecting to a cluster. In practice, the cluster will be hosted on a remote machine that's connected to all other nodes. There will be one computer, called the master that manages splitting up the data and the computations. The master is connected to the rest of the computers in the cluster, which are called worker. The master sends the workers data and calculations to run, and they send their results back to the master.
# 
# We definitely don't need may clusters for Titanic dataset. In addition to that, the syntax for running locally or using many clusters are pretty similar. To start working with Spark DataFrames, we first have to create a SparkSession object from SparkContext. We can think of the SparkContext as the connection to the cluster and SparkSession as the interface with that connection. Let's create a SparkSession. 

# # Beginner Tutorial
# This part is solely for beginners. I recommend starting from here to get a good understanding of the flow. 

# In[4]:


## creating a spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('tutorial').getOrCreate()


# Let's read the dataset. 

# In[5]:


df_train = spark.read.csv('../input/titanic/train.csv', header = True, inferSchema=True)
df_test = spark.read.csv('../input/titanic/test.csv', header = True, inferSchema=True)


# In[6]:


titanic_train = df_train.alias("titanic_train")


# In[7]:


## So, what is df_train?
type(df_train)


# In[8]:


## As you can see it's a Spark dataframe. Let's take a look at the preview of the dataset. 
df_train.show()


# In[9]:


## It looks a bit messi. See what I did there? ;). Anyway, how about using .toPandas() for change. 
df_train.toPandas()


# In[10]:


# I use the toPandas() in a riddiculous amount as you will see in this kernel. 
# It is just convenient and doesn't put a lot of constran in my eye. 
## in addition to that if you know pandas, this can be very helpful 
## for checking your work.
## how about a summary. 
result = df_train.describe().toPandas()


# In[11]:


result


# In[12]:


# getting the total row count
df_train.count()


# In[13]:


# We can also convert a pandas dataframe to spark dataframe. Here is how we do it. 
print(f"Before: {type(result)}")
spark_temp = spark.createDataFrame(result)
print(f"After: {type(spark_temp)}")


# In[14]:


# Cool, Let's print the schema of the df using .printSchema()
df_train.printSchema()


# In[15]:


# similar approach
df_train.dtypes


# The data in the real world is not this clean. We often have to create our own schema and implement it. We will describe more about it in the future. Since we are talking about schema, are you wondering if you would be able to implement sql with Spark?. Yes, you can. 
# 
# One of the best advantage of Spark is that you can run sql commands to do analysis. If you are like that nifty co-worker of mine, you would probably want to use sql with spark. Let's do an example. 

# In[16]:


## First, we need to register a sql temporary view.
df_train.createOrReplaceTempView("mytable");

## Then, we use spark.sql and write sql inside it, which returns a spark Dataframe.  
result = spark.sql("SELECT * FROM mytable ORDER BY Fare DESC LIMIT 10")
result.toPandas()


# Similarly we can also register another sql temp view. 

# In[17]:


df_test.createOrReplaceTempView("df_test")


# Now that we have registered two tables with in this spark session, wondering how we can see which once are registered?

# In[18]:


spark.catalog.listTables()


# In[19]:


# similarly
spark.sql("SHOW views").show()


# In[20]:


# We can also create spark dataframe out of these tables using spark.table
temp_table = spark.table("df_test")
print(type(temp_table))
temp_table.show(5)


# In[21]:


# pretty cool, We will dive deep in sql later. 
# Let's go back to dataFrame and do some nitty-gritty stuff. 
# What if want the column names only. 
df_train.columns


# In[22]:


# What about just a column?
df_train['Age']


# In[23]:


df_train.Age


# In[24]:


type(df_train['Age'])


# In[25]:


# Well, that's not what we pandas users have expected. 
# Yes, in order to get a column we need to use select().  
# df.select(df['Age']).show()
df_train.select('Age').show()


# In[26]:


## What if we want multiple columns?
df_train.select(['Age', 'Fare']).show()


# In[27]:


# similarly 
df_train[['Age', 'Fare']].show()


# In[28]:


# or 
df_train[df_train.Age, 
         df_train.Fare].show()


# As you can see pyspark dataframe syntax is pretty simple with a lot of ways to implement. Which syntex is best implemented depends on what we are trying to accomplish. I will discuss more on this as we go on. Now let's see how we can access a row. 

# In[29]:


df_train.head(1)


# In[30]:


type(df_train.head(1))


# In[31]:


## returns a list. let's get the item in the list
row = df_train.head(1)[0]
row


# In[32]:


type(row)


# In[33]:


## row can be converted into dict using .asDict()
row.asDict()


# In[34]:


## Then the value can be accessed from the row dictionaly. 
row.asDict()['PassengerId']


# In[35]:


## similarly
row.asDict()['Name']


# In[36]:


## let's say we want to change the name of a column. we can use withColumnRenamed
# df.withColumnRenamed('exsisting name', 'anticipated name');
df_train.withColumnRenamed("Age", "newA").limit(5).toPandas()


# In[37]:


# Let's say we want to modify a column, for example, add in this case, adding $20 with every fare. 
## df.withColumn('existing column', 'calculation with the column(we have to put df not just column)')
## so not df.withColumn('Fare', 'Fare' +20).show()
df_train.withColumn('Fare', df_train['Fare']+20).limit(5).show()


# Now this change isn't permanent since we are not assigning it to any variables. 

# In[38]:


## let's say we want to get the average fare.
# we will use the "mean" function from pyspark.sql.functions(this is where all the functions are stored) and
# collect the data using ".collect()" instead of using .show()
# collect returns a list so we need to get the value from the list using index


# In[39]:


from pyspark.sql.functions import mean
fare_mean = df_train.select(mean("Fare")).collect()
fare_mean[0][0]


# In[40]:


fare_mean = fare_mean[0][0]
fare_mean


# #### Filter

# In[41]:


# What if we want to filter data and see all fare above average. 
# there are two approaches of this, we can use sql syntex/passing a string
# or just dataframe approach. 
df_train.filter("Fare > 32.20" ).limit(3).show()


# In[42]:


# similarly 
df_train[df_train.Fare > 32.20].limit(3).show()


# In[43]:


# or we can use the dataframe approach
df_train.filter(df_train['Fare'] > fare_mean).limit(3).show()


# In[44]:


## What if we want to filter by multiple columns.
# passenger with below average fare with a sex equals male
temp_df = df_train.filter((df_train['Fare'] < fare_mean) &
          (df_train['Sex'] ==  'male')
         )
temp_df.show(5)


# In[45]:


# similarly 
df_train[(df_train.Fare < fare_mean) & 
         (df_train.Sex == "male")].show(5)


# In[46]:


# passenger with below average fare and are not male
filter1_less_than_mean_fare = df_train['Fare'] < fare_mean
filter2_sex_not_male = df_train['Sex'] != "male"
df_train.filter((filter1_less_than_mean_fare) &
                (filter2_sex_not_male)).show(10)


# In[47]:


# We can also apply it this way
# passenger with below fare and are not male
# creating filters
filter1_less_than_mean_fare = df_train['Fare'] < fare_mean
filter2_sex_not_male = df_train['Sex'] != "male"
# applying filters
df_train.filter(filter1_less_than_mean_fare).filter(filter2_sex_not_male).show(10)


# In[48]:


# we can also filter by using builtin functions.
# between
df_train.select("PassengerId", "Fare").filter(df_train.Fare.between(10,40)).show()


# In[49]:


df_train.select("PassengerID", df_train.Fare.between(10,40)).show()


# In[50]:


# contains
df_train.select("PassengerId", "Name").filter(df_train.Name.contains("Mr")).show()


# In[51]:


# startswith 
df_train.select("PassengerID", 'Sex').filter(df_train.Sex.startswith("fe")).show()


# In[52]:


# endswith
df_train.select("PassengerID", 'Ticket').filter(df_train.Ticket.endswith("50")).show()


# In[53]:


# isin
df_train[df_train.PassengerId.isin([1,2,3])].show()


# In[54]:


# like
df_train[df_train.Name.like("Br%")].show()


# In[55]:


# substr
df_train.select(df_train.Name.substr(1,5)).show()


# In[56]:


# similarly 
df_train[[df_train.Name.substr(1,5)]].show()


# One interesting thing about substr method is that we can't implement the following syntax while working with substr. This syntax is best implemented in a filter when the return values are boolean not a column.

# In[57]:


# df_train[df_train.Name.substr(1,5)].show()


# #### GroupBy

# In[58]:


## Let's group by Pclass and get the average fare price per Pclass.  
df_train.groupBy("Pclass").mean().toPandas()


# In[59]:


## let's just look at the Pclass and avg(Fare)
df_train.groupBy("Pclass").mean().select('Pclass', 'avg(Fare)').show()


# In[60]:


# Alternative way
df_train.groupBy("Pclass").mean("Fare").show()


# In[61]:


## What if we want just the average of all fare, we can use .agg with the dataframe. 
df_train.agg({'Fare':'mean'}).show()


# In[62]:


## another way this can be done is by importing "mean" funciton from pyspark.sql.functions
from pyspark.sql.functions import mean
df_train.select(mean("Fare")).show()


# In[63]:


## we can also combine the few previous approaches to get similar results. 
temp = df_train.groupBy("Pclass")
temp.agg({"Fare": 'mean'}).show()


# In[64]:


# What if we want to format the results. 
# for example,
# I want to rename the column. this will be accomplished using .alias() method.  
# I want to format the number with only two decimals. this can be done using "format_number"
from pyspark.sql.functions import format_number
temp = df_train.groupBy("Pclass")
temp = temp.agg({"Fare": 'mean'})
temp.select('Pclass', format_number("avg(Fare)", 2).alias("average fare")).show()


# #### OrderBy
# There are many built in functions that we can use to do orderby in spark. Let's look at some of those. 

# In[65]:


## What if I want to order by Fare in ascending order. 
df_train.orderBy("Fare").limit(20).toPandas()


# In[66]:


# similarly
df_train.orderBy(df_train.Fare.asc()).show()


# In[67]:


# What about descending order
# df.orderBy(df['Fare'].desc()).limit(5).show()
# dot notation
df_train.orderBy(df_train.Fare.desc()).limit(5).show()


# In[68]:


df_train.filter(df_train.Embarked.isNull()).count()


# In[69]:


df_train.select('PassengerID','Embarked').orderBy(df_train.Embarked.asc_nulls_first()).show()


# In[70]:


df_train.select('PassengerID','Embarked').orderBy(df_train.Embarked.asc_nulls_last()).tail(5)


# In[71]:


## How do we deal with missing values. 
# df.na.drop(how=("any"/"all"), thresh=(1,2,3,4,5...))
df_train.na.drop(how="any").limit(5).toPandas()


# # Advanced Tutorial
# 

# ### Spark Catalog

# In[72]:


# If you have used Spark for a while now, this is a good time to learn about spark Catalog.
# you can also totally skip this section since it is totally independed of what follows.


# In[73]:


# get all the databases in the database. 
spark.catalog.listDatabases()


# In[74]:


# get the name of the current database
spark.catalog.currentDatabase()


# In[75]:


## lists tables
spark.catalog.listTables()


# In[76]:


# add a table to the catalog
df_train.createOrReplaceTempView("df_train")


# In[77]:


# list tables
spark.catalog.listTables()


# In[78]:


# Caching
# cached table "df_train"
spark.catalog.cacheTable("df_train")


# In[79]:


# checks if the table is cached
spark.catalog.isCached("df_train")


# In[80]:


spark.catalog.isCached("df_test")


# In[81]:


# lets cahche df_test as well
spark.catalog.cacheTable("df_test")


# In[82]:


spark.catalog.isCached("df_test")


# In[83]:


# let's uncache df_train
spark.catalog.uncacheTable("df_train")


# In[84]:


spark.catalog.isCached("df_train")


# In[85]:


spark.catalog.isCached("df_test")


# In[86]:


# How about clearing all cached tables at once. 
spark.catalog.clearCache()


# In[87]:


spark.catalog.isCached("df_train")


# In[ ]:





# In[88]:


# creating a global temp view
df_train.createGlobalTempView("df_train")


# In[89]:


# listing all views in global_temp
spark.sql("SHOW VIEWS IN global_temp;").show()


# In[90]:


# dropping a table. 
spark.catalog.dropGlobalTempView("df_train")


# In[91]:


# checking that global temp view is dropped.
spark.sql("SHOW VIEWS IN global_temp;").show()


# In[92]:


spark.catalog.dropTempView("df_train")


# In[93]:


# checking that global temp view is dropped.
spark.sql("SHOW VIEWS IN global_temp;").show()


# In[94]:


spark.sql("SHOW VIEWS").show()


# ## Dealing with Missing Values
# ### Cabin

# In[95]:


# filling the null values in cabin with "N".
# df.fillna(value, subset=[]);
df_train = df_train.na.fill('N', subset=['Cabin'])
df_test = df_test.na.fill('N', subset=['Cabin'])


# ### Fare

# In[96]:


## how do we find out the rows with missing values?
# we can use .where(condition) with .isNull()
df_test.where(df_test['Fare'].isNull()).show()


# Here, We can take the average of the **Fare** column to fill in the NaN value. However, for the sake of learning and practicing, we will try something else. We can take the average of the values where **Pclass** is ***3***, **Sex** is ***male*** and **Embarked** is ***S***

# In[97]:


missing_value = df_test.filter(
    (df_test['Pclass'] == 3) &
    (df_test.Embarked == 'S') &
    (df_test.Sex == "male")
)
## filling in the null value in the fare column using Fare mean. 
df_test = df_test.na.fill(
    missing_value.select(mean('Fare')).collect()[0][0],
    subset=['Fare']
)


# In[98]:


# Checking
df_test.where(df_test['Fare'].isNull()).show()


# ### Embarked

# In[99]:


df_train.where(df_train['Embarked'].isNull()).show()


# In[100]:


## Replacing the null values in the Embarked column with the mode. 
df_train = df_train.na.fill('C', subset=['Embarked'])


# In[101]:


## checking
df_train.where(df_train['Embarked'].isNull()).show()


# In[102]:


df_test.where(df_test.Embarked.isNull()).show()


# ## Feature Engineering
# ### Cabin

# In[103]:


## this is a code to create a wrapper for function, that works for both python and Pyspark.
from typing import Callable
from pyspark.sql import Column
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, IntegerType, ArrayType, DataType
class py_or_udf:
    def __init__(self, returnType : DataType=StringType()):
        self.spark_udf_type = returnType
        
    def __call__(self, func : Callable):
        def wrapped_func(*args, **kwargs):
            if any([isinstance(arg, Column) for arg in args]) or \
                any([isinstance(vv, Column) for vv in kwargs.values()]):
                return udf(func, self.spark_udf_type)(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapped_func

    
@py_or_udf(returnType=StringType())
def first_char(col):
    return col[0]
    


# In[104]:


df_train = df_train.withColumn('Cabin', first_char(df_train['Cabin']))


# In[105]:


df_test = df_test.withColumn('Cabin', first_char(df_test['Cabin']))


# In[106]:


df_train.limit(5).toPandas()


# We can use the average of the fare column We can use pyspark's ***groupby*** function to get the mean fare of each cabin letter.

# In[107]:


df_train.groupBy('Cabin').mean("Fare").show()


# Now, these mean can help us determine the unknown cabins, if we compare each unknown cabin rows with the given mean's above. Let's write a simple function so that we can give cabin names based on the means. 

# In[108]:


@py_or_udf(returnType=StringType())
def cabin_estimator(i):
    """Grouping cabin feature by the first letter"""
    a = 0
    if i<16:
        a = "G"
    elif i>=16 and i<27:
        a = "F"
    elif i>=27 and i<38:
        a = "T"
    elif i>=38 and i<47:
        a = "A"
    elif i>= 47 and i<53:
        a = "E"
    elif i>= 53 and i<54:
        a = "D"
    elif i>=54 and i<116:
        a = 'C'
    else:
        a = "B"
    return a


# In[109]:


## separating data where Cabin == 'N', remeber we used 'N' for Null. 
df_withN = df_train.filter(df_train['Cabin'] == 'N')
df2 = df_train.filter(df_train['Cabin'] != 'N')

## replacing 'N' using cabin estimated function. 
df_withN = df_withN.withColumn('Cabin', cabin_estimator(df_withN['Fare']))

# putting the dataframe back together. 
df_train = df_withN.union(df2).orderBy('PassengerId') 


# In[110]:


#let's do the same for test set
df_testN = df_test.filter(df_test['Cabin'] == 'N')
df_testNoN = df_test.filter(df_test['Cabin'] != 'N')
df_testN = df_testN.withColumn('Cabin', cabin_estimator(df_testN['Fare']))
df_test = df_testN.union(df_testNoN).orderBy('PassengerId')


# ### Name

# In[111]:


## creating UDF functions
@py_or_udf(returnType=IntegerType())
def name_length(name):
    return len(name)


@py_or_udf(returnType=StringType())
def name_length_group(size):
    a = ''
    if (size <=20):
        a = 'short'
    elif (size <=35):
        a = 'medium'
    elif (size <=45):
        a = 'good'
    else:
        a = 'long'
    return a


# In[112]:


## getting the name length from name. 
df_train = df_train.withColumn("name_length", name_length(df_train['Name']))

## grouping based on name length. 
df_train = df_train.withColumn("nLength_group", name_length_group(df_train['name_length']))


# In[113]:


## Let's do the same for test set. 
df_test = df_test.withColumn("name_length", name_length(df_test['Name']))

df_test = df_test.withColumn("nLength_group", name_length_group(df_test['name_length']))


# ### Title

# In[114]:


## this function helps getting the title from the name. 
@py_or_udf(returnType=StringType())
def get_title(name):
    return name.split('.')[0].split(',')[1].strip()

df_train = df_train.withColumn("title", get_title(df_train['Name']))
df_test = df_test.withColumn('title', get_title(df_test['Name']))


# In[115]:


## we are writing a function that can help us modify title column
@py_or_udf(returnType=StringType())
def fuse_title1(feature):
    """
    This function helps modifying the title column
    """
    if feature in ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col', 'Rev', 'Dona', 'Dr']:
        return 'rare'
    elif feature in ['Ms', 'Mlle']:
        return 'Miss'
    elif feature == 'Mme':
        return 'Mrs'
    else:
        return feature


# In[116]:


df_train = df_train.withColumn("title", fuse_title1(df_train["title"]))


# In[117]:


df_test = df_test.withColumn("title", fuse_title1(df_test['title']))


# In[118]:


print(df_train.toPandas()['title'].unique())
print(df_test.toPandas()['title'].unique())


# ### family_size

# In[119]:


df_train = df_train.withColumn("family_size", df_train['SibSp']+df_train['Parch'])
df_test = df_test.withColumn("family_size", df_test['SibSp']+df_test['Parch'])


# In[120]:


## bin the family size. 
@py_or_udf(returnType=StringType())
def family_group(size):
    """
    This funciton groups(loner, small, large) family based on family size
    """
    
    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a


# In[121]:


df_train = df_train.withColumn("family_group", family_group(df_train['family_size']))
df_test = df_test.withColumn("family_group", family_group(df_test['family_size']))


# ### is_alone

# In[122]:


@py_or_udf(returnType=IntegerType())
def is_alone(num):
    if num<2:
        return 1
    else:
        return 0


# In[123]:


df_train = df_train.withColumn("is_alone", is_alone(df_train['family_size']))
df_test = df_test.withColumn("is_alone", is_alone(df_test["family_size"]))


# ### ticket

# In[124]:


## dropping ticket column
df_train = df_train.drop('ticket')
df_test = df_test.drop("ticket")


# ### calculated_fare

# In[125]:


from pyspark.sql.functions import expr, col, when, coalesce, lit


# In[126]:


## here I am using a something similar to if and else statement, 
#when(condition, value_when_condition_met).otherwise(alt_condition)
df_train = df_train.withColumn(
    "calculated_fare", 
    when((col("Fare")/col("family_size")).isNull(), col('Fare'))
    .otherwise((col("Fare")/col("family_size"))))


# In[127]:


df_test = df_test.withColumn(
    "calculated_fare", 
    when((col("Fare")/col("family_size")).isNull(), col('Fare'))
    .otherwise((col("Fare")/col("family_size"))))


# ### fare_group

# In[128]:


@py_or_udf(returnType=StringType())
def fare_group(fare):
    """
    This function creates a fare group based on the fare provided
    """
    
    a= ''
    if fare <= 4:
        a = 'Very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 20:
        a = 'mid'
    elif fare <= 45:
        a = 'high'
    else:
        a = "very_high"
    return a


# In[129]:


df_train = df_train.withColumn("fare_group", fare_group(col("Fare")))
df_test = df_test.withColumn("fare_group", fare_group(col("Fare")))


# # That's all for today. Let's come back tomorrow when we will learn how to apply machine learning with Pyspark

# In[130]:


# Binarizing, Bucketing & Encoding


# In[131]:


train = spark.read.csv('../input/titanic/train.csv', header = True, inferSchema=True)
test = spark.read.csv('../input/titanic/test.csv', header = True, inferSchema=True)


# In[132]:


train.show()


# In[133]:


# Binarzing
from pyspark.ml.feature import Binarizer
# Cast the data type to double
train = train.withColumn('SibSp', train['SibSp'].cast('double'))
# Create binarzing transform
bin = Binarizer(threshold=0.0, inputCol='SibSp', outputCol='SibSpBin')
# Apply the transform
train = bin.transform(train)


# In[134]:


train.select('SibSp', 'SibSpBin').show(10)


# In[135]:


# Bucketing
from pyspark.ml.feature import Bucketizer
# We are going to bucket the fare column
# Define the split
splits = [0,4,10,20,45, float('Inf')]

# Create bucketing transformer
buck = Bucketizer(splits=splits, inputCol='Fare', outputCol='FareB')

# Apply transformer
train = buck.transform(train)


# In[136]:


train.toPandas().head(10)


# In[137]:


# One Hot Encoding
# it is a two step process
from pyspark.ml.feature import OneHotEncoder, StringIndexer
# Create indexer transformer for Sex Column

# Step 1: Create indexer for texts
stringIndexer = StringIndexer(inputCol='Sex', outputCol='SexIndex')

# fit transform
model = stringIndexer.fit(train)

# Apply transform
indexed = model.transform(train)


# In[138]:


# Step 2: One Hot Encode
# Create encoder transformer
encoder = OneHotEncoder(inputCol='SexIndex', outputCol='Sex_Vec')

# fit model
model = encoder.fit(indexed)

# apply transform
encoded_df = model.transform(indexed)

encoded_df.toPandas().head()


# In[ ]:





# In[ ]:





# In[ ]:





# <div class="alert alert-info">
#     <h1>Resources</h1>
#     <ul>
#         <li><a href="https://docs.databricks.com/spark/latest/spark-sql/udf-python.html">User-defined functions - Python</a></li>
#         <li><a href="https://medium.com/@ayplam/developing-pyspark-udfs-d179db0ccc87">Developing PySpark UDFs</a></li>
#     </ul>
#         <h1>Credits</h1>
#     <ul>
#         <li>To DataCamp, I have learned so much from DataCamp.</li>
#         <li>To Jose Portilla, Such an amazing teacher with all of his resources</li>
#     </ul>
#     
# </div>

# <div class="alert alert-info">
# <h4>If you like to discuss any other projects or just have a chat about data science topics, I'll be more than happy to connect with you on:</h4>
#     <ul>
#         <li><a href="https://www.linkedin.com/in/masumrumi/"><b>LinkedIn</b></a></li>
#         <li><a href="https://github.com/masumrumi"><b>Github</b></a></li>
#         <li><a href="https://masumrumi.com/"><b>masumrumi.com</b></a></li>
#         <li><a href="https://www.youtube.com/channel/UC1mPjGyLcZmsMgZ8SJgrfdw"><b>Youtube</b></a></li>
#     </ul>
# 
# <p>This kernel will always be a work in progress. I will incorporate new concepts of data science as I comprehend them with each update. If you have any idea/suggestions about this notebook, please let me know. Any feedback about further improvements would be genuinely appreciated.</p>
# 
# <h1>If you have come this far, Congratulations!!</h1>
# 
# <h1>If this notebook helped you in any way or you liked it, please upvote and/or leave a comment!! :)</h1></div>

# <div class="alert alert-info">
#     <h1>Versions</h1>
#     <ul>
#         <li>Version 16</li>
#     </ul>
#     
# </div>

# <div class="alert alert-danger">
#     <h1>Work Area</h1>
# </div>

# ### Other DataFrame Methods

# In[139]:


df_train.show(5)


# In[140]:


# agg
df_train.agg({"Age" : "min"}).show()


# In[141]:


# agg
from pyspark.sql import functions as F
df_train.groupBy("Sex").agg(
    F.min("Age").name("min_age"), 
    F.max("Age").alias("max_age")).show()


# In[142]:


# colRegex
df_train.select(df_train.colRegex("`(Sex)?+.+`")).show(5)


# In[143]:


# distinct
df_train[['Pclass', 'Sex']].distinct().show()


# In[144]:


# another way
# dropDuplicates
df_train[['Pclass', 'Sex']].dropDuplicates().show()


# In[145]:


# beware, this is probably not something we want when we try to do dropDuplicates
df_train.dropDuplicates(subset=['Pclass']).show()


# In[146]:


# drop_dupllicates()
# drop_duplicates() is an alias of dropDuplicates()
df_train[['Pclass', 'Sex']].drop_duplicates().show()


# In[147]:


# drop
# dropping a column
df_train.drop('Name').show(5)


# In[148]:


# drop
# dropping multiple columns
df_train.drop("name", "Survived").show(5)


# In[149]:


# dropna
df_train.dropna(how="any", subset=["Age"]).count()


# In[150]:


#similarly
df_train.na.drop(how="any", subset=['Age']).count()


# In[151]:


# exceptAll
# temp dataframes
df1 = spark.createDataFrame(
        [("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b",  3), ("c", 4)], ["C1", "C2"])
df2 = spark.createDataFrame([("a", 1),("a", 1), ("b", 3)], ["C1", "C2"])


# In[152]:


df1.show()


# In[153]:


df2.show()


# In[154]:


df1.exceptAll(df2).show()


# In[155]:


# intersect
df1.intersect(df2).show()


# In[156]:


# intersectAll
# intersectAll preserves the duplicates. 
df1.intersectAll(df2).show()


# In[157]:


# Returns True if the collect() and take() methods can be run locally
df_train.isLocal()


# In[158]:


## fillna
df_train.fillna("N", subset=['Cabin']).show()


# In[159]:


# similarly
# dataFrame.na.fill() is alias of dataFrame.fillna()
df_train.na.fill("N", subset=['Cabin']).show()


# In[160]:


age_mean = df_train.agg({"Age": "mean"}).collect()[0][0]


# In[161]:


age_mean


# In[162]:


df_train.fillna({"Age": age_mean, "Cabin": "N"})[['Age', "Cabin"]].show(10)


# In[163]:


# first
df_train.first()


# In[164]:


def f(passenger):
    print(passenger.Name)


# In[165]:


# foreach
# this prints out in the terminal. 
df_train.foreach(f)


# In[166]:


# freqItems
# this function is meant for exploratory data analysis.
df_train.freqItems(cols=["Cabin"]).show()


# In[167]:


# groupBy
# pandas value_counts() equivalent. 
df_train.groupBy("Fare").count().orderBy("count", ascending=False).show()


# In[168]:


df_train.groupBy(['Sex', 'Pclass']).count().show()


# In[169]:


df_train.hint("broadcast").show()


# In[170]:


# isStreaming
# Returns True if this DataFrame contains one or more sources that continuously return data as it arrives.
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.isStreaming.html
df_train.isStreaming


# In[171]:


# sort/orderBy
df_train.sort('Survived', ascending = False).show()


# In[172]:


# randomSplit
# randomly splits the dataframe into two based on the given weights.
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.randomSplit.html
splits = df_train.randomSplit([1.0, 2.0], seed=42)


# In[173]:


splits[0].count()


# In[174]:


splits[0].show(5)


# In[175]:


splits[1].count()


# In[176]:


splits[1].show(5)


# In[177]:


# replace
df_train.replace("male", "Man").show(5)


# In[178]:


# similarly
df_train.na.replace("male", "Man").show(5)


# In[179]:


# cube
# the following stack overflow explains cube better than official spark page. 
# https://stackoverflow.com/questions/37975227/what-is-the-difference-between-cube-rollup-and-groupby-operators
df = spark.createDataFrame([("foo", 1), ("foo", 2), ("bar", 2), ("bar", 2)]).toDF("x", "y")
df.show()


# In[180]:


temp_df.show()


# In[181]:


df.cube("x", "y").count().show()


# Here is what cube returns
# ```
# // +----+----+-----+     
# // |   x|   y|count|
# // +----+----+-----+
# // | foo|   1|    1|   <- count of records where x = foo AND y = 1
# // | foo|   2|    1|   <- count of records where x = foo AND y = 2
# // | bar|   2|    2|   <- count of records where x = bar AND y = 2
# // |null|null|    4|   <- total count of records
# // |null|   2|    3|   <- count of records where y = 2
# // |null|   1|    1|   <- count of records where y = 1
# // | bar|null|    2|   <- count of records where x = bar
# // | foo|null|    2|   <- count of records where x = foo
# // +----+----+-----+```

# In[182]:


# rollup
df.rollup("x", "y").count().show()


# Here is what rollup's look like
# ```
# // +----+----+-----+
# // |   x|   y|count|
# // +----+----+-----+
# // | foo|null|    2|   <- count where x is fixed to foo
# // | bar|   2|    2|   <- count where x is fixed to bar and y is fixed to  2
# // | foo|   1|    1|   ...
# // | foo|   2|    1|   ...
# // |null|null|    4|   <- count where no column is fixed
# // | bar|null|    2|   <- count where x is fixed to bar
# // +----+----+-----+```

# In[183]:


# sameSemantics
# Returns True when the logical query plans inside both DataFrames are equal and therefore return same results.
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.sameSemantics.html
df1 = spark.range(10)
df2 = spark.range(10)


# In[184]:


df1.show()


# In[185]:


df2.show()


# In[186]:


df1.withColumn("col1", df1.id * 2).sameSemantics(df2.withColumn("col1", df2.id * 2))


# In[187]:


df1.withColumn("col1", df1.id * 2).sameSemantics(df2.withColumn("col1", df2.id + 2))


# In[188]:


df1.withColumn("col1", df1.id * 2).sameSemantics(df2.withColumn("col0", df2.id * 2))


# In[189]:


df_train.schema


# In[190]:


df_train.printSchema()


# In[ ]:





# In[ ]:




