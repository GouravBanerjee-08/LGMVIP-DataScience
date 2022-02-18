#!/usr/bin/env python
# coding: utf-8

# # Let's Grow More❤️

# ### Name: Gourav Banerjee
# Data Science Intern @ LGMVIP Feb-2022
# #### TASK 1 { Beginner Level Task } - Iris Flowers Classification ML Project
# 

# Description of Task:
#     This particular ML project is usually referred to as the “Hello World” of Machine Learning. The iris flowers dataset contains numeric attributes, and it is perfect for beginners to learn about supervised ML algorithms, mainly how to load and handle data. Also, since this is a small dataset, it can easily fit in memory without requiring special transformations or scaling capabilities.
# 

# Dataset: https://archive.ics.uci.edu/ml/datasets/Iris

# ## Importing Libraries

# In[2]:


import numpy as np                   

import pandas as pd                  

import seaborn as sns  

import matplotlib.pyplot as plt                   

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Dataset Description
# The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.
# 
# ### Attribute Information:
# 
# Sepal length in cm
# Sepal width in cm
# Petal length in cm
# Petal width in cm
# Iris flower can be divided into 3 species as per the length and width of their Sepals and Petals:
# 
# 1) Iris Setosa
# 2) Iris Versicolour
# 3) Iris Virginica
# ![lgmviptask1.png](attachment:lgmviptask1.png)

# ## Importing Dataset

# In[3]:


#Reading/Importing dataset
data=pd.read_csv(r'D:\6thSem\Iris.csv')


# In[4]:


#Printing first 5 rows of the dataset
data.head()


# In[5]:


#Printing last 5 rows of the dataset
data.tail()


# #### Data Preprocessing

# In[6]:


#Printing the shape of the dataset
data.shape


# In[7]:


#Summary of Dataset
data.info()


# In[8]:


#Checking the null values
data.isnull()


# In[9]:


#Returns the number of missing values in the dataset.
data.isnull().sum()


# In[10]:


#Statistical Summary of the dataset
data.describe()


# In[11]:


#Columns of Dataset
data.columns


# In[12]:


#To return no, of unique elements in the object
data.nunique()


# In[13]:


data.max()


# In[14]:


data.min()


# In[15]:


# To display no. of samples on each class.
data['Species'].value_counts()


# In[16]:


#Pie plot to show the overall types of Iris classifications
data['Species'].value_counts().plot(kind = 'pie',  autopct = '%1.1f%%', shadow = True, explode = [0.08,0.08,0.08])


# ##### Correlation Matrix

# In[17]:


data.corr()


# ### Heat Map
# 

# In[18]:


#Correlation Heatmap
plt.figure(figsize=(9,7))
sns.heatmap(data.corr(),cmap='CMRmap',annot=True,linewidths=2)
plt.title("Correlation Graph",size=20)
plt.show()


# The diagonal values are 1 as expected as they show relation of the feature with itself. Also, there is high positive correlation for Petal width with Sepal length and Petal length. Also, correlation between Petal length and Sepal length is positively high

# ### Label encoding for categorical variables
# 

# In machine learning, we usually deal with datasets which contains multiple labels in one or more than one columns. These labels can be in the form of words or numbers. Label Encoding refers to converting the labels into numeric form so as to convert it into the machine-readable form

# In[19]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[20]:


data['Species'] = le.fit_transform(data['Species'])
data.head()


# In[21]:


# To display no. of samples on each class.
data['Species'].unique()


# ###  Splitting X and Y into Train and Test datasets

# In[22]:


# Splitting dataset 
from sklearn.model_selection import train_test_split

features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = data.loc[:, features].values   #defining the feature matrix
Y = data.Species

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 40,random_state=0)


# In[23]:


X_Train.shape


# In[24]:


X_Test.shape


# In[25]:


Y_Train.shape


# In[26]:


Y_Test.shape


# ### Data Scaling
# 

# In[27]:


# Feature Scaling to bring all the variables in a single scale.

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_Train = sc.fit_transform(X_Train)
X_Test = sc.transform(X_Test)


# In[28]:


# Importing some metrics for evaluating  models.
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import  classification_report
from sklearn.metrics import confusion_matrix


# ### Model Creation

# #### Logistic Regression
# 

# In[29]:


from sklearn.linear_model import LogisticRegression
log_model= LogisticRegression(random_state = 0)
log_model.fit(X_Train, Y_Train)


# In[30]:


# model training
log_model.fit(X_Train, Y_Train)


# In[31]:


# Predicting
Y_Pred_Test_log_res=log_model.predict(X_Test)


# In[32]:


Y_Pred_Test_log_res


# In[33]:


print("Accuracy:",metrics.accuracy_score(Y_Test, Y_Pred_Test_log_res)*100)


# In[34]:


print(classification_report(Y_Test, Y_Pred_Test_log_res))


# In[35]:


confusion_matrix(Y_Test,Y_Pred_Test_log_res )


# #### KNN(K-Nearest Neighbours)
# 

# In[36]:


# Importing KNeighborsClassifier from sklearn.neighbors library

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto')


# In[37]:


# Importing KNeighborsClassifier 
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)


# In[38]:


# model training
knn_model.fit(X_Train, Y_Train)


# In[39]:


# Predicting
Y_Pred_Test_knn=knn_model.predict(X_Test)


# In[40]:


Y_Pred_Test_knn


# In[41]:


print("Accuracy:",metrics.accuracy_score(Y_Test,Y_Pred_Test_knn)*100)


# In[42]:


print(classification_report(Y_Test,Y_Pred_Test_knn))


# In[43]:


confusion_matrix(Y_Test, Y_Pred_Test_knn)


# #### Decision Tree
# 

# In[44]:


# Importing DecisionTreeClassifier from sklearn.tree library and creating an object of it  with hyper parameters criterion,splitter and max_depth.

from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=6)


# In[45]:


# model training
dec_tree.fit(X_Train, Y_Train)


# In[46]:


# Predicting
Y_Pred_Test_dtr=dec_tree.predict(X_Test)


# In[47]:


Y_Pred_Test_dtr


# In[48]:


print("Accuracy:",metrics.accuracy_score(Y_Test, Y_Pred_Test_dtr)*100)


# In[49]:


print(classification_report(Y_Test, Y_Pred_Test_dtr))


# In[50]:


confusion_matrix(Y_Test, Y_Pred_Test_dtr)


# #### Naive Bayes
# 

# In[51]:


from sklearn.naive_bayes import GaussianNB
nav_byes = GaussianNB()


# In[52]:


# model training
nav_byes.fit(X_Train, Y_Train)


# In[53]:


# Predicting
Y_Pred_Test_nvb=nav_byes.predict(X_Test)


# In[54]:


Y_Pred_Test_nvb


# In[55]:


print("Accuracy:",metrics.accuracy_score(Y_Test, Y_Pred_Test_nvb)*100)


# In[56]:


print(classification_report(Y_Test, Y_Pred_Test_nvb))


# In[57]:


confusion_matrix(Y_Test,Y_Pred_Test_nvb )


# #### Random Forest Classification
# 

# In[58]:


from sklearn.ensemble import RandomForestClassifier
Ran_for = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')


# In[59]:


# model training
Ran_for.fit(X_Train, Y_Train)


# In[60]:


# Predicting
Y_Pred_Test_rf=Ran_for.predict(X_Test)


# In[61]:


Y_Pred_Test_rf


# In[62]:


print("Accuracy:",metrics.accuracy_score(Y_Test,Y_Pred_Test_rf)*100)


# In[63]:


print(classification_report(Y_Test, Y_Pred_Test_rf))


# In[64]:


confusion_matrix(Y_Test,Y_Pred_Test_rf )


# #### SVM
# 

# In[65]:


# Importing SVC from sklearn.svm library

from sklearn.svm import SVC
svm_model=SVC(C=500, kernel='rbf')


# In[66]:


# model training
svm_model.fit(X_Train, Y_Train)


# In[67]:


# Predicting
Y_Pred_Test_svm=svm_model.predict(X_Test)


# In[68]:


Y_Pred_Test_svm


# In[69]:


print("Accuracy:",metrics.accuracy_score(Y_Test,Y_Pred_Test_svm)*100)


# In[70]:


print(classification_report(Y_Test, Y_Pred_Test_svm))


# In[71]:


confusion_matrix(Y_Test,Y_Pred_Test_svm )


# ### Model Evaluation Results
# 
# Model - 	                    Accuracy Score
# 
# Logistic Regression - 	        97.5
# 
# KNN(K-Nearest Neighbours) - 	97.5
# 
# Decision Tree - 	            97.5
# 
# Naive Bayes - 	                100.0
# 
# Random Forest - 	            97.5
# 
# SVM - 	                        97.5

# We got highest accuracy Score of 100 for Naive Bayes
# 
# 

# ### 11. Conclusions

# ​ Our dataset was not very large and consisted of only 150 rows, with all the 3 species uniformly distributed.
# 
# ​ PetalWidthCm was highly correlated with PetalLengthCm
# 
# ​ PetalLengthCm was highly correlated with PetalWidthCm
# 
# ​ We tried with 6 different machine learning Classification models on the Iris Test data set to classify the flower into it's three species:
#     a) Iris Setosa
#     b) Iris Versicolour
#     c) Iris Virginica,
# based on the length and width of the flower's Petals and Sepals.
# 
# ​ We got very high accuracy score for all the models, and even the accuracy score of 100 for KNN and SVM with Linear Kernel models with some hyper parameter tuning maybe due to small size of dataset.

# ## THANK YOU
