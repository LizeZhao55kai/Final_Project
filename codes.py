#!/usr/bin/env python
# coding: utf-8

# In[123]:


import pandas as pd
import numpy as np
#import tensorflow
import os
import sys
import matplotlib.pyplot as plt
import imblearn
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor 
from sklearn.linear_model import ElasticNet,SGDRegressor,BayesianRidge,LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn import utils


# In[124]:


print('python version:'+sys.version)
print('packages:\n'+'\n'.join(f'{m.__name__}={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))


# <font face="Times New Roman" size=5 color=#000000 > 
# Load the Data  <br/>
# <font face="Times New Roman" size=3 color=#000000 >  
# We use 2020-2021 Marketscan data to analyse and we extract the first 5000 rows'data in 'mdcro.csv'.

# In[127]:


df = pd.read_csv('mdcro201.csv',nrows = 1000000,low_memory = False)


# In[128]:


null_val_count = []
for col in df.columns:
    null_val_count.append(df[col].isna().any())
res = True in (x==True for x in null_val_count)
res


# <font face="Times New Roman" size=5 color=#000000 >
#     Data Preprocessing  <br/>
# <font face="Times New Roman" size=3 color=#000000 >
#    We need to do data cleaning and construct them as training sets and testing sets.

# In[129]:


def get_num_cols(data):
    cols = data.columns
    num_cols = data._get_numeric_data().columns
    return num_cols


# In[130]:


def get_categorical_cols(data):
    cols = data.columns
    num_cols = data._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    return cat_cols


# We need to delete the rows which 'sex' colomn is NAN

# In[131]:


df = df[df['SEX'].notna()]
df = df[df['PROCGRP'].notna()]
df.head(n = 10)


# In[132]:


df.shape


# Search the relationship of the data

# In[133]:


pd.crosstab(df.PROC1, df.SEX)


# Assemble several columns as a new datadet

# In[134]:


data = df[['SEQNUM','VERSION','AGE','SEX','SVCDATE','TSVCDAT','PROCGRP','COPAY','DEDUCT','PAY','NETPAY','COB','REVCODE','REGION','COINS','STDPLAC','STDPROV']]
data = data[data['REVCODE'].notna()]
data.head(n=10)


# In[135]:


data.shape


# Use the 'PAY' data and the 'COPAY' data to calculate the 'CLAIM' data.

# In[136]:


data['CLAIM'] = data['PAY']-data['NETPAY']


# In[137]:


data.head(n = 10)


# In[138]:


#Find the columns we use and mess up the order and reset the index
data = data[['SEQNUM','VERSION','AGE','SEX','SVCDATE','TSVCDAT','PROCGRP','CLAIM','NETPAY','REVCODE','REGION','STDPROV','STDPLAC']]

data = data.sample(frac=1.0)
data = data.reset_index()
#data = data.drop()
data.head(n=10)


# In[139]:


#Calculate the total number of days of treatment and consider it as a new colomn of dataset
#df['end time'] = pd.to_datetime(df['end time'])
#df['start time'] = pd.to_datetime(df['start time'])
#df['xxx time'] = (df['end time'] - df['start time']).dt.seconds/60
#dataSet['t'] =dataSet['time'].astype('timedelta64[D]').astype(float)
data['TIME_NUMBER'] = pd.to_datetime(data['TSVCDAT']) - pd.to_datetime(data['SVCDATE'])
data['TIME_NUMBER'] = data['TIME_NUMBER'].astype('timedelta64[D]').astype(float)
#data['TIME_NUMBER'] = data['TIME_NUMBER'].days()
#new_df = pd.DataFrame(pd.to_datetime(data['TSVCDAT']) - pd.to_datetime(data['SVCDATE']))
#new_df.head(n = 10)
#new_df = new_df.rename(columns = {'0':'NUMBER_OF_DAYS'})
#data = data[['SEQNUM','VERSION','AGE','SEX','SVCDATE','TSVCDAT','PROCGRP','CLAIM']]
#data = data.join(new_df,how = 'left')
#data2 = pd.concat([data,new_df],axis = 1)
#data = data2.rename(columns = {'0':'NUMBER_OF_DAYS'},inplace = True)
#data = data2[['SEQNUM','VERSION','AGE','SEX','SVCDATE','TSVCDAT','PROCGRP','CLAIM','NETPAY','REVCODE','REGION','0']]
data.head(n = 10)


# In[140]:


#df = df[df['SEX'].notna()]
data = data.dropna(axis = 0)


# In[141]:


data['REVCODE'].dtypes


# In[159]:


#Devide the data into X and Y

X_data = pd.DataFrame(data, columns=[ 'AGE','SEX','TIME_NUMBER','PROCGRP','REVCODE','REGION'])
Y_data = pd.DataFrame(data, columns = ['CLAIM'])


# In[160]:


X_data.dtypes


# In[161]:


Y_data.dtypes


# Devide all the data into the training data and the test data

# In[162]:


#We have 84984 rows so we choose the first 80%'s data as the training data and the last 20% as the test data
#X_data_training = X_data.head(n=67987)
#Y_data_training = Y_data.head(n=67987)

#X_data_test = X_data.tail(n = 16997)
#Y_data_test = Y_data.tail(n = 16997)
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.33, random_state = 5)
stratified_kfold = StratifiedKFold(n_splits=10)


# In[163]:


X_train.shape


# In[164]:


Y_train.shape


# In[166]:


# Split the training and validation datasets and their labels.
X_train, X_val, y_train, y_val = train_test_split(X_data,Y_data,random_state = 1912)

print('The training and validation datasets and labels have been split.')


# <font face="Times New Roman" size=5 color=#000000 >
#  Dicision Tree Regression

# In[167]:


X_val.shape


# In[168]:


# Perform common cleaning and feature engineering tasks on datasets.
def prep_dataset(dataset):
    
    # PROVIDE MISSING VALUES
    
    # Fill missing Age values with the median age.
    #dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    # Fill missing Fare values with the median fare.
    #dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    # Fill missing Embarked values with the mode.
    #dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    
    # ONE-HOT ENCODING
    
    cols = ['SEX','REGION']
    
    for i in cols:
        dummies = pd.get_dummies(dataset[i], prefix = i, drop_first = False)
        dataset = pd.concat([dataset, dummies], axis = 1)

    return dataset

X_train = prep_dataset(X_train.copy())

X_val = prep_dataset(X_val.copy())

print('The dataset has been cleaned and prepared.')


# In[169]:


# Drop unused columns from datasets.
def drop_unused(dataset):
        
    #dataset = dataset.drop(['PassengerId'], axis = 1)
    #dataset = dataset.drop(['Cabin'], axis = 1)
    #dataset = dataset.drop(['Ticket'], axis = 1)
    #dataset = dataset.drop(['Name'], axis = 1)

    # These have been replaced with one-hot encoding.
    dataset = dataset.drop(['SEX'], axis = 1)
    #dataset = dataset.drop(['REVCODE'], axis = 1)
    dataset = dataset.drop(['REGION'], axis = 1)
    
    return dataset

X_train = drop_unused(X_train.copy())

X_val = drop_unused(X_val.copy())

print('Columns that will not be used for training have been dropped.')


# In[170]:


X_train.head()


# In[171]:


X_val.head()


# In[172]:


from time import time


# In[173]:


y_val.head()


# In[231]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,KFold

tree = DecisionTreeRegressor(random_state = 1024)

start = time()
tree.fit(X_train, np.ravel(y_train.astype(int)))
end = time()
train_time = (end - start) * 1000

prediction = tree.predict(X_val)

kf = KFold(n_splits = 5)

# Score using the validation data.
score = tree.score(X_val, y_val)
score_2 = cross_val_score(tree,X_train,y_train,cv = kf)

print('Decision tree model took {:.2f} milliseconds to fit.'.format(train_time))
print('Accuracy: {:.0f}%'.format(score * 100))
print('CV Accuracy: {:.0f}%'.format(score * 100))


#inputs_labels.argmax(axis=1),predicted.numpy().argmax(axis=1)


# In[214]:


from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO  
import sklearn.externals
from IPython.display import Image, display 
import pydotplus as pdotp

def plot_tree(model, image):
    dot_data = sklearn.externals.StringIO()
    export_graphviz(model, out_file = dot_data, 
                    filled = True,
                    rounded = True,
                    special_characters = True, 
                    feature_names = X_train.columns.values.tolist(),
                    class_names = ['0', '1'])

    graph = pdotp.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png(image)
    Image(graph.create_png())
    
print('A function to plot the decision tree structure has been defined.')


# In[215]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def model_scores(y, prediction):
    acc = accuracy_score(y, prediction)
    print('Accuracy: {:.0f}%'.format(np.round(acc * 100)))
    
    precision = precision_score(y, prediction)
    print('Precision: {:.0f}%'.format(np.round(precision * 100)))
    
    recall = recall_score(y, prediction)
    print('Recall: {:.0f}%'.format(np.round(recall * 100)))
    
    f1 = f1_score(y, prediction)
    print('F1: {:.0f}%'.format(np.round(f1 * 100)))
    
print('A function to compute the model scores has been defined.')


# In[216]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def roc(y, prediction_proba):
    fpr, tpr, thresholds = roc_curve(y, prediction_proba)
    
    plt.plot(fpr, tpr);
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.0]);
    plt.title('ROC Curve');
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.grid(True);
    
    auc = roc_auc_score(y, prediction_proba)
    print('Area Under Curve: {:.2f}'.format(auc))
    
print('A function to generate the ROC curve and compute AUC has been defined.')


# In[217]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def prc(y, prediction_proba):
    precision, recall, thresholds = precision_recall_curve(y, prediction_proba)
    
    plt.plot(recall, precision);
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.0]);
    plt.title('Precision–Recall Curve');
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    plt.grid(True);
    
    ap = average_precision_score(y, prediction_proba)
    print('Average Precision: {:.2f}'.format(ap))
    
print('A function to generate the PRC and compute average precision has been defined.')


# <font face="Times New Roman" size=5 color=#000000 >
# Prediction: Random Forest Regression

# In[233]:


from sklearn.ensemble import RandomForestRegressor

start = time()
forest = RandomForestRegressor(n_estimators = 100,
                                criterion = 'squared_error',
                                bootstrap = True,
                                oob_score = True,
                                random_state = 1024)

forest.fit(X_train, np.ravel(y_train))
end = time()
train_time = (end - start) * 1000

prediction = forest.predict(X_val)

# Score using the validation data.
score = forest.score(X_val, y_val)

print('Decision tree model took {:.2f} milliseconds to fit.'.format(train_time))
print('Accuracy: {:.0f}%'.format(score * 100))


# <font face="Times New Roman" size=5 color=#000000 >
# Prediction: Simple Linear Regression（Multiple dimensions)

# In[72]:


data.describe()


# In[73]:


import seaborn as sns


# In[74]:


data['AGE'].value_counts().plot(kind='bar')
plt.title('number of Ages')
plt.xlabel('Age')
plt.ylabel('Count')
sns.despine


# In[75]:


data['SEX'].value_counts().plot(kind='bar')
plt.title('number of men and women')
plt.xlabel('Sex')
plt.ylabel('Count')
sns.despine


# In 'sex' part, 1.0 represents Male and 2.0 represents Female

# In[76]:


data['REGION'].value_counts().plot(kind='bar')
plt.title('Region in Categories')
plt.xlabel('Region')
plt.ylabel('Count')
sns.despine


# In[77]:


plt.figure(figsize=(10,10))
sns.jointplot(x=data.CLAIM.values, y=data.TIME_NUMBER.values, height=10)
plt.ylabel('Claim', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.show()
sns.despine


# In[78]:


plt.scatter(data.CLAIM,data.TIME_NUMBER)
plt.xlabel("Claim")
plt.ylabel('Time')
plt.title("Claim vs Time")


# In[79]:


reg = LinearRegression()


# In[117]:


labels = data['CLAIM']
conv_dates = [1 if values == 0.0 else values for values in data.TIME_NUMBER]
data['TIME_NUMBER'] = conv_dates
#train1 = data.drop(['id', 'price'],axis=1)


# In[81]:


reg.fit(X_train,Y_train)
reg.score(X_test,Y_test)


# This model is very weak

# In[82]:


from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')


# In[84]:


clf.fit(X_train, Y_train)


# In[85]:


clf.score(X_test,Y_test)


# In[107]:


import numpy as np
import mpl_toolkits 


# In[108]:


t_sc = np.zeros([400,400])


# In[109]:


y_pred = reg.predict(X_test)


# In[115]:


#for i,y_pred in enumerate(clf.staged_predict(X_test)):
#    t_sc[i]=clf.loss_(Y_test,y_pred)


# In[111]:


testsc = np.arange(0,400,1)+1


# In[114]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(testsc,clf.train_score_,'b-',label= 'Set dev train')
#plt.plot(testsc,t_sc,'r-',label = 'set dev test')


# This is all about the single feature. Remember we have 8 features.

# In[ ]:





# <font face="Times New Roman" size=5 color=#000000 >
# Prediction: Logistic Regression and SGD Regression

# In[55]:


X_train.describe()


# In[91]:


X_train.shape


# In[56]:


Y_train.describe()


# In[205]:


Y_train.shape


# In[202]:


#convert y values to categorical values
lab = preprocessing.LabelEncoder()
#Y_train = np.ravel(Y_train,order = 'C')
y_transformed = lab.fit_transform(Y_train)


# In[203]:


#Train model
log_reg = LogisticRegression(random_state = 42)
log_reg.fit(X_train,y_transformed)


# In[206]:


#Build an array for visualisation
X_1d_new = np.linspace(-610, 32128, num = 19719).reshape(-1, 1)
y_1d_proba = log_reg.predict_proba(X_1d_new)

plt.figure(figsize=(8,4))
plt.plot(X_1d_new, y_1d_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(X_1d_new, y_1d_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14);


# In[ ]:




