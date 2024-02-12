# Source code for the article "DDOS Prevention in IoT Networks by Analyzing Source-Side Inter-Bot Traffic Using Machine Learning Techniques"

# Authors:
#           Saba Malekzadeh
#           Dr. Saleh Yousefi [s.yousefi@urmia.ac.ir]
#           Dr. Mir Saman Tajbakhsh (corresponding author) [ms.tajbakhsh@urmia.ac.ir]

# Running the code:
#                   Change csv_data to one of the data, dataFull, dataProbable and it will run all the scenarios.
import numpy
import matplotlib.pyplot as plt
import pandas
import math
import logging
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import roc_curve, auc, log_loss
from collections import Counter
from keras.layers import Layer
import keras.backend as K
from keras.layers import Input, Dense, SimpleRNN
from keras import Model

# fix random seed for reproducibility
numpy.random.seed(7)

# data dataFull dataProbable
csv_data = 'data'
file = csv_data + '.csv'
epoc_count = 20
verbos_level = 0
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def makecategorical(dataframe, columns):
    for i in dataframe[columns]:
        dataframe[i] = pandas.Categorical(dataframe[i])
    return dataframe

def makeintegral(dataframe, columns):
    for i in dataframe[columns]:
        dataframe[i] = dataframe[i].astype(int)
    return dataframe

def makefloatical(dataframe, columns):
    for i in dataframe[columns]:
        dataframe[i] = dataframe[i].astype(float)
    return dataframe

def removecolumn(dataframe, columns):
    dataframe = dataframe.drop(columns, axis = 1)
    return dataframe

'''
   Usage: assignlabeltonumerical(dataframe, 'sport', [-1,1024,2048,99999], ['lowPORT', 'midPORT', 'highPORT'])
'''
def assignlabeltonumerical(dataframe, theColumn, splitNums, newLabels):
    dataframe[theColumn] = pandas.cut(x=dataframe[theColumn],
                                         bins=splitNums,
                                         labels=newLabels)
    return dataframe
    
def removenulls(dataframe):
    return dataframe.dropna(how='any', axis=0)

def create_baseline():
    model = Sequential()
    
    model.add(Bidirectional(LSTM(64, activation='tanh', kernel_regularizer='l2')))
    model.add(Dense(128, activation = 'relu', kernel_regularizer='l2'))
    model.add(Dense(1, activation = 'sigmoid', kernel_regularizer='l2'))
    
    #model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.compile('adam', 'binary_crossentropy')
    return model

    

# load the dataset
dataframe = pandas.read_csv(file, engine='python')

# Purify
# attack,category,subcategory,AttackProfile
dataframe = removenulls(dataframe)

if ("Full" in file):
    dataframe = removecolumn(dataframe, ['pkSeqID','stime','ltime','category','subcategory','AttackProfile','flgs','proto','state','saddr','daddr'])
    dataframe = makeintegral(dataframe, ['sport','dport','pkts','bytes','spkts','dpkts','sbytes','dbytes'])
elif ("Probable" in file):
    dataframe = removecolumn(dataframe, ['pkSeqID','stime','ltime','category','subcategory','AttackProfile','flgs','proto','state','saddr','daddr'])
    dataframe = makeintegral(dataframe, ['sport','dport','pkts','bytes','spkts','dpkts','sbytes','dbytes', 'ProbableAttack'])
else:
    dataframe = removecolumn(dataframe, ['pkSeqID','stime','ltime','category','subcategory','flgs','proto','state','saddr','daddr'])
    dataframe = makeintegral(dataframe, ['sport','dport','pkts','bytes','spkts','dpkts','sbytes','dbytes'])
    
dataframe = makefloatical(dataframe, ['seq','dur','mean','stddev','sum','min','max','rate','srate','drate'])

Y = dataframe.pop('attack')
#Attack is 0 or 1 by default
X = dataframe
# Standardise the data
scalar = StandardScaler(copy=True, with_mean=True, with_std=True)

scalar.fit(X)
X = scalar.transform(X)

# Shape the data
features = len(X[0])
samples = X.shape[0]
train_len = 25
input_len = samples - train_len
I = numpy.zeros((samples - train_len, train_len, features))
for i in range(input_len):
    temp = numpy.zeros((train_len, features))
    for j in range(i, i + train_len - 1):
        temp[j-i] = X[j]
    I[i] = temp
X.shape
X_train, X_test, Y_train, Y_test = train_test_split(I, Y[25:100000], test_size = 0.2)

model = create_baseline()
history = model.fit(X_train, Y_train, epochs = epoc_count,validation_split=0.2, verbose = verbos_level)
predict = model.predict(X_test, verbose=verbos_level)
tp = 0
tn = 0
fp = 0
fn = 0
predictn = predict.flatten().round()
predictn = predictn.tolist()
Y_testn = Y_test.tolist()
for i in range(len(Y_testn)):
  if predictn[i]==1 and Y_testn[i]==1:
    tp+=1
  elif predictn[i]==0 and Y_testn[i]==0:
    tn+=1
  elif predictn[i]==0 and Y_testn[i]==1:
    fp+=1
  elif predictn[i]==1 and Y_testn[i]==0:
    fn+=1
to_heat_map =[[tn,fp],[fn,tp]]
precision_m =  tp / (tp + fp)
recall_m = tp / (tp + fn)
fscore_m = (2 * precision_m * recall_m) / (precision_m + recall_m)
false_positive_rate_m = fp / (fp + tn)
false_negative_rate_m = fn / (tp + fn)
true_positive_rate_m = recall_m
specifity_m = 1 - (tn / (tn+fp))
lift_m = (tp/(tp+fn)) / ((tp+fp) / (tp+fn+fp+tn))
print("TP:\t" + str(tp) + "\r\nTN:\t" + str(tn) + "\r\nFP:\t" + str(fp) + "\r\nFN:\t" + str(fn) + "\r\n")
print("Precision:\t\t\t\t" + str(precision_m) + "\r\n" + 
      "Recall:\t\t\t\t\t" + str(recall_m) + "\r\n" + 
      "fScore:\t\t\t\t\t" + str(fscore_m) + "\r\n" + 
      "False Positive Rate:\t" + str(false_positive_rate_m) + "\r\n" +
      "False Negative Rate:\t" + str(false_negative_rate_m) + "\r\n" +
      "True Positive Rate:\t\t" + str(true_positive_rate_m) + "\r\n" +
      "Specifity Rate:\t\t\t" + str(specifity_m) + "\r\n" +
      "Lift:\t\t\t\t\t" + str(lift_m) + "\r\n")
fpr0, tpr0, thresholds = roc_curve(Y_testn, predictn, pos_label=1.0)

print ("########## LSTM")
print (str(fpr0) + " " + str(tpr0))
print ("##########")

to_heat_map = pandas.DataFrame(to_heat_map, index = ["Attack","Normal"],columns = ["Attack","Normal"])
ax = sns.heatmap(to_heat_map,annot=True, fmt="d")
plt.figure()





##################
# Attention Method
##################
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

class Attention(Layer):
    
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention,self).__init__()
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(Attention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)
    
def create_attention_baseline():
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Attention(return_sequences=True)) # receive 3D and output 3D
    model.add(LSTM(64))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile('adam', 'binary_crossentropy')
    return model

    

# load the dataset
dataframe = pandas.read_csv(file, engine='python')

# Purify
# attack,category,subcategory,AttackProfile
dataframe = removenulls(dataframe)
if ("Full" in file):
    dataframe = removecolumn(dataframe, ['pkSeqID','stime','ltime','category','subcategory','AttackProfile','flgs','proto','state','saddr','daddr'])
    dataframe = makeintegral(dataframe, ['sport','dport','pkts','bytes','spkts','dpkts','sbytes','dbytes'])
elif ("Probable" in file):
    dataframe = removecolumn(dataframe, ['pkSeqID','stime','ltime','category','subcategory','AttackProfile','flgs','proto','state','saddr','daddr'])
    dataframe = makeintegral(dataframe, ['sport','dport','pkts','bytes','spkts','dpkts','sbytes','dbytes', 'ProbableAttack'])
else:
    dataframe = removecolumn(dataframe, ['pkSeqID','stime','ltime','category','subcategory','flgs','proto','state','saddr','daddr'])
    dataframe = makeintegral(dataframe, ['sport','dport','pkts','bytes','spkts','dpkts','sbytes','dbytes'])
    
dataframe = makefloatical(dataframe, ['seq','dur','mean','stddev','sum','min','max','rate','srate','drate'])

Y = dataframe.pop('attack')
#Attack is 0 or 1 by default
X = dataframe

# Standardise the data
scalar = StandardScaler(copy=True, with_mean=True, with_std=True)

scalar.fit(X)
X = scalar.transform(X)

# Shape the data
features = len(X[0])
samples = X.shape[0]
train_len = 25
input_len = samples - train_len
I = numpy.zeros((samples - train_len, train_len, features))

for i in range(input_len):
    temp = numpy.zeros((train_len, features))
    for j in range(i, i + train_len - 1):
        temp[j-i] = X[j]
    I[i] = temp

X.shape


X_train, X_test, Y_train, Y_test = train_test_split(I, Y[25:100000], test_size = 0.2)
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

model = create_attention_baseline()
history = model.fit(X_train, Y_train, epochs = epoc_count,validation_split=0.2, verbose = verbos_level)

predict = model.predict(X_test, verbose=verbos_level)

tp = 0
tn = 0
fp = 0
fn = 0
predictn = predict.flatten().round()
predictn = predictn.tolist()
Y_testn = Y_test.tolist()
for i in range(len(Y_testn)):
  if predictn[i]==1 and Y_testn[i]==1:
    tp+=1
  elif predictn[i]==0 and Y_testn[i]==0:
    tn+=1
  elif predictn[i]==0 and Y_testn[i]==1:
    fp+=1
  elif predictn[i]==1 and Y_testn[i]==0:
    fn+=1

to_heat_map =[[tn,fp],[fn,tp]]
precision_m =  tp / (tp + fp)
recall_m = tp / (tp + fn)
fscore_m = (2 * precision_m * recall_m) / (precision_m + recall_m)
false_positive_rate_m = fp / (fp + tn)
false_negative_rate_m = fn / (tp + fn)
true_positive_rate_m = recall_m
specifity_m = 1 - (tn / (tn+fp))
lift_m = (tp/(tp+fn)) / ((tp+fp) / (tp+fn+fp+tn))

print("TP:\t" + str(tp) + "\r\nTN:\t" + str(tn) + "\r\nFP:\t" + str(fp) + "\r\nFN:\t" + str(fn) + "\r\n")
print("Precision:\t\t\t\t" + str(precision_m) + "\r\n" + 
      "Recall:\t\t\t\t\t" + str(recall_m) + "\r\n" + 
      "fScore:\t\t\t\t\t" + str(fscore_m) + "\r\n" + 
      "False Positive Rate:\t" + str(false_positive_rate_m) + "\r\n" +
      "False Negative Rate:\t" + str(false_negative_rate_m) + "\r\n" +
      "True Positive Rate:\t\t" + str(true_positive_rate_m) + "\r\n" +
      "Specifity Rate:\t\t\t" + str(specifity_m) + "\r\n" +
      "Lift:\t\t\t\t\t" + str(lift_m) + "\r\n")
fpr15, tpr15, thresholds = roc_curve(Y_testn, predictn, pos_label=1.0)

print ("########## LSTM + Att")
print (str(fpr15) + " " + str(tpr15))
print ("##########")

to_heat_map = pandas.DataFrame(to_heat_map, index = ["Attack","Normal"],columns = ["Attack","Normal"])
ax = sns.heatmap(to_heat_map,annot=True, fmt="d")
plt.figure()







#################
# Classic Methods
#################
def create_LogisticRegression():
    model = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
    return model

def create_svm():
    model = svm.LinearSVC()
    return model

def create_RandomForest():
    model = RandomForestClassifier(n_estimators=5, max_depth=1, random_state=1)
    return model

def create_NaiveBayes():
    model = GaussianNB()
    return model

# load the dataset
dataframe = pandas.read_csv(file, engine='python')

dataframe = removenulls(dataframe)
if ("Full" in file):
    dataframe = removecolumn(dataframe, ['pkSeqID','stime','ltime','category','subcategory','AttackProfile','flgs','proto','state','saddr','daddr'])
    dataframe = makeintegral(dataframe, ['sport','dport','pkts','bytes','spkts','dpkts','sbytes','dbytes'])
elif ("Probable" in file):
    dataframe = removecolumn(dataframe, ['pkSeqID','stime','ltime','category','subcategory','AttackProfile','flgs','proto','state','saddr','daddr'])
    dataframe = makeintegral(dataframe, ['sport','dport','pkts','bytes','spkts','dpkts','sbytes','dbytes', 'ProbableAttack'])
else:
    dataframe = removecolumn(dataframe, ['pkSeqID','stime','ltime','category','subcategory','flgs','proto','state','saddr','daddr'])
    dataframe = makeintegral(dataframe, ['sport','dport','pkts','bytes','spkts','dpkts','sbytes','dbytes'])
    
dataframe = makefloatical(dataframe, ['seq','dur','mean','stddev','sum','min','max','rate','srate','drate'])

Y = dataframe.pop('attack')
#Attack is 0 or 1 by default
X = dataframe

# Standardise the data
scalar = StandardScaler(copy=True, with_mean=True, with_std=True)

scalar.fit(X)
X = scalar.transform(X)

# Shape the data
features = len(X[0])
samples = X.shape[0]
train_len = 25
input_len = samples - train_len
I = numpy.zeros((samples - train_len, train_len, features))

for i in range(input_len):
    temp = numpy.zeros((train_len, features))
    for j in range(i, i + train_len - 1):
        temp[j-i] = X[j]
    I[i] = temp

X.shape


#X_train, X_test, Y_train, Y_test = train_test_split(I, Y[25:100000], test_size = 0.2)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,shuffle=True)

# Naive Bayes
model = create_NaiveBayes()
history = model.fit(X_train, Y_train)
predict = model.predict(X_test)
tp = 0
tn = 0
fp = 0
fn = 0
predictn = predict.flatten().round()
predictn = predictn.tolist()
Y_testn = Y_test.tolist()
for i in range(len(Y_testn)):
  if predictn[i]==1 and Y_testn[i]==1:
    tp+=1
  elif predictn[i]==0 and Y_testn[i]==0:
    tn+=1
  elif predictn[i]==0 and Y_testn[i]==1:
    fp+=1
  elif predictn[i]==1 and Y_testn[i]==0:
    fn+=1
to_heat_map =[[tn,fp],[fn,tp]]
precision_m =  tp / (tp + fp)
recall_m = tp / (tp + fn)
fscore_m = (2 * precision_m * recall_m) / (precision_m + recall_m)
false_positive_rate_m = fp / (fp + tn)
false_negative_rate_m = fn / (tp + fn)
true_positive_rate_m = recall_m
specifity_m = 1 - (tn / (tn+fp))
lift_m = (tp/(tp+fn)) / ((tp+fp) / (tp+fn+fp+tn))
print("TP:\t" + str(tp) + "\r\nTN:\t" + str(tn) + "\r\nFP:\t" + str(fp) + "\r\nFN:\t" + str(fn) + "\r\n")
print("Precision:\t\t\t\t" + str(precision_m) + "\r\n" + 
      "Recall:\t\t\t\t\t" + str(recall_m) + "\r\n" + 
      "fScore:\t\t\t\t\t" + str(fscore_m) + "\r\n" + 
      "False Positive Rate:\t" + str(false_positive_rate_m) + "\r\n" +
      "False Negative Rate:\t" + str(false_negative_rate_m) + "\r\n" +
      "True Positive Rate:\t\t" + str(true_positive_rate_m) + "\r\n" +
      "Specifity Rate:\t\t\t" + str(specifity_m) + "\r\n" +
      "Lift:\t\t\t\t\t" + str(lift_m) + "\r\n")
fpr1, tpr1, thresholds = roc_curve(Y_testn, predictn, pos_label=1.0)

print ("########## NB")
print (str(fpr1) + " " + str(tpr1))
print ("##########")

to_heat_map = pandas.DataFrame(to_heat_map, index = ["Attack","Normal"],columns = ["Attack","Normal"])
ax = sns.heatmap(to_heat_map,annot=True, fmt="d")
log_loss(Y_testn, predictn, eps=1e-15)
plt.figure()

# Random Forest
model = create_RandomForest()
history = model.fit(X_train, Y_train)
predict = model.predict(X_test)
tp = 0
tn = 0
fp = 0
fn = 0
predictn = predict.flatten().round()
predictn = predictn.tolist()
Y_testn = Y_test.tolist()
for i in range(len(Y_testn)):
  if predictn[i]==1 and Y_testn[i]==1:
    tp+=1
  elif predictn[i]==0 and Y_testn[i]==0:
    tn+=1
  elif predictn[i]==0 and Y_testn[i]==1:
    fp+=1
  elif predictn[i]==1 and Y_testn[i]==0:
    fn+=1
to_heat_map =[[tn,fp],[fn,tp]]
precision_m =  tp / (tp + fp)
recall_m = tp / (tp + fn)
fscore_m = (2 * precision_m * recall_m) / (precision_m + recall_m)
false_positive_rate_m = fp / (fp + tn)
false_negative_rate_m = fn / (tp + fn)
true_positive_rate_m = recall_m
specifity_m = 1 - (tn / (tn+fp))
lift_m = (tp/(tp+fn)) / ((tp+fp) / (tp+fn+fp+tn))
print("TP:\t" + str(tp) + "\r\nTN:\t" + str(tn) + "\r\nFP:\t" + str(fp) + "\r\nFN:\t" + str(fn) + "\r\n")
print("Precision:\t\t\t\t" + str(precision_m) + "\r\n" + 
      "Recall:\t\t\t\t\t" + str(recall_m) + "\r\n" + 
      "fScore:\t\t\t\t\t" + str(fscore_m) + "\r\n" + 
      "False Positive Rate:\t" + str(false_positive_rate_m) + "\r\n" +
      "False Negative Rate:\t" + str(false_negative_rate_m) + "\r\n" +
      "True Positive Rate:\t\t" + str(true_positive_rate_m) + "\r\n" +
      "Specifity Rate:\t\t\t" + str(specifity_m) + "\r\n" +
      "Lift:\t\t\t\t\t" + str(lift_m) + "\r\n")
fpr2, tpr2, thresholds = roc_curve(Y_testn, predictn, pos_label=1.0)

print ("########## RF")
print (str(fpr2) + " " + str(tpr2))
print ("##########")

to_heat_map = pandas.DataFrame(to_heat_map, index = ["Attack","Normal"],columns = ["Attack","Normal"])
ax = sns.heatmap(to_heat_map,annot=True, fmt="d")
log_loss(Y_testn, predictn, eps=1e-15)
plt.figure()

# SVM
model = create_svm()
history = model.fit(X_train, Y_train)
predict = model.predict(X_test)
tp = 0
tn = 0
fp = 0
fn = 0
predictn = predict.flatten().round()
predictn = predictn.tolist()
Y_testn = Y_test.tolist()
for i in range(len(Y_testn)):
  if predictn[i]==1 and Y_testn[i]==1:
    tp+=1
  elif predictn[i]==0 and Y_testn[i]==0:
    tn+=1
  elif predictn[i]==0 and Y_testn[i]==1:
    fp+=1
  elif predictn[i]==1 and Y_testn[i]==0:
    fn+=1
to_heat_map =[[tn,fp],[fn,tp]]
precision_m =  tp / (tp + fp)
recall_m = tp / (tp + fn)
fscore_m = (2 * precision_m * recall_m) / (precision_m + recall_m)
false_positive_rate_m = fp / (fp + tn)
false_negative_rate_m = fn / (tp + fn)
true_positive_rate_m = recall_m
specifity_m = 1 - (tn / (tn+fp))
lift_m = (tp/(tp+fn)) / ((tp+fp) / (tp+fn+fp+tn))
print("TP:\t" + str(tp) + "\r\nTN:\t" + str(tn) + "\r\nFP:\t" + str(fp) + "\r\nFN:\t" + str(fn) + "\r\n")
print("Precision:\t\t\t\t" + str(precision_m) + "\r\n" + 
      "Recall:\t\t\t\t\t" + str(recall_m) + "\r\n" + 
      "fScore:\t\t\t\t\t" + str(fscore_m) + "\r\n" + 
      "False Positive Rate:\t" + str(false_positive_rate_m) + "\r\n" +
      "False Negative Rate:\t" + str(false_negative_rate_m) + "\r\n" +
      "True Positive Rate:\t\t" + str(true_positive_rate_m) + "\r\n" +
      "Specifity Rate:\t\t\t" + str(specifity_m) + "\r\n" +
      "Lift:\t\t\t\t\t" + str(lift_m) + "\r\n")
fpr3, tpr3, thresholds = roc_curve(Y_testn, predictn, pos_label=1.0)

print ("########## SVM")
print (str(fpr3) + " " + str(tpr3))
print ("##########")

to_heat_map = pandas.DataFrame(to_heat_map, index = ["Attack","Normal"],columns = ["Attack","Normal"])
ax = sns.heatmap(to_heat_map,annot=True, fmt="d")
log_loss(Y_testn, predictn, eps=1e-15)
plt.figure()

# Logistic Regression
model = create_LogisticRegression()
history = model.fit(X_train, Y_train)
predict = model.predict(X_test)
tp = 0
tn = 0
fp = 0
fn = 0
predictn = predict.flatten().round()
predictn = predictn.tolist()
Y_testn = Y_test.tolist()
for i in range(len(Y_testn)):
  if predictn[i]==1 and Y_testn[i]==1:
    tp+=1
  elif predictn[i]==0 and Y_testn[i]==0:
    tn+=1
  elif predictn[i]==0 and Y_testn[i]==1:
    fp+=1
  elif predictn[i]==1 and Y_testn[i]==0:
    fn+=1
to_heat_map =[[tn,fp],[fn,tp]]
precision_m =  tp / (tp + fp)
recall_m = tp / (tp + fn)
fscore_m = (2 * precision_m * recall_m) / (precision_m + recall_m)
false_positive_rate_m = fp / (fp + tn)
false_negative_rate_m = fn / (tp + fn)
true_positive_rate_m = recall_m
specifity_m = 1 - (tn / (tn+fp))
lift_m = (tp/(tp+fn)) / ((tp+fp) / (tp+fn+fp+tn))
print("TP:\t" + str(tp) + "\r\nTN:\t" + str(tn) + "\r\nFP:\t" + str(fp) + "\r\nFN:\t" + str(fn) + "\r\n")
print("Precision:\t\t\t\t" + str(precision_m) + "\r\n" + 
      "Recall:\t\t\t\t\t" + str(recall_m) + "\r\n" + 
      "fScore:\t\t\t\t\t" + str(fscore_m) + "\r\n" + 
      "False Positive Rate:\t" + str(false_positive_rate_m) + "\r\n" +
      "False Negative Rate:\t" + str(false_negative_rate_m) + "\r\n" +
      "True Positive Rate:\t\t" + str(true_positive_rate_m) + "\r\n" +
      "Specifity Rate:\t\t\t" + str(specifity_m) + "\r\n" +
      "Lift:\t\t\t\t\t" + str(lift_m) + "\r\n")
fpr4, tpr4, thresholds = roc_curve(Y_testn, predictn, pos_label=1.0)

print ("########## LR")
print (str(fpr4) + " " + str(tpr4))
print ("##########")

to_heat_map = pandas.DataFrame(to_heat_map, index = ["Attack","Normal"],columns = ["Attack","Normal"])
ax = sns.heatmap(to_heat_map,annot=True, fmt="d")
log_loss(Y_testn, predictn, eps=1e-15)

plt.figure()
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr0, tpr0, label= "DL")
plt.plot(fpr15, tpr15, label= "Att")
plt.plot(fpr1, tpr1, label= "NB")
plt.plot(fpr2, tpr2, label= "RF")
plt.plot(fpr3, tpr3, label= "SVM")
plt.plot(fpr4, tpr4, label= "LR")
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title('Receiver Operating Characteristic')
plt.figure()