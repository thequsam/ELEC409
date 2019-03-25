# ELEC 409 Final Assignment
# Sam Johnston 
# Queen's University
# 2019

import pandas as pd
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from scipy import stats
from sklearn.decomposition import PCA


# Load data
fileName = '/Users/Sam/Documents/Devolopment/ELEC409/Dataset_C_MD_outcome2.csv'
data = pd.read_csv(fileName, usecols=[*range(2, 62)], header=None, skiprows=3)
data_X = np.array(data)
data_X = data_X.reshape(60,7129)

# Principal component analysis
# pca = PCA(n_components=2)
# data_X = pca.fit_transform(data_X)

# Create the array of class labels
# 1 for non-responders and zero for responders
Y_1 = np.ones(21)
Y_0 = np.zeros(39)
data_Y = np.append(Y_1, Y_0)

# Create the testing and training sets split in half
x_train, x_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.5, random_state=8)

# Normalize the data 
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
x_train = scaler.fit_transform(x_train) 
x_test = scaler.fit_transform(x_test) 

# Train KNN
k_range = range(1,26)
scores = {}
scores_list = []
max_score = np.zeros([30,26])

loo = LeaveOneOut()
loo.get_n_splits(x_train)

for train_index, test_index in loo.split(x_train):
    x_train_loo, x_test_loo = x_train[train_index], x_train[test_index]
    y_train_loo, y_test_loo = y_train[train_index], y_train[test_index]

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train_loo,y_train_loo)
        y_pred = knn.predict(x_test_loo)
        scores[k] = metrics.accuracy_score(y_test_loo,y_pred)
        scores_list.append(metrics.accuracy_score(y_test_loo,y_pred))
        max_score[test_index,k] = scores[k]
   
    scores = {}
    scores_list = []
    
max_k = np.sum(max_score, axis=0)
max_k_value = np.argmax(max_k)+1

knn = KNeighborsClassifier(n_neighbors=max_k_value)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
score_max = metrics.accuracy_score(y_test,y_pred)

print(score_max)


