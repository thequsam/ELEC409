# ELEC 409 Final Assignment
# Building an RNN for classification
# Sam Johnston 
# Queen's University
# 2019

import pandas as pd
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data
fileName = '/Users/Sam/Documents/Devolopment/ELEC409/Dataset_C_MD_outcome2.csv'
data = pd.read_csv(fileName, usecols=[*range(2, 62)], header=None, skiprows=3)
data_X = np.array(data)
data_X = data_X.reshape(60,7129)

# Create the array of class labels
# 1 for non-responders and zero for responders
Y_1 = np.ones(21)
Y_0 = np.zeros(39)
data_Y = np.append(Y_1, Y_0)

# Normalize the data 
# scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler = MinMaxScaler(feature_range=(0,1))
data_X_norm = scaler.fit_transform(data_X)

# Declare optimal values for K and the number of samples used
k_value = 2
num_samples = 78

# Keep only values of statistic significance 
#data_X = SelectKBest(chi2, k=num_samples).fit_transform(data_X_norm, data_Y)

# Create the testing and training sets split in half
# Changing the random_state to a different number changes the random seed and therefore 
# changes how the data is split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.5, random_state=8)

# Train RNN


# Compute contengency matrix, Fischers exact test ant Matthew's correlation coeffecient
contingency_matrix = metrics.confusion_matrix(y_test,y_pred)
odds_ratio, p_value = stats.fisher_exact(contingency_matrix)
m_coeff = metrics.matthews_corrcoef(y_test,y_pred)

print(p_value)
print(m_coeff)

