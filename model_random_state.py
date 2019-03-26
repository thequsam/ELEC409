# ELEC 409 Final Assignment
# Testing selected parameters
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

# Intialize range of random state and zero array's for p value and Matthews correlation coefficient
j_range = range(1,100)
p_value = np.zeros(100)
m_coeff = np.zeros(100)

# Intialize arrays to count number of responders in test and train sets
resp_count_test = np.zeros(100)
resp_count_train = np.zeros(100)

for j in j_range:
    # Keep only values of statistic significance 
    data_X = SelectKBest(chi2, k=num_samples).fit_transform(data_X_norm, data_Y)

    # Create the testing and training sets split in half
    # Changing the random_state to a different number changes the random seed and therefore 
    # changes how the data is split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.5, random_state=j)

    # Count number of responders in test and train sets
    resp_count_test[j] = sum(y_test)
    resp_count_train[j] = sum(y_train)

    # Train KNN on selected max k value
    knn = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    score_max = metrics.accuracy_score(y_test,y_pred)

    # Compute contengency matrix, Fischers exact test ant Matthew's correlation coeffecient
    contingency_matrix = metrics.confusion_matrix(y_test,y_pred)
    odds_ratio, p_value[j] = stats.fisher_exact(contingency_matrix)
    m_coeff[j] = metrics.matthews_corrcoef(y_test,y_pred)


# Plot the p value and Matthews correlation coefficent changes with increasing number of gene samples
plt.subplot(3, 1, 1)
plt.plot(p_value, '.-')
plt.title('Measure of classification with varying gene sample size')
plt.ylabel('P value')

plt.subplot(3, 1, 2)
plt.plot(m_coeff, '.-')
plt.ylabel('Matthews correlation coefficient')

plt.subplot(3, 1, 3)
plt.plot(resp_count_test, 'g.-', label='test set')
plt.plot(resp_count_train, 'r.-', label='train set')
plt.title('Number of non-responders in train and test set')
plt.ylabel('Non-responders')
plt.xlabel('Random state input')

plt.subplots_adjust(hspace=0.3)
plt.show()