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

# Preform principle component analysis
pca = PCA(n_components=2, svd_solver='full')
PCA_X = pca.fit_transform(data_X_norm) 

j_range = range(50,200)
p_value = np.zeros(150)
m_coeff = np.zeros(150)

for j in j_range:
    # Keep only values of statistic significance 
    data_X = SelectKBest(chi2, k=j).fit_transform(data_X_norm, data_Y)

    # Create the testing and training sets split in half
    # Changing the random_state to a different number changes the random seed and therefore 
    # changes how the data is split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.5, random_state=8)

    # Intialize array's for KNN scores and k range
    k_range = range(1,26)
    scores = {}
    scores_list = []
    max_score = np.zeros([30,26])

    # Get leave on out splits
    loo = LeaveOneOut()
    loo.get_n_splits(x_train)

    # Loop through all leave one out splits
    for train_index, test_index in loo.split(x_train):
        x_train_loo, x_test_loo = x_train[train_index], x_train[test_index]
        y_train_loo, y_test_loo = y_train[train_index], y_train[test_index]

        # Loop through k value 1 through 26
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
            knn.fit(x_train_loo,y_train_loo)
            y_pred = knn.predict(x_test_loo)
            scores[k] = metrics.accuracy_score(y_test_loo,y_pred)
            scores_list.append(metrics.accuracy_score(y_test_loo,y_pred))
            max_score[test_index,k] = scores[k]
    
        scores = {}
        scores_list = []
        
    # Save k value that maximizes score
    max_k = np.sum(max_score, axis=0)
    max_k_value = np.argmax(max_k)+1

    # Train KNN on selected max k value
    knn = KNeighborsClassifier(n_neighbors=max_k_value, weights='distance')
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    score_max = metrics.accuracy_score(y_test,y_pred)

    # Compute contengency matrix, Fischers exact test ant Matthew's correlation coeffecient
    contingency_matrix = metrics.cluster.contingency_matrix(y_test,y_pred)
    odds_ratio, p_value[j-50] = stats.fisher_exact(contingency_matrix)
    m_coeff[j-50] = metrics.matthews_corrcoef(y_test,y_pred)

    print(j)

p_value_max = (np.argmin(p_value))+51
m_coeff_max = (np.argmax(m_coeff))+51

print(p_value_max)
print(m_coeff_max)
print(p_value[(p_value_max)-51])
print(m_coeff[(m_coeff_max)-51])
print(max_k_value)

# Plot the p value and Matthews correlation coefficent changes with increasing number of gene samples
fig1 = plt.figure()
fig2 = plt.figure()

ax1 = fig1.add_subplot(2, 1, 1)
ax1.plot(j_range, p_value, '.-')
ax1.set_title('Measure of classification with varying gene sample size')
ax1.set_ylabel('P value')

ax2 = fig1.add_subplot(2, 1, 2)
ax2.plot(j_range, m_coeff, '.-')
ax2.set_xlabel('Number of Genes used')
ax2.set_ylabel('Matthews correlation coefficient')

ax3 = fig2.add_subplot(1, 1, 1)
ax3.scatter(PCA_X[:,0],PCA_X[:,1])
ax3.set_title('PCA')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')

plt.show()
