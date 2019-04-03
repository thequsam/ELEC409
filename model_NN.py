# ELEC 409 Final Assignment
# Building an NN for classification
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

import tensorflow as tf
import os  
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.layers import Dense, Flatten, Dropout, Reshape
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras import backend as K
from keras.utils import np_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

# Declare the number of samples used
num_samples = 1000

# Create the testing and training sets split in half
# Changing the random_state to a different number changes the random seed and therefore 
# changes how the data is split into train and test sets
data_X = SelectKBest(chi2, k=num_samples).fit_transform(data_X_norm, data_Y)

x_train, x_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.5, random_state=10)
samples_train, genes_train = x_train.shape
samples_test, genes_test = x_test.shape

# Train a-NN
# Model parameters
epochs = 100
batch_size = 1
verbose = 1

def build_model():
    model = Sequential()
    model.add(Dense(200, activation='relu', input_dim=genes_train))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

num_itr = 100

# Intialize range of random state and zero array's for p value and Matthews correlation coefficient
j_range = range(1,num_itr)
p_value = np.zeros(num_itr)
m_coeff = np.zeros(num_itr)

# Intialize arrays to count number of responders in test and train sets
resp_count_test = np.zeros(num_itr)
resp_count_train = np.zeros(num_itr)

for j in j_range:

    model = build_model()

    # Keep only values of statistic significance 
    data_X = SelectKBest(chi2, k=num_samples).fit_transform(data_X_norm, data_Y)

    # Create the testing and training sets split in half
    # Changing the random_state to a different number changes the random seed and therefore 
    # changes how the data is split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.5, random_state=j)

    # Count number of responders in test and train sets
    resp_count_test[j] = sum(y_test)
    resp_count_train[j] = sum(y_train)

    if resp_count_train[j] != (10 or 11):
        continue

        # Fit model
    model.fit(
        x_train, 
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    # Evaluate model
    _, accuracy = model.evaluate(
        x_test, 
        y_test, 
        batch_size=batch_size, 
        verbose=verbose
    )

    # Save model in HDF5 file
    # model.save('ELEC490.h5')

    print('Accuracy of model:')
    print(accuracy*100.0)
    print('')

    # x_test = x_test.swapaxes(1,0)
    # print(x_test.shape)
    # #Predict probability of each class

    y_pred = model.predict_classes(x_test, batch_size=batch_size)

    # Compute contengency matrix, Fischers exact test ant Matthew's correlation coeffecient
    contingency_matrix = metrics.confusion_matrix(y_test,y_pred)
    odds_ratio, p_value[j] = stats.fisher_exact(contingency_matrix)
    m_coeff[j] = metrics.matthews_corrcoef(y_test,y_pred)


p_value_z = []
m_coeff_z = []

for j in j_range:
    if p_value[j] == 0:
        continue
    else:
        p_value_z.append(p_value[j])
        m_coeff_z.append(m_coeff[j])

# Plot the p value and Matthews correlation coefficent changes with increasing number of gene samples
plt.subplot(2, 1, 1)
plt.plot(p_value_z, '.-')
plt.title('Measure of classification with varying random state')
plt.ylabel('P value')

plt.subplot(2, 1, 2)
plt.plot(m_coeff_z, '.-')
plt.ylabel('Matthews correlation coefficient')
plt.xlabel('Random state value')

plt.subplots_adjust(hspace=0.3)
plt.show()
