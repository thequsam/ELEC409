# ELEC 409 Final Assignment
# Sam Johnston 
# Queen's University
# 2019

import pandas as pd
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy import stats

# Load data
fileName = '/Users/Sam/Documents/Devolopment/ELEC409/Dataset_C_MD_outcome2.csv'
data = pd.read_csv(fileName, header=None, skiprows=3)
data_X = np.array(data)
data_X = data_X.reshape(62,7129)

# Create the array of class labels
# 1 for non-responders and zero for responders
Y_1 = np.ones(21)
Y_0 = np.zeros(39)
data_Y = np.append(Y_1, Y_0)
#data_Y = data_Y.reshape(60,1)

x_train, x_test, y_train, y_test = train_test_split(data_X[2:62,:], data_Y, test_size=0.5, random_state=42)




