# -*- coding: utf-8 -*-
"""
Python script for machine learning and deep learning applied to 
breast cancer prediction
Version: December 31, 2022
@author: Rolando Gonzales Martinez
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
# from sklearn.model_selection import StratifiedKFold
#import numpy as np
import pandas as pd

#%% data source route
db_source = '...\\data\\'
#db_source = 'Z:\\UpWork\\UpWork2023\\Daan\\df\\'


#%% Loading data
df = pd.read_csv(db_source + 'dfz_selected.csv')
df.drop(['Unnamed: 0'], axis = 1, inplace = True)
df.columns

# Defining target (dependent) variable
df.rename(columns={'breastcancer': 'y'}, inplace=True) 

X = df.drop('y', axis=1).values
y = df.y.values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
y_train

#%%############################################################################
#################   P a r t  II : D e e p   l e a r n i n g   #################
###############################################################################
# Importing DL libraries and regularizers
import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
'''
conda install hdf5
'''
import theano
print('theano: %s' % theano.__version__)
# tensorflow
# import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import tensorflow
print('tensorflow: %s' % tensorflow.__version__)
# keras
import keras
print('keras: %s' % keras.__version__)
# Deep learning libraries:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

#%% Estimating DL models with K-fold Monte Carlo 
from sklearn.model_selection import KFold
os.getcwd()
os.chdir('Z:\\UpWork\\UpWork2023\\Daan\\output')
train_per = 0.75
simulations = 1000

# Deep learning functions:
def evaluate_DLmodel(cv):
    # DL model with principal components:
    number_of_predictors = X.shape[1]
    DLmodel = Sequential()
    DLmodel.add(Dense(100, input_shape=(number_of_predictors,), activation='relu'))
    DLmodel.add(Dense(100, activation='relu',
                      kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                      bias_regularizer=regularizers.L2(1e-4),
                      activity_regularizer=regularizers.L2(.666)))
    DLmodel.add(Dense(100, activation='relu',activity_regularizer=regularizers.L2(.666)))
    DLmodel.add(Dense(100, activation='relu',activity_regularizer=regularizers.L2(.666)))
    DLmodel.add(Dense(1, activation='sigmoid')) # output layer
    DLmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    DLmodel.fit(X, y, validation_split=(1-train_per), epochs=300, batch_size=10, verbose=0)
    performance = DLmodel.evaluate(X, y)
    accuracy = performance[1]*100
    return accuracy

# ------------------ Monte Carlo k-flod cross-validation ----------------------
# k-fold monte carlo cross-validation DL model 1:
M0per_DLcvmc =[] # accuracy of DL, k-fold monte carlo cross-validation 
for sim in range(simulations): 
    print('M0 simulation: ',sim+1)
    # define the test condition
    cv = KFold(n_splits=10, shuffle=True)
    # evaluate k value
    accuracy_DL = evaluate_DLmodel(cv)
    M0per_DLcvmc.append(accuracy_DL)
M0res = pd.DataFrame(M0per_DLcvmc)
M0res.columns= ["accuracy"]
M0res.to_csv('Results_DL_tensor.csv')

########################## end of script #####################################
