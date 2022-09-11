# -*- coding: utf-8 -*-
"""
Python script for deep learning
Version: Aug 30, 2022
@author: Rolando Gonzales Martinez
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
import pandas as pd
import featurewiz as FW
import numpy as np

#%% data source route
db_source = 'Z:/GitHub/Deeplearning_breastcancer/data/'

#%%############################################################################
############## Part I: Feature selection and feature engineering ##############
###############################################################################
# Loading data and z-standarization
df = pd.read_csv(db_source + 'dataR2.csv')
# Below I make the target variable binary (recode 1=cancer, 0=healthy)
df['Classification'] = df['Classification'].replace(1,0)
df['Classification'] = df['Classification'].replace(2,1)
df.rename(columns={"Classification": "breastcancer"}, inplace=True)

# z-Standarization:
from sklearn.preprocessing import StandardScaler
dfz = pd.DataFrame(StandardScaler().fit_transform(df),columns=df.columns)
dfz.breastcancer = df.breastcancer

# Data Set Information:
    # There are 10 predictors, all quantitative, and a binary dependent variable, indicating the presence or absence of breast cancer.
    # The predictors are anthropometric data and parameters which can be gathered in routine blood analysis.
    # Prediction models based on these predictors, if accurate, can potentially be used as a biomarker of breast cancer.
    # --- Attribute Information ---
    # Quantitative Attributes:
    # Age (years)
    # BMI (kg/m2)
    # Glucose (mg/dL)
    # Insulin (µU/mL)
    # HOMA
    # Leptin (ng/mL)
    # Adiponectin (µg/mL)
    # Resistin (ng/mL)
    # MCP-1(pg/dL)
    #---------------------------
    # Labels:
    # 1=Healthy controls
    # 2=Patients
    # https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra

#%% Feature selection
# Recursive loop to choose a correlation threshold:
# features = []
# for tau in range(1, 99, 1):
#     f, fdf = FW.featurewiz(dfz,'breastcancer', corr_limit=tau/100, verbose=0,  
#         header=0, test_data='',feature_engg='', category_encoders='',
#         dask_xgboost_flag=False)
#     features.append(f)
# fr = pd.DataFrame(features) 
# #saving the dataframe 
# fr.to_csv('features.csv') 

# Subset of relevant variables:
tau  = 0.30 # correlation threshold, variables v between -tau < v < tau will be removed

fs, fds = FW.featurewiz(dfz,'breastcancer', corr_limit=tau, verbose=2, sep=',', 
        header=0, test_data='',feature_engg='', category_encoders='',
        dask_xgboost_flag=False)
#['Glucose', 'Age', 'Resistin', 'Adiponectin']

# Dropping age to avoid non-sensical interactions in feature engineering:
fds.drop(['Age'], axis=1, inplace = True)

# Feature engineering:
fsi, fdi = FW.featurewiz(fds,'breastcancer', corr_limit=.5, verbose=2, sep=',', 
        header=0, test_data='',feature_engg='interactions', category_encoders='',
        dask_xgboost_flag=False)

fds['age']= dfz.Age

fdi['age']= dfz.Age
fdi['glucose'] = dfz.Glucose
fdi['resistin'] = dfz.Resistin
fdi['adiponectin'] = dfz.Adiponectin

#%% Keeping only relevant dataframes:
del df, fs, fsi, tau

#%%############################################################################
#################   P a r t  II : D e e p   l e a r n i n g   #################
###############################################################################
# Importing DL libraries and regularizers
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

#%% Data preparation for DL:
fde = fdi
dfz  = dfz.to_numpy()
fds = fds.to_numpy()
fdi = fdi.to_numpy()
Xall = dfz[:,0:9]
Xsfs = fds[:,[0,1,2,4]] # Selected features (sfs)
Xsef = fdi[:,[0,1,2,3,4,5,7,8,9,10]] # Selected and engineered features (efs)
Xefs = fdi[:,[0,1,2,3,4,5]] # Engineered features (efs)
y = dfz[:,9]
del dfz, fdi, fds, fde

#%% Custom performance metrics:
from keras import backend as K
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# def specificity(y_true, y_pred):
#     true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
#     possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
#     return true_negatives / (possible_negatives + K.epsilon())

def specificity(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity

#%% Estimating base DL model to calibrate the arquitecture
evilseed = 666
np.random.seed(evilseed)
tensorflow.random.set_seed(evilseed)
train_per = 0.75
number_of_predictors = Xall.shape[1]
#------------------------------------------------------------------------------
DLmodel = Sequential()
DLmodel.add(Dense(100, input_shape=(number_of_predictors,), activation='relu'))
DLmodel.add(Dense(100, activation='relu',
                  kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.L2(1e-4),
                  activity_regularizer=regularizers.L2(.666)))
DLmodel.add(Dense(100, activation='relu',activity_regularizer=regularizers.L2(.666)))
DLmodel.add(Dense(100, activation='relu',activity_regularizer=regularizers.L2(.666)))
# DLmodel.add(Dense(100, activation='relu',activity_regularizer=regularizers.L2(.666)))
DLmodel.add(Dense(1, activation='sigmoid')) # output layer
# compile the keras model
DLmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#DLmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', specificity])
#DLmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.SensitivityAtSpecificity(0.5),keras.metrics.SpecificityAtSensitivity(0.5),'accuracy'])
# fit the keras model on the dataset
DLmodel.fit(Xall, y, validation_split=(1-train_per), epochs=200, batch_size=10)
# evaluate the keras model
accuracy = DLmodel.evaluate(Xall, y)
print("%s: %.2f%%" % (DLmodel.metrics_names[1], accuracy[1]*100))
# 89.66%
# better results are obtained using a relu compared to a sigmoid activation function
# regularized hidden units (dense layers) were added until no improvement in accuracy was observed (4 hidden units = worst)
del accuracy, evilseed, number_of_predictors
#%% Estimating DL models with K-fold Monte Carlo 
from sklearn.model_selection import KFold

# Deep learning functions:
def evaluate_DLmodel1(cv):
    # DL model with all predictors:
    number_of_predictors = Xall.shape[1]
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
    DLmodel.fit(Xall, y, validation_split=(1-train_per), epochs=300, batch_size=10, verbose=0)
    performance = DLmodel.evaluate(Xall, y)
    accuracy = performance[1]*100
    return accuracy

def evaluate_DLmodel2(cv):
    # DL model with feature selection:
    number_of_predictors = Xsfs.shape[1]
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
    DLmodel.fit(Xsfs, y, validation_split=(1-train_per), epochs=300, batch_size=10, verbose=0)
    performance = DLmodel.evaluate(Xsfs, y)
    accuracy = performance[1]*100
    return accuracy

def evaluate_DLmodel3(cv):
    # DL model with feature selection:
    number_of_predictors = Xsef.shape[1]
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
    DLmodel.fit(Xsef, y, validation_split=(1-train_per), epochs=300, batch_size=10, verbose=0)
    performance = DLmodel.evaluate(Xsef, y)
    accuracy = performance[1]*100
    return accuracy

def evaluate_DLmodel4(cv):
    # DL model with feature selection:
    number_of_predictors = Xefs.shape[1]
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
    DLmodel.fit(Xefs, y, validation_split=(1-train_per), epochs=300, batch_size=10, verbose=0)
    performance = DLmodel.evaluate(Xefs, y)
    accuracy = performance[1]*100
    return accuracy

simulations = 1000

# k-fold monte carlo cross-validation DL model 1:
M1per_DLcvmc =[] # accuracy of DL, k-fold monte carlo cross-validation 
for sim in range(simulations): 
    print('M1 simulation: ',sim+1)
    # define the test condition
    cv = KFold(n_splits=10, shuffle=True)
    # evaluate k value
    accuracy_DL = evaluate_DLmodel1(cv)
    M1per_DLcvmc.append(accuracy_DL)
M1res = pd.DataFrame(M1per_DLcvmc)
M1res.columns= ["accuracy"]
M1res.to_csv('M1results.csv')

# k-fold monte carlo cross-validation DL model 2:
M2per_DLcvmc =[] # accuracy of DL, k-fold monte carlo cross-validation 
for sim in range(simulations):  
    print('M2 simulation: ',sim+1)
    # define the test condition
    cv = KFold(n_splits=10, shuffle=True)
    # evaluate k value
    accuracy_DL = evaluate_DLmodel2(cv)
    M2per_DLcvmc.append(accuracy_DL)   
M2res = pd.DataFrame(M2per_DLcvmc)
M2res.columns= ["accuracy"]
M2res.to_csv('M2results.csv')

# k-fold monte carlo cross-validation DL model 3:
M3per_DLcvmc =[] # accuracy of DL, k-fold monte carlo cross-validation 
for sim in range(simulations):  
    print('M3 simulation: ',sim+1)
    # define the test condition
    cv = KFold(n_splits=10, shuffle=True)
    # evaluate k value
    accuracy_DL = evaluate_DLmodel3(cv)
    M3per_DLcvmc.append(accuracy_DL)
M3res = pd.DataFrame(M3per_DLcvmc)
M3res.columns= ["accuracy"]
M3res.to_csv('M3results.csv')

# k-fold monte carlo cross-validation DL model 4:
M4per_DLcvmc =[] # accuracy of DL, k-fold monte carlo cross-validation 
for sim in range(simulations):  
    print('M4 simulation: ',sim+1)
    # define the test condition
    cv = KFold(n_splits=10, shuffle=True)
    # evaluate k value
    accuracy_DL = evaluate_DLmodel4(cv)
    M4per_DLcvmc.append(accuracy_DL)
M4res = pd.DataFrame(M4per_DLcvmc)
M4res.columns= ["accuracy"]
M4res.to_csv('M4results.csv')

########################## end of script #####################################
