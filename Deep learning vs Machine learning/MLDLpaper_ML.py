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

#%% Loading data
df = pd.read_csv(db_source + 'dfz_selected.csv')
df.drop(['Unnamed: 0'], axis = 1, inplace = True)
df.columns

# Defining target (dependent) variable
df.rename(columns={'breastcancer': 'y'}, inplace=True) 

X = df.drop('y', axis=1)
y = df.y

#%%############################################################################
##############   P a r t  I : M a c h i n e   l e a r n i n g   ###############
###############################################################################

#%% Machine learning
# Import libraries:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier # Stochastic gradient descending
import xgboost as xgb
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.ensemble import RandomForestClassifier # Random forest
import numpy as np

# Parameters
train_percentage = 0.75  # Percentage of data in the train sample (75%), test: 25%


#%% Defining functions for models (now including household weights):
    
def log_reg(X_train, y_train, X_test, y_test):
    clf = LogisticRegression().fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    return auc

def ranforest(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(max_features=3,n_estimators=10,n_jobs = -1)
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    return auc

def nBayes(X_train, y_train, X_test, y_test):
    naive_bayes = GaussianNB()
    clf = naive_bayes.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    return auc

def nn(X_train, y_train, X_test, y_test):
    # Note: MLPClassifier does not currently supports the use of sample weights
    clf = MLPClassifier(solver='adam',hidden_layer_sizes=(10,10), max_iter = 500)
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    return auc

def sgdclf(X_train, y_train, X_test, y_test):
    clf = SGDClassifier(random_state=666,loss='modified_huber',penalty='l1')
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    return auc

def xgbclf(X_train, y_train, X_test, y_test):
    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    return auc

def svmclf(X_train, y_train, X_test, y_test):
    clf = svm.SVC(probability=True)
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    return auc

#%% Main Spatial Machine Learning function
def main_ML(X, y):
    # X.columns = X.columns.str.translate("".maketrans({"[":"{", "]":"}","<":"^"}))
    
    # This part below splits the sample:
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(1-train_percentage))
    
    # This part below estimates different models:
    y1_auc = log_reg(X_train, y_train, X_test, y_test)
    y2_auc = ranforest(X_train, y_train, X_test, y_test)
    y3_auc = nBayes(X_train, y_train, X_test, y_test)
    y4_auc = nn(X_train, y_train, X_test, y_test)
    y5_auc = sgdclf(X_train, y_train, X_test, y_test)
    y6_auc = xgbclf(X_train, y_train, X_test, y_test)
    y7_auc = svmclf(X_train, y_train, X_test, y_test)
    
    aucs = np.array([y1_auc, y2_auc, y3_auc, y4_auc, y5_auc, y6_auc, y7_auc])
    
    results = pd.DataFrame(aucs)
    results.columns = ['AUC']
    results.index = ['Logistic regression', 'Random forest', 'Naive bayes', 'Neural network', 'Stochastic gradient', 'XGBoost', 'SVM']
         
    return results

#%% Master function: this function implements the whole analysis pipeline:
# (1) Splits the data
# (2) Estimates the models
# (3) Calculates performance metrics in the test sample

from sklearn.model_selection import KFold
simulations = 1000
# k-fold monte carlo cross-validation DL model 1:
results = [] # accuracy of DL, k-fold monte carlo cross-validation 
for sim in range(simulations): 
    print('K-fold simulations: ',sim+1)
    # define the test condition
    cv = KFold(n_splits=10, shuffle=True)
    # evaluate k value
    res = main_ML(X, y)
    results.append(res)
results = pd.concat(results)

########################## end of script #####################################
