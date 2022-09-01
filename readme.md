## Python script to reproduce the results of the study: _Predicting breast cancer with deep learning: do feature selection and feature engineering improve the accuracy of predictions with bio-markers and demographic information?_

Deep learning algorithms applied to demographic information of patients and biological markers obtained from routine blood samples offer an affordable, non-invasive and radiation free alternative for breast cancer pre-screening. 

The python script [`DLbcancer_fsfe.py`](script/DLbcancer_fsfe.py) evaluates the consequences of implementing feature selection and feature engineering in deep learning architectures aimed to predict breast cancer. The minimum redundancy maximum relevance (MRMR) algorithm of SULOV-gradient boosting  is applied to select a minimal-optimal subset of features that can be combined to improve the accuracy of prediction of breast cancer with deep learning algorithms based on L1-L2 regularized dense layers.  The data used in the study is the Coimbra dataset that can be found in the [UCI Machine Learning Repository of the Center for Machine Learning and Intelligent Systems](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra) and is available also in the `data` folder of this repository ([`dataR2.csv`](data/dataR2.csv)).




Monte Carlo experiments with k-fold cross-validation were performed to compare the α-accuracy of 4 different deep learning architectures in the input layer of the neural network: 

1. All features (saturated input layer architecture without feature selection and without engineering)
2. With feature selection
3. With feature selection and with feature engineering of the selected features
4. With feature engineering alone

![image](https://user-images.githubusercontent.com/62504422/187880563-ed734bd6-435d-454f-bff2-1eb06a7271ec.png)
