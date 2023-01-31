## Pre-screening breast cancer with machine learning and deep learning

<p align="justify"> Deep learning has been widely applied in breast cancer screening to analyze images obtained from X-rays, ultrasounds, magnetic resonances, and biopsies. We suggest that deep learning can also be used for pre-screening cancer by analyzing demographic and anthropometric information of patients, as well as biological markers obtained from routine blood samples and relative risks obtained from meta-analysis and international databases. We applied feature selection algorithms to a database of 116 women, including 52 healthy women and 64 women diagnosed with breast cancer, to identify the best pre-screening predictors of cancer. We utilized the best predictors to perform k-fold Monte Carlo cross-validation experiments that compare deep learning against traditional machine learning algorithms. Our results indicate that a deep learning model with an input-layer architecture that is fine-tuned using feature selection can effectively distinguish between patients with and without cancer, since the average area under the curve (AUC) of a deep learning model is 87%, with a 95% confidence interval between 82% and 91%. Additionally, compared to machine learning, deep learning has the lowest uncertainty in its predictions, as indicated by its standard deviation of AUC equal to 0.0345. </p>

<p align="justify"> The python script [`DLbcancer_fsfe.py`](script/DLbcancer_fsfe.py) evaluates the consequences of implementing feature selection and feature engineering in deep learning architectures aimed to predict breast cancer. The minimum redundancy maximum relevance (MRMR) algorithm of SULOV-gradient boosting  is applied to select a minimal-optimal subset of features that can be combined to improve the accuracy of prediction of breast cancer with deep learning algorithms based on L1-L2 regularized dense layers.  The data used in the study is the Coimbra dataset that can be found in the [UCI Machine Learning Repository of the Center for Machine Learning and Intelligent Systems](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra) and is available also in the `data` folder of this repository ([`dataR2.csv`](data/dataR2.csv)). </p>

Monte Carlo experiments with k-fold cross-validation were performed to compare the α-accuracy of 4 different deep learning architectures in the input layer of the neural network: 

1. All features (saturated input layer architecture without feature selection and without engineering)
2. With feature selection
3. With feature selection and with feature engineering of the selected features
4. With feature engineering alone

![image](https://user-images.githubusercontent.com/62504422/187880563-ed734bd6-435d-454f-bff2-1eb06a7271ec.png)

The Deep learning models with optimal predictors were compared with machine learning models:

<img src="https://user-images.githubusercontent.com/62504422/215833763-c4d92255-49cf-4f13-b091-b0aa10075ce2.png" width=55% height=55%>
