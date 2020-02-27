# Publication Details
Improving the accuracy of prediction of lung cancer patient survival time using LSTM Neural Networks.

International Conference of Sabaragamuwa University of SriLanka 
Nov 2019

# Survival-Time-Prediction-of-a-patient-using-LSTM-MLP-DBN
This repo is about predicting survival time of a patient using different deep learning techniques.
Basically many models were developed before in literature .

we tried to developed the system such that RMSE is minimum compared to works done before in literature. 

## Methods Used
1.MLP<br/>
2.DBM<br/>
3.LSTM

  
# Abstract: 
Outcomes for cancer patients have been previously estimated by applying various
machine learning techniques to large data-sets such as the Surveillance, Epidemiology, and
End Results (SEER) program database. In particular for lung cancer, it is not well under-
stood which types of techniques would yield more predictive information, and which data
attributes should be used in order to determine this information. In this study, a number of
supervised learning techniques is applied to the SEER database to classify lung cancer patients
in terms of survival, including linear regression, Decision Trees, Gradient Boosting Machines
(GBM), Support Vector Machines (SVM), CNN and Deep belief model. Key data attributes
in applying these methods include tumor grade, tumor size, gender, age, stage, and number
of primaries, with the goal to enable comparison of predictive power between the various
methods. The prediction is treated like a continuous target, rather than a classification into
categories, as a first step towards improving survival prediction. We conclude that application
of these supervised learning techniques to lung cancer data in the SEER database may be of
use to estimate patient survival time with the ultimate goal to inform patient care decisions,
and that the performance of these techniques with this particular data-set may be on par with
that of classical methods.
 
 # Results 
 
 Models    RMSE      Standard Deviation   Mean of Predictions    Mean of residuals <br/>
  LSTM     10.53           14.2652             42.8517                7.5264<br/>
  MLP 1    14.8787         11.5504             45.4631                9.3820<br/>
  MLP 2    14.9684         11.6146             46.3205                9.4452<br/>
  DBN      16.399          7.4902              40.0                   14.5900
