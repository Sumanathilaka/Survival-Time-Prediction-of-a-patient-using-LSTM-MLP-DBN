# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:39:36 2019

@author: desha
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:49:17 2019

@author: desha
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import math
from dbn.tensorflow import SupervisedDBNClassification

puredata = pd.read_csv('final3.csv' ,names =["Age","Grade","Radiation-Sequence-with-surgery","No-of-Primaries","T","N","M","Radiation","Stage","Primary-Site","First-Malignant-Primary-Indicator","Sequence-Number","CS-Lymphnodes","Histology-Recode-Broad-Groupings","RXSumm-ScopeRegLNSur(2003+)","RXSumm-SurgPrimSite(1998+)","DerivedSS1977","Tumor-Size","Survival-Time"])
print (puredata.head())

print (puredata.describe().transpose())

X = puredata.drop('Survival-Time',axis=1)
y = puredata['Survival-Time']

X_train, X_test, y_train, y_test = train_test_split(X, y)
y_test=y_test.tolist()
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


mlp = SupervisedDBNClassification(hidden_layers_structure=[19,30,19],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=50,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)

mlp.fit(X_train,y_train)
# Save the model
mlp.save('model.pkl')
# Restoreit
mlp = SupervisedDBNClassification.load('model.pkl')

predictions = mlp.predict(X_test)

RMSE_sum=0

list=[]
for x in range(0,len(X_test)):
    
    RMSE_sum=RMSE_sum+ ((y_test[x]-predictions[x])**2)
    list.append(abs(y_test[x]-predictions[x]))

RMSE=math.sqrt (RMSE_sum/len(X_test))
print ("RMSE :", RMSE)
print ("Mean of predictions : ",np.mean(predictions))
print ("Mean of residuals : " , np.mean(list))
print ("Standard deviation : ",np.std(list,ddof=1))

