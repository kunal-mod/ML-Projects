import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv("50_Startups.csv")
features=dataset.iloc[:,:-1].values
labels=dataset.iloc[:,-1].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder='passthrough')
features=np.array(ct.fit_transform(features),dtype=np.str)
features=features[:,1:]

import statsmodels.regression.linear_model as sm

features_obj = features[:,[0,1,2,3,4,]]
features_obj =np.append(arr=np.ones((50,1)).astype(int),values=features,axis=1)

while True:
    features_obj=features_obj.astype(float)
    labels=labels.astype(float)
    regressor_ols = sm.OLS(endog=labels,exog=features_obj).fit()
    p_values=regressor_ols.pvalues
    if p_values.max() > 0.05:
        features_obj = np.delete(features_obj,p_values.argmax(),1)
    else:
        break
    
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features_obj,labels,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
r = LinearRegression()
r.fit(features_train,labels_train)
labels_pre = r.predict(features_test)


from sklearn.model_selection import train_test_split
features_train1,features_test1,labels_train1,labels_test1=train_test_split(features,labels,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(features_train1,labels_train1)
labels_pre1 = reg.predict(np.array(features_test1,dtype="float64"))


for i in range(len(labels_pre)):
    print("Diff in back: " , (labels_test[i]-labels_pre[i]) , "Diff without: " , (labels_test[i]-labels_pre1[i]))

