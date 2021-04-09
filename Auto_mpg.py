import numpy as np 
import pandas as pd

data = pd.read_csv('Auto_mpg.txt',sep="\s+" )


data["mpg"].idxmax(axis = 0)
data["car name"][322]

data.isin(['?']).sum(axis=0)
data.isnull().sum()

features=data.iloc[:,1:-1].values
labels=data.iloc[:,0:1].values

np.where(features=="?")



"""
data.isin(['?']).sum(axis=0)
data.replace(to_replace ="?", value ='nan',inplace=True)
"""
features=np.where(features=="?", "nan", features)
np.where(features=="nan")

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(features[:,2:3])
features[:,2:3]=imputer.transform(features[:,2:3])


from sklearn.model_selection import train_test_split
features_train , features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.2,random_state=40)

from sklearn.tree import DecisionTreeRegressor
classifier_dc=DecisionTreeRegressor(criterion='mse',random_state=0)
classifier_dc.fit(features_train,labels_train)

#Predicting the test set 
labels_pred_dc=classifier_dc.predict(features_test)
classifier_dc.score(features_test,labels_test)



from sklearn.ensemble import RandomForestRegressor
classifier_rf=RandomForestRegressor(n_estimators=10,random_state=0)
classifier_rf.fit(features_train,labels_train)

#Predicting the test set 
labels_pred_rf=classifier_rf.predict(features_test)

classifier_rf.score(features_test,labels_test)

pv=[[6,215,100.0,2630,22.2,80,3]]
dcp=classifier_dc.predict(pv)
rfp=classifier_rf.predict(pv)
l1=labels_pred_dc
l2=labels_pred_rf

