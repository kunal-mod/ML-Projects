import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv("PastHires.csv")
features=dataset.iloc[:,:-1].values
labels=dataset.iloc[:,-1].values



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
labels=le.fit_transform(labels)


from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder='passthrough')
features=np.array(ct.fit_transform(features),dtype=np.str)
features=features[:,1:]

ct1=ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder='passthrough')
features=np.array(ct1.fit_transform(features),dtype=np.str)
features=features[:,1:]

ct2=ColumnTransformer([('encoder',OneHotEncoder(),[5])],remainder='passthrough')
features=np.array(ct2.fit_transform(features),dtype=np.str)
features=features[:,1:]

ct3=ColumnTransformer([('encoder',OneHotEncoder(),[6])],remainder='passthrough')
features=np.array(ct3.fit_transform(features),dtype=np.str)
features=features[:,1:]


features=np.array(features,dtype="float64")

from sklearn.model_selection import train_test_split
features_train , features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.2,random_state=40)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(features_train,labels_train)

#Predicting the test set 
labels_pred=classifier.predict(features_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test, labels_pred)


# Accuracy Score 
from sklearn.metrics import accuracy_score
ac=accuracy_score(labels_test,labels_pred)


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(features_train,labels_train)

#Predicting the test set 
labels_pred=classifier.predict(features_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test, labels_pred)


# Accuracy Score 
from sklearn.metrics import accuracy_score
ac=accuracy_score(labels_test,labels_pred)

p=np.array([[10,'Y',4,'BS','Y','N'],[10,'N',4,'MS','N','Y']])


p=np.array(ct.transform(p),dtype=np.str)
p=p[:,1:]
p=np.array(ct1.transform(p),dtype=np.str)
p=p[:,1:]
p=np.array(ct2.transform(p),dtype=np.str)
p=p[:,1:]
p=np.array(ct3.transform(p),dtype=np.str)
p=p[:,1:]
p=np.array(p,dtype="float64")
p_pred=classifier.predict(p)
l_predt=le.inverse_transform(p_pred)
