import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

df = pd.read_csv("breast_cancer.csv")
features = df.iloc[:,1:10].values
labels = df.iloc[:,-1].values




from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputer=imputer.fit(features[:,5:6])
features[:,5:6]=imputer.transform(features[:,5:6])

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.fit_transform(features_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(features_train,labels_train)

labels_pre = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pre)

from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_test,labels_pre)


#using knn

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=15,metric='minkowski',p=2)
classifier.fit(features_train,labels_train)
labels_pre=classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pre)

from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_test,labels_pre) 


#using SVM
from sklearn.svm import SVC
classifier =SVC(kernel='rbf',random_state=0)
classifier.fit(features_train,labels_train)

labels_pred =classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_test,labels_pred)


l=[[6,2,5,3,9,4,7,2,2]]
l=sc.fit_transform(l)
pre=classifier.predict(l)