import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
dataset = load_iris()
features = dataset['data']
labels = dataset['target']

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.30,random_state=0)

#Standard scaling

"""
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.fit_transform(features_test)
"""


#using GNB

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(features_train,labels_train)

labels_pred =classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_test,labels_pred)


#Using SVC

"""
from sklearn.svm import SVC
classifier =SVC(kernel='rbf',random_state=0)
classifier.fit(features_train,labels_train)

labels_pred =classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_test,labels_pred)
"""


#using knn
"""
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10,metric='minkowski',p=2)
classifier.fit(features_train,labels_train)
labels_pre=classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pre)

from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_test,labels_pre)
"""




