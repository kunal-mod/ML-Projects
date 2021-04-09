import pandas as pd
import numpy as np
import matplotlib as ply

df = pd.read_csv("caesarian.csv")
features = df.iloc[:,:-1].values
labels = df.iloc[:,-1].values

n=df.isnull().sum()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features = sc.fit_transform(features)

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels, test_size=0.3, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(features_train,labels_train)
labels_pre=classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pre)

from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_test,labels_pre) 
