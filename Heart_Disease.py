import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


df = pd.read_csv("Heart_Disease.csv")

a= df.isnull().sum()

features = df.iloc[:,0:9].values
labels = df.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features = sc.fit_transform(features)

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression
classifire = LogisticRegression()
classifire.fit(features_train,labels_train)

labels_pre = classifire.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pre)

from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_test,labels_pre) 

