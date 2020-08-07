import pandas as pd
import numpy as np

df = pd.read_csv("affairs.csv")
features = df.iloc[:,:-1].values
labels = df.iloc[:,-1].values


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


ct = ColumnTransformer([('encoder', OneHotEncoder(), [6])], remainder='passthrough')

features = np.array(ct.fit_transform(features), dtype = np.str)

features=features[:,1:]

ct1 = ColumnTransformer([('encoder', OneHotEncoder(), [11])], remainder='passthrough')

features = np.array(ct1.fit_transform(features), dtype = np.str)

features=features[:,1:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features = sc.fit_transform(features)

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression
classifire = LogisticRegression()
classifire.fit(features_train,labels_train)

labels_pre = classifire.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pre)

from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_test,labels_pre)

did = np.count_nonzero(labels_pre == 1)
didnt = np.count_nonzero(labels_pre == 0)


per=(did/1548)*100
 

