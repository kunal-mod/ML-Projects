import pandas as pd
import numpy as np

df = pd.read_csv('Foodtruck.csv')
features = df.iloc[:,0].values
labels = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
r = LinearRegression()
r.fit(features_train.reshape(-1,1),labels_train.reshape(-1,1))
lable_pre = r.predict(features_test.reshape(-1,1))
print(r.predict([[3.073]]))