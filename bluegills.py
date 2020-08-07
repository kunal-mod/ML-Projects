import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("bluegills.csv")
features=dataset.iloc[:,0:-1].values
labels=dataset.iloc[:,-1].values

plt.scatter(features, labels)
plt.show()

lr=LinearRegression()
lr.fit(features,labels)
lr_pred=lr.predict(features)
lr.predict([[5]])

plt.scatter(features,labels,color='red')
plt.plot(features,lr.predict(features),color='blue')
plt.title('PLR')
plt.xlabel('Year')
plt.ylabel("Cost")
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_object =PolynomialFeatures(degree = 2)
features_poly = poly_object.fit_transform(features)

lin_reg = LinearRegression()
lin_reg.fit(features_poly,labels)
lrp_pred=lin_reg.predict(features_poly)

lin_reg.predict(poly_object.transform([[5]]))

#Visaulize the polynomial set 
#features_grid=np.arange(min(features),max(features),0.01)
#features_grid=features_grid.reshape(len(features_grid),1)
plt.scatter(features,labels,color='red')
plt.plot(features,lin_reg.predict(poly_object.fit_transform(features)),color='blue')
plt.title('PLR')
plt.xlabel('Year')
plt.ylabel("Cost")
plt.show()