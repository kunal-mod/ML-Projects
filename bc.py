import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv("breast_cancer.csv")
features = df.iloc[:,1:10].values
labels = df.iloc[:,-1].values




from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputer=imputer.fit(features[:,5:6])
features[:,5:6]=imputer.transform(features[:,5:6])







p=pd.DataFrame(features)
p = p.drop([2],axis=1)
s=p.isnull().sum()

X=p.iloc[:,:]
# Calculating VIF
vif = pd.DataFrame()
vif["columns"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    

