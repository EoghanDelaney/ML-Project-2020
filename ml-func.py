import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as lin
import sklearn.model_selection as ms

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv (r'powerproduction.csv')
df_remove_power_Zero =  df[df.power != 0]

X = df_remove_power_Zero.values[:,0].reshape(-1, 1)
Y = df_remove_power_Zero.values[:,1]

#X_reshape =  X.
#X_train, X_validation, Y_train, Y_validation = ms.train_test_split(X_reshape, Y, test_size=0.20, random_state=7)

#lin_reg_model = lin.LinearRegression().fit(X_train,Y_train)

#r = lin_reg_model.score(X_validation, Y_validation)
#p = [lin_reg_model.intercept_, lin_reg_model.coef_[0]]



neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, Y)

print(neigh.predict_proba([[6.4]]))