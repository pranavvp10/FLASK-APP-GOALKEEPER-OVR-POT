# Importing useful libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as pre
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 
from sklearn.neural_network import MLPRegressor

# importing data
#data_path= 'fifa.csv'
#da = pd.read_csv(data_path)
#da.head()

# Model to find the best regression model
df=pd.read_csv('C:\\Users\\praveen\\PycharmProjects\\pvp\\pvp\\fifa.csv',encoding="cp1252")
gk=['potential','age','overall','position','gkdiving','gkhandling','gkkicking','gkpositioning','gkreflexes']
gkf = pd.DataFrame(df, columns = gk)
golk=['age','overall','position','gkdiving','gkhandling','gkkicking','gkpositioning','gkreflexes']
gk2 = pd.DataFrame(gkf, columns = golk)
ft=gk2.loc[gk2['position']=='GK']
dt=ft.drop('position',axis=1)

#model_data.plot.scatter(x='carat', y='price', s=1);


target_name = 'overall'
#scaler = sk.preprocessing
#robust_scaler = pre.RobustScaler()
X = dt.drop('overall',axis=1)
#X = robust_scaler.fit_transform(X)
y=dt[target_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
models = pd.DataFrame(index=['train_mse', 'test_mse'], columns=['NULL','MLR', 'Ridge','KNN', 'LASSO'])

#Null Model
y_pred_null = y_train.mean()
models.loc['train_mse','NULL'] = mean_squared_error(y_pred=np.repeat(y_pred_null, y_train.size),
                                                  y_true=y_train)
models.loc['test_mse','NULL'] = mean_squared_error(y_pred=np.repeat(y_pred_null, y_test.size),
                                                   y_true=y_test)

#Linear Regression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
models.loc['train_mse','MLR'] = mean_squared_error(y_pred=linear_regression.predict(X_train), y_true=y_train)
models.loc['test_mse','MLR'] = mean_squared_error(y_pred=linear_regression.predict(X_test), y_true=y_test)


#Ridge Regreesion
ridge_regression = Ridge()
ridge_regression.fit(X_train, y_train)
models.loc['train_mse','Ridge'] = mean_squared_error(y_pred=ridge_regression.predict(X_train), y_true=y_train)
models.loc['test_mse','Ridge'] = mean_squared_error(y_pred=ridge_regression.predict(X_test), y_true=y_test)

#new_diamond = OrderedDict([('carat',0.45), ('depth',62.3), ('table',59.0), ('x',3.95),
#                           ('y',3.92), ('z',2.45), ('cut_Good',0.0), ('cut_Ideal',0.0),
#                           ('cut_Premium',1.0), ('cut_Very Good',0.0), ('color_E',0.0),
#                           ('color_F',0.0), ('color_G',1.0), ('color_H',0.0), ('color_I',0.0),
#                           ('color_J',0.0), ('clarity_IF',0.0), ('clarity_SI1',0.0),
#                           ('clarity_SI2',0.0), ('clarity_VS1',0.0), ('clarity_VS2',0.0),
#                           ('clarity_VVS1',1.0), ('clarity_VVS2',0.0), ('carat_squared',0.0576)])
#new_diamond = pd.Series(new_diamond).values.reshape(1,-1)
#from sklearn.model_selection import GridSearchCV
#parameters = {'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
#ridge = Ridge()
#gridridge = GridSearchCV(ridge, parameters)
#gridridge.fit(X_train, y_train)
#y_pred=gridridge.predict(new_diamond)
#final_price = np.abs(y_pred)
#print(y_pred)

#KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=10, weights='distance', metric='euclidean', n_jobs=-1)
knn.fit(X_train, y_train)
models.loc['train_mse','KNN'] = mean_squared_error(y_pred=knn.predict(X_train), y_true=y_train)
models.loc['test_mse','KNN'] = mean_squared_error(y_pred=knn.predict(X_test), y_true=y_test)


# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
models.loc['train_mse','LASSO'] = mean_squared_error(y_pred=lasso.predict(X_train), y_true=y_train)
models.loc['test_mse','LASSO'] = mean_squared_error(y_pred=lasso.predict(X_test), y_true=y_test)

# Random Forest Regressor

clf_rf = RandomForestRegressor()
clf_rf.fit(X_train, y_train)
models.loc['train_mse','RF'] = mean_squared_error(y_pred=clf_rf.predict(X_train), y_true=y_train)
models.loc['test_mse','RF'] = mean_squared_error(y_pred=clf_rf.predict(X_test), y_true=y_test)


rf_final = RandomForestRegressor(n_estimators =50, n_jobs=-1)
rf_final.fit(X, y)
#new_data = OrderedDict([('age',30),('gkdiving',80),('gkhandling',82),('gkkicking',83),('gkpositioning',81),('gkreflexes',84)])
#new_data = pd.Series(new_data).values.reshape(1,-1)
#print(rf_final.predict(new_data))
#print(linear_regression.predict(new_data))
#print(lasso.predict(new_data))
#print(ridge_regression.predict(new_data))
#print(knn.predict(new_data))
