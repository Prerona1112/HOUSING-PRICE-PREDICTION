# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# importing the dataset
dataset = pd.read_csv('USA_Housing.csv')
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, -2].values
# splitting up the dataset
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(dataset.shape)
print(X.shape)
print(y.shape)
print(Xtrain.shape)
print(Xtest.shape)
print(ytrain.shape)
print(ytest.shape)
# preprocessing
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
Xtrain = sc_X.fit_transform(Xtrain)
Xtest = sc_X.transform(Xtest)
# fitting the model into linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Xtrain, ytrain)
lin_intercept =("C=",regressor.intercept_)
lin_coef = ("M=",regressor.coef_)
print(lin_intercept)
print('\n')
print(lin_coef)
# predicting the results
y_pred = regressor.predict(Xtest)
print(y_pred)
# Visualising the Training set results
plt.plot(Xtrain, regressor.predict(Xtrain), color = 'blue')
plt.title('Address vs Price (Training set)')
plt.xlabel('Address')
plt.ylabel('Population')
plt.show()
# Visualising the Test set results
print(y_pred.shape)
print(ytest.shape)
plt.plot(Xtest, regressor.predict(Xtest), color = 'blue')
plt.title('address vs price(Test set)')
plt.xlabel('Address')
plt.ylabel('Population')
plt.show()
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(ytest, y_pred)))
accuracy = regressor.score(Xtest, ytest)
print(accuracy)
accuracy_train = regressor.score(Xtrain, ytrain)
print(accuracy_train)
