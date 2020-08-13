import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#IMPORTING THE DATASET
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values


#true value of level 6.5 = $160000
true = [68000, 130000, 160000 ]
#dictionary of all predicted values
results = {}

 
#LINEAR REGRESSION
from sklearn.linear_model import LinearRegression 
lr = LinearRegression()
lr.fit(x,y)
y_lr = lr.predict([[3.5],[5.5],[6.5]])

results['Linear_Regression'] = y_lr

plt.scatter(x, y, color = 'red')
plt.plot(x, lr.predict(x), color = 'blue')
plt.title('(Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree =4)
x_poly = pr.fit_transform(x) # generates x,x^2,x^3...
pr.fit(x_poly)

lr2 = LinearRegression()
lr2.fit(x_poly,y)
y_pr = lr2.predict(pr.fit_transform([[3.5],[5.5],[6.5]]))

results['Polynomial_Regression'] = y_pr

plt.scatter(x, y, color = 'red')
plt.plot(x, lr2.predict(pr.fit_transform(x)), color = 'blue')
plt.title('(Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#SUPPORT VECTOR REGRESSION

from sklearn.preprocessing import StandardScaler
y1 = dataset.iloc[:, 2:].values

sc_X = StandardScaler()
sc_y = StandardScaler()
x_svr = sc_X.fit_transform(x)
y_svr = sc_y.fit_transform(y1)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_svr, y_svr)

# Predicting a new result
y_pred = regressor.predict(sc_X.transform([[3.5],[5.5],[6.5]]))
y_pred = sc_y.inverse_transform(y_pred)

results['Support_vector_Regression'] = y_pred
# Visualising the SVR results
plt.scatter(x_svr, y_svr, color = 'red')
plt.plot(x_svr, regressor.predict(x_svr), color = 'blue')
plt.title('(SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#DESICION TREE REGRESSION
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x,y)
y_dt = dt.predict([[3.5],[5.5],[6.5]])

results['Decision_tree_Regression'] = y_dt

plt.scatter(x,y,color='red')
plt.plot(x,dt.predict(x),color='blue')
plt.title('(Desiction Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#RANDOM FORREST REGRESSION
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=300)
rf.fit(x,y)
y_rf = rf.predict([[3.5],[5.5],[6.5]])
results['Random_forest_Regression'] = y_rf

plt.scatter(x,y,color='red')
plt.plot(x,rf.predict(x),color='blue')
plt.title('(Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



for i in results:
    print(i)
    print(r2_score(true,results[i]))
    print("\n")




