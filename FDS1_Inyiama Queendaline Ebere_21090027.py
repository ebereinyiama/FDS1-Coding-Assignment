# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 20:20:55 2022

@author: EbereInyiama
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

### Question 1
### Reading the file through pandas dataframe.

df_inputdata7 = pd.read_csv("inputdata7.csv")
print(f"the input data 7: \n {df_inputdata7}")

### Reading the data file into python numpy array

df_inputdata7_array = np.array(pd.read_csv('inputdata7.csv'))
print("the input data 7 as an array: \n", df_inputdata7_array)

### creating the columns as numpy array variables

np_array_rainfall = df_inputdata7_array[:, 0]
np_array_productivity = df_inputdata7_array[:, 1]

### Another method of creating numpy arrays of the columns

#np_array_rainfall = np.array(df_inputdata7['Rainfall'])
#np_array_productivity = np.array(df_inputdata7['Productivity'])

print("the rainfall column: \n", np_array_rainfall)

print("the productivity column: \n", np_array_productivity)

### Question 2
###  plotting the inputdata7 as a two-dimensional scatter plot

plt.figure(figsize=(8,6))
plt.scatter(np_array_rainfall, np_array_productivity, color = 'magenta')
plt.title('Rainfall vs Productivity', fontsize = 20)
plt.xlabel('Rainfall', fontsize = 15)
plt.ylabel('Productivity', fontsize = 15)
plt.show()

### Fitting using numpy

"""a simple linear equation is given as y = ax + b
plotting np_array_rainfall as (x) and np_array_productivity as a function of rainfall f(x) or y"""

plt.figure(figsize=(8,6))
a, b = np.polyfit(np_array_rainfall, np_array_productivity, 1) # polynomial of one degree
plt.scatter(np_array_rainfall, np_array_productivity, color = 'magenta')
plt.plot(np_array_rainfall, a*np_array_rainfall+b, color = 'blue')
plt.title('Fitting Rainfall vs Productivity using Numpy', fontsize = 15)
plt.xlabel('Rainfall', fontsize = 15)
plt.ylabel('Productivity', fontsize = 15)
plt.show()

### Another method of fitting with numpy

f, g = np.polyfit(np_array_rainfall, np_array_productivity, 1)
predict = np.poly1d([f, g])
y_pred = predict(np_array_rainfall)

plt.figure(figsize=(8,6))
plt.scatter(np_array_rainfall, np_array_productivity,color='magenta')
plt.plot(np_array_rainfall, y_pred,color='blue')

plt.title('Fitting Rainfall vs Productivity using Numpy 2', fontsize = 15)
plt.xlabel('Rainfall', fontsize = 15)
plt.ylabel('Productivity', fontsize = 15)
plt.show()


### Question 3
### creating a linear regression model based on the data 
# Linear Regression already imported

model = LinearRegression()

X = np_array_rainfall[:, np.newaxis] #re-shaping rainfall column to a matrix for prediction
print(X.shape) # to confirm X is a matrix

print(model.fit(X, np_array_productivity)) # to confirm the variable 'model' is a linear regression fit

y_predictions = model.predict(X) #predicting productivity using re-shaped x (rainfall)

print()
print("the predicted values of y from linear regression model is: \n", y_predictions)


### Question 4
### Plotting the corresponding line over the original data

plt.figure(figsize=(8,6))
plt.scatter(np_array_rainfall, np_array_productivity, color = 'black')
plt.plot(X, y_predictions, color = 'blue')
plt.title('Fitting Rainfall vs Productivity using Linear Regression', fontsize = 15)
plt.xlabel('Rainfall', fontsize = 15)
plt.ylabel('Productivity', fontsize = 15)
plt.show()


### plotting only the re-shaped rainfall column (X) and the y predictions from linear regression as a scatter plot 

plt.figure(figsize = (8,6))
plt.scatter(X, y_predictions)
plt.xlabel('Rainfall (X)', fontsize = 15)
plt.ylabel('Predicted Productivity', fontsize = 15)
plt.title('Rainfall vs Predicted Productivity', fontsize = 15)
plt.show() ### Plot depicts a linear graph


### Question 5

"""Using the linear regression model to evaluate the productivity 
coefficient of the field if the amount of precipitations is 310 mm 
"""

print("the intercept is: ", model.intercept_) # the intercept or the constant of a linear equation
print("the regression coefficient is: ", model.coef_) #regression coefficient also known as slope

intercept = model.intercept_
slope = model.coef_

"""linear equation y = a + bx
where a is intercept, b is coefficient/slope and x is 310mm"""

x = 310
y_310 = intercept + (slope*x) 

print("The productivity coefficient of the field if the amount of precipitation is 310mm: ", y_310)

### Showing the resulting coefficient value on the plot

plt.figure(figsize = (8,6))
plt.scatter(X, y_predictions, color = "magenta")
plt.scatter(x, y_310, color = "blue" )
plt.xlabel('Rainfall', fontsize = 15)
plt.ylabel('Predicted Productivity', fontsize = 15)
plt.title('Plot showing Resulting Productivity Coefficient', fontsize = 15)
plt.show()












