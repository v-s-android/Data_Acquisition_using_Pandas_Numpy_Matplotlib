'''
Use Linear Regression in one variable to fit the parameters to a model
Use Linear Regression in multiple variables to fit the parameters to a model
Use Polynomial Regression in single variable tofit the parameters to a model
Create a pipeline for performing linear regression using multiple features in polynomial scaling
Evaluate the performance of different forms of regression on basis of MSE and R^2 parameters
'''
import piplite
await piplite.install('seaborn')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

file_name="laptops.csv"
path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"
await download(path, file_name)

df.read_csv(file_name)
df.head()

# Task 1 : Single Linear Regression
'''
Create a single feature Linear Regression model that fits the pair of "CPU_frequency" and "Price" to find the model for prediction.
'''
x = df[['CPU_frequency']]
y = df['Price']

lm = LinearRegression()
lm

lm.fit(x, y)
cpu_freq_predict = lm.predict(x)
cpu_freq_predict[0:5] # array([1073.07834392, 1277.93263722, 1636.42765051, 1073.07834392, 1175.50549057])

# Generate the Distribution plot for the predicted values and that of the actual values. How well did the model perform?

ax1 = sns.distplot( df['Price'], hist = False, color = 'r', label = "Actual Value")

sns.distplot(cpu_freq_predict , hist = False, color = 'b' , ax = ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of laptops')
plt.legend(['Actual Value', 'Predicted Value'])
plt.show()

# Evaluate the Mean Squared Error and R^2 score values for the model.

# lm.fit(x,y) just fir reference
print("R-squared ", lm.score(x, y)) # R-squared  0.13444363210243238

#cpu_freq_predict = lm.predict(x) just for reference
mse = mean_squared_error(df['Price'], cpu_freq_predict)
print("mean_squared_error is ", mse) # mean_squared_error is  284583.44058686297

#Task 2 - Multiple Linear Regression¶
'''
The parameters which have a low enough p-value so as to indicate strong relationship with the 'Price' value are 'CPU_frequency', 'RAM_GB', 
'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU' and 'Category'. Use all these variables to create a Multiple Linear Regression system.
'''
z = df[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']]
lm2 = LinearRegression()
lm2

lm2.fit(z, df['Price'])
z_predict = lm2.predict(z)
print(z_predict[0:5]) # array([1345.51622771,  710.44905496, 1552.37242687, 1295.00681012, 1543.13847022])

# Plot the Distribution graph of the predicted values as well as the Actual values

ax2 = sns.distplot( df['Price'], hist = False, color = 'r', label = "Actual Values") # Actual values
sns.distplot(z_predict, hist = False, color = 'b', label="Fitted Values", ax = ax2)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of laptops')
plt.legend(['Actual Value', 'Predicted Value'])
plt.show()

# Find the R^2 score and the MSE value for this fit. Is this better or worst than the performance of Single Linear Regression?

print("R squared ", lm2.score(z ,df['Price'])) # R squared  0.5082509055187374

mse2 = mean_squared_error(df['Price'], z_predict)
print("mean_squared_error of Multiple Linear Regression",mse2) mean_squared_error of Multiple Linear Regression 161680.57263893107


# Task 3 - Polynomial Regression
'''
Use the variable "CPU_frequency" to create Polynomial features. Try this for 3 different values of polynomial degrees.
Remember that polynomial fits are done using `numpy.polyfit`. 
'''




