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
f1 = np.polyfit(df['CPU_frequency'], df['Price'], 1)
p1 = np.poly1d(f1)

f3 = np.polyfit(df['CPU_frequency'], df['Price'], 3)
p3 = np.poly1d(f3)

f5 = np.polyfit(df['CPU_frequency'], df['Price'], 5)
p5 = np.poly1d(f5)

# Plot the regression output against the actual data points to note how the data fits in each case. To plot the polynomial response over 
# the actual data points, you have the function shown below.
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(independent_variable.min(),independent_variable.max(),100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title(f'Polynomial Fit for Price ~ {Name}')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of laptops')

# Call for function of degree 1
PlotPolly(p1, df['CPU_frequency'], df['Price'] , 'CPU_frequency')
# Call for function of degree 3
PlotPolly( p3, df['CPU_frequency'], df['Price'] , 'CPU_frequency')
# Call for function of degree 5
PlotPolly( p5, df['CPU_frequency'], df['Price'] , 'CPU_frequency')

# calculate the R^2 and MSE values for these fits. For polynomial functions, the function sklearn.metrics.r2_score will be used to calculate R^2 values.
from sklearn.metrics import r2_score

X = df[['CPU_frequency']]
Y = df['Price'] # you can use r2_score(Y, p1(x))
print('The R-square value for 1st degree polynomial is: ', r2_score(df['Price'], p1(X)))
print('the mean_squared_error for 1st degree polynomial is: ',mean_squared_error(df['Price'], p1(X)))

print('The R-square value for 3rd degree polynomial is: ', r2_score(df['Price'], p3(X)))
print('the mean_squared_error for 3rd degree polynomial is: ',mean_squared_error(df['Price'], p3(X)))

print('The R-square value for 5th degree polynomial is: ', r2_score(df['Price'], p5(X)))
print('the mean_squared_error for 5th degree polynomial is: ',mean_squared_error(df['Price'], p5(X)))
'''
The R-square value for 1st degree polynomial is:  0.13444363210243282
the mean_squared_error for 1st degree polynomial is:  284583.4405868628
The R-square value for 3rd degree polynomial is:  0.26692640796530986
the mean_squared_error for 3rd degree polynomial is:  241024.8630384881
The R-square value for 5th degree polynomial is:  0.3030822706443803
the mean_squared_error for 5th degree polynomial is:  229137.29548053825
'''

# Task 4 - Pipeline
'''
Create a pipeline that performs parameter scaling, Polynomial Feature generation and Linear regression. Use the set of multiple features as before to create this pipeline.
'''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Z = df[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']]
Input = [('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe = Pipeline(Input)
Z = Z.astype(float) # converting all values to float 
pipe.fit( Z, df['Price']) 
Y_pipe = pipe.predict(Z) 
print(Y_pipe[0:5]) # array([1371. , 1159.5, 1389. ,  -52.5, 1602.5])

# Evaluate the MSE and R^2 values for the this predicted output.
print("R^2 for multi-variable polynomial pipeline is: ", r2_score(df['Price'], Y_pipe ))
print("MSE for multi-variable polynomial pipeline is: ", mean_squared_error(df['Price'], Y_pipe))
'''
R^2 for multi-variable polynomial pipeline is:  0.2563325298426047
MSE for multi-variable polynomial pipeline is:  244507.9894957983

You should now have seen that the values of R^2 increase as we go from Single Linear Regression to Multiple Linear Regression. 
Further, if we go for multiple linear regression extended with polynomial features, we get an even better R^2 value.
'''




