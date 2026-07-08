'''
Develop prediction models : we will develop several models that will predict the price of the car using the variables or features
'''
#install specific version of libraries
#! mamba install pandas==1.3.3-y
#! mamba install numpy=1.21.2-y
#! mamba install sklearn=0.20.1-y
import piplite
await piplite.install('seaborn')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyodide.http import pyfetch
from sklearn.linear_model import LinearRegression

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

file_path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"
file_name="usedcars.csv"
await download(file_path, file_name)

df = pd.read_csv(file_name)
df.head()

# Create the linear regression object:
lm = LinearRegression()
print(lm)

# Using simple linear regression, we will create a linear function with "highway-mpg" as the predictor variable and the "price" as the response variable.
X = df[['highway-mpg']]
Y = df['price']

# Fit the linear model using highway-mpg:
lm.fit(X,Y)

# We can output a prediction:
Yhat = lm.predict(X)
print(Yhat[0:5]) # array([16236.50464347, 16236.50464347, 17058.23802179, 13771.3045085 , 20345.17153508])

# the value of the intercept (a)
print(lm.intercept_) # 38423.30585815743

# the value of the slope (b)
print(lm.coef_) # array([-821.73337832])

#Create a linear regression object called "lm1".
lm1 = LinearRegression()
print(lm1)
#Train the model using "engine-size" as the independent variable and "price" as the dependent variable
Xeng = df[['engine-size']]
Yprice = df['price']

lm1.fit(X,Y) # or alternativley lm1.fit(df[['engine-size']], df[['price']])
Yhat = lm1.predict(X)
print(Yhat[0:5]) # array([16236.50464347, 16236.50464347, 17058.23802179, 13771.3045085 , 20345.17153508])

# Intercept and slope
print(lm1.intercept_)
print(lm1.coef_)

'''
Multiple Linear Regression : 

Other good predictors of price could be: Horsepower, Curb-weight, Engine-size, Highway-mpg
Developing a model using these variables as the predictor variables.
'''
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
#Fit the linear model using the four above-mentioned variables.
lm.fit(Z , df['price'])

Yhat = lm.predict(Z)
Yhat[0:5]

lm.intercept_ # -15806.62462632922
lm.coef_ # array([53.49574423,  4.70770099, 81.53026382, 36.05748882])
# Price = -15678.742628061467 + 52.65851272 x horsepower + 4.69878948 x curb-weight + 81.95906216 x engine-size + 33.58258185 x highway-mpg

# Similarly : Create and train a Multiple Linear Regression model "lm2" where the response variable is "price", and the predictor variable is "normalized-losses" and "highway-mpg".
lm2 = LinearRegression()
lm2.fit(df[['normalized-losses' , 'highway-mpg']],df['price'])

print(lm2.coef_) # [   1.49789586 -820.45434016]
print(lm2.intercept_) # 38201.313272457344

# 2. Model Evaluation Using Visualization
import seaborn as sns

'''
This plot will show a combination of a scattered data points (a scatterplot), as well as the fitted linear regression line going through the data.
This will give us a reasonable estimate of the relationship between the two variables, the strength of the correlation,
as well as the direction (positive or negative correlation).
'''
# Let's visualize highway-mpg as potential predictor variable of price:

width = 8
height = 6
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

# similarly for peak-rpm

plt.figure(figsize = (width, height))
sns.regplot(x = 'peak-rpm', y = 'price', data = df)
plt.ylim(0,)

# Given the regression plots above, is "peak-rpm" or "highway-mpg" more strongly correlated with "price"? Use the method ".corr()"
df[['peak-rpm','ighway-mpg','price']].corr()
'''
	            peak-rpm	highway-mpg	       price
peak-rpm	    1.000000	    -0.058598	  -0.101616
highway-mpg	    -0.058598	    1.000000	   -0.704692
price	        -0.101616	    -0.704692	    1.000000

# The variable "highway-mpg" has a stronger correlation with "price", it is approximate -0.704692  compared to "peak-rpm" which is approximate -0.101616.
'''
# Residual Plot
'''
What is a residual?

The difference between the observed value (y) and the predicted value (Yhat) is called the residual (e).
When we look at a regression plot, the residual is the distance from the data point to the fitted regression line.

So what is a residual plot?

A residual plot is a graph that shows the residuals on the vertical y-axis and the independent variable on the horizontal x-axis.
'''
plt.figure(figsize =(width, height))
sns.residplot( x = df['highway-mpg'], y = df['price'] ) # notice we are using the directly the dataframe column
plt.show()


