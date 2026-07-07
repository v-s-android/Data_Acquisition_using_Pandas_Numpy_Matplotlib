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

