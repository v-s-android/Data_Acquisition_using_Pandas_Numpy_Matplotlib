'''
Objectives:
Explore features or characteristics to predict price of car
Analyze patterns and run descriptive statistical analysis
Group data based on identified parameters and create pivot tables
Identify the effect of independent attributes on price of cars
'''
#install specific version of libraries used in lab
#! mamba install pandas==1.3.3
#! mamba install numpy=1.21.2
#! mamba install scipy=1.7.1-y
#!  mamba install seaborn=0.9.0-y
import pandas as pd
import numpy as np
import piplite
await piplite.install('seaborn')
from pyodide.http import pyfetch
import matplotlib.pyplot as plt
import seaborn as sns

#doenload the file from url
async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

file_path_url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"
file_name = 'usedcars.csv'
await download(file_path_url, file_name)

df = pd.read_csv(file_name, header = 0) #The header parameter tells Pandas which row contains the column names. eg: make,model,price (column names)
df.head()

## Analyzing Individual Feature Patterns Using Visualization
df.dtypes()
df['peak-rpm'].dtypes # dtype('float64')

# we can calculate the correlation between variables of type "int64" or "float64" using the method "corr":

numeric_df = df.select_dtypes(include = ['float64','int64'])
numeric_df.corr()

# correlation between the following columns: bore, stroke, compression-ratio, and horsepower.
df[['bore','stroke','compression-ratio','horsepower']].corr()

#Continuous Numerical Variables:
#Positive Linear Relationship : find the scatterplot of "engine-size" and "price".
sns.regplot( x = 'engine-size', y = 'price', data = df)
plt.ylim(0,)

#examine the correlation between 'engine-size' and 'price' 
df[['engine-size','price']].corr()
'''
	         engine-size	price
engine-size	1.000000	0.872335
price      	0.872335	1.000000

'''
# Highway mpg is a potential predictor variable of price. Let's find the scatterplot of "highway-mpg" and "price".

sns.regplot( x = 'highway-mpg' ,y =  'price' , data = df)

#We can examine the correlation between 'highway-mpg' and 'price' and see it's approximately -0.704.

df[['highway-mpg', 'price']].corr()
'''
	          highway-mpg	price
highway-mpg	1.000000	-0.704692
price	      -0.704692	1.000000
'''
# Let's see if "peak-rpm" is a predictor variable of "price".
sns.regplot(x = 'peak-rpm', y='price', data = df)
df[['peak-rpm','price']].corr()
'''
	          peak-rpm	price
peak-rpm	1.000000	-0.101616
price	    -0.101616	1.000000
'''

