'''
Data Analytics for House Pricing Data Set
- Determining the market price of a house given a set of features. You will analyze and predict housing prices using attributes or features
such as square footage, number of bedrooms, number of floors, and so on.
'''
# Import the required libraries
# All Libraries required for this lab are listed below. The libraries pre-installed on Skills Network Labs are commented.
# !mamba install -qy pandas==1.3.4 numpy==1.21.4 seaborn==0.9.0 matplotlib==3.5.0 scikit-learn==0.20.1
# Note: If your environment doesn't support "!mamba install", use "!pip install"

# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
#!pip install -U scikit-learn

import piplite
await piplite.install('seaborn')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Module 1: Importing Data Sets
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
await download(filepath, "housing.csv")
file_name="housing.csv"

df = pd.read_csv(file_name)
df.head()

# Question 1
# Display the data types of each column using the function dtypes. Take a screenshot of your code and output. You will need to submit the screenshot for the final project.
df.info()
'''
Data columns (total 22 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Unnamed: 0     21613 non-null  int64  
 1   id             21613 non-null  int64  
 2   date           21613 non-null  object 
 3   price          21613 non-null  float64
 4   bedrooms       21600 non-null  float64
 5   bathrooms      21603 non-null  float64
 6   sqft_living    21613 non-null  int64  
 7   sqft_lot       21613 non-null  int64  
 8   floors         21613 non-null  float64
 9   waterfront     21613 non-null  int64  
 10  view           21613 non-null  int64  
 11  condition      21613 non-null  int64  
 12  grade          21613 non-null  int64  
 13  sqft_above     21613 non-null  int64  
 14  sqft_basement  21613 non-null  int64  
 15  yr_built       21613 non-null  int64  
 16  yr_renovated   21613 non-null  int64  
 17  zipcode        21613 non-null  int64  
 18  lat            21613 non-null  float64
 19  long           21613 non-null  float64
 20  sqft_living15  21613 non-null  int64  
 21  sqft_lot15     21613 non-null  int64  
 '''
# We use the method describe to obtain a statistical summary of the dataframe.
df.describe()

#Module 2: Data Wrangling
#Question 2
'''
Drop the columns "id" and "Unnamed: 0" from axis 1 using the method drop(), then use the method describe() to obtain a statistical summary of the data.
Make sure the inplace parameter is set to True. Take a screenshot of your code and output. You will need to submit the screenshot for the final project.
'''
df.drop(["id", "Unnamed: 0"] , axis = 1, inplace = True)
df.describe()

# We can see we have missing values for the columns  bedrooms and  bathrooms  from df.info() output above
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
'''
number of NaN values for the column bedrooms : 13
number of NaN values for the column bathrooms : 10
'''

'''
We can replace the missing values of the column 'bedrooms' with the mean of the column 'bedrooms'  using the method replace().
We also replace the missing values of the column 'bathrooms' with the mean of the column 'bathrooms'  using the method replace().
Don't forget to set the inplace parameter to True
'''

mean = df['bedrooms'].mean()
df['bedrooms'].replace(np.nan , mean, inplace = True)

mean_bathrooms= df['bathrooms'].mean()
df['bathrooms'].replace(np.nan , mean_bathrooms , inplace = True)

# Now print

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
'''
number of NaN values for the column bedrooms : 0
number of NaN values for the column bathrooms : 0
'''
