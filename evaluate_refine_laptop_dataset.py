'''
Objectives
- Use training, testing and cross validation to improve the performance of the dataset.
- Identify the point of overfitting of a model
- Use Ridge Regression to identify the change in performance of a model based on its hyperparameters
- Use Grid Search to identify the best performing model using different hyperparameters
'''

import piplite
await piplite.install('seaborn')

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

# Importing the Dataset
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv'
await download(filepath, "laptops.csv")
file_name="laptops.csv"
df = pd.read_csv( file_name , header = 0)
df.head()
#Drop the two unnecessary columns that have been added into the file, 'Unnamed: 0' and 'Unnamed: 0.1'. Use drop to delete these columns.
df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1, inplace=True))

# Task 1 : Using Cross validation to improve the model
'''
Divide the dataset into x_data and y_data parameters. Here y_data is the "Price" attribute, and x_data has all other attributes in the data set.
'''
y_data = df['Price']
x_data = df.drop(['Price'], axis = 1) # dropping price column so that we will have the rest of the data

# Split the data set into training and testing subests such that you reserve 10% of the data set for testing purposes.

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data , test_size=0.10, random_state=1)
print("Number of testing samples: " , x_test.shape[0])
print("Munber of training samples: ", x_train.shape[0])
'''
Number of testing samples:  24
Nunber of training samples:  214
'''
# Create a single variable linear regression model using "CPU_frequency" parameter. Print the R^2 value of this model for the training and testing subsets.

lr = LinearRegression()

lr.fit(x_train[['CPU_frequency']], y_train)

print("R^2 value of training subset", lr.score(x_train[['CPU_frequency']], y_train))
print("R^2 value of testing subset", lr.score(x_test[['CPU_frequency']], y_test))
'''
R^2 value of training subset 0.14829792099817962
R^2 value of testing subset -0.06599437350393766
'''
# Run a 4-fold cross validation on the model and print the mean value of R^2 score along with its standard deviation.
c_v_s = cross_val_score( lr,  # linear regression model
                x_train[['CPU_frequency']],
                y_train,
                cv = 4
               )
print("mean of the R^2 score ", c_v_s.mean())
print("standard deviation ", c_v_s.std())
'''
mean of the R^2 score  0.12738818019555026
standard deviation  0.08317058010912008
'''
