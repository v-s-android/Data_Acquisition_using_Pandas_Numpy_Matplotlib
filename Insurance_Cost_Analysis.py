'''
Perform analytics operations on an insurance database that uses the below mentioned parameters.

Parameter	      Description	    Content type
age	            Age in years	-    integer
gender	        Male or Female	-    integer (1 or 2)
bmi	            Body mass index	  -  float
no_of_children	Number of children	- integer
smoker	        Whether smoker or not	- integer (0 or 1)
region	        Which US region - NW, NE, SW, SE	-  integer (1,2,3 or 4 respectively)
charges	        Annual Insurance charges in USD	   -  float
'''

'''
Objectives¶

- Load the data as a pandas dataframe
- Clean the data, taking care of the blank entries
- Run exploratory data analysis (EDA) and identify the attributes that most affect the charges
- Develop single variable and multi variable Linear Regression models for predicting the charges
- Use Ridge regression to refine the performance of Linear regression models.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
file_name = "insurance.csv"
await download(filepath, file_name )

df = pd.read_csv(file_name, header=None)
df.head()
# Add the headers to the dataframe, as mentioned in the project scenario.
headers = ["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]
df.columns = headers
df.head()

# cleaning the data
# Now, replace the '?' entries with 'NaN' values. 
df.replace( '?' , np.nan, inplace = True)

# Task 2 : Data Wrangling
# Use dataframe.info() to identify the columns that have some 'Null' (or NaN) information.
df.info()
'''
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   age             2767 non-null   object 
 1   gender          2771 non-null   int64  
 2   bmi             2771 non-null   float64
 3   no_of_children  2771 non-null   int64  
 4   smoker          2764 non-null   object 
 5   region          2771 non-null   int64  
 6   charges         2771 non-null   float64
'''
# Handle missing data:
'''
- For continuous attributes (e.g., age), replace missing values with the mean.
- For categorical attributes (e.g., smoker), replace missing values with the most frequent value.
- Update the data types of the respective columns.
- Verify the update using df.info().
'''

print(df['smoker'].value_counts())
'''
smoker
0    2201
1     563
'''
smoker_frequency = df['smoker'].value_counts().idxmax()
df['smoker'] = df['smoker'].replace(np.nan , smoker_frequency)

mean_age = df['age'].astype('float').mean(axis = 0) # using astype as int gives an error
df['age'] = df['age'].fillna(mean_age)
# or you can do df['age'].fillna(mean_age, inplace=True)
'''
axis=0 → perform the operation down the rows (vertically), producing a result for each column.
axis=1 → perform the operation across the columns (horizontally), producing a result for each row.
'''
# Update data types
df[["age","smoker"]] = df[["age","smoker"]].astype('int')

'''
Also note, that the `charges` column has values which are more than 2 decimal places long. Update the `charges` column such that all values are rounded 
to nearest 2 decimal places. Verify conversion by printing the first 5 values of the updated dataframe.
'''
df['charges'] = np.round(df['charges'],2)
df.head()
