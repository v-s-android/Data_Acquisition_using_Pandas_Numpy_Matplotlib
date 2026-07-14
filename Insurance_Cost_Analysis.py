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
