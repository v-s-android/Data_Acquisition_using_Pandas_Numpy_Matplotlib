"""
Handle missing values

Correct data formatting

Standardize and normalize data
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from pyodide.http import pyfetch

file_path_url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
file_name = "usedcars.csv"

async def download(file_path_url,file_name):
  response = await pyfetch(file_path_url)
  if response.status == 200:
    with open(file_name , 'wb') as file:
      file.write(response.bytes())

# call the download function: this downloads the data into usedcars.csv
await download(file_path_url , file_name)

# create a  list headers containing name of headers
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

# read the downloaded local csv and assign headers
# You can use df.columns = headers
df = pd.read_csv(file_name , names = headers) 

print(df.head())

#identify missing values
'''
Convert "?" to NaN
In the car data set, missing data comes with the question mark "?". We replace "?" with NaN (Not a Number), Python's default missing value marker for reasons of computational speed and convenience. Use the function:
.replace(A, B, inplace = True) 
to replace A by B.
'''
# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
print(df.head(5))

'''
Evaluating for Missing Data
The missing values are converted by default. Use the following functions to identify these missing values. You can use two methods to detect missing data:

.isnull()
.notnull()
The output is a boolean value indicating whether the value that is passed into the argument is in fact missing data.
"True" means the value is a missing value while "False" means the value is not a missing value.
'''
missing_data_df = df.isnull()
print(missing_data_df.head(5))

'''
Count missing values in each column
Using a for loop in Python, you can quickly figure out the number of missing values in each column. As mentioned above,
"True" represents a missing value and "False" means the value is present in the data set. In the body of the for loop the method ".value_counts()" counts 
the number of "True" values.
'''
for column in missing_data_df.columns.values.tolist():
  print(missing_data_df[column].value_counts())
  print("")

'''
output:
symboling
False    205
Name: count, dtype: int64

normalized-losses
False    164
True      41
Name: count, dtype: int64
'''

# Calculate the mean value for the "normalized-losses" column
avg_norm_losses = df["normalized-losses"].astype("float").mean(axis = 0)
print("Average/Mean of normalized-losses:",avg_norm_losses) # output: Average of normalized-losses: 122.0

#Replace NaN in "normalized-losses" column with avg_norm_losses
df["normalized-losses"].replace(np.nan , avg_norm_losses , inplace = True) # Alt: df["normalized-losses"] = df["normalized-losses"].replace(np.nan, avg_norm_loss)

#Calculate the mean valure if "bore" column and replace NaN with avg_bore_val
avg_bore_val = df["bore"].astype("float").mean(axis = 0)
print("Average of bore: ", avg_bore_val) # Average of bore: 3.3297512437810943
df["bore"] = df["bore"].replace( np.nan ,avg_bore_val) 
print(df["bore"],head(5))

'''
output: 
0    3.47
1    3.47
2    2.68
3    3.19
4    3.19
Name: bore, dtype: object
'''



