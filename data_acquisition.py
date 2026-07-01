import pandas as pd
import numpy as np
from pyodide.http import pyfetch

# The functions below will download the dataset
async def download(file_path_url, filename):
  response = await pyfetch(url)
  if response.status == 200:
    with open(filename, 'wb') as file:
      file.write(await response.bytes())

file_path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv'
file_name = "auto.csv"
# obtain the dataset
await download(file_path , file_name)

#Utilize the Pandas method read_csv() to load the data into a dataframe.
df = pd.read_csv(file_name)

# show the first 5 rows using dataframe.head() method
print("The first 5 rows of the dataframe") 
print(df.head(5))

# Bottom 10 rows
print(df.tail(10))

# Adding header : create a list "headers"
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df.columns = headers
print(df.head(3))

# we need to replace the "?" symbol with NaN so the dropna() can remove the missing values
df1 = df.replace( '?' , np.NaN)
# axis=0 means that the contents along the entire row will be dropped wherever the entity 'price' is found to be NaN
df = df1.dropna(subset=["price"], axis=0)
df.head(10)

# This method will provide various summary statistics, excluding NaN (Not a Number) values.
print(df.describe())

# describe all the columns in "df" 
df.describe(include = "all")

# dataframe[[' column 1 ',column 2', 'column 3'] ].describe() statistics for selective columns
df[['symboling','normalized-losses', 'fuel-type']].describe()
df[['length', 'compression-ratio']].describe()

# This method prints information about a data frame including the index dtype and columns, non-null values and memory usage.
df.info()




