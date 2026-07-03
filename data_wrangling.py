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

#To see which values are present in a particular column, we can use the ".value_counts()" method:
df['num-of-doors'].value_counts()
'''
num-of-doors
four    114
two      89
Name: count, dtype: int64
'''

#You can see that four doors is the most common type. We can also use the ".idxmax()" method to calculate the most common type automatically:
df['num-of-doors'].value_counts().idxmax() # output: 'four'

#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"] = df["num-of-doors"].replace(np.nan, "four")

#Finally, drop all rows that do not have price data:
# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)
print(df.head())

df.dtypes
'''
normalized-losses     object
bore                  object
stroke                object
peak-rpm              object
price                 object
'''
# Convert data types to proper format¶ : All numercial values need to be changed from object to float/int
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

# Data Standardization
# Standardization is the process of transforming data into a common format, allowing the researcher to make the meaningful comparison.
# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]
# check your transformed data 
df.head()

#transform mpg to L/100km in the column of "highway-mpg" and change the name of column to "highway-L/100km".
# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df["highway-mpg"] = 235/df["highway-mpg"]

# rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)

# check your transformed data 
df.head()

#Data Normalization
df['length'] = df['length']/df['length'].max()
#similary
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()

df[["length","width","height"]].head() # print these columns

# Binning
# Binning is a process of transforming continuous numerical variables into discrete categorical 'bins' for grouped analysis.
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
'''
Find 3 bins of equal size bandwidth by using Numpy's linspace(start_value, end_value, numbers_generated function.
Since you want to include the minimum value of horsepower, set start_value = min(df["horsepower"]).
Since you want to include the maximum value of horsepower, set end_value = max(df["horsepower"]).
Since you are building 3 bins of equal length, you need 4 dividers, so numbers_generated = 4.
'''
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
print(bins)

#Set group names:
group_names = ['Low', 'Medium', 'High']
# Apply the pandas function "cut" to determine what each value of `df['horsepower']` belongs to. 

df['horsepower-binned'] = pd.cut( df['horsepower'], bins , label = group_names, include_lowest = True)

print(df[['horsepower-binned', 'horsepower']])
'''
	horsepower	horsepower-binned
0	111	Low
1	111	Low
2	154	Medium
3	102	Low
4	115	Low
'''
df["horsepower-binned"].value_counts()
'''
horsepower-binned
Low       153
Medium     43
High        5
Name: count, dtype: int64
'''

# Plot the distribution of each bin:

pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

# Indicator Variable

# The column "fuel-type" has two unique values: "gas" or "diesel". Regression doesn't understand words, only numbers.
#To use this attribute in regression analysis, you can convert "fuel-type" to indicator variables.

fuel_df = pd.get_dummies(df['fuel-type'])
print(fuel_df.head())
'''
diesel	gas
0	False	True
1	False	True
2	False	True
3	False	True
4	False	True
'''

# Change the column names for clarity:
fuel_df.rename( columns = {'diesel' : 'fuel-type-diesel', 'gas' : 'fuel-type-gas' }, inplace= True)
print(fuel_df.head())
'''
	fuel-type-diesel	fuel-type-gas
0	  False	  True
1	  False	  True
2	  False	  True
3	  False	  True
4	  False	  True
'''

# In the data frame, column 'fuel-type' now has values for 'gas' and 'diesel' as 0s and 1s.

# merge data frame "df" and "fuel_df" 
df = pd.concat([df, fuel_df], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

# Convert aspirations
df['aspiration'].value_counts()

aspiration_df = pd.get_dummies(df['aspiration'])
print(aspiration_df.head())
'''
	std	turbo
0	True	False
1	True	False
2	True	False
3	True	False
4	True	False
'''

# beak into columns
aspiration_df.rename(columns = {'std' : 'aspiration-std', 'turbo' : 'aspiration-turbo'}, inplace = True)
print(aspiration_df.head())
'''
	aspiration-std	aspiration-turbo
0	  True	  False
1	  True	  False
2	  True	  False
3	  True	  False
4	  True	  False
'''
# merge the new dataframe to the original datafram
df = pd.concat([df , aspiration_df ], axis =1)
#  drop original column "aspiration" from "df"
df.drop("aspiration", axis = 1, inplace = True)
df.to_csv('clean_df.csv') # creates a new .csv file 
