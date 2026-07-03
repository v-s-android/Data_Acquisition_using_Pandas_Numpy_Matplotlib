'''
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   Unnamed: 0      238 non-null    int64  
 1   Manufacturer    238 non-null    object 
 2   Category        238 non-null    int64  
 3   Screen          238 non-null    object 
 4   GPU             238 non-null    int64  
 5   OS              238 non-null    int64  
 6   CPU_core        238 non-null    int64  
 7   Screen_Size_cm  234 non-null    float64
 8   CPU_frequency   238 non-null    float64
 9   RAM_GB          238 non-null    int64  
 10  Storage_GB_SSD  238 non-null    int64  
 11  Weight_kg       233 non-null    float64
 12  Price           238 non-null    int64  
dtypes: float64(3), int64(8), object(2)
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyodide.http import pyfetch

# download the dataset
async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

file_path_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod1.csv"
file_name = "laptops.csv"

await download(file_path_url , file_name) # this downloads the file

#load data into a pandas.DataFrame
df = pd.read_csv(file_name, header = 0)

# Verify loading by displaying the dataframe summary using dataframe.info()
print(df.info())

# update the Screen_Size_cm column such that all values are rounded to nearest 2 decimal places by using numpy.round()
df['Screen_Size_cm'] = numpy.round(df['Screen_Size_cm'], 2)
'''
0    35.56
1    39.62
2    39.62
3    33.78
4    39.62
'''
# Task 1: Evaluate the dataset for missing data : code that identifies which columns have missing data.
df.replace('?', np.nan, inplace = True)
missing_data_df = df.isnull()
for each_column in missing_data_df.columns.values.tolist():
    print(missing_data_df[each_column].value_counts())
    print("")
  '''
Screen_Size_cm
False    234
True       4
Name: count, dtype: int64
...
  '''

# Task 2: Replace with mean
'''
Missing values in attributes that have continuous data are best replaced using Mean value. We note that values in "Weight_kg" attribute are continuous in nature,
and some values are missing. Therefore, write a code to replace the missing values of weight with the average value of the attribute.
'''
# replacing missing data with mean
# astype() function converts the values to the desired data type
# axis=0 indicates that the mean value is to calculated across all column elements in a row.
avg_weight_kg = df['Weight_kg'].astype('float').mean(axis=0)
df.replace(np.nan , avg_weight_kg, inplace = True)

# Task 3: Replace with the most frequent value
'''
### Replace with the most frequent value
Missing values in attributes that have categorical data are best replaced using the most frequent value. We note that values in "Screen_Size_cm" attribute are categorical in nature,
and some values are missing. Therefore, write a code to replace the missing values of Screen Size with the most frequent value of the attribute.
'''
freq = df['Screen_Size_cm'].value_counts().idxmax()
print("Frequent item : ", freq)
df['Screen_Size_cm'].replace(np.nan , freq , inplace= True)
print(df['Screen_Size_cm'])

# Task 4 : Fixing the data types(Data Cleaning)
'''
Both "Weight_kg" and "Screen_Size_cm" are seen to have the data type "Object", while both of them should be having a data type of "float".
Write a code to fix the data type of these two columns.
'''
df["Weight_kg"] = df["Weight_kg"].astype('float')
df["Screen_Size_cm"] = df["Screen_Size_cm"].astype('float')
print(df[["Weight_kg","Screen_Size_cm"]].head())
# or simply df[["Weight_kg","Screen_Size_cm"]] = df[["Weight_kg","Screen_Size_cm"]].astype("float")

# Data Standardization
'''
The value of Screen_size usually has a standard unit of inches. Similarly, weight of the laptop is needed to be in pounds.
'''
# Data standardization: convert screen size from cm to inch
df['Screen_Size_cm'] = df['Screen_Size_cm'] / 2.54
# Data standardization: convert weight from kg to pounds
df['Weight_kg'] = df['Weight_kg'] * 2.205 
# Rename the columns
df.rename(columns = {'Screen_Size_cm':'Screen_Size_inch', 'Weight_kg': 'Weight_P'} , inplace = True)
print(df[['Screen_Size_inch','Weight_P']].head(10))

#Data Normalization
'''
Often it is required to normalize a continuous data attribute. Write a code to normalize the "CPU_frequency" attribute with respect to the maximum value available in the dataset.
just divide each element in column with max value of that column
'''
max_value_cpu_freq = df['CPU_frequency'].max()
df['CPU_frequency'] = df['CPU_frequency'] / max_value_cpu_freq
print(df['CPU_frequency'].head(10))
'''
0    0.551724
1    0.689655
2    0.931034
3    0.551724
'''

#Binning
'''
Binning is a process of creating a categorical attribute which splits the values of a continuous data into a specified number of GROUPS.
In this case, write a code to create 3 bins for the attribute "Price". These bins would be named "Low", "Medium" and "High". 
The new attribute will be named "Price-binned".
'''
bins = np.linspace(min(df["Price"]),max(df["Price"]), 4)
labels = ["Low", "Medium" , "High"]
df['Price-binned'] = pd.cut(df['Price'], bins, labels= labels, include_lowest = True)
print(df['Price-binned'])
'''
0         Low
...
235    Medium
236       Low
'''

# Plot a bar graph
#import matplotlib.pyplot as plt
plt.bar(labels, df['Price-binned'].value_counts())
plt.xlabel("Price")
plt.ylabel("group")
plt.title("Price Bar graph")

# Task - 6

### Indicator variables
'''
Convert the "Screen" attribute of the dataset into 2 indicator variables, "Screen-IPS_panel" and "Screen-Full_HD". Then drop the "Screen" attribute from the dataset.
'''
new_screen_df = pd.get_dummies(df['Screen'], prefix="Screen")
print(new_screen_df.head())
'''
   Screen_Full HD  Screen_IPS Panel
0           False              True
1            True             False
2            True             False
3           False              True
4            True             False
'''
df = pd.concat([df,new_screen_df], axis=1) # concatinate column wise axis=0 is for row, axis =1 is for operate on column

# drop original column "Screen" from "df"
df.drop("Screen", axis = 1, inplace=True)
df.head()

'''
   Unnamed: 0 Manufacturer  Category  GPU  OS  CPU_core  Screen_Size_inch  \
0           0         Acer         4    2   1         5         14.000000   
1           1         Dell         3    1   1         3         15.598425   
2           2         Dell         3    1   1         7         15.598425   
3           3         Dell         4    2   1         5         13.299213   
4           4           HP         4    2   1         7         15.598425   

   CPU_frequency  RAM_GB  Storage_GB_SSD  Weight_P  Price Price-binned  \
0       0.551724       8             256   3.52800    978          Low   
1       0.689655       4             256   4.85100    634          Low   
2       0.931034       8             256   4.85100    946          Low   
3       0.551724       8             128   2.69010   1244          Low   
4       0.620690       8             256   4.21155    837          Low   

   Screen_Full HD  Screen_IPS Panel  Screen_Full HD  Screen_IPS Panel  
0           False              True           False              True  
1            True             False            True             False  
2            True             False            True             False  
3           False              True           False              True  
4            True             False            True             False  
'''


