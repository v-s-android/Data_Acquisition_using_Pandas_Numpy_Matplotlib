import pandas as pd
import numpy as np
import piplite
await piplite.install('seaborn')
from pyodide.http import pyfetch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#doenload the file from url
async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

file_path_url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"
file_name = 'laptops.csv'
await download(file_path_url, file_name)

df = pd.read_csv(file_name, header = 0) #The header parameter tells Pandas which row contains the column names. eg: make,model,price (column names)
df.head()

# Task 1 - Visualize individual feature patterns

### Continuous valued features
'''
Generate regression plots for each of the parameters "CPU_frequency", "Screen_Size_inch" and "Weight_pounds" against "Price".
Also, print the value of correlation of each feature with "Price".
'''
sns.regplot( x = "CPU_frequency", y= "Price", data = df)
plt.ylim(0,)
df[['CPU_frequency','Price']].corr()

sns.regplot(x = "Screen_Size_inch", y = "Price", data = df)
plt.ylim(0,)
df[['Screen_Size_inch','Price']].corr()

sns.regplot(x='Weight_pounds', y='Price', data = df)
plt.ylim(0,)
df[['Weight_pounds', 'Price']].corr()

#Alternatively to find the corr between the parameters and Price
for param in ["CPU_frequency", "Screen_Size_inch","Weight_pounds"]:
    print(f"Correlation of Price and {param} is ", df[[param,"Price"]].corr())

### Categorical features
'''
Generate Box plots for the different feature that hold categorical values. These features would be "Category", "GPU", "OS", "CPU_core", "RAM_GB", "Storage_GB_SSD"
'''
sns.boxplot(x = "Category", y ="Price", data=df)
sns.boxplot(x = "GPU", y ="Price", data=df)
sns.boxplot(x = "OS", y ="Price", data=df)
sns.boxplot(x = "CPU_core", y ="Price", data=df)
sns.boxplot(x = "RAM_GB", y ="Price", data=df)
sns.boxplot(x = "Storage_GB_SSD", y ="Price", data=df)

#Task 2 - Descriptive Statistical Analysis
'''
Generate the statistical description of all the features being used in the data set. Include "object" data types as well.
'''
df.describe()
df.describe(include = ["object"])
'''
	    Manufacturer	Price-binned
count	    238	          238
unique	   11	            3
top	      Dell	        Low
freq	     71          	160
'''

# Task 3 - GroupBy and Pivot Tables
'''
Group the parameters "GPU", "CPU_core" and "Price" to make a pivot table and visualize this connection using the pcolor plot.
'''
new_df = df[["GPU", "CPU_core", "Price"]]
grouped_new_df = new_df.groupby(["GPU", "CPU_core"],as_index=False).mean()
print(grouped_new_df)
'''
	GPU	CPU_core	Price
0	1	  3	     769.250000
1	1	  5	     998.500000
2	1	  7	    1167.941176
3	2	  3	    785.076923
4	2	  5	    1462.197674
5	2	  7	    1744.621622
6	3	  3	    784.000000
7	3	  5	    1220.680000
8	3	  7	    1945.097561
'''
# Create the Pivot table
pivoted_df = grouped_new_df.pivot(index = 'GPU', columns = "CPU_core")
print(pivoted_df)

#create the heat map
plt.pcolor(pivoted_df, cmap = "RdBu")
plt.colorbar()
plt.show()

# or alternatively add the labels to x axis and y axis
fig, ax = plt.subplots()
im = ax.pcolor(pivoted_df, cmap='RdBu')

#label names
row_labels = pivoted_df.columns.levels[1]
col_labels = pivoted_df.index

#move ticks and labels to the center
ax.set_xticks(np.arange(pivoted_df.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(pivoted_df.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

#Task 4 - Pearson Correlation and p-values
'''
Use the scipy.stats.pearsonr() function to evaluate the Pearson Coefficient and the p-values for each parameter tested above.
This will help you determine the parameters most likely to have a strong effect on the price of the laptops.
'''

perason_coeff, p_value = stats.pearsonr(df['GPU'],df['Price'])
print("the pearson-coeff is", perason_coeff, "with p_value" , p_value)

# or Alternatively
colmn_list = ['RAM_GB','CPU_frequency','Storage_GB_SSD','Screen_Size_inch','Weight_pounds','CPU_core','OS','GPU','Category']

for col in colmn_list:
  perason_coeff, p_value = stats.pearsonr(df['GPU'],df['Price'])
  print(f"the pearson-coeff is {col} is {perason_coeff} with p_value : {p_value} ")

'''the pearson-coeff is RAM_GB is 0.2882981988881427 with p_value : 6.166949698364507e-06 
the pearson-coeff is CPU_frequency is 0.2882981988881427 with p_value : 6.166949698364507e-06 
the pearson-coeff is Storage_GB_SSD is 0.2882981988881427 with p_value : 6.166949698364507e-06 
the pearson-coeff is Screen_Size_inch is 0.2882981988881427 with p_value : 6.166949698364507e-06 
the pearson-coeff is Weight_pounds is 0.2882981988881427 with p_value : 6.166949698364507e-06 
the pearson-coeff is CPU_core is 0.2882981988881427 with p_value : 6.166949698364507e-06 
the pearson-coeff is OS is 0.2882981988881427 with p_value : 6.166949698364507e-06 
the pearson-coeff is GPU is 0.2882981988881427 with p_value : 6.166949698364507e-06 
the pearson-coeff is Category is 0.2882981988881427 with p_value : 6.166949698364507e-06 
'''
  
  
