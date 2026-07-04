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
from scipy import stats

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

sns.regplot(x="stroke", y="price", data = df)
df[["stroke","price"]].corr()

# Categorical Variables
'''
These are variables that describe a 'characteristic' of a data unit, and are selected from a small group of categories. 
The categorical variables can have the type "object" or "int64". A good way to visualize categorical variables is by using boxplots.
'''
sns.boxplot( x = 'body-style', y = 'price', data= df)

sns.boxplot(x="engine-location", y="price", data=df)
# Let's examine "drive-wheels" and "price".
sns.boxplot( x = 'drive-wheels', y= 'price', data = df)

# Descriptive Statistical Analysis
df.describe()
df.describe(include=['object'])

# value_counts(): It can br ONLY applied on column(panda series) and not the entire dataframe df
'''
Value counts is a good way of understanding how many units of each characteristic/variable we have. 
'''
df['drive-wheels'].value_counts()
#We can convert the series to a dataframe as follows:
df['drive-wheels'].value_counts().to_frame()
'''
	          count
drive-wheels	
fwd			   118
rwd				75
4wd				8
'''
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.reset_index(inplace=True)
drive_wheels_counts=drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'})
print(drive_wheels_counts)
# let's rename the index to 'drive-wheels':
drive_wheels_counts.index.name = "drive-wheels"
print(drive_wheels_counts)
'''
				value_counts	count
drive-wheels		
0					fwd			118
1					rwd			75
2					4wd			8
'''
# We can repeat the above process for the variable 'engine-location'.
# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)
'''
	            count
engine-location	
front	 		198
rear			3
'''

# Grouping
df['drive-wheels'].unique() # array(['rwd', 'fwd', '4wd'], dtype=object)
# We can select the columns 'drive-wheels', 'body-style' and 'price', then assign it to the variable "df_group_one".

group_1_df = df[['drive-wheels','body-style','price']]
new_group_1= group_1_df.groupby(df['drive-wheels'], as_index = False).agg({'price' : 'mean'})
print(new_group_1)

#let's group by both 'drive-wheels' and 'body-style

group_2 = df[['drive-wheels','body-style','price']]
new_group_2 = group_2.groupby(df[['drive-wheels','body-style']]).mean()
print(new_group_2)

# pivoting the df
pivot_df = new_group_1.pivot(index = 'drive-wheels', columns = 'body-style')
pivot_df = pivot_df.fillna(0) #fill missing values with 0
print(pivot_df)

# Use the "groupby" function to find the average "price" of each car based on "body-style".
df_3 = df[['body-style','price']]
grouped_data_3= df_3.groupby(['body-style'], as_index=False ).mean()
grouped_data_3

# heat map to visualize the relationship between Body Style vs Price.
#use the grouped results
plt.pcolor(pivot_df, cmap='RdBu')
plt.colorbar()
plt.show()

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

# Correlation and Causation
'''
Pearson Correlation

The Pearson Correlation measures the linear dependence between two variables X and Y.

The resulting coefficient is a value between -1 and 1 inclusive, where:

1: Perfect positive linear correlation.
0: No linear correlation, the two variables most likely do not affect each other.
-1: Perfect negative linear correlation.

Pearson Correlation is the default method of the function "corr". Like before, we can calculate the Pearson Correlation of the of the 'int64' or 'float64' variables.
'''
df.select_dtypes(include = ["number"]).corr()

#calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'.
# from scipy import stats
pearson_coff, p_value = stats.pearsonr(df['wheel-base'],df['price'])
print("the pearson-coeff is", pearson_coff, "with p_value" , p_value)
'''
The Pearson Correlation Coefficient is 0.5846418222655085  with a P-value of P = 8.076488270732338e-20
Since the p-value is <0.001, the correlation between wheel-base and price is statistically significant, although the linear relationship isn't extremely strong (~0.585).
'''

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
'''
The Pearson Correlation Coefficient is 0.6906283804483643  with a P-value of P =  8.016477466158871e-30
Conclusion:
Since the p-value is <0.001, the correlation between length and price is statistically significant, and the linear relationship is moderately strong (~0.691).
'''

# calculate the Pearson Correlation Coefficient and P-value of 'width' and 'price':

pearson_coef, p_value = stats.pearsonr(df['width'],df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value) 
'''
The Pearson Correlation Coefficient is 0.7512653440522663  with a P-value of P = 9.200335510485071e-38
Since the p-value is < 0.001, the correlation between width and price is statistically significant, and the linear relationship is quite strong (~0.751).
'''

pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value) 
'''
The Pearson Correlation Coefficient is -0.6865710067844684  with a P-value of P =  2.3211320655672357e-29
Since the p-value is  0.001, the correlation between city-mpg and price is statistically significant, and the coefficient of about -0.687 shows
that the relationship is negative and moderately strong.
'''
