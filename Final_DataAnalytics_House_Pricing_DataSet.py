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

'''
Module 3: Exploratory Data Analysis
Question 3
Use the method value_counts to count the number of houses with unique floor values, use the method .to_frame() to convert it to a data frame. 
'''
df['floors'].value_counts().to_frame()

'''
Question 4
Use the function boxplot in the seaborn library to determine whether houses with a waterfront view or without a waterfront view have more price outliers.
'''
sns.boxplot(x = "waterfront" , y = "price" , data = df)

'''
Question 5
Use the function regplot in the seaborn library to determine if the feature sqft_above is negatively or positively correlated with price. 
'''
sns.regplot(x = "sqft_above" , y = "price", data = df)

# We can use the Pandas method corr() to find the feature other than price that is most correlated with price.
df_numeric = df.select_dtypes(include=[np.number])
df_numeric.corr()['price'].sort_values()
'''
zipcode         -0.053203 # except zipcode, everything else is posivilty correlated
long             0.021626
condition        0.036362
yr_built         0.054012
sqft_lot15       0.082447
sqft_lot         0.089661
yr_renovated     0.126434
floors           0.256794
waterfront       0.266369
lat              0.307003
bedrooms         0.308797
sqft_basement    0.323816
view             0.397293
bathrooms        0.525738
sqft_living15    0.585379
sqft_above       0.605567
grade            0.667434
sqft_living      0.702035
price            1.000000
Name: price, dtype: float64
'''

'''
Module 4: Model Development
We can Fit a linear regression model using the longitude feature 'long' and caculate the R^2.
'''
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y) # 0.00046769430149007363
'''
Question 6
Fit a linear regression model to predict the 'price' using the feature 'sqft_living' then calculate the R^2. 
'''
lm.fit(df[['sqft_living']], Y)
print("R^2: ", lm.score(df[['sqft_living']] , Y)) # R^2:  0.4928532179037931

'''
Question 7
Fit a linear regression model to predict the 'price' using the list of features,
Then calculate the R^2. Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project.
'''
features = df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]
lm.fit(features , Y)
print("R^2: ", lm.score(features, Y)) # R^2:  0.6576890354915759

'''
This will help with Question 8:
Create a list of tuples, the first element in the tuple contains the name of the estimator:

'scale'
'polynomial'
'model'

The second element in the tuple contains the model constructor

StandardScaler()
PolynomialFeatures(include_bias=False)
LinearRegression()
'''
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

'''
Question 8
Use the list to create a pipeline object to predict the 'price', fit the object using the features in the list features, and calculate the R^2. 
'''
from sklearn.metrics import r2_score

pipe = Pipeline(Input)
features = features.astype(float) # converting all values to float 
pipe.fit(features, df['price'])
y_hat = pipe.predict(features)
print("r2_score is :", r2_score(df['price'], y_hat)) # r2_score is : 0.7512051345272872

'''
Module 5: Model Evaluation and Refinement
Import the necessary modules:
'''
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# We will split the data into training and testing sets:

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])
'''
number of test samples: 3242
number of training samples: 18371
'''

'''
Question 9
Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data.
'''
from sklearn.linear_model import Ridge
RidgeModel = Ridge(alpha = 0.1)
RidgeModel.fit(x_train , y_train)
y_predict = RidgeModel.predict(x_test)
print("r2_score is: ", r2_score(y_test, y_predict)) # r2_score is:  0.647875916393907

'''
Question 10
Perform a second order polynomial transform on both the training data and testing data. Create and fit a Ridge regression object using the training data,
set the regularisation parameter to 0.1, and calculate the R^2 utilising the test data provided.
'''
pr = PolynomialFeatures(degree = 2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RR = Ridge(alpha = 0.1)
RR.fit(x_train_pr, y_train)
y_rr_predict = RR.predict(x_test_pr)
print("r2_score", r2_score(y_test, y_rr_predict)) # r2_score 0.7002744263583341
