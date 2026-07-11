'''
Evaluate and refine prediction models
'''
'''
install the following libraries
import piplite
await piplite.install(['pandas'])
await piplite.install(['matplotlib'])
await piplite.install(['scipy'])
await piplite.install(['scikit-learn'])
await piplite.install(['seaborn'])
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())
          
file_url_path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'
file_name = 'module_5_auto.csv'

await download(file_url_path ,file_name)
df = pd.read_csv(file_name, header=0)
print(df.head())

# First, let's only use numeric data
df = df._get_numeric_data()
df.head()

# remove the columns 'Unnamed:0.1' and 'Unnamed:0' since they do not provide any value to the models.
df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis = 1, inplace = True)
df.head()

# Libraries for plotting:
%pip install micropip
import micropip
await micropip.install(['ipywidgets'], keep_going=True)
import ipywidgets
from ipywidgets import interact, interactive, fixed, interact_manual

# Functions for Plotting

def DistributionPlot(Redfunction, Bluefunction, Redname, Bluename, title):
  width = 6
  height = 8
  plt.figure(figsize = (width, height))
  ax1 = sns.kdeplot(Redfunction , color = 'r' , label = Redname)
  ax2 = sns.kdeplot(Bluefunction , color = 'b', label = Bluename, ax = ax1)
  plt.title(title)
  plt.xlabel('Price (in dollars)')
  plt.ylabel('Proportion of Cars')
  plt.show()
  plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()

'''
Part 1: Training and Testing
An important step in testing your model is to split your data into training and testing data. We will place the target data price in a separate dataframe y_data:
'''
y_data = df['price'] # This contains the target values (labels or dependent variable).

#now drop the price column in df and assign the table to x_data
x_data = df.drop('price', axis = 1) # This contains your input features (independent variables).
x_data.head()

'''
***Now, we randomly split our data into training and testing data using the function train_test_split.***
'''
from sklearn.model_selection import train_test_split
# x_data is the table without the price
# y_data is just the price column
# The test_size parameter sets the proportion of data that is split into the testing set. In the above, the testing set is 10% of the total dataset.
# random_state : Without it, each time you run the code, the data is split differently.and  with it you get the same split every time, making your results reproducible.
x_train, x_test, y_train , y_test = train_test_split( x_data, y_data , test_size = 0.1, random_state = 1)


print("number of test sample ", x_test.shape[0])
print("number of train sample", x_train.shape[0])
'''
number of test samples : 21
number of training samples: 180

test_size=0.10

This means:

10% of the data goes into the test set.
90% goes into the training set.

For example, if you have 1,000 samples:

x_train → 900 samples
x_test → 100 samples
y_train → 900 labels
y_test → 100 labels
'''
# Similarly Q1: Use the function "train_test_split" to split up the dataset such that 40% of the data samples will be utilized for testing. Set the parameter "random_state" equal to zero.
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data,y_data, test_size= 0.4, random_state = 0)
print("number of test samples :", x_test1.shape[0])
print("number of training samples:",x_train1.shape[0])
'''
number of test samples : 81
number of training samples: 120
'''

# Let's import LinearRegression from the module linear_model.
from sklearn.linear_model import LinearRegression

lre = LinearRegression()
# We fit the model using the feature "horsepower":
lre.fit(x_train[['horsepower']], y_train) # training data

# calculate the R^2 on the test data:
print(lre.score(x_test[['horsepower']] , y_test)) # 0.3635875575078824
# calculate the R^2 on the train data:
print(lre.score(x_train[['horsepower']], y_train)) # 0.6619724197515103
'''
We can see the R^2 is much smaller using the test data compared to the training data.
'''
# Q2 : Find the R^2 on the test data using 40% of the dataset for testing.
# x_data is the table without the price
# y_data is just the price column
# The test_size parameter sets the proportion of data that is split into the testing set. In the above, the testing set is 10% of the total dataset.
x_train1, x_test1, y_train1, y_test1 = train_test_split( x_data, y_data, test_size = 0.4, random_state = 0)

lre1 = LinearRegression()

lre1.fit(x_train1[['horsepower']], y_train1)

print("test data R^2 value ", lre1.score(x_test1[['horsepower']], y_test1)) # test data R^2 value  0.7139364665406973

print("train data R^2 value ", lre1.score(x_train1[['horsepower']], y_train1)) # train data R^2 value  0.5754067463583004
'''
R^2 value of test data is more than R^2 value of train data
'''

#Cross-Validation Score

# Let's import cross_val_score from the module model_selection.
from sklearn.model_selection import cross_val_score

# We input the object, the feature ("horsepower"), and the target data (y_data). The parameter 'cv' determines the number of folds. In this case, it is 4.
r_cross = cross_val_score( lre , x_data[['horsepower']] , y_data , cv = 4)
# The default scoring is R^2. Each element in the array has the average R^2 value for the fold:
print(r_cross) # [0.7746232  0.51716687 0.74785353 0.04839605]

# We can calculate the average and standard deviation of our estimate:
print("The mean of the folds are ", r_cross.mean())
print("standard deviation ", r_cross.std())
'''
The mean of the folds are 0.5220099150421197 and the standard deviation is 0.29118394447560203
'''
# NEXT, We can use negative squared error as a score by setting the parameter 'scoring' metric to 'neg_mean_squared_error'.

print("the score (neg_mean_squared_error) " , -1 * cross_val_score( lre , x_data[['horsepower']], y_data, cv = 4, scoring = 'neg_mean_squared_error'))
# [20254142.84026702 43745493.26505171 12539630.34014929 17561927.72247586]

# Q3: Calculate the average R^2 using two folds, then find the average R^2 for the second fold utilizing the "horsepower" feature:
#The R2 score is nothing but the cross_val_score
rc = cross_val_score( lre, x_data[['horsepower']], y_data, cv = 2)
print(rc.mean())
'''
0.5166761697127429
'''


