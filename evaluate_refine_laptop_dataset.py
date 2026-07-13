'''
Objectives
- Use training, testing and cross validation to improve the performance of the dataset.
- Identify the point of overfitting of a model
- Use Ridge Regression to identify the change in performance of a model based on its hyperparameters
- Use Grid Search to identify the best performing model using different hyperparameters
'''

import piplite
await piplite.install('seaborn')

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

# Importing the Dataset
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv'
await download(filepath, "laptops.csv")
file_name="laptops.csv"
df = pd.read_csv( file_name , header = 0)
df.head()
#Drop the two unnecessary columns that have been added into the file, 'Unnamed: 0' and 'Unnamed: 0.1'. Use drop to delete these columns.
df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1, inplace=True))

# Task 1 : Using Cross validation to improve the model
'''
Divide the dataset into x_data and y_data parameters. Here y_data is the "Price" attribute, and x_data has all other attributes in the data set.
'''
y_data = df['Price']
x_data = df.drop(['Price'], axis = 1) # dropping price column so that we will have the rest of the data

# Split the data set into training and testing subests such that you reserve 10% of the data set for testing purposes.

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data , test_size=0.10, random_state=1)
print("Number of testing samples: " , x_test.shape[0])
print("Munber of training samples: ", x_train.shape[0])
'''
Number of testing samples:  24
Nunber of training samples:  214
'''
# Create a single variable linear regression model using "CPU_frequency" parameter. Print the R^2 value of this model for the training and testing subsets.

lr = LinearRegression()

lr.fit(x_train[['CPU_frequency']], y_train)

print("R^2 value of training subset", lr.score(x_train[['CPU_frequency']], y_train))
print("R^2 value of testing subset", lr.score(x_test[['CPU_frequency']], y_test))
'''
R^2 value of training subset 0.14829792099817962
R^2 value of testing subset -0.06599437350393766
'''
# Run a 4-fold cross validation on the model and print the mean value of R^2 score along with its standard deviation.
c_v_s = cross_val_score( lr,  # linear regression model
                x_train[['CPU_frequency']],
                y_train,
                cv = 4
               )
print("mean of the R^2 score ", c_v_s.mean())
print("standard deviation ", c_v_s.std())
'''
mean of the R^2 score  0.12738818019555026
standard deviation  0.08317058010912008
'''

#Task 2: Overfitting
'''
Split the data set into training and testing components again, this time reserving 50% of the data set for testing.
'''
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.5, random_state = 0)
x_train.head() # just for checking

'''
To identify the point of overfitting the model on the parameter "CPU_frequency", you'll need to create polynomial features using the single attribute.
You need to evaluate the R^2 scores of the model created using different degrees of polynomial features, ranging from 1 to 5. Save this set of values of R^2 score as a list.
'''

order = [1,2,3,4,5]
lr = LinearRegression()
res = []

for n in order:
    pr = PolynomialFeatures(degree = n)
    x_train_pr = pr.fit_transform(x_train[['CPU_frequency']])
    x_test_pr = pr.fit_transform(x_test[['CPU_frequency']])
    lr.fit(x_train_pr, y_train) # fit the training set
    # find the R^2 score on the testing set
    res.append(lr.score(x_test_pr, y_test))

print("R^2 scores list ", res) # R^2 scores list  [0.05322174176198158, -0.026920818679001313, 0.05156345792558126, -0.9948137915492898, -1.3759360955769684]

# Plot the values of R^2 scores against the order. Note the point where the score drops.
# import matplotlib.pyplot as plt

plt.plot(order, r_square_test)
plt.xlabel("Order")
plt.ylabel("R^2 scores")
plt.title("R^2 Using Test Data")
plt.show() # the score drops at 3

# Task 3 : Ridge Regression

'''
Now consider that you have multiple features, i.e. 'CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS','GPU' and 'Category'.
Create a polynomial feature model that uses all these parameters with degree=2. Also create the training and testing attribute sets.
'''
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.5, random_state = 0)
pr = PolynomialFeatures(degree = 2)

x_train_pr = pr.fit_transform(x_train[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS','GPU', 'Category']])
x_test_pr = pr.fit_transform(x_test[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS','GPU', 'Category']])

'''
Create a Ridge Regression model and evaluate it using values of the hyperparameter alpha ranging from 0.001 to 1 with increments of 0.001.
Create a list of all Ridge Regression R^2 scores for training and testing data.
'''
r_square_train = []
r_square_test = []

alpha_number = np.arange(0.001 , 1, 0.001) # 0.001 increment
progress_bar = tqdm(alpha_number)
# print(alpha_number) [0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01  0.011 0.012 0.013 0.014 0.015 0.016 ..]

for alpha in progress_bar :
    RidgeModel = Ridge(alpha = alpha) # alpha increments 0.001 each time
    RidgeModel.fit(x_train_pr, y_train)

    # R^2 scores for training and testing data.
    train_score = RidgeModel.score(x_train_pr, y_train)
    test_score = RidgeModel.score(x_test_pr , y_test)
    
    progress_bar.set_postfix({"train score" : train_score, "Test score": test_score }) # not required, but good to have
    
    r_square_train.append(train_score)
    r_square_test.append(test_score)

# Plot the R^2 values for training and testing sets with respect to the value of alpha
plt.figure(figsize = (8,6))
plt.plot(alpha_number , r_square_test , label = 'validation data')
plt.plot(alpha_number, r_square_train, 'r' , label = 'training data')
plt.xlabel('alpha')
plt.ylabel('R^2 values')
plt.title('R^2 using Training and testing data ')
plt.ylim(0, 1)
plt.legend()
plt.show()

# Task 4: Grid Search
'''
Using the raw data and the same set of features as used above, use GridSearchCV to identify the value of alpha for which the model performs best. 
Assume the set of alpha values to be used as {0.0001, 0.001, 0.01, 0.1, 1, 10}
'''
