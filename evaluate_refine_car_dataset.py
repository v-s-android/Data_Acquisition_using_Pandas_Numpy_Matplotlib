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

'''
You can also use the function 'cross_val_predict' to predict the output. The function splits up the data into the specified number of folds,
with one fold for testing and the other folds are used for training. First, import the function:
'''
from sklearn.model_selection import cross_val_predict
#We input the object, the feature "horsepower", and the target data y_data. The parameter 'cv' determines the number of folds. In this case, it is 4. We can produce an output:
r_cross_predict = cross_val_predict( lre, x_data[['horsepower']], y_data, cv =4 )
print(r_cross_predict[0:5])
# array([14141.63807508, 14141.63807508, 20814.29423473, 12745.03562306, 14762.35027598])

# Part 2: Overfitting, Underfitting and Model Selection
'''
It turns out these differences are more apparent in Multiple Linear Regression and Polynomial Regression so we will explore overfitting in that context.

Let's create Multiple Linear Regression objects and train the model using 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg' as features.
'''
lr = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.10, random_state = 1)
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train) # we always use the training set for fir the lodel

#Prediction using training data:
yhat_train_predict = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(yhat_train_predict[0:5])  # array([ 7426.6731551 , 28323.75090803, 14213.38819709,  4052.34146983, 34500.19124244])
#Prediction using test data:
yhat_test_predict = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(yhat_test_predict[0:5]) #array([11349.35089149,  5884.11059106, 11208.6928275 ,  6641.07786278, 15565.79920282])

# Let's perform some model evaluation using our training and testing data separately. First, we import the seaborn and matplotlib library for plotting.

import seaborn as sns
import matplotlib.pyplot as plt

# Let's examine the distribution of the predicted values of the training data.
Title = "Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution"
DistributionPlot(y_train , yhat_train_predict, "Actual Values (Train)", "Predicted Values (Train)", Title)
# plots the graphs

# Let's examine the distribution of the predicted values of the test data.

Title = "Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data"
DistributionPlot(y_test, yhat_test_predict, "Actual Values (Test)","Predicted Values (Test)", Title)
# plots the graphs
'''
Overfitting
Overfitting occurs when the model fits the noise, but not the underlying process. Therefore, when testing your model using the test set, your model does not perform as well since it is modelling noise, not the underlying process that generated the relationship.
Let's create a degree 5 polynomial model.
'''
from sklearn.preprocessing import PolynomialFeatures

#Let's use 55 percent of the data for training and the rest for testing:
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.45, random_state = 0)

# We will perform a degree 5 polynomial transformation on the feature 'horsepower'.
pr = PolynomialFeatures(degree = 5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])

# now lets create a linearregression model and train it
lr = LinearRegression()
lr.fit(x_train_pr, y_train)
# We can see the output of our model using the method "predict." We assign the values to "yhat".

yhat = lr.predict(x_test_pr)
yhat[0:5] # array([ 6728.58641321,  7307.91998787, 12213.73753589, 18893.37919224, 19996.10612156])

# Let's take the first five predicted values and compare it to the actual targets.

print("Predicted values",yhat[0:5] )
print("Actual values ",y_test[0:5].values)
'''
Predicted values: [ 6728.58641321  7307.91998787 12213.73753589 18893.37919224 19996.10612156]
Actual values: [ 6295.     10698.     13860.     13499. 1    5750.]
'''
# We will use the function "PollyPlot" that we defined at the beginning of the lab to display the training data, testing data, and the predicted function.
# PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform)
PollyPlot( x_train['horsepower'], x_test['horsepower'], y_train, y_test, lr, pr) # plots the graph

# R^2 of the training data:
print(lr.score(x_train_pr , y_train)) # 0.5567716897754004

# R^2 of the test data:
print(lr.score(x_test_pr, y_test)) # -29.87099623387278
'''
We see the R^2 for the training data is 0.5567 while the R^2 on the test data was -29.87. The lower the R^2, the worse the model. A negative R^2 is a sign of overfitting.
'''

# Let's see how the R^2 changes on the test data for different order polynomials and then plot the results:
Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    
    
    lr.fit(x_train_pr, y_train)
    
    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')  # plots the graph

# The following function will be used in the next section. Please run the cell below.
def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly,pr)

# The following interface allows you to experiment with different polynomial orders and different amounts of data.
interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05)) # plots teh graph

#Question #4a): We can perform polynomial transformations with more than one feature. Create a "PolynomialFeatures" object "pr1" of degree two.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.45, random_state=0)
pr1 = PolynomialFeatures(degree=2)

# Question #4b) Transform the training and testing samples for the features 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg'. Hint: use the method "fit_transform".
x_train_pr1 = pr1.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
x_test_pr1 = pr1.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

# Question #4c) How many dimensions does the new feature have? Hint: use the attribute "shape".
print(x_train_pr1.shape) # (110, 15) #there are now 15 features
print(x_test_pr1.shape) # (91, 15) #there are now 15 features

# Question #4d): Create a linear regression model "poly1". Train the object using the method "fit" using the polynomial features.
lr = LinearRegression()
lr.fit(x_train_pr1 , y_train)

# Question #4e): Use the method "predict" to predict an output on the polynomial features, then use the function "DistributionPlot"
#to display the distribution of the predicted test output vs. the actual test data.
y_hat = lr.predict(x_test_pr1)
title = 'Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test, y_hat, "Actual values(Test)", "Predicted Values (Test)" ,title) # plots the graph

# Question #4f):  Using the distribution plot above, describe (in words) the two regions where the predicted prices are less accurate than the actual prices.
'''
The predicted value is higher than actual value for cars where the price $10,000 range, conversely the predicted price is lower than the price cost in the $30,000
to $40,000 range. As such the model is not as accurate in these ranges.
'''

#Part 3: Ridge Regression
'''
we will review Ridge Regression and see how the parameter alpha changes the model. Just a note, here our test data will be used as validation data.

Let's perform a degree two polynomial transformation on our data.
'''

pr = PolynomialFeatures(degree = 2)
x_train_pr = pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr = pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
# import Ridge
from sklearn.linear_model import Ridge

# Let's create a Ridge regression object, setting the regularization parameter (alpha) to 0.1
RidgeModel = Ridge(alpha = 1)
# Like regular regression, you can fit the model using the method fit.
RidgeModel.fit(x_train_pr, y_train)
# Similarly, you can obtain a prediction:
y_hat = RidgeModel.predict(x_test_pr)

print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)
'''
predicted: [ 6570.82441941  9636.24891471 20949.92322738 19403.60313255]
test set : [ 6295. 10698. 13860. 13499.]
'''

# We select the value of alpha that minimizes the test error. To do so, we can use a for loop. We have also created a progress bar to see how many iterations we have completed so far.
from tqdm import tqdm

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)
'''
100%|██████████| 1000/1000 [00:01<00:00, 810.37it/s, Test Score=0.564, Train Score=0.859]
'''
# We can plot out the value of R^2 for different alphas:

width = 8
height = 6
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()

# Question #5):  Perform Ridge regression. Calculate the R^2 using the polynomial features, use the training data to train the model and use the test data
#to test the model. The parameter alpha should be set to 10.
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

ridge_model = Ridge(alpha = 10) # setting alpha to 10
ridge_model.fit(x_train_pr , y_train)
y_hat = ridge_model.predict(x_test_pr)

print('predicted:', y_hat[0:5])
print('test set :', y_test[0:5].values)
print("R^2 ", RigeModel.score(x_test_pr, y_test))
'''
predicted: [ 6472.05406775  9537.15585237 21078.08955884 19750.77444841
 21339.40311655]
test set : [ 6295. 10698. 13860. 13499. 15750.]
R^2  0.5637701868993872
'''

# Part 4: Grid Search
'''
The term alpha is a hyperparameter. Sklearn has the class GridSearchCV to make the process of finding the best hyperparameter simpler.
'''
from sklearn.model_selection import GridSearchCV

# We create a dictionary of parameter values:
parameters1= [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000]}]
parameters1
# Create a Ridge regression object:
RR = Ridge()
# Create a ridge grid search object:
grid_serach_cv_model = GridSearchCV( RR , parameters1 , cv = 4)
# Fit the model y_data = df['price']:
grid_serach_cv_model.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)

# The object finds the best parameter values on the validation data. We can obtain the estimator with the best parameters and assign it to the variable BestRR as follows:
BestRR = grid_serach_cv_model.best_estimator_
print(BestRR)
# We now test our model on the test data:
BestRR.score(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data) 
'''
0.8411649831036152
'''
# Question #6):  Perform a grid search to find the best alpha value and check if using feature scaling improves the model.

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

# Define parameter grid without 'normalize'
params = [{'alpha' : [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000] }]

# Perform Grid Search
ridge = Ridge()
grid_s_model = GridSearchCV(ridge , params, cv = 4)
grid_s_model.fit(x_scaled, y_data)

# Best model
print("Best estimator: ",grid_s_model.best_estimator_)
print("Best Params: ",grid_s_model.best_params_['alpha'])
'''
Best estimator:  Ridge(alpha=100)
Best Params:  100
'''
