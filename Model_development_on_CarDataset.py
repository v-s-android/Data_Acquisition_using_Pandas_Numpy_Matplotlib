'''
Develop prediction models : we will develop several models that will predict the price of the car using the variables or features
'''
#install specific version of libraries
#! mamba install pandas==1.3.3-y
#! mamba install numpy=1.21.2-y
#! mamba install sklearn=0.20.1-y
import piplite
await piplite.install('seaborn')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyodide.http import pyfetch
from sklearn.linear_model import LinearRegression

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

file_path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"
file_name="usedcars.csv"
await download(file_path, file_name)

df = pd.read_csv(file_name)
df.head()

# Create the linear regression object:
lm = LinearRegression()
print(lm)

# Using simple linear regression, we will create a linear function with "highway-mpg" as the predictor variable and the "price" as the response variable.
X = df[['highway-mpg']]
Y = df['price']

# Fit the linear model using highway-mpg:
lm.fit(X,Y)

# We can output a prediction:
Yhat = lm.predict(X)
print(Yhat[0:5]) # array([16236.50464347, 16236.50464347, 17058.23802179, 13771.3045085 , 20345.17153508])

# the value of the intercept (a)
print(lm.intercept_) # 38423.30585815743

# the value of the slope (b)
print(lm.coef_) # array([-821.73337832])

#Create a linear regression object called "lm1".
lm1 = LinearRegression()
print(lm1)
#Train the model using "engine-size" as the independent variable and "price" as the dependent variable
Xeng = df[['engine-size']]
Yprice = df['price']

lm1.fit(X,Y) # or alternativley lm1.fit(df[['engine-size']], df[['price']])
Yhat = lm1.predict(X)
print(Yhat[0:5]) # array([16236.50464347, 16236.50464347, 17058.23802179, 13771.3045085 , 20345.17153508])

# Intercept and slope
print(lm1.intercept_)
print(lm1.coef_)

'''
Multiple Linear Regression : 

Other good predictors of price could be: Horsepower, Curb-weight, Engine-size, Highway-mpg
Developing a model using these variables as the predictor variables.
'''
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
#Fit the linear model using the four above-mentioned variables.
lm.fit(Z , df['price'])

Yhat = lm.predict(Z)
Yhat[0:5]

lm.intercept_ # -15806.62462632922
lm.coef_ # array([53.49574423,  4.70770099, 81.53026382, 36.05748882])
# Price = -15678.742628061467 + 52.65851272 x horsepower + 4.69878948 x curb-weight + 81.95906216 x engine-size + 33.58258185 x highway-mpg

# Similarly : Create and train a Multiple Linear Regression model "lm2" where the response variable is "price", and the predictor variable is "normalized-losses" and "highway-mpg".
lm2 = LinearRegression()
lm2.fit(df[['normalized-losses' , 'highway-mpg']],df['price'])

print(lm2.coef_) # [   1.49789586 -820.45434016]
print(lm2.intercept_) # 38201.313272457344

# 2. Model Evaluation Using Visualization
import seaborn as sns

'''
This plot will show a combination of a scattered data points (a scatterplot), as well as the fitted linear regression line going through the data.
This will give us a reasonable estimate of the relationship between the two variables, the strength of the correlation,
as well as the direction (positive or negative correlation).
'''
# Let's visualize highway-mpg as potential predictor variable of price:

width = 8
height = 6
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

# similarly for peak-rpm

plt.figure(figsize = (width, height))
sns.regplot(x = 'peak-rpm', y = 'price', data = df)
plt.ylim(0,)

# Given the regression plots above, is "peak-rpm" or "highway-mpg" more strongly correlated with "price"? Use the method ".corr()"
df[['peak-rpm','ighway-mpg','price']].corr()
'''
	            peak-rpm	highway-mpg	       price
peak-rpm	    1.000000	    -0.058598	  -0.101616
highway-mpg	    -0.058598	    1.000000	   -0.704692
price	        -0.101616	    -0.704692	    1.000000

# The variable "highway-mpg" has a stronger correlation with "price", it is approximate -0.704692  compared to "peak-rpm" which is approximate -0.101616.
'''
# Residual Plot
'''
What is a residual?

The difference between the observed value (y) and the predicted value (Yhat) is called the residual (e).
When we look at a regression plot, the residual is the distance from the data point to the fitted regression line.

So what is a residual plot?

A residual plot is a graph that shows the residuals on the vertical y-axis and the independent variable on the horizontal x-axis.
'''
plt.figure(figsize =(width, height))
sns.residplot( x = df['highway-mpg'], y = df['price'] ) # notice we are using the directly the dataframe column
plt.show()

# Multiple Linear Regression: Distribution plot
'''
How do we visualize a model for Multiple Linear Regression? This gets a bit more complicated because you can't visualize it with regression or residual plot.

One way to look at the fit of the model is by looking at the distribution plot. We can look at the distribution of the fitted values that result from the model and
compare it to the distribution of the actual values.
'''
Y_hat = lm.predict(Z) # first make a pridiction
plt.figure(figsize=(width,height))

ax1 = sns.distplot(df['price'], hist = False, color = 'r' , label = 'Actual Value')
sns.distplot(Y_hat, hist = False, color = 'r' , label = 'Fitted Value')

plt.title("Actual vs Fitted Values for Price")
plt.xlabel("Price (in dollars)")
plt.ylabel("Proportion of Cars")
plt.show()
plt.close()

# Polynomial regression
'''
Polynomial regression is a particular case of the general linear regression model or multiple linear regression models.
We get non-linear relationships by squaring or setting higher-order terms of the predictor variables.
'''
def PlotPolly(model, independent_variable, dependent_variabble, x_label):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(x_label)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()
#The variables:
x = df['highway-mpg']
y = df['price']

# fit the polynomial using the function "polyfit", then use the function "poly1d" to display the polynomial function.
# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit( x, y, 3)
p = np.poly1d(f)
print(p)
'''
        3         2
-1.557 x + 204.8 x - 8965 x + 1.379e+05
'''
# plot the function:
PlotPolly( p , x , y , "highway-mpg") # shows the plot

# Similarly Create 11 order polynomial model with the variables x and y from above.

f1 = np.polyfit(x, y , 11)
p1 = np.poly1d(f1)
print(p1)
'''
11             10             9           8         7
-1.243e-08 x  + 4.722e-06 x  - 0.0008028 x + 0.08056 x - 5.297 x
          6        5             4             3             2
 + 239.5 x - 7588 x + 1.684e+05 x - 2.565e+06 x + 2.551e+07 x - 1.491e+08 x + 3.879e+08
'''
PlotPolly( p1, x , y, 'highway-mpg')

# We can perform a polynomial transform on multiple features
from sklearn.preprocessing import PolynomialFeatures

# create a PolynomialFeatures object of degree 2
pf = PolynomialFeatures(degree = 2 )
Z_pf = pf.fit_transform(Z)
Z.shape # (201, 2) In the original data, there are 201 samples and 2 features.
Z_pf.shape # (201 , 6) After the transformation, there are 201 samples and 6 features.

# Piplines
'''
Data Pipelines simplify the steps of processing the data. We use the module Pipeline to create a pipeline. We also use StandardScaler as a step in our pipeline.
'''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# We create the pipeline by creating a list of tuples including the name of the model or estimator and its corresponding constructor.
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

# We input the list as an argument to the pipeline constructor:
pipe = Pipeline(Input)
print(pipe)

# First, we convert the data type Z to type float to avoid conversion warnings that may appear as a result of StandardScaler taking float inputs.
#Then, we can normalize the data, perform a transform and fit the model simultaneously.
# Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Z = Z.astype(float)
pipe.fit(Z, y)

# Similarly, we can normalize the data, perform a transform and produce a prediction simultaneously.
ypipe=pipe.predict(Z)
ypipe[0:4] # array([15388.77780567, 15388.77780567, 16771.84474515, 11641.85647791])

# Create a pipeline that standardizes the data, then produce a prediction using a linear regression model using the features Z and target y.

Input = [('scale', StandardScaler()),('model' , LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(Z, y) # Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

ypipe = pipe.predict(Z)
ypipe[0:10]
'''
array([16231.78938339, 16231.78938339, 17052.24372355, 13833.33798916,
       20396.97271047, 17872.69806371, 17926.6223148 , 17872.69806371,
       22028.89401561, 14695.7334135 ])
'''

# Two very important measures that are often used in Statistics to determine the accuracy of a model are:

# R^2 / R-squared
# Mean Squared Error (MSE)

# Model 1: Simple Linear Regression
#highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y)) # The R-square is:  0.4965911884339176

'''
We can say that ~49.659% of the variation of the price is explained by this simple linear model "horsepower_fit".

Let's calculate the MSE:

We can predict the output i.e., "yhat" using the predict method, where X is the input variable:
'''
Yhat = lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4]) 
# The output of the first four predicted value is:  [16236.50464347 16236.50464347 17058.23802179 13771.3045085 ]

# import the function mean_squared_error from the module metrics:
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)
# The mean square error of price and predicted value is:  31635042.944639888

# Model 2: Multiple Linear Regression
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Y = df['price']
lm.fit( Z , Y )
print("R-squared ", lm.score(Z , Y))
'''
The R-square is:  0.8093562806577457
We can say that ~80.896 % of the variation of price is explained by this multiple linear regression "multi_fit".
'''

Y_predict_multifit = lm.predict(Z)

mse1 = mean_squared_error(Z, Y_predict_multifit)
print("mean squared error" , mse1)
'''
The mean square error of price and predicted value using multifit is:  11980366.87072649
'''

# Model 3: Polynomial Fit
from sklearn.metrics import r2_score
x = df['highway-mpg']
y = df['price']

# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)

r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)
'''
The R-square value is:  0.674194666390652
We can say that ~67.419 % of the variation of price is explained by this polynomial fit.
'''
# We can also calculate the MSE:
mean_squared_error(y , p(x))
'''
20474146.426361218
'''

# 5. Prediction and Decision Making
'''
Prediction¶
In the previous section, we trained the model using the method fit. Now we will use the method predict to produce a prediction.
Lets import pyplot for plotting; we will also be using some functions from numpy.
'''
# Create a new input:
new_input = np.arange(1, 100, 1).reshape(-1, 1)

X = df[['highway-mpg']]
Y = df['price']

lm.fit(X, Y)

Y_hat = lm.predict(new_input)
print(Y_hat[0:5]) # array([37601.57247984, 36779.83910151, 35958.10572319, 35136.37234487, 34314.63896655])

plt.plot(new_input, Y_hat)
plt.show()

'''
Decision Making: Determining a Good Model Fit

Now that we have visualized the different models, and generated the R-squared and MSE values for the fits, how do we determine a good model fit?

What is a good R-squared value?
When comparing models, the model with the higher R-squared value is a better fit for the data.

What is a good MSE?
When comparing models, the model with the smallest MSE value is a better fit for the data.

Let's take a look at the values for the different models.
Simple Linear Regression: Using Highway-mpg as a Predictor Variable of Price.

R-squared: 0.49659118843391759
MSE: 3.16 x10^7
Multiple Linear Regression: Using Horsepower, Curb-weight, Engine-size, and Highway-mpg as Predictor Variables of Price.

R-squared: 0.80896354913783497
MSE: 1.2 x10^7
Polynomial Fit: Using Highway-mpg as a Predictor Variable of Price.

R-squared: 0.6741946663906514
MSE: 2.05 x 10^7

Simple Linear Regression Model (SLR) vs Multiple Linear Regression Model (MLR)
Usually, the more variables you have, the better your model is at predicting, but this is not always true. Sometimes you may not have enough data,
you may run into numerical problems, or many of the variables may not be useful and even act as noise. As a result, you should always check the MSE and R^2.

In order to compare the results of the MLR vs SLR models, we look at a combination of both the R-squared and MSE to make the best conclusion about the fit of the model.

MSE: The MSE of SLR is 3.16x10^7 while MLR has an MSE of 1.2 x10^7. The MSE of MLR is much smaller.
R-squared: In this case, we can also see that there is a big difference between the R-squared of the SLR and the R-squared of the MLR. The R-squared for the SLR (~0.497)
is very small compared to the R-squared for the MLR (~0.809).
This R-squared in combination with the MSE show that MLR seems like the better model fit in this case compared to SLR.

Simple Linear Model (SLR) vs. Polynomial Fit
MSE: We can see that Polynomial Fit brought down the MSE, since this MSE is smaller than the one from the SLR.
R-squared: The R-squared for the Polynomial Fit is larger than the R-squared for the SLR, so the Polynomial Fit also brought up the R-squared quite a bit.
Since the Polynomial Fit resulted in a lower MSE and a higher R-squared, we can conclude that this was a better fit model than the simple linear regression for
predicting "price" with "highway-mpg" as a predictor variable.

Multiple Linear Regression (MLR) vs. Polynomial Fit
MSE: The MSE for the MLR is smaller than the MSE for the Polynomial Fit.
R-squared: The R-squared for the MLR is also much larger than for the Polynomial Fit.

Conclusion
Comparing these three models, we conclude that the MLR model is the best model to be able to predict price from our dataset. This result makes sense since we have 27 variables in total and we know that more than one of those variables are potential predictors of the final car price.
'''
