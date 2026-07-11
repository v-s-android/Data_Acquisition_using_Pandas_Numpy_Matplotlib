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





