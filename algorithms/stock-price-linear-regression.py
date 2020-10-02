from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import tensorflow
import keras
import quandl
import numpy as np

# Fetches the stock data
dataFrame = quandl.get("WIKI/TSLA")

# Formats the data to only look for the close price
dataFrame = dataFrame[["Adj. Close"]]

# Outputs first five rows of data to the console to make sure everything is working correctly
print(dataFrame.head())

# How many days in the future to predict
predictionDays = 10

# Creates a new column x amount of days ahead of the present
dataFrame["Prediction"] = dataFrame[["Adj. Close"]].shift(-predictionDays)

# Outputs the new data set
print(dataFrame.tail())

# Converts the dataframe into a numpy array which becomes the training set for the model
trainingSet = np.array(dataFrame.drop(["Prediction"], 1)) # axis = 1

# Removes the predicted rows
trainingSet = trainingSet[:-dataFrame]

# Prints out the new set
print(trainingSet)
