# Britney Clark
# CSC525: Principles of Machine Learning
# Critical Thinking 3
# Dr. Joseph Issa
# April 10, 2022

import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load and view data
boston_data = load_boston()
print(boston_data.keys())
boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
print(boston.head())

# Add Target column
boston['MEDV'] = boston_data.target
# Verify no missing data
print(boston.isnull().sum())

# Determine Correlation
correlation_matrix = boston.corr().round(2)
ax = sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

# Based on the correlation chart, RM and LSTAT will be used as the main features since they have the
# strongest correlation with MEDV at 0.7 and -0.74 respectively

features = ['RM', 'LSTAT']
target = boston['MEDV']

# Iterate through values in features
for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
plt.show()

# Prepare data for the training model
X = pd.DataFrame(numpy.c_[boston['RM'], boston['LSTAT']], columns=['RM', 'LSTAT'])
Y = boston['MEDV']

# Split Data in Train and Test
# Test size = 20% for 80% accuracy
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# Training and testing the model using Linear regression and Mean Squared Error.
# Model will be evaluated using Root Mean Square Error and R2-Score
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# Evaluation for Training Set
Y_train_predict = lin_model.predict(X_train)
rmse_train = (numpy.sqrt(mean_squared_error(Y_train, Y_train_predict)))
r2_train = r2_score(Y_train, Y_train_predict)

print('The model performance training set')
print('----------------------------------')
print('RMSE_Train is {}'.format(rmse_train))
print('R2 score Train is {}'.format(r2_train))
print('\n')

# Evaluation for Testing Set
Y_test_predict = lin_model.predict(X_test)
rmse_test = (numpy.sqrt(mean_squared_error(Y_test, Y_test_predict)))
r2_test = r2_score(Y_test, Y_test_predict)

print('The model performance testing set')
print('----------------------------------')
print('RMSE_Test is {}'.format(rmse_test))
print('R2 score Test is {}'.format(r2_test))
print('\n')

# Predicted Values
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_test_predict})
print(df)

