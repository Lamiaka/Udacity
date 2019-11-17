# TODO: Add import statements
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data.csv',delimiter = ',',header = None)
X = train_data.iloc[:,:-1]
Y = train_data.iloc[:,-1]

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()
#line_reg = LinearRegression()
# TODO: Fit the model.
model_fit = lasso_reg.fit(X,Y)
#model_fit_linear = line_reg.fit(X,Y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
#line_reg_coef = line_reg.coef_
print(reg_coef)
#print(line_reg_coef)