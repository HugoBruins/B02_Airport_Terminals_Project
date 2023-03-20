import pandas as pd
from sklearn.model_selection import train_test_split
#Reading in the dataset
df = pd.read_csv('fraud_prediction.csv')
#Dropping the index
df = df.drop(['Unnamed: 0'], axis = 1)
#Creating the features and target arrays
features = df.drop('amount', axis = 1).values
target = df['amount'].values
#Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target,
test_size = 0.3, random_state = 42)

from sklearn.ensemble import RandomForestRegressor

#Initiliazing an Random Forest Regressor with default parameters
rf_reg = RandomForestRegressor(max_depth = 10, min_samples_leaf = 0.2,
random_state = 50)

#Fitting the regressor on the training data
rf_reg.fit(X_train, y_train)
import RandomForestRegressor

