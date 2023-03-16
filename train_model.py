import handle_data
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


data = handle_data.csv_to_dataframe("logfiles.csv")
training, val, test = handle_data.split_data(data)
training = handle_data.manipulate_data(training)
val = handle_data.manipulate_data(val)
test = handle_data.manipulate_data(test)

print(training["Input"])

# Create a random forest classifier object
rf_clf = RandomForestRegressor(n_estimators=50, random_state=23)

# Train the random forest classifier on the training data
rf_clf.fit(training['Input'], training["Output"])

# Use the trained model to predict on the test data
y_pred = rf_clf.predict(test["Input"])

# Compute the accuracy of the model
mse = mean_squared_error(test["Output"], y_pred)

print(f"MSE : {mse}")

print("Hugo")
print("Now you can see me")

