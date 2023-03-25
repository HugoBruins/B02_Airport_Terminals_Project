import handle_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Import the data
data = handle_data.csv_to_dataframe("logfiles.csv")
training, val, test = handle_data.split_data(data)

training = handle_data.manipulate_data(training)
val = handle_data.manipulate_data(val)
test = handle_data.manipulate_data(test)

# Loop over all the output variables individually
for key in list(training["Output"]):
    # Create a random forest classifier object
    rf_clf = RandomForestRegressor(n_estimators=50,  random_state=23)

    # Train the random forest classifier on the training data
    rf_clf.fit(training['Input'], training["Output"][key])

    # Use the trained model to predict on the test data
    y_pred = rf_clf.predict(val["Input"])

    mean_pred = y_pred.mean()
    mean_real = val["Output"][key].mean()
    # Compute the accuracy of the model and print it out
    mse = mean_squared_error(val["Output"][key], y_pred)
    print(f"Predicted variable:\t{key}\nRMSE:\t\t\t\t{np.sqrt(mse)}\nMean Predicted:\t\t{mean_pred}\nMean Real:"
          f"\t\t\t{mean_real}\n")
