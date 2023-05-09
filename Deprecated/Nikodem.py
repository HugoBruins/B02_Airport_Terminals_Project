import handle_data

if __name__ == '__main__':
    # handle_data.folders_to_csv("logfiles", "logfiles.csv")
    data = handle_data.csv_to_dataframe("logfiles.csv")
    data = handle_data.manipulate_data(data)


from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import handle_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import copy
import matplotlib.pyplot as plt


import handle_data

if __name__ == '__main__':
    data = handle_data.csv_to_dataframe("logfiles.csv")
    data = handle_data.manipulate_data(data)


from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import handle_data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import copy
import matplotlib.pyplot as plt


def import_data():
    # Import the data
    data = handle_data.csv_to_dataframe("logfiles.csv")
    data = handle_data.strategies(data, "check_in_strategies.csv", "security_strategies.csv", True, True, True)
    training, val, test = handle_data.split_data(data)
    # training = handle_data.strategies(training, "check_in_strategies.csv", "security_strategies.csv", True, True, True)
    # val = handle_data.strategies(test, "check_in_strategies.csv", "security_strategies.csv", True, True, True)
    # test = handle_data.strategies(test, "check_in_strategies.csv", "security_strategies.csv", True, True, True)
    training = handle_data.manipulate_data(training)
    val = handle_data.manipulate_data(val)
    test = handle_data.manipulate_data(test)

    return training, val, test

training, val, test = import_data()

def train_model(training, hyperparameters):
    # Loop over all the output variables individually
    models = {}
    for key in list(training["Output"]):
        if key == "AvgTimeToGate":
            # Create a random forest classifier object
            rf_clf = RandomForestRegressor(**hyperparameters)

            # Train the random forest classifier on the training data
            rf_clf.fit(training['Input'], training["Output"][key])

            models[key] = rf_clf
    return models


def test_model(models, test):
    for key in models:
        if key == "AvgTimeToGate":
            # Use the trained model to predict on the test data
            y_pred = models[key].predict(test["Input"])

            mean_pred = y_pred.mean()
            mean_real = test["Output"][key].mean()
            # Compute the accuracy of the model and print it out
            mse = mean_squared_error(test["Output"][key], y_pred)
            print(f"Predicted variable:\t{key}\nRMSE:\t\t\t\t{np.sqrt(mse)}\nMean Predicted:\t\t{mean_pred}\nMean Real:"
            f"\t\t\t{mean_real}\n")


# Define the hyperparameter search space
param_space = {
    'n_estimators': Integer(50, 1000),
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 1.0),
    'bootstrap': [True, False]
}

# Define the objective function to optimize
def objective(params):
    rf = RandomForestRegressor(**params)
    return -np.mean(cross_val_score(rf, training['Input'], training["Output"]["AvgTimeToGate"], scoring='neg_mean_squared_error', cv=5))

# Define the BayesSearchCV optimizer

optimizer = BayesSearchCV(
    estimator=RandomForestRegressor(),
    search_spaces=param_space,
    n_iter=50,
    scoring='neg_mean_squared_error',
    random_state=42,
    verbose=0,
    cv=None,  # Disable k-fold cross-validation
)

# Run the optimizer to find the best hyperparameters
best_params = optimizer.fit(training['Input'], training["Output"]["AvgTimeToGate"], X_val=val['Input'], y_val=val['Output']["AvgTimeToGate"]).best_params_

# Train the model with the best hyperparameters
models = train_model(training, val, best_params)

# Print the best hyperparameters and objective value
print("Best hyperparameters: ", optimizer.best_params_)
print("Best objective value: ", -optimizer.best_score_)


