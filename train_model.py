import handle_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import copy
import matplotlib.pyplot as plt


def import_data_main():
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


def train_model_main(training, hyperparameters):
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


def hyperparamaters_main(training, val):  #(https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
    # Number of trees in random forest (initial range from: https://mljar.com/blog/how-many-trees-in-random-forest/)
    n_estimators = list(np.arange(1, 50, 1))  # start = 200, stop = 2000, num = 10
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = list(np.arange(1, 151, 25, dtype=int))
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node (https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/)
    min_samples_leaf = range(1, 5)
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    test_ranges = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    initial_parameters = {'n_estimators': 100,
                          'max_features': 1.0,
                          'max_depth': None,
                          'min_samples_split': 2,
                          'min_samples_leaf': 1,
                          'bootstrap': True}

    previous_parameters = copy.deepcopy(initial_parameters)
    number_of_iterations = 2
    keys = list(initial_parameters.keys())

    # For each iteration as specified above
    for i in range(number_of_iterations):
        print(f'starting iteration {i+1}...')
        current_parameters = previous_parameters
        # Remove a parameter, find the optimal one using the function, put it back
        for key in keys:
            print(f'optimizing {key} in iteration {i+1}...')
            current_parameters.pop(key)
            new_value = hyperparameter_optimize_parameter(key, test_ranges[key], current_parameters, training, val)
            current_parameters[key] = new_value
        previous_parameters = current_parameters
    # The best parameters ar the most recent ones
    best_parameters = previous_parameters
    return best_parameters


def hyperparameter_optimize_parameter(test_parameter_key, test_parameter_range, current_parameters, training, val):
    history = []
    for test_parameter_value in test_parameter_range:
        # put the test parameter back from where it was removed in the other function
        current_parameters[test_parameter_key] = test_parameter_value
        # Create a random forest classifier object
        rf_clf = RandomForestRegressor(**current_parameters)

        # Train the random forest classifier on the training data
        rf_clf.fit(training['Input'], training["Output"]["AvgTimeToGate"])
        mse = hyperparameters_evaluate_accuracy(rf_clf, val)
        print(f'{current_parameters}\nmse:{mse}')
        history.append([test_parameter_value, mse])
    min_error = min([x[1] for x in history])
    for sublist in history:
        if sublist[1] == min_error:
            optimal_parameter = sublist[0]
            break
    print(f"Found optimal parameter {test_parameter_key} value of {optimal_parameter} with error {min_error}")
    hyperparameters_plot_history(history, test_parameter_key)
    return optimal_parameter


def hyperparameters_evaluate_accuracy(rf_clf, val):
    y_pred = rf_clf.predict(val["Input"])
    # mean_pred = y_pred.mean()
    # mean_real = val["Output"].mean()
    mse = mean_absolute_error(val["Output"]["AvgTimeToGate"], y_pred)
    return mse


def hyperparameters_plot_history(history, name):

    x_values = [x[0] for x in history]
    y_values = [x[1] for x in history]

    plt.plot(x_values, y_values)
    plt.xlabel(name)
    plt.ylabel('error')
    plt.title(f'{name} vs error')
    plt.show()


def main():
    training, val, test = import_data_main()
    hyperparameters = hyperparamaters_main(training, val)
    print(hyperparameters)
    # hyperparameters = {'n_estimators': 355, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 200, 'bootstrap': False}
    #models = train_model(training, hyperparameters)
    #test_model(models, test)

main()