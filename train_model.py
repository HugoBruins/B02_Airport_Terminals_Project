import time

import handle_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np
import copy
import matplotlib.pyplot as plt
import time

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


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


def hyperparamaters_main(training, val, patience, increment_percentage, shrink_percentage, max_iterations, max_time):  #(https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
    # Number of trees in random forest (initial range from: https://mljar.com/blog/how-many-trees-in-random-forest/)
    n_estimators = list(np.arange(51, 221, 4))  # start = 200, stop = 2000, num = 10
    # Number of features to consider at every split
    max_features = list(np.arange(1, 27, 2, dtype=int))
    # Maximum number of levels in tree
    max_depth = list(np.arange(251, 351, 5, dtype=int))
    # Minimum number of samples required to split a node
    min_samples_split = list(np.arange(101, 501, 10))
    # Minimum number of samples required at each leaf node (https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/)
    min_samples_leaf = list(np.arange(51, 251, 1))
    # Method of selecting samples for training each tree
    min_weight_fraction_leaf = list(np.arange(0.00, 0.06, 0.01, dtype=float))
    # max_leaf_nodes = list(np.arange(150, 500, 10, dtype=int))
    min_impurity_decrease = list(np.arange(0.0000, 0.0011, 0.0001, dtype=float))
    ccp_alpha = list(np.arange(0.0000, 0.01, 0.0005, dtype=float))
    bootstrap = [True, False]
    # Create the random grid
    test_ranges = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'min_weight_fraction_leaf': min_weight_fraction_leaf,
                #    'max_leaf_nodes': max_leaf_nodes,
                   'min_impurity_decrease': min_impurity_decrease,
                   'ccp_alpha': ccp_alpha, 
                   'bootstrap': bootstrap}

    initial_parameters = {'n_estimators': 101,
                          'max_features': 13,
                          'max_depth': 301,
                          'min_samples_split': 7,
                          'min_samples_leaf': 5,
                            'min_weight_fraction_leaf':0.00,
                            # 'max_leaf_nodes':350,
                            'min_impurity_decrease':0.0000,
                            'ccp_alpha': 0.000, 
                          'bootstrap': True}
    
    old_min = {}
    old_max = {}

    previous_parameters = copy.deepcopy(initial_parameters)
    number_of_iterations = max_iterations + 1
    keys = list(initial_parameters.keys())

    # For each iteration as specified above
    time_start = time.process_time()
    # Stop if max_time is exceeded
    for i in range(number_of_iterations):
        print(f'current test_range: {test_ranges}')
        if (time.process_time() - time_start) > max_time:
            break
        print(f'starting iteration {i+1}...')
        current_parameters = previous_parameters
        # Remove a parameter, find the optimal one using the function, put it back
        for key in keys:
            print(f'optimizing {key} in iteration {i+1}...')
            print(f'current parameters: {current_parameters}')
            current_parameters.pop(key)
            new_value, slope_is_positive = hyperparameter_optimize_parameter(key, test_ranges[key], current_parameters, training, val, patience, (i+1))
            current_parameters[key] = new_value
            # if error slope of previous iteration was postive shift to right, else to left
            try:
                if key not in ["bootstrap"]:
                    if i == 0:
                        if key in ['min_weight_fraction_leaf', 'min_impurity_decrease', 'ccp_alpha']:
                            increment = (test_ranges[key][-1] - test_ranges[key][0]) * (increment_percentage / 100)
                            #step_size = (test_ranges[key][-1] - test_ranges[key][0])/60
                        else:
                            increment = round((test_ranges[key][-1] - test_ranges[key][0]) * (increment_percentage / 100))
                            #step_size = round((test_ranges[key][-1] - test_ranges[key][0])/60)
                        if step_size < 1:
                            step_size = 1
                        if slope_is_positive:
                            new_max = new_value + increment
                            new_min = new_max - (test_ranges[key][-1] - test_ranges[key][0])* (shrink_percentage/ 100)
                        else:
                            new_min = new_value - increment
                            new_max = new_min + (test_ranges[key][-1] - test_ranges[key][0])* (shrink_percentage/ 100)
                        
                        old_min[key] = new_min
                        old_max[key] = new_max
                        if key not in ['min_weight_fraction_leaf', 'min_impurity_decrease', 'ccp_alpha']:
                            new_min = round(new_min)
                            new_max = round(new_max)

                        if new_min < 0:
                            new_min = 1
                        test_ranges[key] = list(np.arange(new_min, new_max, step_size))
                    else:
                        if key in ['min_weight_fraction_leaf', 'min_impurity_decrease', 'ccp_alpha']:
                            increment = (old_max[key] - old_min[key]) * (increment_percentage / 100)
                        else:
                            increment = round((old_max[key] - old_min[key]) * (increment_percentage / 100))
                            #step_size = round((old_max[key] - old_min[key])/60)
                        if step_size < 1:
                            step_size = 1
                        if slope_is_positive:
                            new_max = new_value + increment
                            new_min = new_max - (old_max[key] - old_min[key])* (shrink_percentage/ 100)
                        else:
                            new_min = new_value - increment
                            new_max = new_min + (old_max[key] - old_min[key])* (shrink_percentage/ 100)
                        
                        old_min[key] = new_min
                        old_max[key] = new_max
                        if key not in ['min_weight_fraction_leaf', 'min_impurity_decrease', 'ccp_alpha']:
                            new_min = round(new_min)
                            new_max = round(new_max)

                        if new_min < 0:
                            new_min = 1
                        test_ranges[key] = list(np.arange(new_min, new_max, step_size))
            except Exception as e:
                pass

        previous_parameters = current_parameters
    # The best parameters ar the most recent ones
    best_parameters = previous_parameters
    return best_parameters


def hyperparameter_optimize_parameter(test_parameter_key, test_parameter_range, current_parameters, training, val, patience, current_iteration):
    patience_counter = 0  # Number of times the error curve shoots up
    history = []
    for index, test_parameter_value in enumerate(test_parameter_range):
        if patience_counter > patience:
            break
        print(f'trying {test_parameter_key}: {test_parameter_value}')
        # put the test parameter back from where it was removed in the other function
        current_parameters[test_parameter_key] = test_parameter_value
        # train a model on training data with the current fixed parameters and the variable parameter
        rf_clf = RandomForestRegressor(**current_parameters)
        rf_clf.fit(training['Input'], training["Output"]["AvgTimeToGate"])
        # compute the error for these parameters using validation data
        mse = hyperparameters_evaluate_accuracy(rf_clf, val)
        print(f'{current_parameters}\nmse:{mse}')
        # append the current variable parameter and error [parameter, error] to history
        if index != 0:
            if mse > history[-1][1]:  # If error > previous_error and not on first iteration, count up patience counter
                patience_counter += 1
        history.append([test_parameter_value, mse])
    # find the value of the minimum error
    print(history)
    min_error = min([x[1] for x in history])
    # find the parameter that this minimum error corresponds to
    for sublist in history:
        if sublist[1] == min_error:
            optimal_parameter = sublist[0]
            break
    print(f"Found optimal parameter {test_parameter_key} value of {optimal_parameter} with error {min_error}")
    # hyperparameters_plot_history(history, test_parameter_key, current_iteration)

    # compute if the error function is decreasing or increasing in this iteration
    linear_regression_model = LinearRegression()
    x = []
    y = []
    for item in history:
        x.append([item[0]])
        y.append(item[1])
    try:
        linear_regression_model.fit(x, y)
        slope = linear_regression_model.coef_
        slope_is_positive = False
        if slope[0] > 0:
            slope_is_positive = True
        print('marker')
        print(slope)
    except Exception as e:
        slope_is_positive = True
        pass
    return optimal_parameter, slope_is_positive


def hyperparameters_evaluate_accuracy(rf_clf, val):
    y_pred = rf_clf.predict(val["Input"])
    # mean_pred = y_pred.mean()
    # mean_real = val["Output"].mean()
    mse = np.sqrt(mean_squared_error(val["Output"]["AvgTimeToGate"], y_pred))
    return mse


def hyperparameters_plot_history(history, name, current_iteration):

    x_values = [x[0] for x in history]
    y_values = [x[1] for x in history]

    plt.plot(x_values, y_values)
    plt.xlabel(name)
    plt.ylabel('error')
    plt.title(f'{name} vs error ({current_iteration})')
    plt.show()


def main():
    print('newrun')
    training, val, test = import_data_main()
    hyperparameters = hyperparamaters_main(training, val, 20, 10, 99, 500000, 1800)  #input: training data, val data, patience, increment_percentage, shrink_percentage, max_iterations, max_time
#     print(f'final hyperparemeters are: {hyperparameters}')
    # hyperparameters = {'ccp_alpha': 0.001, 'bootstrap': False, 'n_estimators': 63, 'max_features': 11, 'max_depth': 261, 'min_samples_split': 221, 'min_samples_leaf': 85, 'min_weight_fraction_leaf': 0.0, 'min_impurity_decrease': 0.0008}
    # models = train_model_main(training, hyperparameters)
    # test_model(models, test)
    """""
    1.
    Order
    importance
    2.
    Implement
    max.up
    tolerance

    3.
    Implement
    move
    range
    to
    left / right & remove
    outliers

    4.
    Implement
    max.iterations
    5.
    Implement
    max.time

    6.
    Plot
    min
    error
    between
    iterations"""""
main()