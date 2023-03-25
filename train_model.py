import handle_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import RandomizedSearchCV


def import_data():
    # Import the data
    data = handle_data.csv_to_dataframe("logfiles.csv")
    training, val, test = handle_data.split_data(data)

    training = handle_data.manipulate_data(training)
    val = handle_data.manipulate_data(val)
    test = handle_data.manipulate_data(test)

    return training, val, test


def train_model(training, hyperparameters):
    # Loop over all the output variables individually
    models = {}
    for key in list(training["Output"]):
        if key == "TotalExpenditure":
            # Create a random forest classifier object
            rf_clf = RandomForestRegressor(**hyperparameters)

            # Train the random forest classifier on the training data
            rf_clf.fit(training['Input'], training["Output"][key])

            models[key] = rf_clf
    return models


def test_model(models, test):
    for key in models:
        if key == "TotalExpenditure":
            # Use the trained model to predict on the test data
            y_pred = models[key].predict(test["Input"])

            mean_pred = y_pred.mean()
            mean_real = test["Output"][key].mean()
            # Compute the accuracy of the model and print it out
            mse = mean_squared_error(test["Output"][key], y_pred)
            print(f"Predicted variable:\t{key}\nRMSE:\t\t\t\t{np.sqrt(mse)}\nMean Predicted:\t\t{mean_pred}\nMean Real:"
            f"\t\t\t{mean_real}\n")


def tune_hyperparamaters(val):  #(https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
    # Number of trees in random forest (initial range from: https://mljar.com/blog/how-many-trees-in-random-forest/)
    n_estimators = list(np.arange(950, 1150, 1))  # start = 200, stop = 2000, num = 10
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = list(np.arange(1, 150, 2, dtype=int))
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node (https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/)
    min_samples_leaf = range(1, 5)
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    print(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, verbose=3, n_jobs=2)  # Next session, manually write this function!!! and put val data into test_model, leave test data out
    # Fit the random search model
    rf_random.fit(val["Input"], val["Output"]["TotalExpenditure"])

    print(rf_random.best_params_)
    return rf_random.best_params_




training, val, test = import_data()
hyperparameters = tune_hyperparamaters(val)

# hyperparameters = {'n_estimators': 355, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 200, 'bootstrap': False}
models = train_model(training, hyperparameters)
test_model(models, test)
