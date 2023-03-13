import handle_data
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def separate_data(data):
    # Create a dictionary to store the separate DataFrames
    separated_data = {}

    # Get the unique values of the "identifier" column
    identifiers = data["Input"]["IdentifierType"].unique()

    # Iterate over the identifiers and create a separate DataFrame for each one
    for identifier in identifiers:
        separated_data[identifier] = data[data["Input"]["IdentifierType"] == identifier].reset_index(drop=True)

    return separated_data


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.3, random_state=42)

# Create a random forest classifier object
rf_clf = RandomForestRegressor(n_estimators=190, random_state=23)

# Train the random forest classifier on the training data
rf_clf.fit(X_train, y_train)

# Use the trained model to predict on the test data
y_pred = rf_clf.predict(X_test)

# Compute the accuracy of the model
mse = mean_squared_error(y_test, y_pred)

print(f"MSE : {mse}")


# handle_data.folders_to_csv("logfiles", "logfiles.csv")
data = handle_data.csv_to_dataframe("logfiles.csv")

data = handle_data.manipulate_data(data)
# data_avg = handle_data.average_data(data)
handle_data.manual_check_data(data)

# generate_plots(data_avg)

separated_data = separate_data(data)

INI_input = separated_data["INI"]
ADA_input = separated_data["ADA"]
VAL_input = separated_data["VAL"]
TEST_input = separated_data["TEST"]

print(INI_input)
print('marker')
print(ADA_input)
