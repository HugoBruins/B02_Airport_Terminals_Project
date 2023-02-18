import os
import time
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA


def folders_to_csv(folder_name: str, file_name: str) -> None:
    """
    Generates a single csv file from all the read input data

    :param folder_name: The name of the folder where the data should be read from
    :param file_name: The name of the output csv file
    :return: None
    """
    print(f"[DEBUG] Processing all folders in {folder_name}, if this is not desired take read_files() out of main")
    start_time = time.time()
    output_file = open(file_name, "w")

    # Go through all the folders in the logfiles folder
    for root, dirs, files in os.walk(f".\\{folder_name}"):
        if not files == []:
            file_name = root + '\\' + files[0]

            # Open the text file in this logfiles sub-folder, combine the 2 lines in this file, so they appear next
            # to each other in the output file.
            with open(file_name, "r") as current_file:
                for n, line in enumerate(current_file):
                    if n == 0:
                        output_file.write(line.strip() + ",")
                    if n == 1:
                        output_file.write(line.replace("_",  ","))
    print(f"[DEBUG] It took this long to read all the folders: {time.time()-start_time}")
    output_file.close()


def csv_to_dataframe(filename: str) -> dict:
    """
    Converts the csv to a dictionary with on "Input" a dataframe with all the input parameters and on "Output" a
    dataframe containing all the output parameters

    :param filename: The name of the (csv) file to read
    :return: A dictionary containing the input and output data with correctly named headers
    """
    output = dict()
    data = pd.read_csv(filename, header=None)
    data.columns = ["AvgQueueTime_Cl1",	"AvgQueueTime_Cl2",	"AvgQueueTime_Cl3",	"AvgQueueTime_Cl4", "AvgQueueTime_SC",
                    "AvgTimeToGate",	"PaxCompleted_SC",	"PaxCompleted_Cl",	"NumMissedFlights",	"TotalExpenditure",
                    "MaxPaxInQueue_Cl1",	"MaxPaxInQueue_Cl2",	"MaxPaxInQueue_Cl3",	"MaxPaxInQueue_Cl4",
                    "MaxPaxInQueue_SC",	"Seed1",	"Gui",	"Timestep",	"EndTimeSimulation",	"Seed2",
                    "CheckInStrategy", "SecurityCheckpointStrategy",	"CallToGateStrategy",	"ScaleCheckpointTime",
                    "Timeslot1", "Timeslot2",	"Timeslot3",	"Timeslot4",	"Timeslot5",	"Timeslot6", "Timeslot7",
                    "ActiveGate1",	"ActiveGate2",	"ActiveGate3",	"ActiveGate4",	"ActiveGate5",	"ActiveGate6",
                    "ActiveGate7", "NumberPaxFlight1",	"NumberPaxFlight2",	"NumberPaxFlight3",	"NumberPaxFlight4",
                    "NumberPaxFlight5", "NumberPaxFlight6",	"NumberPaxFlight7",	"FlightPath1",	"FlightPath2",
                    "FlightPath3", "FlightPath4",	"FlightPath5",	"FlightPath6",	"FlightPath7",	"MinimumOccupancy1",
                    "MinimumOccupancy2",	"MinimumOccupancy3",	"MinimumOccupancy4",	"MinimumOccupancy5",
                    "MinimumOccupancy6",	"MinimumOccupancy7",	"Identifier"]

    output_data = data.iloc[:, :data.columns.get_loc("Gui")]
    input_data = data.iloc[:, data.columns.get_loc("Gui"):]

    output["Input"] = input_data
    output["Output"] = output_data

    return output


def manipulate_data(data: dict) -> dict:
    """
    :param data: A dict containing in and output dataframes generated from the read_data_to_dataframe function
    :return: A dict where the average queue-time and max passengers in queue of check in is averaged on the Output data
    in the same format as the input dict
    """
    output_data = data["Output"]

    avg_cl_time = (output_data["AvgQueueTime_Cl1"] + output_data["AvgQueueTime_Cl2"]
                   + output_data["AvgQueueTime_Cl3"] + output_data["AvgQueueTime_Cl4"]) / 4
    output_data = output_data.drop(["AvgQueueTime_Cl1", "AvgQueueTime_Cl2", "AvgQueueTime_Cl3", "AvgQueueTime_Cl4"],
                                   axis=1)
    output_data.insert(0, "AvgQueueTimeCl", avg_cl_time)

    # Averaging maximum passengers in check in queue, and replacing the old columns with the average
    avg_cl_pax = (output_data["MaxPaxInQueue_Cl1"] + output_data["MaxPaxInQueue_Cl2"]
                  + output_data["MaxPaxInQueue_Cl3"] + output_data["MaxPaxInQueue_Cl4"]) / 4
    output_data = output_data.drop(["MaxPaxInQueue_Cl1", "MaxPaxInQueue_Cl2", "MaxPaxInQueue_Cl3", "MaxPaxInQueue_Cl4"],
                                   axis=1)
    output_data.insert(output_data.columns.get_loc("TotalExpenditure") + 1, "AvgMaxPaxInQueueCl", avg_cl_pax)

    data["Output"] = output_data
    return data


def pca_on_input(data: dict, k: int) -> (dict, PCA):
    """
    Does a Principal Component Analysis transformation on the data that is input of the function to reduce the
    dimensions of the input data.

    :param data: A dictionary containing both the in and output data
    :param k: The total amount of principal components that are desired
    :return: A dict in the same format as the function input data ata but with the PCA transformed input data. And also
    returns the pca fit that was used to transform the data
    """
    # Removing everything with letters instead of numbers
    output = dict()
    pca_input = data["Input"].drop(
        ["Gui", "FlightPath1", "FlightPath2", "FlightPath3", "FlightPath4", "FlightPath5", "FlightPath6", "FlightPath7",
         "Identifier"], axis=1)
    scaler = preprocessing.RobustScaler()
    scaler.fit(pca_input)
    scaled_input = scaler.transform(pca_input)

    pca = PCA(n_components=k)
    pca.fit(scaled_input)
    output["Input"] = pd.DataFrame(pca.transform(scaled_input))
    output["Output"] = data["Output"]

    return output, pca


def manual_check_data(data: dict) -> None:
    """
    This function will print the column name and the respective value in the exact same row as the
    instructive PowerPoint listed, which can be used to cross validate data changes

    :param data: dataframe to compare
    :return: None
    """
    print("[DEBUG] Begin of manual data check, you can compare these values with what is in the powerpoint")
    input_row = data["Input"].loc[data["Input"]["Identifier"] == "ADA-117-158"]
    input_values = input_row.values.tolist()
    print("\nINPUT VALUES\n")
    for n, column_name in enumerate(list(input_row.columns)):
        print(f"{column_name}: {input_values[0][n]}")

    output_row = data["Output"].loc[data["Input"]["Identifier"] == "ADA-117-158"]
    output_values = output_row.values.tolist()
    print("\nOUTPUT VALUES\n")
    for n, column_name in enumerate(list(output_row.columns)):
        print(f"{column_name}: {output_values[0][n]}")
    print("[DEBUG] End of manual data check, you can compare these values with what is in the powerpoint")
