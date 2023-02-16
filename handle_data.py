import os
import time
import pandas as pd


def read_files(filename: str) -> None:
    """
    Function to read all the data from every separate file in logfiles folder and output the data
    in a csv file.

    Currently, it combines both the input and output lines into one single row in the csv file, to make it easier
    to manipulate later

    This function only needs to be run once, unless changes to the data formatting are desired.
    """
    print("[DEBUG] Processing all folders in logfiles, if this is not desired take read_files() out of main")
    start_time = time.time()
    output_file = open(filename, "w")

    # Go through all the folders in the logfiles folder
    for root, dirs, files in os.walk(".\\logfiles"):
        if not files == []:
            filename = root + '\\' + files[0]

            # Open the text file in this logfiles sub-folder, combine the 2 lines in this file, so they appear next
            # to each other in the output file.
            with open(filename, "r") as current_file:
                for n, line in enumerate(current_file):
                    if n == 0:
                        output_file.write(line.strip() + ",")
                    if n == 1:
                        output_file.write(line.replace("_",  ","))
    print(f"[DEBUG] It took this long to read all the folders: {time.time()-start_time}")
    output_file.close()


def process_data(filename: str) -> pd.DataFrame:
    """
    Function that reads the csv file generated from "read_files" as a pandas dataframe, and that adds correct headers
    to it. It also averages the average queue time and maximum passengers in queue of the check in.
    """
    data = pd.read_csv(filename, header=None)
    data.columns = ["AvgQueueTime_Cl1",	"AvgQueueTime_Cl2",	"AvgQueueTime_Cl3",	"AvgQueueTime_Cl4", "AvgQueueTime_SC",
                    "AvgTimeToGate",	"PaxCompleted_SC",	"PaxCompleted_Cl",	"NumMissedFlights",	"TotalExpenditure",
                    "MaxPaxInQueue_Cl1",	"MaxPaxInQueue_Cl2",	"MaxPaxInQueue_Cl3",	"MaxPaxInQueue_Cl4",
                    "MaxPaxInQueue_SC",	"Seed1",	"Gui",	"Timestep",	"EndTimeSimulation",	"Seed2",	"CheckInStrategy",
                    "SecurityCheckpointStrategy",	"CallToGateStrategy",	"ScaleCheckpointTime",	"Timeslot1",
                    "Timeslot2",	"Timeslot3",	"Timeslot4",	"Timeslot5",	"Timeslot6",	"Timeslot7",
                    "Gate1",	"Gate2",	"Gate3",	"Gate4",	"Gate5",	"Gate6",	"Gate7",
                    "NumberPaxFlight1",	"NumberPaxFlight2",	"NumberPaxFlight3",	"NumberPaxFlight4",	"NumberPaxFlight5",
                    "NumberPaxFlight6",	"NumberPaxFlight7",	"FlightPath1",	"FlightPath2",	"FlightPath3",
                    "FlightPath4",	"FlightPath5",	"FlightPath6",	"FlightPath7",	"MinimumOccupancy1",
                    "MinimumOccupancy2",	"MinimumOccupancy3",	"MinimumOccupancy4",	"MinimumOccupancy5",
                    "MinimumOccupancy6",	"MinimumOccupancy7",	"Identifier"]

    # validate_data(data)

    # Averaging check in queue time, and replacing the old columns with the average
    avg_cl_time = (data["AvgQueueTime_Cl1"] + data["AvgQueueTime_Cl2"] + data["AvgQueueTime_Cl3"] +
                   data["AvgQueueTime_Cl4"])/4
    data = data.drop(["AvgQueueTime_Cl1", "AvgQueueTime_Cl2", "AvgQueueTime_Cl3", "AvgQueueTime_Cl4"], axis=1)
    data.insert(0, "AvgQueueTimeCl", avg_cl_time)

    # Averaging maximum passengers in check in queue, and replacing the old columns with the average
    avg_cl_pax = (data["MaxPaxInQueue_Cl1"] + data["MaxPaxInQueue_Cl2"] + data["MaxPaxInQueue_Cl3"] +
                  data["MaxPaxInQueue_Cl4"])/4
    data = data.drop(["MaxPaxInQueue_Cl1", "MaxPaxInQueue_Cl2", "MaxPaxInQueue_Cl3", "MaxPaxInQueue_Cl4"], axis=1)
    data.insert(data.columns.get_loc("TotalExpenditure") + 1, "AvgMaxPaxInQueueCl", avg_cl_pax)

    # validate_data(data)
    return data


def manual_check_data(data: pd.DataFrame) -> None:
    """
    This function will print the column name and the respective value in the exact same row as the
    instructive PowerPoint listed, which can be used to cross validate data changes
    """
    print("[DEBUG] Begin of manual data check, you can compare these values with what is in the powerpoint")
    row_from_ppt = data.loc[data["Identifier"] == "ADA-117-158"]
    values = row_from_ppt.values.tolist()
    for n, column_name in enumerate(list(row_from_ppt.columns)):
        print(f"{column_name}: {values[0][n]}")
    print("[DEBUG] End of manual data check, you can compare these values with what is in the powerpoint")
