import os
import time
import logging
import pandas as pd

"""
Function to read all the data from every seperate file in .\logfiles and output the data
in a comma seperated CSV file
"""
#
def read_files(filename: str) -> None:
    print("[DEBUG] Processing all folders in logfiles, if this is not desired take read_files() out of main")
    start_time = time.time()
    output_file = open(filename, "w");
    for root, dirs, files in os.walk(".\logfiles"):
        if not files == []:
            filename = root + '\\' + files[0];
            with open(filename, "r") as current_file:
                for n, line in enumerate(current_file):
                    if n == 0:
                        output_file.write(line.strip() + ",");
                    if n == 1:
                        output_file.write(line.replace("_",  ","))
    logging.debug(f"[DEBUG] It took this long to read all the folders: {time.time()-start_time}")
    output_file.close()

def process_data(filename: str) -> None:
    data = pd.read_csv(filename, header=None)
    data.columns = ["AvgQueueTime_Cl1",	"AvgQueueTime_Cl2",	"AvgQueueTime_Cl3",	"AvgQueueTime_Cl4"	,"AvgQueueTime_SC",	"AvgTimeToGate",	"PaxCompleted_SC",	"PaxCompleted_Cl",	"NumMissedFlights",	"TotalExpenditure",	"MaxPaxInQueue_Cl1",	"MaxPaxInQueue_Cl2",	"MaxPaxInQueue_Cl3",	"MaxPaxInQueue_Cl4",	"MaxPaxInQueue_SC",	"Seed",	"Gui",	"Timestep",	"EndTimeSimulation",	"Seed",	"CheckInStrategy",	"SecurityCheckpointStrategy",	"CallToGateStrategy",	"ScaleCheckpointTime",	"Timeslot1",	"Timeslot2",	"Timeslot3",	"Timeslot4",	"Timeslot5",	"Timeslot6",	"Timeslot7",	"Gate1",	"Gate2",	"Gate3",	"Gate4",	"Gate5",	"Gate6",	"Gate7",	"NumberPaxFlight1",	"NumberPaxFlight2",	"NumberPaxFlight3",	"NumberPaxFlight4",	"NumberPaxFlight5",	"NumberPaxFlight6",	"NumberPaxFlight7",	"FlightPath1",	"FlightPath2",	"FlightPath3",	"FlightPath4",	"FlightPath5",	"FlightPath6",	"FlightPath7",	"MinimumOccupancy1",	"MinimumOccupancy2",	"MinimumOccupancy3",	"MinimumOccupancy4",	"MinimumOccupancy5",	"MinimumOccupancy6",	"MinimumOccupancy7",	"Identifier"]

    #avg_cl_time = (data[0] + data[1] + data[2] + data[3])/4

    print(data)

    #data.replace([0,1,2,3], avg_cl_time.transpose)


