import handle_data
import matplotlib.pyplot as plt


def generate_plots(average_data):
    total_passengers = average_data["Input"]["NumberPaxFlight1"] + average_data["Input"]["NumberPaxFlight2"] +\
                       average_data["Input"]["NumberPaxFlight3"] + average_data["Input"]["NumberPaxFlight4"] +\
                       average_data["Input"]["NumberPaxFlight5"] + average_data["Input"]["NumberPaxFlight6"] +\
                       average_data["Input"]["NumberPaxFlight7"]

    plt.scatter(total_passengers, average_data["Output"]["AvgQueueTimeCl"])
    plt.xlabel("Total passengers [-]")
    plt.ylabel("Average Check-In Queue time [s]")
    plt.savefig("plots\\Total passengers vs Average Check in Queue time.png")
    plt.close()

    plt.scatter(total_passengers, average_data["Output"]["AvgQueueTime_SC"])
    plt.xlabel("Total passengers [-]")
    plt.ylabel("Average Security Queue time [s]")
    plt.savefig("plots\\Total passengers vs Average Security Queue Time.png")
    plt.close()

    plt.scatter(total_passengers, average_data["Output"]["AvgTimeToGate"])
    plt.xlabel("Total passengers [-]")
    plt.ylabel("Average Time to Gate [s]")
    plt.savefig("plots\\Total passengers vs Average Time To Gate.png")
    plt.close()

    plt.scatter(total_passengers, average_data["Output"]["TotalExpenditure"])
    plt.xlabel("Total passengers [-]")
    plt.ylabel("Total Expenditure [€]")
    plt.savefig("plots\\Total passengers vs Total Expenditure.png")
    plt.close()

    plt.scatter(total_passengers, average_data["Output"]["NumMissedFlights"])
    plt.xlabel("Total passengers [-]")
    plt.ylabel("Total number of missed flights [-]")
    plt.savefig("plots\\Total passengers vs Number of Missed Flights.png")
    plt.close()


if __name__ == '__main__':
    # handle_data.folders_to_csv("logfiles", "logfiles.csv")
    data = handle_data.csv_to_dataframe("logfiles.csv")

    data = handle_data.manipulate_data(data)
    # data_avg = handle_data.average_data(data)
    handle_data.manual_check_data(data)

    print(data.keys())
    generate_plots(data_avg)
