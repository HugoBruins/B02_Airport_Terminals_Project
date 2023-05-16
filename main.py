import handle_data
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # handle_data.folders_to_csv("logfiles", "logfiles.csv")
    data = handle_data.csv_to_dataframe("logfiles.csv")
    # handle_data.remove_faulty(data, "TotalExpenditure", 1.5, ["INI", "ADA", "VAL"])
    # data = handle_data.manipulate_data(data)
    # data = handle_data.remove_faulty(data, "Total Expenditure")
    # data = handle_data.manipulate_data(data)

    # data_avg = handle_data.average_data(data)
    # handle_data.manual_check_data(data)

    # print(data)
    # generate_plots(data)