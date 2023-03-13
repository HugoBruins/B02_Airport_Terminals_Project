import handle_data

if __name__ == '__main__':
    # handle_data.folders_to_csv("logfiles", "logfiles.csv")
    data = handle_data.csv_to_dataframe("logfiles.csv")

    data = handle_data.manipulate_data(data)
    data_avg = handle_data.average_data(data)
    handle_data.manual_check_data(data)