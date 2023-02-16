import handle_data


if __name__ == '__main__':
    # handle_data.read_files("logfiles.csv")
    data = handle_data.process_data("logfiles.csv")
    handle_data.manual_check_data(data)
