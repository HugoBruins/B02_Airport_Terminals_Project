import numpy as np

import handle_data

if __name__ == '__main__':
    # handle_data.folders_to_csv("logfiles", "logfiles.csv")
    data = handle_data.csv_to_dataframe("logfiles.csv")

    data = handle_data.manipulate_data(data)

    handle_data.manual_check_data(data)

    pca_reduced_data, input_pca = handle_data.pca_on_input(data, 10)
    print(data["Input"].shape)
    print(pca_reduced_data["Input"].shape)
    print(np.cumsum(input_pca.explained_variance_ratio_))
