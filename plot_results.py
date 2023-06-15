import matplotlib.pyplot as plt
import statsmodels.api as sm

def qq_plot(x_train, x_test, y_train, y_test, xlabel, ylabel, c_train, c_test):
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
    fig.set_figwidth(15)

    # Put all the brothers together
    plt.subplots_adjust(wspace=0.0)

    # Plot training plots
    axs[0].scatter(x_train, y_train, alpha=0.5, c=c_train, label="Training Data")
    axs[1].hist(y_train, bins=50, orientation='horizontal', color=c_train, ec=c_train)

    # Plot test plots
    axs[0].scatter(x_test, y_test, alpha=0.5, c=c_test, label="Test data")
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[0].grid(True)
    axs[0].legend(loc='upper right')

    axs[1].hist(y_test, bins=50, orientation='horizontal', color=c_test, ec=c_test)
    axs[1].set_xlabel("Frequency")
    axs[1].grid(True)
    axs[1].set_yticklabels([])

# #pulling and reading the file
# logfile= pd.read_csv('logfiles_averaged_scenarios.csv')
# df = pd.DataFrame(data=logfile)
# print(df)
#
# #plotting the graph in 5D, this can change, and the axis labels need to be changed accordingly
# fig = px.scatter_3d(df, x='CallToGateStrategy', y='AvgTimeToGate', z='NumMissedFlights', color='TotalExpenditure',
#                     labels={"CallToGateStrategy": "Call To Gate Strategy",
#                             "AvgTimeToGate": "Average Time To Gate",
#                             "NumMissedFlights": "Number of Missed Flights",
#                             "TotalExpenditure": "Total Expenditure"
#                             },
#                     title="TITLE OF THE PLOT"
#                     )
# fig.show()
#
# # for a 2D plot
# fig = px.scatter(df, x='CallToGateStrategy', y='TotalExpenditure',
#                     labels={"CallToGateStrategy": "Call to Gate Strategy",
#                             "TotalExpenditure": "Total Expenditure"
#                             },
#                     title="TITLE"
#                     )
# fig.show()