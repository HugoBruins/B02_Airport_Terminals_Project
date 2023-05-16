import plotly.express as px

from numpy import *
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import handle_data

#pulling and reading the file
# logfile= pd.read_csv('logfiles_averaged_scenarios.csv')
data = handle_data.csv_to_dataframe("logfiles.csv")
data = handle_data.manipulate_data(data)
data = handle_data.strategies(data, "check_in_strategies.csv", "security_strategies.csv", True, True, True)

df = pd.concat([data["Input"], data["Output"]], axis=1)




# Totaling the no. pax. missed flights, since there were 7
df.insert(0, "Total passengers", df[['NumberPaxFlight1', 'NumberPaxFlight2', "NumberPaxFlight3", "NumberPaxFlight4", "NumberPaxFlight5", "NumberPaxFlight6", "NumberPaxFlight7"]].sum(axis=1))
# df.loc[-1] = (df[['NumberPaxFlight1', 'NumberPaxFlight2', "NumberPaxFlight3", "NumberPaxFlight4", "NumberPaxFlight5", "NumberPaxFlight6", "NumberPaxFlight7"]])
# df = df.drop(['NumberPaxFlight1', 'NumberPaxFlight2', "NumberPaxFlight3", "NumberPaxFlight4", "NumberPaxFlight5", "NumberPaxFlight6", "NumberPaxFlight7"], axis=1)

# creating the heat map
corr_matrix = df.corr()
corr_matrix.dropna(axis=0, how='all', inplace=True)
corr_matrix.dropna(axis=1, how='all', inplace=True)

column_names = corr_matrix.columns.tolist()
print(column_names)

corr_matrix = corr_matrix.drop(column_names[1: 10], axis = 1)
corr_matrix = corr_matrix.drop(column_names[1: 10], axis = 0)

print(corr_matrix)
sn.heatmap(corr_matrix, annot=True, cmap="Reds", linewidth=.5, vmin=-1, vmax=1)

#naming the ticks appropriately
# plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],[" ", "Avg. Que TimeCl", "Avg. Que TimeSC", "Avg. Time to Gate", "Pax. Check-In Compl.SC", "Pax. Check-In Compl.CL", "No. Missed Flights", "Total Expenditure", "Avg. Max Pax in QueCL", "Max Pax. in QueSC", "Check-In Strat.", "Security Checkpoint Strat.", "Call to Gate Strat.", "Identifier Scenario", "Identifier Run", "Tot. No. Pax Missed Flights"], rotation=45, ha='right', minor=False)
#plt.yticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],[" ", "Avg. Que TimeCl", "Avg. Que TimeSC", "Avg. Time to Gate", "Pax. Check-In Compl.SC", "Pax. Check-In Compl.CL", "No. Missed Flights", "Total Expenditure", "Avg. Max Pax in QueCL", "Max Pax. in QueSC", "Check-In Strat.", "Security Checkpoint Strat.", "Call to Gate Strat.", "Identifier Scenario", "Identifier Run", "Tot. No. Pax Missed Flights"])
#plt.xticks([1,2,3,4,5,6,7,8,9], ["Avg. Que TimeCl", "Avg. Que TimeSC", "Avg. Time to Gate", "Pax. Check-In Compl.SC", "Pax. Check-In Compl.CL", "No. Missed Flights", "Total Expenditure", "Avg. Max Pax in QueCL", "Max Pax. in QueSC"], rotation=45, ha='right', minor=False)

abbrev_inputs = ["TotPax", "CallToGateStrat", "NPax1", "NPax2", "NPax3", "NPax4", "NPax5", "NPax6", "NPax7", "C7200-6300", "C6300-5400", "C5400-4500", "C4500-3600", "C3600-2700", "S<1800", "S1800-3600", "S7200-9000", "S9000-10800", "S>10800"]
abbrev_outputs = ["TotPax", "CallToGateStrat", "NPax1", "NPax2", "NPax3", "NPax4", "NPax5", "NPax6", "NPax7", "C7200-6300", "C6300-5400", "C5400-4500", "C4500-3600", "C3600-2700", "S<1800", "S1800-3600", "S7200-9000", "S9000-10800", "S>10800"]
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], abbrev_outputs)
plt.yticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], abbrev_inputs)


#plotting the graph
# plt.title('Correlation Matrix', fontsize=16)
plt.show()