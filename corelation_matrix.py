import plotly.express as px
from numpy import *
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

#pulling and reading the file
logfile= pd.read_csv('logfiles_averaged_scenarios.csv')
df = pd.DataFrame(data=logfile)
#print(df)
#xaxis = ["Average Queue Time Cl", "Average Queue Time SC", "Average Time to Gate", "# Passengers Check-In Completed SC", "# Passengers Check-In Completed CL", "Number of Missed Flights", "Total Expenditure","Average Max Passenger in Queue CL", "Max Passenger in Queue SC", "Seed 1", "Chekc-In Strategy", "Security Check Point Stratega","Call to Gate Strategy", "Number of Passenger Missed Flights 1", "Number of Missed Flights 2", "Number of Missed Flights 3","Number of Missed Flights 4", "Number of Missed Flights 5", "Number of Missed Flights 6", "Number of Missed Flights 7","Identifier Scenario", "Identifie Run"]





corr_matrix = df.corr()
#plt.figure(figsize=(12,10))
g = sn.heatmap(corr_matrix, cmap="Reds", linewidth=.5)
#g.set_xticks(range(len(df)), labels=range("Average Queue Time Cl", "Average Queue Time SC", "Average Time to Gate", "Number Passengers Check-In Completed SC", "Number Passengers Check-In Completed CL", "Number of Missed Flights", "Total Expenditure", "Average Max Passenger in Queue CL", "Max Passenger in Queue SC", "Seed 1", "Chekc-In Strategy", "Security Check Point Stratega", "Call to Gate Strategy", "Number of Passenger Missed Flights 1", "Number of Missed Flights 2", "Number of Missed Flights 3", "Number of Missed Flights 4", "Number of Missed Flights 5", "Number of Missed Flights 6", "Number of Missed Flights 7", "Identifier Scenario", "Identifie Run"))

plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],["Unnamed", "Avg. Que TimeCl", "Avg. Que TimeSC", "Avg. Time to Gate", "Pax. Check-In Compl.SC", "Pax. Check-In Compl.CL", "No. Missed Flights", "Total Expenditure", "Avg. Max Pax in QueCL", "Max Pax. in QueSC", "Seed 1", "Check-In Strat.", "Security Checkpoint Strat.", "Call to Gate Strat.", "No. Pax Missed Flights1", "No. Pax Missed Flights2", "No. Pax Missed Flights3", "No. Pax Missed Flights4", "No. Pax Missed Flights5", "No. Pax Missed Flights6", "No. Pax Missed Flights7", "Identifier Scenario", "Identifier Run"], rotation=45, ha='right', minor=False)
plt.yticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],["Unnamed", "Avg. Que TimeCl", "Avg. Que TimeSC", "Avg. Time to Gate", "Pax. Check-In Compl.SC", "Pax. Check-In Compl.CL", "No. Missed Flights", "Total Expenditure", "Avg. Max Pax in QueCL", "Max Pax. in QueSC", "Seed 1", "Check-In Strat.", "Security Checkpoint Strat.", "Call to Gate Strat.", "No. Pax Missed Flights1", "No. Pax Missed Flights2", "No. Pax Missed Flights3", "No. Pax Missed Flights4", "No. Pax Missed Flights5", "No. Pax Missed Flights6", "No. Pax Missed Flights7", "Identifier Scenario", "Identifier Run"])


plt.title('Correlation Matrix', fontsize=16);
plt.show()