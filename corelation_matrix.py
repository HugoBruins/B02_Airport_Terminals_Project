import plotly.express as px
from numpy import *
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

#pulling and reading the file
logfile= pd.read_csv('logfiles_averaged_scenarios.csv')
df = pd.DataFrame(data=logfile)

#creating the heat map
corr_matrix = df.corr()
sn.heatmap(corr_matrix, cmap="Reds", linewidth=.5)

#naming the ticks appropriately 
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[" ", "Avg. Que TimeCl", "Avg. Que TimeSC", "Avg. Time to Gate", "Pax. Check-In Compl.SC", "Pax. Check-In Compl.CL", "No. Missed Flights", "Total Expenditure", "Avg. Max Pax in QueCL", "Max Pax. in QueSC", "Seed 1", "Check-In Strat.", "Security Checkpoint Strat.", "Call to Gate Strat.", "No. Pax Missed Flights1", "No. Pax Missed Flights2", "No. Pax Missed Flights3", "No. Pax Missed Flights4", "No. Pax Missed Flights5", "No. Pax Missed Flights6", "No. Pax Missed Flights7", "Identifier Scenario", "Identifier Run"], rotation=45, ha='right', minor=False)
plt.yticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[" ", "Avg. Que TimeCl", "Avg. Que TimeSC", "Avg. Time to Gate", "Pax. Check-In Compl.SC", "Pax. Check-In Compl.CL", "No. Missed Flights", "Total Expenditure", "Avg. Max Pax in QueCL", "Max Pax. in QueSC", "Seed 1", "Check-In Strat.", "Security Checkpoint Strat.", "Call to Gate Strat.", "No. Pax Missed Flights1", "No. Pax Missed Flights2", "No. Pax Missed Flights3", "No. Pax Missed Flights4", "No. Pax Missed Flights5", "No. Pax Missed Flights6", "No. Pax Missed Flights7", "Identifier Scenario", "Identifier Run"])

#plotting the graph
plt.title('Correlation Matrix for Averaged Scenarios', fontsize=16)
plt.show()