import plotly.express as px
from numpy import *
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

#pulling and reading the file
logfile= pd.read_csv('logfiles_averaged_scenarios.csv')
df = pd.DataFrame(data=logfile)

# Totaling the no. pax. missed flights, since there were 7
df["total_pax_missed_flights"] = df[['NumberPaxFlight1', 'NumberPaxFlight2', "NumberPaxFlight3", "NumberPaxFlight4", "NumberPaxFlight5", "NumberPaxFlight6", "NumberPaxFlight7"]].sum(axis=1)
df = df.drop(['NumberPaxFlight1', 'NumberPaxFlight2', "NumberPaxFlight3", "NumberPaxFlight4", "NumberPaxFlight5", "NumberPaxFlight6", "NumberPaxFlight7"],  axis=1)
df = df.drop(["Seed1"], axis=1)
print(list(df))

#creating the heat map
corr_matrix = df.corr()
sn.heatmap(corr_matrix, cmap="Reds", linewidth=.5)

#naming the ticks appropriately 
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],[" ", "Avg. Que TimeCl", "Avg. Que TimeSC", "Avg. Time to Gate", "Pax. Check-In Compl.SC", "Pax. Check-In Compl.CL", "No. Missed Flights", "Total Expenditure", "Avg. Max Pax in QueCL", "Max Pax. in QueSC", "Check-In Strat.", "Security Checkpoint Strat.", "Call to Gate Strat.", "Identifier Scenario", "Identifier Run", "Tot. No. Pax Missed Flights"], rotation=45, ha='right', minor=False)
plt.yticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],[" ", "Avg. Que TimeCl", "Avg. Que TimeSC", "Avg. Time to Gate", "Pax. Check-In Compl.SC", "Pax. Check-In Compl.CL", "No. Missed Flights", "Total Expenditure", "Avg. Max Pax in QueCL", "Max Pax. in QueSC", "Check-In Strat.", "Security Checkpoint Strat.", "Call to Gate Strat.", "Identifier Scenario", "Identifier Run", "Tot. No. Pax Missed Flights"])


#plotting the graph
plt.title('Correlation Matrix for Averaged Scenarios', fontsize=16)
plt.show()