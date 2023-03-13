
import plotly.express as px
from numpy import *
import pandas as pd

#pulling and reading the file
logfile= pd.read_csv('logfiles_averaged_scenarios.csv')
df = pd.DataFrame(data=logfile)
print(df)

#plotting the graph in 5D, this can change, and the axis labels need to be changed accordingly
fig = px.scatter_3d(df, x='CallToGateStrategy', y='AvgTimeToGate', z='NumMissedFlights', color='TotalExpenditure',
                    labels={"CallToGateStrategy": "Call To Gate Strategy",
                            "AvgTimeToGate": "Average Time To Gate",
                            "NumMissedFlights": "Number of Missed Flights",
                            "TotalExpenditure": "Total Expenditure"
                            },
                    title="TITLE OF THE PLOT"
                    )
fig.show()
