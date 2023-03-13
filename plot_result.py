import plotly.express as px
from numpy import *
import pandas as pd

#pulling and reading the file
logfile= pd.read_csv('logfile.csv')
df = pd.DataFrame(data=logfile)
print(df)

#plotting the graph in 5D, this can change, and the axis labels need to be changed accordingly
fig = px.scatter_3d(df, x='0.0', y='0.0.1', z='0.0.2', color='0.0.3', size='0.0.9',
        labels={"0.0": "x-axis [-]",
                    "0.0.1": "y-axis [-]",
                    "0.0.2": "z-axis [-]",
                    "0.0.3": "Colors",
                    "0.0.9": "Size"
                 },
                title="TITLE OF THE PLOT"
                    )
fig.show()