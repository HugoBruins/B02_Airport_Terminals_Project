
import plotly.express as px
from numpy import *
import pandas as pd

logfile= pd.read_csv('logfile.csv')
df = pd.DataFrame(data=logfile)
print(df)

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