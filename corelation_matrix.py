import plotly.express as px
from numpy import *
import pandas as pd

#pulling and reading the file
logfile= pd.read_csv('logfiles_averaged_scenarios.csv')
df = pd.DataFrame(data=logfile)
print(df)

corr_matrix = df.corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()