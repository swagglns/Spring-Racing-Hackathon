import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

data = pd.read_csv('all_tracks_hackathon.csv')

#starts, wins, seconds, thirds
column = 'win_time'

finishes = []
values = []
for x in range(10000):
    finishes.append(data['finish'][x])
    values.append(data[column][x])

plt.scatter(finishes, values)

# Add labels and title
plt.xlabel('Finish')
plt.ylabel(column)
plt.title(str(column) + ' vs Placement')

# Show the plot
plt.show()