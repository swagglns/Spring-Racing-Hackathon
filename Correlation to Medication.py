import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

data = pd.read_csv('all_tracks_hackathon.csv')

categories = ['B','L','C','M','A','']
values = [0,0,0,0,0,0]
Lcount = 0
for x in range(10000):
    if data['finish'][x] == 1:
        if str(data['medication'][x]) == 'nan':
            values[5] += 1
            continue
        values[categories.index(data['medication'][x])] += 1
    if data['medication'][x] == 'L':
        Lcount += 1

print(Lcount / 10000)
print(values[1]/sum(values))
print(categories)
print(values)
plt.bar(categories, values)

# Add labels and title
plt.xlabel('Odds')
plt.ylabel('Commonality')
plt.title('Commonality of odds winning')

# Show the plot
plt.show()