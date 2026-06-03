import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

data = pd.read_csv('all_tracks_hackathon.csv')

categories = []
values = []
for x in range(10000):
    if data['dollar_odds'][x] == 0:
        continue
    if data['finish'][x] == 1:
        found = False
        for y in range(len(values)):
            if categories[y] == float(data['dollar_odds'][x]):
                values[y] += 1
                found = True
                break
        if not found:
            categories.append(float(data['dollar_odds'][x]))
            values.append(1)

print(np.average(categories,weights=values))
#highestWinner = max(values)
#print(categories[values.index(highestWinner)])
print(categories)
print(values)
plt.bar(categories, values)

# Add labels and title
plt.xlabel('Odds')
plt.ylabel('Commonality')
plt.title('Commonality of odds winning')

# Show the plot
plt.show()