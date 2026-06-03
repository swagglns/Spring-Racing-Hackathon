import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

data = pd.read_csv('all_tracks_hackathon.csv')

medications = ['B','L','C','M','A','None']
counts = [0,0,0,0,0,0]
for x in data['medication']:
    match x:
        case 'B':
            counts[0] += 1
        case 'L':
            counts[1] += 1
        case 'C':
            counts[2] += 1
        case 'M':
            counts[3] += 1
        case 'A':
            counts[4] += 1
        case _:
            counts[5] += 1

for x in range(len(counts)):
    print(medications[x] + ":",counts[x] / len(data))