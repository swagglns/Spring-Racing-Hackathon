import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

# load data
df = pd.read_csv(r"races\race9.csv", low_memory=False)
print("Data read")

# collums to drop final version
drop_cols = ['horse_name','purse','post_time','win_time','horse_name','program_num','post_position','comment','jockey','trainer',
            'owner','last_race_track','last_race_date','last_race_number','track_code','track_name','race_date','race_number']

df['race_uid'] = df['race_number'].astype(str) + "_" + df['race_date'].astype(str) + "_" + df['track_code'].astype(str)
df = df.drop(columns=drop_cols)

# converting 

df.to_csv('converted_test_one_race.csv', index=False) 