import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import random

model = joblib.load('top3classifier.joblib')

with open("delete_unwanted_columns.py") as f:
    code = f.read()
    exec(code)

with open("Data Conversion.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb)

# Define features and labels

df = pd.read_csv(r"converted_test_one_race.csv",low_memory=False)

df['win_rate'] = df['num_past_wins'] / df['num_past_starts']
# Avoid division by zero
df['win_rate'] = df['win_rate'].fillna(0)

# Success rate: Number of past wins divided by (number of past wins + number of past losses or non-wins)
df['success_rate'] = df['num_past_wins'] / (df['num_past_wins'] + df['num_past_thirds'] + df['num_past_seconds'])
df['success_rate'] = df['success_rate'].fillna(0)

# Ratio of third-place finishes to total past starts
df['thirds_ratio'] = df['num_past_thirds'] / df['num_past_starts']
df['thirds_ratio'] = df['thirds_ratio'].fillna(0)

df.drop(columns=['num_past_seconds','num_past_starts','num_past_thirds','num_past_wins','finish','distance_unit','race_uid'], inplace=True)

model = joblib.load("top3classifier.joblib")

result = model.predict(df)

count = 0
for r in result:
    print(r)

# Find the indices of all items predicted as True
true_indices = [i for i, val in enumerate(result) if val == True]

# Generate rankings from 1 to count and shuffle them
rankings = list(range(1, len(true_indices) + 1))
random.shuffle(rankings)

# Ensure exactly 3 top picks
if len(true_indices) < 3:
    # Get all indices sorted by lowest dollar_odds
    sorted_by_odds = df['dollar_odds'].argsort().tolist()
    for idx in sorted_by_odds:
        if idx not in true_indices:
            true_indices.append(idx)
            rankings.append(len(true_indices))
            if len(true_indices) == 3:
                break

# Trim if more than 3
while len(true_indices) > 3:
    max_rank_idx = rankings.index(max(rankings))
    del true_indices[max_rank_idx]
    del rankings[max_rank_idx]


for x in range(len(true_indices)):
    print(rankings[x], '- Post Position:', true_indices[x]+1)