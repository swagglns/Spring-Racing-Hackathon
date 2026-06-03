import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_csv(r"C:\Users\jtvin\OneDrive\Documents\GitHub\hackathon_data\all_tracks_hackathon.csv", low_memory=False)
print("Data read")

# Drop columns with very high cardinality or not useful
drop_cols = [
    'horse_name', 'track_code', 'track_name', 'race_date', 'post_time', 'weather',
    'last_race_track', 'last_race_date', 'last_race_number', 'last_race_finish',
    'breed', 'comment'
]

# Filter to just races we care about (top 3 finishers)
df = df[df['finish'].isin([1, 2, 3])]

# Add race_uid for race-based split
df['race_uid'] = df['race_number'].astype(str) + "_" + df['race_date'].astype(str)
df = df.drop(columns=drop_cols)

# Split by race
unique_races = df['race_uid'].unique()
train_races = unique_races[:int(0.8 * len(unique_races))]
test_races = unique_races[int(0.8 * len(unique_races)):]

train_df = df[df['race_uid'].isin(train_races)]
test_df = df[df['race_uid'].isin(test_races)]

# Define target
y_train = train_df['finish']
y_test = test_df['finish']

# Drop target + race ID
X_train = train_df.drop(columns=['finish', 'race_uid'])
X_test = test_df.drop(columns=['finish', 'race_uid'])

# One-hot encode only relevant low-cardinality categorical variables
cat_cols = ['race_type', 'distance_unit', 'course', 'surface', 'track_condition', 'sex', 'medication']
X_train = pd.get_dummies(X_train, columns=cat_cols)
X_test = pd.get_dummies(X_test, columns=cat_cols)

# Align features
X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)
#only inlcuding things that are all numbers
X_train = X_train.select_dtypes(include=['number'])
X_test = X_test.select_dtypes(include=['number'])

#New code
X_combined = pd.get_dummies(pd.concat([X_train, X_test]))

# Now split it back into training and test sets
X_train = X_combined.iloc[:len(X_train), :]
X_test = X_combined.iloc[len(X_train):, :]

# Downcast numerics to save memory
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print("Training has begun...")

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict & report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
joblib.dump(model, 'initial_forest_model.joblib')

loaded_model = joblib.load('initial_forest_model.joblib')

# Now you can use the loaded model to make predictions

#testing accuracy on the trifecta




# Create a function to check if the trifecta is correctly predicted
def check_trifecta(actual, predicted):
    # Extract the top 3 finishers
    top_3_actual = actual.sort_values().head(3).index.tolist()  # actual top 3
    top_3_predicted = predicted.sort_values().head(3).index.tolist()  # predicted top 3

    # Check if the predicted top 3 match the actual top 3 in exact order
    return top_3_actual == top_3_predicted

# Apply the check_trifecta function to each race
correct_trifecta_count = 0
total_races = len(y_test)

for race in test_df['race_uid'].unique():
    race_data = test_df[test_df['race_uid'] == race]
    actual_finish = race_data['finish']
    predicted_finish = model.predict(race_data.drop(columns=['finish', 'race_uid']))

    if check_trifecta(actual_finish, predicted_finish):
        correct_trifecta_count += 1

# Calculate trifecta prediction accuracy
trifecta_accuracy = correct_trifecta_count / total_races
print(f"Trifecta Prediction Accuracy: {trifecta_accuracy * 100:.2f}%")