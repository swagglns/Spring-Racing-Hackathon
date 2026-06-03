import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib



df = pd.read_csv(r"C:\Users\jtvin\OneDrive\Documents\GitHub\Spring-Racing-Hackathon\parsed_data.csv",low_memory=False)


#this lists all of the unique race ids that then are sorted into train and test data sets


#data interactions
# Win rate: Number of past wins divided by number of past starts
df['win_rate'] = df['num_past_wins'] / df['num_past_starts']
# Avoid division by zero
df['win_rate'] = df['win_rate'].fillna(0)

# Success rate: Number of past wins divided by (number of past wins + number of past losses or non-wins)
df['success_rate'] = df['num_past_wins'] / (df['num_past_wins'] + df['num_past_thirds'] + df['num_past_seconds'])
df['success_rate'] = df['success_rate'].fillna(0)

# Ratio of third-place finishes to total past starts
df['thirds_ratio'] = df['num_past_thirds'] / df['num_past_starts']
df['thirds_ratio'] = df['thirds_ratio'].fillna(0)

#only care about hte top three
df['top3'] = df['finish'].apply(lambda x: 1 if x in [1, 2, 3] else 0)

# Define features and labels


df.drop(columns=['num_past_seconds','num_past_starts','num_past_thirds','num_past_wins'], inplace=True)



unique_races = df['race_uid'].unique()
train_races = unique_races[:int(0.8 * len(unique_races))]
test_races = unique_races[int(0.8 * len(unique_races)):]
#df = df.dropna(subset=['dollar_odds'], inplace=True)

#passing the split races into data frames with all of the horse data
train_df = df[df['race_uid'].isin(train_races)]
test_df = df[df['race_uid'].isin(test_races)]

#define the X and Y train data

X_train = train_df.drop(columns=['finish','race_uid','top3'])
y_train = train_df['top3']

X_test = test_df.drop(columns=['finish','race_uid','top3'])
y_test = test_df['top3']

X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_train.mean(), inplace=True)

#Aligning the models
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
print("Model is Training...")

#current best params
model = RandomForestClassifier(n_estimators=200, random_state=42,n_jobs=-1,
                               min_samples_leaf=5,warm_start=False,class_weight='balanced',max_depth=10,min_samples_split=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Predict on training data
y_train_pred = model.predict(X_train)
print("=== Train Set Report ===")
print(classification_report(y_train, y_train_pred))

# Predict on test data (you already have this)
y_test_pred = model.predict(X_test)
print("=== Test Set Report ===")
print(classification_report(y_test, y_test_pred))




joblib.dump(model,"top3classifier.joblib")

