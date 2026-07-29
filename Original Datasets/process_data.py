import json
import pandas as pd
import json
import pandas as pd

# Load the JSON dataset
with open("Original Datasets/intents.json", "r") as file:
    data = json.load(file)


# Convert JSON into a tabular format
rows = []

for intent in data["intents"]:
    tag = intent["tag"]

    for pattern in intent["patterns"]:
        rows.append({
            "tag": tag,
            "pattern": pattern,
            "response": intent["responses"][0]
        })

df = pd.DataFrame(rows)

print(df.info())

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# -----------------------------
# Data Quality Analysis
# -----------------------------

print("\nData Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# -----------------------------
# Feature Engineering
# -----------------------------

# Create a new feature: number of words in the pattern
df["pattern_length"] = df["pattern"].apply(lambda x: len(x.split()))

# Create another feature: number of words in the response
df["response_length"] = df["response"].apply(lambda x: len(x.split()))

print("\nFeature Engineering:")
print(df.head())

# Save processed dataset
df.to_csv("Processed Datasets/processed_data.csv", index=False)

# Save column names
columns = pd.DataFrame(df.columns, columns=["Column_Name"])

columns.to_csv(
    "Processed Datasets/processed_data_columns.csv",
    index=False
)

print("\nProcessed files saved successfully!")
