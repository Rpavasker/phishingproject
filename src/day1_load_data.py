import pandas as pd

data = pd.read_csv("data/spam.csv", encoding="latin-1")

# Keep only the useful columns (this dataset typically has extra unnamed columns)
data = data[["v1", "v2"]]
data.columns = ["label", "message"]

print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

print("\nClass distribution:")
print(data["label"].value_counts())
