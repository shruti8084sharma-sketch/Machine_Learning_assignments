import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Load Wine dataset
wine = load_wine()

# Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

print("WINE DATASET")
print(df.head())

# User input
n = int(input("\nEnter number of rows to display: "))

print("\nSelected Rows:")
print(df.head(n))

# Calculate correlation matrix
corr = df.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap - Wine Dataset")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Find strongest positive correlation
corr_matrix = corr.copy()

# Remove diagonal values
for i in range(len(corr_matrix)):
    corr_matrix.iloc[i, i] = -1

# Find pair with highest positive correlation
max_corr = corr_matrix.max().max()
feature1 = corr_matrix.max().idxmax()
feature2 = corr_matrix[feature1].idxmax()

print("\nStrongest Positive Correlation:")
print(feature1, "and", feature2)
print("Correlation value:", round(max_corr, 4))