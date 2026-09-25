import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Load Wine dataset
wine = load_wine()

# Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

print("WINE DATASET")
print(df.head())

# Display all numerical attributes
print("\nAvailable Numerical Attributes:")
for i, column in enumerate(df.columns, 1):
    print(i, ".", column)

# User input
choice = input("\nEnter attribute numbers separated by comma (e.g., 1,2,3): ")

# Convert input into column names
numbers = [int(x.strip()) for x in choice.split(",")]
selected_columns = [df.columns[i - 1] for i in numbers]

# Display selected data
print("\nSelected Attributes:")
print(df[selected_columns].head())

# Plot boxplots
plt.figure(figsize=(12, 7))
df[selected_columns].boxplot()

plt.title("Boxplots of Selected Numerical Attributes")
plt.xlabel("Attributes")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()