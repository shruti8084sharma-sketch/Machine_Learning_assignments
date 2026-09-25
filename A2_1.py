import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Load Wine dataset
wine = load_wine()

# Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["Target"] = wine.target

print("========== WINE DATASET ==========")

# User input
n = int(input("Enter number of rows to display: "))

# Validate input
if n > len(df):
    print("Only", len(df), "rows are available.")
    n = len(df)

if n <= 0:
    print("Please enter a positive number.")
else:

    # Display user-selected rows
    print("\n========== SELECTED ROWS ==========")
    print(df.head(n))

    # Shape
    print("\n========== DATASET SHAPE ==========")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    # Information
    print("\n========== DATASET INFORMATION ==========")
    df.info()

    # Statistical summary
    print("\n========== STATISTICAL SUMMARY ==========")
    print(df.describe())

    # Missing values
    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    # Duplicate values
    print("\n========== DUPLICATE ROWS ==========")
    print("Duplicate rows:", df.duplicated().sum())

    # Target distribution
    print("\n========== TARGET DISTRIBUTION ==========")
    print(df["Target"].value_counts().sort_index())

    # Histogram
    df.hist(figsize=(15, 12), bins=15)
    plt.suptitle("Wine Dataset Histograms")
    plt.tight_layout()
    plt.show()

    # Boxplot
    plt.figure(figsize=(15, 7))
    df.drop(columns=["Target"]).boxplot()
    plt.title("Boxplots of Wine Dataset")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Correlation matrix
    print("\n========== CORRELATION MATRIX ==========")
    print(df.corr())

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(df.corr(), cmap="coolwarm")
    plt.colorbar()
    plt.xticks(range(len(df.columns)), df.columns, rotation=90)
    plt.yticks(range(len(df.columns)), df.columns)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    print("\n========== EDA COMPLETED ==========")