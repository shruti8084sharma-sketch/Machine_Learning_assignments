import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# 1. Load dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["target"] = iris.target

# 2. Basic Data Exploration
print("\n--- First Five Rows ---")
print(df.head())

print("\n--- Dataset Information ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

# 3. Check Missing Values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 4. Correlation Matrix
print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))

# 5. Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="target",
    palette="viridis"
)

plt.title("Sepal Length vs Petal Length by Class")
plt.show()

# 6. Histogram: Distribution of Sepal Length
plt.figure(figsize=(7, 5))

sns.histplot(
    df["sepal length (cm)"],
    kde=True,
    color="blue"
)

plt.title("Distribution of Sepal Length")
plt.show()