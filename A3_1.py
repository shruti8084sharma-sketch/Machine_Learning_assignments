import pandas as pd
import numpy as np

# Number of records
n = int(input("Enter number of records: "))

age = []
salary = []
department = []
experience = []

# User input
for i in range(n):
    print("\nEnter details for employee", i + 1)

    age.append(float(input("Enter Age: ")))
    salary.append(float(input("Enter Salary: ")))
    department.append(input("Enter Department: "))
    experience.append(float(input("Enter Years of Experience: ")))

# Create DataFrame
df = pd.DataFrame({
    "Age": age,
    "Salary": salary,
    "Department": department,
    "Years of Experience": experience
})

# Inject missing values deliberately
df.loc[0, "Age"] = np.nan
if n > 1:
    df.loc[1, "Salary"] = np.nan
if n > 2:
    df.loc[2, "Department"] = np.nan
if n > 3:
    df.loc[3, "Years of Experience"] = np.nan

print("\nDataset with Missing Values:")
print(df)

# Preprocessing
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
df["Years of Experience"] = df["Years of Experience"].fillna(
    df["Years of Experience"].mean()
)
df["Department"] = df["Department"].fillna(df["Department"].mode()[0])

print("\nPreprocessed Dataset:")
print(df)