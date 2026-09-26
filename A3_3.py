import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# User input
n = int(input("Enter number of values: "))

data = []

for i in range(n):
    value = float(input("Enter value: "))
    data.append([value])

# StandardScaler
standard = StandardScaler()
standard_data = standard.fit_transform(data)

# MinMaxScaler
minmax = MinMaxScaler()
minmax_data = minmax.fit_transform(data)

print("\nOriginal Data:")
print(np.array(data).flatten())

print("\nStandardScaler:")
print(standard_data.flatten())

print("\nMinMaxScaler:")
print(minmax_data.flatten())

print("\nRange of StandardScaler:")
print(standard_data.min(), "to", standard_data.max())

print("\nRange of MinMaxScaler:")
print(minmax_data.min(), "to",
 minmax_data.max())