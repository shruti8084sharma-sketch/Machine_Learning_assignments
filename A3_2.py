import numpy as np
from sklearn.preprocessing import MinMaxScaler

# User input
n = int(input("Enter number of values: "))

data = []

for i in range(n):
    value = float(input("Enter value: "))
    data.append([value])

# Create scaler
scaler = MinMaxScaler()

# Apply MinMaxScaler
scaled_data = scaler.fit_transform(data)

print("\nOriginal Data:")
print(np.array(data).flatten())

print("\nMinMax Scaled Data:")
print(scaled_data.flatten())