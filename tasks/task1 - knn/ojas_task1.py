import pandas as pd
import random
import math


num_rows = int(input("Enter the number of rows: "))
num_columns = int(input("Enter the number of columns: "))

# Generate random data
data = []
for x in range(num_rows):
    row = [random.randint(1, 20) for x in range(num_columns)]
    data.append(row)

columns = [f"Column{i+1}" for i in range(num_columns)]


df = pd.DataFrame(data, columns=columns)


print("Generated Data (without Sum column):")
print(df)


target = [random.randint(1, 20) for y in range(num_columns)]
df_target = pd.DataFrame([target], columns=columns)

print("\nTarget Data:")
print(df_target)


results = []
for i, row in df.iterrows():
    differences = [abs(target[j] - row[j]) for j in range(num_columns)]  
    row_sum = sum(differences)  
    results.append(row_sum)
    

df["Sum"] = results


print("\nFinal Data with Sums:")
print(df)


df_least_sum = df.nsmallest(3, "Sum")

print("\nRows with the Least Sum Values:")
print(df_least_sum)
