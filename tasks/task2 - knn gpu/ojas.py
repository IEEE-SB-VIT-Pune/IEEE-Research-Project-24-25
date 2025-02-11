import pandas as pd
import random
import cupy as cp


num_rows = int(input("Enter the number of rows: "))
num_columns = int(input("Enter the number of columns: "))


data = [[random.randint(1, 20) for x in range(num_columns)] for x in range(num_rows)]
columns = [f"Column{i+1}" for i in range(num_columns)]
df = pd.DataFrame(data, columns=columns)

print("Generated Data (without Sum column):")
print(df)


target = [random.randint(1, 20) for y in range(num_columns)]
df_target = pd.DataFrame([target], columns=columns)

print("\nTarget Data:")
print(df_target)

data_gpu = cp.array(data)
target_gpu = cp.array(target)


differences_gpu = cp.abs(data_gpu - target_gpu)  
row_sums_gpu = cp.sum(differences_gpu, axis=1)   

results = cp.asnumpy(row_sums_gpu)


df["Sum"] = results

print("\nFinal Data with Sums:")
print(df)


df_least_sum = df.nsmallest(3, "Sum")
print("\nRows with the Least Sum Values:")
print(df_least_sum)
