import pandas as pd
import numpy as np

# Load the dataset
file_path = "/home/jialuyu/Data_Final_Project/Revised_James_Code/final_project_mVAE_pipeline/data_processing/original_data_only_t.csv"
data = pd.read_csv(file_path)

# Check for NaN values
nan_rows = data.isnull().any(axis=1)
nan_cols = data.isnull().any(axis=0)

# Print summary
print(f"Total rows with NaN values: {nan_rows.sum()}")
print(f"Total columns with NaN values: {nan_cols.sum()}")

# Identify the specific rows and columns
rows_with_nan = data[nan_rows]
columns_with_nan = data.columns[data.isnull().any()]

print("\nRows with NaN values:")
print(rows_with_nan)

print("\nColumns with NaN values:")
print(columns_with_nan.tolist())

# Save the rows with NaN values for inspection if needed
rows_with_nan.to_csv("rows_with_nan.csv", index=False)
