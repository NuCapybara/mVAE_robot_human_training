import pandas as pd
from tabulate import tabulate
# Adjust display settings
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 2000)  # Increase display width
pd.set_option('display.float_format', '{:.6f}'.format)  # Format floats for better readability

# Load your CSV file
file_path = "/home/jialuyu/Data_Final_Project/Revised_James_Code/final_project_mVAE_pipeline/data_processing/original_data_with_cur_prev.csv"
df = pd.read_csv(file_path)

# Display a sample of the data
# print(df.head())
# df.to_csv("output.csv", index=False)
# Save the first 100 rows to a CSV file
df.head(100).to_csv("original_data_with_cur_prev_first100.csv", index=False)
print("First 100 rows saved to '.csv'.")

print("Data saved to 'output.csv'. Open it in Excel for better visualization.")
print(tabulate(df, headers='keys', tablefmt='pretty'))