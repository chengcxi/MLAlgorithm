import pandas as pd

# Load the CSV file
df = pd.read_csv('tqqq.csv')

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%b%d-%Y')

# Save the updated DataFrame to a new CSV file
df.to_csv('tqqq_updated.csv', index=False)