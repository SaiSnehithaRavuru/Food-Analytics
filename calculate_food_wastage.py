import pandas as pd

# Load the dataset
df = pd.read_csv('food-data.csv')

# Calculate food wastage
df['food_wastage'] = df['served-qty'] - df['consumed-qty']

# Save the updated dataset with food wastage column
df.to_csv('food-data-with-wastage.csv', index=False)
