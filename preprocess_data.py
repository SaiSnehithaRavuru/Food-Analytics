import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
data = pd.read_csv('food-data.csv')

# Drop rows with missing values
data = data.dropna()

# Remove duplicates
data = data.drop_duplicates()

# Encode categorical variables into numerical representations
label_encoder = LabelEncoder()
data['event'] = label_encoder.fit_transform(data['event'])
data['menu'] = label_encoder.fit_transform(data['menu'])
data['diet'] = label_encoder.fit_transform(data['diet'])
data['flavor_profile'] = label_encoder.fit_transform(data['flavor_profile'])

# Save the preprocessed data to a new CSV file
data.to_csv('preprocessed_data.csv', index=False)
