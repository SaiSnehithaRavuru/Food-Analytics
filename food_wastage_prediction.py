import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('food-data.csv')

# Convert the date column to a numerical format and extract the day of the week
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data['dayofweek'] = data['date'].dt.dayofweek
data['date'] = (data['date'] - pd.to_datetime('1970-01-01')).dt.days

# Calculate the food wastage
data['wastage'] = data['served-qty'] - data['consumed-qty']

# Train a random forest classifier to predict the menu
menu_features = ['event', 'dayofweek']
menu_target = 'menu'
menu_encoder = OneHotEncoder(sparse=False)
encoded_features = menu_encoder.fit_transform(data[menu_features])
encoded_feature_names = menu_encoder.get_feature_names_out(menu_features)
X_train, X_test, y_train, y_test = train_test_split(encoded_features, data[menu_target], test_size=0.2)
menu_model = RandomForestClassifier()
menu_model.fit(X_train, y_train)

# Train a random forest regressor to predict the wastage
wastage_categorical_features = ['event', 'menu-id', 'menu', 'diet', 'flavor_profile']
wastage_numerical_features = ['date', 'calorielevel-per-100gm', 'headcount']
wastage_features = wastage_categorical_features + wastage_numerical_features + ['dayofweek']
wastage_target = 'wastage'
categorical_encoder = OneHotEncoder(sparse=False)
numerical_scaler = MinMaxScaler()
encoded_categorical_features = categorical_encoder.fit_transform(data[wastage_categorical_features])
scaled_numerical_features = numerical_scaler.fit_transform(data[wastage_numerical_features])
encoded_feature_names = list(categorical_encoder.get_feature_names_out(wastage_categorical_features)) + wastage_numerical_features + ['dayofweek']
X = pd.DataFrame(np.hstack([encoded_categorical_features, scaled_numerical_features, data[['dayofweek']]]), columns=encoded_feature_names)
y = data[wastage_target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
wastage_model = RandomForestRegressor()
wastage_model.fit(X_train, y_train)

# Make predictions for a specific date and event
date_str = input('Enter a date (DD-MM-YYYY): ')
date_dt = pd.to_datetime(date_str, format='%d-%m-%Y')
date_num = (date_dt - pd.to_datetime('1970-01-01')).days
dayofweek_num = date_dt.dayofweek
events = ['breakfast', 'lunch', 'snacks']
for event in events:
    # Predict the menu
    new_row = pd.DataFrame([[event, dayofweek_num]], columns=['event', 'dayofweek'])
    new_row_encoded = menu_encoder.transform(new_row)
    menu_prediction_proba = menu_model.predict_proba(new_row_encoded)[0]
    top_menu_items_idx = np.argsort(menu_prediction_proba)[::-1][:3]
    top_menu_items_proba = menu_prediction_proba[top_menu_items_idx]
    top_menu_items_name = menu_model.classes_[top_menu_items_idx]
    print(f'Predicted menu for {event} on {date_str}:')
    for name, proba in zip(top_menu_items_name, top_menu_items_proba):
        print(f'  {name}: {proba:.2f}')
    # Predict the wastage for each menu item
    print(f'Predicted wastage for {event} on {date_str}:')
    wastage_predictions = []
    for name in top_menu_items_name:
        new_row['menu'] = name
        for col in ['menu-id', 'diet', 'flavor_profile']:
            new_row[col] = data[col].mode()[0]
        for col in ['calorielevel-per-100gm', 'headcount']:
            new_row[col] = data[col].mean()
        new_row['date'] = date_num
        new_row_encoded_categorical = categorical_encoder.transform(new_row[wastage_categorical_features])
        new_row_scaled_numerical = numerical_scaler.transform(new_row[wastage_numerical_features])
        new_row_X = pd.DataFrame(np.hstack([new_row_encoded_categorical, new_row_scaled_numerical, new_row[['dayofweek']]]), columns=encoded_feature_names)
        wastage_prediction = wastage_model.predict(new_row_X)
        wastage_predictions.append(wastage_prediction[0])
        print(f'  {name}: {wastage_prediction[0]:.2f}')
    
    # Create a bar graph
    plt.bar(top_menu_items_name, wastage_predictions)
    plt.title(f'Predicted Wastage for {event.capitalize()} on {date_str}')
    plt.xlabel('Menu Item')
    plt.ylabel('Wastage')
    plt.show()
