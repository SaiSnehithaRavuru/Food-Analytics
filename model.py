import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Load the data
data = pd.read_csv('food-data.csv')

# Convert the date column to a numerical format
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data['date'] = (data['date'] - pd.to_datetime('1970-01-01')).dt.days

# Calculate the food wastage
data['wastage'] = data['served-qty'] - data['consumed-qty']

# Train a decision tree classifier to predict the menu
menu_features = ['date', 'event']
menu_target = 'menu'
menu_encoder = OneHotEncoder(sparse=False)
encoded_features = menu_encoder.fit_transform(data[menu_features])
encoded_feature_names = menu_encoder.get_feature_names_out(menu_features)
X_train, X_test, y_train, y_test = train_test_split(encoded_features, data[menu_target], test_size=0.2)
menu_model = DecisionTreeClassifier()
menu_model.fit(X_train, y_train)

# Train a linear regression model to predict the wastage
wastage_categorical_features = ['event', 'menu-id', 'menu', 'diet', 'flavor_profile']
wastage_numerical_features = ['date', 'calorielevel-per-100gm', 'headcount']
wastage_features = wastage_categorical_features + wastage_numerical_features
wastage_target = 'wastage'
categorical_encoder = OneHotEncoder(sparse=False)
numerical_scaler = MinMaxScaler()
encoded_categorical_features = categorical_encoder.fit_transform(data[wastage_categorical_features])
scaled_numerical_features = numerical_scaler.fit_transform(data[wastage_numerical_features])
encoded_feature_names = list(categorical_encoder.get_feature_names_out(wastage_categorical_features)) + wastage_numerical_features
X = pd.DataFrame(np.hstack([encoded_categorical_features, scaled_numerical_features]), columns=encoded_feature_names)
y = data[wastage_target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
wastage_model = LinearRegression()
wastage_model.fit(X_train, y_train)

def predict(date, event):
    # Check if the date is valid
    try:
        datetime.strptime(date, '%d-%m-%Y')
    except ValueError:
        print('Invalid date format. Please enter a date in the format DD-MM-YYYY.')
        return

    # Predict the menu and wastage
    menu_prediction, wastage_prediction = predict(date, event)

    # Print the results
    print('Predicted menu:', menu_prediction)
    print('Predicted wastage:', wastage_prediction)

if __name__ == '__main__':
    date = input('Enter a date (DD-MM-YYYY): ')
    event = input('Enter an event: ')
    menu_prediction, wastage_prediction = predict(date, event)
    print('Predicted menu:', menu_prediction)
    print('Predicted wastage:', wastage_prediction)
