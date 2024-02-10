import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging

def load_dataset(dataset_path):
    # Load and preprocess dataset
    pf = pd.read_excel(dataset_path, sheet_name="POFA")
    pf = pf.rename(columns={'W/B ratio': 'w/b_ratio', '%Addition': 'perc_add', 'Cement': 'cement', 'POFA': 'pofa',
                            'Density of Concrete': 'concrete_density', 'Fine Agg.': 'fine_agg', 'Coarse Agg.': 'coarse_agg',
                            'Water': 'water', 'LOI': 'loi', 'Curing Age': 'age', 'Compressive Strength': 'compr_strength',
                            'SuperPlasticiser': 'superplasticiser',
                            })
    pf = pf.drop(['References', 'Link', 'MIX RATIO', 'concrete_density'], axis=1)
    pf.dropna(subset=['compr_strength'], inplace=True)

    # Calculate 'water' values
    water = (pf['water'].isnull()) & (~pf['pofa'].isnull() & ~pf['cement'].isnull())
    pf.loc[water, 'pofa'] = (pf.loc[water, 'pofa'] + pf.loc[water, 'cement']) * pf.loc[water, 'w/b_ratio']

    pf.dropna(subset=['fine_agg', 'SiO2', 'water'], inplace=True)

    # Feature engineering
    pf['w/s'] = pf['water'] / pf['fine_agg']
    pf['pl/b'] = pf['superplasticiser'] / (pf['cement'] + pf['pofa'])
    pf['density'] = pf['cement'] + pf['fine_agg'] + pf['coarse_agg']
    pf['w/d'] = pf['water'] / (pf['cement'] + pf['fine_agg'] + pf['coarse_agg'])

    return pf

def split_data(X, y, test_size=0.2, random_state=42):
    # Split data into train, test, and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_data(X_train, X_val, X_test):
    # Scale numeric features using Min/Max scaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def train_model(X_train_scaled, y_train, n_estimators=64, max_depth=4, learning_rate=0.20025, n_jobs=-1, random_state=23):
    # Train the XGBoost model
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, n_jobs=n_jobs, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    return model

def evaluate_model(model, X_test_scaled, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation metrics
    print('Mean Squared Error: %f', mse)
    print('Mean Absolute Error: %f', mae)
    print('R-squared: %f', r2)

    return mse, mae, r2

def save_model(model, filename):
    # Save the trained model using joblib
    joblib.dump(model, filename)
    print('Model saved to %s', filename)

def save_scaler(scaler, filename):
    # Save the scaler using joblib
    joblib.dump(scaler, filename)
    print('Scaler saved to %s', filename)

def main():
    dataset_path = "./ML PCC data.xlsx"
    model_filename = 'xgboost_model.pkl'
    scaler_filename = 'scaler.pkl'

    # Load and preprocess dataset
    pf = load_dataset(dataset_path)

    # Define input and target variables
    X = pf.drop(['compr_strength'], axis=1)
    y = pf['compr_strength']

    # Split data into train, test, and validation sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Scale numeric features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_data(X_train, X_val, X_test)

    # Train the XGBoost model
    model = train_model(X_train_scaled, y_train)

    # Evaluate the model
    evaluate_model(model, X_test_scaled, y_test)

    # Save the trained model and scaler
    save_model(model, model_filename)
    save_scaler(scaler, scaler_filename)

if __name__ == "__main__":
    main()
