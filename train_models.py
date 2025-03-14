import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Define folders
DATA_DIR = "data"
MODEL_DIR = "models"

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Get all CSV files (assuming each file is named after the city)
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

for file in csv_files:
    city = file.replace(".csv", "")  # Extract city name
    print(f"Training models for {city}...")

    # Load city data
    data_path = os.path.join(DATA_DIR, file)
    data = pd.read_csv(data_path)

    # Feature selection
    features = data[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']]
    target = data['AQI']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train models
    models = {
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor(objective="reg:squarederror"),
        "Gradient Boosting": GradientBoostingRegressor(),
        "SVM": SVR()
    }

    best_model = None
    best_r2 = -np.inf

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)

        print(f"{model_name} R² Score for {city}: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    # Save the best model for the city
    model_path = os.path.join(MODEL_DIR, f"{city}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"Best model for {city} saved at {model_path}\n")

print("Model training completed!")
