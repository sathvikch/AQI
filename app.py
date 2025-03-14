from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load city models
models = {}
data_folder = "models"
for file in os.listdir(data_folder):
    if file.endswith(".pkl"):
        city_name = file.replace(".pkl", "")
        with open(os.path.join(data_folder, file), "rb") as f:
            models[city_name] = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html", cities=list(models.keys()))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        city = request.form["city"]
        features = [float(request.form[key]) for key in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]]

        if city not in models:
            return jsonify({"error": "Model for selected city not found"}), 400

        model = models[city]
        aqi = model.predict([features])[0]
        return jsonify({"predicted_AQI": round(aqi)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
