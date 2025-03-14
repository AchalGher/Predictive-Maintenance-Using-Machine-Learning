import pandas as pd
import numpy as np
import os
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

app = FastAPI()

# Load dataset
def load_data():
    file_path = "CMAPSSData/train_FD001.txt"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}. Please check the path.")
    
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    
    column_names = ['unit_number', 'time_in_cycles', 'operational_setting_1', 'operational_setting_2', 
                    'operational_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    df.columns = column_names
    
    df['RUL'] = df.groupby('unit_number')['time_in_cycles'].transform("max") - df['time_in_cycles']
    df['failure'] = df['RUL'].apply(lambda x: 1 if x <= 30 else 0)
    
    return df

df = load_data()

# Feature selection
features = [f'sensor_{i}' for i in range(1, 22)]
X = df[features]
y = df['failure']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model & scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Load saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class SensorData(BaseModel):
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float

@app.get("/")
def home():
    return {"message": "Welcome to Predictive Maintenance API"}

@app.post("/predict")
def predict_failure(data: SensorData):
    input_data = np.array([[getattr(data, f"sensor_{i}") for i in range(1, 22)]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    return {"failure_prediction": int(prediction)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
