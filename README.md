## Predictive Maintenance Using Machine Learning

### Overview

This project predicts equipment failures using sensor data, allowing for proactive maintenance scheduling. It leverages machine learning models trained on the NASA CMAPSS dataset to classify whether a machine is at risk of failure.

Features

✅ Data Preprocessing – Cleans and prepares sensor data for modeling.

✅ Feature Engineering – Extracts meaningful features to improve predictions.

✅ Machine Learning Models – Uses Random Forest, XGBoost, and Logistic Regression.

✅ Model Evaluation – Assesses performance using accuracy, precision, recall, and F1-score.

✅ API Deployment – Deploys the trained model as a FastAPI web service.

Dataset
The dataset used is from NASA's Turbofan Engine Degradation Simulation (CMAPSS) dataset. It contains multiple sensor readings that indicate engine health over time.

Installation

To run this project locally, follow these steps:

1️⃣ Clone the Repository

git clone https://github.com/your-username/Predictive-Maintenance-Using-Machine-Learning.git  
cd Predictive-Maintenance-Using-Machine-Learning  

2️⃣ Install Dependencies

Ensure you have Python installed, then install the required packages:
pip install -r requirements.txt  

3️⃣ Train the Model

Run the following script to preprocess data and train the model:
python train.py  

4️⃣ Run the API

To deploy the trained model using FastAPI:

uvicorn app:app --host 0.0.0.0 --port 8000 --reload  

✅ API Deployment Link:

👉 https://predictive-maintenance-using-machine.onrender.com

Now, your API is available at: http://127.0.0.1:8000 (for local testing).

Usage

1️⃣ Make a Prediction

Send a POST request to the API with sensor data:

Example Request:

{

  "sensor_1": 0.5, "sensor_2": 0.2, "sensor_3": 0.3, "sensor_4": 0.6, "sensor_5": 0.1,
  "sensor_6": 0.4, "sensor_7": 0.7, "sensor_8": 0.9, "sensor_9": 0.2, "sensor_10": 0.5,
  "sensor_11": 0.1, "sensor_12": 0.3, "sensor_13": 0.6, "sensor_14": 0.4, "sensor_15": 0.8,
  "sensor_16": 0.2, "sensor_17": 0.5, "sensor_18": 0.7, "sensor_19": 0.9, "sensor_20": 0.4,
  "sensor_21": 0.6
  
}

Example Response:

{

  "failure_prediction": 1
  
}

(1 = Failure, 0 = No Failure)

