from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)
# Use a very permissive CORS configuration
CORS(app)

# Add a simple test route for CORS
@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"message": "CORS is working"})

# Define the same features as in model.py
categorical_features = [
    "Location", "Warehouse Type", "Zoning Regulations", "Environmental Concerns",
    "Neighboring Land Use", "Condition of Property", "Security Features", "Energy Efficiency Features"
]
numerical_features = [
    "Price per SqFt", "Total Square Footage", "Age of Warehouse",
    "Distance to Highways (miles)", "Distance to Ports/Airports (miles)",
    "NOI", "Cap Rate (%)", "Year of Last Renovation", "Number of Loading Docks",
    "Clear Height (ft)", "Parking and Storage Capacity"
]

# Define global variables
kmeans = None
encoder = None
scaler = None
q1_by_cluster = None
q3_by_cluster = None
iqr_by_cluster = None

# Get the absolute path to the models directory
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')

# Load the trained model and preprocessors
def load_model():
    global kmeans, encoder, scaler, q1_by_cluster, q3_by_cluster, iqr_by_cluster
    
    # Load the KMeans model
    with open(os.path.join(models_dir, 'kmeans_model.pkl'), 'rb') as f:
        kmeans = pickle.load(f)
    
    # Load the preprocessors
    with open(os.path.join(models_dir, 'encoder.pkl'), 'rb') as f:
        encoder = pickle.load(f)
    
    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load the cluster statistics for outlier detection
    with open(os.path.join(models_dir, 'cluster_stats.pkl'), 'rb') as f:
        cluster_stats = pickle.load(f)
        q1_by_cluster = cluster_stats['q1']
        q3_by_cluster = cluster_stats['q3']
        iqr_by_cluster = cluster_stats['iqr']

# Load the model when the app starts
try:
    load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run save_model.py first to generate the model files.")

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight requests
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    print("Predicting...")
    if kmeans is None:
        return jsonify({"error": "Model not loaded. Please run save_model.py first."}), 500
        
    data = request.json
    
    # Convert input data to DataFrame with a single row
    input_df = pd.DataFrame([data])
    
    # Preprocess the data
    # One-hot encode categorical features
    encoded_categorical = encoder.transform(input_df[categorical_features])
    encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out())
    
    # Standardize numerical features
    scaled_numerical = scaler.transform(input_df[numerical_features])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_features)
    
    # Combine features
    processed_data = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1)
    
    # Predict cluster
    cluster = int(kmeans.predict(processed_data)[0])
    
    # Determine if it's an outlier
    price_per_sqft = float(input_df["Price per SqFt"])
    lower_bound = q1_by_cluster[cluster] - 1.5 * iqr_by_cluster[cluster]
    upper_bound = q3_by_cluster[cluster] + 1.5 * iqr_by_cluster[cluster]
    is_outlier = (price_per_sqft < lower_bound) or (price_per_sqft > upper_bound)
    
    # Create response with CORS headers
    response = jsonify({
        "cluster": cluster,
        "is_outlier": bool(is_outlier),
        "price_stats": {
            "q1": float(q1_by_cluster[cluster]),
            "q3": float(q3_by_cluster[cluster]),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "submitted_price": price_per_sqft
        }
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(debug=True, port=8080) 