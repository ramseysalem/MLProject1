from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)
# Use a very permissive CORS configuration
CORS(app)

# Add a simple test route for CORS
@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"message": "CORS is working"})

# Define global variables
model = None

# Get the absolute path to the models directory
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')

# Load the trained model
def load_model():
    global model
    
    # Create output directory for results if it doesn't exist
    output_dir = os.path.join(current_dir, "investment_analysis_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate timestamp for this analysis run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check if model file exists
    model_file = os.path.join(output_dir, f"warehouse_investment_model_{timestamp}.pkl")
    
    if not os.path.exists(model_file):
        # Train new model if it doesn't exist
        from undervaluedWarehouses import WarehouseInvestmentModel
        
        print("Training new investment model...")
        model = WarehouseInvestmentModel()
        model.train("mock_warehouse_data_large.csv")
        model.save(model_file)
        print(f"Model trained and saved to {model_file}")
    else:
        # Load existing model
        print(f"Loading existing model from {model_file}")
        model = joblib.load(model_file)
    
    print("Model loaded successfully!")

# Load the model when the app starts
try:
    load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure mock_warehouse_data_large.csv exists in the api directory.")

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
    if model is None:
        return jsonify({"error": "Model not loaded. Please ensure the model is properly initialized."}), 500
        
    data = request.json
    
    # Convert input data to DataFrame with a single row
    input_df = pd.DataFrame([data])
    
    # Add required fields that might be missing
    
    # Calculate Total Price from Price per SqFt and Total Square Footage if not provided
    if 'Total Price' not in input_df.columns:
        input_df['Total Price'] = input_df['Price per SqFt'] * input_df['Total Square Footage']
        print(f"Calculated Total Price: {input_df['Total Price'].iloc[0]}")
    
    # Add Status column as required by the model
    input_df['Status'] = 'For Sale'
    
    # Add Warehouse Name if not provided
    if 'Warehouse Name' not in input_df.columns:
        input_df['Warehouse Name'] = 'Submitted Warehouse'
    
    # Analyze the property
    results = model.analyze_properties(input_df)
    
    # Extract the relevant information for the response
    result = results.iloc[0]
    
    # Create response with CORS headers
    response = jsonify({
        "warehouse_name": result['Warehouse Name'],
        "investment_score": int(result['Investment Score']),
        "is_undervalued": bool(result['Is Undervalued']),
        "estimated_annual_roi": float(result['Estimated Annual ROI (%)']),
        "sale_probability": float(result['Sale Probability']),
        "market_segment": int(result['Market Segment']),
        "value_index": float(result['Value Index']),
        "opportunity_tier": result['Opportunity Tier'],
        "price_stats": {
            "price_per_sqft": float(result['Price per SqFt']),
            "cap_rate": float(result['Cap Rate (%)']),
            "total_price": float(result['Total Price']),
            "noi": float(result['NOI'])
        }
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(debug=True, port=8080) 