# Warehouse Clustering UI

A web application for analyzing warehouses using KMeans clustering. This application allows users to input warehouse data and determine which cluster the warehouse belongs to and whether its price is an outlier compared to similar properties.

## Project Structure

- `model.py`: Original model implementation with KMeans clustering
- `api/`: Backend Flask API for serving the model
  - `app.py`: Flask API for predictions
  - `save_model.py`: Script to train and save the model
  - `requirements.txt`: Dependencies for the backend
- `warehouse-clustering-ui/`: Frontend React application
  - `src/`: Source code for the React app
    - `components/`: React components
      - `WarehouseForm.js`: Form for inputting warehouse data
      - `PredictionResult.js`: Component to display prediction results

## Setup and Installation

### Backend Setup

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Train and save the model:
   ```
   cd api
   python save_model.py
   ```

4. Start the Flask API:
   ```
   python app.py
   ```

### Frontend Setup

1. Navigate to the React app directory:
   ```
   cd warehouse-clustering-ui
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```

4. Open your browser and go to http://localhost:3000

## How to Use

1. Fill out the warehouse information form with details about your warehouse.
2. Click "Analyze Warehouse" to submit the data.
3. View the results showing:
   - Which cluster your warehouse belongs to
   - Whether the price is an outlier
   - Detailed price statistics relative to other warehouses in the same cluster

## Model Details

This application uses KMeans clustering with k=4 to group warehouses with similar characteristics. The model takes into account both numerical features (e.g., price, square footage, age) and categorical features (e.g., location, warehouse type, zoning regulations).

The model also detects price outliers within each cluster using the Interquartile Range (IQR) method.