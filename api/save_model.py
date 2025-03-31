import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

# Get the absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
models_dir = os.path.join(current_dir, 'models')

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# Load the dataset
try: 
    file_path = os.path.join(parent_dir, "mock_warehouse_data_for_sale_sold.csv")
    warehouse_data = pd.read_csv(file_path)
    print(f"Successfully loaded data from {file_path}")
except FileNotFoundError as e:
    print(f"File path {file_path} does not exist; Error: {e}")
    # If we can't find the file, exit the script
    raise e

# Feature selection
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

# One-hot encode categorical features
encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
encoded_categorical = encoder.fit_transform(warehouse_data[categorical_features])
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(), index=warehouse_data.index)

# Standardize numerical features
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(warehouse_data[numerical_features])
scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_features, index=warehouse_data.index)

# Combine features
processed_data = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1)

# Apply KMeans with chosen k
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
warehouse_data["Cluster"] = kmeans.fit_predict(processed_data)

# Calculate cluster statistics for outlier detection
cluster_stats = {}
q1_by_cluster = []
q3_by_cluster = []
iqr_by_cluster = []

# Calculate IQR values for each cluster
for i in range(optimal_k):
    cluster_data = warehouse_data[warehouse_data["Cluster"] == i]
    q1 = cluster_data["Price per SqFt"].quantile(0.25)
    q3 = cluster_data["Price per SqFt"].quantile(0.75)
    iqr = q3 - q1
    q1_by_cluster.append(q1)
    q3_by_cluster.append(q3)
    iqr_by_cluster.append(iqr)

cluster_stats = {
    'q1': q1_by_cluster,
    'q3': q3_by_cluster,
    'iqr': iqr_by_cluster
}

# Save the model, preprocessors, and cluster statistics
with open(os.path.join(models_dir, 'kmeans_model.pkl'), 'wb') as f:
    pickle.dump(kmeans, f)

with open(os.path.join(models_dir, 'encoder.pkl'), 'wb') as f:
    pickle.dump(encoder, f)
    
with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
    
with open(os.path.join(models_dir, 'cluster_stats.pkl'), 'wb') as f:
    pickle.dump(cluster_stats, f)

print("Model and preprocessors saved successfully!")

# Print some information about the clusters
for i in range(optimal_k):
    cluster_data = warehouse_data[warehouse_data["Cluster"] == i]
    print(f"Cluster {i}:")
    print(f"  Number of warehouses: {len(cluster_data)}")
    print(f"  Average Price per SqFt: ${cluster_data['Price per SqFt'].mean():.2f}")
    print(f"  Price range: ${cluster_data['Price per SqFt'].min():.2f} - ${cluster_data['Price per SqFt'].max():.2f}")
    print(f"  IQR bounds for outlier detection: ${q1_by_cluster[i] - 1.5 * iqr_by_cluster[i]:.2f} - ${q3_by_cluster[i] + 1.5 * iqr_by_cluster[i]:.2f}")
    print() 