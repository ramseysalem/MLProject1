import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "/Users/ramseysalem/Documents/Machine Learning 1/mock_warehouse_data_for_sale_sold.csv"
warehouse_data = pd.read_csv(file_path)

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

# Elbow method to find optimal number of clusters
inertia = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(processed_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# Apply KMeans with chosen k
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
warehouse_data["Cluster"] = kmeans.fit_predict(processed_data)

# Outlier detection by IQR within clusters
q1 = warehouse_data.groupby("Cluster")["Price per SqFt"].transform(lambda x: x.quantile(0.25))
q3 = warehouse_data.groupby("Cluster")["Price per SqFt"].transform(lambda x: x.quantile(0.75))
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

warehouse_data["Outlier"] = (warehouse_data["Price per SqFt"] < lower_bound) | (warehouse_data["Price per SqFt"] > upper_bound)

# Isolate for-sale outliers
for_sale_data = warehouse_data[warehouse_data["Status"] == "For Sale"]
outliers = for_sale_data[for_sale_data["Outlier"]]

# Visualize
plt.figure(figsize=(12, 6))
sns.boxplot(x="Cluster", y="Price per SqFt", data=for_sale_data, showfliers=False)
plt.scatter(outliers["Cluster"], outliers["Price per SqFt"], color="red", label="Outliers", alpha=0.7)
plt.xlabel("Cluster")
plt.ylabel("Price per SqFt")
plt.title("Price per SqFt Distribution Across Clusters (For Sale Warehouses)")
plt.legend()
plt.show()
