import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings('ignore')

class WarehouseInvestmentModel:
    """
    A specialized model for identifying undervalued warehouse properties with high investment potential.
    Focuses on detecting outliers (undervalued properties) that are likely to be sold
    and represent strong investment opportunities.
    """
    
    def __init__(self):
        """Initialize the model with feature definitions and empty model containers"""
        # Define feature categories
        self.categorical_features = [
            "Location", "Warehouse Type", "Zoning Regulations", "Environmental Concerns",
            "Neighboring Land Use", "Condition of Property", "Security Features", 
            "Energy Efficiency Features"
        ]
        
        self.numerical_features = [
            "Price per SqFt", "Total Square Footage", "Age of Warehouse", "Total Price",
            "Distance to Highways (miles)", "Distance to Ports/Airports (miles)",
            "NOI", "Cap Rate (%)", "Year of Last Renovation", "Number of Loading Docks",
            "Clear Height (ft)", "Parking and Storage Capacity"
        ]
        
        # Investment-specific features will be calculated during preprocessing
        self.investment_features = [
            "Price_to_NOI_Ratio", 
            "Cap_Rate_to_Avg_Ratio",
            "Price_per_SqFt_to_Avg_Ratio",
            "Location_Price_Ratio",
            "Accessibility_Score",
            "Modernization_Score", 
            "Value_Index"
        ]
        
        # Initialize models
        self.preprocessor = None
        self.classification_model = None
        self.outlier_detector = None
        self.cluster_model = None
        self.location_avg_prices = {}
        self.location_avg_cap_rates = {}
        self.avg_price_per_sqft = 0
        self.avg_cap_rate = 0
    
    def engineer_investment_features(self, df, training=False):
        """
        Create specialized features to highlight investment value
        
        Parameters:
        df: DataFrame with warehouse data
        training: Boolean indicating if this is being used during training
        
        Returns:
        DataFrame with additional investment-focused features
        """
        enhanced_df = df.copy()
        
        # Calculate basic investment ratios
        enhanced_df['Price_to_NOI_Ratio'] = enhanced_df['Total Price'] / enhanced_df['NOI']
        
        # During training, calculate location-based averages for reference
        if training:
            # Calculate location averages for price per sqft
            self.location_avg_prices = enhanced_df.groupby('Location')['Price per SqFt'].mean().to_dict()
            self.location_avg_cap_rates = enhanced_df.groupby('Location')['Cap Rate (%)'].mean().to_dict()
            self.avg_price_per_sqft = enhanced_df['Price per SqFt'].mean()
            self.avg_cap_rate = enhanced_df['Cap Rate (%)'].mean()
        
        # Calculate price ratios compared to location averages (lower is better)
        enhanced_df['Location_Price_Ratio'] = enhanced_df.apply(
            lambda row: row['Price per SqFt'] / self.location_avg_prices.get(row['Location'], self.avg_price_per_sqft), 
            axis=1
        )
        
        # Calculate cap rate compared to average (higher is better)
        enhanced_df['Cap_Rate_to_Avg_Ratio'] = enhanced_df.apply(
            lambda row: row['Cap Rate (%)'] / self.location_avg_cap_rates.get(row['Location'], self.avg_cap_rate),
            axis=1
        )
        
        # Price per sqft compared to overall average (lower is better)
        enhanced_df['Price_per_SqFt_to_Avg_Ratio'] = enhanced_df['Price per SqFt'] / self.avg_price_per_sqft
        
        # Calculate accessibility score (higher is better)
        # Normalize distances so lower distances give higher scores
        max_highway_dist = max(enhanced_df['Distance to Highways (miles)'].max(), 15)
        max_port_dist = max(enhanced_df['Distance to Ports/Airports (miles)'].max(), 50)
        
        highway_score = 1 - (enhanced_df['Distance to Highways (miles)'] / max_highway_dist)
        port_score = 1 - (enhanced_df['Distance to Ports/Airports (miles)'] / max_port_dist)
        
        enhanced_df['Accessibility_Score'] = (highway_score * 0.6) + (port_score * 0.4)
        
        # Calculate modernization score (higher is better)
        current_year = 2025
        years_since_renovation = current_year - enhanced_df['Year of Last Renovation']
        max_years = max(years_since_renovation.max(), 50)  # Cap at 50 years
        renovation_score = 1 - (years_since_renovation / max_years)
        
        # Age score (newer is better)
        max_age = max(enhanced_df['Age of Warehouse'].max(), 100)  # Cap at 100 years
        age_score = 1 - (enhanced_df['Age of Warehouse'] / max_age)
        
        enhanced_df['Modernization_Score'] = (renovation_score * 0.6) + (age_score * 0.4)
        
        # Create value index (higher is better) - key metric for identifying undervalued properties
        # Combines multiple factors that indicate good value
        enhanced_df['Value_Index'] = (
            (1 / enhanced_df['Price_to_NOI_Ratio'].clip(lower=0.01)) * 0.25 +  # Higher NOI relative to price
            enhanced_df['Cap_Rate_to_Avg_Ratio'] * 0.25 +                      # Higher cap rate relative to location
            (1 / enhanced_df['Location_Price_Ratio'].clip(lower=0.01)) * 0.2 +  # Lower price relative to location
            enhanced_df['Accessibility_Score'] * 0.15 +                        # Better accessibility
            enhanced_df['Modernization_Score'] * 0.15                          # More modern facility
        )
        
        return enhanced_df
    
    def create_preprocessor(self):
        """Create a preprocessing pipeline for the model"""
        # Numerical transformer
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Categorical transformer
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])
        
        # Column transformer
        all_numerical_features = self.numerical_features + self.investment_features
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, all_numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return preprocessor
    
    def train(self, data_path):
        """
        Train the complete warehouse investment model
        
        Parameters:
        data_path: Path to the CSV file with warehouse data
        
        Returns:
        Self (for chaining)
        """
        print("Loading and preparing warehouse data...")
        warehouse_data = pd.read_csv(data_path)
        
        # Create target variable (1 for Sold, 0 for For Sale)
        y = (warehouse_data["Status"] == "Sold").astype(int)
        
        # Engineer investment-focused features
        print("Engineering investment features...")
        enhanced_data = self.engineer_investment_features(warehouse_data, training=True)
        
        # Remove non-feature columns
        X = enhanced_data.drop(["Warehouse Name", "Status"], axis=1)
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Create and fit the preprocessor
        print("Creating preprocessing pipeline...")
        self.preprocessor = self.create_preprocessor()
        
        # Prepare the classification model
        print("Training classification model...")
        model_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Define parameter grid
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 15, 25],
            'classifier__min_samples_split': [2, 5],
            'classifier__class_weight': [None, 'balanced']
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            model_pipeline, param_grid, cv=5, 
            scoring='f1', verbose=1, n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.classification_model = grid_search.best_estimator_
        
        # Evaluate the model
        y_pred = self.classification_model.predict(X_test)
        y_prob = self.classification_model.predict_proba(X_test)[:, 1]
        
        print("\nClassification Model Performance:")
        print(classification_report(y_test, y_pred))
        
        # Train outlier detector to identify undervalued properties
        print("\nTraining outlier detector for identifying undervalued properties...")
        
        # Prepare investment metrics only for properties that were sold
        sold_data = enhanced_data[enhanced_data['Status'] == 'Sold']
        sold_X = sold_data.drop(["Warehouse Name", "Status"], axis=1)
        
        # Use only relevant investment features for the outlier detection
        investment_cols = [
            'Price_to_NOI_Ratio', 'Location_Price_Ratio', 'Cap_Rate_to_Avg_Ratio',
            'Price_per_SqFt_to_Avg_Ratio', 'Value_Index'
        ]
        
        # Preprocess the investment features
        outlier_preprocessor = StandardScaler()
        outlier_features = outlier_preprocessor.fit_transform(sold_X[investment_cols])
        
        # Train Isolation Forest for outlier detection
        self.outlier_detector = IsolationForest(
            contamination=0.1,  # Assume ~10% of properties are significantly undervalued
            random_state=42
        )
        self.outlier_detector.fit(outlier_features)
        
        # Train a clustering model to segment properties
        print("\nTraining clustering model for market segmentation...")
        
        # Preprocess all data for clustering
        X_preprocessed = self.preprocessor.fit_transform(X)
        
        # Find optimal number of clusters using silhouette score
        from sklearn.metrics import silhouette_score
        
        # Try several different cluster counts
        k_range = range(3, 8)
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_preprocessed)
            score = silhouette_score(X_preprocessed, cluster_labels)
            silhouette_scores.append(score)
            print(f"K={k}, Silhouette Score: {score:.3f}")
        
        # Select the best k
        best_k = k_range[np.argmax(silhouette_scores)]
        print(f"Best number of clusters: {best_k}")
        
        # Train the final clustering model
        self.cluster_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        self.cluster_model.fit(X_preprocessed)
        
        print("Investment model training complete!")
        return self
    
    def analyze_properties(self, data):
        """
        Analyze properties to identify investment opportunities
        
        Parameters:
        data: DataFrame with warehouse data
        
        Returns:
        DataFrame with analysis results including:
          - Sale probability
          - Investment opportunity score
          - Undervalued flag
          - Market segment (cluster)
          - Value index
        """
        # Make a copy to avoid modifying the original
        data_copy = data.copy()
        
        # Engineer investment features
        enhanced_data = self.engineer_investment_features(data_copy)
        
        # Remove non-feature columns if present
        analysis_data = enhanced_data.copy()
        if "Status" in analysis_data.columns:
            analysis_data = analysis_data.drop("Status", axis=1)
        
        warehouse_names = analysis_data.get("Warehouse Name", pd.Series([f"Warehouse_{i}" for i in range(len(analysis_data))]))
        
        if "Warehouse Name" in analysis_data.columns:
            analysis_data = analysis_data.drop("Warehouse Name", axis=1)
        
        # Make sale probability predictions
        sale_probabilities = self.classification_model.predict_proba(analysis_data)[:, 1]
        
        # Get investment metrics for outlier detection
        investment_cols = [
            'Price_to_NOI_Ratio', 'Location_Price_Ratio', 'Cap_Rate_to_Avg_Ratio',
            'Price_per_SqFt_to_Avg_Ratio', 'Value_Index'
        ]
        
        # Check if these columns exist in the data
        for col in investment_cols:
            if col not in analysis_data.columns:
                raise ValueError(f"Column {col} is missing from the data. Make sure feature engineering was applied.")
        
        # Preprocess investment features
        outlier_preprocessor = StandardScaler()
        outlier_features = outlier_preprocessor.fit_transform(analysis_data[investment_cols])
        
        # Detect undervalued properties
        # Isolation Forest: -1 for outliers, 1 for inliers, we invert to get 1 for outliers
        outlier_scores = -1 * self.outlier_detector.decision_function(outlier_features)
        is_undervalued = (self.outlier_detector.predict(outlier_features) == -1).astype(int)
        
        # Get market segments
        X_preprocessed = self.preprocessor.transform(analysis_data)
        market_segments = self.cluster_model.predict(X_preprocessed)
        
        # Calculate investment opportunity score (0-100)
        # Combines: sale probability, value index, and outlier score
        investment_score = (
            sale_probabilities * 0.4 +                                    # Higher probability of being sold
            enhanced_data['Value_Index'] / enhanced_data['Value_Index'].max() * 0.4 +  # Higher value index 
            (outlier_scores - outlier_scores.min()) / 
            (outlier_scores.max() - outlier_scores.min() + 1e-10) * 0.2   # Higher outlier score (more undervalued)
        ) * 100
        
        # Round scores to integers
        investment_score = investment_score.round().astype(int)
        
        # Calculate estimated annual ROI
        # Base ROI is the cap rate, adjusted for undervalued properties and investment score
        estimated_roi = enhanced_data['Cap Rate (%)'] * (
            1 + (is_undervalued * 0.02) +  # 2% boost for undervalued properties
            ((investment_score / 100) * 0.03)  # Up to 3% additional boost based on investment score
        )
        
        # Assign opportunity tier
        def assign_tier(score):
            if score >= 90:
                return 'Premium'
            elif score >= 80:
                return 'High'
            elif score >= 75:
                return 'Good'
            elif score >= 60:
                return 'Fair'
            else:
                return 'Low'
        
        opportunity_tiers = [assign_tier(score) for score in investment_score]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Warehouse Name': warehouse_names,
            'Sale Probability': sale_probabilities.round(3),
            'Investment Score': investment_score,
            'Opportunity Tier': opportunity_tiers,
            'Is Undervalued': is_undervalued,
            'Estimated Annual ROI (%)': estimated_roi.round(2),
            'Market Segment': market_segments,
            'Value Index': enhanced_data['Value_Index'].round(2),
            'Price per SqFt': enhanced_data['Price per SqFt'].round(2),
            'Cap Rate (%)': enhanced_data['Cap Rate (%)'].round(2),
            'Location': enhanced_data['Location'],
            'NOI': enhanced_data['NOI'].round(0),
            'Total Price': enhanced_data['Total Price'].round(0)
        })
        
        # Sort by investment score (descending)
        results = results.sort_values('Investment Score', ascending=False).reset_index(drop=True)
        
        return results
    
    def save(self, filename='warehouse_investment_model.pkl'):
        """Save the model to a file"""
        joblib.dump(self, filename)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename='warehouse_investment_model.pkl'):
        """Load a saved model"""
        return joblib.load(filename) 