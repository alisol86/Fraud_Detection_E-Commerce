import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineering:
    def __init__(self, config, data, ip_country_data):
        self.config = config
        self.data = data
        self.lower_bound_ips = ip_country_data['lower_bound_ip_address'].astype(int).values
        self.upper_bound_ips = ip_country_data['upper_bound_ip_address'].astype(int).values
        self.countries = ip_country_data['country'].values

    def preprocessing(self):
        col_to_drop = self.config['column_to_drop']
        """ drop irrelevant column from the self.data"""
        print("#### Feature Extraction -> preprocessing")
        return self.data.drop(col_to_drop, axis=1)

    def convert_sex_column(self):
        # covert 'sex' column to 0 or 1 in order to be used in the mdoel 
        self.data['sex'] = (self.data['sex'] == 'M').astype(int)
        print("#### Feature Extraction -> converting column type")
        return self.data

    def add_time_feature(self):
        """add a feature to capture the difference between signup_time and purchase_time"""
        # Converting signup_time and purchase_time columns to datetime format
        self.data['signup_time'] = pd.to_datetime(self.data['signup_time'])
        self.data['purchase_time'] = pd.to_datetime(self.data['purchase_time'])

        # Calculating the time difference between signup and purchase in hours
        self.data['time_to_purchase'] = (self.data['purchase_time'] - self.data['signup_time']).dt.total_seconds() / 3600  # hours
        self.data['time_to_purchase'] = self.data['time_to_purchase'].round(2)
        print("#### Feature Extraction -> adding time feature")
        return self.data

    def add_fraud_spike_feature(self):
        """add a feature to capture time-of-year effect (e.g., a "seasonal" feature) between start_date and end_date"""
        # # Add `in_fraud_spike_period` as a binary feature (e.g., high fraud risk Jan 1 - Jan 13)
        start_date = pd.Timestamp(self.config['fraud_spike_period']['start_date'])
        end_date = pd.Timestamp(self.config['fraud_spike_period']['end_date'])
        self.data['in_fraud_spike_period'] = self.data['purchase_time'].apply(lambda x: 1 if x >= start_date and x <= end_date else 0)
        print("#### Feature Extraction -> adding fraud spike feature")
        return self.data

    def add_device_id_feature(self):
        """add a feature to count number of users for each device"""
        # Count the number of unique users for each device
        device_user_count = self.data.groupby('device_id')['user_id'].nunique().reset_index()
        device_user_count.columns = ['device_id', 'user_count_per_device']

        # Merge this count back into the original self.dataset
        self.data = self.data.merge(device_user_count, on='device_id', how='left')
        print("#### Feature Extraction -> adding device count feature")
        return self.data

    def add_ip_shared_feature(self):
        """add feature for number of IP addresses shared"""
        # add column for number of shared IPs
        self.data['n_ip_shared'] = self.data.ip_address.map(self.data.ip_address.value_counts())
        print("#### Feature Extraction -> adding ip_shared feature")
        return self.data

    def _find_country(self, ip):
        """Helper function for IP-to-country lookup."""
        idx = np.searchsorted(self.upper_bound_ips, ip, side='left')
        if 0 <= idx < len(self.lower_bound_ips) and self.lower_bound_ips[idx] <= ip <= self.upper_bound_ips[idx]:
            return self.countries[idx]
        return 'Unknown'

    def add_country_feature_from_ip(self):
            """Add 'country' feature based on IP address."""
            self.data['country'] = self.data['ip_address'].apply(self._find_country)
            print("#### Feature Extraction -> adding country feature")
            return self.data

    def add_country_risk_category_feature(self):
        """Add a risk category based on the number of fraud cases per country."""
        fraud_count_by_country = self.data.groupby('country')['class'].sum()
        
        # Define thresholds for risk categories
        high_risk_threshold = fraud_count_by_country.quantile(self.config['country_risk_category']['high_threshold'])
        low_risk_threshold = fraud_count_by_country.quantile(self.config['country_risk_category']['low_threshold'])

        # Function to assign risk categories
        def categorize_risk(fraud_count):
            if fraud_count >= high_risk_threshold:
                return 'High-Risk'
            elif fraud_count <= low_risk_threshold:
                return 'Low-Risk'
            else:
                return 'Mid-Risk'

        fraud_count_by_country = fraud_count_by_country.apply(categorize_risk)
        self.data['country_risk_category'] = self.data['country'].map(fraud_count_by_country)
        print("#### Feature Extraction -> adding country risk category feature")
        return self.data

    def add_one_hot_encoding_feature(self):
        """Apply one-hot encoding to categorical features."""
        categorical_features = self.config['categorical_features']
        self.data = pd.get_dummies(self.data, columns=categorical_features, dtype='int')
        return self.data

    def select_relevant_features(self):
        """Select relevant features from the self.dataset."""
        features_sel = self.config['selected_features']
        self.data = self.data.loc[:, features_sel]
        return self.data

    def correlation_analysis(self):
        """Perform correlation analysis and drop specified columns."""
        correlation_matrix = self.data.corr()
        # plt.figure(figsize=(12, 8))
        # sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
        # plt.title("Correlation Matrix of Fraud self.data Features")
        # plt.show()
        
        columns_to_drop = self.config['correlation_analysis']['column_to_drop']
        self.data = self.data.drop(columns_to_drop, axis=1)
        return self.data

    def scale_features(self):
        """Scale features using MinMaxScaler."""
        scaler = MinMaxScaler()
        features_to_scale = self.config['features_to_scale']
        self.data[features_to_scale] = scaler.fit_transform(self.data[features_to_scale])
        return self.data
    
    def feature_engineering_pipeline(self):
        """Run all preprocessing steps."""
        self.preprocessing()
        self.convert_sex_column()
        self.add_time_feature()
        self.add_fraud_spike_feature()
        self.add_device_id_feature()
        self.add_ip_shared_feature()
        self.add_country_feature_from_ip()
        self.add_country_risk_category_feature()
        self.add_one_hot_encoding_feature()
        self.select_relevant_features()
        self.correlation_analysis()
        self.scale_features()
        return self.data  # Return the processed data
    
    def __call__(self):
        """Allow the class instance to be called like a function."""
        return self.feature_engineering_pipeline()




