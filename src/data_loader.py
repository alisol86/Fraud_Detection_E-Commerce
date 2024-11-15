import pandas as pd
import re

class DataLoader:
    def __init__(self, config):
        self.config = config
  
    def read_fraud_data(self):
        print("#### Data Loader -> Reading Fraud Data")
        data_path = self.config['data']['fraud_data_file']
        fraud_data = pd.read_csv(data_path)
        return fraud_data

    def read_ip_mapping(self):
        print("#### Data Loader -> Reading IP to Country mapping Data")
        ip_mapping_path = self.config['data']['ip_country_file']
        ip_mapping_data = pd.read_excel(ip_mapping_path)
        return ip_mapping_data

 