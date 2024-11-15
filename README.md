# Fraud Detection in E-Commerce Transaction

This project is designed to detect fraudulent transactions for an E-Commerce store.

## File Structure
```bash
project-root/
│
├── config/
│   └── data_loader.yaml                            # Configuration for data loading
│   └── feature_engineering.yaml.yaml               # Configuration for feature engineering
│
├── src/
│   ├── data_loader.py                              # Script for reading data
│   ├── exploratory_analysis.py                     # Script for EDA reports and statistics
│   ├── feature_engineering.py                      # Script for feature engineering
│   ├── model_train.py                              # Script for training different models
│   ├── model_inference.py                          # Script for making predictions based on models
│   └── __init__.py                                 # Project root setup
│
├── tests/
│   └── test_feature_engineering.py                 # Unit tests for feature_engineering.py script
│
├── data/
│   └──  Fraud_data.csv                             # Fraud dataset
│   └──  IpAddress_to_Country.xlsx                  # IP to Country mapping dataset
│
├── Novelis_TechnicalAssessment_AliSoleymani.ipynb  # Notebook describing the methodology
├── README.md                     # Project documentation
```

The {PROJECT_ROOT_PATH} in the config files is a placeholder which is dynamically replaced with the actual project root directory at runtime.