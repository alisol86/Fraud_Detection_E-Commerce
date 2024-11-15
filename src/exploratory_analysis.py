import yaml
from ydata_profiling import ProfileReport
from data_loader import DataLoader
from utils import read_config

def standard_checks(data):
    print("data sample: \n", data.head())
    print("data shape: ", data.shape)
    print("column names: \n", data.columns)
    print("column types: \n", data.dtypes)
    print("null check: \n", data.isna().sum())


def profile_report(data):
    profile = ProfileReport(data, title="Profiling Report")
    profile.to_file("Fraud_EDA_report.html")

def run_exploratory_analysis(data):
    standard_checks(data)
    profile_report(data)


if __name__ == "__main__":
    cfg = read_config()
    data = DataLoader(cfg).read_fraud_data()
    run_exploratory_analysis(data)