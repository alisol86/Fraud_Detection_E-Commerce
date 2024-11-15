import unittest
import numpy as np
import pandas as pd
from src.feature_engineering import FeatureEngineering

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Mock configuration and input data
        self.config = {}
        self.data = None
        self.ip_country_data = pd.DataFrame({
            'lower_bound_ip_address': [10, 20, 30],
            'upper_bound_ip_address': [15, 25, 35],
            'country': ['CountryA', 'CountryB', 'CountryC']
        })

        # Initialize FeatureEngineering instance
        self.feature_engineering = FeatureEngineering(self.config, self.data, self.ip_country_data)

    def test_find_country(self):
        # Test IPs within the bounds
        self.assertEqual(self.feature_engineering._find_country(12), 'CountryA')
        self.assertEqual(self.feature_engineering._find_country(22), 'CountryB')
        self.assertEqual(self.feature_engineering._find_country(33), 'CountryC')

        # Test IPs outside the bounds
        self.assertEqual(self.feature_engineering._find_country(5), 'Unknown')
        self.assertEqual(self.feature_engineering._find_country(40), 'Unknown')

        # Test edge cases
        self.assertEqual(self.feature_engineering._find_country(10), 'CountryA')
        self.assertEqual(self.feature_engineering._find_country(15), 'CountryA')
        self.assertEqual(self.feature_engineering._find_country(20), 'CountryB')
        self.assertEqual(self.feature_engineering._find_country(35), 'CountryC')

if __name__ == '__main__':
    unittest.main()
