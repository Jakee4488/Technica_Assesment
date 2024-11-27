import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
from Mushroom_Classification_src.entity.config_entity import DataValidationConfig
from Mushroom_Classification_src.components.data_validation import DataValiadtion

class TestDataValidation(unittest.TestCase):
    @patch("pandas.read_csv")
    @patch("builtins.open", new_callable=mock_open)
    def test_validate_all_columns(self, mock_open_file, mock_read_csv):
        # Mock the configuration
        mock_config = MagicMock(spec=DataValidationConfig)
        mock_config.unzip_data_dir = "fake_unzip_data_dir.csv"
        mock_config.STATUS_FILE = "fake_status_file.txt"
        mock_config.all_schema = {"col1": "int", "col2": "float", "col3": "str"}  # Example schema

        # Mock the dataset
        data = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [1.1, 2.2, 3.3],
            "col3": ["a", "b", "c"]
        })
        mock_read_csv.return_value = data

        # Create an instance of DataValiadtion
        validator = DataValiadtion(config=mock_config)

        # Call the method
        result = validator.validate_all_columns()

        # Check if read_csv was called with the correct path
        mock_read_csv.assert_called_once_with(mock_config.unzip_data_dir)

        # Verify that the status file was written correctly
        mock_open_file.assert_called()
        mock_open_file().write.assert_called_with("Validation status: True")

        # Verify the validation result
        self.assertTrue(result)

    @patch("pandas.read_csv")
    @patch("builtins.open", new_callable=mock_open)
    def test_validate_all_columns_invalid(self, mock_open_file, mock_read_csv):
        # Mock the configuration
        mock_config = MagicMock(spec=DataValidationConfig)
        mock_config.unzip_data_dir = "fake_unzip_data_dir.csv"
        mock_config.STATUS_FILE = "fake_status_file.txt"
        mock_config.all_schema = {"col1": "int", "col2": "float"}  # Schema missing 'col3'

        # Mock the dataset
        data = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [1.1, 2.2, 3.3],
            "col3": ["a", "b", "c"]  # Extra column not in the schema
        })
        mock_read_csv.return_value = data

        # Create an instance of DataValiadtion
        validator = DataValiadtion(config=mock_config)

        # Call the method
        result = validator.validate_all_columns()

        # Check if read_csv was called with the correct path
        mock_read_csv.assert_called_once_with(mock_config.unzip_data_dir)

        # Verify that the status file was written correctly
        mock_open_file.assert_called()
        mock_open_file().write.assert_called_with("Validation status: False")

        # Verify the validation result
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
