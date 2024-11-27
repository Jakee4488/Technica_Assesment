import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
from Mushroom_Classification_src.entity.config_entity import DataTransformationConfig
from Mushroom_Classification_src.components.data_transformation import DataTransformation

class TestDataTransformation(unittest.TestCase):
    @patch("pandas.read_csv")
    @patch("pandas.DataFrame.to_csv")
    @patch("Mushroom_Classification_src.logger")
    def test_train_test_spliting(self, mock_logger, mock_to_csv, mock_read_csv):
        # Mock the configuration
        mock_config = MagicMock(spec=DataTransformationConfig)
        mock_config.data_path = "fake_data_path.csv"
        mock_config.root_dir = "fake_root_dir"

        # Mock the dataset
        data = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8],
            "label": [0, 1, 0, 1]
        })
        mock_read_csv.return_value = data

        # Create an instance of DataTransformation
        transformer = DataTransformation(config=mock_config)

        # Call the method
        transformer.train_test_spliting()

        # Check if read_csv was called with the correct path
        mock_read_csv.assert_called_once_with(mock_config.data_path)

        # Check if to_csv was called twice (once for train, once for test)
        self.assertEqual(mock_to_csv.call_count, 2)

        # Verify logger calls
        mock_logger.info.assert_any_call("Splited data into training and test sets")
        self.assertEqual(mock_logger.info.call_count, 3)

        # Verify the train-test split
        train_call_args, test_call_args = mock_to_csv.call_args_list
        train_file_path = train_call_args[0][0]
        test_file_path = test_call_args[0][0]
        
        self.assertTrue("train.csv" in train_file_path)
        self.assertTrue("test.csv" in test_file_path)

if __name__ == "__main__":
    unittest.main()
