import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from Mushroom_Classification_src.components.model_trainer import ModelTrainer
from Mushroom_Classification_src.entity.config_entity import ModelTrainerConfig

class TestModelTrainer(unittest.TestCase):
    @patch("pandas.read_csv")
    @patch("joblib.dump")
    @patch("os.path.join", return_value="fake_model_path.pkl")
    def test_train(self, mock_path_join, mock_joblib_dump, mock_read_csv):
        # Mock configuration
        mock_config = MagicMock(spec=ModelTrainerConfig)
        mock_config.train_data_path = "fake_train_data.csv"
        mock_config.test_data_path = "fake_test_data.csv"
        mock_config.target_column = "target"
        mock_config.alpha = 0.1
        mock_config.l1_ratio = 0.5
        mock_config.root_dir = "fake_root_dir"
        mock_config.model_name = "model.pkl"

        # Mock training and testing data
        train_data = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
            "target": [7.0, 8.0, 9.0]
        })
        test_data = pd.DataFrame({
            "feature1": [10.0, 11.0, 12.0],
            "feature2": [13.0, 14.0, 15.0],
            "target": [16.0, 17.0, 18.0]
        })
        mock_read_csv.side_effect = [train_data, test_data]  # Mock `read_csv` for train and test data

        # Instantiate the ModelTrainer
        trainer = ModelTrainer(config=mock_config)

        # Call the `train` method
        trainer.train()

        # Verify that `read_csv` was called with correct paths
        mock_read_csv.assert_any_call(mock_config.train_data_path)
        mock_read_csv.assert_any_call(mock_config.test_data_path)

        # Check if `ElasticNet.fit` was called with the right data
        expected_train_x = train_data.drop(columns=["target"])
        expected_train_y = train_data[["target"]]
        mock_joblib_dump.assert_called_once()  # Ensure the model was saved

        # Verify `joblib.dump` call with correct model path
        mock_joblib_dump.assert_called_with(
            unittest.mock.ANY,  # The trained ElasticNet model
            "fake_model_path.pkl"
        )

        # Verify the model path creation
        mock_path_join.assert_called_once_with(mock_config.root_dir, mock_config.model_name)

if __name__ == "__main__":
    unittest.main()
