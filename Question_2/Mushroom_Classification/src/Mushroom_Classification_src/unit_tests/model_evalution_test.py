import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
from Mushroom_Classification_src.components.model_evaluation import ModelEvaluation
from Mushroom_Classification_src.entity.config_entity import ModelEvaluationConfig

class TestModelEvaluation(unittest.TestCase):
    @patch("pandas.read_csv")
    @patch("joblib.load")
    @patch("Mushroom_Classification_src.utils.common.save_json")
    @patch("mlflow.start_run")
    @patch("mlflow.log_metric")
    @patch("mlflow.log_params")
    @patch("mlflow.sklearn.log_model")
    def test_log_into_mlflow(
        self,
        mock_log_model,
        mock_log_params,
        mock_log_metric,
        mock_start_run,
        mock_save_json,
        mock_joblib_load,
        mock_read_csv
    ):
        # Mock configuration
        mock_config = MagicMock(spec=ModelEvaluationConfig)
        mock_config.test_data_path = "fake_test_data.csv"
        mock_config.model_path = "fake_model_path.pkl"
        mock_config.target_column = "target"
        mock_config.mlflow_uri = "http://fake_mlflow_uri"
        mock_config.metric_file_name = "fake_metrics.json"
        mock_config.all_params = {"param1": "value1", "param2": "value2"}

        # Mock test data
        test_data = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
            "target": [7.0, 8.0, 9.0]
        })
        mock_read_csv.return_value = test_data

        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([7.1, 7.9, 8.8])  # Predictions
        mock_joblib_load.return_value = mock_model

        # Instantiate the class
        evaluator = ModelEvaluation(config=mock_config)

        # Call the method
        evaluator.log_into_mlflow()

        # Check if test data was read
        mock_read_csv.assert_called_once_with(mock_config.test_data_path)

        # Check if model was loaded
        mock_joblib_load.assert_called_once_with(mock_config.model_path)

        # Verify predictions
        mock_model.predict.assert_called_once_with(test_data.drop(["target"], axis=1))

        # Verify metrics
        expected_rmse = np.sqrt(((test_data["target"] - [7.1, 7.9, 8.8]) ** 2).mean())
        expected_mae = (abs(test_data["target"] - [7.1, 7.9, 8.8])).mean()
        expected_r2 = 1 - (((test_data["target"] - [7.1, 7.9, 8.8]) ** 2).sum() /
                           ((test_data["target"] - test_data["target"].mean()) ** 2).sum())

        mock_save_json.assert_called_once_with(
            path="fake_metrics.json",
            data={"rmse": expected_rmse, "mae": expected_mae, "r2": expected_r2}
        )

        # Verify MLflow calls
        mock_log_params.assert_called_once_with({"param1": "value1", "param2": "value2"})
        mock_log_metric.assert_any_call("rmse", expected_rmse)
        mock_log_metric.assert_any_call("mae", expected_mae)
        mock_log_metric.assert_any_call("r2", expected_r2)
        mock_log_model.assert_called()

        # Ensure MLflow context is used
        mock_start_run.assert_called_once()

if __name__ == "__main__":
    unittest.main()
