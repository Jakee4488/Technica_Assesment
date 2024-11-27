import unittest
from unittest.mock import patch, MagicMock
from Mushroom_Classification_src.components.data_ingestion import DataIngestion, DataIngestionConfig
from pathlib import Path

class TestDataIngestion(unittest.TestCase):

    @patch('data_ingestion.logger')
    @patch('data_ingestion.request.urlretrieve')
    @patch('data_ingestion.os.path.exists')
    def test_download_file_when_file_does_not_exist(self, mock_exists, mock_urlretrieve, mock_logger):
        # Arrange
        mock_exists.return_value = False
        mock_urlretrieve.return_value = ('local_file_path', 'headers')
        config = DataIngestionConfig()
        ingestion = DataIngestion(config)

        # Act
        ingestion.download_file()

        # Assert
        mock_urlretrieve.assert_called_once_with(
            url=config.source_URL,
            filename=config.local_data_file
        )
        mock_logger.info.assert_called_once()
        mock_exists.assert_called_once_with(config.local_data_file)

    @patch('data_ingestion.logger')
    @patch('data_ingestion.get_size')
    @patch('data_ingestion.os.path.exists')
    def test_download_file_when_file_exists(self, mock_exists, mock_get_size, mock_logger):
        # Arrange
        mock_exists.return_value = True
        mock_get_size.return_value = '1MB'
        config = DataIngestionConfig()
        ingestion = DataIngestion(config)

        # Act
        ingestion.download_file()

        # Assert
        mock_logger.info.assert_called_once()
        mock_get_size.assert_called_once_with(Path(config.local_data_file))
        mock_exists.assert_called_once_with(config.local_data_file)

    @patch('data_ingestion.zipfile.ZipFile')
    @patch('data_ingestion.os.makedirs')
    def test_extract_zip_file(self, mock_makedirs, mock_zipfile):
        # Arrange
        mock_zip_ref = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_ref
        config = DataIngestionConfig()
        ingestion = DataIngestion(config)

        # Act
        ingestion.extract_zip_file()

        # Assert
        mock_makedirs.assert_called_once_with(config.unzip_dir, exist_ok=True)
        mock_zipfile.assert_called_once_with(config.local_data_file, 'r')
        mock_zip_ref.extractall.assert_called_once_with(config.unzip_dir)

if __name__ == '__main__':
    unittest.main()