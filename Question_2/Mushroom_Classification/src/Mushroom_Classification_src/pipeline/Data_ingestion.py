from Mushroom_Classification_src.components.data_ingestion import DataIngestion
from Mushroom_Classification_src.config.configuration import ConfigurationManager
from Mushroom_Classification_src import logger   


STAGE_NAME="Data Ingestion"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__ == "__main__":
    try:
        logger.info(f"Running {STAGE_NAME} stage")
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
        logger.info(f"Completed {STAGE_NAME} stage")
    except Exception as e:  
        logger.error(f"Failed to run {STAGE_NAME} stage")
        logger.error(e)
        raise e