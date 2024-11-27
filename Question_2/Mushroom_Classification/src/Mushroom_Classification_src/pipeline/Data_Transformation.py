from Mushroom_Classification_src.components.data_transformation import DataTransformation
from Mushroom_Classification_src.config.configuration import ConfigurationManager
from Mushroom_Classification_src import logger   
from pathlib import Path




STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
               config = ConfigurationManager()
               data_transformation_config = config.get_data_transformation_config()
    
                # Create an instance of DataTransformation with the configuration
               data_transformation = DataTransformation(config=data_transformation_config)
    
                # Perform the full data transformation process
               train_pca, test_pca, target_train, target_test = data_transformation.transform()
    
                # Save transformed data
               data_transformation.save_transformed_data(train_pca, test_pca, target_train, target_test)
    
               logger.info("Data transformation pipeline completed successfully and data saved.")

            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e)





if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


