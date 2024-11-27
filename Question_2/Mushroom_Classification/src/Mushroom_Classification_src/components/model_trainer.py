import pandas as pd
import os
from Mushroom_Classification_src.entity.config_entity import ModelTrainerConfig
from Mushroom_Classification_src import logger
from sklearn.linear_model import ElasticNet
import joblib


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path).values
        train_data_target=pd.read_csv(self.config.train_data_target).values
        




        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        lr.fit(train_data, train_data_target)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))

        logger.info(f"Model trained successfully and saved at {os.path.join(self.config.root_dir, self.config.model_name)}")