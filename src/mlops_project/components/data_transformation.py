import os
from src.mlops_project import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from src.mlops_project.entity.config_entity import DataTransformationConfig



class DataTransformation:
    def  __init__(self, config: DataTransformationConfig) -> None:
        self.config=config
    
    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)

        data['CHDRisk'] = data['CHDRisk'].replace({'yes': 1, 'no': 0})


        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index = False)

        logger.info(f"Train/Test Split Created from {self.config.data_path}")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)