    
from src.mlops_project.config.configuration import ConfigurationManager
from src.mlops_project.components.model_training import ModelTraining
from src.mlops_project import logger

STAGE_NAME = "Model Training Stage"



class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training_config = ModelTraining(config=model_training_config)
        model_training_config.train()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
