from src.mlops_project import logger
from src.mlops_project.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline

logger.info("Starting logs.")

STAGE_NAME = "Data ingestion Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e