from src.CNNClassifier import logging
from src.CNNClassifier.pipeline.stage_01_data_ingetion import DataIngestionTrainingPipeline



STAGE_NAME = "Data Ingestion Stage"
try:
    logging.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logging.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e