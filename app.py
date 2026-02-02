from src.driver_drowsiness_project.components.data_ingestion import DataIngestion
from src.driver_drowsiness_project.components.data_validation import DataValidation
from src.driver_drowsiness_project.components.data_transformation import DataTransformation
from src.driver_drowsiness_project.components.model_tranier import ModelTrainer
from src.driver_drowsiness_project.components.model_evalution import ModelEvaluation

if __name__ == "__main__":
    
    # data_ingestion = DataIngestion()
    # data_ingestion.ingest()
    # data_validation = DataValidation()
    # data_validation.validate()
    # data_trans = DataTransformation()
    # data_trans.transform()
    # model_train = ModelTrainer()
    # model_train.train()
    model_eval = ModelEvaluation()
    model_eval.evaluate()

