from src.driver_drowsiness_project.components.data_ingestion import DataIngestion
from src.driver_drowsiness_project.components.data_validation import DataValidation
from src.driver_drowsiness_project.components.data_transformation import DataTransformation

if __name__ == "__main__":
    
    data_ingestion = DataIngestion()
    data_ingestion.ingest()
    data_validation = DataValidation()
    data_validation.validate()
    data_trans = DataTransformation()
    data_trans.transform()

