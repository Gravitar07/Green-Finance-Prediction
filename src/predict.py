import numpy as np
from src.llm import LLM
from src.model_store import models
from src.exception import *
from src.logger import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self):
        try:
            self.scaler = models["scaler"]
            logger.info("Scaler loaded successfully.")
        except Exception as e:
            log_exception(e, "Error loading scaler in DataPreprocessor.")
            raise ModelLoadingError("Could not load the scaler for data preprocessing.")
        
    def preprocess(self, input_data):
        try:
            scaled_data = self.scaler.transform(np.array(input_data).reshape(1, -1))
            logger.info("Data preprocessing completed successfully.")
            return scaled_data
        except Exception as e:
            log_exception(e, "Error in preprocessing data in DataPreprocessor.")
            raise PreprocessingError("Preprocessing failed. Ensure input data format is correct.") from e
        
    
class ML_Model_Predictor:
    def __init__(self):
        try:
            self.model = models["ml_model"]
            logger.info("ML model loaded successfully.")
        except Exception as e:
            log_exception(e, "Failed to load ML model in ML_Model_Predictor.")
            raise ModelLoadingError("Could not load ML model. Please check model path and format.") from e

    def predict(self, preprocessed_data):
        try:
            prediction = self.model.predict_proba(preprocessed_data)
            logger.info("ML model prediction completed successfully.")
            return prediction
        except Exception as e:
            log_exception(e, "Error during ML model prediction in ML_Model_Predictor.")
            raise PredictionError("Prediction failed. Ensure input data format is correct.") from e


def prediction(impact_area_community, impact_area_environment, impact_area_customers, impact_area_governance, certification_cycle, input_raw_data):
    try:
        # Initialize classes
        preprocessor = DataPreprocessor()
        ml_predictor = ML_Model_Predictor()
        llm = LLM()
        
        # Prepare structured data input for ML model
        structured_data = [impact_area_community, impact_area_environment, impact_area_customers, impact_area_governance, certification_cycle]
        preprocessed_data = preprocessor.preprocess(structured_data)
        
        # Get predictions from ML model
        ml_prediction_result = ml_predictor.predict(preprocessed_data)

        result = f"""
        Green Finance Report:
        
        **ML Model Risk Probability Prediction:** {ml_prediction_result}
        
        {input_raw_data}
        """
        
        # Generate LLM report
        report = llm.inference(result=result)
        logger.info("LLM report generated successfully.")
        return report
    
    except Exception as e:
        log_exception(e, "Error in prediction function.")
        raise PredictionError("Prediction function encountered an error. Check inputs and model paths.") from e
