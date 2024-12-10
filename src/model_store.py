import os
import joblib
from src.config import Config

models = {}

def initialize_models():
    try:
        scaler_path = f"{Config.MODELS_PATH}/scaler_object.joblib"
        ml_model_path = f"{Config.MODELS_PATH}/random_forest.joblib"
        
        if os.path.exists(scaler_path) and os.path.exists(ml_model_path):
            models["scaler"] = joblib.load(scaler_path)
            models["ml_model"] = joblib.load(ml_model_path)
            print("Models initialized successfully...")
        else:
            raise FileNotFoundError("Model files not found in the specified path.")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e
