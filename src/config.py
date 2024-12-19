import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = Path(os.path.dirname(curr_dir))
    
    DATA_PATH = Path(os.path.join(BASE_DIR, "data", "unique_companies_dataset.csv"))

    MODELS_PATH = Path(os.path.join(BASE_DIR, "final_models"))

    GROQ_API_KEY = os.getenv('GROQ_API_KEY')

    GROQ_MODEL_NAME = 'llama3-8b-8192'

    # ESG Weights
    ESG_WEIGHTS = {
        'environment': 0.5,
        'social': 0.3,
        'governance': 0.2
    }