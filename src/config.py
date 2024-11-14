import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
PREDICTIONS_CSV_PATH = os.path.join(DATA_DIR, "predictions.csv")
FEEDBACK_JSON_PATH = os.path.join(DATA_DIR, "feedback_data.json")
PROVENANCE_LOGS_PATH = os.path.join(DATA_DIR, "provenance_logs.json")
FEEDBACK_LOG_PATH = os.path.join(DATA_DIR, "feedback_log.json")

MODEL_ID_PATH = os.path.join(BASE_DIR, "model_id.txt")
