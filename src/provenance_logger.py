import os
import json
from logging_config import setup_logging
from config import FEEDBACK_JSON_PATH, PROVENANCE_LOGS_PATH

logger = setup_logging()

os.makedirs(os.path.dirname(PROVENANCE_LOGS_PATH), exist_ok=True)

if not os.path.exists(FEEDBACK_JSON_PATH):
    logger.error(f"{FEEDBACK_JSON_PATH} not found. Ensure feedback data exists.")
    raise FileNotFoundError(f"{FEEDBACK_JSON_PATH} not found.")

logger.info("Loading feedback data...")
with open(FEEDBACK_JSON_PATH, 'r') as f:
    feedback_data = json.load(f)
logger.info("Feedback data loaded successfully.")

logger.info("Generating provenance logs...")
provenance_logs = [
    {
        "prediction_id": entry.get('prediction_id'),
        "input_data": entry.get('input_data'),
        "predicted_value": entry.get('predicted')
    }
    for entry in feedback_data
]

logger.info(f"Saving provenance logs to {PROVENANCE_LOGS_PATH}...")
with open(PROVENANCE_LOGS_PATH, 'w') as f:
    json.dump(provenance_logs, f, indent=4)

logger.info(f"Provenance logs successfully saved to {PROVENANCE_LOGS_PATH}.")
