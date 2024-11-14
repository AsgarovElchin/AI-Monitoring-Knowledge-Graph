import os
import json
import logging
from datetime import datetime
from data_utils import fetch_and_preprocess_dataset  
from neo4j_config import get_driver  
from config import FEEDBACK_JSON_PATH, FEEDBACK_LOG_PATH

driver = get_driver()

if not os.path.exists(FEEDBACK_JSON_PATH):
    logging.error(f"{FEEDBACK_JSON_PATH} not found. Ensure predictions have been generated.")
    raise FileNotFoundError(f"{FEEDBACK_JSON_PATH} not found.")

with open(FEEDBACK_JSON_PATH, 'r') as f:
    predictions = json.load(f)
logging.info(f"Loaded feedback data from {FEEDBACK_JSON_PATH}.")

feedback_log = []

def create_feedback_node(tx, prediction_id, correct, notes, timestamp):
    """
    Create Feedback nodes and link them to Prediction nodes in Neo4j.
    """
    result = tx.run(
        """
        MATCH (p:Prediction {prediction_id: $prediction_id})
        CREATE (f:Feedback {correct: $correct, notes: $notes, timestamp: $timestamp})
        MERGE (p)-[:HAS_FEEDBACK]->(f)
        RETURN p, f
        """,
        prediction_id=prediction_id,
        correct=correct,
        notes=notes,
        timestamp=timestamp,
    )
    if not result.peek():
        logging.warning(f"No Prediction node found with prediction_id: {prediction_id}")
    else:
        logging.info(f"Feedback linked to Prediction ID: {prediction_id}")

logging.info("Starting feedback collection process...")
with driver.session() as session:
    for entry in predictions:
        prediction_id = entry.get('prediction_id')
        input_data = entry.get('input_data')
        predicted_value = entry.get('predicted')
        actual_value = entry.get('actual')

        print("\n--- Review Prediction ---")
        print(f"Prediction ID: {prediction_id}")
        print(f"Input Data: {json.dumps(input_data, indent=2)}")
        print(f"Predicted: {predicted_value}, Actual: {actual_value}")

        while True:
            correct_response = input("Is this prediction correct? (y/n): ").strip().lower()
            if correct_response in ['y', 'n']:
                correct = correct_response == 'y'
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

        notes = input("Additional notes (optional): ").strip()
        timestamp = datetime.now().isoformat()

        session.execute_write(create_feedback_node, prediction_id, correct, notes, timestamp)

        feedback_entry = {
            'prediction_id': prediction_id,
            'input_data': input_data,
            'predicted': predicted_value,
            'actual': actual_value,
            'correct': correct,
            'notes': notes,
            'timestamp': timestamp,
        }
        feedback_log.append(feedback_entry)

        continue_response = input("Do you want to continue to the next prediction? (y/n): ").strip().lower()
        if continue_response != 'y':
            logging.info("User exited the feedback collection process.")
            break

os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)
with open(FEEDBACK_LOG_PATH, 'w') as f:
    json.dump(feedback_log, f, indent=4)
logging.info(f"Feedback log saved to {FEEDBACK_LOG_PATH}.")

driver.close()
logging.info("Feedback collection process completed.")
