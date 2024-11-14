import os
import json
import csv
import uuid
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_utils import fetch_and_preprocess_dataset  
from neo4j_config import get_driver  
import logging

driver = get_driver()

logging.info("Fetching and preprocessing dataset...")
X, y = fetch_and_preprocess_dataset()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
logging.info(f"Model training completed with accuracy: {accuracy:.4f}")

model_id = str(uuid.uuid4())
model_version = "1.0"
training_date = datetime.now().isoformat()
training_params = {"max_iter": 1000, "test_size": 0.2, "random_state": 42}

def create_or_update_model_node(tx, model_id, model_version, training_date, accuracy, training_params):
    tx.run(
        """
        MERGE (m:Model {model_version: $model_version})
        ON CREATE SET 
            m.model_id = $model_id, 
            m.training_date = $training_date, 
            m.accuracy = $accuracy, 
            m.training_params = $training_params
        ON MATCH SET 
            m.training_date = $training_date, 
            m.accuracy = $accuracy, 
            m.training_params = $training_params
        """,
        model_id=model_id,
        model_version=model_version,
        training_date=training_date,
        accuracy=accuracy,
        training_params=json.dumps(training_params),
    )

def create_dataset_node(tx, dataset_name, dataset_version):
    tx.run(
        """
        MERGE (d:Dataset {name: $dataset_name, version: $dataset_version})
        """,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
    )

def link_model_to_dataset(tx, model_id, dataset_name, dataset_version):
    tx.run(
        """
        MATCH (m:Model {model_id: $model_id})
        MATCH (d:Dataset {name: $dataset_name, version: $dataset_version})
        MERGE (m)-[:TRAINED_ON]->(d)
        """,
        model_id=model_id,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
    )

def create_prediction_node(tx, prediction_id, input_data, predicted_value, actual_value):
    tx.run(
        """
        CREATE (p:Prediction {
            prediction_id: $prediction_id, 
            input_data: $input_data, 
            predicted_value: $predicted_value, 
            actual_value: $actual_value
        })
        """,
        prediction_id=prediction_id,
        input_data=json.dumps(input_data),
        predicted_value=predicted_value,
        actual_value=actual_value,
    )

def link_prediction_to_model(tx, prediction_id, model_id):
    tx.run(
        """
        MATCH (p:Prediction {prediction_id: $prediction_id})
        MATCH (m:Model {model_id: $model_id})
        MERGE (m)-[:GENERATED]->(p)
        """,
        prediction_id=prediction_id,
        model_id=model_id,
    )

logging.info("Generating predictions...")
predictions = model.predict(X_test)
feedback_data = []

for i, prediction in enumerate(predictions):
    feedback_entry = {
        'prediction_id': str(uuid.uuid4()),
        'input_data': {key: float(value) for key, value in zip(X.columns, X_test[i])},
        'predicted': int(prediction),
        'actual': int(y_test.iloc[i].item()),
    }
    feedback_data.append(feedback_entry)

logging.info("Saving predictions to CSV...")
os.makedirs('./data', exist_ok=True)
predictions_csv_path = './data/predictions.csv'
with open(predictions_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['prediction_id', 'input_data', 'predicted', 'actual'])
    writer.writeheader()
    writer.writerows(feedback_data)
logging.info(f"Predictions saved to {predictions_csv_path}.")

feedback_data_path = './data/feedback_data.json'
with open(feedback_data_path, 'w') as f:
    json.dump(feedback_data, f, indent=4)
logging.info(f"Feedback data saved to {feedback_data_path}.")

with driver.session() as session:
    session.execute_write(create_or_update_model_node, model_id, model_version, training_date, accuracy, training_params)
    session.execute_write(create_dataset_node, "Heart Disease Dataset", "1.0")
    session.execute_write(link_model_to_dataset, model_id, "Heart Disease Dataset", "1.0")

    for feedback_entry in feedback_data:
        prediction_id = feedback_entry['prediction_id']
        session.execute_write(create_prediction_node, prediction_id, feedback_entry['input_data'], feedback_entry['predicted'], feedback_entry['actual'])
        session.execute_write(link_prediction_to_model, prediction_id, model_id)

model_id_path = './model_id.txt'
with open(model_id_path, 'w') as f:
    f.write(model_id)
logging.info(f"Model ID saved to {model_id_path}.")

driver.close()
logging.info("Model, dataset, and predictions data successfully stored in Neo4j.")
