import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_utils import fetch_and_preprocess_dataset 
from neo4j_config import get_driver  
from config import MODEL_ID_PATH
from risk_management import create_risk_node
import logging
import uuid

driver = get_driver()

logging.info("Fetching and preprocessing dataset...")
X, y = fetch_and_preprocess_dataset()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train.values.ravel())

try:
    with open(MODEL_ID_PATH, 'r') as f:
        model_id = f.read().strip()
    logging.info(f"Loaded model ID: {model_id}")
except FileNotFoundError:
    logging.error(f"{MODEL_ID_PATH} not found. Ensure model_training.py has been executed.")
    raise

def calculate_psi(expected, actual, bucket_type='quantiles', buckets=10):
    """
    Calculates Population Stability Index (PSI).
    """
    def get_bins(data, bucket_type, buckets):
        if bucket_type == 'quantiles':
            return np.percentile(data, np.linspace(0, 100, buckets + 1))
        elif bucket_type == 'intervals':
            return np.linspace(np.min(data), np.max(data), buckets + 1)

    bins = get_bins(expected, bucket_type, buckets)
    expected_percents = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=bins)[0] / len(actual)

    psi = np.sum((expected_percents - actual_percents) * np.log(np.where(expected_percents != 0, expected_percents / actual_percents, 1)))
    return psi

def detect_ks_drift(train_data, new_data):
    """
    Performs KS test for each feature to detect drift.
    """
    drift_results = {}
    for column in train_data.columns:
        statistic, p_value = ks_2samp(train_data[column], new_data[column])
        drift_results[column] = {'KS Statistic': statistic, 'p-value': p_value}
    return drift_results

def detect_drift(train_data, new_data, psi_threshold=0.1, ks_pvalue_threshold=0.05):
    """
    Detects drift using PSI and KS test for each feature.
    """
    psi_values = {}
    ks_drift = detect_ks_drift(train_data, new_data)
    drift_detected = False

    for column in train_data.columns:
        psi_value = calculate_psi(train_data[column], new_data[column])
        psi_values[column] = psi_value
        if psi_value > psi_threshold or ks_drift[column]['p-value'] < ks_pvalue_threshold:
            drift_detected = True
            logging.warning(
                f"Drift detected in feature '{column}': PSI={psi_value:.4f}, KS p-value={ks_drift[column]['p-value']:.4f}"
            )

    return drift_detected, psi_values, ks_drift

logging.info("Performing drift detection...")
train_data_df = pd.DataFrame(X_train, columns=X.columns)
test_data_df = pd.DataFrame(X_test, columns=X.columns)
drift_detected, psi_values, ks_drift = detect_drift(train_data_df, test_data_df)

def create_drift_detection(tx, model_id, drift_detected, psi_values, ks_drift, detection_date):
    """
    Store drift detection results in Neo4j and connect to the Model node.
    """
    drift_id = str(uuid.uuid4())
    tx.run(
        """
        CREATE (d:DriftDetection {
            drift_id: $drift_id, 
            detection_date: $detection_date, 
            drift_detected: $drift_detected
        })
        """,
        drift_id=drift_id,
        detection_date=detection_date,
        drift_detected=drift_detected,
    )
    tx.run(
        """
        MATCH (m:Model {model_id: $model_id})
        MATCH (d:DriftDetection {drift_id: $drift_id})
        MERGE (m)-[:HAS_DRIFT_DETECTION]->(d)
        """,
        model_id=model_id,
        drift_id=drift_id,
    )
    for feature, psi_value in psi_values.items():
        ks_stat = ks_drift[feature]["KS Statistic"]
        p_value = ks_drift[feature]["p-value"]
        tx.run(
            """
            MATCH (d:DriftDetection {drift_id: $drift_id})
            CREATE (dm:DriftMetric {
                feature: $feature, 
                psi: $psi, 
                ks_statistic: $ks_stat, 
                p_value: $p_value
            })
            MERGE (d)-[:HAS_DRIFT_METRIC]->(dm)
            """,
            drift_id=drift_id,
            feature=feature,
            psi=psi_value,
            ks_stat=ks_stat,
            p_value=p_value,
        )

detection_date = datetime.now().isoformat()
with driver.session() as session:
    session.execute_write(create_drift_detection, model_id, drift_detected, psi_values, ks_drift, detection_date)

    for feature, psi_value in psi_values.items():
        ks_stat = ks_drift[feature]["KS Statistic"]
        p_value = ks_drift[feature]["p-value"]

        if psi_value > 0.2 or p_value < 0.01:
            severity = "High"
        elif 0.1 < psi_value <= 0.2 or 0.01 <= p_value < 0.05:
            severity = "Medium"
        else:
            severity = "Low"
            
        description = f"Drift detected in feature '{feature}' with PSI={psi_value:.4f} and KS p-value={p_value:.4f}"
        risk_id = str(uuid.uuid4())

        session.execute_write(
            create_risk_node,
            risk_id=risk_id,
            description=description,
            severity=severity,
            created_at=datetime.now().isoformat(),
            associated_metric_id=feature,  
            metric_type="DriftMetric",
        )

driver.close()
logging.info("Drift detection results successfully stored in Neo4j.")
