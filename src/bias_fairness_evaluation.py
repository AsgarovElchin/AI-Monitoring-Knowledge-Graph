import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from risk_management import create_risk_node
import logging
import uuid
from config import MODEL_ID_PATH
from data_utils import fetch_and_preprocess_dataset  
from neo4j_config import get_driver  

driver = get_driver()

X, y = fetch_and_preprocess_dataset()

sensitive_attribute = X['sex']

X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X, y, sensitive_attribute, test_size=0.2, random_state=42
)
logging.info("Training logistic regression model...")
model = LogisticRegression(max_iter=5000)  
model.fit(X_train, y_train.values.ravel())
logging.info("Model training completed.")

y_pred = model.predict(X_test)

try:
    with open(MODEL_ID_PATH, 'r') as f:
        model_id = f.read().strip()
    logging.info(f"Loaded model ID: {model_id}")
except FileNotFoundError:
    logging.error(f"{MODEL_ID_PATH} not found. Ensure model_training.py has been executed.")
    raise

def demographic_parity(y_pred, sensitive_test):
    """
    Calculate demographic parity.
    """
    positive_rate_male = np.mean(y_pred[sensitive_test == 1])
    positive_rate_female = np.mean(y_pred[sensitive_test == 0])
    dp_diff = abs(positive_rate_male - positive_rate_female)
    return dp_diff, positive_rate_male, positive_rate_female

def equalized_odds(y_test, y_pred, sensitive_test):
    """
    Calculate equalized odds (TPR and FPR parity).
    """
    results = {}
    for group, group_name in zip([1, 0], ["Male", "Female"]):
        cm = confusion_matrix(
            y_test[sensitive_test == group], y_pred[sensitive_test == group]
        ).ravel()
        tn, fp, fn, tp = (cm.tolist() + [0] * (4 - len(cm)))[:4]
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        results[group_name] = {"TPR": tpr, "FPR": fpr}
    tpr_diff = abs(results["Male"]["TPR"] - results["Female"]["TPR"])
    fpr_diff = abs(results["Male"]["FPR"] - results["Female"]["FPR"])
    return results, tpr_diff, fpr_diff

logging.info("Calculating bias and fairness metrics...")
dp_diff, male_rate, female_rate = demographic_parity(y_pred, sensitive_test)
eo_results, tpr_diff, fpr_diff = equalized_odds(y_test, y_pred, sensitive_test)

def add_bias_metrics(tx, model_id, dp_diff, male_rate, female_rate, bias_id):
    """
    Add bias metrics to Neo4j.
    """
    tx.run(
        """
        MATCH (m:Model {model_id: $model_id})
        CREATE (b:BiasMetric {
            metric: "demographic_parity", 
            value: $dp_diff, 
            male_rate: $male_rate, 
            female_rate: $female_rate, 
            bias_id: $bias_id, 
            created_at: $created_at
        })
        MERGE (m)-[:HAS_BIAS_METRIC]->(b)
        """,
        model_id=model_id,
        dp_diff=dp_diff,
        male_rate=male_rate,
        female_rate=female_rate,
        bias_id=bias_id,
        created_at=datetime.now().isoformat(),
    )

def add_fairness_metrics(tx, model_id, tpr_diff, fpr_diff, fairness_id):
    """
    Add fairness metrics to Neo4j.
    """
    tx.run(
        """
        MATCH (m:Model {model_id: $model_id})
        CREATE (f:FairnessMetric {
            metric: "equalized_odds", 
            tpr_diff: $tpr_diff, 
            fpr_diff: $fpr_diff, 
            fairness_id: $fairness_id, 
            created_at: $created_at
        })
        MERGE (m)-[:HAS_FAIRNESS_METRIC]->(f)
        """,
        model_id=model_id,
        tpr_diff=tpr_diff,
        fpr_diff=fpr_diff,
        fairness_id=fairness_id,
        created_at=datetime.now().isoformat(),
    )

logging.info("Storing metrics in Neo4j...")
with driver.session() as session:
    bias_id = str(uuid.uuid4())
    session.execute_write(add_bias_metrics, model_id, dp_diff, male_rate, female_rate, bias_id)
    dp_severity = "High" if dp_diff > 0.2 else "Medium" if dp_diff > 0.1 else "Low"
    session.execute_write(
        create_risk_node,
        risk_id=str(uuid.uuid4()),
        description=f"Demographic parity difference detected: {dp_diff:.4f}",
        severity=dp_severity,
        created_at=datetime.now().isoformat(),
        associated_metric_id=bias_id,
        metric_type="BiasMetric",
    )

    fairness_id = str(uuid.uuid4())
    session.execute_write(add_fairness_metrics, model_id, tpr_diff, fpr_diff, fairness_id)
    eo_severity = "High" if tpr_diff > 0.1 or fpr_diff > 0.1 else "Medium" if tpr_diff > 0.05 or fpr_diff > 0.05 else "Low"
    session.execute_write(
        create_risk_node,
        risk_id=str(uuid.uuid4()),
        description=f"Equalized odds difference detected: TPR diff={tpr_diff:.4f}, FPR diff={fpr_diff:.4f}",
        severity=eo_severity,
        created_at=datetime.now().isoformat(),
        associated_metric_id=fairness_id,
        metric_type="FairnessMetric",
    )

driver.close()
logging.info("Bias and fairness metrics successfully added to Neo4j.")
