import pandas as pd
from ucimlrepo import fetch_ucirepo
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fetch_and_preprocess_dataset():
    """
    Fetch dataset from UCI Machine Learning Repository and preprocess.
    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target
    """
    logging.info("Fetching dataset...")
    dataset = fetch_ucirepo(id=45) 
    X = dataset.data.features
    y = dataset.data.targets

    logging.info(f"Features type: {type(X)}, Target type: {type(y)}")
    logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")


    if isinstance(y, pd.DataFrame) or len(y.shape) > 1:
        logging.warning("Converting target `y` DataFrame to Pandas Series...")
        y = pd.Series(y.squeeze(), name="target")


    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    logging.info("Dataset preprocessing completed.")
    return X, y
