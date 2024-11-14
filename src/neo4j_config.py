import os
from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

if not uri or not user or not password:
    raise ValueError("Missing one or more required environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(user, password))

def get_driver():
    """
    Returns the Neo4j driver instance.
    """
    return driver
