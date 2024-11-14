import uuid
from datetime import datetime

def create_risk_node(tx, risk_id, description, severity, created_at, associated_metric_id=None, metric_type=None):
    """
    Create a Risk node and optionally link it to an associated metric in Neo4j.
    """
    tx.run(
        """
        CREATE (r:Risk {
            risk_id: $risk_id,
            description: $description,
            severity: $severity,
            created_at: $created_at
        })
        """,
        risk_id=risk_id,
        description=description,
        severity=severity,
        created_at=created_at,
    )

    if associated_metric_id and metric_type:
        if metric_type == "DriftMetric":
            tx.run(
                """
                MATCH (m:DriftMetric {feature: $associated_metric_id})
                MATCH (r:Risk {risk_id: $risk_id})
                MERGE (m)-[:POSES_RISK]->(r)
                """,
                associated_metric_id=associated_metric_id,
                risk_id=risk_id,
            )
        elif metric_type == "BiasMetric":
            tx.run(
                """
                MATCH (m:BiasMetric {bias_id: $associated_metric_id})
                MATCH (r:Risk {risk_id: $risk_id})
                MERGE (m)-[:POSES_RISK]->(r)
                """,
                associated_metric_id=associated_metric_id,
                risk_id=risk_id,
            )
        elif metric_type == "FairnessMetric":
            tx.run(
                """
                MATCH (m:FairnessMetric {fairness_id: $associated_metric_id})
                MATCH (r:Risk {risk_id: $risk_id})
                MERGE (m)-[:POSES_RISK]->(r)
                """,
                associated_metric_id=associated_metric_id,
                risk_id=risk_id,
            )
