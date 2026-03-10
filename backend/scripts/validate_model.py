#!/usr/bin/env python
"""
Model validation script for CI/CD.
Validates that model metrics meet defined thresholds before promotion.
"""

import argparse
import json
import os
import sys

import mlflow


def validate_model(run_id: str, thresholds: dict) -> bool:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    if run.status != "FINISHED":
        print(f"Run {run_id} is not finished. Status: {run.status}")
        return False

    metrics = run.data.metrics

    print(f"Validating model run {run_id}")
    print(f"Metrics: {metrics}")
    print(f"Thresholds: {thresholds}")

    all_passed = True
    for metric_name, threshold in thresholds.items():
        if metric_name not in metrics:
            print(f"ERROR: Metric '{metric_name}' not found in run metrics")
            all_passed = False
            continue

        value = metrics[metric_name]
        passed = value >= threshold
        status = "PASS" if passed else "FAIL"
        print(f"  {metric_name}: {value:.4f} >= {threshold:.4f} [{status}]")

        if not passed:
            all_passed = False

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate ML model metrics")
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument(
        "--thresholds",
        required=True,
        help="JSON string of metric thresholds (e.g. '{\"accuracy\": 0.85}')",
    )
    args = parser.parse_args()

    thresholds = json.loads(args.thresholds)
    passed = validate_model(args.run_id, thresholds)

    if passed:
        print("\n✓ Model validation PASSED")
        sys.exit(0)
    else:
        print("\n✗ Model validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
