#!/usr/bin/env python3
import json
import math
import boto3
import logging
import os
from datetime import datetime
from typing import Optional

s3 = boto3.client("s3")
logger = logging.getLogger(__name__)


class BaselineManager:
    """
    Maintains a per-channel running baseline using Welford's online algorithm,
    which computes mean and variance incrementally without storing all past data.
    """

    def __init__(
        self,
        bucket: str,
        baseline_key: str = "state/baseline.json",
        log_file: str = "app.log",
        log_s3_key: str = "logs/app.log"
    ):
        self.bucket = bucket
        self.baseline_key = baseline_key
        self.log_file = log_file
        self.log_s3_key = log_s3_key

    def load(self) -> dict:
        try:
            logger.info(f"Loading baseline from s3://{self.bucket}/{self.baseline_key}")
            response = s3.get_object(Bucket=self.bucket, Key=self.baseline_key)
            baseline = json.loads(response["Body"].read())
            logger.info("Baseline loaded successfully")
            return baseline

        except Exception as e:
            error_code = getattr(e, "response", {}).get("Error", {}).get("Code")
            if error_code in ("NoSuchKey", "404"):
                logger.warning("No existing baseline found in S3; starting with empty baseline")
                return {}

            logger.exception(f"Failed to load baseline: {e}")
            raise

    def save(self, baseline: dict):
        try:
            baseline["last_updated"] = datetime.utcnow().isoformat()

            s3.put_object(
                Bucket=self.bucket,
                Key=self.baseline_key,
                Body=json.dumps(baseline, indent=2),
                ContentType="application/json"
            )
            logger.info(f"Baseline saved to s3://{self.bucket}/{self.baseline_key}")

            # Upload the local log file to S3 as backup
            if os.path.exists(self.log_file):
                with open(self.log_file, "rb") as f:
                    s3.put_object(
                        Bucket=self.bucket,
                        Key=self.log_s3_key,
                        Body=f,
                        ContentType="text/plain"
                    )
                logger.info(f"Log file uploaded to s3://{self.bucket}/{self.log_s3_key}")
            else:
                logger.warning(f"Log file '{self.log_file}' not found; skipping log upload")

        except Exception as e:
            logger.exception(f"Failed to save baseline or upload log file: {e}")
            raise

    def update(self, baseline: dict, channel: str, new_values: list[float]) -> dict:
        """
        Welford's online algorithm for numerically stable mean and variance.
        Each channel tracks: count, mean, M2 (sum of squared deviations).
        Variance = M2 / count, std = sqrt(variance).
        """
        try:
            logger.info(f"Updating baseline for channel '{channel}' with {len(new_values)} new value(s)")

            if channel not in baseline:
                baseline[channel] = {"count": 0, "mean": 0.0, "M2": 0.0}
                logger.info(f"Initialized new baseline channel '{channel}'")

            state = baseline[channel]

            for value in new_values:
                state["count"] += 1
                delta = value - state["mean"]
                state["mean"] += delta / state["count"]
                delta2 = value - state["mean"]
                state["M2"] += delta * delta2

            # Only compute std once we have enough observations
            if state["count"] >= 2:
                variance = state["M2"] / state["count"]
                state["std"] = math.sqrt(variance)
            else:
                state["std"] = 0.0

            baseline[channel] = state

            logger.info(
                f"Updated channel '{channel}': "
                f"count={state['count']}, mean={state['mean']:.4f}, std={state['std']:.4f}"
            )

            return baseline

        except Exception as e:
            logger.exception(f"Failed to update baseline for channel '{channel}': {e}")
            raise

    def get_stats(self, baseline: dict, channel: str) -> Optional[dict]:
        try:
            logger.info(f"Fetching stats for channel '{channel}'")
            return baseline.get(channel)
        except Exception as e:
            logger.exception(f"Failed to get stats for channel '{channel}': {e}")
            raise
        