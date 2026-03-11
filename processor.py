#!/usr/bin/env python3
import json
import io
import boto3
import pandas as pd
import logging
from datetime import datetime

from baseline import BaselineManager
from detector import AnomalyDetector

s3 = boto3.client("s3")
logger = logging.getLogger(__name__)

NUMERIC_COLS = ["temperature", "humidity", "pressure", "wind_speed"]  # students configure this

def process_file(bucket: str, key: str):
    try:
        logger.info(f"Processing started for s3://{bucket}/{key}")

        # Download raw file
        response = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(response["Body"].read()))
        logger.info(f"Loaded {len(df)} rows from s3://{bucket}/{key}")
        logger.info(f"CSV columns found: {list(df.columns)}")

        #  Load current baseline
        baseline_mgr = BaselineManager(bucket=bucket)
        baseline = baseline_mgr.load()
        logger.info("Baseline loaded successfully")

        # Update baseline with values from this batch BEFORE scoring
        for col in NUMERIC_COLS:
            if col in df.columns:
                clean_values = df[col].dropna().tolist()
                if clean_values:
                    logger.info(f"Updating baseline for column '{col}' with {len(clean_values)} values")
                    baseline = baseline_mgr.update(baseline, col, clean_values)
                else:
                    logger.info(f"Column '{col}' exists but has no non-null values")
            else:
                logger.info(f"Column '{col}' not found in input file")

        #  Run detection
        detector = AnomalyDetector(z_threshold=3.0, contamination=0.05)
        scored_df = detector.run(df, NUMERIC_COLS, baseline, method="both")
        logger.info("Anomaly detection completed successfully")

        # Write scored file to processed/ prefix
        output_key = key.replace("raw/", "processed/")
        csv_buffer = io.StringIO()
        scored_df.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv"
        )
        logger.info(f"Processed CSV uploaded to s3://{bucket}/{output_key}")

        #  Save updated baseline back to S3
        baseline_mgr.save(baseline)
        logger.info("Updated baseline saved to S3")

        # Build and return a processing summary
        anomaly_count = int(scored_df["anomaly"].sum()) if "anomaly" in scored_df else 0
        summary = {
            "source_key": key,
            "output_key": output_key,
            "processed_at": datetime.utcnow().isoformat(),
            "total_rows": len(df),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(df), 4) if len(df) > 0 else 0,
            "baseline_observation_counts": {
                col: baseline.get(col, {}).get("count", 0) for col in NUMERIC_COLS
            }
        }

        # Write summary JSON alongside the processed file
        summary_key = output_key.replace(".csv", "_summary.json")
        s3.put_object(
            Bucket=bucket,
            Key=summary_key,
            Body=json.dumps(summary, indent=2),
            ContentType="application/json"
        )
        logger.info(f"Summary JSON uploaded to s3://{bucket}/{summary_key}")

        logger.info(f"Processing complete: {anomaly_count}/{len(df)} rows flagged as anomalies")
        return summary

    except Exception as e:
        logger.exception(f"Processing failed for s3://{bucket}/{key}: {e}")
        raise

    