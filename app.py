# app.py
import io
import json
import os
import boto3
import pandas as pd
import requests
import logging
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from baseline import BaselineManager
from processor import process_file

app = FastAPI(title="Anomaly Detection Pipeline")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("FastAPI anomaly detection service starting up")

s3 = boto3.client("s3")
BUCKET_NAME = os.environ["BUCKET_NAME"]

# ── SNS subscription confirmation + message handler ──────────────────────────

@app.post("/notify")
async def handle_sns(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
        msg_type = request.headers.get("x-amz-sns-message-type")

        logger.info(f"Received /notify request with SNS message type: {msg_type}")

        # SNS sends a SubscriptionConfirmation before it will deliver any messages.
        if msg_type == "SubscriptionConfirmation":
            confirm_url = body.get("SubscribeURL")
            if not confirm_url:
                logger.error("SubscriptionConfirmation received without SubscribeURL")
                raise HTTPException(status_code=400, detail="Missing SubscribeURL")

            logger.info(f"Confirming SNS subscription at URL: {confirm_url}")
            response = requests.get(confirm_url, timeout=10)
            response.raise_for_status()

            logger.info("SNS subscription confirmed successfully")
            return {"status": "confirmed"}

        if msg_type == "Notification":
            message = body.get("Message")
            if not message:
                logger.error("Notification received without Message field")
                raise HTTPException(status_code=400, detail="Missing SNS Message")

            s3_event = json.loads(message)
            records = s3_event.get("Records", [])
            logger.info(f"Notification contains {len(records)} record(s)")

            for record in records:
                try:
                    key = record["s3"]["object"]["key"]
                    logger.info(f"New file detected: {key}")

                    if key.startswith("raw/") and key.endswith(".csv"):
                        logger.info(f"Queueing background processing for file: {key}")
                        background_tasks.add_task(process_file, BUCKET_NAME, key)
                    else:
                        logger.info(f"Skipping non-matching file: {key}")

                except Exception as record_error:
                    logger.exception(f"Failed to process SNS record: {record_error}")

        return {"status": "ok"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error handling SNS notification: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ── Query endpoints ───────────────────────────────────────────────────────────

@app.get("/anomalies/recent")
def get_recent_anomalies(limit: int = 50):
    """Return rows flagged as anomalies across the 10 most recent processed files."""
    try:
        logger.info(f"/anomalies/recent called with limit={limit}")

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")

        keys = sorted(
            [
                obj["Key"]
                for page in pages
                for obj in page.get("Contents", [])
                if obj["Key"].endswith(".csv")
            ],
            reverse=True,
        )[:10]

        logger.info(f"Found {len(keys)} recent processed CSV file(s)")

        all_anomalies = []
        for key in keys:
            response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            df = pd.read_csv(io.BytesIO(response["Body"].read()))
            if "anomaly" in df.columns:
                flagged = df[df["anomaly"] == True].copy()
                flagged["source_file"] = key
                all_anomalies.append(flagged)

        if not all_anomalies:
            logger.info("No anomalies found in recent processed files")
            return {"count": 0, "anomalies": []}

        combined = pd.concat(all_anomalies).head(limit)
        logger.info(f"Returning {len(combined)} anomaly row(s)")
        return {"count": len(combined), "anomalies": combined.to_dict(orient="records")}

    except Exception as e:
        logger.exception(f"Error in /anomalies/recent: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve recent anomalies")



@app.get("/anomalies/summary")
def get_anomaly_summary():
    """Aggregate anomaly rates across all processed files using their summary JSONs."""
    try:
        logger.info("/anomalies/summary called")

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")

        summaries = []
        for page in pages:
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("_summary.json"):
                    response = s3.get_object(Bucket=BUCKET_NAME, Key=obj["Key"])
                    summaries.append(json.loads(response["Body"].read()))

        if not summaries:
            logger.info("No processed summary files found yet")
            return {"message": "No processed files yet."}

        total_rows = sum(s["total_rows"] for s in summaries)
        total_anomalies = sum(s["anomaly_count"] for s in summaries)

        logger.info(
            f"Summary calculated: files={len(summaries)}, rows={total_rows}, anomalies={total_anomalies}"
        )

        return {
            "files_processed": len(summaries),
            "total_rows_scored": total_rows,
            "total_anomalies": total_anomalies,
            "overall_anomaly_rate": round(total_anomalies / total_rows, 4) if total_rows > 0 else 0,
            "most_recent": sorted(summaries, key=lambda x: x["processed_at"], reverse=True)[:5],
        }

    except Exception as e:
        logger.exception(f"Error in /anomalies/summary: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve anomaly summary")


@app.get("/baseline/current")
def get_current_baseline():
    """Show the current per-channel statistics the detector is working from."""
    try:
        logger.info("/baseline/current called")

        baseline_mgr = BaselineManager(bucket=BUCKET_NAME)
        baseline = baseline_mgr.load()

        channels = {}
        for channel, stats in baseline.items():
            if channel == "last_updated":
                continue
            channels[channel] = {
                "observations": stats["count"],
                "mean": round(stats["mean"], 4),
                "std": round(stats.get("std", 0.0), 4),
                "baseline_mature": stats["count"] >= 30,
            }

        logger.info(f"Returning baseline for {len(channels)} channel(s)")

        return {
            "last_updated": baseline.get("last_updated"),
            "channels": channels,
        }

    except Exception as e:
        logger.exception(f"Error in /baseline/current: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve baseline")

@app.get("/health")
def health():
    logger.info("/health called")
    return {
        "status": "ok",
        "bucket": BUCKET_NAME,
        "timestamp": datetime.utcnow().isoformat()
    }
