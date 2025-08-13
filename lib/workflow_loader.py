from __future__ import annotations

import os
import json
import boto3
import urllib.request
from typing import Any, Dict


def load_workflow_from_s3(s3_key: str, bucket: str | None = None) -> Dict[str, Any]:
    """Download workflow JSON from S3 and return as dict."""
    bucket = bucket or os.environ.get("AWS_BUCKET")
    if not bucket:
        raise RuntimeError("Missing AWS_BUCKET env for S3 workflow load")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    data = obj["Body"].read()
    return json.loads(data)


def load_workflow_from_url(url: str) -> Dict[str, Any]:
    """Fetch workflow JSON from HTTPS URL (presigned) and return as dict."""
    with urllib.request.urlopen(url, timeout=20) as resp:
        body = resp.read()
        return json.loads(body)


