from __future__ import annotations

import os
import json
import boto3
from pathlib import Path
import urllib.request
from typing import Any, Dict


def load_workflow_from_s3(s3_key: str, bucket: str | None = None) -> Dict[str, Any]:
    """Load workflow JSON, preferring mounted /workflows path; fallback to S3.

    Accepts keys with or without the "workflows/" prefix. When using the mount,
    
    /workflows maps to s3://<bucket>/workflows/.
    """
    key = s3_key.lstrip("/")
    # Normalize local relative path under /workflows
    local_rel = key[len("workflows/"):] if key.startswith("workflows/") else key
    local_path = Path("/workflows") / local_rel
    if local_path.exists():
        try:
            return json.loads(local_path.read_text())
        except Exception:
            pass
    bucket = bucket or os.environ.get("AWS_BUCKET")
    if not bucket:
        raise RuntimeError("Missing AWS_BUCKET env for S3 workflow load")
    s3 = boto3.client("s3")
    # Try key as-provided; then try with workflows/ prefix if missing
    candidates = [key]
    if not key.startswith("workflows/"):
        candidates.append(f"workflows/{key}")
    last_err: Exception | None = None
    for k in candidates:
        try:
            obj = s3.get_object(Bucket=bucket, Key=k)
            data = obj["Body"].read()
            return json.loads(data)
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("Workflow load failed")


def load_workflow_from_url(url: str) -> Dict[str, Any]:
    """Fetch workflow JSON from HTTPS URL (presigned) and return as dict."""
    with urllib.request.urlopen(url, timeout=20) as resp:
        body = resp.read()
        return json.loads(body)


