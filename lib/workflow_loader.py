from __future__ import annotations

import json
from pathlib import Path
import urllib.request
from typing import Any, Dict


def load_workflow_from_s3(s3_key: str, bucket: str | None = None) -> Dict[str, Any]:
    """Load workflow JSON from mounted /workflows path.

    Accepts keys with or without the "workflows/" prefix.
    /workflows maps to s3://<bucket>/workflows/ via Modal mount.
    """
    key = s3_key.lstrip("/")
    # Normalize local relative path under /workflows
    local_rel = key[len("workflows/"):] if key.startswith("workflows/") else key
    local_path = Path("/workflows") / local_rel

    # Primary: mounted S3 (/workflows)
    if local_path.exists():
        try:
            wf = json.loads(local_path.read_text())
            print(f"✅ Loaded workflow from S3 mount: {local_path}")
            return wf
        except Exception as e:
            raise RuntimeError(f"Failed to load workflow from {local_path}: {e}")

    # Fallback: baked-in image directory (/root/workflows) for local/dev
    baked_path = Path("/root/workflows") / local_rel
    if baked_path.exists():
        try:
            wf = json.loads(baked_path.read_text())
            print(f"✅ Loaded workflow from baked path: {baked_path}")
            return wf
        except Exception as e:
            raise RuntimeError(f"Failed to load workflow from {baked_path}: {e}")

    # If file doesn't exist anywhere, raise a clear error
    raise FileNotFoundError(f"Workflow not found in /workflows or /root/workflows (key: {s3_key})")


def load_workflow_from_url(url: str) -> Dict[str, Any]:
    """Fetch workflow JSON from HTTPS URL (presigned) and return as dict."""
    with urllib.request.urlopen(url, timeout=20) as resp:
        body = resp.read()
        return json.loads(body)


