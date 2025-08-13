from __future__ import annotations

import os
import json as _json
import threading
import urllib.request
from typing import Dict, Any


def submit_to_comfyui_api_async(payload: Dict[str, Any]) -> None:
    """Submit the job to comfyui-api dynamic endpoint asynchronously.

    Env vars used:
    - WEBHOOK_URL: optional webhook to receive progress/completion
    - COMFY_API_BASE: base URL (default http://127.0.0.1:9000)
    - COMFY_WORKFLOW_ENDPOINT: route (default /workflows/image_default/run)
    Route selection:
    - If payload contains 'workflow_s3' or 'workflow_url', send to /workflows/run
    - Otherwise, use COMFY_WORKFLOW_ENDPOINT (defaults to /workflows/image_default/run)
    """

    def _submit() -> None:
        try:
            payload_with_hook = dict(payload)

            # Optional webhook
            webhook_url = os.environ.get("WEBHOOK_URL")
            if webhook_url:
                payload_with_hook["webhook_url"] = webhook_url

            # Base and route
            base_url = os.environ.get("COMFY_API_BASE", "http://127.0.0.1:9000")
            if payload.get("workflow_s3") or payload.get("workflow_url"):
                route = "/workflows/run"
            else:
                route = os.environ.get("COMFY_WORKFLOW_ENDPOINT", "/workflows/image_default/run")
            url = base_url.rstrip("/") + "/" + route.lstrip("/")

            # Submit
            req = urllib.request.Request(
                url=url,
                data=_json.dumps(payload_with_hook).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
            print(f"📨 Submitted job to comfyui-api: {payload_with_hook.get('job_id')}")
        except Exception as e:
            print(f"❌ comfyui-api submission failed: {e}")

    threading.Thread(target=_submit, daemon=True).start()


