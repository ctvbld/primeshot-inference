from __future__ import annotations

import os
import json as _json
from typing import Any, Dict

from .s3_artifacts import find_s3_entries, partition_artifacts


def handle_webhook(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process comfyui-api webhook payload and call inference-complete EF.

    Returns a structured response used as HTTP body by the endpoint.
    """
    try:
        user_id = payload.get("user_id")
        job_id = payload.get("job_id") or payload.get("inference_id")

        pairs = find_s3_entries(payload)
        artifacts = partition_artifacts(pairs)

        # Optional: include basic metadata if present in payload
        # (full metadata extraction via HEAD was moved to EF for consistency)

        # Call inference-complete Edge Function to persist and trigger queue
        try:
            import requests
            supabase_url = os.environ.get('SUPABASE_URL')
            service_role_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
            if supabase_url and service_role_key and job_id:
                url = f"{supabase_url}/functions/v1/inference-complete"
                headers = {
                    'Authorization': f'Bearer {service_role_key}',
                    'Content-Type': 'application/json',
                    'apikey': service_role_key,
                }
                body = {
                    'job_id': job_id,
                    'success': True,
                    'artifacts': artifacts,
                }
                try:
                    resp = requests.post(url, json=body, headers=headers, timeout=20)
                    if not resp.ok:
                        print(f"⚠️ inference-complete EF error: {resp.status_code} {resp.text}")
                except Exception as ef_e:
                    print(f"⚠️ inference-complete EF call failed: {ef_e}")
            else:
                print("⚠️ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY or job_id; skipping EF call")
        except Exception as _ef:
            print(f"⚠️ EF block error: {_ef}")

        return {
            "status": "ok",
            "user_id": user_id,
            "job_id": job_id,
            "artifacts": artifacts,
            "received": payload,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "received": payload}


