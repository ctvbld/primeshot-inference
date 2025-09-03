from __future__ import annotations

import os
import json
import urllib.request
from typing import Any, Dict


def fetch_inference_prep(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call Supabase Edge Function inference-prepare and return JSON."""
    supabase_url = os.environ.get('SUPABASE_URL')
    service_role_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
    if not supabase_url or not service_role_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
    url = f"{supabase_url}/functions/v1/inference-prepare"
    webhook_secret = os.environ.get('WEBHOOK_SECRET') or os.environ.get('INFERENCE_WEBHOOK_SECRET')
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {webhook_secret or service_role_key}',
            'apikey': service_role_key,
        },
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())


