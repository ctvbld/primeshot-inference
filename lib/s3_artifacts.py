from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def find_s3_entries(data: Any) -> List[Tuple[str, str]]:
    """Return list of (bucket, key) pairs discovered in an arbitrary payload.
    Accepts formats like s3://bucket/key or separate bucket/key fields.
    """
    found: List[Tuple[str, str]] = []
    s3_url_re = re.compile(r"^s3://([^/]+)/(.+)$")

    if isinstance(data, dict):
        bucket = data.get("bucket")
        key = data.get("key") or data.get("s3_key") or data.get("path")
        url = data.get("url") or data.get("s3")
        if isinstance(url, str):
            m = s3_url_re.match(url)
            if m:
                found.append((m.group(1), m.group(2)))
        if bucket and key:
            found.append((str(bucket), str(key)))
        for v in data.values():
            found.extend(find_s3_entries(v))
    elif isinstance(data, list):
        for v in data:
            found.extend(find_s3_entries(v))
    elif isinstance(data, str):
        m = s3_url_re.match(data)
        if m:
            found.append((m.group(1), m.group(2)))
    return found


def partition_artifacts(pairs: List[Tuple[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    web, orig = [], []
    for bucket, key in pairs:
        low = key.lower()
        entry = {"bucket": bucket, "key": key}
        if "/web/" in low or low.startswith("web_"):
            web.append(entry)
        elif "/orig/" in low or low.startswith("orig_"):
            orig.append(entry)
    return {"web": web, "orig": orig}


