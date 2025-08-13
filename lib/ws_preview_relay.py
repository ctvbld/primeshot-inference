from __future__ import annotations

import os
import time
import json as _json
import asyncio
import threading
import websockets


def start_relay(progress_ws_url: str, comfy_ws_url: str, job_id: str, throttle_sec: float = 0.3) -> None:
    """Start a background relay from ComfyUI WS to our progress WS."""

    async def _run():
        try:
            async with websockets.connect(comfy_ws_url, ping_interval=None) as comfy_ws, \
                       websockets.connect(progress_ws_url, ping_interval=None) as out_ws:
                last_preview_ts = 0.0
                while True:
                    msg = await comfy_ws.recv()
                    try:
                        data = _json.loads(msg)
                    except Exception:
                        continue
                    evt = {}
                    if isinstance(data, dict):
                        if "progress" in data:
                            evt["progress"] = data.get("progress")
                        # Preview in 'image' or 'preview'
                        preview = data.get("image") or data.get("preview")
                        if isinstance(preview, str):
                            now = time.time()
                            if now - last_preview_ts > throttle_sec:
                                evt["preview_base64"] = preview
                                last_preview_ts = now
                    if evt:
                        evt.update({"type": "inference_progress", "job_id": job_id})
                        await out_ws.send(_json.dumps(evt))
        except Exception as e:
            print(f"⚠️ WS relay ended: {e}")

    threading.Thread(target=lambda: asyncio.run(_run()), daemon=True).start()


