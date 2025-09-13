"""Workflow loading utilities for ComfyUI inference.

This module provides functions to load workflow JSON files from various sources
including S3 mounts, local paths, and HTTPS URLs.
"""

from __future__ import annotations

import json
from pathlib import Path
import urllib.request
from typing import Any, Dict, Optional
import logging

from .constants import WORKFLOWS_MOUNT_PATH, WORKFLOWS_BAKED_PATH

logger = logging.getLogger(__name__)


def validate_workflow(workflow: Dict[str, Any]) -> None:
    """Validate that a loaded workflow has the expected structure.
    
    Args:
        workflow: The workflow dictionary to validate
        
    Raises:
        ValueError: If the workflow structure is invalid
    """
    if not isinstance(workflow, dict):
        raise ValueError("Workflow must be a dictionary")
    
    if not workflow:
        raise ValueError("Workflow cannot be empty")
    
    # Check that at least one node exists
    node_count = sum(1 for v in workflow.values() if isinstance(v, dict))
    if node_count == 0:
        raise ValueError("Workflow must contain at least one node")
    
    logger.info(f"✅ Validated workflow with {node_count} nodes")


def load_workflow_from_s3(s3_key: str, bucket: str | None = None) -> Dict[str, Any]:
    """Load workflow JSON from mounted /workflows path.

    Accepts keys with or without the "workflows/" prefix.
    /workflows maps to s3://<bucket>/workflows/ via Modal mount.
    
    Args:
        s3_key: S3 key or local path to workflow file
        bucket: S3 bucket name (unused, kept for compatibility)
        
    Returns:
        Loaded and validated workflow dictionary
        
    Raises:
        FileNotFoundError: If workflow file not found
        RuntimeError: If workflow loading fails
        ValueError: If workflow validation fails
    """
    key = s3_key.lstrip("/")
    # Normalize local relative path under /workflows
    local_rel = key[len("workflows/"):] if key.startswith("workflows/") else key
    local_path = Path(WORKFLOWS_MOUNT_PATH) / local_rel

    # Primary: mounted S3 (/workflows)
    if local_path.exists():
        try:
            wf = json.loads(local_path.read_text())
            logger.info(f"✅ Loaded workflow from S3 mount: {local_path}")
            validate_workflow(wf)
            return wf
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in workflow {local_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load workflow from {local_path}: {e}")

    # Fallback: baked-in image directory (/root/workflows) for local/dev
    baked_path = Path(WORKFLOWS_BAKED_PATH) / local_rel
    if baked_path.exists():
        try:
            wf = json.loads(baked_path.read_text())
            logger.info(f"✅ Loaded workflow from baked path: {baked_path}")
            validate_workflow(wf)
            return wf
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in workflow {baked_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load workflow from {baked_path}: {e}")

    # If file doesn't exist anywhere, raise a clear error
    raise FileNotFoundError(
        f"Workflow not found in {WORKFLOWS_MOUNT_PATH} or {WORKFLOWS_BAKED_PATH} (key: {s3_key})"
    )


def load_workflow_from_url(url: str, timeout: int = 20) -> Dict[str, Any]:
    """Fetch workflow JSON from HTTPS URL (presigned) and return as dict.
    
    Args:
        url: HTTPS URL to fetch workflow from
        timeout: Request timeout in seconds
        
    Returns:
        Loaded and validated workflow dictionary
        
    Raises:
        RuntimeError: If workflow loading fails
        ValueError: If workflow validation fails
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
            wf = json.loads(body)
            logger.info(f"✅ Loaded workflow from URL: {url}")
            validate_workflow(wf)
            return wf
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from URL {url}: {e}")
    except urllib.request.URLError as e:
        raise RuntimeError(f"Failed to fetch workflow from {url}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading workflow from {url}: {e}")


