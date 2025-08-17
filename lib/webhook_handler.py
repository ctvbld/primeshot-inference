from __future__ import annotations

import os
import json as _json
from typing import Any, Dict

from .s3_artifacts import find_s3_entries, partition_artifacts



def _create_web_variants(pairs: list[tuple[str, str]], web_prefix: str, job_id: str) -> list[tuple[str, str]]:
    """Create multi-resolution web variants from original images and upload to S3.
    
    Creates multiple WebP variants at different resolutions:
    - Base: 1024px (filename.webp) 
    - Variants: 640px, 320px (filename-w640.webp, filename-w320.webp)
    
    Args:
        pairs: List of (s3_key, local_path) tuples from original uploads
        web_prefix: S3 prefix for web variants (e.g., "user-images/user123/inference/job456/web/")
        job_id: Job ID for logging
    
    Returns:
        Extended list of pairs including both original and all web variants
    """
    try:
        import boto3
        from PIL import Image
        import io
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        bucket = os.environ.get('AWS_BUCKET')
        
        if not bucket:
            print(f"⚠️ AWS_BUCKET not set, skipping web variant creation for job {job_id}")
            return pairs
            
        # Multi-resolution configuration (simplified: base + smaller variants)
        variant_widths = [320, 640, 1024]  # 1024 is the base size
        base_width = 1024
        
        extended_pairs = list(pairs)  # Start with original pairs
        
        for orig_s3_key, orig_local_path in pairs:
            try:
                print(f"🔄 Creating multi-resolution web variants for {orig_s3_key}")
                
                # Download original image from S3
                response = s3_client.get_object(Bucket=bucket, Key=orig_s3_key)
                orig_data = response['Body'].read()
                
                # Generate base filename (replace extension with .webp and clean trailing underscore)
                orig_filename = orig_s3_key.split('/')[-1]
                base_name = orig_filename.rsplit('.', 1)[0]
                # Remove trailing underscore from ComfyUI naming (e.g., "job123_00001_" -> "job123_00001")
                if base_name.endswith('_'):
                    base_name = base_name[:-1]
                base_filename = base_name + '.webp'
                
                # Open original image once
                with Image.open(io.BytesIO(orig_data)) as orig_img:
                    # Convert to RGB if necessary (for WebP compatibility)
                    if orig_img.mode in ('RGBA', 'LA', 'P'):
                        orig_img = orig_img.convert('RGB')
                    
                    # Create all variants
                    for width in variant_widths:
                        try:
                            # Create a copy for this variant
                            img = orig_img.copy()
                            
                            # Resize maintaining aspect ratio
                            img.thumbnail((width, width), Image.Resampling.LANCZOS)
                            
                            # Quality based on size (higher quality for larger sizes)
                            if width >= 1024:
                                quality = 90
                            elif width >= 640:
                                quality = 85
                            else:
                                quality = 80
                            
                            # Save as WebP
                            web_buffer = io.BytesIO()
                            img.save(web_buffer, format='WEBP', quality=quality, optimize=True)
                            web_data = web_buffer.getvalue()
                            
                            # Generate filename: base for 1024px, variants with -w{width} suffix
                            if width == base_width:
                                web_filename = base_filename
                            else:
                                web_filename = base_filename.rsplit('.', 1)[0] + f'-w{width}.webp'
                            
                            web_s3_key = web_prefix + web_filename
                            
                            # Upload variant to S3
                            s3_client.put_object(
                                Bucket=bucket,
                                Key=web_s3_key,
                                Body=web_data,
                                ContentType='image/webp'
                            )
                            
                            # Add to pairs list (use orig_local_path as placeholder)
                            extended_pairs.append((web_s3_key, orig_local_path))
                            
                            print(f"✅ Created web variant ({width}px): {web_s3_key}")
                            
                        except Exception as ve:
                            print(f"⚠️ Failed to create {width}px variant: {ve}")
                            # Continue with other sizes even if one fails
                            continue
                
            except Exception as e:
                print(f"⚠️ Failed to create web variants for {orig_s3_key}: {e}")
                # Continue with other images even if one fails
                continue
        
        return extended_pairs
        
    except Exception as e:
        print(f"⚠️ Error in web variant creation for job {job_id}: {e}")
        return pairs  # Return original pairs if web creation fails


def handle_webhook(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process comfyui-api webhook payload and call inference-complete EF.
    
    Handles dual output creation (web + orig variants) when dual_output flag is present.

    Returns a structured response used as HTTP body by the endpoint.
    """
    try:
        # Debug: log the full payload structure to understand what we're receiving
        try:
            # Create a copy for logging that truncates large base64 image data
            payload_for_logging = payload.copy()
            if 'image' in payload_for_logging and isinstance(payload_for_logging['image'], str):
                img_data = payload_for_logging['image']
                if len(img_data) > 100:  # If it's a large string (likely base64 image)
                    payload_for_logging['image'] = f"{img_data[:50]}...[{len(img_data)} chars total]...{img_data[-50:]}"
            
            print(f"🔍 Webhook received payload: {_json.dumps(payload_for_logging, indent=2)}")
            
            # Also print payload keys and sizes for debugging
            print(f"🔍 Payload structure:")
            for key, value in payload.items():
                if isinstance(value, str):
                    print(f"  - {key}: string ({len(value)} chars)")
                elif isinstance(value, dict):
                    print(f"  - {key}: dict ({len(value)} keys)")
                elif isinstance(value, list):
                    print(f"  - {key}: list ({len(value)} items)")
                else:
                    print(f"  - {key}: {type(value).__name__} = {value}")
                    
        except Exception as log_e:
            print(f"🔍 Webhook received payload (raw): {payload}")
            print(f"⚠️ Logging error: {log_e}")
        
        # Extract job_id - the Edge Function will get user_id from database
        job_id = payload.get("job_id") or payload.get("inference_id") or payload.get("id")
        
        # If no job_id in payload, try to extract from S3 paths
        if not job_id:
            pairs = find_s3_entries(payload)
            if pairs:
                # Extract from S3 path pattern: user-images/{user_id}/inference/{job_id}/
                first_s3_key = pairs[0][0] if pairs else ""
                import re
                s3_match = re.search(r'user-images/([^/]+)/inference/([^/]+)/', first_s3_key)
                if s3_match:
                    job_id = s3_match.group(2)
                    print(f"🔍 Extracted job_id from S3 path: {job_id}")
        
        print(f"🔍 Processing webhook for job_id: {job_id}")

        pairs = find_s3_entries(payload)
        
        # Auto-detect if images are in /orig/ directory and create web variants
        orig_pairs = [pair for pair in pairs if '/orig/' in pair[0]]
        if orig_pairs:
            # Create web variants for all orig images
            job_id_for_logging = job_id or "unknown"
            # Derive web prefix from orig prefix
            first_orig_key = orig_pairs[0][0]
            web_prefix = first_orig_key.replace('/orig/', '/web/').rsplit('/', 1)[0] + '/'
            pairs = _create_web_variants(pairs, web_prefix, job_id_for_logging)
        
        artifacts = partition_artifacts(pairs)

        # Optional: include basic metadata if present in payload
        # (full metadata extraction via HEAD was moved to EF for consistency)

        # Call inference-complete Edge Function to persist and trigger queue
        try:
            import requests
            
            # Get environment from job metadata to use correct Supabase instance
            env_tag = "dev"  # default
            if job_id:
                try:
                    from job_tracker import get_job_tracker
                    tracker = get_job_tracker()
                    if hasattr(tracker, '_job_metadata') and job_id in tracker._job_metadata:
                        metadata = tracker._job_metadata[job_id]
                        env_tag = metadata.get("env", "dev")
                        print(f"🔍 Using environment: {env_tag}")
                except Exception as e:
                    print(f"⚠️ Failed to get env from metadata, using default 'dev': {e}")
            
            # Use environment-specific Supabase credentials
            supabase_url = os.environ.get(f'SUPABASE_URL_{env_tag.upper()}') or os.environ.get('SUPABASE_URL')
            service_role_key = os.environ.get(f'SUPABASE_SERVICE_ROLE_KEY_{env_tag.upper()}') or os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
            
            print(f"🔍 Supabase URL ({env_tag}): {supabase_url[:50] + '...' if supabase_url and len(supabase_url) > 50 else supabase_url}")
            print(f"🔍 Service key ({env_tag}): {'✅ Set' if service_role_key else '❌ Missing'}")
            
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
                print(f"⚠️ Missing Supabase credentials for {env_tag} environment or job_id; skipping EF call")
                print(f"  - supabase_url: {'✅' if supabase_url else '❌'}")
                print(f"  - service_role_key: {'✅' if service_role_key else '❌'}")
                print(f"  - job_id: {'✅' if job_id else '❌'}")
        except Exception as _ef:
            print(f"⚠️ EF block error: {_ef}")

        return {
            "status": "ok", 
            "job_id": job_id,
            "artifacts": artifacts,
            "received": payload,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "received": payload}


