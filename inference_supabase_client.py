"""
Supabase client for ComfyUI inference job progress tracking
"""
import os
import logging
import time
from typing import Optional, Dict, Any, List
from supabase import create_client, Client
from datetime import datetime

logger = logging.getLogger(__name__)

def get_timestamp() -> str:
    """Get properly formatted timestamp for PostgreSQL"""
    now = time.time()
    return time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(now)) + f".{int((now % 1) * 1000000):06d}Z"

class InferenceSupabaseClient:
    """Handles database operations for inference job progress tracking"""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables are required")
        
        # Validate URL format
        if not self.supabase_url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid SUPABASE_URL format: {self.supabase_url}")
        
        try:
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
        except Exception as e:
            logger.error(f"Failed to create Supabase client: {e}")
            logger.error(f"URL format check: {self.supabase_url[:20]}...")
            raise
    
    def create_inference_job(self, user_id: str, workflow_name: str, parameters: Dict[str, Any]) -> str:
        """
        Create a new inference job record
        
        Args:
            user_id: UUID of the user
            workflow_name: Name of the ComfyUI workflow
            parameters: Workflow parameters
            
        Returns:
            job_id: UUID of the created inference job
        """
        try:
            result = self.client.table('inference_jobs').insert({
                'user_id': user_id,
                'workflow_name': workflow_name,
                'parameters': parameters,
                'status': 'queued',
                'progress': 0,
                'started_at': None,
                'completed_at': None
            }).execute()
            
            job_id = result.data[0]['id']
            logger.info(f"Created inference job: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to create inference job: {e}")
            raise
    
    def start_inference_job(self, job_id: str) -> None:
        """
        Mark inference job as started
        
        Args:
            job_id: UUID of the inference job
        """
        try:
            # Check if job is already started to prevent duplicate updates
            current_job = self.client.table('inference_jobs').select('status, started_at').eq('id', job_id).single().execute()
            if current_job.data and current_job.data.get('status') not in ['queued']:
                logger.info(f"Skipping start update for job {job_id} - already {current_job.data.get('status')}")
                return
            
            update_data = {
                'status': 'processing',
                'started_at': get_timestamp(),
                'updated_at': get_timestamp()
            }
            
            self.client.table('inference_jobs').update(update_data).eq('id', job_id).execute()
            logger.info(f"Started inference job {job_id}")
            
        except Exception as e:
            logger.error(f"Failed to start inference job {job_id}: {e}")
            # Don't raise - we don't want progress tracking to break inference

    def update_inference_status(self, job_id: str, status: str, progress: Optional[int] = None) -> None:
        """
        Update inference job status and progress
        
        Args:
            job_id: UUID of the inference job
            status: New inference status ('processing', 'completed', 'failed')
            progress: Optional progress percentage (0-100)
        """
        try:
            # Check if job is already completed/failed to prevent race conditions
            current_job = self.client.table('inference_jobs').select('status').eq('id', job_id).single().execute()
            if current_job.data and current_job.data.get('status') in ['completed', 'failed']:
                logger.info(f"Skipping status update for job {job_id} - already {current_job.data.get('status')}")
                return
            
            update_data = {
                'status': status,
                'updated_at': get_timestamp()
            }
            
            if progress is not None:
                update_data['progress'] = progress
            
            # Set started_at when status changes to processing (if not already set)
            if status == 'processing' and not current_job.data.get('started_at'):
                update_data['started_at'] = get_timestamp()
            
            self.client.table('inference_jobs').update(update_data).eq('id', job_id).execute()
            logger.info(f"Updated inference job {job_id} status to: {status}" + (f" ({progress}%)" if progress else ""))
            
        except Exception as e:
            logger.error(f"Failed to update inference job status {job_id}: {e}")
            # Don't raise - we don't want status tracking to break inference

    def complete_inference_job(
        self, 
        job_id: str, 
        success: bool, 
        error_message: Optional[str] = None,
        output_urls: Optional[list] = None
    ) -> None:
        """
        Mark inference job as completed
        
        Args:
            job_id: UUID of the inference job
            success: Whether inference completed successfully
            error_message: Error message if inference failed
            output_urls: List of generated image URLs
        """
        try:
            update_data = {
                'status': 'completed' if success else 'failed',
                'progress': 100 if success else 0,
                'completed_at': get_timestamp(),
                'updated_at': get_timestamp()
            }
            
            if error_message:
                update_data['error_message'] = error_message
            
            if output_urls:
                update_data['output_urls'] = output_urls
                update_data['images_generated'] = len(output_urls)
            
            self.client.table('inference_jobs').update(update_data).eq('id', job_id).execute()
            
            status_text = "completed successfully" if success else f"failed: {error_message}"
            logger.info(f"Inference job {job_id} {status_text}")
            
        except Exception as e:
            logger.error(f"Failed to complete inference job {job_id}: {e}")
    
    def get_inference_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get inference job status and details
        
        Args:
            job_id: UUID of the inference job
            
        Returns:
            Job details dict or None if not found
        """
        try:
            result = self.client.table('inference_jobs').select('*').eq('id', job_id).single().execute()
            return result.data if result.data else None
            
        except Exception as e:
            logger.error(f"Failed to get inference job status {job_id}: {e}")
            return None
    
    def get_user_inference_jobs(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent inference jobs for a user
        
        Args:
            user_id: UUID of the user
            limit: Maximum number of jobs to return
            
        Returns:
            List of job details
        """
        try:
            result = (
                self.client.table('inference_jobs')
                .select('*')
                .eq('user_id', user_id)
                .order('created_at', desc=True)
                .limit(limit)
                .execute()
            )
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Failed to get user inference jobs for {user_id}: {e}")
            return []

# Global instance
_inference_supabase_client: Optional[InferenceSupabaseClient] = None

def get_inference_supabase_client() -> InferenceSupabaseClient:
    """Get or create Inference Supabase client instance"""
    global _inference_supabase_client
    if _inference_supabase_client is None:
        _inference_supabase_client = InferenceSupabaseClient()
    return _inference_supabase_client