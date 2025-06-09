#!/usr/bin/env python3
"""
Job tracking system for ComfyUI Photography Platform.
Manages job statuses, progress tracking, and results storage.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class JobStatus(Enum):
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class GenerationJob:
    job_id: str
    status: JobStatus
    request_data: Dict[str, Any]
    created_at: float
    updated_at: float
    progress: float = 0.0
    images_completed: int = 0
    total_images: int = 0
    estimated_remaining: str = ""
    output_urls: list = None
    error_message: str = ""
    execution_time: float = 0.0
    cost_estimate: float = 0.0
    
    def __post_init__(self):
        if self.output_urls is None:
            self.output_urls = []

class JobTracker:
    """Simple file-based job tracking system."""
    
    def __init__(self, storage_path: str = "/tmp/job_tracker.json"):
        self.storage_path = storage_path
        self.jobs: Dict[str, GenerationJob] = {}
        self._load_jobs()
    
    def create_job(self, request_data: Dict[str, Any], job_id: str = None) -> str:
        """Create a new generation job."""
        if job_id is None:
            job_id = str(uuid.uuid4())
        current_time = time.time()
        
        # Extract batch size to calculate total images
        batch_size = request_data.get("batch_size", 15)
        
        job = GenerationJob(
            job_id=job_id,
            status=JobStatus.SUBMITTED,
            request_data=request_data,
            created_at=current_time,
            updated_at=current_time,
            total_images=batch_size,
            cost_estimate=batch_size * 0.10  # $0.10 per image target
        )
        
        self.jobs[job_id] = job
        self._save_jobs()
        
        print(f"📋 Created job: {job_id}")
        return job_id
    
    def update_job_status(self, job_id: str, status: JobStatus, **kwargs) -> bool:
        """Update job status and optional additional fields."""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        job.status = status
        job.updated_at = time.time()
        
        # Update optional fields
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        
        # Calculate progress and estimated remaining time
        if status == JobStatus.PROCESSING:
            if job.total_images > 0:
                job.progress = job.images_completed / job.total_images
                
                # Estimate remaining time based on progress
                if job.progress > 0:
                    elapsed = time.time() - job.created_at
                    total_estimated = elapsed / job.progress
                    remaining = total_estimated - elapsed
                    job.estimated_remaining = f"{int(remaining)}s" if remaining > 0 else "0s"
        elif status == JobStatus.COMPLETED:
            job.progress = 1.0
            job.estimated_remaining = "0s"
            job.execution_time = time.time() - job.created_at
        
        self._save_jobs()
        return True
    
    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status in API format."""
        job = self.get_job(job_id)
        if not job:
            return None
        
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": job.progress,
            "images_completed": job.images_completed,
            "total_images": job.total_images,
            "estimated_remaining": job.estimated_remaining,
            "output_urls": job.output_urls,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "execution_time": job.execution_time,
            "cost_estimate": job.cost_estimate,
            "error_message": job.error_message if job.status == JobStatus.FAILED else ""
        }
    
    def mark_processing(self, job_id: str) -> bool:
        """Mark job as processing."""
        return self.update_job_status(job_id, JobStatus.PROCESSING)
    
    def mark_completed(self, job_id: str, output_urls: list, execution_time: float = 0.0) -> bool:
        """Mark job as completed with results."""
        return self.update_job_status(
            job_id, 
            JobStatus.COMPLETED,
            output_urls=output_urls,
            execution_time=execution_time,
            images_completed=len(output_urls)
        )
    
    def mark_failed(self, job_id: str, error_message: str) -> bool:
        """Mark job as failed with error message."""
        return self.update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=error_message
        )
    
    def update_progress(self, job_id: str, images_completed: int) -> bool:
        """Update job progress."""
        return self.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            images_completed=images_completed
        )
    
    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up jobs older than specified hours."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        old_jobs = [
            job_id for job_id, job in self.jobs.items()
            if current_time - job.created_at > max_age_seconds
        ]
        
        for job_id in old_jobs:
            del self.jobs[job_id]
        
        if old_jobs:
            self._save_jobs()
            print(f"🧹 Cleaned up {len(old_jobs)} old jobs")
        
        return len(old_jobs)
    
    def get_active_jobs(self) -> Dict[str, GenerationJob]:
        """Get all non-completed jobs."""
        return {
            job_id: job for job_id, job in self.jobs.items()
            if job.status in [JobStatus.SUBMITTED, JobStatus.PROCESSING]
        }
    
    def _load_jobs(self):
        """Load jobs from storage file."""
        try:
            if Path(self.storage_path).exists():
                with open(self.storage_path, 'r') as f:
                    jobs_data = json.load(f)
                    
                # Convert dict data back to GenerationJob objects
                for job_id, job_data in jobs_data.items():
                    # Convert status string back to enum
                    job_data['status'] = JobStatus(job_data['status'])
                    self.jobs[job_id] = GenerationJob(**job_data)
                
                print(f"📂 Loaded {len(self.jobs)} jobs from storage")
        except Exception as e:
            print(f"⚠️ Could not load jobs from storage: {e}")
            self.jobs = {}
    
    def _save_jobs(self):
        """Save jobs to storage file."""
        try:
            # Convert GenerationJob objects to dict for JSON serialization
            jobs_data = {}
            for job_id, job in self.jobs.items():
                job_dict = asdict(job)
                job_dict['status'] = job.status.value  # Convert enum to string
                jobs_data[job_id] = job_dict
            
            # Ensure directory exists
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.storage_path, 'w') as f:
                json.dump(jobs_data, f, indent=2)
                
        except Exception as e:
            print(f"❌ Could not save jobs to storage: {e}")

# Global job tracker instance
job_tracker = JobTracker()

def get_job_tracker() -> JobTracker:
    """Get the global job tracker instance."""
    return job_tracker 