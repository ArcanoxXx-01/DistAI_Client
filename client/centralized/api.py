"""
Alternative: Separate endpoints for dataset upload and training.
"""

# from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
# from schemas.training import (
#     TrainingRequest, 
#     TrainingResponse, 
#     TrainingStatus,
#     DatasetUploadResponse,
#     TrainingResultsResponse
# )
# from services.storage_service import storage_service
# from services.training_service import training_service


router = APIRouter()


@router.post("/dataset/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV dataset file")
):
    """Upload a dataset and get a dataset_id."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    file_content = await file.read()
    dataset_id = await storage_service.save_dataset(file_content, file.filename)
    
    return DatasetUploadResponse(dataset_id=dataset_id)


@router.post("/train", response_model=TrainingResponse)
async def create_training_job(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a training job with a previously uploaded dataset.
    You need to upload the dataset first using /dataset/upload endpoint.
    The training will run in the background, and you can check its status
    using the /train/{training_id}/status endpoint.
    """
    # Validate that the dataset exists
    if not await storage_service.dataset_exists(request.dataset_id):
        raise HTTPException(
            status_code=404, 
            detail=f"Dataset with id '{request.dataset_id}' not found. Please upload the dataset first."
        )
    
    # Create the training job record and get the training ID
    training_id = await training_service.create_training_job(
        dataset_id=request.dataset_id,
        model_names=request.model_names,
        train_type=request.train_type.value
    )
    
    # Add the actual training execution as a background task
    background_tasks.add_task(
        training_service.execute_training,
        training_id=training_id,
        dataset_id=request.dataset_id,
        model_names=request.model_names,
        train_type=request.train_type.value
    )
    
    return TrainingResponse(training_id=training_id)


@router.get("/train/{training_id}/status", response_model=TrainingStatus)
async def get_training_status(training_id: str):
    """Get the status of a training job."""
    job_info = await training_service.get_job_status(training_id)
    
    if not job_info:
        raise HTTPException(
            status_code=404,
            detail=f"Training job '{training_id}' not found"
        )
    
    return TrainingStatus(
        training_id=training_id,
        status=job_info["status"]
    )


@router.get("/train/{training_id}/results", response_model=TrainingResultsResponse)
async def get_training_results(training_id: str):
    """
    Get comprehensive results for a training job.
    This endpoint retrieves all model results for the specified training ID,
    including metrics, status, and any errors that occurred during training.
    """
    results = await training_service.get_training_results(training_id)
    
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Training job '{training_id}' not found"
        )
    
    return TrainingResultsResponse(**results)
