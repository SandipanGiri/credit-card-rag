from fastapi import APIRouter, UploadFile, File
from src.api.v1.services.upload_services import upload_document

router = APIRouter(prefix="/api/v1/documents")


@router.post("/")
def upload(file: UploadFile = File(...)):
    print("at upload route")
    response = upload_document(file)

    return response
