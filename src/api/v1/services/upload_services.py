from pathlib import Path
from fastapi import UploadFile
import shutil
from src.ingestion.ingestion import run_ingestion

UPLOAD_DIR = Path("data")


def upload_document(file: UploadFile) -> str:

    try:
        if not UPLOAD_DIR.exists():
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        file_path = UPLOAD_DIR / file.filename
        # Using shutil to save the uploaded file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print("File uploaded successfully")
        run_ingestion(str(file_path))
        print("File successfully ingested")
    except Exception as e:
        print(f"Exception occured while ingesting file as {e}")
