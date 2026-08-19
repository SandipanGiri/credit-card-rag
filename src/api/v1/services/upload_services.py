from pathlib import Path
from fastapi import UploadFile
import shutil
from src.ingestion.ingestion import run_ingestion

# from src.ingestion.ingestion import ingest_pdf
# from src.ingestion.ingestion_rerank import ingest_pdf


# async def upload_document(file: UploadFile):
#     print("am in uploadservice")
#     print("filename ", file)
#     Data = Path("data")
#     Data.mkdir(exist_ok=True)
#     file_path = Data / file.filename

#     with open(file_path, "wb") as buffer:
#         buffer.write(await file.read())

#     result = ingest_pdf(file_path)
#     print("*****file ingetsed", result)
#     return {
#         # "message": "File uploaded successfully",
#         # "filename": file.filename,
#         "message": result
#     }


# service to upload file
UPLOAD_DIR = Path("data")


def ingest_file_service(file: UploadFile) -> str:

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
