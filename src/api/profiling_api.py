"""Module 1 API — profile a CSV over HTTP.

Run:
    uvicorn src.api.profiling_api:app --reload
Then open http://127.0.0.1:8000/docs for an interactive page, or:
    curl -F "file=@data/raw/bank_marketing_sample.csv" http://127.0.0.1:8000/profile

Endpoints:
    GET  /health   -> {"status": "ok"}
    POST /profile  -> profiling_report.json for the uploaded CSV
"""
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.profiling.profiler import REPORT_VERSION, profile_file

MAX_UPLOAD_MB = 50   # keep the server safe from huge uploads

app = FastAPI(
    title="Data Gatekeeper — Profiling API",
    description="Upload a CSV, get back its profiling report (Module 1).",
    version=REPORT_VERSION,
)


@app.get("/health")
def health() -> dict:
    """Simple check that the server is running."""
    return {"status": "ok"}


# A normal "def" (not "async def"): profiling is heavy CPU work, and FastAPI
# runs normal functions in a worker thread, so the server stays responsive.
@app.post("/profile")
def profile(file: UploadFile = File(..., description="CSV file to profile")) -> dict:
    """Profile an uploaded CSV and return the report as JSON."""
    name = file.filename or "upload.csv"
    if not name.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    data = file.file.read()
    if not data.strip():
        raise HTTPException(status_code=400, detail="The file is empty")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB} MB")

    # Save the upload to a temporary file so we can reuse profile_file(path).
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "upload.csv"
        path.write_bytes(data)
        try:
            return profile_file(path, dataset_name=name)
        except Exception as e:   # bad CSV (e.g. broken quotes) -> clear 400 instead of a crash
            raise HTTPException(status_code=400, detail=f"Could not read the CSV: {e}") from e
