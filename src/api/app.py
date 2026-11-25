"""Data Gatekeeper API — profile, clean, transform, mask and validate CSVs over HTTP.

Run:   make api          (= uvicorn src.api.app:app --reload)
Docs:  http://127.0.0.1:8000/docs  (interactive page to try every endpoint)

Endpoints (upload one CSV as the form field "file"):
    GET  /health      -> {"status": "ok"}
    POST /profile     -> Module 1 profiling report (JSON)
    POST /clean       -> cleaning log + quality before/after (JSON), or ?output=csv
    POST /transform   -> transform report (JSON), or ?output=csv (train + test rows)
    POST /mask-pii    -> PII report (JSON), or ?output=csv (the protected data)
    POST /validate    -> validation report: contract (+ drift with a reference file,
                         + anomalies with ?anomalies=true)           (Sprint 9)
    POST /drift       -> drift report of "current" against "reference" (Sprint 9)
    POST /anomalies   -> unusual rows with explanations (Sprint 9)

Safety rules:
  * .csv only, at most MAX_UPLOAD_MB and MAX_ROWS -> clear 4xx errors otherwise
  * the upload is read into memory and parsed there; we never save it to disk
    (Starlette may buffer an upload over 1 MB in a temporary file; it is deleted when
    the request ends, and we close it as soon as the bytes are read)
  * unexpected errors -> 500 with a short message; the stack trace and any data value
    stay out of the response (and out of the log: only the error type is logged)
"""
# No "from __future__ import annotations" here: FastAPI reads the parameter types at
# runtime, and Python 3.9 cannot evaluate "str | None", so Optional/Union are used.
import io
import logging
from typing import Literal, Optional, Union

import pandas as pd
import yaml
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from src.cleaning.cleaner import clean_dataset
from src.privacy.mask import PrivacyError, mask_dataframe
from src.profiling.profiler import REPORT_VERSION, profile_dataset, sniff_separator
from src.quality.score import quality_delta, quality_score
from src.transform.pipeline import TEST_SIZE, transform_dataset
from src.validation.anomaly import METHODS, detect
from src.validation.contract import ContractError, parse_contract
from src.validation.drift import drift_report
from src.validation.report import parse_methods, validation_report

MAX_UPLOAD_MB = 20
MAX_ROWS = 200_000
MAX_CONTRACT_KB = 256        # a contract is a small YAML file
PREVIEW_ROWS = 5
FORM_OVERHEAD = 64 * 1024    # multipart headers around the file

log = logging.getLogger("gatekeeper.api")

app = FastAPI(
    title="Data Gatekeeper API",
    description="Upload a CSV: profile it, clean it, transform it for ML, mask its PII, "
                "or validate it (contract, drift, anomalies).",
    version=REPORT_VERSION,
)


@app.middleware("http")
async def guard(request: Request, call_next):  # noqa: ANN001, ANN201
    """Runs around every request.

    1. refuse a too-large upload from its Content-Length, before the body is read
    2. any unexpected error -> a plain 500. We catch it here (not with an exception
       handler, which re-raises) so neither the client nor the server log gets a stack
       trace or an error message that might contain a data value: only the error type.
    """
    size = request.headers.get("content-length", "")
    if size.isdigit() and int(size) > MAX_UPLOAD_MB * 1024 * 1024 + FORM_OVERHEAD:
        return JSONResponse(status_code=413,
                            content={"detail": f"File is larger than {MAX_UPLOAD_MB} MB"})
    try:
        return await call_next(request)
    except Exception as exc:
        log.error("%s %s failed: %s", request.method, request.url.path, type(exc).__name__)
        return JSONResponse(status_code=500,
                            content={"detail": "Internal error while processing the file"})


def read_upload(file: UploadFile) -> tuple[bytes, str]:
    """Check an upload (.csv, not empty, not too big) and return (bytes, file name)."""
    name = file.filename or "upload.csv"
    try:
        if not name.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Please upload a .csv file")
        limit = int(MAX_UPLOAD_MB * 1024 * 1024)
        data = file.file.read(limit + 1)       # never read more than the limit + 1 byte
    finally:
        file.file.close()
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB} MB")
    if not data.strip():
        raise HTTPException(status_code=400, detail="The file is empty")
    return data, name


def parse_csv(data: bytes, as_text: bool = True) -> pd.DataFrame:
    """Parse CSV bytes in memory. as_text=True keeps every value as text (like Module 1)."""
    sep = sniff_separator(data[:64_000].decode("utf-8", errors="replace"))
    try:
        df = pd.read_csv(io.BytesIO(data), sep=sep, dtype=str if as_text else None,
                         encoding_errors="replace")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as e:
        # pandas parser messages describe the structure ("Expected 2 fields in line 3"),
        # not the values; only the first line is kept
        reason = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
        raise HTTPException(status_code=400, detail=f"Could not read the CSV: {reason}") from e
    if df.shape[1] == 0 or df.empty:
        raise HTTPException(status_code=400, detail="The CSV has no data rows")
    if len(df) > MAX_ROWS:
        raise HTTPException(status_code=413,
                            detail=f"The CSV has {len(df)} rows; the limit is {MAX_ROWS}")
    return df


def csv_download(df: pd.DataFrame, filename: str) -> Response:
    """A CSV file as the response (the browser / curl -OJ saves it)."""
    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def preview(df: pd.DataFrame) -> list[dict]:
    """The first rows as JSON-friendly records (NaN -> null)."""
    head = df.head(PREVIEW_ROWS).astype(object)
    return head.where(head.notna(), None).to_dict(orient="records")


@app.get("/health")
def health() -> dict[str, str]:
    """Simple check that the server is running."""
    return {"status": "ok"}


# Normal "def" (not "async def"): this is heavy CPU work, and FastAPI runs normal
# functions in a worker thread, so the server stays responsive meanwhile.
@app.post("/profile")
def profile(file: UploadFile = File(..., description="CSV file to profile")) -> dict:
    """Profile an uploaded CSV and return the Module 1 report as JSON."""
    data, name = read_upload(file)
    return profile_dataset(parse_csv(data), dataset_name=name)


@app.post("/clean", response_model=None)
def clean(file: UploadFile = File(..., description="raw CSV file to clean"),
          imputer: Optional[Literal["median", "knn", "regression", "iterative"]] = Query(
              None, description="imputer for all columns (default: iterative; median is fastest)"),
          outliers: Optional[Literal["flag", "cap", "nan"]] = Query(None),
          output: Literal["json", "csv"] = Query("json", description="csv = download the data"),
          ) -> Union[dict, Response]:
    """Profile + clean an uploaded CSV; JSON log with quality before/after, or the CSV."""
    data, name = read_upload(file)
    raw = parse_csv(data)
    report = profile_dataset(raw, dataset_name=name)
    config = {}
    if imputer:
        config["numeric_strategy"] = config["categorical_strategy"] = imputer
    if outliers:
        config["outlier_action"] = outliers
    cleaned, clean_log = clean_dataset(raw, report, config)
    if output == "csv":
        return csv_download(cleaned, f"cleaned_{name}")
    # score the cleaned data the way a user of the CSV would read it (as text)
    after = profile_dataset(parse_csv(cleaned.to_csv(index=False).encode()), dataset_name=name)
    before_q = quality_score(report)
    after_q = quality_score(after, flag_columns=clean_log["summary"]["columns_added"])
    clean_log["quality"] = {"before": before_q, "after": after_q,
                            "delta": quality_delta(before_q, after_q)}
    return {"dataset": name, "summary": clean_log["summary"], "quality": clean_log["quality"],
            "preview": preview(cleaned), "log": clean_log}


@app.post("/transform", response_model=None)
def transform(file: UploadFile = File(..., description="CLEANED CSV file"),
              target: Optional[str] = Query(None, description="label column, e.g. y"),
              test_size: float = Query(TEST_SIZE, gt=0, lt=1),
              output: Literal["json", "csv"] = Query("json"),
              ) -> Union[dict, Response]:
    """Encode + scale for ML (fitted on the train rows only); report or the rows as CSV."""
    data, name = read_upload(file)
    try:
        result = transform_dataset(parse_csv(data, as_text=False), target=target,
                                   test_size=test_size)
    except ValueError as e:                  # e.g. target not found, no usable columns
        raise HTTPException(status_code=400, detail=str(e)) from e
    if output == "csv":
        rows = pd.concat([result.train.assign(split="train"), result.test.assign(split="test")])
        return csv_download(rows, f"transformed_{name}")
    return {"dataset": name, "report": result.report, "preview": preview(result.train)}


@app.post("/mask-pii", response_model=None)
def mask_pii(file: UploadFile = File(..., description="CSV file with personal data"),
             action: list[str] = Query([], description="COLUMN=ACTION, e.g. email=mask "
                                       "(mask, hash, drop, generalize, keep); repeatable"),
             output: Literal["json", "csv"] = Query("json"),
             ) -> Union[dict, Response]:
    """Detect PII and protect it; JSON report (+ masked preview) or the protected CSV."""
    data, name = read_upload(file)
    actions = {}
    for item in action:
        column, sep, value = item.rpartition("=")
        if not sep or not column:
            raise HTTPException(status_code=400, detail=f"action {item!r}: use COLUMN=ACTION")
        actions[column] = value
    try:
        masked, report = mask_dataframe(parse_csv(data), actions)
    except PrivacyError as e:
        if "salt" in str(e):                 # a server setting, not the client's mistake
            raise HTTPException(status_code=503, detail="Hashing is not set up on this "
                                "server; choose another action, e.g. action=email=mask") from e
        raise HTTPException(status_code=400, detail=str(e)) from e
    if output == "csv":
        return csv_download(masked, f"masked_{name}")
    return {"dataset": name, "report": report, "preview": preview(masked)}


# ------------------------------------------------------------- Sprint 9: Module 3
def read_contract(file: UploadFile) -> dict:
    """Check and parse an uploaded contract (.yaml / .yml, small, valid format)."""
    name = file.filename or "contract.yaml"
    try:
        if not name.lower().endswith((".yaml", ".yml")):
            raise HTTPException(status_code=400, detail="The contract must be a .yaml file")
        limit = MAX_CONTRACT_KB * 1024
        data = file.file.read(limit + 1)
    finally:
        file.file.close()
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Contract is larger than {MAX_CONTRACT_KB} KB")
    try:
        return parse_contract(yaml.safe_load(data.decode("utf-8", errors="replace")),
                              source="contract")
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail="The contract is not valid YAML") from e
    except ContractError as e:                 # names the broken rule, never a data value
        raise HTTPException(status_code=400, detail=str(e)) from e


def read_csv_upload(file: UploadFile) -> tuple[pd.DataFrame, str]:
    """An uploaded CSV with the Sprint 6 limits, every value as text."""
    data, name = read_upload(file)
    return parse_csv(data), name


@app.post("/validate")
def validate(file: UploadFile = File(..., description="CSV file to check"),
             contract: UploadFile = File(..., description="contract YAML"),
             reference: Optional[UploadFile] = File(
                 None, description="optional reference CSV: adds the drift section"),
             anomalies: bool = Query(False, description="add the anomaly section"),
             method: str = Query("iforest", description="iforest, zscore, iqr or all"),
             exclude: list[str] = Query([], description="columns never used, e.g. y"),
             fail_on_share: Optional[float] = Query(None, ge=0, le=1),
             ) -> dict:
    """Contract check (+ drift, + anomalies) -> the validation report (status pass / fail)."""
    spec = read_contract(contract)
    df, name = read_csv_upload(file)
    ref, ref_name = (None, "reference")
    if reference is not None:
        ref, ref_name = read_csv_upload(reference)
    try:
        return validation_report(df, spec, name, reference=ref, reference_name=ref_name,
                                 anomalies=anomalies, methods=parse_methods(method),
                                 exclude=exclude, fail_on_share=fail_on_share)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/drift")
def drift(reference: UploadFile = File(..., description="reference CSV (the good data)"),
          current: UploadFile = File(..., description="new CSV to compare"),
          contract: Optional[UploadFile] = File(None, description="optional contract YAML "
                                                "(drift thresholds, column types)"),
          ) -> dict:
    """PSI + KS per column of current against reference; PII values never shown."""
    spec = read_contract(contract) if contract is not None else None
    ref, ref_name = read_csv_upload(reference)
    cur, cur_name = read_csv_upload(current)
    return drift_report(ref, cur, spec, reference_name=ref_name, current_name=cur_name)


@app.post("/anomalies")
def anomalies_endpoint(file: UploadFile = File(..., description="CSV file"),
                       method: str = Query("all", description=f"{', '.join(METHODS)} or all"),
                       top: int = Query(10, ge=1, le=100),
                       contamination: float = Query(0.01, gt=0, lt=0.5),
                       exclude: list[str] = Query([], description="columns never used, e.g. y"),
                       ) -> dict:
    """Score rows (robust z, IQR, Isolation Forest) and explain the top ones."""
    df, _ = read_csv_upload(file)
    try:
        return detect(df, methods=parse_methods(method), top=top, contamination=contamination,
                      exclude=exclude)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
