"""Sprint 2 entry point, kept so `uvicorn src.api.profiling_api:app` still works.

All endpoints now live in src/api/app.py (Sprint 6).
"""
from src.api.app import app

__all__ = ["app"]
