"""
Main entry point for the Cameroon Agricultural Data Management System
"""

import logging

import uvicorn
from fastapi import FastAPI

from config.settings import get_settings, setup_logging

settings = get_settings()
setup_logging(settings)
logger = logging.getLogger(__name__)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
