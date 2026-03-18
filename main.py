"""
Main entry point for the Cameroon Agricultural Data Management System
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from config.settings import get_settings, setup_logging
from config.database import get_database, get_database_config

settings = get_settings()
setup_logging(settings)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Connect to database on startup, disconnect on shutdown."""
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    try:
        await get_database()
        logger.info("Database connection established successfully")
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error("Failed to start application: %s", e)
        raise
    yield
    db_config = get_database_config()
    await db_config.disconnect()
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Database health check endpoint."""
    db_config = get_database_config()
    return await db_config.health_check()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
