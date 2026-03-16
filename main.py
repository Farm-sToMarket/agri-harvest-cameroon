"""
Main entry point for the Cameroon Agricultural Data Management System
"""

import asyncio
import logging
from config.settings import get_settings, setup_logging
from config.database import get_database


async def main():
    """Main application entry point"""
    settings = get_settings()
    setup_logging(settings)

    logger = logging.getLogger(__name__)
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    try:
        # Initialize database connection
        db = await get_database()
        logger.info("Database connection established successfully")

        # TODO: Add your main application logic here
        logger.info("Application initialized successfully")

    except Exception as e:
        logger.error("Failed to start application: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(main())
