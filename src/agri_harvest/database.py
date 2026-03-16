"""
Database connection utilities
This module provides consistent access to the MongoDB database
"""

from config.database import get_database, get_database_config

# Re-export main database functions for consistency
__all__ = ['get_database', 'get_database_config']
