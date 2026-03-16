"""
File handling utilities for agricultural data processing
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def ensure_directory_exists(directory_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't

    Args:
        directory_path: Path to directory

    Returns:
        Path object of the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json_file(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Read JSON file with error handling

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary from JSON file or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("Error reading JSON file %s: %s", file_path, e)
        return None


def write_json_file(
    data: Dict[str, Any],
    file_path: Union[str, Path],
    indent: int = 2
) -> bool:
    """
    Write data to JSON file with error handling

    Args:
        data: Data to write
        file_path: Path to output file
        indent: JSON indentation level

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error("Error writing JSON file %s: %s", file_path, e)
        return False


def read_csv_file(
    file_path: Union[str, Path],
    delimiter: str = ',',
    encoding: str = 'utf-8'
) -> Optional[List[Dict[str, Any]]]:
    """
    Read CSV file and return list of dictionaries

    Args:
        file_path: Path to CSV file
        delimiter: CSV delimiter
        encoding: File encoding

    Returns:
        List of dictionaries or None if error
    """
    try:
        data = []
        with open(file_path, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                data.append(dict(row))
        return data
    except Exception as e:
        logger.error("Error reading CSV file %s: %s", file_path, e)
        return None


def write_csv_file(
    data: List[Dict[str, Any]],
    file_path: Union[str, Path],
    fieldnames: Optional[List[str]] = None,
    delimiter: str = ',',
    encoding: str = 'utf-8'
) -> bool:
    """
    Write data to CSV file

    Args:
        data: List of dictionaries to write
        file_path: Path to output file
        fieldnames: List of field names (auto-detected if None)
        delimiter: CSV delimiter
        encoding: File encoding

    Returns:
        True if successful, False otherwise
    """
    if not data:
        logger.warning("No data to write to %s", file_path)
        return False

    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        if fieldnames is None:
            fieldnames = list(data[0].keys())

        with open(file_path, 'w', encoding=encoding, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        logger.error("Error writing CSV file %s: %s", file_path, e)
        return False


def get_file_size(file_path: Union[str, Path]) -> Optional[int]:
    """
    Get file size in bytes

    Args:
        file_path: Path to file

    Returns:
        File size in bytes or None if file doesn't exist
    """
    try:
        return Path(file_path).stat().st_size
    except FileNotFoundError:
        return None


def validate_file_extension(
    file_path: Union[str, Path],
    allowed_extensions: List[str]
) -> bool:
    """
    Validate file has allowed extension

    Args:
        file_path: Path to file
        allowed_extensions: List of allowed extensions (e.g., ['.json', '.csv'])

    Returns:
        True if extension is allowed
    """
    file_ext = Path(file_path).suffix.lower()
    return file_ext in [ext.lower() for ext in allowed_extensions]


def clean_filename(filename: str) -> str:
    """
    Clean filename by removing invalid characters

    Args:
        filename: Original filename

    Returns:
        Cleaned filename safe for filesystem
    """
    # Remove invalid characters for most filesystems
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    # Remove extra spaces and dots
    filename = filename.strip(' .')

    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext

    return filename


def backup_file(file_path: Union[str, Path], backup_suffix: str = '.bak') -> bool:
    """
    Create backup of existing file

    Args:
        file_path: Path to file to backup
        backup_suffix: Suffix to add to backup file

    Returns:
        True if backup successful, False otherwise
    """
    try:
        original_path = Path(file_path)
        if not original_path.exists():
            logger.warning("File %s does not exist, cannot backup", file_path)
            return False

        backup_path = original_path.with_suffix(original_path.suffix + backup_suffix)
        original_path.rename(backup_path)
        logger.info("Created backup: %s", backup_path)
        return True
    except Exception as e:
        logger.error("Error creating backup of %s: %s", file_path, e)
        return False


def list_files_in_directory(
    directory_path: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """
    List files in directory matching pattern

    Args:
        directory_path: Directory to search
        pattern: File pattern (e.g., "*.json")
        recursive: Search subdirectories

    Returns:
        List of Path objects
    """
    try:
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            return []

        if recursive:
            return list(path.rglob(pattern))
        else:
            return list(path.glob(pattern))
    except Exception as e:
        logger.error("Error listing files in %s: %s", directory_path, e)
        return []
