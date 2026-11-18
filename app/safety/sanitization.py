"""
File and Input Sanitization utilities.
Validates file extensions, sizes, and cleans filenames.
"""

from pathlib import Path
import re
from typing import Tuple
from app.config import config
from app.utils.logger import logger


class InputSanitizer:
    """Enforces boundaries and security rules for uploads and text inputs."""

    MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Removes dangerous characters, path traversal sequences, and normalizes name."""
        base_name = Path(filename).name
        # Keep alphanumeric, dot, dash, underscore
        clean_name = re.sub(r"[^A-Za-z0-9._-]", "_", base_name)
        return clean_name or "uploaded_document"

    @classmethod
    def validate_file(cls, filename: str, file_size: int) -> Tuple[bool, str]:
        """
        Validates file extension and size.
        Returns: (is_valid: bool, error_message: str)
        """
        ext = Path(filename).suffix.lower()
        if ext not in config.SUPPORTED_EXTENSIONS:
            return False, f"Unsupported file type '{ext}'. Supported: {', '.join(config.SUPPORTED_EXTENSIONS)}"

        if file_size <= 0:
            return False, "File is empty (0 bytes)."

        if file_size > cls.MAX_FILE_SIZE_BYTES:
            max_mb = cls.MAX_FILE_SIZE_BYTES // (1024 * 1024)
            return False, f"File exceeds maximum allowed size of {max_mb} MB."

        return True, "Valid"
