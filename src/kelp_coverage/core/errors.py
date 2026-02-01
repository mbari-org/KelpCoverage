from typing import Optional, Dict, Any
from .constants import (
    ERROR_CODE_SUCCESS, 
    ERROR_CODE_INVALID_INPUT, 
    ERROR_CODE_FILE_NOT_FOUND, 
    ERROR_CODE_DIR_NOT_FOUND, 
    ERROR_CODE_INVALID_IMAGE, 
    ERROR_CODE_MODEL_LOAD_ERROR, 
    ERROR_CODE_PROCESSING_ERROR
)

class KelpError(Exception):
    def __init__(self, msg=None, error_code=ERROR_CODE_PROCESSING_ERROR, context=None):
        self.msg = msg or "an unknown error has occurred"
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self._format_message())

    def _format_message(self):
        formatted = f"[Error {self.error_code}] {self.msg}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            formatted += f" (reason: {context_str})"
        return formatted

class KelpInvalidInputError(KelpError):
    def __init__(self, msg=None, context=None):
        super().__init__(msg or "Invalid input provided", ERROR_CODE_INVALID_INPUT, context)

class KelpFileNotFoundError(KelpError):
    def __init__(self, file_path=None, context=None):
        super().__init__(f"File not found: {file_path or 'unknown path'}", ERROR_CODE_FILE_NOT_FOUND, context)

class KelpDirNotFoundError(KelpError):
    def __init__(self, dir_path=None, context=None):
        super().__init__(f"Directory not found: {dir_path or 'unknown path'}", ERROR_CODE_DIR_NOT_FOUND, context)
