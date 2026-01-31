from typing import Optional
from .constants import ERROR_CODE_PROCESSING_ERROR, ERROR_CODE_INVALID_INPUT, ERROR_CODE_FILE_NOT_FOUND

class KelpError(Exception):
    def __init__(self, msg, error_code=ERROR_CODE_PROCESSING_ERROR, context=None):
        self.msg = msg
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
    def __init__(self, msg, context=None):
        super().__init__(msg, ERROR_CODE_INVALID_INPUT, context)

class KelpFileNotFoundError(KelpError):
    def __init__(self, file_path, context=None):
        msg = f"File not found: {file_path}"
        super().__init__(msg, ERROR_CODE_FILE_NOT_FOUND, context)
