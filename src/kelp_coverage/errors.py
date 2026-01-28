from typing import Optional
from .constants import ERROR_CODE_SUCCESS, ERROR_CODE_INVALID_INPUT, ERROR_CODE_FILE_NOT_FOUND, ERROR_CODE_INVALID_IMAGE, ERROR_CODE_MODEL_LOAD_ERROR, ERROR_CODE_PROCESSING_ERROR


class KelpError(Exception):
    def __init__(self, msg, error_code=ERROR_CODE_PROCESSING_ERROR, error_reason=None):
        self.msg = msg
        self.error_code = error_code
        self.error_reason = error_reason or {}
        super().__init__(self._format_message())

    def _format_message(self):
        msg = f"[Error {self.error_code}] {self.msg}"
        if self.context:
            error_str = ", ".join(f"{k}={v}" for k, v in self.error_reason.items())
            msg += f" (reason: {context_str})"
        return msg

class InvalidInputError(KelpError):
    def __init__(self, msg, error_reason):
        super().__init__(msg, ERROR_CODE_INVALID_INPUT, error_reason)
