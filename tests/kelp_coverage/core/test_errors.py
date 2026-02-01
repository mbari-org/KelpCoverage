import pytest
from kelp_coverage.core.errors import (
    KelpError, 
    KelpInvalidInputError, 
    KelpFileNotFoundError, 
    KelpDirNotFoundError,
    ERROR_CODE_PROCESSING_ERROR,
    ERROR_CODE_INVALID_INPUT,
    ERROR_CODE_FILE_NOT_FOUND,
    ERROR_CODE_DIR_NOT_FOUND
)

class TestKelpError:
    def test_error_default_msg(self):
        error = KelpError()
        assert error.msg == "an unknown error has occurred"
        assert f"[Error {ERROR_CODE_PROCESSING_ERROR}] an unknown error has occurred" in str(error)

    def test_error_custom_msg(self):
        msg = "something went wrong"
        error = KelpError(msg)
        assert error.msg == msg
        assert error.error_code == ERROR_CODE_PROCESSING_ERROR
        assert f"[Error {ERROR_CODE_PROCESSING_ERROR}] {msg}" in str(error)

    def test_error_codes(self):
        code = 99
        error = KelpError(error_code=code)
        assert error.error_code == 99
        assert "[Error 99]" in str(error)
    
    def test_error_context(self, sample_context):
        error = KelpError(context=sample_context)
        assert error.context == sample_context
        assert "(reason:" in str(error)
        assert "key=value" in str(error)
        assert "count=42" in str(error)

class TestKelpInvalidInputError:
    def test_default_values(self):
        error = KelpInvalidInputError()
        assert error.error_code == ERROR_CODE_INVALID_INPUT
        assert "Invalid input provided" in str(error)

class TestKelpFileNotFoundError:
    def test_default_no_path(self):
        error = KelpFileNotFoundError()
        assert "File not found: unknown path" in str(error)
        assert error.error_code == ERROR_CODE_FILE_NOT_FOUND

    def test_with_path(self):
        path = "/tmp/missing.json"
        error = KelpFileNotFoundError(path)
        assert f"File not found: {path}" in str(error)
        assert error.error_code == ERROR_CODE_FILE_NOT_FOUND

class TestKelpDirNotFoundError:
    def test_default_no_path(self):
        error = KelpDirNotFoundError()
        assert "Directory not found: unknown path" in str(error)
        assert error.error_code == ERROR_CODE_DIR_NOT_FOUND

    def test_with_path(self):
        path = "/usr/local/configs"
        error = KelpDirNotFoundError(path)
        assert f"Directory not found: {path}" in str(error)
        assert error.error_code == ERROR_CODE_DIR_NOT_FOUND