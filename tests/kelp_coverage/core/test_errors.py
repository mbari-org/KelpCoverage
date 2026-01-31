import pytest
from kelp_coverage.core.errors import KelpError, KelpInvalidInputError

class TestKelpError:
    def test_basic_error(self):
        msg = "something went wrong"
        error = KelpError(msg)
        print(str(error))
        assert error.msg == msg
        assert error.error_code == 5
        assert "something went wrong" in str(error)
