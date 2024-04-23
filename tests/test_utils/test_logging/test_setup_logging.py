from typing import Optional

import pytest
import structlog
from structlog.testing import capture_logs

from ml_assemblr.utils.logging.setup_logging import setup_logging


@pytest.mark.parametrize(
    "is_pretty_render, is_fast_json_render",
    [
        (None, None),
        (None, True),
        (None, False),
        (True, None),
        (True, True),
        (True, False),
        (False, None),
        (False, True),
        (False, False),
    ],
)
def test_all_setup_structlog_expected_behavior(
    is_pretty_render: Optional[bool], is_fast_json_render: Optional[bool]
):
    setup_logging(is_pretty_render=is_pretty_render, is_fast_json_render=is_fast_json_render)
    with capture_logs() as cap_logs:
        structlog.get_logger(te43=True).warning("trewrw")
        assert len(cap_logs) == 1
