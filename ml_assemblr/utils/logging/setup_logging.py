import logging
from collections.abc import Callable, Collection
from typing import Iterable, Union

import orjson
import structlog
from structlog import processors
from structlog.processors import CallsiteParameter
from structlog.typing import EventDict, Processor, WrappedLogger

from ml_assemblr.utils.notebook_utils import is_in_notebook


class FilteredCallsiteParameterAdder(processors.CallsiteParameterAdder):

    def __init__(
        self,
        parameters: Collection[CallsiteParameter] = processors.CallsiteParameterAdder._all_parameters,
        additional_ignores: Union[list[str], None] = None,
        filtered_min_level=logging.ERROR,
    ) -> None:
        super().__init__(parameters, additional_ignores)
        self.filtered_min_level = filtered_min_level

    def __call__(self, logger: logging.Logger, name: str, event_dict: EventDict) -> EventDict:
        """Add specified parameter to `event_dict` if severity of log is equal or
        higher to `filtered_min_level`."""
        if logging.getLevelName(name.upper()) < self.filtered_min_level:
            return event_dict
        return super().__call__(logger, name, event_dict)


def setup_logging(
    is_pretty_render: Union[bool, None] = None,
    is_fast_json_render: Union[bool, None] = None,
    include_locals_in_traceback: bool = False,
    severity_level: int = logging.INFO,
):
    if is_pretty_render is None and is_in_notebook():
        is_pretty_render = True

    if is_pretty_render:
        is_fast_json_render = False

    if is_fast_json_render is None:
        is_fast_json_render = not is_in_notebook()

    exception_renderer = processors.ExceptionRenderer(
        structlog.tracebacks.ExceptionDictTransformer(show_locals=include_locals_in_traceback)
    )

    logger_factory: Callable[..., WrappedLogger] = structlog.WriteLoggerFactory()

    if is_fast_json_render:
        renderers = [
            exception_renderer,
            processors.JSONRenderer(serializer=orjson.dumps),
        ]
        logger_factory = structlog.BytesLoggerFactory()

    elif is_pretty_render:
        renderers = [structlog.dev.ConsoleRenderer(pad_event=30)]

    else:
        renderers: Iterable[Processor] = [
            exception_renderer,
            processors.JSONRenderer(),
        ]

    structlog.configure(
        cache_logger_on_first_use=True,
        wrapper_class=structlog.make_filtering_bound_logger(severity_level),
        processors=[
            structlog.contextvars.merge_contextvars,
            processors.add_log_level,
            processors.TimeStamper(fmt="iso", utc=True),
            processors.StackInfoRenderer(),
            FilteredCallsiteParameterAdder(
                [
                    CallsiteParameter.FUNC_NAME,
                    CallsiteParameter.PATHNAME,
                ],
                additional_ignores=["ml_assemblr.utils.logging.setup_logging"],
            ),
            *renderers,
        ],
        logger_factory=logger_factory,
    )
