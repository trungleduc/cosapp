import logging
import os
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from cosapp.core.module import Module
from cosapp.drivers import Driver, EulerExplicit, NonLinearSolver
from cosapp.recorders import DataFrameRecorder
from cosapp.systems import System
from cosapp.utils import logging as logging_module
from cosapp.utils.logging import (
    VERBOSE_LEVEL,
    ContextFilter,
    FileLogHandler,
    FilterWithContext,
    HandlerWithContextFilters,
    LogFormat,
    LoggerContext,
    LogLevel,
    StreamLogHandler,
    TimeFilter,
    rollover_logfile,
    set_log,
)


class FakeCoSAppHandler(logging.Handler, HandlerWithContextFilters):
    def __init__(self, *args, **kwargs):
        logging.Handler.__init__(self, *args, **kwargs)
        HandlerWithContextFilters.__init__(self)

    def addFilter(self, filter):
        """Adds the specified filter filter to this handler."""
        super().addFilter(filter)
        self._set_contextual_filters(self.filters)

    def removeFilter(self, filter):
        """Removes the specified filter filter from this handler."""
        super().removeFilter(filter)
        self._set_contextual_filters(self.filters)


def test_HandlerWithContextFilters_contextual_filters():
    h = FakeCoSAppHandler()
    assert list(h.contextual_filters) == list()

    h.addFilter(lambda r: 0)
    assert list(h.contextual_filters) == list()

    time_filter = TimeFilter(22)
    h.addFilter(time_filter)
    assert list(h.contextual_filters) == [time_filter, ]


@pytest.mark.parametrize("filter", [None, lambda r: 0, TimeFilter(22)])
@pytest.mark.parametrize("ctx_param", [None, object(), True, False])
@pytest.mark.parametrize("activation", [None, True, False])
def test_HandlerWithContextFilters_needs_handling(filter, ctx_param, activation):

    ctx_filter = MagicMock(spec=FilterWithContext, current_context=PropertyMock())
    ctx_filter.filter = MagicMock(return_value=1)

    if isinstance(ctx_param, bool):
        ctx = MagicMock(
            spec=LoggerContext, log_debug_message=MagicMock(return_value=ctx_param)
        )
    else:
        ctx = ctx_param

    with patch("logging.StreamHandler.handle") as super_handle:
        h = StreamLogHandler()
        try:
            h.addFilter(ctx_filter)
            if filter is not None:
                h.addFilter(filter)

            rec = logging.LogRecord(
                "log_test", LogLevel.DEBUG, __file__, 22, "msg", dict(), None
            )
            rec.context = ctx
            rec.activate = activation

            h.handle(rec)  # needs_handling called within handle

            if ctx is None:
                super_handle.assert_called_once_with(rec)

            else:
                if activation is not None:
                    assert ctx_filter.current_context is ctx

                if isinstance(ctx, LoggerContext):
                    ctx.log_debug_message.assert_called_once_with(h, rec)
                    if ctx.log_debug_message.return_value:
                        super_handle.assert_called_once_with(rec)
                    else:
                        super_handle.assert_not_called()
                else:
                    super_handle.assert_not_called()

        finally:
            h.close()  # Close the access to the log file


@pytest.mark.parametrize("loglevel", LogLevel)
@pytest.mark.parametrize("handler_type", [HandlerWithContextFilters, StreamLogHandler])
@pytest.mark.parametrize("handler_level", LogLevel)
def test_HandlerWithContextFilters_log(loglevel, handler_type, handler_level):
    handler = handler_type()
    handler.handle = MagicMock()

    if isinstance(handler, logging.Handler):
        handler.setLevel(handler_level)
        handler.log(loglevel, "dummy message")
        if loglevel >= handler_level:
            handler.handle.assert_called_once()
            record = handler.handle.call_args[0][0]
            assert record.levelno == loglevel
            assert record.msg == "dummy message"
        else:
            handler.handle.assert_not_called()
    else:
        with pytest.raises(NotImplementedError):
            handler.log(loglevel, "dummy message")
        handler.handle.assert_not_called()


@pytest.mark.parametrize("loglevel", LogLevel)
def test_HandlerWithContextFilters_log_debug_message_called_only_for_verbose(loglevel):
    root_logger = logging.getLogger()
    root_logger.setLevel(loglevel)
    h = FileLogHandler()
    root_logger.addHandler(h)

    syst = System("dolly")
    syst.log_debug_message = MagicMock(return_value=True)

    with patch("logging.handlers.RotatingFileHandler.handle") as handle:
        syst.call_setup_run()

    if loglevel > VERBOSE_LEVEL:
        syst.log_debug_message.assert_not_called()
        handle.assert_not_called()
    else:
        syst.log_debug_message.assert_called()
        handle.assert_called()

@pytest.mark.parametrize("loglevel", LogLevel)
@pytest.mark.parametrize("suffix", ["", "dummy suffix"])
def test_LoggerContext_log_context(caplog, loglevel, suffix):
    ctx = LoggerContext()

    def check_log(activate):
        if len(caplog.records) == 0:
            assert loglevel > VERBOSE_LEVEL
        else:
            assert loglevel <= VERBOSE_LEVEL
        for record in caplog.records:
            assert hasattr(record, "context")
            assert record.context is ctx
            if len(suffix) > 0:
                assert record.msg.endswith(suffix)
            assert record.activate == activate
            if activate:
                assert record.msg.startswith(LoggerContext.CONTEXT_ENTER_MESSAGE)
            else:
                assert record.msg.startswith(LoggerContext.CONTEXT_EXIT_MESSAGE)

    with caplog.at_level(loglevel):
        caplog.clear()
        with ctx.log_context(suffix):
            check_log(True)
            caplog.clear()
        check_log(False)


def test_LoggerContext_log_debug_message():
    ctx = LoggerContext()
    assert ctx.log_debug_message(object(), object(), LogFormat.RAW)


def test_FilterWithContext():
    called = 0

    class DummyFilter(FilterWithContext):
        def _set_context(self):
            nonlocal called
            called += 1

    f = DummyFilter()
    assert f.current_context is None
    ctx = object()
    f.current_context = ctx
    assert f.current_context is ctx
    assert called == 1


def test_TimeFilter_current_context():
    class FakeContext:
        def __init__(self):
            self.time = 26.0

    ctx = FakeContext()
    filter = TimeFilter(22)
    assert filter.current_context is None
    filter.current_context = ctx
    assert filter.current_context is ctx


@pytest.mark.parametrize("loglevel", LogLevel)
@pytest.mark.parametrize("start_time", [10.0, 26.0, 42])
def test_TimeFilter_filter(loglevel, start_time):
    class FakeContext:
        def __init__(self):
            self.time = 26.0

    ctx = FakeContext()
    filter = TimeFilter(start_time)
    filter.current_context = ctx

    rec = logging.LogRecord("log_test", loglevel, __file__, 22, "msg", dict(), None)

    assert filter.filter(rec) == int(loglevel > VERBOSE_LEVEL or ctx.time >= start_time)


@pytest.mark.parametrize("loglevel", LogLevel)
@pytest.mark.parametrize("name", [None, "small_devil", "big_angel"])
def test_ContextFilter_filter(loglevel, name):
    h = ContextFilter(context=name)
    rec = logging.LogRecord("log_test", loglevel, __file__, 22, "msg", dict(), None)
    assert h.filter(rec) == (name is None or loglevel > VERBOSE_LEVEL)


@pytest.mark.parametrize("handlerlevel", LogLevel)
@pytest.mark.parametrize("name", [None, "small_devil", "big_angel"])
@pytest.mark.parametrize("ctx", [None, System, Driver])
@pytest.mark.parametrize("reclevel", LogLevel)
def test_ContextFilter_filter_integration(handlerlevel, name, ctx, reclevel):
    record_args = ("log_test", reclevel, __file__, 22, "msg", dict(), None)

    with patch("logging.StreamHandler.emit") as emit:
        h = StreamLogHandler()
        try:
            h.setLevel(handlerlevel)
            ctx_filter = ContextFilter(context=name)
            h.addFilter(ctx_filter)

            rec = logging.LogRecord(*record_args)

            # Log with no context
            h.handle(rec)

            if name is None or reclevel > VERBOSE_LEVEL:
                emit.assert_called_once_with(rec)
            else:
                emit.assert_not_called()

            if ctx is not None:
                # Set the context
                rec = logging.LogRecord(*record_args)
                rec.context = ctx(name="small_devil")
                rec.activate = True
                h.handle(rec)

                rec2 = logging.LogRecord(*record_args)
                is_shown = h.filter(rec2)
                assert is_shown == (
                    name is None or name == "small_devil" or reclevel > VERBOSE_LEVEL
                )

                # Set a similar context
                rec = logging.LogRecord(*record_args)
                rec.context = ctx(name="the_devil")
                rec.activate = True
                h.handle(rec)

                assert h.filter(rec2) == is_shown

                # Set a different context
                rec = logging.LogRecord(*record_args)
                rec.context = MagicMock()
                rec.context.configure_mock(name="no_devil")
                rec.activate = True
                h.handle(rec)

                if name is None or name != "small_devil" or reclevel > VERBOSE_LEVEL:
                    assert h.filter(rec2) == is_shown
                else:
                    assert h.filter(rec2) != is_shown

                # Reset the context to stop the filter
                rec = logging.LogRecord(*record_args)
                rec.context = MagicMock(spec=ctx)
                rec.context.configure_mock(name="small_devil")
                rec.activate = True
                h.handle(rec)

                if name is None or name != "small_devil" or reclevel > VERBOSE_LEVEL:
                    assert h.filter(rec2) == is_shown
                else:
                    assert h.filter(rec2) != is_shown

        finally:
            h.close()  # Close the access to the log file


def test_rollover_logfile(tmp_path):
    root_logger = logging.getLogger()
    root_logger.addHandler(FileLogHandler())
    root_logger.addHandler(logging.handlers.RotatingFileHandler(tmp_path / "dummy.log"))
    with patch("logging.handlers.RotatingFileHandler.doRollover") as target:
        rollover_logfile()
    assert target.call_count >= 2


def test_set_log():

    root_logger = logging.getLogger()

    set_log()

    assert root_logger.level == LogLevel.INFO
    hdl = list(filter(lambda h: isinstance(h, HandlerWithContextFilters), root_logger.handlers))
    assert len(hdl) == 2
    for h in hdl:
        assert len(h.filters) == 0

    set_log(start_time=22.0)
    hdl = list(filter(lambda h: isinstance(h, HandlerWithContextFilters), root_logger.handlers))
    assert len(hdl) == 2
    for h in hdl:
        assert len(h.filters) == 1
        assert isinstance(h.filters[0], TimeFilter)

    set_log(context="my_system")
    hdl = list(filter(lambda h: isinstance(h, HandlerWithContextFilters), root_logger.handlers))
    assert len(hdl) == 2
    for h in hdl:
        assert len(h.filters) == 1
        assert isinstance(h.filters[0], ContextFilter)


@pytest.mark.parametrize("loglevel", LogLevel)
def test_set_log_level(loglevel):

    root_logger = logging.getLogger()
    
    set_log(level=loglevel)
    hdl = list(filter(lambda h: isinstance(h, HandlerWithContextFilters), root_logger.handlers))
    assert root_logger.level == loglevel
    for h in hdl:
        assert h.level == loglevel

    set_log(level=int(loglevel))
    hdl = list(filter(lambda h: isinstance(h, HandlerWithContextFilters), root_logger.handlers))
    assert root_logger.level == loglevel
    for h in hdl:
        assert h.level == loglevel


@pytest.mark.parametrize(
    "params, exception",
    [
        (dict(filename=22), TypeError),
        (dict(stream=22), TypeError),
        (dict(level="Jean-Mich"), TypeError),
        (dict(context=False), TypeError),
        (dict(format=22), TypeError),
        (dict(encoding=True), TypeError),
        (dict(backupCount="end"), TypeError),
        (dict(backupCount=-2), ValueError),
    ],
)
def test_set_log_valid_input(params, exception):

    with pytest.raises(exception):
        set_log(**params)


def test_set_log_no_handlers(caplog):
    caplog.clear()
    with caplog.at_level(LogLevel.WARNING, logger=logging_module.__name__):
        set_log(filename=None, stream=None)

    assert len(caplog.records) == 1
    assert caplog.record_tuples[0] == (logging_module.__name__, LogLevel.WARNING, "No CoSApp log handlers added.")
