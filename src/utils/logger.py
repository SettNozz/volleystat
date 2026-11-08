"""
Structured colored logger for the volleyball analytics pipeline.

Usage:
    from src.utils.logger import get_logger
    log = get_logger("GameAnalyzer")
    log.info("Processing frame 42")
    log.debug("Ball at (100, 200)")
    log.warning("Ball not detected for 30 frames")
    log.error("Model file not found")
"""

import sys
import time

# Enable ANSI escape codes on Windows
if sys.platform == "win32":
    import os
    os.system("")

# ANSI codes
_R  = "\033[0m"      # reset
_GRAY   = "\033[90m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"

_LEVEL_CFG = {
    "DEBUG":   (_GRAY,   "DEBUG  "),
    "INFO":    (_GREEN,  "INFO   "),
    "WARNING": (_YELLOW, "WARN   "),
    "ERROR":   (_RED,    "ERROR  "),
}

_t0 = time.time()


def _print_line(line: str, *, flush: bool = False) -> None:
    stream = sys.stdout
    try:
        print(line, file=stream, flush=flush)
    except UnicodeEncodeError as exc:
        # Redirected Windows output can use a legacy code page. Keep logs
        # readable without letting unsupported characters abort analysis.
        encoding = getattr(stream, "encoding", None) or exc.encoding
        safe_line = line.encode(encoding, errors="backslashreplace").decode(encoding)
        print(safe_line, file=stream, flush=flush)


class Logger:
    def __init__(self, name: str):
        self.name = name

    def _log(self, level: str, msg: str):
        elapsed = time.time() - _t0
        color, label = _LEVEL_CFG[level]
        ts = f"{elapsed:7.1f}s"
        line = f"{_GRAY}{ts}{_R} {color}{label}{_R} {_CYAN}[{self.name}]{_R} {msg}"
        _print_line(line, flush=True)

    def debug(self, msg: str):   self._log("DEBUG",   msg)
    def info(self, msg: str):    self._log("INFO",    msg)
    def warning(self, msg: str): self._log("WARNING", msg)
    def error(self, msg: str):   self._log("ERROR",   msg)

    # Convenience: print a section header
    def section(self, title: str):
        sep = "─" * 50
        _print_line(f"\n{_BOLD}{sep}{_R}")
        _print_line(f"{_BOLD}  {title}{_R}")
        _print_line(f"{_BOLD}{sep}{_R}")


def get_logger(name: str) -> Logger:
    """Return a Logger for the given module/component name."""
    return Logger(name)
