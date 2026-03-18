import logging
import sys

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

def set_logger(level: str = "INFO") -> None:

    numeric_level = getattr(logging, level.upper())
    root = logging.getLogger("agentic_pandas")
    root.setLevel(numeric_level)

    if root.handlers:
        return
    
    # INFO or DEBUG handler -> stdout
    info_handler = logging.StreamHandler(sys.stdout)
    info_handler.setLevel(numeric_level)
    info_handler.addFilter(lambda f: f.levelno < logging.WARNING)
    info_handler.setFormatter(logging.Formatter("%(message)s"))

    # WARNING and above -> stderr
    warning_handler = logging.StreamHandler(sys.stderr)
    warning_handler.setLevel(logging.WARNING)
    warning_handler.setFormatter(logging.Formatter("%(levelname)-8s %(name)s — %(message)s"))

    root.addHandler(info_handler)
    root.addHandler(warning_handler)
