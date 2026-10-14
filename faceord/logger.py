"""Copy console output of an experiment to logs/<YYYYmmdd_HHMMSS>_<tag>.txt.

    with LogSession("kfold_faceord"):
        run_cv(cfg)

stdout and stderr are both written to the terminal and to the log file; an
exception is logged with its traceback before it propagates.
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path


class _TeeWriter:
    """Stream wrapper that writes to the original stream and to a log file."""

    def __init__(self, original_stream, log_file):
        self._orig = original_stream
        self._file = log_file

    def write(self, text):
        self._orig.write(text)
        self._file.write(text)

    def flush(self):
        self._orig.flush()
        self._file.flush()

    def isatty(self):
        return self._orig.isatty()

    def __getattr__(self, attr):
        return getattr(self._orig, attr)


class LogSession:
    """Context manager that tees stdout/stderr into a timestamped log file."""

    def __init__(self, tag: str = "run", log_dir: str = "logs"):
        self.tag = "".join(c if c.isalnum() or c in "_-" else "_" for c in tag)
        self.log_dir = Path(log_dir)
        self.path = None
        self._file = self._stdout = self._stderr = None

    def __enter__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.log_dir / f"{stamp}_{self.tag}.txt"
        self._file = open(self.path, "w", encoding="utf-8", buffering=1)
        self._file.write(f"# {datetime.now():%Y-%m-%d %H:%M:%S}  {' '.join(sys.argv)}\n\n")
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeWriter(self._stdout, self._file)
        sys.stderr = _TeeWriter(self._stderr, self._file)
        print(f"[log] {self.path}")
        return self.path

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is not None:
            self._file.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        sys.stdout.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self._file.close()
        return False
