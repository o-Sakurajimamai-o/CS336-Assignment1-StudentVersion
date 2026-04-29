import builtins
from pathlib import Path


_ORIGINAL_OPEN = builtins.open
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _open_with_utf8_fixtures(file, mode="r", *args, **kwargs):
    if "b" not in mode and kwargs.get("encoding") is None:
        try:
            path = Path(file).resolve(strict=False)
        except TypeError:
            path = None

        if path is not None and _FIXTURES_DIR in path.parents:
            kwargs["encoding"] = "utf-8"

    return _ORIGINAL_OPEN(file, mode, *args, **kwargs)


builtins.open = _open_with_utf8_fixtures
