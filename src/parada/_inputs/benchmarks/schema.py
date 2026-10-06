from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

_WINDOWS_ABSOLUTE = re.compile("^[A-Za-z]:[/\\\\]")


def is_portable_sample_id(sample_id: str) -> bool:
    if not sample_id or "\\" in sample_id or _WINDOWS_ABSOLUTE.match(sample_id):
        return False
    path = PurePosixPath(sample_id)
    return not path.is_absolute() and ".." not in path.parts


@dataclass(frozen=True, slots=True)
class SampleEntry:
    sample_id: str
    label: int
    split_source: str
