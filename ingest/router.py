"""Which lane a file belongs to. Extension only.

Content sniffing is not built until a mislabeled file actually breaks something — at which
point the fix is a magic-byte check here and nowhere else.
"""

from pathlib import Path

LANES = {
    ".csv": "tabular",
    ".xlsx": "tabular",
    ".xlsm": "tabular",
    ".pdf": "document",
}


def lane_for(name: str) -> str:
    """'sales.xlsx' -> 'tabular'. Raises on anything we cannot read, rather than guessing."""
    suffix = Path(str(getattr(name, "name", name))).suffix.lower()
    if suffix not in LANES:
        raise ValueError(f"cannot read {suffix or 'a file with no extension'} — "
                         f"accepted: {', '.join(sorted(LANES))}")
    return LANES[suffix]
