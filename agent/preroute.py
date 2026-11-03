"""Tier P: the questions that must never reach a model.

"How many rows is this?" has an exact answer sitting in the Schema Card. Spending a model
call on it is the cheapest thing in the project to fix and the easiest to forget
(plans/datasurf-plan.md §6).

**This file is where a bug is most expensive.** Tier P is the only tier with no model and no
`verify` behind it, so a regex that matches too much answers confidently and wrongly for free.
Every pattern is therefore a `fullmatch` with an explicit tail — `how many rows` hits,
`how many rows have a null discount` misses and falls through to the planner. When in doubt,
miss: a miss costs a tenth of a cent, a false hit costs the user's trust.

The plan's "regex + embedding match" is half-built on purpose. The embedding half is the
semantic cache (`harness/cache.py`, M4.7) and needs `sentence-transformers`, which pulls
~2GB of torch and is deferred until the document lane needs it anyway.
"""

import re

# A trailing clause that adds no constraint. Anything else after the question is a
# constraint, and a constraint means this is not a tier-P question.
TAIL = (r"(?:\s+(?:are\s+there|do\s+(?:we|i)\s+have|does\s+it\s+have|in\s+total|total|here|"
        r"altogether|is\s+(?:it|this|that)|does\s+this\s+have))?\s*[?.!]*")
HOW_MANY = r"(?:how\s+many|what(?:'s|\s+is)?\s+the\s+(?:number|count)\s+of)\s+"
WHATS = r"(?:what(?:'s|\s+is)?\s+the\s+)?"

ROWS = re.compile(rf"\s*{HOW_MANY}(?:rows?|records?|entries|observations){TAIL}", re.I)
COLS = re.compile(rf"\s*{HOW_MANY}(?:columns?|fields|variables){TAIL}", re.I)
COL_NAMES = re.compile(rf"\s*(?:what|which)\s+(?:are\s+the\s+)?(?:columns?|fields|variables)"
                       rf"(?:\s+(?:names?|are\s+(?:there|available)|do\s+(?:we|i)\s+have))?"
                       rf"{TAIL}|\s*list\s+(?:the\s+|all\s+)?(?:columns?|fields|variables)"
                       rf"{TAIL}|\s*column\s+names?{TAIL}", re.I)
TABLES = re.compile(rf"\s*(?:what|which)\s+(?:tables?|sheets?|files?)(?:\s+(?:are\s+(?:there|"
                    rf"loaded|available)|do\s+(?:we|i)\s+have))?{TAIL}|"
                    rf"\s*list\s+(?:the\s+|all\s+)?(?:tables?|sheets?|files?){TAIL}|"
                    rf"\s*how\s+many\s+(?:tables?|sheets?|files?){TAIL}", re.I)
SHAPE = re.compile(rf"\s*{WHATS}(?:shape|dimensions|size)(?:\s+of\s+(?:the\s+)?"
                   rf"(?:data|table|sheet|file))?{TAIL}", re.I)

def answer(question: str, tables: dict) -> tuple[str, str] | None:
    """(answer, what it was read from), or None to fall through to the planner.

    `tables` is `Session.tables`. Everything here reads `.card`, never a DataFrame — the card
    already holds these numbers at full precision, which is the point of having built it.
    """
    if not tables:
        return None
    q = question.strip()

    if TABLES.fullmatch(q):
        named = ", ".join(f"`{n}`" for n in tables)
        return (f"{len(tables)} table{'s' if len(tables) != 1 else ''} loaded: {named}.",
                "the session inventory")

    # Everything below is about one table. With several loaded and none named, the honest move
    # is to fall through rather than silently pick the first.
    name, q = _which(q, tables)
    if name is None:
        return None
    card = tables[name].card
    source = f"the Schema Card for {name}"

    if ROWS.fullmatch(q):
        return f"`{name}` has {card.rows:,} rows.", source
    if COLS.fullmatch(q):
        return f"`{name}` has {len(card.columns)} columns.", source
    if SHAPE.fullmatch(q):
        return f"`{name}` is {card.rows:,} rows × {len(card.columns)} columns.", source
    if COL_NAMES.fullmatch(q):
        named = ", ".join(f"`{c.name}`" for c in card.columns)
        return f"`{name}` has {len(card.columns)} columns: {named}.", source
    return None


def _which(question: str, tables: dict) -> tuple[str | None, str]:
    """(the one table this question is about, the question with that reference removed).

    The reference has to come out before matching, or `how many rows in sales` reads as a
    constrained question and misses — the table name is not a constraint, it is an address.
    """
    named = [n for n in tables if re.search(rf"\b{re.escape(n)}\b", question, re.I)]
    if len(named) > 1:
        return None, question                      # two tables named: that is a real question
    if not named:
        return (next(iter(tables)) if len(tables) == 1 else None), question
    pattern = rf"\s*\b(?:in|of|for|from)?\s*{re.escape(named[0])}\b"
    return named[0], re.sub(pattern, "", question, flags=re.I)
