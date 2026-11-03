"""The only place a model is called, and the only place `usage` is read.

One OpenAI-compatible client pointed at OpenRouter (plans/datasurf-plan.md §4): not
`langchain-openai`, and not provider calls scattered through the nodes. That is what makes
M4.8's three-model comparison an env var instead of a refactor.

`usage: {include: true}` is the load-bearing request option — without it OpenRouter returns
token counts but no charge, and the ledger's whole job is to report the real number.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel

import config
from harness.trace import Usage

# ```sql … ``` — models fence code even when told not to. Cheaper to strip than to re-ask.
_FENCE = re.compile(r"\A```[a-zA-Z]*\s*|\s*```\Z")


class NoKey(RuntimeError):
    """Raised instead of calling out with an empty Authorization header."""


@dataclass
class Reply:
    text: str
    usage: Usage


@lru_cache(maxsize=1)
def client() -> OpenAI:
    if not config.API_KEY:
        raise NoKey("OPENROUTER_API_KEY is not set — copy .env.example to .env and add a key "
                    "from https://openrouter.ai/keys")
    return OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL,
                  default_headers={"X-Title": "DataSurf"})


def chat(system: str, user: str, *, model: str | None = None, max_tokens: int = 700) -> Reply:
    response = client().chat.completions.create(
        model=model or config.MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,                       # SQL for the same question should not vary
        max_tokens=max_tokens,
        extra_body={"usage": {"include": True}},
    )
    return Reply((response.choices[0].message.content or "").strip(), usage(response))


def decide(system: str, user: str, schema: type[BaseModel]) -> tuple[BaseModel, Usage]:
    """A typed reply, via native structured outputs.

    Not a JSON-shaped request plus a parse-repair loop: `gpt-6-luna` constrains generation to
    the schema, so a malformed plan is impossible rather than retried. This is also why the
    free fallback is `nemotron-3-super` and not `nemotron-3-ultra` — the larger model has tool
    calling but no structured outputs, which would put the parse-repair loop back
    (plans/datasurf-plan.md §3).
    """
    response = client().chat.completions.parse(
        model=config.MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=schema,
        temperature=0,
        extra_body={"usage": {"include": True}},
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:                   # refusal, or a model that ignored the schema
        raise RuntimeError(f"{config.MODEL} returned no parsable "
                           f"{schema.__name__}: {response.choices[0].message.content!r}")
    return parsed, usage(response)


def usage(response) -> Usage:
    """OpenRouter's extra fields ride along on the usage object; the SDK keeps unknown keys."""
    raw = getattr(response, "usage", None)
    if raw is None:
        return Usage()
    details = getattr(raw, "prompt_tokens_details", None)
    return Usage(
        tok_in=raw.prompt_tokens or 0,
        tok_out=raw.completion_tokens or 0,
        cached=getattr(details, "cached_tokens", 0) or 0,
        usd=getattr(raw, "cost", None),
    )


def unfence(text: str) -> str:
    return _FENCE.sub("", text.strip())
