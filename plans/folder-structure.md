# DataSurf — Folder Structure Plan

**2026-09-29** · companion to [modules.md](modules.md) and [datasurf-plan.md](datasurf-plan.md)

The rule that governs this whole document:

> **No directory is created before the module that fills it. No abstraction until its second caller
> exists.**

A tree this size looks like a lot of scaffolding. It isn't — M1 creates **six files**. Everything else
appears when the module that needs it starts. The `Module` column is the schedule; treat any file
outside your current module as not existing yet.

---

## The tree

```
datasurf/
├── app.py                      M1   streamlit entry: upload · tables panel · chat · presets · ledger
├── config.py                   M2   env, model id, budgets, paths — one place, read once
├── check_env.py                 ✓   venv smoke test (exists)
├── requirements.txt             ✓   (exists)
│
├── ingest/                     M1   file in → queryable table + card out. No LLM anywhere.
│   ├── router.py               M1   sniff type → tabular lane or document lane
│   ├── tabular.py              M1   read csv/xlsx · header detection · normalize · changelog
│   ├── profile.py              M1   Schema Card builder — highest-leverage file in the repo
│   ├── store.py                M1   session registry: duckdb connection + table names + handles
│   └── documents.py            M1   pdf → page-keyed raw text (M3 extends: chunk, embed)
│
├── agent/                      M2   the graph and everything that feeds a prompt
│   ├── llm.py                  M2   OpenRouter adapter — the only place a model name appears
│   ├── state.py                M2   pydantic State + tool arg/return models
│   ├── graph.py                M2   LangGraph wiring: nodes, conditional edges, checkpointer
│   ├── nodes.py                M2   pre_route · plan · clarify · act · verify · repair · synthesize
│   │                                no `route` node — the lane edge replaced it (plan §7)
│   ├── preroute.py             M2   deterministic tier-P matcher — zero model calls
│   ├── lanes.py                M2   lane edge + per-lane tool sets and verify rules
│   ├── context.py              M2   prompt assembly: cards, handles, rolling summary, cache breaks
│   └── prompts/                M2   markdown, not python string soup
│       ├── base.md             M2
│       ├── tabular.md          M2
│       └── document.md         M3
│
├── domains/                    M2   domain packs — NOT separate agents (see modules.md §M2)
│   ├── detect.py               M2   Schema Card → domain. Pure function, directly testable.
│   ├── generic.py              M2   the always-present fallback pack
│   ├── financial.py            M2   first real pack
│   ├── scientific.py           M2
│   ├── medical.py              M2   data-description scope only
│   └── legal.py                M2   data-description scope only
│
├── analysis/                   M2   deterministic statistics. No LLM. Each file = one preset.
│   ├── health.py               M2   nulls · dupes · type mismatches · constant cols · IQR outliers
│   ├── eda.py                  M2   distributions · cardinality · key/measure/dimension guess
│   ├── matrices.py             M2   correlation · covariance · contingency · overlap
│   ├── anomalies.py            M2   z-scores · per-segment deviation from group mean
│   ├── trends.py               M3*  resample · decompose · baseline forecast  (*date-column gated)
│   ├── segments.py             M2   metric diff with effect size AND sample size
│   └── joins.py                M2   candidate join keys by name, dtype, value overlap
│
├── documents/                  M3   unstructured lane
│   ├── chunk.py                M3   layout-aware chunking
│   ├── embed.py                M3   local MiniLM → float32 matrix
│   ├── index.py                M3   numpy cosine search; swap point for a real vector store
│   ├── card.py                 M3   Doc Card builder
│   ├── classify.py             M3   document type → selects the same domain pack
│   └── verdicts.py             M3   unprompted first-read findings
│
├── tools/                      M2   the MCP surface
│   ├── server.py               M2   FastMCP: every tool, one implementation
│   ├── client.py               M2   MCP client + per-specialist tool subsetting
│   ├── tabular.py              M2   list_tables · schema_card · column_stat · run_sql · describe · plot
│   └── documents.py            M3   search_docs · get_page · quote
│
├── harness/                    M4   but trace.py lands during M2.1 — see modules.md §6
│   ├── trace.py                M2!  JSONL per run, every node in/out  ← FIRST, not last
│   ├── ledger.py               M2!  tokens in/out/cached + cost, sidebar panel
│   ├── budget.py               M2   token · step · wall-clock ceilings
│   ├── loop.py                 M2   run loop enforcing the above
│   ├── cache.py                M4   semantic cache: question embedding → stored SQL
│   ├── replay.py               M4   re-run a recorded trace offline
│   ├── judge.py                M4   model scoring for narrative answers
│   └── knowledge.py            M4   knowledgebase read/write
│
├── evals/                      M4
│   ├── golden.yaml             M4   ~30 questions × 3 fixtures, expected values
│   ├── test_golden.py          M4   runs across luna · nemotron-super · deepseek-flash
│   ├── test_ingest.py          M1   header detection + normalization against the nasty fixture
│   ├── test_preroute.py        M2   tier-P matcher: pure, fast, no network
│   ├── test_lanes.py           M2   lane selection from session inventory — must stay token-free
│   └── test_detect.py          M2   domain detection from Schema Cards
│
├── data/fixtures/              M1   committed — .gitignore fixed, see below
│   ├── messy_sales.xlsx        M1   headers row 4 · merged cells · totals row · serial dates
│   ├── reps.csv                M1   join target for M2.4
│   └── report.pdf              M3   claims that agree and disagree with the workbook
│
└── plans/                       ✓
    ├── datasurf-plan.md         ✓   architecture, diagrams, roadmap, risks
    ├── modules.md               ✓   M1–M4, milestones, build order
    └── folder-structure.md      ✓   this document
```

---

## What exists at the end of each module

| After | Files | Runnable proof |
|---|---:|---|
| **M1** | 6 + 3 fixtures | `streamlit run app.py` → upload `messy_sales.xlsx` → correct Schema Card, no API key set |
| **M2** | ~28 | Free-form question → verified answer + visible SQL; broken column self-repairs; "how many rows" costs 0 calls |
| **M3** | ~36 | Hybrid question → DuckDB number + PDF quote + real page number |
| **M4** | ~42 | Two README tables: naive vs optimized, and one harness across three models |

Six files for a working, demoable module 1. That is the point of the ordering.

---

## Three structural decisions worth defending

**`analysis/` holds computation and presets together.** A separate `presets/` directory would be one
wrapper file per analysis file, each forwarding a single call. One directory; each file computes and
exposes one preset entry point. Split it only if a preset ever composes three or more analyses.

**`domains/` is data and prompt fragments, not classes.** A pack is a module-level dict plus a
`detect_hints` list. No `BaseDomain` ABC with one method until there is a second thing that needs the
abstraction.

**`harness/trace.py` and `ledger.py` live in the M4 directory but are written during M2.1.** The
directory groups by concern; the `Module` column schedules by need. Writing the ledger last means
having no baseline to measure optimizations against — the one sequencing mistake that cannot be
repaired afterwards.

---

## The .gitignore fixture rule — applied 2026-09-29

`data/fixtures/` is now committable while everything else under `data/` stays ignored. The fix was
**not** the one first sketched here: keeping `data/` and adding negations under it does not work,
because git will not descend into an excluded *directory* — so no negation can rescue a file whose
parent is excluded. It had to become `data/*`.

```gitignore
# Data - user uploads & generated output
# `data/*` not `data/`: git will not descend into an excluded directory, so a
# negation can never rescue a file whose parent directory is excluded.
data/*
uploads/
outputs/
*.csv
*.xlsx
*.xls
*.xlsb
*.ods
*.parquet
*.pdf
*.docx

# ...but the synthetic test fixtures ship with the repo — the evals must be
# reproducible, and committed fixtures keep real data away from a third-party
# endpoint. These negations must come after the patterns they override.
!data/fixtures/
!data/fixtures/**
```

Two rules are load-bearing and easy to get wrong: **a negation must come after the patterns it
overrides**, and **it cannot rescue a file under an excluded directory**. The `*.csv` / `*.xlsx` /
`*.pdf` globs sit above the negations precisely so `!data/fixtures/**` wins for paths inside fixtures
while those globs still catch the same extensions everywhere else.

Verified with `git check-ignore`:

| Path | Expected | Result |
|---|---|---|
| `data/fixtures/messy_sales.xlsx` | tracked | ✓ |
| `data/fixtures/reps.csv` | tracked | ✓ |
| `data/fixtures/report.pdf` | tracked | ✓ |
| `data/uploads/user_upload.xlsx` | ignored | ✓ |
| `data/outputs/export.csv` | ignored | ✓ |
| `data/scratch.csv` | ignored | ✓ |
| `uploads/someones.csv` | ignored | ✓ |
| `root_level.csv` | ignored | ✓ |
| `.env` | ignored | ✓ |

Re-run that check if the fixture set changes:

```bash
git check-ignore -v data/fixtures/messy_sales.xlsx   # no output = tracked, which is correct
git add -An data/                                    # should list only data/fixtures/*
```

---

## Naming conventions

- **Modules are nouns, functions are verbs.** `profile.py` exposes `build_schema_card()`.
- **No `utils.py`, no `helpers.py`, no `common.py`.** A function whose home is unclear belongs next to
  its only caller until a second one appears.
- **`tools/` mirrors the lanes.** `tools/tabular.py` ↔ `ingest/tabular.py` ↔ `analysis/`;
  `tools/documents.py` ↔ `documents/`.
- **Tests are named for what breaks**, not for the file under test: `test_preroute.py` protects the
  zero-token path, and its failure means money is being spent that shouldn't be.
