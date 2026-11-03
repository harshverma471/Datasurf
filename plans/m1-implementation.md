# M1 — UI and Ingestion: implementation record

**2026-10-02** · what was built, what was decided, and what broke along the way ·
companion to [modules.md](modules.md), [folder-structure.md](folder-structure.md) and
[datasurf-plan.md](datasurf-plan.md)

Those three documents say what to build and why. This one says what exists.

---

## Status: complete

All seven milestones ship. The done-check passes: upload a workbook with headers on row 4,
merged cells, a stray totals row and dates as serial integers, and the app shows a correct
Schema Card without hand-holding — with no model call anywhere in the path.

| | Milestone | Where it lives |
|---|---|---|
| M1.1 | Streamlit shell | `app.py` |
| M1.2 | Tabular read, every sheet | `ingest/tabular.py` — `read_raw` |
| M1.3 | Header detection + override | `ingest/tabular.py` — `detect_header` · `app.py` — `header_panel` |
| M1.4 | Normalization + changelog | `ingest/tabular.py` — `normalize` |
| M1.5 | Schema Card | `ingest/profile.py` |
| M1.6 | DuckDB registry | `ingest/store.py` — `Session.add_table` |
| M1.7 | PDF intake | `ingest/documents.py` |

24 tests pass in `evals/test_ingest.py`.

---

## What shipped

```
app.py                          344   upload · header override · changelog · columns · card
ingest/tabular.py               255   read · detect_header · normalize        ← the hard part
ingest/profile.py               252   Schema Card: per-column stats, notes, findings
ingest/store.py                  76   Session: DuckDB connection, raw frames, tables, documents
ingest/router.py                 23   extension → tabular | document
ingest/documents.py              19   pdf → page-keyed text
evals/test_ingest.py            229   24 tests, including the no-LLM guard
data/fixtures/make_fixtures.py  125   writes the deliberately awful workbook
```

Plus two committed fixtures: `messy_sales.xlsx` and `reps.csv`.

The folder plan predicted six files. Eight shipped: the two extras are the test file and the
fixture generator, both of which that plan lists under M1 anyway.

---

## The one architectural decision

**Read the file once into raw frames; derive everything else from those.**

```
file ──► raw frames (header=None, cached on Session)
             │
             ├─ detect_header   ──► row index + margin
             ├─ normalize       ──► clean frame + changelog
             ├─ build_schema_card
             └─ con.register
```

Changing the header row re-derives from the cached frames and never reopens the file. This
makes the override instant and keeps Streamlit's top-to-bottom rerun cheap. It also forces
`read_raw` to decide nothing, which is why `header=None` appears on every read.

`Session` is Seam 2 from the main plan §4a. One object owns the DuckDB connection, the raw
frames, the tables, the documents and the header overrides. It lives in `st.session_state`,
never `@st.cache_data` — it holds a live DuckDB connection, which is neither hashable nor
picklable.

---

## How a file travels

1. `router.lane_for(name)` sorts the upload by extension into `tabular` or `document`.
2. `tabular.read_raw(file)` returns every sheet unparsed. CSVs are read with `dtype=str` so
   both lanes reach the detector looking identical.
3. `tabular.detect_header(raw)` scores the first twenty rows and returns the winner plus its
   margin over the runner-up.
4. `tabular.normalize(raw, header_row)` returns a clean frame and a changelog.
5. `profile.build_schema_card(...)` turns 240 rows into about 277 tokens.
6. `Session.add_table(...)` registers the frame in DuckDB and records the card together,
   because those two must never drift apart.

Header detection scores four signals and takes the argmax:

| Signal | Weight | Catches |
|---|---|---|
| fill — non-null share of the row | 0.30 | blank spacer rows |
| stringiness — cells that are not numbers | 0.25 | data rows |
| uniqueness — distinct / non-null | 0.20 | a repeated banner |
| type shift — text here, numbers in the six rows below | 0.25 | the header→data boundary |

The last three signals are multiplied by fill a second time, which zeroes out blank spacer
rows: rows 1 and 3 of the demo workbook both score 0.000.

Two signals do the discriminating that matters. **Uniqueness** separates a title banner from a
header. The banner forward-fills to seven copies of one string and scores 0.143 where the real
header scores 1.000, and that gap is most of the distance between them. **Type shift** is the
only signal that looks at the relationship between a row and what follows it, which is the
evidence that actually locates a header: `Revenue ($)` sitting above `$94,897.17`.

Scored against the demo workbook:

| Row | Score | What it is |
|---|---:|---|
| 4 | **0.929** | the real header |
| 2 | 0.757 | `CONFIDENTIAL` — the runner-up, hence a margin of 0.17 |
| 0 | 0.721 | the merged title banner |
| 5–9 | 0.571 | data rows |
| 1, 3 | 0.000 | blank spacers |

---

## Decisions taken during the build

**Every sheet registers; the picker only chooses the view.** A three-sheet workbook becomes
three DuckDB tables. M2's join work needs them all present, and registering a frame DuckDB
already holds costs nothing.

**Serial-date conversion is gated on the column name *and* the value range.** Range alone
would destroy an ID column whose numbers land between 20,000 and 60,000. Name alone would
mangle a `year` column of 2023s. Requiring both makes the conversion safe to run unattended.

**An identifier is named like one or is 95% unique.** Cardinality alone called `units` an
identifier — 195 distinct values in 240 rows — and printed a distinct count where μ and σ
belong.

**A join key must be near-unique on the target side.** `region` matched `reps.territory` at
100% value overlap and is not a key at all.

**Header overrides live on `Session`, not in widget state.** Streamlit discards state for
widgets that stop rendering, and only the selected table renders a header input, so switching
tables would have silently lost the override.

**The card distinguishes a guess from an answer.** It reads `header detected row 4` or
`header set by user to row 4`. M2 reads this card, and a guess deserves less trust than a
human's answer.

**Findings carry two renderings.** `Finding.text` is terse, for the card, where tokens are
budgeted. `Finding.human` is a sentence, for the screen. The card says
`discount_pct nulls cluster in region=East (97% of them)`; the UI says why that matters.

---

## Bugs found and fixed

Each of these shipped broken and was caught by running the thing, not by reading it.

**`revenue` stayed a string.** The coercion gate read `s.dtype == object`, but pandas 3 gives
pure-string columns a `str` dtype. Currency parsing silently never ran. Fixed by testing for
"neither numeric nor datetime" instead.

**Merged header cells defeated the detector.** A merged cell arrives as value-then-`NaN`, so
the real header looked emptier than the data below it, and on a four-column sheet the data row
won. The margin honestly reported 0.0 while the answer was wrong. `_score` now forward-fills
the candidate row, scoring it the way `normalize` will read it.

**The sidebar reported the previous run.** `sidebar()` drew the tables panel before `rebuild()`
had created the tables, so a fresh upload showed "Nothing registered yet" beside a fully
rendered Schema Card. Split into `upload_panel()` and `tables_panel()`, called either side of
`rebuild`.

**Stale cards crashed the app after an edit.** Streamlit evicts a changed module from
`sys.modules` and re-imports it, which makes `profile.SchemaCard` a new class object — but
`Session` survives in `st.session_state`, still holding instances of the old one. `rebuild`
returned early because the header row had not changed, so new code met old objects and raised
`AttributeError`. The staleness check now includes
`isinstance(t.card, profile.SchemaCard)`, which is `False` across a reload: exactly the signal
that these were derived by code that no longer exists.

**The changelog claimed strips that never happened.** `rep_id` was logged as "stripped currency,
separators and units" when nothing was stripped. `_to_numeric` now reports whether it removed
anything, and the changelog says "parsed as numbers" instead.

**A CSV reported a sheet it does not have.** `read_raw` keys a CSV by its filename stem, and
the card printed `sheet "reps"`. The UI now passes a sheet only when the source has more than
one.

---

## Verification

```bash
.venv/bin/python data/fixtures/make_fixtures.py     # regenerate both fixtures
.venv/bin/pytest evals/test_ingest.py -q            # 24 tests
.venv/bin/python check_env.py                       # unchanged
.venv/bin/streamlit run app.py
```

By hand, with no `.env` file and no API key set:

1. Upload `messy_sales.xlsx`. The header reports row 4. `order_date` shows dates, not `45543`.
   `revenue` is a float. The `TOTAL` row is gone. `discount_pct` carries a clustered-missingness
   note.
2. Change the header row to 2. Everything re-derives to 241 rows without reopening the file.
   Change it back for 240.
3. Add `reps.csv`. The candidate-key note appears on `rep_id`.
4. Upload any PDF. Page count and a text preview appear, and nothing else.

The last test in the suite is the one that matters most: it imports every module under
`ingest/` and asserts that no `openai`, `langchain`, `langgraph`, `fastmcp` or `anthropic` key
appeared in `sys.modules`. Planting `import openai` in `router.py` fails it. M1 is defined by
spending zero model calls, and that deserves a test rather than a promise.

---

## Known gaps

**No committed PDF fixture.** pdfplumber reads but cannot write, and adding `reportlab` for one
test file fails the dependency test in `requirements.txt`. M1.7 was verified against a
throwaway PDF. `data/fixtures/report.pdf` arrives with M3, which needs a document carrying real
claims anyway. Until then the tabular path has regression tests and the document path does not.

**The detection margin narrowed.** Forward-filling the candidate row also fills the title
banner, which raises its score. The messy sheet's margin fell from 0.357 to 0.171. Every
fixture still detects correctly, and low uniqueness across a filled banner is what keeps it
losing — but that is the number to watch when a future sheet detects wrong.

**Two defensive leftovers.** The deferred import in `profile.py` and the `card: object`
annotation in `store.py` both guard against an import cycle that does not exist. Among the four
`ingest` modules there is exactly one dependency edge: `profile → tabular`, for `renames`.
Neither leftover costs anything, and `card: object` does keep `store` independent of `profile`.

---

## What M2 inherits

The contract from [modules.md](modules.md) §8 holds:

- **M2 gets a registered DuckDB table and a `SchemaCard`.** It never reads a file. That is why
  `Session` owns the connection and `add_table` does both jobs in one call.
- **M3 gets `{page_no: text}`.** It never opens a PDF. Keying by page number from the first
  line is what makes citation possible without a migration.
- **The lane edge is free.** `router.lane_for` sorts every upload at ingest time, so the first
  conditional edge in M2's graph reads the session inventory rather than spending a model call.

One change to anticipate: `SchemaCard.to_text()` is the string M2 will cache as a prompt
prefix. Its format is fixed in `profile.py` and rendered verbatim by the UI. Change it there,
never in `app.py`.
