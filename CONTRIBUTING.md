# Contributing

Setup and code conventions. For design/rationale see
[ARCHITECTURE.md](ARCHITECTURE.md); for the task list see
[DEVELOPMENT.md](DEVELOPMENT.md).

if you work with agents:

1. Please no agent-based commits/pull-requests. Agent-based _code_ is fine:
   commits should be human-checked/branch-pushed. Pull-requests/merges
   fully human-only.
2. during your human-check, make sure single-commit changes stay
   overseeable: not too big, conceptually clear, clear naming.
3. Agent-based changes to documentation should be triple-checked by
   humans: they're often the most detrimental changes (imo).

## Setup

```bash
uv sync
uv run pytest -q
```

No pinned pass count here, the suite is actively changing — see
[DEVELOPMENT.md](DEVELOPMENT.md) for the current baseline. `nanoom[chem]`
(rdkit/bblean) comes in via the `test` dependency group, so plain `uv sync`
is enough to run everything.

## Reporting issues

Open a [GitHub issue](https://github.com/lucinamay/nanoom/issues) with the
exact command/snippet, the full traceback, what you expected, your OS and
Python version, and whether it reproduces from a clean `uv sync`. For
data-shape problems, the smallest polars frame that reproduces it beats
any description.

## Contributing code

1. Branch off `main`, snake_case name.
2. Commit in small steps, `PREFIX: description`:

   |       |                                               |
   | ----- | --------------------------------------------- |
   | `NEW` | new functionality                              |
   | `ADD` | tests, data, docs for existing functionality   |
   | `FIX` | bug fixes                                      |
   | `IMP` | clearer naming, structure, docs                |
   | `REF` | refactor, no behaviour change                  |
   | `DEL` | removals                                       |

3. Keep tests green, add tests for what you changed:

   ```bash
   uv run pytest -q
   uv run ruff check && uv run ruff format
   ```

   Ruff-format!!! Include import ordering.

4. Update the docs in the same commit: the module docstring if the
   module's job changed, [DEVELOPMENT.md](DEVELOPMENT.md) if you finished
   or invalidated a task, [ARCHITECTURE.md](ARCHITECTURE.md) if you moved
   a boundary or made a design choice worth recording.
5. Open a PR against `main` and tag the maintainer. Review is informal,
   but useful for repo-history reasons — state the rationale behind
   implementation choices.

## Tests

- New behaviour needs a test, especially bug fixes (test what failed
  before the fix).
- Offline and fast: no network, no downloaded weights.
- **never write into the repo/actual directories**, only to `tmp_path`.
- Name tests (_clearly_) after the behaviour, not the function:
  `test_split_raises_on_mismatched_cluster_length`.
- Clustering and splitting take explicit seeds/`random_state`; if you
  touch `clustering.py` or `splitting.py`, show the output is stable given
  the same seed.

## Coding principles: style / conventions

`currently` bullets are worked examples of how a principle is presently
interpreted, not the principle itself — if principle and example
disagree, the example changes.

**Reproducibility & Transparency** — someone should be able to rebuild a
split later without asking us anything.

- `pyproject.toml` and `uv.lock` committed and kept in sync.
- Explicit seeds/`random_state` for clustering and splitting; the LP
  solver in `splitting.py` runs with a fixed `n_jobs` in tests, for
  reproducibility.

**Conceptual clarity** — structure and naming should match how we think
about the problem.

- `nanoom.cluster` produces a `cluster_col`, `nanoom.split` consumes it — swap
  in your own cluster assignment without touching the splitter.
- Free functions over classes: transforms return new frames, no mutation to keep
  in mind while looking at code.

**Simplicity** — YAGNI. Also, if something's a shortcut for later, say so
explicitly rather than hiding it.

**Readability** — someone who didn't write it should be able to follow
it.

**Scientific best practices** — no silent fails or fallbacks.

- A wrong-length input to an audit function in `eval.py` raises instead
  of silently computing the wrong answer.
