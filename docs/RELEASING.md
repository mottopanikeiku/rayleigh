# Releasing

I tag releases from `main` after CI passes. The distribution name stays `eval-power` because the PyPI name `rayleigh` belongs to an unrelated project; releases are GitHub tags with attached build artifacts, not PyPI uploads.

1. Choose the version and set it in `pyproject.toml` and `CITATION.cff`. Add `date-released` to `CITATION.cff` and replace "(unreleased)" in `CHANGELOG.md` with the release date.
2. Refresh the lockfile and run every check from a clean sync:

   ```bash
   uv lock
   uv sync --locked
   uv run ruff check . && uv run ruff format --check .
   uv run pytest -q
   uv run python scripts/check_results.py
   uv run python site/export_fixture.py --check
   node --test site/*.test.mjs
   ```

3. Build and install the wheel outside the checkout, then run the CLI example:

   ```bash
   uv build --out-dir dist
   uv venv ../wheel-check
   uv pip install --python ../wheel-check dist/eval_power-*.whl
   example="$PWD/examples/gsm8k_qwen7_phi35_pilot.csv"
   (cd .. && wheel-check/bin/rayleigh --difference 0.05 --pilot "$example")
   ```

   The output must report `"bound_items_per_model": 787`.

4. Commit as "Release <version>", push, and wait for CI on `main`.
5. Tag and publish with the changelog section as notes:

   ```bash
   git tag -a v<version> -m "rayleigh <version>"
   git push origin v<version>
   gh release create v<version> dist/* --title "rayleigh <version>" --notes-file <notes.md>
   ```

6. Move "(unreleased)" to a new section in `CHANGELOG.md` for the next version.
