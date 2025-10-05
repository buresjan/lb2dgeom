# Installation Guide

This document describes two supported workflows for installing **lb2dgeom** on a
local machine:

- A reproducible Conda environment that captures Python and binary
  dependencies.
- A lightweight virtual environment using the Python standard library.

Both paths install the package in editable mode so source edits take effect
immediately—ideal for development and experimentation.

---

## 1. Prerequisites

- **Operating system**: Linux, macOS, or Windows.
- **Python**: Version 3.9 or newer.
- **Build tools**: A C compiler is *not* required—the project depends only on
  pure-Python wheels.
- **Git**: Optional but recommended for cloning the repository.

If you plan to use the Conda workflow, install either Anaconda or Miniconda.
For the native virtual environment path, ensure `python` and `pip` are on your
`PATH`.

---

## 2. Clone the repository

```bash
git clone <repo-url>
cd lb2dgeom
```

If you already have a local checkout, update it instead:

```bash
git pull
```

---

## 3. Option A – Conda environment (recommended for new users)

1. Create the environment from the provided specification:
   ```bash
   conda env create -f environment.yml
   ```
2. Activate it:
   ```bash
   conda activate lb2dgeom
   ```
3. Confirm the editable installation succeeded:
   ```bash
   python -c "import lb2dgeom; print('lb2dgeom imported:', lb2dgeom.Grid)"
   ```
4. (Optional) update the environment after pulling new changes:
   ```bash
   conda env update -f environment.yml --prune
   ```

*What you get*: Python 3.10, `numpy`, `matplotlib`, `pytest`, and an editable
install of `lb2dgeom`—all isolated from your base environment.

---

## 4. Option B – Native Python virtual environment

Use this path if you prefer to manage dependencies with the standard library or
`pipx`.

1. Ensure `pip` is current:
   ```bash
   python -m pip install --upgrade pip
   ```
2. Create and activate a virtual environment (example uses `.venv`):
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # macOS/Linux
   # .venv\Scripts\activate.bat     # Windows (cmd)
   ```
3. Install runtime dependencies and the package in editable mode:
   ```bash
   pip install -e .
   ```
4. (Optional) install development tooling (formatting, linting, tests):
   ```bash
   pip install pytest black mypy matplotlib numpy
   ```
5. Verify the import:
   ```bash
   python -c "import lb2dgeom; print('lb2dgeom imported:', lb2dgeom.Grid)"
   ```

Deactivate the virtual environment with `deactivate` when you are done.

---

## 5. Sanity checks

Run a quick test suite to confirm everything is wired correctly:

```bash
pytest -q
```

You can also execute an example script and inspect generated outputs:

```bash
python examples/demo_ellipse.py --out-dir examples/output
ls examples/output
```

---

## 6. Upgrading or reinstalling

- **Conda environment**: `conda env update -f environment.yml --prune`
- **Virtual environment**: Re-run `pip install -e .` after pulling changes.
  Dependencies are minimal, so updates are typically fast.

If you encounter conflicts, consider wiping the environment and recreating it.

---

## 7. Troubleshooting tips

- Ensure the active Python interpreter matches the environment you intend to
  use (`which python` or `where python`).
- If `matplotlib` fails to import due to backend issues, set the Agg backend
  before running scripts:
  ```bash
  export MPLBACKEND=Agg
  ```
- For Windows users encountering "Scripts is not on PATH" warnings, either the
  activation step was skipped or PowerShell execution policy is blocking
  activation. Launch a new shell, re-run the activation command, and try again.

With either workflow configured, you can modify the source under `src/` and rerun
examples or tests without reinstalling.
