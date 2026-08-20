"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based

"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

import shutil
from os import environ
from pathlib import Path

from settings import config

DOIT_CONFIG = {"backend": "sqlite3", "dep_file": "./.doit-db.sqlite"}


BASE_DIR = config("BASE_DIR")
DATA_DIR = config("DATA_DIR")
MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
OS_TYPE = config("OS_TYPE")
# USER = config("USER")

## Helpers for handling Jupyter Notebook tasks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# Force UTF-8 file I/O. On Windows, latexmk resets the console code page to 437,
# after which the chartbook build fails reading UTF-8 sources with cp437.
environ["PYTHONUTF8"] = "1"

# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook_path):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
def jupyter_to_html(notebook_path, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} {notebook_path}"
def jupyter_to_md(notebook_path, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir={output_dir} {notebook_path}"
def jupyter_clear_output(notebook_path):
    """Clear the output of a notebook"""
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
# fmt: on


def mv(from_path, to_path):
    """Move a file to a folder"""
    from_path = Path(from_path)
    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)
    if OS_TYPE == "nix":
        command = f"mv {from_path} {to_path}"
    else:
        command = f"move {from_path} {to_path}"
    return command


def copy_file(origin_path, destination_path, mkdir=True):
    """Create a Python action for copying a file."""

    def _copy_file():
        origin = Path(origin_path)
        dest = Path(destination_path)
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)

    return _copy_file


##################################
## Begin rest of PyDoit tasks here
##################################


def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": ["python ./src/settings.py"],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": [],
    }


def task_pull():
    """Pull raw data from WRDS (requires WRDS_USERNAME in .env)."""
    yield {
        "name": "crsp_index",
        "doc": "Pull CRSP monthly market index (vwretd/vwretx) from WRDS",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_CRSP_index.py",
        ],
        "targets": [DATA_DIR / "crsp_msi.parquet"],
        "file_dep": ["./src/settings.py", "./src/pull_CRSP_index.py"],
        "clean": [],
    }
    yield {
        "name": "ff",
        "doc": "Pull Fama-French monthly factors (risk-free rate) from WRDS",
        "actions": [
            "python ./src/settings.py",
            "python ./src/pull_fama_french.py",
        ],
        "targets": [DATA_DIR / "ff_factors.parquet"],
        "file_dep": ["./src/settings.py", "./src/pull_fama_french.py"],
        "clean": [],
    }

def task_clean_data():
    """Build the tidy predictor panel from the raw pulls."""
    return {
        "actions": ["python ./src/calc_predictor_data.py"],
        "targets": [DATA_DIR / "predictor_panel.parquet"],
        "file_dep": [
            "./src/calc_predictor_data.py",
            DATA_DIR / "crsp_msi.parquet",
            DATA_DIR / "ff_factors.parquet",
        ],
        "clean": [],
    }

def task_table_01():
    """Build Table 1 (Parts A/B/C) as LaTeX."""
    return {
        "actions": ["python ./src/create_table_01.py"],
        "targets": [OUTPUT_DIR / "table_01.tex"],
        "file_dep": [
            "./src/create_table_01.py",
            "./src/create_table_01_partC.py",
            "./src/monte_carlo.py",
            DATA_DIR / "predictor_panel.parquet",
        ],
        "clean": True,
    }


def task_figure_01():
    """Build Figure 1 (beta vs rho by method and subsample)."""
    return {
        "actions": ["python ./src/create_figure_01.py"],
        "targets": [OUTPUT_DIR / "figure_01.png"],
        "file_dep": [
            "./src/create_figure_01.py",
            "./src/stambaugh_bias.py",
            "./src/create_table_01_partC.py",
            DATA_DIR / "predictor_panel.parquet",
        ],
        "clean": True,
    }

def task_table_02():
    """Build Table 2 (Bayesian posteriors, specs A-D) as LaTeX."""
    return {
        "actions": ["python ./src/create_table_02.py"],
        "targets": [OUTPUT_DIR / "table_02.tex"],
        "file_dep": [
            "./src/create_table_02.py",
            "./src/bayesian.py",
            "./src/create_table_01_partC.py",
            DATA_DIR / "predictor_panel.parquet",
        ],
        "clean": True,
    }

def task_table_01_updated():
    """Build Table 1 on samples extended through the most recent data."""
    return {
        "actions": ["python ./src/create_table_01.py --updated"],
        "targets": [OUTPUT_DIR / "table_01_updated.tex"],
        "file_dep": [
            "./src/create_table_01.py",
            "./src/create_table_01_partC.py",
            "./src/monte_carlo.py",
            DATA_DIR / "predictor_panel.parquet",
        ],
        "clean": True,
    }


def task_table_02_updated():
    """Build Table 2 on samples extended through the most recent data."""
    return {
        "actions": ["python ./src/create_table_02.py --updated"],
        "targets": [OUTPUT_DIR / "table_02_updated.tex"],
        "file_dep": [
            "./src/create_table_02.py",
            "./src/bayesian.py",
            "./src/create_table_01_partC.py",
            DATA_DIR / "predictor_panel.parquet",
        ],
        "clean": True,
    }

def task_update_comparison():
    """Build the paper-era vs updated-era comparison exhibit."""
    return {
        "actions": ["python ./src/create_update_comparison.py"],
        "targets": [OUTPUT_DIR / "update_comparison.tex"],
        "file_dep": [
            "./src/create_update_comparison.py",
            "./src/create_table_01_partC.py",
            "./src/monte_carlo.py",
            "./src/stambaugh_bias.py",
            DATA_DIR / "predictor_panel.parquet",
        ],
        "clean": True,
    }

def task_notebook():
    """Execute the walkthrough notebook, export HTML, and stage the executed
    .ipynb in OUTPUT_DIR for the chartbook site build."""
    nb_py = "./src/01_walkthrough.ipynb.py"
    nb = "./src/01_walkthrough.ipynb"

    def copy_executed_notebook():
        shutil.copy2(nb, OUTPUT_DIR / "01_walkthrough.ipynb")

    return {
        "actions": [
            f"jupytext --to notebook --output {nb} {nb_py}",
            f"jupyter nbconvert --execute --to notebook --inplace {nb}",
            f"jupyter nbconvert --to html --output-dir={OUTPUT_DIR} {nb}",
            copy_executed_notebook,
        ],
        "targets": [
            OUTPUT_DIR / "01_walkthrough.html",
            OUTPUT_DIR / "01_walkthrough.ipynb",
        ],
        "file_dep": [
            nb_py,
            DATA_DIR / "predictor_panel.parquet",
            "./src/create_table_01.py",
            "./src/create_table_02.py",
            "./src/create_figure_01.py",
        ],
        "clean": True,
    }
# def task_summary_stats():
#     """Generate summary statistics tables"""
#     file_dep = ["./src/example_table.py"]
#     file_output = [
#         "example_table.tex",
#         "pandas_to_latex_simple_table1.tex",
#     ]
#     targets = [OUTPUT_DIR / file for file in file_output]

#     return {
#         "actions": [
#             "python ./src/example_table.py",
#             "python ./src/pandas_to_latex_demo.py",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "clean": True,
#     }


# notebook_tasks = {
#     "01_example_notebook_interactive.ipynb.py": {
#         "path": "./src/01_example_notebook_interactive.ipynb.py",
#         "file_dep": [],
#         "targets": [],
#     },
# }


# fmt: off
# def task_run_notebooks():
#     """Preps the notebooks for presentation format.
#     Execute notebooks if the script version of it has been changed.
#     """
#     for notebook in notebook_tasks.keys():
#         pyfile_path = Path(notebook_tasks[notebook]["path"])
#         notebook_path = pyfile_path.with_suffix("")  # strips .py, leaves .ipynb
#         notebook_name = notebook_path.stem  # e.g. "01_example_notebook_interactive"
#         yield {
#             "name": notebook,
#             "actions": [
#                 """python -c "import sys; from datetime import datetime; print(f'Start """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
#                 f"jupytext --to notebook --output {notebook_path} {pyfile_path}",
#                 jupyter_execute_notebook(notebook_path),
#                 jupyter_to_html(notebook_path),
#                 mv(notebook_path, OUTPUT_DIR),
#                 """python -c "import sys; from datetime import datetime; print(f'End """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
#             ],
#             "file_dep": [
#                 pyfile_path,
#                 *notebook_tasks[notebook]["file_dep"],
#             ],
#             "targets": [
#                 OUTPUT_DIR / f"{notebook_name}.html",
#                 *notebook_tasks[notebook]["targets"],
#             ],
#             "clean": True,
#         }
# fmt: on

###############################################################
## Task below is for LaTeX compilation
###############################################################


def task_compile_latex_docs():
    """Compile the project report to PDF."""
    return {
        "actions": [
            "latexmk -pdf -halt-on-error -cd ./reports/report.tex",
            "latexmk -pdf -halt-on-error -c -cd ./reports/report.tex",
        ],
        "targets": ["./reports/report.pdf"],
        "file_dep": [
            "./reports/report.tex",
            "./reports/my_article_header.sty",
            "./reports/my_common_header.sty",
            OUTPUT_DIR / "table_01.tex",
            OUTPUT_DIR / "table_02.tex",
            OUTPUT_DIR / "table_01_updated.tex",
            OUTPUT_DIR / "table_02_updated.tex",
            OUTPUT_DIR / "figure_01.png",
        ],
        "clean": True,
    }


def task_build_chartbook_site():
    """Build the chartbook static site into ./docs for GitHub Pages."""

    def copy_static_assets():
        # Sphinx does not copy loose non-source files; ensure they ship.
        for f in ["playground.html", "report.pdf"]:
            src = Path("./docs_src/site") / f
            if src.exists():
                shutil.copy2(src, Path("./docs") / f)
        (Path("./docs") / ".nojekyll").touch()

    # Glob the site sources so adding a page triggers a rebuild. Listing files
    # individually meant new pages were silently skipped as up-to-date.
    site_sources = sorted(str(p) for p in Path("./docs_src").rglob("*.md"))

    return {
        "actions": [
            "chartbook build -f",
            copy_static_assets,
        ],
        "file_dep": [
            "chartbook.toml",
            *site_sources,
            "./docs_src/site/playground.html",
            OUTPUT_DIR / "01_walkthrough.ipynb",
        ],
        "targets": ["./docs/index.html"],
        "verbosity": 2,
    }

def task_run_pytest():
    """Run pytest and save results to OUTPUT_DIR"""
    src_py_files = list(Path("./src").glob("*.py"))
    test_output = OUTPUT_DIR / "pytest_results.xml"

    def run_pytest():
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "pytest", f"--junitxml={test_output}"],
        )
        if result.returncode != 0:
            # Remove the XML so doit won't consider the target up-to-date
            Path(test_output).unlink(missing_ok=True)
            raise RuntimeError(f"pytest failed with exit code {result.returncode}")

    return {
        "actions": [run_pytest],
        "targets": [test_output],
        "file_dep": src_py_files,
        "clean": True,
        "verbosity": 2,
    }
