#!/usr/bin/env bash
# Regenerate the HTML API documentation and the pytest test report.
#
# Usage (from the repository root):
#   python -m pip install pytest pytest-html pytest-cov pdoc numpy pandas scikit-learn
#   bash scripts/build_docs_and_test_report.sh
#
# Outputs
#   docs/                         HTML documentation of src/cmapss (pdoc)
#   tests/report/test_report.html Self-contained pytest report (open in a browser)
#   tests/report/test_report.xml  JUnit XML (machine-readable)
#   tests/report/test_report.txt  Plain-text log with versions and coverage
#   tests/report/coverage/        HTML line-coverage report for src/cmapss
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p tests/report

# 1) Tests: HTML + XML reports, and a text log that records the environment.
{
  echo "Test report generated: $(date -u '+%Y-%m-%d %H:%M UTC')"
  echo "Python $(python --version 2>&1 | cut -d' ' -f2) | $(python -c 'import sklearn,pandas,numpy,pytest;print(f"scikit-learn {sklearn.__version__} | pandas {pandas.__version__} | numpy {numpy.__version__} | pytest {pytest.__version__}")')"
  echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
  echo
  python -m pytest -v \
    --cov=src/cmapss --cov-report=term-missing --cov-report=html:tests/report/coverage \
    --html=tests/report/test_report.html --self-contained-html \
    --junitxml=tests/report/test_report.xml
} 2>&1 | tee tests/report/test_report.txt

# 2) API documentation (HTML) generated from the docstrings.
python -m pdoc -o docs --docformat numpy -d numpy src/cmapss
echo "Docs written to docs/ (open docs/index.html)"
