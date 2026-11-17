#!/bin/bash
# Run the tests in tests/testthat on tiny fixtures (about 2 minutes).
# Usage: bash run_tests.sh [Rscript]
set -euo pipefail
R=${1:-Rscript}
cd "$(dirname "$0")"
"$R" -e 'testthat::test_dir("tests/testthat", reporter = "summary", stop_on_failure = TRUE)'
