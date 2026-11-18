#!/usr/bin/env bash
# End-to-end: tests -> train -> evaluate (+figures) -> notebooks -> app smoke test
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -q tests/test_pipeline.py
python -m src.train            # add --device mps on Apple Silicon
python -m src.evaluate
python run_notebooks.py all
python tests/test_app.py
echo "Done. Launch the dashboard with:  streamlit run app/streamlit_app.py"
