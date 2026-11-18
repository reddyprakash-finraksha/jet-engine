"""Execute notebooks in place:  python run_notebooks.py [01|02|all]"""
import sys, nbformat
from nbclient import NotebookClient
which = sys.argv[1] if len(sys.argv) > 1 else "all"
for f in ["notebooks/01_eda_preprocessing.ipynb", "notebooks/02_train_evaluate.ipynb"]:
    if which != "all" and which not in f:
        continue
    nb = nbformat.read(f, 4)
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": "notebooks"}}).execute()
    nbformat.write(nb, f); print("executed", f)
