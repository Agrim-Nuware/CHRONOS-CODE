"""Headless-execute a notebook's code cells in order, for smoke-testing without Colab/Jupyter."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nb_path = sys.argv[1]
os.chdir(os.path.dirname(os.path.abspath(nb_path)))

with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

namespace = {"__name__": "__main__"}
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    if source.strip().startswith("!"):
        print(f"[cell {i}] skipping shell command: {source.strip()[:60]}")
        continue
    try:
        exec(compile(source, f"<cell {i}>", "exec"), namespace)
    except Exception:
        print(f"FAILED at cell {i}:\n{source}")
        raise
    plt.close("all")

print("ALL CELLS EXECUTED SUCCESSFULLY")
