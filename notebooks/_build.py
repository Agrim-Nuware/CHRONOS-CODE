import json


def write_notebook(cells, path):
    """cells: list of (cell_type, source_string) tuples."""
    nb_cells = []
    for cell_type, source in cells:
        lines = source.splitlines(keepends=True)
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": lines,
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)

    notebook = {
        "cells": nb_cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"wrote {path} ({len(nb_cells)} cells)")


def md(text):
    return ("markdown", text)


def code(text):
    return ("code", text)
