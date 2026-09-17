#!/usr/bin/env python3
"""Minimal 'percent format' (.py) → .ipynb dönüştürücü (jupytext yok).

`# %%`            → kod hücresi
`# %% [markdown]` → markdown hücresi (sonraki '# ' önekli satırlar gövde)

    python _py2nb.py kaynak.py [cikti.ipynb]
"""
import json
import sys
import uuid
from pathlib import Path


def convert(src: str) -> dict:
    lines = src.splitlines()
    cells, cur, kind = [], [], "code"

    def flush():
        if not cur:
            return
        body = "\n".join(cur).strip("\n")
        if not body.strip():
            return
        cid = uuid.uuid4().hex[:12]
        if kind == "markdown":
            txt = "\n".join(l[2:] if l.startswith("# ") else l[1:] if l == "#" else l
                            for l in body.splitlines())
            cells.append({"cell_type": "markdown", "id": cid, "metadata": {},
                          "source": txt.splitlines(keepends=True)})
        else:
            cells.append({"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None,
                          "outputs": [], "source": body.splitlines(keepends=True)})

    for line in lines:
        s = line.strip()
        if s.startswith("# %%"):
            flush()
            cur, kind = [], ("markdown" if "markdown" in s else "code")
        else:
            cur.append(line)
    flush()

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": ".venv (3.11.8)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.8"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    src_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else src_path.with_suffix(".ipynb")
    out_path.write_text(json.dumps(convert(src_path.read_text()), ensure_ascii=False, indent=1))
    print(f"→ {out_path}")
