"""将 规划.md、教程.md 编译为 PDF（需 ai25 环境 pandoc + xelatex）。"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PANDOC_CANDIDATES = (
    Path(r"E:\Anaconda3\envs\ai25\Library\bin\pandoc.exe"),
    shutil.which("pandoc"),
)
HEADER = ROOT / "pandoc-pdf-header.tex"


def find_pandoc() -> str:
    for p in PANDOC_CANDIDATES:
        if p and Path(p).is_file():
            return str(p)
    raise FileNotFoundError("未找到 pandoc，请: conda install -n ai25 -c conda-forge pandoc")


def build(md: Path, pdf: Path, pandoc: str) -> None:
    cmd = [
        pandoc,
        str(md),
        "-o",
        str(pdf),
        "--pdf-engine=xelatex",
        "-f",
        "markdown+raw_html+tex_math_dollars",
        "-V",
        "CJKmainfont=SimSun",
        "-V",
        "geometry:margin=2cm",
        "-V",
        "documentclass=ctexart",
        "-V",
        "fontsize=11pt",
    ]
    if HEADER.is_file():
        cmd.extend(["--include-in-header", str(HEADER)])
    print(f"  -> {pdf.name}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    pandoc = find_pandoc()
    print(f"Using pandoc: {pandoc}")
    for name in ("规划.md", "教程.md"):
        md = ROOT / name
        if not md.is_file():
            print(f"MISSING: {md}", file=sys.stderr)
            return 1
        build(md, md.with_suffix(".pdf"), pandoc)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
