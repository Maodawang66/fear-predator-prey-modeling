"""读取 Peacor 等 (2022) 元分析数据，提取 PLP 研究表供 φ 先验分析。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent / "raw"

DEFAULT_XLSX = RAW / "12_peacor_risk_meta" / "PeacorEtAl_Data_ELE-00137-2022.xlsx"
DEFAULT_CSV = RAW / "12_peacor_risk_meta" / "PLP_studies.csv"
PLP_SHEET = "PLP studies"


def _is_valid_xlsx(path: Path) -> bool:
    if not path.is_file():
        return False
    head = path.read_bytes()[:64]
    return head.startswith(b"PK") and not head.lstrip().startswith(b"<!")


def load_peacor_plp(path: str | Path | None = None) -> pd.DataFrame:
    """加载「PLP studies」表（64 篇 TMIE/NCE 分类清单）。"""
    if path is not None:
        p = Path(path)
        if p.suffix.lower() in (".csv", ".txt"):
            return pd.read_csv(p)
        return pd.read_excel(p, sheet_name=PLP_SHEET, engine="openpyxl")

    if _is_valid_xlsx(DEFAULT_XLSX):
        return pd.read_excel(DEFAULT_XLSX, sheet_name=PLP_SHEET, engine="openpyxl")
    if DEFAULT_CSV.is_file():
        return pd.read_csv(DEFAULT_CSV)

    raise FileNotFoundError(
        f"未找到 Peacor 数据：{DEFAULT_XLSX} 或 {DEFAULT_CSV}\n"
        "请从 Dryad doi:10.5061/dryad.ffbg79cxt 下载 xlsx，"
        "或运行 python data/deep_data_analysis.py（需已有 PLP_studies.csv）。"
    )


def list_peacor_sheets(path: str | Path | None = None) -> list[str]:
    xlsx = Path(path) if path else DEFAULT_XLSX
    if not _is_valid_xlsx(xlsx):
        return ["PLP studies (CSV fallback)"]
    return pd.ExcelFile(xlsx, engine="openpyxl").sheet_names


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    print("sheets:", list_peacor_sheets())
    df = load_peacor_plp()
    print("PLP studies:", df.shape)
    print(df.columns.tolist()[:8])
    if "Predation effect" in df.columns:
        print(df["Predation effect"].value_counts())
