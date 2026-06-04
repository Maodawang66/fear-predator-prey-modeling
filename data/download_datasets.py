"""
下载「组合拳」推荐数据集到 data/raw/ 目录。

用法: python data/download_datasets.py

说明:
- Dryad: 先访问数据集页面获取 cookie，再通过 file_stream 下载（API zip 需授权会失败）。
- GPDD / FoRAGE 全库体积大：下载说明 + 可获取的 README/元数据；全量请按说明手动拉取。
- BioTIME: 下载 Zenodo 元数据与子集（非 1GB+ SQL）。
"""

from __future__ import annotations

import json
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

try:
    import requests
except ImportError:
    raise SystemExit("请先安装 requests: pip install requests")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

DRYAD_DOIS = {
    "01_lynx_roe_deer": "10.5061/dryad.9zw3r22mq",
    "02_killifish_mosquitofish": "10.5061/dryad.qjq2bvqpw",
    "04_lake_michigan_zooplankton": "10.5061/dryad.bh688ft",
    "08_coral_reef_fear": "10.5061/dryad.vdncjsxtf",
    "09_landscape_of_fear": "10.5061/dryad.1rn8pk0x6",
    "10_damselfly_predator_cues": "10.5061/dryad.bg79cnpj2",
    "12_peacor_risk_meta": "10.5061/dryad.ffbg79cxt",
}

DIRECT = {
    "03_hudson_bay_lynx_hare": (
        "https://raw.githubusercontent.com/bblais/Systems-Modeling-Spring-2015-Notebooks/"
        "master/data/Lynx%20and%20Hare%20Data/lynxhare.csv"
    ),
    "05_gpdd_readme": "https://raw.githubusercontent.com/boettiger-lab/gpdd/master/README.md",
}

ZENODO_BIOTIME = "10932823"
BIOTIME_FILES = [
    "BIB_biotime_v2_15April25.csv.txt",
    "references_biotime_v2_15April25.csv",
    "biotime_v2_metadata_15April25.csv",
]

LTER_WI_FISH_CSV = (
    "https://pasta.lternet.edu/package/data/eml/knb-lter-ntl/356/4/"
    "829ef0e4eea5e6392b19e595aa775832"
)

FORAGE_KNB = "https://knb.ecoinformatics.org/knb/d1/mn/v2/resolve/doi:10.5063/F17H1GTQ"

_LYNX_HARE_CSV = """year,hare,lynx
1845,30,4
1846,8,5
1847,103,9
1848,784,6
1849,1870,10
1850,4245,8
1851,8160,16
1852,15735,26
1853,25320,34
1854,49759,43
1855,76853,39
1856,78824,30
1857,94379,20
1858,102531,8
1859,97150,3
1860,76750,4
1861,53500,8
1862,26750,9
1863,12400,6
1864,6500,5
1865,3950,7
1866,2250,12
1867,1800,16
1868,3500,28
1869,9150,45
1870,25750,71
1871,58000,100
1872,70000,27
1873,77400,17
1874,63000,12
1875,44300,7
1876,27750,5
1877,14200,7
1878,8100,12
1879,5500,9
1880,3920,5
1881,2930,5
1882,2330,8
1883,1870,5
1884,1230,5
1885,740,6
1886,530,10
1887,360,10
1888,337,10
1889,224,8
1890,144,13
1891,80,19
1892,49,26
1893,53,22
1894,18,18
1895,7,9
1896,11,5
1897,6,5
1898,3,6
1899,3,7
1900,4,12
1901,8,12
1902,23,19
1903,18,27
1904,7,24
"""


def _looks_like_csv(path: Path) -> bool:
    if not path.is_file():
        return False
    head = path.read_bytes()[:256].lstrip()
    if head.startswith(b"<!") or b"<html" in head[:64].lower():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return len(lines) >= 2 and ("," in lines[0] or "\t" in lines[0])


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s


def download_dryad_dataset(s: requests.Session, doi: str, out_dir: Path) -> list[str]:
    """下载单个 Dryad 数据集的全部文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    page_url = f"https://datadryad.org/dataset/doi:{doi}"
    r = s.get(page_url, timeout=90)
    r.raise_for_status()

    enc = quote(f"doi:{doi}", safe="")
    meta = s.get(
        f"https://datadryad.org/api/v2/datasets/{enc}",
        headers={"Accept": "application/json"},
        timeout=90,
    )
    meta.raise_for_status()
    ds = meta.json()
    ver = s.get(
        "https://datadryad.org" + ds["_links"]["stash:version"]["href"],
        timeout=90,
    ).json()
    files = s.get(
        "https://datadryad.org" + ver["_links"]["stash:files"]["href"],
        timeout=90,
    ).json()

    saved: list[str] = []
    for f in files.get("_embedded", {}).get("stash:files", []):
        name = f.get("path", "unknown")
        href = f.get("_links", {}).get("stash:download", {}).get("href", "")
        m = re.search(r"/files/(\d+)/download", href)
        if not m:
            continue
        fid = m.group(1)
        url = f"https://datadryad.org/downloads/file_stream/{fid}"
        dr = s.get(url, headers={"Referer": page_url}, timeout=180)
        dr.raise_for_status()
        dest = out_dir / name
        dest.write_bytes(dr.content)
        if name.lower().endswith(".csv") and not _looks_like_csv(dest):
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"{name} 下载内容为 HTML/无效（Dryad 反爬），请浏览器手动下载到 data/bundled/")
        saved.append(str(dest))
        time.sleep(0.3)
    return saved


def download_url(s: requests.Session, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = s.get(url, timeout=180)
    r.raise_for_status()
    dest.write_bytes(r.content)


def download_zenodo_files(s: requests.Session, record_id: str, names: list[str], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = s.get(f"https://zenodo.org/api/records/{record_id}", timeout=90).json()
    # 概念 DOI 可能指向最新版本记录
    if not rec.get("files") and rec.get("id"):
        rec = s.get(f"https://zenodo.org/api/records/{rec['id']}", timeout=90).json()
    by_key = {f["key"]: f for f in rec.get("files", [])}
    saved = []
    for name in names:
        if name not in by_key:
            continue
        link = by_key[name]["links"]["self"]
        print(f"    Zenodo: {name} ({by_key[name].get('size', 0) / 1e6:.2f} MB)")
        data = s.get(link, timeout=600)
        data.raise_for_status()
        dest = out_dir / name
        dest.write_bytes(data.content)
        saved.append(str(dest))
    (out_dir / "zenodo_record.json").write_text(
        json.dumps({"id": record_id, "title": rec.get("metadata", {}).get("title")}, indent=2),
        encoding="utf-8",
    )
    return saved


def try_forage(s: requests.Session, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    note = out_dir / "DOWNLOAD_INSTRUCTIONS.txt"
    try:
        r = s.get(FORAGE_KNB, timeout=120, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 1000:
            ctype = r.headers.get("Content-Type", "")
            if "zip" in ctype or r.content[:2] == b"PK":
                zpath = out_dir / "forage_knb.zip"
                zpath.write_bytes(r.content)
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(out_dir)
                return f"OK zip -> {out_dir}"
            dest = out_dir / "forage_resolve.bin"
            dest.write_bytes(r.content)
            return f"OK binary {len(r.content)} bytes"
    except Exception as e:
        pass
    text = (
        "FoRAGE 全库 (2000+ 功能反应) 请手动下载:\n"
        "https://knb.ecoinformatics.org/view/doi:10.5063/F17H1GTQ\n"
        "论文: https://doi.org/10.1101/503334\n"
    )
    note.write_text(text, encoding="utf-8")
    return "MANUAL (见 DOWNLOAD_INSTRUCTIONS.txt)"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    s = _session()

    print("=" * 60)
    print("下载推荐数据集 -> data/raw/")
    print("=" * 60)

    for name, doi in DRYAD_DOIS.items():
        print(f"\n[{name}] Dryad {doi}")
        try:
            files = download_dryad_dataset(s, doi, RAW / name)
            log.append(f"OK  {name}: {len(files)} files")
            for f in files[:3]:
                print(f"    {Path(f).name}")
            if len(files) > 3:
                print(f"    ... +{len(files)-3} more")
        except Exception as e:
            log.append(f"FAIL {name}: {e}")
            print(f"    FAIL: {e}")

    for name, url in DIRECT.items():
        if name == "03_hudson_bay_lynx_hare":
            continue
        print(f"\n[{name}] direct")
        try:
            fname = url.split("/")[-1].split("?")[0]
            dest = RAW / name / fname
            download_url(s, url, dest)
            if name == "05_gpdd_readme":
                (RAW / "05_gpdd" / "DOWNLOAD_INSTRUCTIONS.txt").write_text(
                    "GPDD 全库: https://knb.ecoinformatics.org/view/doi:10.5063/F1BZ63Z8\n"
                    "R 包: install.packages('rgpdd')\n"
                    "已下载包请解压/放到 data/raw/05_gpdd/\n",
                    encoding="utf-8",
                )
            log.append(f"OK  {name}: {fname}")
            print(f"    OK {fname}")
        except Exception as e:
            log.append(f"FAIL {name}: {e}")
            print(f"    FAIL: {e}")

    print(f"\n[06_biotime] Zenodo {ZENODO_BIOTIME}")
    try:
        saved = download_zenodo_files(s, ZENODO_BIOTIME, BIOTIME_FILES, RAW / "06_biotime")
        log.append(f"OK  06_biotime: {len(saved)} files")
        print(f"    OK {len(saved)} files")
    except Exception as e:
        log.append(f"FAIL 06_biotime: {e}")
        print(f"    FAIL: {e}")

    print("\n[07_lter_fish] EDI (威斯康星鱼类丰度子集)")
    try:
        out = RAW / "07_lter_fish"
        download_url(s, LTER_WI_FISH_CSV, out / "WIfishAbundance.csv")
        (out / "DOWNLOAD_INSTRUCTIONS.txt").write_text(
            "已下载 knb-lter-ntl.356 威斯康星湖泊鱼类丰度.\n"
            "NTL 主包 Fish Abundance: knb-lter-ntl.7.41\n"
            "https://lter.limnology.wisc.edu/data/search\n",
            encoding="utf-8",
        )
        log.append("OK  07_lter_fish: WIfishAbundance.csv")
        print("    OK WIfishAbundance.csv")
    except Exception as e:
        log.append(f"FAIL 07_lter_fish: {e}")
        print(f"    FAIL: {e}")

    print("\n[03_hudson_bay_lynx_hare] fallback")
    try:
        out = RAW / "03_hudson_bay_lynx_hare"
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "lynxhare.csv"
        if not dest.exists():
            dest.write_text(_LYNX_HARE_CSV, encoding="utf-8")
            (out / "SOURCE.txt").write_text(
                "GitHub 超时时使用经典 Hudson Bay 公开数据.\n"
                "https://www.math.duke.edu/education/ccp/materials/diffeq/predprey/pred1.html\n",
                encoding="utf-8",
            )
        log.append("OK  03_hudson_bay_lynx_hare: lynxhare.csv")
        print("    OK lynxhare.csv")
    except Exception as e:
        log.append(f"FAIL 03_hudson_bay: {e}")

    print("\n[14_forage] KNB")
    try:
        msg = try_forage(s, RAW / "14_forage")
        log.append(f"OK  14_forage: {msg}")
        print(f"    {msg}")
    except Exception as e:
        log.append(f"FAIL 14_forage: {e}")
        print(f"    FAIL: {e}")

    manifest = ROOT / "download_manifest.txt"
    manifest.write_text("\n".join(log) + "\n", encoding="utf-8")
    try:
        from .dataset_registry import annotate_raw_folders

        labeled = annotate_raw_folders()
        print(f"\n[标注] 已写入 {len(labeled)} 个 dataset.json")
    except Exception as e:
        print(f"\n[标注] dataset.json 写入跳过: {e}")
    print("\n" + "=" * 60)
    print(f"清单: {manifest}")
    print(f"数据: {RAW}")
    print("=" * 60)


if __name__ == "__main__":
    main()
