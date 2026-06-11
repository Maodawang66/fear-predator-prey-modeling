# 数据目录

| 路径 | 内容 |
|------|------|
| `数据来源.md` | 各数据集说明、引用与论文用法 |
| `download_datasets.py` | 一键下载脚本 |
| `download_manifest.txt` | 最近一次下载结果 |
| `dataset_catalog.json` | 全部数据集角色、列名、DOI 标注（源） |
| `dataset_registry.py` | 读取 catalog；同步 `raw/*/dataset.json` |
| `raw/` | 原始数据（按编号分子文件夹，含 `dataset.json`） |
| `raw/05_gpdd/` | GPDD 大样本拓展（~5156 序列，~18 万条观测） |

## 快速开始

```bash
python data/download_datasets.py
```

核心 CSV 示例:
- `raw/01_lynx_roe_deer/Andren_lynx_roedeer_data.csv`
- `raw/02_killifish_mosquitofish/TimeSeriesLogMeans.csv`
- `raw/03_hudson_bay_lynx_hare/lynxhare.csv`
- `raw/04_lake_michigan_zooplankton/GLERL_M110_Zoop_1994-2012.txt`
- `raw/07_lter_fish/WIfishAbundance.csv`
- `raw/12_peacor_risk_meta/PeacorEtAl_Data_ELE-00137-2022.xlsx`
- `raw/15_isle_royale_wolf_moose/isle_royale_wolf_moose_pre_2018.csv`

`raw/03_hudson_bay_lynx_hare/lynxhare.csv` 保留作历史资料，但毛皮交易代理量不进入正式 15 条种群序列分析。Windermere 北/南湖盆与 Komi 配对由 `raw/05_gpdd/data/` 中的 GPDD 记录显式加载。
