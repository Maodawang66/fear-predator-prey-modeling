# 恐惧效应下的捕食者—猎物动力学

USTC 数学建模课程项目。建立多种生态数学模型（ODE 和 PDE），研究恐惧/记忆效应对捕食者—猎物系统稳定性的影响，并与 12 组真实生态时间序列数据拟合校准。

## 环境配置

```bash
conda activate ai25
pip install -r requirements.txt
```

依赖：`numpy>=1.24`, `scipy>=1.10`, `matplotlib>=3.7`, `requests>=2.28`, `openpyxl>=3.1`

## 项目结构

```
pro/
├── main.py                  # 主模型 ODE 数值实验入口
├── pde2d_turing.py          # 2D 反应—扩散 PDE + Turing 稳定性诊断
├── k_damping_analysis.py    # k 参数削弱振荡专项分析
├── compile_docs.py          # pandoc → PDF 文档编译
├── compile_docs.bat         # 编译规划/教程为 PDF
├── compile_report.bat       # 编译 report.tex 为 PDF
├── run_deep_analysis.bat    # 运行深度数据分析
├── requirements.txt         # Python 依赖
├── CLAUDE.md                # Claude Code 辅助配置
├── src/                     # 核心库包
│   ├── model.py             # ODE 右端项函数（7 种恐惧机制）
│   ├── parameters.py        # 参数集定义与预设情景
│   ├── simulate.py          # 数值积分包装器
│   ├── analysis.py          # 参数扫描与敏感性分析
│   ├── literature.py        # 多机制统一对比框架
│   ├── visualize.py         # matplotlib 绑图函数集
│   ├── fit.py               # ODE 参数拟合（差分进化 + L-BFGS-B）
│   ├── k_damping.py         # k 参数削弱振荡分析模块
│   └── pde2d.py             # 2D 反应—扩散 PDE 有限差分求解器
├── data/                    # 数据模块
│   ├── common.py            # 共享工具（路径解析、CSV 读取）
│   ├── series.py            # 时间序列数据容器
│   ├── dataset_registry.py  # 数据集注册表
│   ├── dataset_catalog.json # 12 组数据集的元数据目录
│   ├── download_datasets.py # 一键下载所有数据集
│   ├── auto_discover.py     # 启发式自动发现捕食者—猎物时间序列
│   ├── calibrate_datasets.py# 手动标定（5 组数据集）
│   ├── calibrate_bda.py     # 自动发现 + 批量标定管线
│   ├── deep_data_analysis.py# 深度分析（双轨验证 + 实验先验）
│   ├── load_lynx_hare.py    # Hudson Bay 猞猁—雪兔数据加载器
│   ├── load_lynx_roe.py     # 欧亚猞猁—狍数据加载器
│   ├── load_killifish.py    # 鳉鱼—食蚊鱼数据加载器
│   ├── load_zooplankton.py  # 密歇根湖浮游动物数据加载器
│   ├── load_lter_fish.py    # Wisconsin LTER 鱼类数据加载器
│   └── load_peacor.py       # Peacor 荟萃分析数据加载器
├── report/                  # 报告文档
│   ├── report.tex/pdf       # 最终论文
│   ├── 摘要.tex/pdf         # 摘要
│   ├── 规划.md/pdf          # 项目规划文档
│   ├── 教程.md/pdf          # 原理教程
│   └── fit_table_rows.tex   # 拟合表格片段
└── results/                 # 所有实验输出（图表、CSV、JSON）
```

## 各文件功能说明

### 根目录入口脚本

| 文件 | 功能 |
|---|---|
| `main.py` | 主模型数值实验入口。运行基线 vs 恐惧+记忆 ODE 对比、参数扫描（φ, δ）、敏感性分析、七种文献机制对比、B-D+恐惧模型对比，以及 k-damping 分析。生成约 12 张图表到 `results/` |
| `pde2d_turing.py` | 2D 反应—扩散 PDE + Turing 线性稳定性诊断。支持命令行参数：`--fig demo/all/3/4/5/6` 选择图组，`--quick` 快速试跑，`--skip-pde` 仅输出稳定性曲线 |
| `k_damping_analysis.py` | k 参数削弱振荡独立分析。执行 Jacobian 特征值扫描、数值振幅扫描、峰值衰减比拟合，输出 5 张图 + CSV + JSON 到 `results/k_damping/` |
| `compile_docs.py` | 使用 pandoc + xelatex 将 `report/规划.md` 和 `report/教程.md` 编译为 PDF |
| `compile_docs.bat` | Windows 批处理，调用 `compile_docs.py` |
| `compile_report.bat` | Windows 批处理，使用 latexmk + xelatex 编译 `report/report.tex` |
| `run_deep_analysis.bat` | Windows 批处理，运行 `data/deep_data_analysis.py` |

### src/ 核心库

| 文件 | 功能 |
|---|---|
| `src/model.py` | 7 种 ODE 右端项函数：`baseline_rhs`（logistic + Holling II）、`fear_memory_rhs`（Wang 2019 记忆核）、`fear_instant_rhs`（无记忆瞬时恐惧）、`fear_saturating_rhs`（Zanette 型饱和恐惧）、`fear_foraging_rhs`（Lima/Preisser 觅食抑制）、`fear_handling_rhs`（处理时间延长）、`bd_fear_rhs`（B-D + Wang 恐惧因子） |
| `src/parameters.py` | Frozen dataclass 参数集（`BaselineParams`, `FearMemoryParams`, `FearSaturatingParams`, `FearForagingParams`, `FearHandlingParams`, `BDAFearParams`），`MechanismId` 枚举，预设情景实例 |
| `src/simulate.py` | `scipy.solve_ivp`（RK45）包装，均匀网格重采样（800 点），`long_term_mean` 长期均值，`is_extinct` 灭绝判定 |
| `src/analysis.py` | 参数扫描（`scan_phi`, `scan_delta`, `scan_bda_fear_k`），局部敏感性分析（`sensitivity_local`），平衡点数值求解（`equilibrium_baseline`, `equilibrium_bda_fear`） |
| `src/literature.py` | 统一多机制对比框架：`run_mechanism` / `run_all_mechanisms`，支持 7 种机制 ID |
| `src/visualize.py` | 所有 matplotlib 绑图：时间序列对比、相图、参数扫描、敏感性条形图、机制对比、2D 斑图、拟合结果、k-damping 分析图 |
| `src/fit.py` | ODE 参数拟合：`differential_evolution`（≥5 参数）+ `L-BFGS-B`（<5 参数），支持 baseline / fear_memory / B-D+fear 三种模型，`profile_bda_k` k 参数可辨识性分析 |
| `src/k_damping.py` | k 削弱振荡分析：Jacobian 矩阵求解、特征值扫描、Hopf 分岔阈值估计、峰值衰减指标、多 k 联合扫描 |
| `src/pde2d.py` | 2D 反应—扩散显式 Euler 有限差分（Neumann 边界），Turing 稳定性 λ(k) 分析，`scan_d2_patterns` 扩散系数扫描 |

### data/ 数据模块

| 文件 | 功能 |
|---|---|
| `data/common.py` | 共享工具：`resolve_data_file`（raw/ → bundled/ → 绝对路径 fallback）、`is_valid_csv`（Dryad 反爬检测）、`read_csv_dicts`、`normalize_time_years` |
| `data/series.py` | `PredatorPreySeries` dataclass：统一时间序列容器（t, prey, predator + 元数据），支持 `scaled_copy` 归一化 |
| `data/dataset_registry.py` | 读取 `dataset_catalog.json`，同步注解到 `data/raw/<id>/dataset.json`，`is_ode_fit_path` 判定 |
| `data/dataset_catalog.json` | 12 组数据集的完整注册信息（DOI、角色、列映射、加载器引用） |
| `data/download_datasets.py` | 一键下载所有数据集（Dryad API + GitHub + Zenodo），含反爬检测和 manifest 输出 |
| `data/auto_discover.py` | 启发式自动发现引擎：扫描 CSV/TSV，识别时间/猎物/捕食者列，处理堆叠区域数据、长格式表格等复杂情况 |
| `data/calibrate_datasets.py` | 手动标定 5 组数据集（hudson_bay, lynx_roe_r3, killifish_tp, zooplankton, lter_fish），分别拟合 baseline / fear_memory / B-D+fear |
| `data/calibrate_bda.py` | 自动发现标定管线：扫描所有数据 → 自动识别 → 三种模型拟合 → 输出图表/JSON/CSV 到 `results/calibration_bda/` |
| `data/deep_data_analysis.py` | 深度分析（双轨验证）：Track A 跨系统 η(k,v) 计算 + RMSE 改进 + k 可辨识性；Track B Peacor 荟萃分析 + 珊瑚礁 + 豆娘实验先验 |
| `data/load_lynx_hare.py` | Hudson Bay 猞猁—雪兔毛皮记录（1845–1935，90+ 年）加载器 |
| `data/load_lynx_roe.py` | Andren 等欧亚猞猁—狍 7 区域数据加载器，支持单区域或全区域 |
| `data/load_killifish.py` | 鳉鱼—食蚊鱼月度 log 密度数据（3 站点 × 5 年 × 12 月）加载器 |
| `data/load_zooplankton.py` | Lake Michigan GLERL 浮游动物监测数据（1994–2012）加载器 |
| `data/load_lter_fish.py` | Wisconsin LTER 鱼类丰度数据加载器（默认 Bluegill vs Largemouth Bass） |
| `data/load_peacor.py` | Peacor 等 (2022) 荟萃分析数据（PLP studies）XLSX/CSV 加载器 |

## 常用命令

```bash
conda activate ai25

# 主模型数值实验
python main.py

# 2D 反应—扩散 PDE + Turing 稳定性
python pde2d_turing.py                # 默认图组
python pde2d_turing.py --fig all      # 全部图组
python pde2d_turing.py --quick        # 64×64 快速试跑
python pde2d_turing.py --skip-pde     # 仅稳定性曲线

# k 削弱振荡专项分析
python k_damping_analysis.py

# 数据下载与标定
python data/download_datasets.py      # 下载所有数据集
python data/calibrate_datasets.py     # 手动标定
python data/calibrate_bda.py          # 自动发现 + 批量标定
python data/deep_data_analysis.py     # 深度分析

# 文档编译
call compile_docs.bat                 # 规划/教程 → PDF
call compile_report.bat               # report.tex → PDF
```

## 两种模型体系

- **经典体系**（x=猎物, y=捕食者, 可选 M=记忆）：logistic 增长 + Holling II 功能响应，参数为 `BaselineParams` / `FearMemoryParams`
- **B-D 体系**（u=猎物, v=捕食者）：Beddington-DeAngelis 功能响应 + Wang 恐惧因子 `1/(1+kv)`，参数为 `BDAFearParams`，基于 Myint et al. (2025, arXiv:2506.22070)

## 参考文献

Myint, S. S., Wang, Y., & Preisser, E. L. (2025). Fear-induced destabilization in a Beddington-DeAngelis predator-prey model. *arXiv:2506.22070*.
