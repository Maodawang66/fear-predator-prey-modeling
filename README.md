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
│   ├── model.py             # ODE 右端项函数（7 种动力学形式）
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
| `main.py` | 主模型数值实验入口。运行基线 vs 恐惧+记忆 ODE、参数扫描、敏感性分析、按 20% 等效恐惧强度校准的机制对比、B-D+恐惧对比和 k-damping 分析 |
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
| `src/parameters.py` | Frozen dataclass 参数集，8 个 `MechanismId` 比较情景（含 Holling/B-D 各自无恐惧基线），以及预设参数实例 |
| `src/simulate.py` | `scipy.solve_ivp`（RK45）包装，均匀网格重采样（800 点），`long_term_mean` 长期均值，`is_extinct` 灭绝判定 |
| `src/analysis.py` | 参数扫描、局部敏感性分析、带 RHS 残差校验的平衡点求解、20% 等效恐惧参数反解，以及相对各模型体系无恐惧基线的机制比较 |
| `src/literature.py` | 统一多机制运行框架：`run_mechanism` / `run_all_mechanisms`，支持传入校准参数并分开 Holling/B-D 模型尺度 |
| `src/visualize.py` | 所有 matplotlib 绑图：时间序列对比、相图、参数扫描、敏感性条形图、机制对比、2D 斑图、拟合结果、k-damping 分析图 |
| `src/fit.py` | 三模型 ODE 参数拟合；输出训练/留后验证 RMSE、AIC/AICc/BIC、优化诊断；实现 B-D `k` profile likelihood，并保留条件敏感性扫描 |
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
| `data/calibrate_bda.py` | 自动发现标定管线：三模型仅在前 80% 时间点拟合，再对末尾 20% 做连续多步预测；输出 RMSE、信息准则和优化诊断 |
| `data/deep_data_analysis.py` | 深度分析：按留后验证误差与 AICc 比较模型；重新优化干扰参数的 `k` profile；在每条序列观测捕食者中位数处计算恐惧抑制 |
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
python data/deep_data_analysis.py --skip-profile-all --skip-lter  # 代表序列 profile，较快
python data/deep_data_analysis.py     # 全序列 profile + LTER，耗时很长

# 文档编译
call compile_docs.bat                 # 规划/教程 → PDF
call compile_report.bat               # report.tex → PDF
```

## 两种模型体系

- **经典体系**（x=猎物, y=捕食者, 可选 M=记忆）：logistic 增长 + Holling II 功能响应，参数为 `BaselineParams` / `FearMemoryParams`
- **B-D 体系**（u=猎物, v=捕食者）：Beddington-DeAngelis 功能响应 + Wang 恐惧因子 `1/(1+kv)`，参数为 `BDAFearParams`，基于 Myint et al. (2025, arXiv:2506.22070)

## 拟合与模型比较口径

- 三种模型只使用前 80% 时间点拟合，归一化尺度也只由训练段确定；末尾 20% 用于从序列初始状态连续积分得到的多步留后预测。
- 三种模型均输出训练段 `rmse_normalized_*` / `rmse_raw_*` 和留后段 `validation_rmse_*`。跨模型胜负优先依据留后验证误差与 AICc，不能仅依据训练 RMSE。
- AIC、AICc 和 BIC 使用统一归一化训练残差，并按各模型实际拟合参数数量惩罚复杂度。
- `optimization_status` 分为 `success`、`usable_limit` 和 `failed`；失败拟合保留用于诊断，但不进入模型胜负统计。
- 每条拟合记录包含优化终止原因、目标函数值和参数触边情况。Holling II 搜索区间已覆盖当前默认 `theta=0.0052`，`fit_e_mu=True` 会实际拟合 `e`、`mu`。
- B-D 正平衡点通过降维求根计算，返回前必须满足正值约束并通过 `bd_fear_rhs` 残差检查。

## 公平机制比较与 B-D 恐惧强度

- Holling 与 B-D 使用不同量纲，绝对时间序列只能分图展示；机制胜负使用相对各自无恐惧基线的长期均值变化和相对振幅 `A/mean` 变化。
- 核心机制对比在各体系无恐惧正平衡点处校准为 20% 等效抑制。原始默认参数结果仅作为 `results/supplementary/` 中的补充实验。
- B-D 必须与 `k=0` 的 B-D 基线比较。
- `profile_bda_k` 会对每个固定 `k` 重新优化其余 B-D 参数并输出近似 95% profile 区间；`conditional_bda_k_scan` 才是固定其余参数的条件敏感性扫描。
- 跨系统恐惧抑制使用每条序列归一化捕食者的观测中位数：`eta_at_observed_median`。`eta_v5_theoretical_extrapolation` 仅为理论外推；当前 `v=5` 对 12/12 条序列均超出观测范围。

主要结果位于：

- `results/calibration_bda/fit_summary.csv`：36 条三模型拟合记录。
- `results/deep_analysis/tier1/rmse_improvement.csv`：训练 RMSE、留后验证 RMSE、AICc/BIC 和两类胜者。
- `results/equivalent_fear_calibration.json`：20% 等效恐惧参数与参考状态。
- `results/deep_analysis/tier1/k_profile_long.csv`：重新优化干扰参数后的 profile likelihood。
- `results/deep_analysis/tier1/k_conditional_sensitivity_long.csv`：固定其余参数的条件扫描。
- `results/deep_analysis/tier1/cross_system_k_eta.csv`：观测中位捕食者密度、观测范围内 `eta` 和明确标记的 `v=5` 理论外推。

`report/` 中的论文和规划文档尚未完成 TODO 18 的系统更新，可能包含修复前结论；当前实验口径以代码和上述机器可读结果为准。

## 参考文献

Myint, S. S., Wang, Y., & Preisser, E. L. (2025). Fear-induced destabilization in a Beddington-DeAngelis predator-prey model. *arXiv:2506.22070*.
