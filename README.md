# 恐惧效应下的捕食者—猎物动力学

USTC 数学建模课程项目。建立多种生态数学模型（ODE），研究恐惧/记忆效应的机制路径依赖性，并通过 15 条真实种群时间序列评估留后预测、复杂度、参数可辨识性和证据边界。

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
├── run_deep_analysis.bat    # 运行深度数据分析
├── requirements.txt         # Python 依赖
├── src/                     # 核心库包
│   ├── model.py             # ODE 右端项函数（6 种 Holling II 动力学形式）
│   ├── parameters.py        # 参数集定义与预设情景
│   ├── simulate.py          # 数值积分包装器
│   ├── analysis.py          # 参数扫描与敏感性分析
│   ├── literature.py        # 多机制统一对比框架
│   ├── visualize.py         # matplotlib 绑图函数集
│   ├── fit.py               # ODE 参数拟合（差分进化 + L-BFGS-B）
│   └── fear_pathway_fit.py  # 等参数化 Holling II 恐惧通道拟合
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
│   ├── generate_report_protocol_seven_heatmap.py # 报告口径六模型汇总、热力图与改善幅度图
│   ├── load_lynx_hare.py    # Hudson Bay 猞猁—雪兔数据加载器
│   ├── load_lynx_roe.py     # 欧亚猞猁—狍数据加载器
│   ├── load_killifish.py    # 鳉鱼—食蚊鱼数据加载器
│   ├── load_zooplankton.py  # 密歇根湖浮游动物数据加载器
│   ├── load_lter_fish.py    # Wisconsin LTER 鱼类数据加载器
│   └── load_peacor.py       # Peacor 荟萃分析数据加载器
├── report/                  # 报告文档
│   ├── report.tex/pdf       # 最终论文
│   └── fit_table_rows.tex   # 拟合表格片段
└── results/                 # 所有实验输出（图表、CSV、JSON）
```

## 各文件功能说明

### 根目录入口脚本

| 文件 | 功能 |
|---|---|
| `main.py` | 主模型数值实验入口。运行基线 vs 恐惧+记忆 ODE、参数扫描、敏感性分析、按 20% 等效恐惧强度校准的机制对比 |
| `run_deep_analysis.bat` | Windows 批处理，运行 `data/deep_data_analysis.py` |

### src/ 核心库

| 文件 | 功能 |
|---|---|
| `src/model.py` | 6 种 ODE 右端项函数：`baseline_rhs`（logistic + Holling II）、`fear_memory_rhs`（Wang 2019 记忆核）、`fear_instant_rhs`（无记忆瞬时恐惧）、`fear_saturating_rhs`（Zanette 型饱和恐惧）、`fear_foraging_rhs`（Lima/Preisser 觅食抑制）、`fear_handling_rhs`（处理时间延长） |
| `src/parameters.py` | Frozen dataclass 参数集，6 个 `MechanismId` 比较情景（含无恐惧基线），以及预设参数实例 |
| `src/simulate.py` | `scipy.solve_ivp`（RK45）包装，均匀网格重采样（800 点），`long_term_mean` 长期均值，`is_extinct` 灭绝判定 |
| `src/analysis.py` | 参数扫描、局部敏感性分析、带 RHS 残差校验的平衡点求解、20% 等效恐惧参数反解，以及相对各模型体系无恐惧基线的机制比较 |
| `src/literature.py` | 统一多机制运行框架：`run_mechanism` / `run_all_mechanisms`，支持传入校准参数 |
| `src/visualize.py` | 所有 matplotlib 绑图：时间序列对比、相图、参数扫描、敏感性条形图、机制对比、拟合结果 |
| `src/fit.py` | baseline + fear-memory ODE 参数拟合；输出训练/留后验证 RMSE、AIC/AICc/BIC、优化诊断 |
| `src/fear_pathway_fit.py` | 在相同 Holling II baseline 上拟合瞬时繁殖、记忆繁殖、饱和繁殖、觅食/攻击率抑制和处理时间延长通道；每个候选仅增加一个主要恐惧参数 |

### data/ 数据模块

| 文件 | 功能 |
|---|---|
| `data/common.py` | 共享工具：`resolve_data_file`（raw/ → bundled/ → 绝对路径 fallback）、`is_valid_csv`（Dryad 反爬检测）、`read_csv_dicts`、`normalize_time_years` |
| `data/series.py` | `PredatorPreySeries` dataclass：统一时间序列容器（t, prey, predator + 元数据），支持 `scaled_copy` 归一化 |
| `data/dataset_registry.py` | 读取 `dataset_catalog.json`，同步注解到 `data/raw/<id>/dataset.json`，`is_ode_fit_path` 判定 |
| `data/dataset_catalog.json` | 12 组数据集的完整注册信息（DOI、角色、列映射、加载器引用） |
| `data/download_datasets.py` | 一键下载所有数据集（Dryad API + GitHub + Zenodo），含反爬检测和 manifest 输出 |
| `data/auto_discover.py` | 启发式自动发现引擎：扫描 CSV/TSV，识别时间/猎物/捕食者列，处理堆叠区域数据、长格式表格等复杂情况 |
| `data/calibrate_datasets.py` | 手动标定 5 组数据集（hudson_bay, lynx_roe_r3, killifish_tp, zooplankton, lter_fish），分别拟合 baseline / fear_memory |
| `data/calibrate_bda.py` | 自动发现标定管线：baseline + fear-memory 仅在前 80% 时间点拟合，再对末尾 20% 做连续多步预测；输出 RMSE、信息准则和优化诊断 |
| `data/deep_data_analysis.py` | 深度分析：按留后验证误差与 AICc 比较模型 |
| `data/generate_report_protocol_seven_heatmap.py` | 将 `report.tex` 的正式拟合口径扩展到六模型。四个新增通道可按 `--model` 断点运行，`--aggregate-only` 汇总正式六模型指标、热力图，以及最佳恐惧模型相对 baseline 的留后 RMSE 改善 CSV/排序图 |
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

# 数据下载与标定
python data/download_datasets.py      # 下载所有数据集
python data/calibrate_datasets.py     # 手动标定
python data/calibrate_bda.py          # 自动发现 + 批量标定
python data/deep_data_analysis.py --skip-profile-all --skip-lter  # 代表序列 profile，较快
python data/deep_data_analysis.py     # 全序列 profile + LTER，耗时很长

# 与 report.tex 相同口径的六模型扩展；四个新增通道可分别运行并断点续跑
python data/generate_report_protocol_seven_heatmap.py --model fear_instant
python data/generate_report_protocol_seven_heatmap.py --model fear_saturating
python data/generate_report_protocol_seven_heatmap.py --model fear_foraging
python data/generate_report_protocol_seven_heatmap.py --model fear_handling
python data/generate_report_protocol_seven_heatmap.py --aggregate-only

# 若 latexmk 启动器异常，直接连续运行两次 XeLaTeX
xelatex -interaction=nonstopmode -output-directory=report report/report.tex
xelatex -interaction=nonstopmode -output-directory=report report/report.tex
```

## 当前主要结论

- 在无恐惧正平衡点处校准为 20% 等效抑制后，五种恐惧机制均保持共存，但长期种群变化方向明显不同。
- 记忆模型的 `phi` 扫描 31/31 在当前积分范围内未灭绝，其中 20 个情景通过长期收敛诊断，11 个未通过。
- 六模型（baseline + 5 种 Holling II 恐惧通道）留后验证 RMSE 最优为 baseline 3 条、fear-memory 4 条、fear-instant 2 条、fear-foraging 2 条、fear-handling 4 条；AICc 最优为 baseline 8 条、fear-memory 1 条、fear-instant 2 条、fear-saturating 1 条、fear-foraging 3 条。
- 六模型 90/90 次拟合均可比较；不同序列由不同通道获胜，无单一恐惧通道一致胜出，预测胜出不能替代参数可辨识性证据。
- 最佳可比较恐惧模型相对 baseline 的留后 RMSE 在 12/15 条序列上数值改善；8/15 至少改善 1%，7/15 至少改善 5%，6/15 至少改善 10%。12 条正改善序列的中位改善为 10.47%，全部 15 条的中位改善为 1.17%，范围为 -2.65% 至 72.23%。
- 最大改善来自包含未建模刺网移除干预的 Windermere North，不解释为恐惧机制证据。
- Peacor 数据中的 TMIE 37、NCE 27 是研究类型计数，不是效应量。Andrén 七区、Killifish 三站和 Windermere 两湖盆存在研究内重复，序列级胜负不是 15 个独立证据。

最终论文位于 `report/report.tex`，结构为：

`引言 → 相关工作 → 预备知识 → 模型与方法 → 理论与动力学分析 → 实验与讨论 → 结论 → 参考文献 → 附录`

## 模型体系

- **经典体系**（x=猎物, y=捕食者, 可选 M=记忆）：logistic 增长 + Holling II 功能响应，参数为 `BaselineParams` / `FearMemoryParams`

## 拟合与模型比较口径

- 各模型只使用前 80% 时间点拟合，归一化尺度也只由训练段确定；末尾 20% 用于从序列初始状态连续积分得到的多步留后预测。
- 各模型均输出训练段 `rmse_normalized_*` / `rmse_raw_*` 和留后段 `validation_rmse_*`。跨模型胜负优先依据留后验证误差与 AICc，不能仅依据训练 RMSE。
- AIC、AICc 和 BIC 使用统一归一化训练残差，并按各模型实际拟合参数数量惩罚复杂度。
- `optimization_status` 分为 `success`、`usable_limit` 和 `failed`；失败拟合保留用于诊断，但不进入模型胜负统计。
- baseline 默认逐序列拟合 `r,K,a,theta,e,mu`；fear-memory 默认逐序列拟合 `r,K,a,theta,e,mu,phi`，仅固定 `delta=1`。`fit_e_mu=False` 可复现固定 `e、mu` 的旧模式。
- 所有拟合均使用固定种子 `(0,1,2)` 的差分进化，选择最佳候选后进行局部精修；每条记录包含逐种子目标值、选中种子、优化终止原因、最终目标函数值和参数触边情况。
- 正式六模型扩展沿用同一口径：训练段尺度归一化、20% 连续多步留后、固定种子 `(0,1,2)` 差分进化和局部精修。
- 六模型改善率定义为 `100 * (baseline 留后 RMSE - 最佳可比较恐惧模型留后 RMSE) / baseline 留后 RMSE`；只使用正式六模型指标和可比较拟合。
- `e、mu` 的点估计只在对应序列时间单位内解释；部分 `e` 拟合触及下界，不能进行未经可辨识性分析的跨系统生态比较。

## 公平机制比较

- 核心机制对比在无恐惧正平衡点处校准为 20% 等效抑制。原始默认参数结果仅作为 `results/supplementary/` 中的补充实验。
- 15 条主序列是拟合单位，不是独立生态重复；LTER 额外拟合因物种配对、时间尺度和筛选规则不同，仅作为探索性输出。

主要结果位于：

- `results/calibration_bda/fit_summary.csv`：30 条 baseline + fear-memory 拟合记录。
- `results/seven_model_real_fits/report_protocol_six_model_metrics.csv`：与 `report.tex` 相同口径的 15 条序列 × 6 模型，共 90 条正式指标。
- `results/seven_model_real_fits/report_protocol_six_model_validation_heatmap.png`：正式六模型留后验证热力图。
- `results/seven_model_real_fits/report_protocol_six_model_holdout_improvement.csv`：逐序列 baseline、最佳恐惧模型、绝对改善和相对改善率。
- `results/seven_model_real_fits/report_protocol_six_model_holdout_improvement.png`：按改善率排序、按最佳恐惧通道配色的水平条形图。
- `results/seven_model_real_fits/report_protocol_fear_*_metrics.csv`：四个新增通道的正式拟合断点缓存。
- `results/deep_analysis/tier1/rmse_improvement.csv`：训练 RMSE、留后验证 RMSE、AICc/BIC 和两类胜者。
- `results/equivalent_fear_calibration.json`：20% 等效恐惧参数与参考状态。

真实数据与诊断输出位于：

- `data/raw/`：下载得到的原始真实数据。
- `data/bundled/`：原始下载不可用时的项目内打包数据。
- `data/dataset_catalog.json`：数据集 DOI、用途、列映射和加载器注册信息。
- `results/calibration_bda/identification_report.json`：15 条正式序列的来源文件和元数据。
- `results/calibration_bda/params/`、`results/calibration_bda/figures/`：正式 baseline + fear-memory 参数与拟合图。
- `results/fear_pathway_comparison/`：等参数化通道开发、合成恢复和 profile 诊断。该目录使用的早期优化预算与正式报告口径不同，不用于正式六模型胜负统计。

`report/report.tex` 已按当前机器可读结果重写。报告用于组织和解释证据；若报告文字与代码或机器可读输出冲突，以代码和上述输出为准。

`REBUTTAL.md` 按当前正式六模型结果审计 reviewer 质疑、已有回应和仍可追加的不确定性分析；不再使用旧 TODO 或行号引用。

## 参考文献

见 `report/report.tex` 中的完整文献列表。
