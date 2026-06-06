# TODO List

> 执行规则：每完成一条 TODO，完成对应验证后单独创建一次 Git 提交。不要将多个 TODO 合并到同一提交中。

## P0：会导致核心结论无效的问题

1. 问题：三种数据拟合模型使用的 RMSE 尺度不一致。`src/fit.py` 中 baseline 和 fear_memory 的最终 `rmse_total`、`rmse_prey`、`rmse_predator` 使用原始丰度计算，而 bda_fear 使用按各物种最大值归一化后的丰度计算。`data/deep_data_analysis.py` 又直接计算 baseline/B-D RMSE 比值，导致报告中的“B-D 改进 10^2--10^6 倍”和“12/12 最优”缺乏公平比较基础。
   修复方法：在 `src/fit.py` 中定义统一的模型比较误差口径，建议所有模型同时输出归一化 RMSE 和原始尺度 RMSE，并明确字段名称；让 `data/calibrate_bda.py`、`data/calibrate_datasets.py` 和 `data/deep_data_analysis.py` 只使用统一的归一化 RMSE 进行跨模型比较。重新运行 12 条序列的三模型拟合并更新比较表。
   Git 提交说明：单独提交，例如 `fix(fit): use comparable normalized RMSE across models`。

2. 问题：`src/analysis.py::equilibrium_bda_fear` 返回的 B-D 正平衡点不满足 `bd_fear_rhs=0`。默认参数下函数返回约 `(5.166, 5.371)`，但代回方程得到非零 `du/dt`。这会污染 B-D Jacobian、特征值、稳定性分类和 Hopf 判断。
   修复方法：重新推导并实现 B-D 正平衡点求解；可使用解析关系降维后求根，或使用带正值约束和多初值的数值求根。返回结果前必须检查状态为正且 RHS 残差低于容差。同步检查 `src/k_damping.py` 中所有依赖该平衡点的计算。
   Git 提交说明：单独提交，例如 `fix(analysis): compute valid B-D coexistence equilibrium`。

3. 问题：现有拟合结果大量未成功收敛，但仍被写入汇总并用于报告结论。当前结果中 baseline 仅 7/12、fear_memory 仅 2/12、bda_fear 仅 1/12 标记为 `success=True`。
   修复方法：在 `data/calibrate_bda.py` 和 `data/calibrate_datasets.py` 中区分成功、达到迭代上限但可用、失败三种状态；失败拟合不得进入模型胜负统计。记录优化终止原因、目标函数值和参数是否触边，并为失败序列增加重试或明确标记。
   Git 提交说明：单独提交，例如 `fix(calibration): exclude failed optimizations from model comparison`。

4. 问题：Holling II 数据拟合仍使用旧参数范围。`src/fit.py` 中 baseline/fear 的 `theta` 初值约为 `20`、下界为 `0.1`，无法搜索到当前合理默认值 `theta=0.0052`；`fit_e_mu` 参数存在但没有实际作用。
   修复方法：根据当前模型尺度重新设置 `a`、`theta`、`e`、`mu` 的初值和边界；实现或删除无效的 `fit_e_mu` 参数。使用修正后的边界重新拟合 12 条序列，并检查参数触边情况。
   Git 提交说明：单独提交，例如 `fix(fit): align Holling parameter bounds with calibrated regime`。

## P1：实验比较设计欠缺

5. 问题：`src/literature.py` 和 `src/analysis.py::compare_mechanisms` 将 Holling 系列与 B-D 模型的绝对长期均值直接画在同一柱状图中。两类模型使用不同量纲、参数体系和初值，因此 `20` 与 `5` 不具有直接定量可比性。
   修复方法：将机制比较指标改为相对各自无恐惧基线的变化比例，例如猎物和捕食者长期均值变化百分比、相对振幅变化；如仍展示绝对均值，应拆分图表并明确标注模型尺度。B-D 必须与 `k=0` 的 B-D 基线比较。
   Git 提交说明：单独提交，例如 `fix(analysis): compare fear mechanisms relative to model-specific baselines`。

6. 问题：不同恐惧机制使用的默认 `phi`、`psi`、`h`、`k` 没有校准到相同的等效恐惧强度，当前机制对照同时混合了“机制形式差异”和“参数强度差异”。
   修复方法：定义统一的参考状态或等效抑制指标，例如在指定捕食者密度下使猎物增长或捕食率降低相同百分比；据此为各机制反解参数，再重新运行机制对照。保留原始默认参数实验作为补充而非公平比较结论。
   Git 提交说明：单独提交，例如 `feat(analysis): calibrate equivalent fear strength across mechanisms`。

7. 问题：三模型复杂度不同，当前直接比较训练集 RMSE。baseline、fear_memory 和 B-D 拟合参数数量不同，B-D 更灵活，训练误差更低不能单独证明模型更优。
   修复方法：基于统一残差尺度增加 AIC、AICc 和 BIC；同时实现按时间顺序划分的留后验证或滚动时间序列验证，报告训练误差和验证误差。模型胜负结论优先依据验证误差与 AICc。
   Git 提交说明：单独提交，例如 `feat(fit): add complexity-aware and out-of-sample model comparison`。

8. 问题：`src/fit.py::profile_bda_k` 固定其他 B-D 参数后仅扫描 `k`，不是真正的 profile likelihood，可能低估 `k` 与其他参数的 trade-off 和不确定性。
   修复方法：对每个固定 `k` 重新优化其余 B-D 参数，输出 profile RMSE 曲线和近似置信区间；保留当前固定其他参数的扫描，但改名为条件敏感性扫描，避免误称 profile likelihood。
   Git 提交说明：单独提交，例如 `feat(fit): reoptimize nuisance parameters in B-D k profile`。

9. 问题：B-D 拟合将捕食者归一化到约 `[0,1]`，但跨系统比较使用 `eta(v=5)`，该参考值通常位于拟合数据范围之外，属于外推。
   修复方法：将等效抑制改为数据支持范围内的参考量，例如每条序列的归一化捕食者中位数、均值或指定分位数；跨系统汇总时同时报告参考密度。若保留 `eta(v=5)`，必须明确标记为理论外推。
   Git 提交说明：单独提交，例如 `fix(analysis): evaluate B-D fear suppression within observed predator range`。

## P1：默认参数标定和状态判定欠缺

10. 问题：`data/calibrate_holling_defaults.py` 通过自动发现后按置信度和点数排序截取前 12 条序列，未显式锁定报告使用的 12 条序列。数据文件变化后，默认参数标定样本可能悄然改变。
    修复方法：建立显式的报告序列清单或稳定 ID，并让标定脚本校验实际加载序列与清单完全一致；输出序列清单和数据摘要，缺失或新增时直接失败。
    Git 提交说明：单独提交，例如 `fix(calibration): pin Holling default calibration to report series manifest`。

11. 问题：`data/calibrate_holling_defaults.py` 使用每条序列最大值归一化，容易受异常峰值影响；使用后半段均值代表长期平衡，也没有先检验序列是否平稳或是否仍处于趋势/周期状态。
    修复方法：比较最大值、稳健分位数和 z-score 等缩放方法；增加趋势、周期性和尾段稳定性诊断。对于明显非平稳或周期序列，应使用周期均值、状态空间估计或从平衡目标样本中排除，并做敏感性分析。
    Git 提交说明：单独提交，例如 `feat(calibration): add robust scaling and stationarity checks for equilibrium target`。

12. 问题：Holling 默认参数的目标函数包含人为弱先验和硬编码平衡区间，但当前输出没有量化参数不可辨识性，也没有说明结果对先验和搜索范围的敏感程度。
    修复方法：输出接近最优的参数集合、参数分布和目标函数等高线；分别改变先验权重、搜索边界和目标平衡定义，报告默认参数和平衡点的敏感性。将弱先验和硬约束写入输出说明。
    Git 提交说明：单独提交，例如 `feat(calibration): report identifiability and sensitivity of Holling defaults`。

13. 问题：`src/simulate.py::is_extinct` 使用末期窗口最小值判断灭绝。具有正常振荡的种群只要短暂低于阈值，就会被标记为灭绝；固定绝对阈值也无法跨不同量纲模型公平使用。
    修复方法：将灭绝判定改为持续低于阈值一定时间，或结合尾部均值、分位数和恢复趋势；阈值应支持相对模型尺度配置。增加对稳定共存、低谷振荡和真实灭绝轨迹的测试。
    Git 提交说明：单独提交，例如 `fix(simulate): make extinction classification persistent and scale-aware`。

14. 问题：长期均值、振幅和稳定状态使用固定积分终点及固定 burn-in 比例，没有检查轨迹是否已经收敛。不同参数下收敛速度差异会导致扫描结果偏差。
    修复方法：增加收敛诊断，例如比较相邻尾部窗口均值和振幅、检查状态导数或平衡残差；未收敛时自动延长积分或标记为 `not_converged`。扫描输出必须包含收敛状态。
    Git 提交说明：单独提交，例如 `feat(simulate): add convergence diagnostics to long-term metrics`。

## P2：验证、复现和报告欠缺

15. 问题：项目缺少自动化测试，核心公式和数据比较错误没有回归保护。
    修复方法：新增测试覆盖：Holling 正平衡点 RHS 残差、B-D 正平衡点 RHS 残差、Jacobian 数值差分校验、统一 RMSE 尺度、灭绝分类、默认参数共存、扫描结果结构和标定序列清单。测试应使用小网格或短数据以保持运行速度。
    Git 提交说明：单独提交，例如 `test: add regression coverage for equilibria fitting and classification`。

16. 问题：拟合使用单个差分进化随机种子和有限优化预算，结果可能依赖随机初始化；当前没有报告多次拟合的参数和目标函数稳定性。
    修复方法：对每条序列和模型使用多个随机种子重复拟合，选择最佳结果并报告均值、方差、成功率和参数分散程度；确保不同模型使用可比较的优化预算。
    Git 提交说明：单独提交，例如 `feat(fit): add multi-seed calibration stability checks`。

17. 问题：12 条时间序列并不都是独立样本，例如 Andrén 七个区域来自同一研究，直接将其作为 12 个独立胜负样本会夸大证据量。
    修复方法：在模型比较中按数据集或研究来源分组，分别报告序列级和研究级结果；如计算总体统计，采用分层汇总或组内聚合，避免伪重复。
    Git 提交说明：单独提交，例如 `fix(analysis): account for grouped and non-independent time series`。

18. 问题：报告 `report/report.tex` 仍包含与当前实验冲突的旧结论，包括“phi 扫描 31/31 灭绝”“仅 B-D 共存”，并包含基于不可比 RMSE 的大幅改进结论。
    修复方法：完成上述核心代码修复并重新生成全部实验结果后，系统更新摘要、方法、结果、结论、表格和图注；明确区分绝对密度、相对变化、归一化 RMSE 和原始尺度 RMSE。不得在核心修复完成前仅修改文字结论。
    Git 提交说明：单独提交，例如 `docs(report): align conclusions with corrected experiment pipeline`。

19. 问题：运行 `main.py` 会直接覆盖大量 `results/` 图表，但结果文件缺少参数快照、代码版本和运行时间等溯源信息，难以判断图表对应哪一版实验。
    修复方法：每次实验输出机器可读的运行清单，至少记录 Git commit、参数、命令、时间、数据序列 ID 和关键指标；考虑按运行 ID 输出到独立目录，再显式选择用于报告的结果。
    Git 提交说明：单独提交，例如 `feat(results): add reproducible experiment manifests and run directories`。

20. 问题：当前实验流程没有统一的一键校验入口，主实验、标定、深度分析和报告可能使用不同时间点生成的结果。
    修复方法：新增只读检查或流水线命令，按顺序执行核心测试、默认参数标定、模型拟合、机制比较和结果一致性检查；在生成报告前验证所有表格、图和结论来自同一运行清单。
    Git 提交说明：单独提交，例如 `feat(workflow): add end-to-end experiment consistency check`。
