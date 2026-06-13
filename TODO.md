# TODO List

> 执行规则：每完成一条 TODO，完成对应验证后单独创建一次 Git 提交。不要将多个 TODO 合并到同一提交中。
>
> 分为三个阶段：Phase 1（移除 B-D fear）→ Phase 2（修正过度宣称、重写结论、补充实验、重组图表、统一语言）→ Phase 3（完善恐惧机制区分与三层叙事结构）。
>
> **Phase 1（项目 1--24）已于 2026-06-10 完成。** 共 8 次提交，涵盖标题/摘要/关键词、引言/相关工作、预备知识/模型方法、理论/实验、结论/参考文献、附录、图表引用和编译验证。六模型热力图重绘（项目 44）也在 Phase 1 收尾时提前完成。

---

## Phase 2：修正过度宣称、重组论文

### P1 — 修正过度宣称和不严谨表述

25. **第 6.1 节 "31/31 共存"**：11 个情景未通过收敛诊断。改为"31/31 在当前积分范围内未灭绝（其中 11 个情景未通过长期收敛诊断）"。
    Git 提交说明：`fix(report): qualify coexistence claim with convergence caveat`

26. **全文"15 条序列"的计数**：在首次出现处加脚注说明"含研究内重复（Andrén 七区、Killifish 三站、Windermere 两湖盆来自同一研究）"。汇总统计处改为"15 个拟合单位"或"序列级"。
    Git 提交说明：`fix(report): qualify per-series counts with non-independence caveat`

27. **可辨识性结论的限定**：注明 M(0)/$\delta$ profile 仅覆盖 3 条代表序列；选取标准需在正文中说明；其余序列不可假定同样不可辨识。
    Git 提交说明：`fix(report): limit identifiability conclusions to profiled series`

### P2 — 重写结论使其与现有实验一致

28. **结论重构**：从当前 5-6 条改为三段——(a) 理论层：机制路径依赖性；(b) 预测层：六模型比较，无单一恐惧通道一致胜出；(c) 机制层：为什么预测改善≠机制识别。每条标注证据强度。
    Git 提交说明：`refactor(report): restructure conclusion around three core questions`

29. **讨论增加与文献的对话**：增加一段说明"理论模型关心可能存在什么，数据模型检验在当前数据中实际被支持什么"；明确"15 个拟合单位中没有任何一个产生了恐惧强度的可靠估计"（如果属实）。
    Git 提交说明：`feat(report): connect discussion to existing literature`

### P3 — 补充最必要的实验与分析

30. **退化检验**：数值验证 $\phi=0$ 时 fear-memory 的 ODE 轨迹与 baseline 完全重合。对 AICc 支持 baseline 的序列，检查 fitted $\phi$ 是否集中在接近 0 的区域。绘制 $\Delta$AICc vs fitted $\phi$ 散点图。
    Git 提交说明：`feat(analysis): add null-model degeneration checks`

31. **$\Delta$AICc 实质阈值**：对每条序列计算 baseline 与最佳恐惧模型的 $\Delta$AICc；报告 $\Delta$AICc > 2、> 4、> 7 的序列各有多少。
    Git 提交说明：`feat(analysis): report substantial AICc differences with thresholds`

32. **参数不确定性量化**：对 baseline 的 15 条序列，使用 Fisher 信息矩阵或 parametric bootstrap（从残差重采样 10 次）报告参数 95% CI。标注 CI 包含 0 的参数。
    Git 提交说明：`feat(analysis): add parameter uncertainty quantification`

33. **研究级聚合胜负分析**：Andrén 七区取组内验证 RMSE 中位数；Killifish 三站同理；Windermere 两湖盆同理。重新计算 N≈6 的研究级胜负表。正文同时报告序列级和研究级。
    Git 提交说明：`fix(analysis): aggregate model comparison to study level`

### P4 — 重组图表和结果呈现

34. **正文图表精简**：正文保留 4 个核心 exhibit——(a) 六模型留后验证热力图；(b) 六模型胜负汇总表；(c) M(0) 和 $\delta$ profile（新 6.5 节）；(d) 一个代表性拟合图。其余理论分析图（时序图、$\phi$ 扫描、机制柱状图、$\delta$ 扫描等）移至附录。
    Git 提交说明：`refactor(report): streamline figures to 4 core exhibits`

35. **Windermere 案例分析归属**：从 6.4 节移入 6.6 节"讨论"，缩短正文版本，细节移至附录。
    Git 提交说明：`refactor(report): relocate Windermere case study to discussion`

36. **证据边界表扩展**：当前表 5 仅两行。扩展为三列多行，覆盖预测能力、机制识别、恐惧强度估计、跨系统比较四个维度。
    Git 提交说明：`feat(report): expand evidence-boundary table`

### P5 — 统一论文语言风格

37. **模型名统一**：全文统一模型引用——`fear_memory`（代码）→ "fear-memory"（正文），首次出现加中文注释。同样统一 fear-instant、fear-saturating、fear-foraging、fear-handling、baseline。
    Git 提交说明：`style(report): unify model name conventions`

38. **口语化/辩护性表达替换**：见 REBUTTAL.md 附录 A 完整清单。逐句替换。
    Git 提交说明：`style(report): replace colloquial and defensive phrasing`

39. **相关工作加入批判性评述**：每项文献后增加 1 句评价（"X 提出了 Y，但未检验 Z"）。
    Git 提交说明：`feat(report): add critical engagement to related work`

40. **高密度段落拆分**：L156-159（拟合程序段）等拆分为短句，每个方法论决策紧跟 1 句理由。
    Git 提交说明：`style(report): unpack dense methodological paragraphs`

---

## Phase 3：完善恐惧机制区分与三层叙事结构

> 执行前提：Phase 1（删 B-D）和 Phase 2（修正宣称/重组）已全部完成。
> 以下任务基于已确认结论编写，按编号顺序执行。

41. **正文说明 fear-memory 与 fear-instant 的区分**：
    - 在 4.3 节（等效恐惧强度校准段）增加一句："fear-memory 与 fear-instant 的 $\phi$ 值相同（0.00783431，$\delta=1$），在平衡点处恐惧项 $r\phi y x$ 完全等价；两者差异仅来自记忆方程 $dM/dt = y-\delta M$ 产生的一阶滞后（$\tau=1$），离开平衡点后时变恐惧强度不同。"
    - 在 6.2 节机制对照图中，在 fear-memory 和 fear-instant 的柱/条上标注"$\phi$ 相同"
    - 删除任何暗示"两者恐惧强度不同"的表述
    - 背景：合成恢复实验确证结构差异可辨识（核心参数固定时交叉拟合 RMSE 差约 50 倍），但真实数据中 fear-instant 在中位数 AICc 上优于 fear-memory 约 30 单位（-147.96 vs -118.03），记忆滞后未得到数据支持
    Git 提交说明：`docs(report): clarify fear-memory vs fear-instant distinction`

42. **正文核心比较改为三模型递进叙事**：
    - 叙事改为三层递进：
      (a) baseline vs fear-memory → 检验恐惧效应是否存在
      (b) fear-memory vs fear-instant → 检验记忆滞后是否重要（两者 $\phi$ 相同、$\delta=1$，差异仅来自 $dM/dt$ 的时滞）
      (c) 附录给出全部六模型，正文提炼一句话："无单一恐惧通道在留后预测和 AICc 上一致胜出"
    - 与 Phase 1 第 14 条协同：正文胜负表（表 2、表 8、表 9）聚焦 baseline / fear-memory / fear-instant 三列；fear-saturating / fear-foraging / fear-handling 三列移至附录
    - 六模型热力图正文保留（概览），但主表聚焦三模型
    - 正文首次列出三模型时注明："三者共享等效 $\phi=0.00783431$，构成一个干净的可比较三元组（无恐惧 → 恐惧+记忆 → 恐惧无记忆）"
    Git 提交说明：`refactor(report): three-model core comparison with tiered narrative`

43. **正文补充固定 $M(0)=y(0)$ 和 $\delta=1$ 的原因**：
    - 与 Phase 2 第 15 条协同：M(0)/$\delta$ profile 从附录升级到正文（新 6.5 节）时，增加方法论解释
    - 文字："固定 $\delta=1$ 提供统一的记忆时间尺度，避免在短序列中与 $\phi$ 相互补偿；固定 $M(0)=y(0)$ 使初始记忆与首个可观测捕食压力一致，避免引入额外的弱可辨识初始状态"
    - 如 Phase 2 第 27 条所述，注明此分析仅覆盖 3 条代表序列
    Git 提交说明：`docs(report): justify fixed M(0) and delta assumptions`

44. **重新生成六模型热力图**：
    - 从 `results/seven_model_real_fits/report_protocol_seven_model_metrics.csv` 读取数据
    - 删除 B-D 列（保留 baseline + 5 个 Holling II 恐惧通道）
    - 用与当前热力图相同的绘图参数重新生成
    - 输出到 `results/seven_model_real_fits/validation_rmse_heatmap_six_model.png`
    - 更新正文引用（与 Phase 1 第 14 条协调）
    - 不重新运行优化
    Git 提交说明：`fix(fig): regenerate heatmap without B-D column`
