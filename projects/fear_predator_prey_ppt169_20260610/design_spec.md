# fear_predator_prey - Design Spec

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | fear_predator_prey |
| **Canvas Format** | PPT 16:9 (1280×720) |
| **Page Count** | 13 |
| **Design Style** | academic defense / 极简学术 |
| **Target Audience** | USTC 数学建模课程答辩评委（老师 + 同学） |
| **Use Case** | 课程结题答辩，约 10-12 分钟 |
| **Created Date** | 2026-06-10 |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280×720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | left/right 60px, top 50px, bottom 40px |
| **Content Area** | 1160×630 (within margins) |

---

## III. Visual Theme

### Theme Style

- **Style**: academic defense / 极简学术
- **Theme**: Light theme
- **Tone**: professional, restrained, evidence-driven

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#FAFBFC` | Page background, slightly cool white |
| **Secondary bg** | `#F0F3F7` | Card background, section background |
| **Primary** | `#1B3A5C` | Title decorations, key sections, icons |
| **Accent** | `#2563A0` | Data highlights, key information, links |
| **Secondary accent** | `#4A90B8` | Secondary emphasis, gradient transitions |
| **Body text** | `#1E293B` | Main body text (not pure black) |
| **Secondary text** | `#64748B` | Captions, annotations |
| **Tertiary text** | `#94A3B8` | Supplementary info, footers |
| **Border/divider** | `#CBD5E1` | Card borders, divider lines |
| **Success** | `#16A34A` | Positive indicators (green family) |
| **Warning** | `#DC2626` | Issue markers (red family) |

### Gradient Scheme

```xml
<linearGradient id="titleGradient" x1="0%" y1="0%" x2="100%" y2="100%">
  <stop offset="0%" stop-color="#1B3A5C"/>
  <stop offset="100%" stop-color="#4A90B8"/>
</linearGradient>
```

---

## IV. Typography System

### Font Plan

**Typography direction**: academic CJK serif + modern sans

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | "Noto Serif CJK SC", SimSun | Georgia | serif |
| **Body** | "Noto Sans CJK SC", "Microsoft YaHei" | Arial | sans-serif |
| **Emphasis** | SimSun | Georgia | serif |
| **Code** | — | Consolas, "Courier New" | monospace |

**Per-role font stacks**:

- Title: `Georgia, "Noto Serif CJK SC", SimSun, serif`
- Body: `"Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif`
- Emphasis: `Georgia, SimSun, serif`
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

**Baseline**: Body font size = 20px

| Purpose | Ratio to body | Actual px | Weight |
| ------- | ------------- | --------- | ------ |
| Cover title | 2.8x | 56px | Bold |
| Page title | 1.6x | 32px | Bold |
| Subtitle | 1.2x | 24px | SemiBold |
| **Body content** | **1x** | **20px** | Regular |
| Annotation / caption | 0.7x | 14px | Regular |
| Page number | 0.55x | 11px | Regular |

---

## V. Layout Principles

### Page Structure

- **Header area**: Top 50px — page title with left-aligned primary-colored accent bar (4px × title height) + title text
- **Content area**: 50px to 680px — flexible grid
- **Footer area**: Bottom 40px — page number right-aligned, subtle tertiary text

### Layout Pattern Library

| Pattern | Suitable Scenarios |
| ------- | ----------------- |
| **Single column centered** | Covers, conclusions (P01, P12, P13) |
| **Asymmetric split (3:7)** | Image-heavy pages — chart left 30% title + takeaway, right 70% figure (P05, P06, P09) |
| **Symmetric split (5:5)** | Side-by-side comparisons — baseline vs fear model equations (P04) |
| **Top-bottom split** | Protocol flow + data summary (P07) |
| **Three column cards** | Three models overview, three profile plots (P04, P10) |
| **Two-column asymmetric (6:4)** | Text conclusion + supporting graphic (P08, P11) |
| **Z-pattern** | Story-driven pages — problem→method→result (P03) |

### Spacing Specification

**Universal**:

| Element | Value |
| ------- | ----- |
| Safe margin from canvas edge | 60px |
| Content block gap | 32px |
| Icon-text gap | 12px |

**Card-based layouts**:

| Element | Value |
| ------- | ----- |
| Card gap | 24px |
| Card padding | 24px |
| Card border radius | 10px |

---

## VI. Icon Usage Specification

### Source

- **Built-in icon library**: `templates/icons/` — tabler-outline
- **stroke_width**: 2

### Recommended Icon List

| Purpose | Icon Path | Page |
| ------- | --------- | ---- |
| Research question | `tabler-outline/search` | P03 |
| Model / equation | `tabler-outline/math-function` | P04 |
| Mechanism comparison | `tabler-outline/git-branch` | P05 |
| Parameter scan | `tabler-outline/chart-scatter` | P06 |
| Data / protocol | `tabler-outline/database` | P07 |
| Model comparison | `tabler-outline/scale` | P08, P09 |
| Identifiability | `tabler-outline/fingerprint` | P10 |
| Cross-system | `tabler-outline/world` | P11 |
| Conclusion | `tabler-outline/bulb` | P12 |
| Summary table | `tabler-outline/table` | P13 |

---

## VII. Visualization Reference List

No catalog chart templates used — all data visualizations are imported PNG images from project results.

---

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Layout pattern | Acquire Via | Status | text_policy | page_role |
| -------- | --------- | ----- | ------- | ---- | -------------- | ----------- | ------ | ----------- | --------- |
| 09_literature_mechanisms_bars.png | [from analyze] | [from analyze] | Six mechanism relative change comparison bar chart | Diagram | #44 background image + native diagram | user | Existing | none | local |
| 05_phi_scan.png | [from analyze] | [from analyze] | Phi parameter scan: prey/predator means vs phi | Diagram | #44 background image + native diagram | user | Existing | none | local |
| 12_isle_royale_wolf_moose_pre_2018_baseline.png | [from analyze] | [from analyze] | Isle Royale baseline fit | Data Chart | #12 asymmetric split | user | Existing | none | local |
| 12_isle_royale_wolf_moose_pre_2018_bda_fear.png | [from analyze] | [from analyze] | Isle Royale B-D fear fit | Data Chart | #12 asymmetric split | user | Existing | none | local |
| 12_isle_royale_wolf_moose_pre_2018_fear_memory.png | [from analyze] | [from analyze] | Isle Royale fear-memory fit | Data Chart | #12 asymmetric split | user | Existing | none | local |
| validation_rmse_heatmap.png | [from analyze] | [from analyze] | Seven-model validation RMSE heatmap | Data Chart | #44 background image + native diagram | user | Existing | none | local |
| k_profile_isle_royale_wolf_moose_pre_2018.png | [from analyze] | [from analyze] | k profile: Isle Royale | Data Chart | #10 three column cards | user | Existing | none | local |
| k_profile_glerl_m110_zoop_1994-201.png | [from analyze] | [from analyze] | k profile: GLERL zooplankton | Data Chart | #10 three column cards | user | Existing | none | local |
| k_profile_timeserieslogmeans_TP.png | [from analyze] | [from analyze] | k profile: TP killifish | Data Chart | #10 three column cards | user | Existing | none | local |
| eta_by_group.png | [from analyze] | [from analyze] | Eta by ecological group | Data Chart | #12 asymmetric split | user | Existing | none | local |
| rmse_improvement.png | [from analyze] | [from analyze] | RMSE improvement over baseline | Data Chart | #12 asymmetric split | user | Existing | none | local |

---

## IX. Content Outline

### Part 1: 引言与模型

#### Slide 01 — Cover
- **Layout**: Single column centered, full-bleed deep blue (#1B3A5C) background with centered white title
- **Rhythm**: `anchor`
- **Title**: 恐惧效应下的捕食者—猎物动力学
- **Subtitle**: 机制路径、预测比较与参数可辨识性
- **Info**: 贺小轩 · 李松茂 · 王玉麟 / USTC 数学建模课程

#### Slide 02 — 目录
- **Layout**: Two-column card grid (4 items)
- **Rhythm**: `anchor`
- **Title**: 目录
- **Content**: 四段式导航 —
  1. 研究背景与模型
  2. 动力学分析与机制比较
  3. 真实数据拟合与预测
  4. 参数可辨识性与结论

#### Slide 03 — 研究背景与三个核心问题
- **Layout**: Z-pattern: top-bar problem statement → left-right Q&A flow
- **Rhythm**: `breathing`
- **Title**: 研究背景与核心问题
- **Core message**: 捕食者不仅直接捕食，还通过风险线索改变猎物行为——恐惧以何种机制进入模型？数据能否判明？
- **Content**:
  - 非消耗性效应（NCE）：猎物感知风险→改变繁殖、觅食、活动
  - **问题一**：不同恐惧机制是否产生一致的种群动力学结论？
  - **问题二**：含恐惧模型能否改善真实时间序列的留后预测？
  - **问题三**：当模型复杂度增加时，数据能否可靠识别恐惧参数？

#### Slide 04 — 模型体系
- **Layout**: Symmetric split (5:5): left column ODE equations, right column model comparison cards
- **Rhythm**: `dense`
- **Title**: 模型体系
- **Core message**: 三种核心动力学形式——从简单到复杂，从经典到高灵活度竞争模型
- **Content**:
  - **Baseline**（logistic + Holling II）：2 ODE, 6 参数
  - **Fear-Memory**（指数记忆核）：3 ODE, 7 参数，M(t) 为捕食压力的指数加权累积
  - **B-D + Fear**（Beddington-DeAngelis + Wang 恐惧因子）：2 ODE, 8 参数，恐惧因子 f(k,v)=1/(1+kv)
- **Visualization**: 三列卡片水平排列，每列含渲染的 ODE 公式 PNG

#### Slide 05 — 六种恐惧机制与公平比较
- **Layout**: Asymmetric split (3:7): left explanation, right bar chart image
- **Rhythm**: `dense`
- **Title**: 六种恐惧机制与公平比较
- **Core message**: 20% 等效抑制下六种机制均保持共存，但长期种群变化方向明显不同——恐惧效应具有机制路径依赖性
- **Content**:
  - 校准原则：各自无恐惧正平衡点处 20% 抑制
  - 六种通道：瞬时繁殖 / 记忆繁殖 / 饱和繁殖 / 觅食抑制 / 处理时间延长 / B-D 恐惧
  - 繁殖抑制降低捕食者 ~17%；觅食/处理时间使两者上升；B-D 使两者下降 ~30%
- **Visualization**: `09_literature_mechanisms_bars.png`

### Part 2: 动力学分析

#### Slide 06 — φ 扫描与共存动力学
- **Layout**: Asymmetric split (3:7): left key numbers, right scan plot
- **Rhythm**: `dense`
- **Title**: φ 扫描：共存与收敛
- **Core message**: 31/31 φ 情景共存，但 11 个高 φ 情景未通过长期收敛诊断——共存不等于动力学稳态
- **Content**:
  - 猎物均值 ≈ 20.02，基本不变
  - 捕食者均值随 φ 递增而递减
  - 高 φ 区域需更长积分达到收敛
- **Visualization**: `05_phi_scan.png`

### Part 3: 真实数据拟合

#### Slide 07 — 真实数据拟合协议
- **Layout**: Top-bottom split: top protocol flow, bottom data overview
- **Rhythm**: `dense`
- **Title**: 真实数据拟合：15 条序列，统一协议
- **Core message**: 前 80% 拟合 · 后 20% 连续多步留后 · 归一化 RMSE + AICc 双重评价
- **Content**:
  - 数据：哺乳类（猞猁-雪兔、猞猁-狍、狼-驼鹿）、鱼类（鳉鱼、Windermere）、浮游动物（GLERL）
  - 拟合：差分进化（3 种子）+ L-BFGS-B 局部精修
  - 验证：从序列初始状态连续积分，不做验证段重初始化
  - 注意：Andrén 七区、Killifish 三站、Windermere 两湖盆存在研究内重复
- **Visualization**: 物种分组信息卡

#### Slide 08 — 三模型预测比较
- **Layout**: Two-column asymmetric (6:4): left fitting figure, right result stats
- **Rhythm**: `dense`
- **Title**: 三模型预测比较
- **Core message**: B-D 留后验证胜 9/15，但 AICc 胜 0/15；baseline AICc 胜 13/15——预测最优 ≠ 机制最优
- **Content**:
  - 留后验证单独最优：B-D 9 / baseline 3 / fear-memory 2 / GLERL 三并列 1
  - AICc 最优：baseline 13 / fear-memory 2 / B-D 0
  - 45 次拟合均可用（success 或 usable_limit）
- **Visualization**: Isle Royale 三模型拟合图（baseline / B-D / fear-memory 拼接）

#### Slide 09 — 七模型扩展
- **Layout**: Asymmetric split (3:7): left text, right heatmap
- **Rhythm**: `dense`
- **Title**: 七模型扩展：无单一通道普遍最优
- **Core message**: 105/105 拟合可比较，不同序列由不同通道获胜——不支持任一恐惧通道普遍正确
- **Content**:
  - 留后单独最优：B-D 6 / fear-instant 3 / baseline 2 / fear-memory 2 / fear-handling 1 / GLERL 七并列 1
  - 预测胜出不能替代参数可辨识性证据
- **Visualization**: `validation_rmse_heatmap.png`

### Part 4: 可辨识性与结论

#### Slide 10 — k 参数可辨识性
- **Layout**: Three column cards: 3 profile plots side-by-side
- **Rhythm**: `dense`
- **Title**: k 参数可辨识性：数据不足以区分恐惧强度
- **Core message**: 三条件的 95% 置信集均覆盖整个扫描区间 [0.0001, 0.5]——预测误差低 ≠ k 可辨识
- **Content**:
  - Isle Royale（狼-驼鹿）：profile 不连续，高低 k 区均有低目标值
  - GLERL（浮游动物）：profile 平坦，U 形特征弱
  - TP（鳉鱼）：profile 连续但近似平坦
  - 每个固定 k 下重新优化其余参数（真 profile，非条件扫描）
- **Visualization**: `k_profile_isle_royale_*.png` / `k_profile_glerl_*.png` / `k_profile_timeserieslogmeans_TP.png`

#### Slide 11 — 跨系统 η 与讨论
- **Layout**: Two-column asymmetric (6:4): left discussion points, right eta figure
- **Rhythm**: `dense`
- **Title**: 预测改善 ≠ 机制识别
- **Core message**: 四项独立理由表明，更低的留后误差不能直接解释为恐惧机制的真实存在
- **Content**:
  - ① 15 条序列不是 15 个独立生态重复（研究内重复）
  - ② AICc 在 13/15 序列上支持最简单的 baseline
  - ③ 参数补偿与触边——不同参数组合可产生相近轨迹
  - ④ Windermere 刺网移除破坏封闭 ODE 假设
  - η 类群中位数：哺乳类 0.00275 / 鱼类 0.0000123 / 浮游动物 0.0000500
  - k 不可辨识 → η 同样不可靠
- **Visualization**: `eta_by_group.png`

#### Slide 12 — 结论
- **Layout**: Single column, 6 numbered items with accent left borders
- **Rhythm**: `anchor`
- **Title**: 结论
- **Core message**: 恐惧效应的机制路径依赖性 + 预测与识别的区分
- **Content**: 六条结论，编号排列

#### Slide 13 — 核心结果汇总
- **Layout**: Single column centered, large summary table
- **Rhythm**: `anchor`
- **Title**: 核心结果汇总
- **Core message**: 一页纵览全部关键数值结果
- **Content**: 模型胜负统计 + 关键数值结论汇总表

---

## X. Speaker Notes Requirements

Not required for this project. Skip speaker notes generation.

---

## XI. Technical Constraints Reminder

### SVG Generation Must Follow:

1. viewBox: `0 0 1280 720`
2. Background uses `<rect>` elements
3. Text wrapping uses `<tspan>` (`<foreignObject>` FORBIDDEN)
4. Transparency uses `fill-opacity` / `stroke-opacity`; `rgba()` FORBIDDEN
5. FORBIDDEN: `mask`, `<style>`, `class`, `foreignObject`
6. FORBIDDEN: `textPath`, `animate*`, `script`
7. Raw Unicode for special chars; XML reserved chars escaped
8. `clipPath` only on `<image>` elements

### PPT Compatibility Rules:

- `<g opacity="...">` FORBIDDEN
- Image transparency uses overlay mask layer
- Inline styles only; no external CSS or `@font-face`
