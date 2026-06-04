# 恐惧效应下捕食者—猎物动力学建模与数值分析

> **Fear-effect predator–prey dynamics: ODE modeling, mechanism comparison, and multi-source data calibration**  
> 数学建模课程项目 · 中国科学技术大学

[![GitHub](https://img.shields.io/github/stars/Maodawang66/fear-predator-prey-modeling?style=social)](https://github.com/Maodawang66/fear-predator-prey-modeling)

---

## 项目简介

在经典捕食者—猎物模型中引入**非消耗性恐惧效应（NCE）**与**记忆核**，研究恐惧对种群共存、振荡与崩溃的影响，并与无恐惧基线对比。项目包含：

- **双主线 ODE**：繁殖抑制型恐惧 + 指数记忆（满足题目记忆效应要求）；Myint (2025) Beddington–DeAngelis (B–D) + 饱和恐惧因子（服务真实数据标定）
- **六种文献 NCE 机制对照**、$\phi/\delta/k$ 参数扫描、三层恐惧稳定化检验
- **12 条公开时间序列**三模型标定（`baseline` / `fear_memory` / `bda_fear`）
- **A/B 双轨恐惧验证**：A 轨跨类等效抑制 $\eta$；B 轨 Peacor 元分析 + 珊瑚/豆娘实验先验
- **拓展**：2D 反应–扩散 PDE 与 Turing 线性诊断（附录，不作主结论）

完整论述见 [`report.pdf`](report.pdf)（主论文）、[`摘要.pdf`](摘要.pdf)（开题）、[`规划.pdf`](规划.pdf) / [`教程.pdf`](教程.pdf)（规划与汇报）。

---

## 主要结论（摘要）

| 方向 | 要点 |
|------|------|
| $\phi$ 扫描 | 主模型 31/31 为灭绝类，强恐惧在 Holling II 框架下易破坏共存 |
| 机制对照 | 仅 B–D+恐惧维持稳定共存（$\bar{u}\approx 4.98$），恐惧**进入路径**决定定性结论 |
| 数据标定 | B–D+恐惧 在 12/12 序列 RMSE 最优（0.054–0.209），较 baseline 改进 $10^2$–$10^6$ 倍 |
| $k$ 稳定化 | 共存密度随恐惧增强压缩，$\max\mathrm{Re}\,\lambda$ 由 $-0.628$ 升至 $-0.389$，全程稳定 |
| 双轨验证 | A 轨哺乳/鱼类 $\eta$ 中位数 $\sim 0.07$–0.08；B 轨 Peacor TMIE 58%、无脊椎 TMIE 92%，与豆娘实验 $\sim 6\%$ 同量级 |

---

## 仓库结构

```
pro/
├── main.py                 # 一键数值实验与主图输出
├── k_damping_analysis.py   # 恐惧强度 k 扫描与特征值分析
├── pde2d_turing.py         # 2D PDE + Turing 线性诊断（拓展）
├── src/                    # 模型、仿真、拟合、可视化
├── data/
│   ├── download_datasets.py
│   ├── deep_data_analysis.py   # 双轨 / Peacor / 类群分析
│   ├── calibrate_bda.py
│   └── raw/                    # 公开数据集（Dryad / LTER / GPDD 等）
├── results/                # 图表、标定 JSON、深度分析产出
├── report.tex / report.pdf # 主论文
├── 摘要.tex / 摘要.pdf
├── 规划.md / 教程.md
├── requirements.txt
└── compile_docs.py         # 规划/教程 → PDF
```

---

## 环境要求

- **Python** 3.10+（推荐 Conda 环境 `ai25`）
- **LaTeX**：XeLaTeX + `ctex`（编译论文）
- **Pandoc**（可选，编译 `规划.md` / `教程.md`）

```bash
pip install -r requirements.txt
# 或
conda create -n ai25 python=3.11
conda activate ai25
pip install -r requirements.txt
conda install -c conda-forge pandoc
```

---

## 快速开始

### 1. 数值实验与主图

```bash
conda activate ai25
python main.py
```

输出目录：`results/`（时序、相平面、$\phi/\delta$ 扫描、敏感性、文献机制对照等）。

### 2. 数据下载（可选，仓库已含 `data/raw/`）

```bash
python data/download_datasets.py
```

说明与引用见 [`data/数据来源.md`](data/数据来源.md)。

### 3. 多序列 B–D 标定

```bash
python data/calibrate_bda.py
```

### 4. 深度分析（双轨、Peacor、RMSE 等）

```bash
python data/deep_data_analysis.py
# 或 Windows:
run_deep_analysis.bat
```

### 5. 编译论文与文档

```bash
# 主论文
latexmk -xelatex report.tex

# 开题摘要
latexmk -xelatex 摘要.tex

# 规划 / 教程 Markdown → PDF
python compile_docs.py
```

Windows 也可使用：`compile_report.bat`、`compile_docs.bat`。

---

## 核心模型（简述）

**基线（无恐惧）** — 逻辑斯蒂猎物 + Holling II 捕食：

$$\dot{x} = rx\left(1-\frac{x}{K}\right) - \frac{axy}{1+ahx},\quad
\dot{y} = \frac{eaxy}{1+ahx} - my$$

**恐惧 + 记忆** — 繁殖抑制 $\phi v$ 与指数核状态 $v$：

$$\dot{x} = rx\left(1-\frac{x}{K}-\phi v\right) - \frac{axy}{1+ahx},\quad
\dot{v} = \delta(y - v)$$

**B–D + 饱和恐惧**（数据标定主模型）— 因子 $1/(1+kv)$ 作用于繁殖项。

三模型命名：`baseline`、`fear_memory`、`bda_fear`。

---

## 数据与引用

| 类型 | 示例来源 |
|------|----------|
| 种群时间序列 | 猞猁–狍、鱼–鱼、哈德逊湾、密歇根湖浮游动物、LTER 鱼类、GPDD |
| 恐惧 / NCE | Peacor et al. (2022) 元分析、珊瑚礁、豆娘捕食线索、恐惧景观 |

下载脚本与 DOI 清单见 [`data/数据来源.md`](data/数据来源.md)。使用公开数据时请遵守各平台许可并引用原始文献。

---

## 团队成员

| 姓名 | 学号 |
|------|------|
| 贺小轩 | PB23151820 |
| 李松茂 | PB23151824 |
| 王玉麟 | PB23151808 |

**课程**：数学建模 · 中国科学技术大学  
**日期**：2026 年 5 月

---

## 许可证与说明

本项目为**课程作业与学术研究用途**，代码与文档仅供学习交流。第三方数据集版权归原发布方所有；复现或二次发表请自行核对 Dryad / KNB / LTER 等平台的许可与引用要求。

如有问题或建议，欢迎在 [Issues](https://github.com/Maodawang66/fear-predator-prey-modeling/issues) 中反馈。

---

## 相关链接

- 仓库主页：<https://github.com/Maodawang66/fear-predator-prey-modeling>
- 作者 GitHub：<https://github.com/Maodawang66>
