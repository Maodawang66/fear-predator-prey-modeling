# k 增大削弱振荡：检验报告

## 文献依据
- Wang et al. (2016): fear factor 1/(1+kv) on reproduction
- Wang et al. (2019): NCE can stabilize cyclic predator-prey dynamics
- Myint et al. (2025): B-D + fear, k=0 limit cycle vs k>0 convergence

## 本项目采用的方法
1. Jacobian eigenvalue scan at (u*,v*) vs k
2. Numerical amplitude A=max-min in tail vs k
3. Peak decay ratio (last/first peak deviation from u*)

## 数值摘要
- Hopf 阈值估计 k_H ≈ None
- k=0: Re λ_max=-0.627988391909258, 相对振幅=2.582e-09, 稳定性=stable_node
- 扫描最大 k=0.18: 相对振幅=2.846e-08, Re λ_max=-0.38857639221312
- 最强压缩出现在 k=0.0, 相对振幅=2.582e-09

## 论文写法提示
若随 k 增大 max Re(λ) 由正变负，则存在 Hopf 型阈值 k_H，恐惧使周期解失稳→稳定。默认 B-D 参数下平衡点对所有 k 已局部稳定，此时应解读为：k 改变 Re(λ) 与瞬态振幅，并与 φ 扫描（Wang 2019 主模型）、数据拟合得到的 k 值（calibrate_bda）交叉验证。

图件见本目录 PNG；完整表格见 `k_scan.csv`。