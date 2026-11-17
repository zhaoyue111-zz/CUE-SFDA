# 无标签伪标签质量指标离线诊断

病例数：101（WCH_MRI145 DWI 训练 split）；模型和预测冻结，未运行 Adapt 训练。
所有预测在 nnUNet 重定向后的完整评估空间中生成：192³ 滑窗、step=0.5、Gaussian 加权 logit 融合，再 sigmoid；图像增强只有强度扰动，几何逆变换为恒等。GT 只在预测及 M0–M6 计算完成后用于 Dice、病例排序诊断和病灶体积偏差分析。

## 预测一致性与有效性

- 原 Consensus Cache 有 101 例融合概率和置信度，但无独立 view 预测，也没有足够元数据证明 step=0.5 Gaussian 协议；因此本实验没有将它与其他推理结果混用。逐 view 预测已重新生成并缓存。
- 现有不确定性 CSV 与新生成预测的组件差异审计：101 例有有效源值，最大绝对差 `0.49999995044119816`，均值 `0.014810673126396796`；差异 ≤1e-4：13 例，≤1e-3：34 例。M0 使用重算的同一预测组分数；CSV 仅作可复现性对照。
- 缺失/无效指标病例数：{"M0": 0, "M1": 0, "M2": 0, "M3": 0, "M4": 0, "M5": 0, "M6": 0}。M2/M3/M4 双方预测均空时得分为 0。
- 空预测：原图 20 例，最终联合融合 20 例；M1 将原图空预测的 20 例分数置 0，共改变 20 例分数。

## 视图真实 Dice 分布

| 视图组 | n | 均值 | 标准差 | 方差 | 最大 | 最小 |
|---|---:|---:|---:|---:|---:|---:|
| image_views | 606 | 0.3374 | 0.3610 | 0.1303 | 0.9174 | 0.0000 |
| prompt_views | 404 | 0.3541 | 0.3634 | 0.1320 | 0.9263 | 0.0000 |
| all_views | 1010 | 0.3441 | 0.3621 | 0.1311 | 0.9263 | 0.0000 |

## 与联合融合 Dice 的 Spearman 相关性

| 指标 | rho [病例 bootstrap 95% CI] | n | Top-20 平均联合 Dice | Oracle Top-20 重合 | M0 条件偏相关 |
|---|---:|---:|---:|---:|---:|
| M1 | 0.870 [0.795, 0.919] | 101 | 0.780 | 65.0% | 0.824 |
| M5 | 0.870 [0.786, 0.924] | 101 | 0.780 | 65.0% | 0.825 |
| M6 | 0.864 [0.781, 0.922] | 101 | 0.790 | 70.0% | 0.815 |
| M4 | 0.863 [0.778, 0.922] | 101 | 0.787 | 70.0% | 0.817 |
| M3 | 0.836 [0.740, 0.904] | 101 | 0.791 | 70.0% | 0.781 |
| M2 | 0.712 [0.555, 0.829] | 101 | 0.655 | 50.0% | 0.674 |
| M0 | 0.506 [0.300, 0.697] | 101 | 0.430 | 45.0% | nan |

## 相近 M0 稳定性下的区分能力

病例按 M0 分为 5 个等频组；`m0_strata_diagnostic.csv` 给出每组均值/中位数和每项分数与联合 Dice 的组内 Spearman。`metric_correlations.csv` 另给对 M0 排名控制后的偏 Spearman。

| 指标 | 五组内 Spearman 均值 | M0 控制偏 Spearman |
|---|---:|---:|
| M0 | 0.056 | nan |
| M1 | 0.410 | 0.824 |
| M2 | 0.332 | 0.674 |
| M3 | 0.374 | 0.781 |
| M4 | 0.428 | 0.817 |
| M5 | 0.406 | 0.825 |
| M6 | 0.402 | 0.815 |

## Top-K 筛选和病灶大小偏差

Top-K 汇总见 `topk_screening_summary.csv`，包括原图及联合 Dice、空预测数和两种 Dice Oracle Top-K 重合率。病灶大小关联以 score 与 log10(1+GT 前景体素) Spearman 表示。

| 指标 | score 与 GT 体积 Spearman | Top-20 平均 GT 前景体素 | Top-20 平均联合 Dice |
|---|---:|---:|---:|
| M0 | 0.404 | 7952.9 | 0.430 |
| M1 | 0.385 | 7167.4 | 0.780 |
| M2 | 0.257 | 5331.2 | 0.655 |
| M3 | 0.401 | 7905.4 | 0.791 |
| M4 | 0.373 | 7134.4 | 0.787 |
| M5 | 0.391 | 7167.4 | 0.780 |
| M6 | 0.406 | 7874.9 | 0.790 |

## 典型失败病例

`failure_case_examples.csv` 列出每项指标中 Top-20 高分但联合 Dice<0.1，以及 Bottom-20 低分但联合 Dice>0.7 的例子；该筛选只用于描述，不反馈到指标定义或阈值。

## 图表与数据文件

- `per_case_metrics.csv`：逐病例 M0–M6、原图/分支/联合 Dice、逐视图 Dice 与前景数。
- `per_view_dice.csv`、`view_dice_summary.csv`：逐 view 质量和 Image/Prompt/全部 view 统计。
- `metric_correlations.csv`：四种目标相关性及 bootstrap 95% CI。
- `topk_screening_summary.csv`、`topk_selected_cases.csv`：Top-20/40/59 筛选。
- `m0_strata_diagnostic.csv`、`lesion_size_association.csv`、`failure_case_examples.csv`：增量信息、病灶大小偏差和失败病例。
- `view_prediction_cache/`：逐病例独立概率图与空间/推理元信息，供离线复算。
- `scatter_*.png`：M0–M6 与四种目标的相关散点图，点颜色表示 GT 病灶大小。

GT Dice 与病灶体积仅为离线诊断标签；任何指标公式、阈值、Top-K 大小均未根据 GT 调整。



## 空预测门控与非空病例条件分析

共有 20 例原图预测为空，81 例非空；空预测病例的 M0 平均为 0.668，非空病例为 0.851。在所有非空病例上 M1 与 M0 分数逐例相同（True），因此 M1 的整体提升完全来自排除空预测；它不提供空预测门控之外的新排序信息。

`conditional_nonempty_correlations.csv` 同时给出全体、原图非空和联合预测非空子集上的相关性。该条件分析显示在排除空预测后，M0 与各质量指标仍可进一步比较，不把 M1 的空预测筛除效果误解为新的连续质量信息。

| Cohort | Metric | Target | n | Spearman |
|---|---|---|---:|---:|
| raw_prediction_nonempty | M0 | joint_dice | 81 | 0.844 |
| raw_prediction_nonempty | M1 | joint_dice | 81 | 0.844 |
| raw_prediction_nonempty | M2 | joint_dice | 81 | 0.587 |
| raw_prediction_nonempty | M3 | joint_dice | 81 | 0.788 |
| raw_prediction_nonempty | M4 | joint_dice | 81 | 0.834 |
| raw_prediction_nonempty | M5 | joint_dice | 81 | 0.842 |
| raw_prediction_nonempty | M6 | joint_dice | 81 | 0.831 |
| joint_prediction_nonempty | M0 | joint_dice | 81 | 0.844 |
| joint_prediction_nonempty | M1 | joint_dice | 81 | 0.844 |
| joint_prediction_nonempty | M2 | joint_dice | 81 | 0.587 |
| joint_prediction_nonempty | M3 | joint_dice | 81 | 0.788 |
| joint_prediction_nonempty | M4 | joint_dice | 81 | 0.834 |
| joint_prediction_nonempty | M5 | joint_dice | 81 | 0.842 |
| joint_prediction_nonempty | M6 | joint_dice | 81 | 0.831 |

完整 Top-K 汇总现已同时包括原图空预测和联合预测空预测数；见 `topk_screening_summary.csv`。

## M2 原始过滤比较定义

M2 的原始前景掩码使用仓库 Chebyshev 路径的严格 `p > 0.5` 判定；其余二值比较按统一阈值 0.5 处理。该定义也已从逐 view 缓存重算并同步到 M2/M5 相关性及 Top-K 汇总。

严格阈值复核中，`p == 0.5` 边界改变 M2 的病例数为 6。

## 结果解释与典型失败

- 全体 101 例上，M0 对联合 Dice 的 Spearman 为 0.506；其中有 20 例原图空预测，空预测平均 M0=0.668，且 M0 Top-20 中 10 例为空，说明稳定空输出会抬高 M0 排名。M1 为 0.870，但它只把原图空预测置 0；所有 81 个非空病例的 M1 与 M0 完全相同，非空子集的 rho 都是 0.844。因此 M1 的总体提升只是排除空预测。

- M3/M4 在全体样本上与联合 Dice 相关较强（rho 分别为 0.836/0.863）；控制 M0 的偏 Spearman 分别为 0.781/0.817，五个 M0 稳定性组内的平均 rho 分别为 0.374/0.428，显示它们在相近 M0 组内仍有排序信息。非空子集的全局 rho 则为 M0=0.844、M3=0.788、M4=0.834；因此这项增量是组内局部区分能力，并没有带来非空病例整体排序超过 M0。M5/M6 同样没有超过 M0 的非空全局排序。

- M2 全体 rho=0.712，非空病例 rho=0.587，Top-20 联合 Dice 均值 0.655；过滤一致性高并不能保证过滤后的伪标签正确，不能单独替代病例级稳定性。M4 Top-20 均值 0.787，M3 为 0.791，M1 为 0.780。

- 分数与 log10(1+GT 前景体素) 的 Spearman 为 0.257–0.406；全体平均 GT 前景体素为 3,053，Top-20 被选病例的均值约 5,331–7,953，说明这些指标普遍偏向较大病灶。Top-K 筛选的高 Dice 收益需结合该偏差解释。

- 典型失误：M0 将 0001679853、0017354579、0017452826、0018806748、0019873347 等空预测打到 1.0；M1/M3/M4/M5/M6 对 0010392655 给出高分（Dice=0，GT 前景 124，联合预测约 1,015 体素）；M2 也会把 0000503556、0009818705 评为高分但 Dice=0。它们分别代表稳定空输出、跨视图一致的错误小灶，以及过滤自洽但语义错误。

- 旧不确定性 CSV 与本次同配置预测有未能复现的分量差异，最大的两例是 0009808605（U_T 本轮 0.000、旧 CSV 0.500）和 0018289026（0.833、0.500）。这些值未被静默复用；M0 使用本轮同一预测组重算，完整差值见 `source_uncertainty_reproducibility.csv`。
